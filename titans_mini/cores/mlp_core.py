"""Option A: the MLP memory core (standard Titans).

The memory is the *weights of a two-layer MLP*, one MLP per sequence in the batch.
A read is a forward pass through those weights; a write is one step of gradient
descent on the key/value reconstruction error.

Two things make this work rather than silently detaching the outer graph:

1. The weights are passed through `torch.func.functional_call` on every step. The
   dynamic weight dict is bound onto a stateless module, so the forward pass is an
   ordinary autograd-traced computation: `retrieved` is a differentiable function
   of both `query_t` (stream input) and the weight dict, and nothing is copied into
   or out of module state.
2. The write uses `torch.func.grad` on a *function of the weight dict* to get the
   gradients, then applies `w - lr * g` as plain tensor arithmetic. No `.backward()`
   is called on the surprise loss, no optimizer object owns the weights, and no
   `.data`/`.detach_()` trickery is involved — the outer graph is never touched.
   `torch.func.grad` returns constants (its own graph is not kept), so the GD step
   is a state transition, not part of the outer backward pass. That matches the
   architecture: the memory update is applied to state, and the *surprise loss* is
   the separate signal a caller backpropagates if it wants the update to improve.

The direct consequence, worth stating because it is the difference between the two
options: Option A cannot send gradient from the outer loss back into its own update
step — the update is defined by a derivative we take inside the step, so the chain
through memory is cut by construction. Option B can. That is the experiment.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from ..core import MemoryCore, per_sequence_mse, register_memory_core


class DynamicMLP(nn.Module):
    """A two-layer MLP whose weights are supplied per call, not owned.

    Every op is an explicit `einsum` so that the leading batch dimension of the
    supplied weights is honoured: each sequence has its own MLP.

    The operands are registered as **non-persistent buffers**, which is the precise
    statement of what they are. They are not parameters — nothing here is trained,
    and an optimizer must never see them or apply weight decay to them. They are not
    persistent state — a checkpoint has no business carrying a template. They exist
    only so that `functional_call` has names, shapes and an initialization to bind
    over; their values are replaced on every call and never read at runtime.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.register_buffer("weight_in", torch.empty(d_in, d_hidden), persistent=False)
        self.register_buffer("bias_in", torch.zeros(d_hidden), persistent=False)
        self.register_buffer("weight_out", torch.empty(d_hidden, d_out), persistent=False)
        self.register_buffer("bias_out", torch.zeros(d_out), persistent=False)
        # Fan-in scaling hand-rolled because nn.init's fan_in/fan_out convention
        # assumes [out, in], and these weight matrices are stored [in, out].
        bound_in = 1.0 / math.sqrt(d_in)
        bound_out = 1.0 / math.sqrt(d_hidden)
        nn.init.uniform_(self.weight_in, -bound_in, bound_in)
        nn.init.uniform_(self.weight_out, -bound_out, bound_out)

    def forward(self, x: Tensor) -> Tensor:
        """`x` is `[B, d_in]`; weights are `[B, ...]`-shaped, yielding `[B, d_out]`."""
        hidden = torch.einsum("bi,bih->bh", x, self.weight_in) + self.bias_in
        hidden = torch.nn.functional.gelu(hidden)
        return torch.einsum("bh,bho->bo", hidden, self.weight_out) + self.bias_out


@register_memory_core("MLP")
class MLPMemoryCore(MemoryCore):
    """Memory = weights/bias dict of a per-sequence two-layer MLP.

    Option A's memory has no separate vector size: it *is* the MLP, so its capacity
    is `d_model x hidden_dim`. `d_mem_vector` is accepted and ignored because every
    core takes the same construction arguments — that uniformity is what lets the
    engine build any of them from one config.
    """

    def __init__(self, d_model: int = 128, d_mem_vector: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        del d_mem_vector  # unused by this core; see the class docstring
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.mlp = DynamicMLP(d_model, hidden_dim, d_model)

    # -- memory ---------------------------------------------------------------
    def init_memory_state(
        self, batch_size: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> dict[str, Tensor]:
        """Per-sequence copy of the template MLP, `[B, ...]`-shaped."""
        state: dict[str, Tensor] = {}
        for name, template in self.mlp.named_buffers():
            template = template.detach().to(device=device, dtype=dtype)
            state[name] = template.unsqueeze(0).expand(batch_size, *template.shape).clone()
        return state

    # -- read / write ---------------------------------------------------------
    def retrieve(self, memory_state: dict[str, Tensor], query_t: Tensor) -> Tensor:
        """One forward pass through the dynamic weights."""
        return functional_call(self.mlp, memory_state, (query_t,))

    def update(
        self, memory_state: dict[str, Tensor], key_t: Tensor, value_t: Tensor, inner_lr: float
    ) -> tuple[dict[str, Tensor], Tensor]:
        """One inner gradient-descent step of the memory MLP.

        Each sequence has its own MLP and therefore its own loss, so the gradients
        are taken of the *sum* across the batch: sequence `i`'s step is exactly
        `inner_lr * dMSE_i/dW_i`, with no `1/B` dilution. The scalar `surprise`
        returned is the batch mean of the same per-sequence errors, so it is
        comparable between cores and readable at the scale of the values.
        """

        def per_sequence_errors(weights: dict[str, Tensor]) -> Tensor:
            prediction = functional_call(self.mlp, weights, (key_t,))
            return per_sequence_mse(prediction, value_t)

        surprise = per_sequence_errors(memory_state).mean()
        # Gradients are taken *with respect to a pytree*, so the result is a dict
        # with the same keys and the same per-sequence shapes. Taking them via
        # torch.func leaves the module and the outer graph untouched.
        grads = torch.func.grad(lambda weights: per_sequence_errors(weights).sum())(memory_state)
        new_state = {name: weight - inner_lr * grads[name] for name, weight in memory_state.items()}
        return new_state, surprise
