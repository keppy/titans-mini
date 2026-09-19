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
   The gradients are detached explicitly before the step, so the update is a state
   transition rather than a node the outer backward pass can descend through; see
   the comment in `update` for why that is both the architecture's semantics and
   the only version whose cost is flat in sequence length.

The result for the outer graph, stated precisely, because the loose version of this
sentence was wrong once: a read is differentiable in the stream query (and in the
weights), so the outer loss flows through retrieval into the engine's projections —
that is the invariant the tests pin. The outer loss does **not** flow through the
inner update into the next step's memory. Option B is the option that can do that
(and only while `detach_memory=False`), which is exactly the comparison.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.func import functional_call

from ..core import MemoryCore, per_sequence_mse, register_memory_core

# The template's operands, and therefore the memory's own keys. Named explicitly so
# that adding a buffer to DynamicMLP cannot silently add memory state: see
# `MLPMemoryCore.init_memory_state`.
_MEMORY_OPERANDS = ("weight_in", "bias_in", "weight_out", "bias_out")


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

    Because they are not checkpointed, their values must be a *fixed function of the
    shapes*, and so they are drawn from a local generator rather than the global RNG:
    a fresh process, a second engine, and an engine that just loaded a checkpoint all
    start the memory from the same point, and constructing a core does not perturb the
    global stream. A random template would have made runs irreproducible while looking
    harmless.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, *, init_seed: int = 0) -> None:
        super().__init__()
        self.register_buffer("weight_in", torch.empty(d_in, d_hidden), persistent=False)
        self.register_buffer("bias_in", torch.zeros(d_hidden), persistent=False)
        self.register_buffer("weight_out", torch.empty(d_hidden, d_out), persistent=False)
        self.register_buffer("bias_out", torch.zeros(d_out), persistent=False)
        # Fan-in scaling hand-rolled because nn.init's fan_in/fan_out convention
        # assumes [out, in], and these weight matrices are stored [in, out].
        bound_in = 1.0 / math.sqrt(d_in)
        bound_out = 1.0 / math.sqrt(d_hidden)
        generator = torch.Generator(device="cpu").manual_seed(init_seed)
        with torch.no_grad():
            self.weight_in.uniform_(-bound_in, bound_in, generator=generator)
            self.weight_out.uniform_(-bound_out, bound_out, generator=generator)

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
            if name not in _MEMORY_OPERANDS:
                # The memory's shapes are read off the template, so any buffer added
                # to DynamicMLP would silently become memory state. Fail loudly
                # instead: this is the one place where that coupling is checked.
                raise RuntimeError(f"unexpected template operand {name!r}; expected {_MEMORY_OPERANDS}")
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

        def total_and_errors(weights: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
            """`(sum of per-sequence errors, the errors themselves)` — one forward pass.

            The scalar is what the step is derived from; the vector is what we report.
            Splitting them this way (rather than recomputing) is why a step costs one
            forward, one backward, and nothing else.
            """
            errors = per_sequence_mse(functional_call(self.mlp, weights, (key_t,)), value_t)
            return errors.sum(), errors

        # Gradients are taken *with respect to a pytree*, so the result is a dict
        # with the same keys and the same per-sequence shapes. Taking them via
        # torch.func leaves the module and the outer graph untouched.
        grads, errors = torch.func.grad(total_and_errors, has_aux=True)(memory_state)
        surprise = errors.mean()
        # The step is *defined* by a derivative taken inside it, and that derivative
        # is treated as a constant: the update is a state transition, not a node the
        # outer backward pass descends through. Two reasons, both load-bearing.
        #
        # (1) It is what the architecture says. The memory evolves by its own
        #     gradient descent; the differentiable signal for learning the
        #     projections is the surprise loss returned alongside it, not this step.
        # (2) It is not free to leave in. In torch 2.14 `torch.func.grad` returns
        #     gradients that still carry a graph (verified: they come back with
        #     `grad_fn` set), so the memory would accumulate a second-order graph
        #     across the sequence — a 512-step forward stops being O(1) in seq_len,
        #     and the outer loss quietly starts meta-learning the update rule.
        new_state = {
            name: weight - inner_lr * grads[name].detach() for name, weight in memory_state.items()
        }
        return new_state, surprise
