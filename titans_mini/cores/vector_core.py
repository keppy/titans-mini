"""Option B: highly non-linear vector memory ("data-as-parameters").

The memory is not a weight matrix. It is a flat vector per sequence,
`memory_state = torch.zeros(B, d_mem_vector)`, and the *weights* are generated on
the fly from that vector by a small fixed network. Two static projections do the
translation in both directions:

    hyper_net       memory vector -> (W_temp, B_temp) for a virtual square layer
    state_mutator   (memory, key, value) -> delta applied to the memory vector

Separation of concerns, per the specification: `hyper_net`, `state_mutator` and the
readout normalisation are ordinary `nn.Parameter`-bearing modules — trained by the
outer optimizer, never touched by the update rule. The memory vector is the opposite:
it is *only* ever a tensor produced and consumed inside a forward pass. It is not a
parameter, not a buffer, and never appears in `state_dict()`; the step-to-step
transition (`memory + inner_lr * delta`, then normalised) is all that carries it
forward.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..core import MemoryCore, per_sequence_mse, register_memory_core


@register_memory_core("Vector")
class VectorMemoryCore(MemoryCore):
    """Memory = a per-sequence vector, read and written through learned projections."""

    def __init__(self, d_model: int = 128, d_mem_vector: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_mem_vector = d_mem_vector
        # One square virtual layer (d_model -> d_model) plus its bias, generated
        # per sequence: d_model**2 + d_model numbers out of the memory vector.
        generated_width = d_model * d_model + d_model
        self.hyper_net = nn.Sequential(
            nn.Linear(d_mem_vector, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, generated_width),
        )
        self.state_mutator = nn.Sequential(
            nn.Linear(d_mem_vector + 2 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_mem_vector),
        )
        self.memory_norm = nn.LayerNorm(d_mem_vector)
        # The generated weights are fed straight into a matmul with activations of
        # unit-ish scale, so they are scaled by the standard 1/sqrt(fan_in) of a
        # layer at initialization; without it the virtual layer's pre-activation
        # grows with d_model and saturates the GeLU.
        self.generated_weight_scale = 1.0 / math.sqrt(d_model)

    # -- memory ---------------------------------------------------------------
    def init_memory_state(
        self, batch_size: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Tensor:
        """Step-0 memory: zeros, shape `[B, d_mem_vector]`."""
        return torch.zeros(batch_size, self.d_mem_vector, device=device, dtype=dtype)

    # -- generated weights ----------------------------------------------------
    def _dynamic_weights(self, memory_state: Tensor) -> tuple[Tensor, Tensor]:
        """`hyper_net`: memory `[B, d_mem_vector]` -> `(W_temp [B,d,d], B_temp [B,d])`."""
        generated = self.hyper_net(memory_state)
        weight_flat, bias = generated.split([self.d_model * self.d_model, self.d_model], dim=-1)
        weight = weight_flat.view(-1, self.d_model, self.d_model) * self.generated_weight_scale
        return weight, bias

    def _virtual_layer(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """`gelu(x @ W_temp + B_temp)`, batched over per-sequence weights."""
        return torch.nn.functional.gelu(torch.einsum("bi,bij->bj", x, weight) + bias)

    # -- read / write ---------------------------------------------------------
    def retrieve(self, memory_state: Tensor, query_t: Tensor) -> Tensor:
        weight, bias = self._dynamic_weights(memory_state)
        return self._virtual_layer(query_t, weight, bias)

    def update(
        self, memory_state: Tensor, key_t: Tensor, value_t: Tensor, inner_lr: float
    ) -> tuple[Tensor, Tensor]:
        """Surprise-scored vector update."""
        weight, bias = self._dynamic_weights(memory_state)
        # How well the *current* dynamic weights predict this key's value.
        surprise = per_sequence_mse(self._virtual_layer(key_t, weight, bias), value_t).mean()

        delta = self.state_mutator(torch.cat([memory_state, key_t, value_t], dim=-1))
        new_state = self.memory_norm(memory_state + inner_lr * delta)
        return new_state, surprise
