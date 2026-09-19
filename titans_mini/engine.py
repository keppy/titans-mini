"""The outer streaming engine: one loop over time, any memory core underneath."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import TitansConfig
from .core import MemoryCore, build_memory_core

# `memory_state` is a dict for one core and a tensor for another, so the engine
# detaches it as a pytree rather than guessing its type.
from torch.utils._pytree import tree_map


class ModularTitansEngine(nn.Module):
    """Streaming token processor with a swappable test-time memory core.

    Static parameters: the QKV projection, the output projection, and whatever the
    core registers (Option B's `hyper_net` / `state_mutator` / LayerNorm; Option A
    registers only the *template* MLP the memory state is copied from).
    Dynamic state: the memory, created inside `forward` and never registered.

    `inner_lr` is a plain float on purpose. It is the memory's learning rate, i.e. a
    hyperparameter of the update rule, not a trained weight — making it an
    `nn.Parameter` would invite the outer optimizer to reshape the dynamics the
    experiment is trying to measure.
    """

    def __init__(
        self,
        memory_core_type: str = "MLP",
        *,
        d_model: int = 128,
        d_mem_vector: int = 256,
        inner_lr: float = 0.05,
        hidden_dim: int = 256,
        detach_memory: bool = False,
    ) -> None:
        super().__init__()
        self.memory_core_type = memory_core_type
        self.d_model = d_model
        self.inner_lr = inner_lr
        self.detach_memory = detach_memory

        self.to_qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.memory_core: MemoryCore = build_memory_core(
            memory_core_type, d_model=d_model, d_mem_vector=d_mem_vector, hidden_dim=hidden_dim
        )

        self.last_surprise_loss: Tensor | None = None

    @classmethod
    def from_config(cls, config: TitansConfig) -> "ModularTitansEngine":
        """Build from a `TitansConfig`, dropping the fields the engine doesn't take."""
        return cls(
            config.memory_core_type,
            d_model=config.d_model,
            d_mem_vector=config.d_mem_vector,
            inner_lr=config.inner_lr,
            hidden_dim=config.hidden_dim,
            detach_memory=config.detach_memory,
        )

    def forward(self, x: Tensor) -> Tensor:
        """`x`: token embeddings `[B, T, d_model]` -> processed output `[B, T, d_model]`.

        Side effect: `self.last_surprise_loss` holds the mean per-step surprise for
        this pass. It is a differentiable tensor (it is the signal that trains the
        memory core's static projections), so callers that never backward it should
        read `.item()` or set the reference free.
        """
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected [batch, seq, {self.d_model}] embeddings, got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        memory_state = self.memory_core.init_memory_state(batch_size, device=x.device, dtype=x.dtype)

        outputs: list[Tensor] = []
        surprises: list[Tensor] = []
        for t in range(seq_len):
            x_t = x[:, t]
            query_t, key_t, value_t = self.to_qkv(x_t).chunk(3, dim=-1)

            retrieved = self.memory_core.retrieve(memory_state, query_t)
            memory_state, surprise = self.memory_core.update(memory_state, key_t, value_t, self.inner_lr)
            if self.detach_memory:
                memory_state = tree_map(lambda leaf: leaf.detach(), memory_state)

            outputs.append(self.out_proj(retrieved) + x_t)
            surprises.append(surprise)

        self.last_surprise_loss = torch.stack(surprises).mean()
        return torch.stack(outputs, dim=1)
