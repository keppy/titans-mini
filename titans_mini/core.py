"""The swappable memory-core interface and its registry.

A memory core owns *storage and retrieval*, never the stream: `ModularTitansEngine`
runs the loop over time and calls the core once per step. Any core is therefore
described completely by three methods:

    init_memory_state(batch_size)      -> whatever the core calls a memory
    retrieve(memory_state, query_t)    -> retrieved_value
    update(memory_state, key_t, value_t, inner_lr) -> (new_memory_state, surprise_loss)

`memory_state` is deliberately untyped (a dict of tensors for Option A, a single
tensor for Option B). The engine only ever passes it back to the core that made it,
so one core's state format cannot leak into another's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import torch
from torch import Tensor, nn

# A core's memory: opaque to the engine, passed back unexamined.
MemoryState = Any


def per_sequence_mse(prediction: Tensor, value: Tensor) -> Tensor:
    """Squared error per sequence: `[B, d]` inputs -> `[B]`.

    Shared by both cores so that "surprise" means the same thing in each and the
    two are comparable: the error of one sequence's memory, reduced over the
    feature dimension only, never across the batch.
    """
    return ((prediction - value) ** 2).mean(dim=-1)


class MemoryCore(nn.Module, ABC):
    """Base class for a swappable test-time memory core.

    Subclasses hold only *static* parameters (the projections that read and write
    the memory). The memory itself is never a parameter or a buffer — it is created
    per forward pass and returned step by step, so it is per-sequence state, not
    model weights.
    """

    @abstractmethod
    def init_memory_state(
        self, batch_size: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> MemoryState:
        """Build the zero step-0 memory for a batch of `batch_size` sequences."""

    @abstractmethod
    def retrieve(self, memory_state: MemoryState, query_t: Tensor) -> Tensor:
        """Read: map the memory and the current query to a retrieved value `[B, d_model]`."""

    @abstractmethod
    def update(
        self, memory_state: MemoryState, key_t: Tensor, value_t: Tensor, inner_lr: float
    ) -> tuple[MemoryState, Tensor]:
        """Write: evolve the memory toward storing `(key_t -> value_t)`.

        Returns the new memory state and the scalar "surprise" (prediction error)
        the update was driven by.
        """


CoreT = TypeVar("CoreT", bound=MemoryCore)
_CORES: dict[str, type[MemoryCore]] = {}


def register_memory_core(name: str) -> Callable[[type[CoreT]], type[CoreT]]:
    """Register a core class under a case-insensitive name."""

    key = name.strip().lower()
    if not key:
        raise ValueError("memory core name must be non-empty")

    def decorator(cls: type[CoreT]) -> type[CoreT]:
        if key in _CORES:
            raise ValueError(f"memory core {name!r} is already registered")
        _CORES[key] = cls
        return cls

    return decorator


def memory_core_names() -> tuple[str, ...]:
    """Registered core names, lowercased, in registration order."""
    return tuple(_CORES)


def build_memory_core(name: str, **kwargs: Any) -> MemoryCore:
    """Instantiate a registered core. `name` is matched case-insensitively."""
    key = name.strip().lower()
    try:
        cls = _CORES[key]
    except KeyError:
        raise ValueError(f"unknown memory core {name!r}; registered: {sorted(_CORES)}") from None
    return cls(**kwargs)
