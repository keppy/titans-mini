"""Global hyperparameters for Titans-Mini.

Defaults are the values in the architecture specification: batch_size 16, seq_len
512, d_model 128, d_mem_vector 256, inner_lr 0.05.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TitansConfig:
    """Shape and optimizer settings, shared by both memory cores."""

    batch_size: int = 16
    seq_len: int = 512
    d_model: int = 128
    d_mem_vector: int = 256
    inner_lr: float = 0.05
    # Which MemoryCore the engine builds. "MLP" or "Vector"; see titans_mini.core.
    memory_core_type: str = "MLP"
    # Width of the hidden layer in both cores: the inner layer of Option A's
    # two-layer MLP, and the hidden width of Option B's hyper_net / state_mutator.
    hidden_dim: int = 256
    # Cut the graph at every memory-state transition. False (default) lets the
    # outer loss reach Option B's state_mutator through the stream, which is what
    # trains the update rule; True is the O(1)-memory, no-BPTT setting. It is a
    # no-op for Option A, whose inner gradient-descent step is a state transition by
    # construction — that equality is asserted in the tests rather than assumed.
    detach_memory: bool = False
