"""Shared fixtures: shapes small enough to debug by eye, big enough to be real."""

from __future__ import annotations

import pytest
import torch

from titans_mini import ModularTitansEngine

BATCH = 3
SEQ = 5
D_MODEL = 8
D_MEM = 16
HIDDEN = 12
INNER_LR = 0.05


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)


@pytest.fixture
def engine_kwargs() -> dict[str, int | float]:
    return {
        "d_model": D_MODEL,
        "d_mem_vector": D_MEM,
        "hidden_dim": HIDDEN,
        "inner_lr": INNER_LR,
    }


@pytest.fixture(params=["MLP", "Vector"])
def engine(request, engine_kwargs) -> ModularTitansEngine:
    """Every core, same shapes — the whole point of the interface."""
    return ModularTitansEngine(request.param, **engine_kwargs)


@pytest.fixture
def embeddings() -> torch.Tensor:
    """Stand-in for the incoming token embeddings."""
    return torch.randn(BATCH, SEQ, D_MODEL)
