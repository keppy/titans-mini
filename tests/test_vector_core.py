"""Option B: static projections are parameters; the memory is a flowing tensor."""

from __future__ import annotations

import torch
from torch import nn

from titans_mini import ModularTitansEngine, VectorMemoryCore

D_MODEL = 8
D_MEM = 16
HIDDEN = 12
BATCH = 3
SEQ = 5
INNER_LR = 0.05


def make_core() -> VectorMemoryCore:
    return VectorMemoryCore(d_model=D_MODEL, d_mem_vector=D_MEM, hidden_dim=HIDDEN)


def make_engine(**overrides) -> ModularTitansEngine:
    kwargs = {
        "d_model": D_MODEL,
        "d_mem_vector": D_MEM,
        "hidden_dim": HIDDEN,
        "inner_lr": INNER_LR,
    }
    kwargs.update(overrides)
    return ModularTitansEngine("Vector", **kwargs)


def test_static_modules_are_registered_parameters():
    core = make_core()
    names = dict(core.named_parameters())

    assert all(isinstance(param, nn.Parameter) for param in names.values())
    for prefix in ("hyper_net.0.weight", "hyper_net.2.weight", "state_mutator.0.weight", "state_mutator.2.weight"):
        assert prefix in names, prefix
    # The readout normalisation is static too.
    assert "memory_norm.weight" in names
    # The generated weight block is a square layer plus its bias.
    assert names["hyper_net.2.weight"].shape == (D_MODEL * D_MODEL + D_MODEL, HIDDEN)
    assert names["state_mutator.0.weight"].shape == (HIDDEN, D_MEM + 2 * D_MODEL)


def test_memory_state_is_an_unregistered_hidden_tensor():
    core = make_core()
    state = core.init_memory_state(BATCH)

    assert isinstance(state, torch.Tensor)
    assert not isinstance(state, nn.Parameter)
    assert state.shape == (BATCH, D_MEM)
    assert torch.count_nonzero(state) == 0
    assert not state.requires_grad  # a step-0 state is data, not a leaf to train

    # Nothing about the memory is model state: it is nowhere in state_dict, and the
    # key set does not depend on the batch or the sequence length.
    engine = make_engine()
    small = set(engine.state_dict())
    assert not any("memory_state" in key for key in small)
    engine(torch.randn(BATCH + 2, SEQ + 3, D_MODEL))
    assert set(engine.state_dict()) == small


def test_retrieve_shape_and_update_normalisation():
    core = make_core()
    state = core.init_memory_state(BATCH)
    query = torch.randn(BATCH, D_MODEL)
    key = torch.randn(BATCH, D_MODEL)
    value = torch.randn(BATCH, D_MODEL)

    assert core.retrieve(state, query).shape == (BATCH, D_MODEL)

    new_state, surprise = core.update(state, key, value, INNER_LR)
    assert new_state.shape == (BATCH, D_MEM)
    assert surprise.dim() == 0 and surprise.requires_grad

    # The update is `LayerNorm(memory + inner_lr * delta)`: mean is exactly zero,
    # and the std is `sqrt(var / (var + eps))` — just under 1 because the freshly
    # initialised memory has variance of the same order as LayerNorm's eps.
    pre_norm = state + INNER_LR * core.state_mutator(torch.cat([state, key, value], dim=-1))
    assert torch.allclose(new_state, torch.nn.functional.layer_norm(pre_norm, (D_MEM,), core.memory_norm.weight, core.memory_norm.bias))
    assert torch.allclose(new_state.mean(dim=-1), torch.zeros(BATCH), atol=1e-6)
    std = new_state.std(dim=-1, unbiased=False)
    assert torch.all(std <= 1.0 + 1e-4) and torch.all(std > 0.5)


def test_hidden_state_flows_step_to_step_and_is_never_a_parameter():
    engine = make_engine()
    x = torch.randn(BATCH, SEQ, D_MODEL)
    engine(x)

    # The engine holds it only long enough to report the surprise loss.
    assert engine.last_surprise_loss is not None
    assert engine.last_surprise_loss.dim() == 0
    assert not any("memory" in name for name, _ in engine.memory_core.named_buffers())


def test_update_rule_and_hyper_net_are_trained_through_the_stream():
    """With the graph kept, the outer loss reaches both static projections."""
    engine = make_engine(detach_memory=False)
    x = torch.randn(BATCH, SEQ, D_MODEL)

    output = engine(x)
    (output.sum() + engine.last_surprise_loss).backward()

    grads = {name: param.grad for name, param in engine.memory_core.named_parameters()}
    assert all(grad is not None for grad in grads.values()), "some static parameter got no gradient"
    assert all(grad.abs().sum() > 0 for grad in grads.values()), "zero gradient"
    # Specifically: the state_mutator only ever influences the output through the
    # memory chain, so its gradient is proof the chain is intact.
    mutator = [grad for name, grad in grads.items() if name.startswith("state_mutator")]
    assert len(mutator) == 4 and all(grad.abs().sum() > 0 for grad in mutator)


def test_detach_memory_cuts_the_chain_but_still_trains_the_hyper_net():
    """The documented tradeoff of the flag: O(1) memory, no learning signal for delta."""
    engine = make_engine(detach_memory=True)
    x = torch.randn(BATCH, SEQ, D_MODEL)

    output = engine(x)
    (output.sum() + engine.last_surprise_loss).backward()

    grads = {name: param.grad for name, param in engine.memory_core.named_parameters()}
    assert all(grad is None for name, grad in grads.items() if name.startswith("state_mutator"))
    assert all(grad is not None for name, grad in grads.items() if name.startswith("hyper_net"))
    # The retrieval path is still live where it matters: memory read -> output.
    assert engine.to_qkv.weight.grad is not None
