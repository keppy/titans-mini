"""Option A: the MLP memory core's state is a weight dict, and it stays in the graph."""

from __future__ import annotations

import torch

from titans_mini import ModularTitansEngine
from titans_mini.cores.mlp_core import MLPMemoryCore

D_MODEL = 8
HIDDEN = 12
BATCH = 3
INNER_LR = 0.05


def make_core() -> MLPMemoryCore:
    return MLPMemoryCore(d_model=D_MODEL, hidden_dim=HIDDEN)


def test_memory_state_is_a_per_sequence_weight_dict():
    core = make_core()
    state = core.init_memory_state(BATCH)

    assert set(state) == set(dict(core.mlp.named_buffers()))
    # One MLP per sequence: every tensor carries the batch dimension.
    for name, tensor in state.items():
        template = dict(core.mlp.named_buffers())[name]
        assert tensor.shape == (BATCH, *template.shape)
        assert tensor is not template  # a copy, not the module's tensor


def test_template_is_neither_a_parameter_nor_checkpointed_state():
    """The template must be invisible to optimizers and to a checkpoint."""
    core = make_core()
    assert list(core.mlp.parameters()) == []
    assert core.mlp.state_dict() == {}

    engine = ModularTitansEngine("MLP", d_model=D_MODEL, d_mem_vector=16, hidden_dim=HIDDEN)
    assert not any("weight_in" in key for key in engine.state_dict())
    # The memory's shapes are the engine's business, not the checkpoint's.
    before = {name for name, _ in engine.named_parameters()}
    engine(torch.randn(BATCH, 3, D_MODEL))
    assert {name for name, _ in engine.named_parameters()} == before


def test_retrieve_is_differentiable_in_query_and_in_the_generated_weights():
    core = make_core()
    state = core.init_memory_state(BATCH)

    query = torch.randn(BATCH, D_MODEL, requires_grad=True)
    retrieved = core.retrieve(state, query)
    assert retrieved.shape == (BATCH, D_MODEL)

    # The memory dict is bound by functional_call, so a backward pass through the
    # retrieved value reaches the query — i.e. the outer graph is intact.
    retrieved.sum().backward()
    assert query.grad is not None
    assert query.grad.abs().sum() > 0

    # ...and the retrieve path is differentiable w.r.t. the weight dict as well,
    # which is what the inner update differentiates.
    with torch.no_grad():
        grads = torch.func.grad(lambda weights: core.retrieve(weights, query.detach()).sum())(state)
    assert set(grads) == set(state)
    assert all(grad.shape == state[name].shape for name, grad in grads.items())
    assert any(grad.abs().sum() > 0 for grad in grads.values())


def test_update_lowers_surprise_and_does_not_mutate_the_old_state():
    core = make_core()
    state = core.init_memory_state(BATCH)
    snapshot = {name: tensor.clone() for name, tensor in state.items()}

    key = 0.5 * torch.randn(BATCH, D_MODEL)
    value = 0.5 * torch.randn(BATCH, D_MODEL)

    _, surprise_before = core.update(state, key, value, INNER_LR)
    for _ in range(20):
        state, surprise = core.update(state, key, value, INNER_LR)

    # The update is gradient descent on that exact error, so it must descend.
    assert surprise.item() < surprise_before.item()

    # The update is functional: the tensor that was passed in is untouched.
    for name, before in snapshot.items():
        assert torch.equal(state[name].detach(), before) is False
    fresh = core.init_memory_state(BATCH)
    updated = core.update(fresh, key, value, INNER_LR)[0]
    assert all(not torch.equal(updated[name], fresh[name]) for name in updated)


def test_template_module_is_never_in_the_graph_or_updated():
    """The declared module is a template: functional_call replaces it wholesale."""
    core = make_core()
    state = core.init_memory_state(BATCH)
    # A stream input that carries gradient, as the engine's projections do.
    key = torch.randn(BATCH, D_MODEL, requires_grad=True)
    value = torch.randn(BATCH, D_MODEL)

    retrieved = core.retrieve(state, key)
    assert retrieved.requires_grad  # the read is a live graph node...
    _, surprise = core.update(state, key, value, INNER_LR)
    (retrieved.sum() + surprise).backward()

    assert key.grad is not None and key.grad.abs().sum() > 0
    # ...but no gradient reaches the template: the weights that ran were the state's.
    assert all(buffer.grad is None for buffer in core.mlp.buffers())
    assert list(core.mlp.parameters()) == []
    # And the state is storage of its own.
    assert all(
        state[name].data_ptr() != template.data_ptr() for name, template in core.mlp.named_buffers()
    )
