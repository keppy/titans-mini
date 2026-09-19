"""The outer engine: one loop, both cores, and an intact autograd graph."""

from __future__ import annotations

import pytest
import torch

from titans_mini import ModularTitansEngine, build_memory_core, memory_core_names

D_MODEL = 8
D_MEM = 16
HIDDEN = 12
BATCH = 3
SEQ = 5

ENGINE_KWARGS = dict(d_model=D_MODEL, d_mem_vector=D_MEM, hidden_dim=HIDDEN, inner_lr=0.05)


@pytest.mark.parametrize("core_type", ["MLP", "Vector"])
def test_forward_returns_the_input_shape(core_type):
    engine = ModularTitansEngine(core_type, **ENGINE_KWARGS)
    x = torch.randn(BATCH, SEQ, D_MODEL)

    output = engine(x)

    assert output.shape == (BATCH, SEQ, D_MODEL)
    assert torch.isfinite(output).all()
    assert engine.last_surprise_loss.item() >= 0.0


@pytest.mark.parametrize("core_type", ["MLP", "Vector"])
def test_outer_autograd_graph_survives_the_inner_loop(core_type):
    engine = ModularTitansEngine(core_type, **ENGINE_KWARGS)
    x = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)

    output = engine(x)
    assert output.requires_grad
    assert engine.last_surprise_loss.requires_grad

    (output.sum() + engine.last_surprise_loss).backward()

    assert x.grad is not None and x.grad.abs().sum() > 0
    assert engine.to_qkv.weight.grad is not None and engine.to_qkv.weight.grad.abs().sum() > 0
    assert engine.out_proj.weight.grad is not None and engine.out_proj.weight.grad.abs().sum() > 0


def test_core_names_are_case_insensitive_and_unknown_ones_are_rejected():
    assert set(memory_core_names()) == {"mlp", "vector"}
    assert isinstance(build_memory_core("vector", d_model=D_MODEL, d_mem_vector=D_MEM, hidden_dim=HIDDEN), torch.nn.Module)
    assert isinstance(
        ModularTitansEngine("vector", **ENGINE_KWARGS).memory_core, torch.nn.Module
    )
    with pytest.raises(ValueError, match="unknown memory core"):
        ModularTitansEngine("KDA", **ENGINE_KWARGS)


def test_forward_rejects_a_shape_it_cannot_process():
    engine = ModularTitansEngine("MLP", **ENGINE_KWARGS)
    with pytest.raises(ValueError, match="expected"):
        engine(torch.randn(BATCH, SEQ))
    with pytest.raises(ValueError, match="expected"):
        engine(torch.randn(BATCH, SEQ, D_MODEL + 1))


@pytest.mark.parametrize("core_type", ["MLP", "Vector"])
def test_swapping_the_core_is_the_only_difference(core_type):
    """Same seed, same engine code path — the core is the only variable."""
    torch.manual_seed(1234)
    first = ModularTitansEngine(core_type, **ENGINE_KWARGS)
    torch.manual_seed(1234)
    second = ModularTitansEngine(core_type, **ENGINE_KWARGS)

    x = torch.randn(BATCH, SEQ, D_MODEL)
    assert torch.equal(first(x), second(x))

    # The two cores really are different objects with different state layouts.
    mlp_keys = set(ModularTitansEngine("MLP", **ENGINE_KWARGS).state_dict())
    vector_keys = set(ModularTitansEngine("Vector", **ENGINE_KWARGS).state_dict())
    assert mlp_keys != vector_keys


@pytest.mark.parametrize("core_type", ["MLP", "Vector"])
def test_checkpoint_round_trip_reproduces_outputs(core_type):
    """A fresh engine that loads a checkpoint must reproduce it bit for bit.

    This is what pins the MLP core's template to a fixed initialization: the template
    is deliberately not checkpointed, so if it were drawn from the global RNG instead,
    this would drift silently.
    """
    source = ModularTitansEngine(core_type, **ENGINE_KWARGS)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    expected = source(x)

    restored = ModularTitansEngine(core_type, **ENGINE_KWARGS)
    restored.load_state_dict(source.state_dict())

    assert torch.equal(restored(x), expected)


def test_detach_memory_is_a_noop_for_the_mlp_core():
    """Option A's step is a state transition either way; assert it, don't assume it."""
    keep = ModularTitansEngine("MLP", detach_memory=False, **ENGINE_KWARGS)
    cut = ModularTitansEngine("MLP", detach_memory=True, **ENGINE_KWARGS)
    cut.load_state_dict(keep.state_dict())

    x = torch.randn(BATCH, SEQ, D_MODEL)
    kept, cut_output = keep(x), cut(x)
    assert torch.equal(kept, cut_output)

    grad_kept = torch.autograd.grad(kept.pow(2).mean(), keep.to_qkv.weight)[0]
    grad_cut = torch.autograd.grad(cut_output.pow(2).mean(), cut.to_qkv.weight)[0]
    assert torch.equal(grad_kept, grad_cut)


def test_inner_lr_is_a_hyperparameter_not_a_trained_parameter():
    engine = ModularTitansEngine("Vector", **ENGINE_KWARGS)
    assert engine.inner_lr == 0.05
    assert not any("inner_lr" in name for name, _ in engine.named_parameters())


def test_one_optimizer_step_trains_both_cores():
    """The end-to-end path: forward, loss, backward, step, loss goes down."""
    for core_type in ("MLP", "Vector"):
        torch.manual_seed(7)
        engine = ModularTitansEngine(core_type, **ENGINE_KWARGS)
        optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-2)

        target = torch.randn(BATCH, SEQ, D_MODEL)
        losses = []
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            output = engine(target)
            loss = torch.nn.functional.mse_loss(output, target) + engine.last_surprise_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())

        assert losses[-1] < losses[0], f"{core_type} did not improve: {losses}"
