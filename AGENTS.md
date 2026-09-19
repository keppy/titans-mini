# AGENTS.md — entry point for any agent or harness

Read this, then [README.md](README.md) for the design reasoning behind each option.

## What this repo is

A clean-room PyTorch implementation of the Titans-Mini specification: a streaming
token engine whose test-time **memory core is swappable**. Two cores exist —
Option A (memory is the weights of a 2-layer MLP) and Option B (memory is a flat
vector; the weights are generated from it). The experiment is the comparison, so the
interface between the engine and a core is the thing that must stay narrow:

    retrieve(memory_state, query_t)        -> retrieved_value
    update(memory_state, key_t, value_t, inner_lr) -> (new_memory_state, surprise_loss)

## Where state lives

| file | authority over |
|---|---|
| `README.md` | **design.** Why `functional_call` / `torch.func.grad`, why LayerNorm on the memory, what `detach_memory` buys and costs. |
| `titans_mini/config.py` | the hyperparameters from the spec. |
| `titans_mini/core.py` | the `MemoryCore` interface, the registry, the shared per-sequence MSE. |
| `titans_mini/engine.py` | the only loop over time. |
| `tests/` | every claim in the README, as an executable assertion. |

## Rules that are load-bearing

1. **The engine must not know which core it holds.** No `isinstance` checks on
   `memory_state`, no per-core branches in `engine.py`. If a new core needs something
   from the loop, widen `MemoryCore` — not the loop.
2. **The memory is never model state.** It is created inside `forward` and passed
   step to step. It must not become an `nn.Parameter` or a *persistent* buffer; a key
   appearing in `state_dict()` per forward is the failure mode this repo exists to
   avoid. Option A's template MLP is the one module that touches the memory and it is
   non-persistent buffers for exactly this reason — keep it that way.
   `tests/test_vector_core.py::test_memory_state_is_an_unregistered_hidden_tensor`
   and `tests/test_mlp_core.py::test_template_is_neither_a_parameter_nor_checkpointed_state`
   pin it.
3. **Never train the memory with `.backward()`.** Option A's inner step goes through
   `torch.func` (`functional_call` + `grad`) and applies `w - lr * g` as tensor
   arithmetic. An optimizer or a `.backward()` on the surprise loss would both
   mutate module state and sever the outer graph — the two things the spec asks for.
4. **Static parameters and dynamic state stay separate.** In Option B,
   `hyper_net` / `state_mutator` / `memory_norm` are trained; the memory vector is
   not. Gradients reaching the former and never the latter is the invariant.
5. **Log the why.** A change to the update rule, the surprise definition, or
   `detach_memory` changes what the comparison measures. Say which failure the
   change was made against.

## Traps already hit here

- **LayerNorm's eps is not negligible on a fresh memory.** `LayerNorm(memory + lr*ΔM)`
  on a near-zero memory gives std ≈ `sqrt(var/(var+eps))` ≈ 0.93, not 1.0. Not a bug;
  don't "fix" it by loosening a test into meaninglessness — assert the layer norm
  relation itself.
- **`functional_call` binds by name and by shape.** The template module's parameter
  shapes are the *memory state's* shapes minus the batch dimension, and the forward
  pass must use explicit `einsum` (batched weights + `F.linear` do not mean what they
  look like).
- **A read is only a graph node if one of its inputs requires grad.** A detached
  memory state plus a plain query silently produces a constant. Tests that assert
  gradient flow must feed `requires_grad=True` inputs, as the engine's projections do.
- **Every core takes the same construction arguments** (`d_model`, `d_mem_vector`,
  `hidden_dim`) even where one is unused — that uniformity is what lets the engine
  build any core from one config.
- **`functional_call` binds non-persistent buffers too.** That is what lets Option A's
  template be invisible to `parameters()` and `state_dict()` while still giving the
  memory dict its names and shapes; switching it back to `nn.Parameter` reintroduces
  dead weights in checkpoints.
- **No NumPy in this venv.** Don't reach for it in tests or scripts.

## Running it

```bash
uv sync
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/smoke_train.py
```

CPU-only by design: the shapes here are small, and this repo has no GPU path. Nothing
here runs on Modal or touches a training corpus.
