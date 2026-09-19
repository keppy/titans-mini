# Titans-Mini — a swappable test-time memory core

A streaming token engine over a memory that is *updated during the forward pass*,
with the memory paradigm factored out so two can be compared without touching the
loop. The specification is a scaled-down Titans; this repo is a clean-room PyTorch
implementation of it, small enough to read in one sitting and to debug on a CPU.

```
titans_mini/
  config.py            hyperparameters (batch 16, seq 512, d_model 128, d_mem 256, inner_lr 0.05)
  core.py              MemoryCore interface + registry + the shared per-sequence MSE
  engine.py            ModularTitansEngine: the one loop over time
  cores/mlp_core.py    Option A — memory is the weights of a 2-layer MLP
  cores/vector_core.py Option B — memory is a flat vector, weights are generated from it
tests/                 21 tests: shapes, state layout, and every autograd claim below
scripts/smoke_train.py both cores end to end at spec scale
```

## The interface

```python
retrieve(memory_state, query_t) -> retrieved_value
update(memory_state, key_t, value_t, inner_lr) -> (new_memory_state, surprise_loss)
```

Plus `init_memory_state(batch_size)` for step 0. `memory_state` is opaque to the
engine — a dict of tensors for A, one tensor for B — and the engine only ever hands
it back to the core that made it, so no core's state format leaks into another's.

```python
from titans_mini import ModularTitansEngine, TitansConfig

engine = ModularTitansEngine("Vector", d_model=128, d_mem_vector=256, inner_lr=0.05)
output = engine(embeddings)          # [B, T, d_model] -> [B, T, d_model]
surprise = engine.last_surprise_loss # scalar; the memory's own prediction error
```

`inner_lr` is a plain float, never a parameter: it is a hyperparameter of the update
rule, and letting the outer optimizer tune it would reshape the dynamics the
experiment exists to measure. Swap the core by changing the string; add one by
subclassing `MemoryCore` and decorating with `@register_memory_core("name")`.

## Option A — the MLP core

Memory is the weights and biases of a two-layer MLP, **one MLP per sequence**
(`weight_in [B, d, h]`, `bias_in [B, h]`, `weight_out [B, h, d]`, `bias_out [B, d]`).
A read is a forward pass through those weights; a write is one gradient-descent step
on the key→value reconstruction error.

The two mechanisms that make this work rather than quietly sever the outer graph:

- **`torch.func.functional_call`** binds the weight dict onto a stateless module on
  every step, so the read is an ordinary autograd-traced computation that is
  differentiable in both the query (the stream) and the weights (the memory).
- **`torch.func.grad`** differentiates *a function of the weight dict*, and the update
  `w - lr * g` is then plain tensor arithmetic. Nothing calls `.backward()` on the
  surprise, no optimizer owns the memory, no `.data`/`detach_()` tricks are involved,
  and module state is never mutated.

The declared `DynamicMLP` is a *template*, not a model: `functional_call` replaces it
wholesale, so it never enters the graph and never receives a gradient. Its operands
are registered as **non-persistent buffers** — not parameters (no optimizer and no
weight decay ever sees them, so nothing can drift the memory's initialization) and
not persistent state (no checkpoint carries them). `test_template_is_neither_a_parameter_nor_checkpointed_state`
is the assertion of that, alongside
`test_template_module_is_never_in_the_graph_or_updated`.

Every op is an explicit `einsum` so the leading batch dimension of the dynamic weights
is honoured — these are per-sequence memories, not one shared MLP.

Consequence worth stating, because it is the difference between the options: **A
cannot send outer gradient into its own update step.** The step is defined by a
derivative taken inside it, so the chain through memory is cut by construction.
`inner_lr` per sequence is exactly the spec's `w - inner_lr * grad(w)`; the scalar
`surprise` reported is the batch mean of the same per-sequence errors.

## Option B — vector memory ("data-as-parameters")

Memory is a flat vector per sequence, `[B, d_mem_vector]`, and the weights are
*generated* from it by static projections:

- `hyper_net` — memory → `(W_temp [B,d,d], B_temp [B,d])`; read is
  `gelu(query @ W_temp + B_temp)`,
- `state_mutator` — `(memory, key, value)` → `ΔM`; write is
  `LayerNorm(memory + inner_lr * ΔM)`,
- `memory_norm` — the LayerNorm, with learned weight/bias.

`hyper_net`, `state_mutator` and `memory_norm` are ordinary `nn.Parameter`-bearing
modules, trained by the outer optimizer and never touched by the update rule. The
memory vector is the exact opposite: it is only ever a tensor produced and consumed
inside `forward`, is not a parameter, is not a buffer, and never appears in
`state_dict()` — `test_memory_state_is_an_unregistered_hidden_tensor` asserts that,
and asserts the key set does not change with batch or sequence length.

Generated weights are scaled by `1/sqrt(d_model)` on the way into the virtual layer.
Without it the pre-activation of a square layer grows with `d_model` and saturates
the GeLU; this is a stability constant, not a learned one.

## `detach_memory` — the flag that decides whether the mutator can learn

Default `False`: the graph is kept across steps, so the outer loss reaches
`state_mutator` through the memory chain. That is what trains the update rule — its
only path to the loss *is* the stream — and it is the capability Option A does not
have. Set it `True` for O(1) memory and no BPTT through the chain; then
`state_mutator` receives no gradient at all while `hyper_net` still does (both
behaviours are pinned by tests, so the tradeoff can't silently change).

## Running it

```bash
uv sync
.venv/Scripts/python.exe -m pytest -q            # 20 tests, CPU, ~6s
.venv/Scripts/python.exe scripts/smoke_train.py  # both cores at spec scale, random tokens
```

The smoke run is a smoke, not a result: the tokens are random, so the loss numbers
mean nothing. It exists to prove the wiring at spec shapes and to report where the
time goes (the MLP core backpropagates through 512 inner gradient steps per outer
step; the vector core is 512 cheap transitions).

## What this repo deliberately does not do

No attention, no positional encoding, no training data, no Modal/GPU path, no
distributed anything. It is the memory-core experiment: the engine owns the loop, the
core owns storage, and everything else is out of scope. If a core here proves out, it
is meant to be lifted into a real model as a pluggable component — that is what the
registry and the narrow interface are for.
