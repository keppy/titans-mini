"""Smoke run: both cores end to end at the specified hyperparameters.

This is a *smoke*, not a measurement: the data is random tokens, so the loss curve
says nothing about quality. What it does show is that each core (a) runs at spec
scale, (b) keeps its memory out of `state_dict()`, and (c) trains — the surprise the
memory is scored on and the outer loss both move.

    .venv/Scripts/python.exe scripts/smoke_train.py [--steps 3] [--cores MLP Vector]

The inner loop is 512 sequential steps with a gradient step inside each one for the
MLP core, so a single outer step backpropagates through 512 inner updates on the
CPU: expect seconds, not milliseconds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from titans_mini import ModularTitansEngine, TitansConfig  # noqa: E402

VOCAB = 256


class TinyLM(nn.Module):
    """The smallest thing that produces the engine's input: token ids -> embeddings."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, config.d_model)
        self.engine = ModularTitansEngine.from_config(config)
        self.lm_head = nn.Linear(config.d_model, VOCAB)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.engine(self.embed(tokens)))


def run_core(core_type: str, steps: int, batch_size: int, seq_len: int) -> None:
    config = TitansConfig(memory_core_type=core_type, batch_size=batch_size, seq_len=seq_len)
    torch.manual_seed(0)
    model = TinyLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    engine_state = set(model.engine.state_dict())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    memory_keys = sorted(key for key in engine_state if "memory_state" in key)
    print(f"\n=== {core_type} core ===")
    print(f"  trained parameters      : {trained:,}")
    print(f"  memory in engine state  : {memory_keys or 'none (created per forward)'}")

    tokens = torch.randint(0, VOCAB, (batch_size, seq_len + 1))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    for step in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        outer = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        surprise = model.engine.last_surprise_loss
        loss = outer + surprise
        loss.backward()
        optimizer.step()
        print(
            f"  step {step + 1}/{steps}  outer_ce {outer.item():.4f}  surprise {surprise.item():.4f}"
            f"  engine_state_keys {len(model.engine.state_dict())}  {time.perf_counter() - started:.2f}s"
        )

    assert set(model.engine.state_dict()) == engine_state, "the engine grew state during the run"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", nargs="+", default=["MLP", "Vector"])
    parser.add_argument("--steps", type=int, default=3)
    # Defaults come from TitansConfig, i.e. the specified hyperparameters.
    parser.add_argument("--batch-size", type=int, default=TitansConfig().batch_size)
    parser.add_argument("--seq-len", type=int, default=TitansConfig().seq_len)
    args = parser.parse_args()

    print(f"device {torch.get_default_device() or 'cpu'} · torch {torch.__version__}")
    print(f"batch {args.batch_size} · seq {args.seq_len} · vocab {VOCAB} · random tokens")
    for core_type in args.cores:
        run_core(core_type, args.steps, args.batch_size, args.seq_len)


if __name__ == "__main__":
    main()
