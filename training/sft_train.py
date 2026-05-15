#!/usr/bin/env python3
"""SFT entrypoint with a low-resource local smoke-test mode.

Two intended usage modes:

1. --dry-run
   Validate config and dataset paths without doing any training.
   Does NOT require torch.

2. --smoke-test
   Run a tiny offline training loop on a local fixture dataset using a randomly
   initialised miniature GRU model (CPU only). This verifies that tokenisation,
   batching, loss computation, and optimiser steps can run locally with negligible
   resource usage. Requires torch (CPU-only install is sufficient).

The actual full SFT path for AutoDL is intentionally left as a future
implementation requiring Unsloth + TRL + QLoRA.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_sft_data import convert_example

FIXTURE_PATH = PROJECT_ROOT / "data/fixtures/hotpotqa_dev_sample.jsonl"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and paths without starting training. No torch needed.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny offline training loop for local pipeline validation (needs torch).",
    )
    parser.add_argument("--smoke-max-steps", type=int, default=2)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--smoke-max-length", type=int, default=128)
    parser.add_argument("--smoke-output-dir", default="outputs/smoke_sft")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as src:
        return yaml.safe_load(src)


def validate_paths(config: dict) -> list[str]:
    warnings = []
    for key in ("train_file", "eval_file"):
        p = Path(config[key])
        if not p.exists():
            warnings.append(f"Missing {key}: {p}")
    return warnings


# ---------------------------------------------------------------------------
# Smoke-test implementation (all torch usage is confined here)
# ---------------------------------------------------------------------------

def _whitespace_tokenize(text: str) -> list[str]:
    return text.replace("\n", " \n ").split()


def _build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, EOS_TOKEN: 3}
    counter: Counter = Counter()
    for text in texts:
        counter.update(_whitespace_tokenize(text))
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def _render_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        parts.append(f"<{msg['role']}>\n{msg['content'].strip()}\n</{msg['role']}>")
    return "\n".join(parts)


def _load_fixture_examples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Smoke-test fixture not found: {path}")
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    if len(rows) < 2:
        raise ValueError("Fixture must have at least 2 rows.")
    return rows


def run_smoke_test(args: argparse.Namespace) -> None:
    # Lazy torch import — only needed for smoke-test, not for --dry-run.
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        raise SystemExit(
            "torch is not installed. "
            "For --smoke-test install it with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    # ---- inline Dataset class ----
    class TokenizedTextDataset(Dataset):
        def __init__(self, records: list[dict]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, idx: int) -> dict:
            return self.records[idx]

    # ---- inline tiny model ----
    class TinyCausalLM(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, attention_mask=None, labels=None):
            embedded = self.embedding(input_ids)
            hidden_states, _ = self.rnn(embedded)
            logits = self.lm_head(hidden_states)
            output = {"logits": logits}
            if labels is not None:
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                output["loss"] = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            return output

    # ---- build dataset ----
    def encode_text(text: str, vocab: dict[str, int], max_len: int) -> dict:
        tokens = [BOS_TOKEN] + _whitespace_tokenize(text) + [EOS_TOKEN]
        ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens[:max_len]]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(vocab[PAD_TOKEN])
            mask.append(0)
        labels = [i if m else -100 for i, m in zip(ids, mask)]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    rows = _load_fixture_examples(FIXTURE_PATH)
    converted = [convert_example(row) for row in rows]
    texts = [_render_messages(r["messages"]) for r in converted]
    vocab = _build_vocab(texts)

    train_ds = TokenizedTextDataset([encode_text(t, vocab, args.smoke_max_length) for t in texts[1:]])
    eval_ds = TokenizedTextDataset([encode_text(t, vocab, args.smoke_max_length) for t in texts[:1]])

    device = torch.device("cpu")
    model = TinyCausalLM(vocab_size=len(vocab)).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.smoke_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    step_losses: list[float] = []
    train_iter = iter(train_loader)
    for step in range(args.smoke_max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        out["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_val = float(out["loss"].detach())
        step_losses.append(loss_val)
        print(f"smoke-step {step + 1}/{args.smoke_max_steps}  loss={loss_val:.4f}")

    model.eval()
    eval_losses = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            eval_losses.append(float(model(**batch)["loss"]))
    eval_loss = sum(eval_losses) / max(len(eval_losses), 1)

    metrics = {
        "mode": "smoke-test",
        "train_examples": len(train_ds),
        "eval_examples": len(eval_ds),
        "vocab_size": len(vocab),
        "max_steps": args.smoke_max_steps,
        "train_loss_last": step_losses[-1],
        "eval_loss": eval_loss,
    }
    out_dir = Path(args.smoke_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Smoke-test complete.")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    warnings = validate_paths(config)

    print(f"Config:     {args.config}")
    print(f"Model path: {config['model_name_or_path']}")
    print(f"Output dir: {config['output_dir']}")
    for w in warnings:
        print(f"WARNING: {w}")

    if args.dry_run:
        print("Dry-run complete. No training started.")
        return

    if args.smoke_test:
        run_smoke_test(args)
        return

    raise SystemExit(
        "Full SFT requires Unsloth + TRL on AutoDL. "
        "Use --dry-run for config validation or --smoke-test for local pipeline check."
    )


if __name__ == "__main__":
    main()
