#!/usr/bin/env python3
"""SFT entrypoint with a low-resource local smoke-test mode.

Two intended usage modes:

1. --dry-run
   Validate config and dataset paths without doing any training.

2. --smoke-test
   Run a tiny offline training loop on a local fixture dataset using a randomly
   initialised miniature GPT-2 model. This is designed only to verify that the
   local training pipeline can start on a machine with limited disk and VRAM.

The actual full SFT path for AutoDL is still intentionally left as a future
implementation, because it will need the real model loading, LoRA/QLoRA, and
the final high-resource training stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_sft_data import convert_example

FIXTURE_PATH = PROJECT_ROOT / "data/fixtures/hotpotqa_dev_sample.jsonl"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


class TokenizedTextDataset(Dataset):
    def __init__(self, records: list[dict[str, torch.Tensor]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.records[index]


class TinyCausalLM(nn.Module):
    """A tiny causal LM used only for local smoke-testing.

    This is intentionally not the final training model. Its only job is to
    verify that tokenisation, batching, loss computation, and optimiser steps
    can run locally with negligible resource usage.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        hidden_states, _ = self.rnn(embedded)
        logits = self.lm_head(hidden_states)

        output = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and paths without starting training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny offline training loop for local pipeline validation.",
    )
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        default=2,
        help="Number of optimisation steps for the local smoke-test.",
    )
    parser.add_argument(
        "--smoke-batch-size",
        type=int,
        default=2,
        help="Batch size for the local smoke-test.",
    )
    parser.add_argument(
        "--smoke-max-length",
        type=int,
        default=128,
        help="Sequence length for the local smoke-test.",
    )
    parser.add_argument(
        "--smoke-output-dir",
        default="outputs/smoke_sft",
        help="Directory for tiny smoke-test metrics only. No model checkpoint is saved by default.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as src:
        return yaml.safe_load(src)


def validate_paths(config: dict) -> list[str]:
    warnings = []
    train_file = Path(config["train_file"])
    eval_file = Path(config["eval_file"])
    if not train_file.exists():
        warnings.append(f"Missing train file: {train_file}")
    if not eval_file.exists():
        warnings.append(f"Missing eval file: {eval_file}")
    return warnings


def load_fixture_examples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Smoke-test fixture not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < 2:
        raise ValueError("Smoke-test fixture must contain at least 2 rows.")
    return rows


def render_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"<{role}>\n{content}\n</{role}>")
    return "\n".join(parts)


def whitespace_tokenize(text: str) -> list[str]:
    return text.replace("\n", " \n ").split()


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        BOS_TOKEN: 2,
        EOS_TOKEN: 3,
    }
    counter = Counter()
    for text in texts:
        counter.update(whitespace_tokenize(text))
    for token, _count in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> dict[str, torch.Tensor]:
    tokens = [BOS_TOKEN] + whitespace_tokenize(text) + [EOS_TOKEN]
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens[:max_length]]
    attention_mask = [1] * len(token_ids)

    while len(token_ids) < max_length:
        token_ids.append(vocab[PAD_TOKEN])
        attention_mask.append(0)

    labels = []
    for token_id, is_active in zip(token_ids, attention_mask):
        labels.append(token_id if is_active else -100)

    return {
        "input_ids": torch.tensor(token_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_smoke_datasets(max_length: int) -> tuple[Dataset, Dataset, int]:
    examples = load_fixture_examples(FIXTURE_PATH)
    converted = [convert_example(row) for row in examples]
    texts = [render_messages(row["messages"]) for row in converted]
    vocab = build_vocab(texts)

    eval_texts = texts[:1]
    train_texts = texts[1:]

    train_records = [encode_text(text, vocab, max_length) for text in train_texts]
    eval_records = [encode_text(text, vocab, max_length) for text in eval_texts]
    return TokenizedTextDataset(train_records), TokenizedTextDataset(eval_records), len(vocab)


def evaluate_model(
    model: TinyCausalLM,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs["loss"].detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def run_smoke_test(args: argparse.Namespace) -> None:
    train_dataset, eval_dataset, vocab_size = build_smoke_datasets(args.smoke_max_length)
    device = torch.device("cpu")
    model = TinyCausalLM(vocab_size=vocab_size, hidden_size=64).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.smoke_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    train_iterator = iter(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    step_losses: list[float] = []

    for step in range(args.smoke_max_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = float(loss.detach().cpu())
        step_losses.append(loss_value)
        print(f"smoke-step {step + 1}/{args.smoke_max_steps} loss={loss_value:.4f}")

    eval_loss = evaluate_model(model, eval_loader, device)
    metrics = {
        "mode": "smoke-test",
        "device": str(device),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "vocab_size": vocab_size,
        "max_length": args.smoke_max_length,
        "max_steps": args.smoke_max_steps,
        "train_loss_last": step_losses[-1],
        "eval_loss": eval_loss,
    }

    output_dir = Path(args.smoke_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Smoke-test complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Metrics written to {metrics_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    warnings = validate_paths(config)

    print(f"Loaded config: {args.config}")
    print(f"Model path: {config['model_name_or_path']}")
    print(f"Output dir: {config['output_dir']}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    if args.dry_run:
        print("Dry run complete. No training started.")
        return

    if args.smoke_test:
        run_smoke_test(args)
        return

    raise SystemExit(
        "The full SFT path is intentionally still pending. "
        "Use --dry-run for config validation or --smoke-test for a tiny local training run. "
        "On AutoDL, this file should be upgraded to the final Unsloth/TRL training backend."
    )


if __name__ == "__main__":
    main()
