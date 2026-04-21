#!/usr/bin/env python3
"""Download benchmark datasets through the datasets library."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a benchmark dataset.")
    parser.add_argument("--name", default="hotpotqa", help="Logical dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to save.")
    parser.add_argument(
        "--output",
        default="data/raw/hotpotqa_train.jsonl",
        help="Target JSONL path.",
    )
    return parser.parse_args()


def resolve_dataset(name: str):
    normalized = name.lower()
    if normalized == "hotpotqa":
        return "hotpot_qa", "fullwiki"
    if normalized == "musique":
        return "dgslibisey/MuSiQue"
    raise ValueError(f"Unsupported dataset name: {name}")


def main() -> None:
    args = parse_args()
    dataset_spec = resolve_dataset(args.name)
    dataset = load_dataset(*dataset_spec, split=args.split)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(output_path), lines=True, force_ascii=False)
    print(f"Saved {len(dataset)} rows to {output_path}")


if __name__ == "__main__":
    main()
