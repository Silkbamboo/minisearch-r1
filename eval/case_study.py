#!/usr/bin/env python3
"""Surface low-quality predictions for manual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import exact_match, token_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract hard cases from predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--output", default="outputs/hard_cases.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with Path(args.predictions).open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            row["em"] = exact_match(row["prediction"], row["reference"])
            row["f1"] = token_f1(row["prediction"], row["reference"])
            rows.append(row)

    rows.sort(key=lambda item: (item["em"], item["f1"]))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        for row in rows[: args.limit]:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {min(len(rows), args.limit)} hard cases to {output_path}")


if __name__ == "__main__":
    main()
