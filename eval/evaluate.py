#!/usr/bin/env python3
"""Batch evaluation script for prediction JSONL files."""

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
    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL path.")
    parser.add_argument("--output", default="outputs/metrics.json", help="Metrics path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    em_sum = 0.0
    f1_sum = 0.0

    with Path(args.predictions).open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            prediction = row["prediction"]
            reference = row["reference"]
            em_sum += exact_match(prediction, reference)
            f1_sum += token_f1(prediction, reference)
            total += 1

    metrics = {
        "count": total,
        "exact_match": em_sum / max(total, 1),
        "token_f1": f1_sum / max(total, 1),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
