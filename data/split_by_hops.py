#!/usr/bin/env python3
"""Split multi-hop datasets by hop count for curriculum learning."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset rows by hop count.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where split JSONL files will be stored.",
    )
    return parser.parse_args()


def infer_hops(example: dict[str, Any]) -> int:
    if "num_hops" in example:
        return int(example["num_hops"])
    supporting = example.get("supporting_facts")
    if isinstance(supporting, list) and supporting:
        return max(1, min(4, len(supporting)))
    context = example.get("context")
    if isinstance(context, list) and context:
        return max(1, min(4, len(context)))
    return 1


def main() -> None:
    args = parse_args()
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with Path(args.input).open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            buckets[infer_hops(row)].append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for hops, rows in sorted(buckets.items()):
        target = output_dir / f"{hops}_hop.jsonl"
        with target.open("w", encoding="utf-8") as dst:
            for row in rows:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} rows to {target}")


if __name__ == "__main__":
    main()
