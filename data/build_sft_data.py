#!/usr/bin/env python3
"""Build a simple SFT dataset from benchmark-style QA data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> "
    "and </think> first every time you get new information. After reasoning, if you find you "
    "lack some knowledge, you can call a search engine by <search> query </search>, and it "
    "will return the top searched results between <information> and </information>. You can "
    "search as many times as you want. If you find no further external knowledge needed, you "
    "can directly provide the answer inside <answer> and </answer> without detailed "
    "illustrations. For example, <answer> xxx </answer>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT training data.")
    parser.add_argument("--input", required=True, help="Input benchmark JSONL path.")
    parser.add_argument("--output", required=True, help="Output SFT JSONL path.")
    return parser.parse_args()


def extract_question(example: dict[str, Any]) -> str:
    return example.get("question", "").strip()


def extract_answer(example: dict[str, Any]) -> str:
    answer = example.get("answer")
    if isinstance(answer, list):
        return ", ".join(str(item) for item in answer)
    return str(answer).strip()


def convert_example(example: dict[str, Any]) -> dict[str, Any]:
    question = extract_question(example)
    answer = extract_answer(example)
    assistant = (
        "<think>Let me find information to answer this question.</think>\n"
        f"<answer>{answer}</answer>"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source": example.get("id"),
            "answer": answer,
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            example = json.loads(line)
            converted = convert_example(example)
            dst.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Wrote SFT data to {output_path}")


if __name__ == "__main__":
    main()
