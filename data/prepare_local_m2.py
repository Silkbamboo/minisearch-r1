#!/usr/bin/env python3
"""Prepare small local M2 artifacts from a benchmark JSONL file."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.build_sft_data import convert_example
from data.split_by_hops import infer_hops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local M2 development artifacts.")
    parser.add_argument("--input", required=True, help="Input benchmark JSONL path.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to store processed outputs.",
    )
    parser.add_argument("--train-size", type=int, default=384)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--max-corpus-docs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(example: dict[str, Any]) -> str:
    answer = example.get("answer", "")
    if isinstance(answer, list):
        return ", ".join(str(item) for item in answer)
    return str(answer).strip()


def normalize_context(example: dict[str, Any]) -> list[dict[str, str]]:
    context = example.get("context", [])
    normalized: list[dict[str, str]] = []
    # HuggingFace datasets format: {"title": [...], "sentences": [[...], ...]}
    if isinstance(context, dict) and "title" in context and "sentences" in context:
        for title, sentences in zip(context["title"], context["sentences"]):
            text = " ".join(str(s) for s in sentences).strip() if isinstance(sentences, list) else str(sentences).strip()
            if title or text:
                normalized.append({"title": str(title).strip(), "text": text})
        return normalized
    for item in context:
        title = ""
        text = ""
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            if "sentences" in item and isinstance(item["sentences"], list):
                text = " ".join(str(sentence) for sentence in item["sentences"]).strip()
            else:
                text = str(item.get("text", "")).strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title = str(item[0]).strip()
            payload = item[1]
            if isinstance(payload, list):
                text = " ".join(str(sentence) for sentence in payload).strip()
            else:
                text = str(payload).strip()
        if title or text:
            normalized.append({"title": title, "text": text})
    return normalized


def extract_gold_titles(example: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    supporting = example.get("supporting_facts", [])
    # HuggingFace datasets format: {"title": [...], "sent_id": [...]}
    if isinstance(supporting, dict) and "title" in supporting:
        return sorted(set(str(t) for t in supporting["title"]))
    for item in supporting:
        if isinstance(item, (list, tuple)) and item:
            titles.append(str(item[0]))
        elif isinstance(item, dict) and "title" in item:
            titles.append(str(item["title"]))
    return sorted(set(titles))


def to_grpo_example(example: dict[str, Any], index: int) -> dict[str, Any]:
    context = normalize_context(example)
    return {
        "id": str(example.get("id", f"example-{index}")),
        "question": str(example.get("question", "")).strip(),
        "answer": normalize_answer(example),
        "num_hops": infer_hops(example),
        "gold_titles": extract_gold_titles(example),
        "context": context,
    }


def build_corpus(examples: list[dict[str, Any]], max_docs: int) -> list[dict[str, str]]:
    corpus: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    next_id = 0

    for example in examples:
        for item in normalize_context(example):
            key = (item["title"], item["text"])
            if key in seen:
                continue
            seen.add(key)
            corpus.append(
                {
                    "doc_id": f"doc-{next_id}",
                    "title": item["title"],
                    "text": item["text"],
                }
            )
            next_id += 1
            if len(corpus) >= max_docs:
                return corpus
    return corpus


def ensure_enough_rows(rows: list[dict[str, Any]], train_size: int, eval_size: int) -> tuple[int, int]:
    total_requested = train_size + eval_size
    if total_requested <= len(rows):
        return train_size, eval_size
    if len(rows) < 2:
        raise ValueError("Need at least 2 rows to build train and eval splits.")
    adjusted_eval = max(1, min(eval_size, len(rows) // 5))
    adjusted_train = max(1, len(rows) - adjusted_eval)
    return adjusted_train, adjusted_eval


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    train_size, eval_size = ensure_enough_rows(rows, args.train_size, args.eval_size)

    rng = random.Random(args.seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size : eval_size + train_size]
    output_dir = Path(args.output_dir)
    hops_dir = output_dir / "hops"

    sft_train = [convert_example(row) for row in train_rows]
    sft_eval = [convert_example(row) for row in eval_rows]
    grpo_train = [to_grpo_example(row, idx) for idx, row in enumerate(train_rows)]
    grpo_eval = [to_grpo_example(row, idx) for idx, row in enumerate(eval_rows)]
    corpus = build_corpus(train_rows + eval_rows, args.max_corpus_docs)

    dump_jsonl(output_dir / "sft_train.jsonl", sft_train)
    dump_jsonl(output_dir / "sft_eval.jsonl", sft_eval)
    dump_jsonl(output_dir / "grpo_train.jsonl", grpo_train)
    dump_jsonl(output_dir / "grpo_eval.jsonl", grpo_eval)
    dump_jsonl(output_dir / "corpus.jsonl", corpus)

    hop_buckets: dict[int, list[dict[str, Any]]] = {hop: [] for hop in range(1, 5)}
    for row in grpo_train:
        hop_buckets.setdefault(int(row["num_hops"]), []).append(row)
    for hop_count, bucket in sorted(hop_buckets.items()):
        dump_jsonl(hops_dir / f"{hop_count}_hop.jsonl", bucket)

    summary = {
        "input": str(input_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "corpus_documents": len(corpus),
        "hop_counts": {str(hop): len(bucket) for hop, bucket in sorted(hop_buckets.items())},
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
