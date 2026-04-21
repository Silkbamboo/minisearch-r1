#!/usr/bin/env python3
"""Prepare a Pyserini-compatible corpus file for BM25 indexing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BM25 corpus.")
    parser.add_argument("--input", required=True, help="Input corpus JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Output corpus directory.")
    return parser.parse_args()


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    doc_id = str(row.get("doc_id") or row.get("id"))
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    contents = f"{title}\n{text}".strip()
    return {"id": doc_id, "contents": contents}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"

    count = 0
    with Path(args.input).open("r", encoding="utf-8") as src, corpus_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            row = normalize_row(json.loads(line))
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Prepared {count} BM25 documents at {corpus_path}")
    print("Next step:")
    print(
        "python -m pyserini.index.lucene "
        "--collection JsonCollection "
        f"--input {output_dir} "
        f"--index {output_dir / 'index'} "
        "--generator DefaultLuceneDocumentGenerator "
        "--threads 4"
    )


if __name__ == "__main__":
    main()
