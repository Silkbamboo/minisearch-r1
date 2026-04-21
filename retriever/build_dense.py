#!/usr/bin/env python3
"""Build a dense FAISS index from a JSONL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dense FAISS index.")
    parser.add_argument("--input", required=True, help="Input corpus JSONL path.")
    parser.add_argument("--index-path", required=True, help="Output FAISS index path.")
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Output metadata JSONL path aligned with the index.",
    )
    parser.add_argument(
        "--model-name",
        default="intfloat/multilingual-e5-base",
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_corpus(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            text = f"{row.get('title', '')}\n{row.get('text', '')}".strip()
            rows.append({"doc_id": row.get("doc_id") or row.get("id"), "text": text})
    return rows


def main() -> None:
    args = parse_args()
    corpus = load_corpus(Path(args.input))
    model = SentenceTransformer(args.model_name)

    embeddings = []
    texts = [f"passage: {row['text']}" for row in corpus]
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[start : start + args.batch_size]
        vectors = model.encode(batch, normalize_embeddings=True)
        embeddings.append(vectors)

    matrix = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_path = Path(args.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    metadata_path = Path(args.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as dst:
        for row in corpus:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved dense index to {index_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
