#!/usr/bin/env python3
"""Minimal retriever service with RRF fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retriever.rrf import reciprocal_rank_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local retriever service.")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_corpus(path: str) -> list[dict]:
    corpus_path = Path(path)
    if not corpus_path.exists():
        return []
    rows = []
    with corpus_path.open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            rows.append(
                {
                    "doc_id": str(row.get("doc_id") or row.get("id")),
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }
            )
    return rows


def lexical_search(query: str, corpus: list[dict], top_k: int) -> list[dict]:
    tokens = {token for token in query.lower().split() if token}
    scored = []
    for row in corpus:
        content = f"{row['title']} {row['text']}".lower()
        overlap = sum(1 for token in tokens if token in content)
        if overlap > 0:
            scored.append({**row, "score": float(overlap)})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def dense_stub(query: str, corpus: list[dict], top_k: int) -> list[dict]:
    if not query:
        return []
    # Placeholder ranking for local smoke testing before a real FAISS backend is attached.
    return lexical_search(query, corpus, top_k)


def create_app(corpus: list[dict], top_k: int) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "documents": len(corpus)})

    @app.post("/search")
    def search():
        payload = request.get_json(force=True, silent=True) or {}
        query = str(payload.get("query", "")).strip()
        lexical = lexical_search(query, corpus, top_k)
        dense = dense_stub(query, corpus, top_k)
        fused = reciprocal_rank_fusion([lexical, dense])[:top_k]
        return jsonify({"query": query, "results": fused})

    return app


def main() -> None:
    args = parse_args()
    corpus = load_corpus(args.corpus)
    app = create_app(corpus, args.top_k)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
