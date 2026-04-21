"""Reciprocal rank fusion helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[dict]],
    *,
    k: int = 60,
    id_key: str = "doc_id",
) -> list[dict]:
    """Fuse multiple ranked lists using reciprocal rank fusion."""
    scores: dict[str, float] = defaultdict(float)
    payload: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = str(item[id_key])
            scores[doc_id] += 1.0 / (k + rank)
            payload.setdefault(doc_id, item)

    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        doc = dict(payload[doc_id])
        doc["rrf_score"] = score
        fused.append(doc)
    return fused
