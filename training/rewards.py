"""Reward helpers for search-augmented reasoning training."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def answer_match_reward(prediction: str, reference: str) -> float:
    return float(normalize_text(reference) in normalize_text(prediction))


def format_reward(prediction: str) -> float:
    has_plan = "<plan>" in prediction and "</plan>" in prediction
    has_answer = "<answer>" in prediction and "</answer>" in prediction
    return float(has_plan and has_answer)


def retrieval_usage_reward(retrieved_docs: list[dict]) -> float:
    return 1.0 if retrieved_docs else 0.0


def total_reward(
    prediction: str,
    reference: str,
    retrieved_docs: list[dict],
    *,
    answer_weight: float = 1.0,
    format_weight: float = 0.2,
    retrieval_weight: float = 0.3,
) -> float:
    return (
        answer_weight * answer_match_reward(prediction, reference)
        + format_weight * format_reward(prediction)
        + retrieval_weight * retrieval_usage_reward(retrieved_docs)
    )
