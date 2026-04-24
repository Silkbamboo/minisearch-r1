"""Reward functions for GRPO/DAPO training.

Three rewards combined with weights [1.0, 0.2, 0.1]:
  f1_reward            – SQuAD-style token F1 (0.0~1.0)
  format_reward        – checks <think> and <answer> tags (0 or 1)
  search_quality_reward – correct answer + search efficiency (0/0.1/0.3)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Text normalisation (SQuAD standard)
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES.sub(" ", text)
    text = " ".join(text.split())
    return text


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _count_searches(text: str) -> int:
    return len(re.findall(r"<search>", text))


# ---------------------------------------------------------------------------
# Individual reward functions (TRL GRPOTrainer signature: list in, list out)
# ---------------------------------------------------------------------------

def f1_reward(completions: list[str], ground_truth: list[Any], **kwargs: Any) -> list[float]:
    """Token-level F1 between extracted <answer> and ground truth."""
    scores = []
    for completion, ref in zip(completions, ground_truth):
        pred = _extract_answer(completion)
        ref_str = ref if isinstance(ref, str) else str(ref)
        scores.append(_token_f1(pred, ref_str))
    return scores


def format_reward(completions: list[str], **kwargs: Any) -> list[float]:
    """1.0 if completion has <think>...</think> and <answer>...</answer> with paired <search> tags."""
    scores = []
    for completion in completions:
        has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
        open_tags = len(re.findall(r"<search>", completion))
        close_tags = len(re.findall(r"</search>", completion))
        tags_paired = open_tags == close_tags
        scores.append(float(has_think and has_answer and tags_paired))
    return scores


def search_quality_reward(
    completions: list[str], ground_truth: list[Any], **kwargs: Any
) -> list[float]:
    """Correct answer + efficient search → 0.3; correct but >3 searches → 0.1; wrong → 0."""
    scores = []
    for completion, ref in zip(completions, ground_truth):
        pred = _extract_answer(completion)
        ref_str = ref if isinstance(ref, str) else str(ref)
        correct = _token_f1(pred, ref_str) > 0.5
        n_searches = _count_searches(completion)
        if correct and n_searches <= 3:
            scores.append(0.3)
        elif correct:
            scores.append(0.1)
        else:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Combined reward (for unit testing / standalone use)
# ---------------------------------------------------------------------------

def combined_reward(
    completions: list[str],
    ground_truth: list[Any],
    *,
    answer_weight: float = 1.0,
    format_weight: float = 0.2,
    search_weight: float = 0.1,
) -> list[float]:
    f1 = f1_reward(completions, ground_truth)
    fmt = format_reward(completions)
    sq = search_quality_reward(completions, ground_truth)
    return [
        answer_weight * a + format_weight * b + search_weight * c
        for a, b, c in zip(f1, fmt, sq)
    ]
