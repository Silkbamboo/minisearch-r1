"""Multi-turn rollout environment for search-augmented reasoning.

Drives one complete rollout:
  1. Build the initial prompt from the system prompt + question.
  2. Call the model generate function to produce tokens.
  3. Detect <search>query</search> tags and pause generation.
  4. Query the retrieval server, inject <information>...</information>.
  5. Resume generation until <answer>...</answer> or max_turns reached.
  6. Return the full text and a retrieval_mask (0 for injected tokens, 1 for model tokens).

The retrieval_mask is used by the training loop to zero out loss on injected content.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> "
    "and </think> first every time you get new information. After reasoning, if you find you "
    "lack some knowledge, you can call a search engine by <search> query </search>, and it "
    "will return the top searched results between <information> and </information>. You can "
    "search as many times as you want. If you find no further external knowledge needed, you "
    "can directly provide the answer inside <answer> and </answer> without detailed "
    "illustrations. For example, <answer> xxx </answer>."
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class GenerateFn(Protocol):
    """Any callable that takes a prompt string and returns generated text (not including the prompt)."""
    def __call__(self, prompt: str) -> str: ...


@dataclass
class RolloutResult:
    full_text: str               # complete sequence (prompt + all generated + injected content)
    retrieval_mask: list[int]    # per-character mask: 1=model-generated, 0=injected
    answer: str                  # extracted final answer (empty string if not found)
    num_searches: int            # number of search calls made


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _call_retriever(
    endpoint: str, query: str, topk: int, timeout: float = 3.0
) -> list[dict]:
    """POST to retrieval server; returns empty list on any error."""
    try:
        resp = requests.post(
            endpoint,
            json={"query": query, "topk": topk},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as exc:
        logger.warning("Retrieval failed for query %r: %s", query, exc)
        return []


def _format_results(results: list[dict]) -> str:
    """Format retrieval results into the <information> block injected into the sequence."""
    if not results:
        return "<information>\n(no results found)\n</information>\n"
    lines = []
    for i, doc in enumerate(results, 1):
        text = doc.get("contents", "").strip()
        if len(text) > 500:
            text = text[:500] + "..."
        lines.append(f"[{i}] {text}")
    return "<information>\n" + "\n".join(lines) + "\n</information>\n"


# ---------------------------------------------------------------------------
# Search tag parsing
# ---------------------------------------------------------------------------

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)


def _extract_search_query(text: str) -> str | None:
    """Return the last open (or closed) <search> query in text, or None."""
    # Try closed tag first
    matches = _SEARCH_RE.findall(text)
    if matches:
        return matches[-1].strip()
    # Try unclosed tag (model stopped mid-tag)
    m = re.search(r"<search>(.*?)$", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_answer(text: str) -> str:
    # Use findall to get all matches and take the last one,
    # because the system prompt contains an example <answer> tag.
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


# ---------------------------------------------------------------------------
# Main rollout function
# ---------------------------------------------------------------------------

def run_rollout(
    question: str,
    generate_fn: GenerateFn,
    retriever_endpoint: str = "http://127.0.0.1:8000/retrieve",
    max_turns: int = 3,
    topk: int = 3,
) -> RolloutResult:
    """Run a single multi-turn rollout for one question.

    Returns a RolloutResult with the full text, per-character retrieval_mask,
    extracted answer, and number of searches performed.
    """
    prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}"

    # We track the full sequence and which characters are model-generated vs injected.
    full_text = prompt
    # Prompt characters are model-visible but not trained on → mask=1 (will be handled
    # by the prompt_mask in the trainer; here we just track injected retrieval content).
    retrieval_mask = [1] * len(prompt)

    num_searches = 0

    for _turn in range(max_turns + 1):
        generated = generate_fn(full_text)

        # Append generated text (all model-generated → mask=1)
        full_text += generated
        retrieval_mask += [1] * len(generated)

        # Check if model produced a search query
        # We look only in the newly generated portion for a <search> tag
        query = _extract_search_query(generated)

        if query and num_searches < max_turns:
            # Close the <search> tag if model left it open
            if not re.search(r"</search>", generated):
                close_tag = "</search>\n"
                full_text += close_tag
                retrieval_mask += [1] * len(close_tag)

            # Retrieve and inject results
            results = _call_retriever(retriever_endpoint, query, topk)
            info_block = _format_results(results)

            full_text += info_block
            retrieval_mask += [0] * len(info_block)   # ← injected content: mask=0

            num_searches += 1
            continue  # give model another turn

        # No search → model is done (or forced to finish)
        break
    else:
        # Exceeded max_turns without <answer>: append a nudge (model-generated equivalent)
        nudge = "\n<answer>"
        full_text += nudge
        retrieval_mask += [1] * len(nudge)

    answer = _extract_answer(full_text)
    return RolloutResult(
        full_text=full_text,
        retrieval_mask=retrieval_mask,
        answer=answer,
        num_searches=num_searches,
    )


# ---------------------------------------------------------------------------
# Token-level mask builder (for training loop)
# ---------------------------------------------------------------------------

def build_token_retrieval_mask(
    retrieval_mask: list[int],
    token_offsets: list[tuple[int, int]],
) -> list[int]:
    """Convert character-level retrieval_mask to token-level mask.

    token_offsets: list of (char_start, char_end) for each token in full_text.
    A token is masked out (0) if ANY of its characters fall in an injected region.
    """
    token_mask = []
    for start, end in token_offsets:
        # If all characters in the token span are model-generated, keep it
        if all(retrieval_mask[i] == 1 for i in range(start, end)):
            token_mask.append(1)
        else:
            token_mask.append(0)
    return token_mask
