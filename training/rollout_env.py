"""Minimal multi-turn rollout environment interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


RetrieverFn = Callable[[str, int], list[dict]]


@dataclass
class TurnState:
    query: str
    retrieved_docs: list[dict] = field(default_factory=list)
    answer: str = ""
    done: bool = False


class RolloutEnvironment:
    """A small abstraction that later training code can plug into."""

    def __init__(self, retriever: RetrieverFn, max_turns: int = 4) -> None:
        self.retriever = retriever
        self.max_turns = max_turns

    def reset(self, question: str) -> TurnState:
        return TurnState(query=question)

    def retrieve(self, state: TurnState, top_k: int) -> TurnState:
        state.retrieved_docs = self.retriever(state.query, top_k)
        return state

    def finish(self, state: TurnState, answer: str) -> TurnState:
        state.answer = answer
        state.done = True
        return state
