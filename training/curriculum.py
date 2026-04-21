"""Curriculum helpers for hop-based scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    dataset_path: Path
    max_hops: int


def build_default_curriculum(data_dir: str = "data/processed/hops") -> list[CurriculumStage]:
    root = Path(data_dir)
    return [
        CurriculumStage("easy", root / "1_hop.jsonl", 1),
        CurriculumStage("medium", root / "2_hop.jsonl", 2),
        CurriculumStage("hard", root / "3_hop.jsonl", 3),
        CurriculumStage("mixed", root / "4_hop.jsonl", 4),
    ]
