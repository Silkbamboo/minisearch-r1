#!/usr/bin/env python3
"""Config-driven GRPO entrypoint with a local dry-run mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.curriculum import build_default_curriculum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training.")
    parser.add_argument("--config", default="configs/grpo_config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and curriculum without starting training.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as src:
        return yaml.safe_load(src)


def validate_paths(config: dict) -> list[str]:
    warnings = []
    train_file = Path(config["train_file"])
    eval_file = Path(config["eval_file"])
    if not train_file.exists():
        warnings.append(f"Missing train file: {train_file}")
    if not eval_file.exists():
        warnings.append(f"Missing eval file: {eval_file}")
    for stage in build_default_curriculum():
        if not stage.dataset_path.exists():
            warnings.append(f"Missing curriculum file: {stage.dataset_path}")
    return warnings


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    curriculum = build_default_curriculum()
    warnings = validate_paths(config)

    print(f"Loaded config: {args.config}")
    print(f"Model path: {config['model_name_or_path']}")
    print(f"Retriever endpoint: {config['retriever_endpoint']}")
    print(f"Max rollout turns: {config['rollout']['max_turns']}")
    print("Curriculum stages:")
    for stage in curriculum:
        print(f"- {stage.name}: {stage.dataset_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    if args.dry_run:
        print("Dry run complete. No training started.")
        return

    raise SystemExit(
        "Real GRPO training is intentionally not implemented yet. "
        "Use --dry-run locally, then replace this entrypoint with the final AutoDL trainer."
    )


if __name__ == "__main__":
    main()
