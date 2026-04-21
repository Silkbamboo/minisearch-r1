#!/usr/bin/env python3
"""Config-driven SFT entrypoint with a local dry-run mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and paths without starting training.",
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
    return warnings


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    warnings = validate_paths(config)

    print(f"Loaded config: {args.config}")
    print(f"Model path: {config['model_name_or_path']}")
    print(f"Output dir: {config['output_dir']}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    if args.dry_run:
        print("Dry run complete. No training started.")
        return

    raise SystemExit(
        "Real SFT training is intentionally not implemented yet. "
        "Use --dry-run locally, then replace this entrypoint with the final AutoDL trainer."
    )


if __name__ == "__main__":
    main()
