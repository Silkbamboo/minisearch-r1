# MiniSearch-R1

MiniSearch-R1 is a local-first reproduction workspace for a search-augmented reasoning agent.
The workflow is designed around:

- local macOS development for code, data processing, and analysis
- GitHub as the single source of truth for code
- AutoDL for GPU training and inference
- ModelScope or Hugging Face for model and dataset transfer

## Development Workflow

1. Write and debug code locally on macOS.
2. Commit and push code changes to GitHub.
3. Pull the latest code on AutoDL and run training there.
4. Upload checkpoints to ModelScope or Hugging Face instead of storing them in Git.

## Local Setup

Recommended local scope:

- code authoring
- config management
- CPU-side data preprocessing
- small-sample retriever validation
- evaluation analysis

Training is intentionally deferred to AutoDL.

Install local dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

If you use `pyserini`, install Java 21 first:

```bash
brew install openjdk@21
```

## AutoDL Setup

Use a GPU image for training. A typical bootstrap flow is:

```bash
git pull
bash scripts/autodl_setup.sh
tmux new -s train
python training/grpo_train.py --config configs/grpo_config.yaml --dry-run
```

## Repository Layout

```text
minisearch-r1/
├── configs/
├── data/
├── retriever/
├── training/
├── eval/
├── scripts/
└── notebooks/
```

## Environment Variables

Copy `.env.example` to `.env` locally and on AutoDL, then fill in your real keys.

## Current State

This repository currently contains the M1 scaffold:

- project layout
- dependency split for local vs server
- config templates
- retriever and evaluation skeletons
- training entrypoint skeletons
- AutoDL bootstrap scripts
