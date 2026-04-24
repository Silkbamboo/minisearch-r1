# Local M2 Workflow

This document describes the recommended local-only M2 workflow for macOS development.

## Goal

Do not process the full KILT Wikipedia corpus locally.

Instead, use a small benchmark subset to validate:

- raw benchmark download
- SFT data conversion
- GRPO-style example generation
- hop-based curriculum splits
- local dev corpus creation
- retriever smoke test
- training config dry-run

## Recommended Commands

### 1. Download a small benchmark subset

```bash
python data/download_benchmarks.py \
  --name hotpotqa \
  --split train \
  --limit 512 \
  --output data/raw/hotpotqa_train.dev.jsonl
```

Offline fallback:

Use `data/fixtures/hotpotqa_dev_sample.jsonl` if you want to validate the flow without any network download.

### 2. Build local processed artifacts

```bash
python data/prepare_local_m2.py \
  --input data/raw/hotpotqa_train.dev.jsonl \
  --output-dir data/processed \
  --train-size 384 \
  --eval-size 64 \
  --max-corpus-docs 2000
```

Generated files:

- `data/processed/sft_train.jsonl`
- `data/processed/sft_eval.jsonl`
- `data/processed/grpo_train.jsonl`
- `data/processed/grpo_eval.jsonl`
- `data/processed/hops/*.jsonl`
- `data/processed/corpus.jsonl`
- `data/processed/summary.json`

### 3. Prepare BM25 input if needed

```bash
python retriever/build_bm25.py \
  --input data/processed/corpus.jsonl \
  --output-dir indexes/bm25_dev
```

This only prepares the Pyserini-ready corpus file. Full-scale Lucene indexing is still an AutoDL job.

### 4. Smoke-test the retriever

```bash
python retriever/server.py --corpus data/processed/corpus.jsonl
```

Then in another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which city hosted the 1992 Summer Olympics?","topk":3}'
```

### 5. Validate training entrypoints

```bash
python training/sft_train.py --dry-run
python training/grpo_train.py --dry-run
```

If the processed files exist, dry-run warnings should be minimal or gone.

## When To Move To AutoDL

Switch to AutoDL when you need any of the following:

- full KILT Wikipedia download
- large BM25 or FAISS indexing
- embedding generation over large corpora
- SFT training
- GRPO training

Local M2 is for correctness and workflow validation. AutoDL is for scale.
