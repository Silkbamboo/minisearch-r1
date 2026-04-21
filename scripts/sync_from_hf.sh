#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <provider:hf|ms> <model_id> [cache_dir]"
  exit 1
fi

PROVIDER="$1"
MODEL_ID="$2"
CACHE_DIR="${3:-/root/models}"

case "${PROVIDER}" in
  hf)
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${MODEL_ID}", local_dir="${CACHE_DIR}/${MODEL_ID}")
print("Downloaded ${MODEL_ID} to ${CACHE_DIR}/${MODEL_ID}")
PY
    ;;
  ms)
    python - <<PY
from modelscope import snapshot_download
snapshot_download("${MODEL_ID}", cache_dir="${CACHE_DIR}")
print("Downloaded ${MODEL_ID} to ${CACHE_DIR}")
PY
    ;;
  *)
    echo "Unknown provider: ${PROVIDER}"
    exit 1
    ;;
esac
