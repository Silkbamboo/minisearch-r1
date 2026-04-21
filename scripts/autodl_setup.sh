#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/5] Update apt metadata"
apt-get update

echo "[2/5] Install system packages"
apt-get install -y git tmux openjdk-21-jdk

echo "[3/5] Upgrade pip"
python -m pip install --upgrade pip

echo "[4/5] Install core training stack"
python -m pip install "unsloth==2025.9.0"
python -m pip install "vllm==0.6.3"

echo "[5/5] Install project requirements"
python -m pip install -r "${ROOT_DIR}/requirements-server.txt"

echo "AutoDL bootstrap complete."
echo "Verify with:"
echo "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())\""
