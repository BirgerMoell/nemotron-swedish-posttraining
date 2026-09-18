#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
MODEL_DIR="$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DATA_PATH="$ASSET_ROOT/data/s2-nemotron3-30b-a3b-post-lora-sv-smoke.jsonl"

cd "$REPO_ROOT"
mkdir -p logs runs
test -s "$MODEL_DIR/config.json"
test -s "$MODEL_DIR/model-00013-of-00013.safetensors"
test -s "$DATA_PATH"
sbatch --parsable --export=ALL,ASSET_ROOT="$ASSET_ROOT" lumi/s2_30b_smoke.sbatch
