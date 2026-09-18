#!/usr/bin/env bash
set -euo pipefail

module purge

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CONTAINER="${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
HF_CACHE="${HF_CACHE:-/scratch/project_465002530/users/bmoell/hf_cache}"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray
CANDIDATES="$ASSET_ROOT/data/s2-nemotron3-30b-a3b-post-lora-sv-smoke.jsonl"
ACCEPTED="$ASSET_ROOT/data/s2-nemotron3-30b-a3b-post-lora-sv-smoke-accepted.jsonl"
WINDOW_MANIFEST="$ASSET_ROOT/manifests/s2-nemotron3-30b-a3b-post-lora-sv-smoke-window.json"

mkdir -p "$ASSET_ROOT" "$HF_CACHE"
singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export HF_HOME='$HF_CACHE' HF_DATASETS_CACHE='$HF_CACHE/datasets'
  cd '$REPO_ROOT'
  python scripts/stage_assets.py \
    --config configs/lumi-s2-30b-post-smoke.json \
    --asset-root '$ASSET_ROOT'
  python scripts/prepare_windowed_data.py \
    --config configs/lumi-s2-30b-post-smoke.json \
    --model-dir '$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16' \
    --candidates '$CANDIDATES' \
    --output '$ACCEPTED' \
    --manifest '$WINDOW_MANIFEST'
"
