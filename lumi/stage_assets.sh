#!/usr/bin/env bash
set -euo pipefail

module purge

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CONTAINER="${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
HF_CACHE="${HF_CACHE:-/scratch/project_465002530/users/bmoell/hf_cache}"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray

mkdir -p "$SCRATCH_ROOT" "$HF_CACHE"
singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  export HF_HOME='$HF_CACHE' HF_DATASETS_CACHE='$HF_CACHE/datasets'
  cd '$REPO_ROOT'
  python scripts/stage_assets.py \
    --config configs/lumi-smoke.json \
    --asset-root '$SCRATCH_ROOT'
"

