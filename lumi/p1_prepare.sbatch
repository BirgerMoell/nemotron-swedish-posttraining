#!/usr/bin/env bash
#SBATCH --job-name=nemotron-sv-p1-data
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/p1-data-%j.out
#SBATCH --error=logs/p1-data-%j.err

set -euo pipefail
ulimit -c 0
module purge

REPO_ROOT="$SLURM_SUBMIT_DIR"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CONTAINER="${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
MODEL_DIR="$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16"
CANDIDATES="$ASSET_ROOT/data/p1-nemotron3-4b-lora-sv-100k.jsonl"
OUTPUT="$ASSET_ROOT/data/p1-nemotron3-4b-lora-sv-100k-accepted.jsonl"
MANIFEST="$ASSET_ROOT/manifests/p1-nemotron3-4b-lora-sv-100k-windowed.json"

test -s "$MODEL_DIR/config.json"
test -s "$CANDIDATES"
srun singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  python scripts/prepare_windowed_data.py \
    --config configs/lumi-p1-100k.json \
    --model-dir '$MODEL_DIR' \
    --candidates '$CANDIDATES' \
    --output '$OUTPUT' \
    --manifest '$MANIFEST'
"

