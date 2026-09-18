#!/usr/bin/env bash
#SBATCH --job-name=nemotron-sv-s0
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
ulimit -c 0
module purge

REPO_ROOT="$SLURM_SUBMIT_DIR"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CONTAINER="${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
HF_CACHE="${HF_CACHE:-/scratch/project_465002530/users/bmoell/hf_cache}"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
MODEL_DIR="$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_PATH="$ASSET_ROOT/data/s0-nemotron3-4b-lora-sv.jsonl"
RUN_DIR="$REPO_ROOT/runs/$SLURM_JOB_ID"
REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

test -s "$MODEL_DIR/config.json"
test -s "$DATA_PATH"
mkdir -p "$RUN_DIR"

export CONTAINER REPO_COMMIT
export HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=7

echo "run=$SLURM_JOB_ID commit=$REPO_COMMIT host=$(hostname) started=$(date --iso-8601=seconds)"
srun --label --gpu-bind=closest singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  cd '$REPO_ROOT'
  python scripts/train_lora_smoke.py \
    --config configs/lumi-smoke.json \
    --model-dir '$MODEL_DIR' \
    --data '$DATA_PATH' \
    --output-dir '$RUN_DIR'
  python scripts/validate_run.py '$RUN_DIR'
"
echo "completed=$(date --iso-8601=seconds) run_dir=$RUN_DIR"

