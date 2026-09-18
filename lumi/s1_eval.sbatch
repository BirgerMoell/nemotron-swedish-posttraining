#!/usr/bin/env bash
#SBATCH --job-name=nemotron-sv-s1-eval
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail
ulimit -c 0
module purge
: "${TRAIN_JOB_ID:?submit through lumi/submit_s1.sh}"

REPO_ROOT="$SLURM_SUBMIT_DIR"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CONTAINER="${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
HF_CACHE="${HF_CACHE:-/scratch/project_465002530/users/bmoell/hf_cache}"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
MODEL_DIR="$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16"
RUN_DIR="$REPO_ROOT/runs/$TRAIN_JOB_ID"

test -s "$RUN_DIR/adapter/adapter_model.safetensors"
mkdir -p "$RUN_DIR/eval" "$RUN_DIR/eval-hf-modules"
export HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_MODULES_CACHE="$RUN_DIR/eval-hf-modules"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True

srun --label --gpu-bind=closest singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  python scripts/evaluate_generations.py \
    --model-dir '$MODEL_DIR' \
    --adapter-dir '$RUN_DIR/adapter' \
    --prompts data/eval/swedish_generation_v1.jsonl \
    --output '$RUN_DIR/eval/swedish_generation_v1.json'
"

