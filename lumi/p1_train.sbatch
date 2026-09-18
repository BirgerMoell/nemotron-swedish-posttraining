#!/usr/bin/env bash
#SBATCH --job-name=nemotron-sv-p1
#SBATCH --account=project_465002530
#SBATCH --partition=standard-g
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=10:30:00
#SBATCH --output=logs/p1-train-%j.out
#SBATCH --error=logs/p1-train-%j.err

set -euo pipefail
ulimit -c 0
module purge

REPO_ROOT="$SLURM_SUBMIT_DIR"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CONTAINER="${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
HF_CACHE="${HF_CACHE:-/scratch/project_465002530/users/bmoell/hf_cache}"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
MODEL_DIR="$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_PATH="$ASSET_ROOT/data/p1-nemotron3-4b-lora-sv-100k-accepted.jsonl"
RUN_DIR="$REPO_ROOT/runs/$SLURM_JOB_ID"
REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

test -s "$MODEL_DIR/config.json"
test -s "$DATA_PATH"
mkdir -p "$RUN_DIR"

export CONTAINER REPO_COMMIT
export HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_MODULES_CACHE_BASE="/tmp/nemotron-hf-modules-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=7
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB

echo "run=$SLURM_JOB_ID commit=$REPO_COMMIT master=$MASTER_ADDR:$MASTER_PORT started=$(date --iso-8601=seconds)"
srun --label --gpu-bind=none singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --node_rank=\$SLURM_PROCID \
    --master_addr='$MASTER_ADDR' \
    --master_port='$MASTER_PORT' \
    scripts/train_lora_smoke.py \
    --config configs/lumi-p1-100k.json \
    --model-dir '$MODEL_DIR' \
    --data '$DATA_PATH' \
    --output-dir '$RUN_DIR'
  if [ \"\$SLURM_PROCID\" = 0 ]; then python scripts/validate_run.py '$RUN_DIR'; fi
"
echo "completed=$(date --iso-8601=seconds) run_dir=$RUN_DIR"

