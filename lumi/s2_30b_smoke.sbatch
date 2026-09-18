#!/usr/bin/env bash
#SBATCH --job-name=nemotron-sv-30b-s2
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=logs/s2-30b-%j.out
#SBATCH --error=logs/s2-30b-%j.err

set -euo pipefail
ulimit -c 0
module purge

REPO_ROOT="$SLURM_SUBMIT_DIR"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CONTAINER="${CONTAINER:-/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
HF_CACHE="${HF_CACHE:-/scratch/project_465002530/users/bmoell/hf_cache}"
BIND=/pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd
MODEL_DIR="$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DATA_PATH="$ASSET_ROOT/data/s2-nemotron3-30b-a3b-post-lora-sv-smoke.jsonl"
RUN_DIR="$REPO_ROOT/runs/$SLURM_JOB_ID"
REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="$((29500 + SLURM_JOB_ID % 1000))"

test -s "$MODEL_DIR/config.json"
test -s "$MODEL_DIR/model-00013-of-00013.safetensors"
test -s "$DATA_PATH"
mkdir -p "$RUN_DIR"

export CONTAINER REPO_COMMIT
export HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets"
export HF_MODULES_CACHE_BASE="/tmp/nemotron-30b-hf-modules-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=7
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export ACCELERATE_USE_FSDP=true FSDP_CPU_RAM_EFFICIENT_LOADING=1

echo "run=$SLURM_JOB_ID commit=$REPO_COMMIT master=$MASTER_ADDR:$MASTER_PORT started=$(date --iso-8601=seconds)"
srun --label --gpu-bind=none singularity exec -B "$BIND" "$CONTAINER" bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  accelerate launch \
    --config_file configs/accelerate-lumi-fsdp.yaml \
    --main_process_ip '$MASTER_ADDR' \
    --main_process_port '$MASTER_PORT' \
    --num_machines 1 \
    --num_processes 8 \
    --machine_rank 0 \
    scripts/train_fsdp_lora_smoke.py \
    --config configs/lumi-s2-30b-post-smoke.json \
    --model-dir '$MODEL_DIR' \
    --data '$DATA_PATH' \
    --output-dir '$RUN_DIR'
  python scripts/validate_run.py '$RUN_DIR'
"
echo "completed=$(date --iso-8601=seconds) run_dir=$RUN_DIR"
