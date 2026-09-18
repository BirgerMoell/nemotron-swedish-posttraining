#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"

cd "$REPO_ROOT"
mkdir -p logs runs
test -s "$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/config.json"
test -s "$ASSET_ROOT/data/s1-nemotron3-4b-lora-sv-5k.jsonl"
TRAIN_JOB_ID="$(sbatch --parsable --export=ALL,ASSET_ROOT="$ASSET_ROOT" lumi/s1_train.sbatch)"
EVAL_JOB_ID="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB_ID" \
  --export=ALL,ASSET_ROOT="$ASSET_ROOT",TRAIN_JOB_ID="$TRAIN_JOB_ID" lumi/s1_eval.sbatch)"
echo "training_job=$TRAIN_JOB_ID evaluation_job=$EVAL_JOB_ID"
