#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"
CANDIDATES="$ASSET_ROOT/data/p1-nemotron3-4b-lora-sv-100k.jsonl"

cd "$REPO_ROOT"
mkdir -p logs runs
test -s "$CANDIDATES"
PREP_JOB_ID="$(sbatch --parsable --export=ALL,ASSET_ROOT="$ASSET_ROOT" lumi/p1_prepare.sbatch)"
TRAIN_JOB_ID="$(sbatch --parsable --dependency="afterok:$PREP_JOB_ID" \
  --export=ALL,ASSET_ROOT="$ASSET_ROOT" lumi/p1_train.sbatch)"
EVAL_JOB_ID="$(sbatch --parsable --dependency="afterok:$TRAIN_JOB_ID" \
  --export=ALL,ASSET_ROOT="$ASSET_ROOT",TRAIN_JOB_ID="$TRAIN_JOB_ID" lumi/p1_eval.sbatch)"
echo "preparation_job=$PREP_JOB_ID training_job=$TRAIN_JOB_ID evaluation_job=$EVAL_JOB_ID"
