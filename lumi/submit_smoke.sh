#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining-assets}"

cd "$REPO_ROOT"
mkdir -p logs runs
test -s "$ASSET_ROOT/models/nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/config.json"
test -s "$ASSET_ROOT/data/s0-nemotron3-4b-lora-sv.jsonl"
sbatch --export=ALL,ASSET_ROOT="$ASSET_ROOT" lumi/smoke.sbatch

