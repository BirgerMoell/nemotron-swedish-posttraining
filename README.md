# Swedish post-training for NVIDIA Nemotron

Reproducible experiments for adapting NVIDIA Nemotron 3 to Swedish, with LUMI/ROCm as the first execution target.

The repository deliberately has two lanes:

1. **Compatibility lane on LUMI:** Hugging Face Transformers + PEFT on AMD MI250X. This proves that Nemotron-H can load through its PyTorch fallback, the Swedish chat data is masked correctly, gradients flow, and a LoRA adapter plus run manifest can be saved.
2. **Reference lane:** NVIDIA NeMo/Megatron Bridge recipes on NVIDIA hardware. This is the conformance target for larger SFT and later distillation/RL experiments.

The first LUMI job uses `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and 64 pinned examples from `AI-Sweden-Models/Dolci-Instruct-SFT-translated`. The 4B model is already instruction-tuned and English-oriented, so this job is a **systems smoke test, not a Swedish-quality result**. The first scientifically meaningful SFT candidate is the 30B-A3B base checkpoint.

## Repository map

- `docs/experiment-plan.md` — hypotheses, stage gates, evaluation and scale-up plan
- `docs/data-sources.md` — source-by-source recommendation and licensing notes
- `docs/lumi-runbook.md` — staging, submission, monitoring and recovery
- `data/sources.yaml` — machine-readable source registry
- `configs/lumi-smoke.json` — immutable first-run configuration
- `scripts/stage_assets.py` — stages the model and a deterministic Swedish sample
- `scripts/patch_nemotron_rocm.py` — exact-SHA-guarded PyTorch RMSNorm fallback
- `scripts/train_lora_smoke.py` — two-step, assistant-only LoRA smoke trainer
- `scripts/validate_run.py` — fail-closed artifact validation
- `lumi/` — LUMI staging and Slurm launchers

## Experiment ladder

| Gate | Model | Data | Purpose |
|---|---|---|---|
| S0 | Nemotron 3 Nano 4B BF16 | 64 Swedish Dolci rows | ROCm/custom-code/LoRA plumbing |
| S1 | Nemotron 3 Nano 4B BF16 | 5k Swedish mixture | data and loss sanity; no capability claim |
| S2 | Nemotron 3 Nano 30B-A3B Base BF16 | 1k rows | base-model FSDP compatibility |
| P1 | Nemotron 3 Nano 30B-A3B Base BF16 | 100k controlled mixture | first measured Swedish SFT pilot |
| P2 | same | 500k–1M quality-filtered mixture | scale only after P1 gates pass |

See [the experiment plan](docs/experiment-plan.md) for stop/go criteria.

## First LUMI run

On a LUMI login node, from the repository checkout:

```bash
bash lumi/stage_assets.sh
bash lumi/submit_smoke.sh
```

Compute nodes are offline; staging is intentionally separate from the Slurm job. The job writes a self-contained run under `runs/<job-id>/` with an adapter, resolved configuration, losses, environment metadata and validation report.

## Local checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
```

## Status

- [x] Source registry and staged experiment plan
- [x] Pinned S0 configuration and LUMI launch path
- [x] S0 LUMI result attached under `docs/runs/` — job `22150463`, PASS
- [ ] P1 Swedish capability run — queued as jobs `22151051` → `22151052` → `22151053`
- [ ] 30B-A3B base-model baseline and S2 run
