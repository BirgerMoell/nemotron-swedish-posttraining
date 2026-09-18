# LUMI runbook

## Fixed environment

- Account: `project_465002530`
- User scratch: `/scratch/project_465002530/users/bmoell`
- Container: `/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif`
- Verified stack: PyTorch 2.9.1+ROCm 6.4, Transformers 4.57.3, PEFT 0.18.1, TRL 0.28.0
- Smoke partition: `dev-g`, one MI250X GCD, 30 minutes

Compute nodes have no internet. Run staging on a login node before submitting.

## Stage

```bash
cd /scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining
bash lumi/stage_assets.sh
```

This downloads the exact model revision to scratch, streams a deterministic 64-row sample from the exact dataset revision, validates message structure and writes a staging manifest plus SHA-256.

The pinned NVIDIA remote-code file guards its Mamba fast kernels but imports a
Triton gated-RMSNorm helper unconditionally. Staging therefore applies an
exact-SHA-guarded, autograd-safe PyTorch implementation of that one operation.
Both original and patched code hashes are written to the staging/run manifests;
any upstream source change makes staging fail instead of applying a fuzzy patch.

## Submit and inspect

```bash
bash lumi/submit_smoke.sh
squeue -u "$USER"
tail -f logs/<job-id>.out
tail -f logs/<job-id>.err
```

The job runs fully offline. Success requires `runs/<job-id>/validation.json` with `"status": "pass"` and an adapter under `runs/<job-id>/adapter/`.

## Failure triage

- Missing files: rerun staging on the login node; never allow the compute job to fetch.
- CUDA-only Mamba error: confirm the resolved config has `use_mamba_kernels=false`, the staging manifest records the RMSNorm compatibility patch, and the job uses a fresh `HF_MODULES_CACHE`.
- Unsupported attention backend: keep `attn_implementation=eager` for S0.
- OOM: confirm one process was launched, sequence length is 256 and only listed projection modules are trainable.
- Template-prefix failure: preserve the failing row and template; do not fall back to training on user tokens.
- NaN/non-finite gradients: validation must fail. Capture the environment and stop rather than saving a nominally successful adapter.

## Promote to S1

Only after S0 passes, create a new immutable config. Do not mutate `lumi-smoke.json`. S1 should add an explicit generation probe, a 100-row masking audit, source-stratified sampling and a baseline comparison.

## 30B-A3B post-trained FSDP smoke

The 30B checkpoint is a separate lane and does not replace or modify the 4B
jobs. Its estimate and allocation ceiling are recorded before submission in
`docs/runs/lumi-s2-30b-post-smoke-budget.md`.

```bash
cd /scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining
bash lumi/stage_30b_smoke.sh
bash lumi/submit_30b_smoke.sh
```

Staging downloads the pinned 63 GB BF16 checkpoint on a login node. The smoke
then uses all eight GCDs on one node with `FULL_SHARD`; only global rank zero
loads the CPU checkpoint, avoiding eight full host-memory copies. The 30-minute
allocation has a hard ceiling of 4 GPU-hours.
