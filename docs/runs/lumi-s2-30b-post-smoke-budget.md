# LUMI S2 30B-A3B post-trained smoke — GPU-hour budget

## Purpose

Qualify the post-trained `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
checkpoint for Swedish LoRA continuation on LUMI. This gate tests the pinned
ROCm fallback, rank-zero-only checkpoint loading, eight-way FSDP sharding,
assistant-only labels, a finite optimizer update and adapter serialization. It
makes no Swedish capability claim.

## Immutable design

- Model revision: `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`
- Model storage: 63,174,529,404 bytes (official Hugging Face metadata)
- Allocation: one LUMI-G node, eight MI250X GCDs
- Data: eight accepted Swedish conversations, one per rank
- Update: one global optimizer step, sequence length 128
- Adapter: LoRA rank 8 on attention and Mamba input/output projections
- Backend: BF16, FSDP `FULL_SHARD`, `use_orig_params=true`, eager attention,
  PyTorch Mamba fallback

## Estimate before submission

| Phase | Wall-clock estimate | GCDs | GPU-hours |
|---|---:|---:|---:|
| Rank-zero load, FSDP construction and broadcast | 10 min | 8 | 1.33 |
| One forward/backward/update | 4 min | 8 | 0.53 |
| Full-state adapter consolidation and validation | 6 min | 8 | 0.80 |
| **Expected** | **20 min** | **8** | **2.67** |

The Slurm limit is 30 minutes. The hard allocation ceiling is therefore:

`8 GCDs × 0.5 hours = 4.0 GPU-hours`.

Model and dataset staging run on a login node and consume zero GPU-hours. If
the job reaches the 30-minute limit, it fails closed; no follow-on job is
submitted automatically.

## Pass gates

- all eight ranks initialize from one CPU checkpoint load;
- peak memory stays within one 64 GiB GCD per rank;
- loss and gradient norm are finite and the gradient norm is non-zero;
- assistant-only masking leaves at least 16 prompt and 16 target tokens;
- `adapter/adapter_model.safetensors`, `run_manifest.json` and
  `validation.json` are present;
- validation reports `status: pass`.

Attempt `22157409` used this original configuration and failed at the token
window gate after 89 seconds. See `lumi-s2-30b-post-smoke-22157409.md`; the
corrective rerun has its own pre-submission estimate in
`lumi-s2-30b-post-smoke-rerun-budget.md`.
