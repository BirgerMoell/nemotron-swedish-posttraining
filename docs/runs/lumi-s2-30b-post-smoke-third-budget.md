# LUMI S2 30B-A3B post-trained smoke third attempt — GPU-hour budget

This estimate was recorded after attempts `22157409` and `22157496` failed and
before the BF16-uniform FSDP attempt was submitted.

## Changed input

- Data and model revision are unchanged from attempt `22157496`.
- Frozen FP32 base parameters are explicitly stored as BF16 before FSDP block
  flattening; the LoRA parameters are BF16 and remain the only trainable
  parameters.
- The run fails before training if any floating-point parameter retains a
  non-BF16 dtype.

## Estimate

| Phase | Wall-clock estimate | GCDs | GPU-hours |
|---|---:|---:|---:|
| Rank-zero load, BF16 cast, FSDP construction and broadcast | 11 min | 8 | 1.47 |
| One 256-token forward/backward/update | 6 min | 8 | 0.80 |
| Adapter consolidation and validation | 6 min | 8 | 0.80 |
| **Expected third attempt** | **23 min** | **8** | **3.07** |

The 30-minute hard ceiling remains 4.0 GPU-hours. The two failed attempts used
`0.198 + 0.247 = 0.445 GPU-hours`, so the cumulative expected use after a
successful third attempt is 3.52 GPU-hours and the cumulative hard ceiling is
4.445 GPU-hours.
