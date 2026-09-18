# LUMI S2 30B-A3B post-trained smoke rerun — GPU-hour budget

This estimate was recorded after attempt `22157409` failed at data validation
and before the corrective rerun was submitted.

## Changed input

- Token-window qualification runs before Slurm submission and consumes no GPU
  time.
- The deterministic candidate pool grows from 64 to 1,024 rows.
- Eight distinct rows must preserve at least 16 prompt tokens and 16 assistant
  tokens in a 256-token window.
- Model, model revision, LoRA targets, GCD count and optimizer-step count are
  unchanged.

## Rerun estimate

| Phase | Wall-clock estimate | GCDs | GPU-hours |
|---|---:|---:|---:|
| Rank-zero load, FSDP construction and broadcast | 10 min | 8 | 1.33 |
| One 256-token forward/backward/update | 6 min | 8 | 0.80 |
| Full-state adapter consolidation and validation | 6 min | 8 | 0.80 |
| **Expected rerun** | **22 min** | **8** | **2.93** |

The rerun keeps the 30-minute Slurm limit, so its hard ceiling remains
`8 × 0.5 = 4.0 GPU-hours`. Including the failed attempt's measured 0.198
GPU-hours, the cumulative smoke-test ceiling is 4.198 GPU-hours and expected
cumulative use is 3.13 GPU-hours.
