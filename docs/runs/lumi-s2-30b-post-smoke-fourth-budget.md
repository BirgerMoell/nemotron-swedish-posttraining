# LUMI S2 30B-A3B post-trained smoke fourth attempt — GPU-hour budget

This estimate was recorded after attempt `22157567` isolated the MIOpen
depthwise-convolution failure and before the pure-PyTorch convolution attempt
was submitted.

## Changed input

- Model weights, data, LoRA targets and FSDP layout are unchanged.
- The pinned remote-code patch replaces the four-tap depthwise Mamba Conv1d in
  the slow path with left-pad/unfold/multiply/sum and FP32 accumulation.
- The fully patched 30B model-code SHA is
  `a480c5d1fbf41ed51c0ec071539c783820072828acb756f2934a4e4bf0b5f564`.

## Estimate

| Phase | Wall-clock estimate | GCDs | GPU-hours |
|---|---:|---:|---:|
| Load, BF16 cast, FSDP construction and broadcast | 11 min | 8 | 1.47 |
| One forward/backward/update with convolution fallback | 7 min | 8 | 0.93 |
| Adapter consolidation and validation | 6 min | 8 | 0.80 |
| **Expected fourth attempt** | **24 min** | **8** | **3.20** |

The 30-minute hard ceiling is again 4.0 GPU-hours. The first three attempts
used `0.198 + 0.247 + 0.651 = 1.096 GPU-hours`, so expected cumulative use
after a successful fourth attempt is 4.30 GPU-hours and the cumulative hard
ceiling is 5.096 GPU-hours.
