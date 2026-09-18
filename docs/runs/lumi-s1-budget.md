# LUMI S1 GPU-hour budget

Calculated before submission from S0 job `22150463`.

## Planned run

- 5,120 distinct accepted Swedish conversations from 20,000 deterministic candidates
- maximum sequence length 256; at least 16 prompt tokens masked and 32 assistant tokens supervised
- eight MI250X GCDs with DDP; one example per rank; global batch 8
- 640 synchronized steps, exactly one pass over the accepted set
- separate one-GCD base-vs-adapter generation evaluation

## Calculation

S0's measured trainer time was 14.623 seconds for two steps, or 7.3115 seconds per 256-token step on one GCD. DDP keeps the number of synchronized steps at `5120 / 8 = 640`.

| Component | Calculation | GPU-hours |
|---|---:|---:|
| Measured training projection | `640 × 7.3115 s × 8 / 3600` | 10.40 |
| 15% DDP allowance | `10.40 × 0.15` | 1.56 |
| Load/save allowance | `0.05 h × 8` | 0.40 |
| Held-out generation evaluation | `0.25 h × 1` | 0.25 |
| Expected total | sum | **12.61** |
| 25% uncertainty envelope | `12.61 × 1.25` | **15.76** |

The training allocation is capped at 2 hours × 8 GCDs = 16 GPU-hours. Evaluation is a separate 30-minute one-GCD allocation, so the absolute allocation cap is 16.5 GPU-hours. Exceeding the training wall-time terminates the job rather than silently expanding the budget.

