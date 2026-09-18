# LUMI P1 Swedish capability run — GPU-hour budget

Calculated before submission from S0 job `22150463`. This supersedes the unsubmitted 5k S1 budget.

## Planned run

- 100,032 distinct, token-window-qualified Swedish conversations
- 512-token windows with at least 32 masked prompt and 64 supervised assistant tokens
- rank-16 LoRA on Mamba, attention and MLP projections
- eight LUMI-G nodes, 64 MI250X GCDs, global batch 64
- 1,563 synchronized updates, one pass over the accepted examples
- cosine learning-rate schedule with 50 warmup steps
- separate base-vs-adapter Swedish generation evaluation

## Calculation

S0 measured 7.3115 seconds per 256-token update. Doubling the maximum sequence length gives 14.623 seconds. A 25% allowance covers the broader adapter and eight-node synchronization, giving 18.279 seconds per synchronized step.

| Component | Calculation | GPU-hours |
|---|---:|---:|
| Training compute | `1563 × 18.279 s × 64 / 3600` | 507.9 |
| Distributed load/save allowance | `0.15 h × 64` | 9.6 |
| Held-out generation evaluation | `0.5 h × 1` | 0.5 |
| Expected total | sum | **518.0** |
| 25% uncertainty envelope | `518.0 × 1.25` | **647.5** |

The Slurm training request is 10.5 hours × 64 GCDs = 672 allocated GPU-hours. Evaluation is a separate one-hour, one-GCD allocation. The requested ceiling is therefore 673 GPU-hours; expected consumption is about 518 GPU-hours.

