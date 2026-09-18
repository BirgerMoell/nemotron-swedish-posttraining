# LUMI P1 launch — 2026-09-18

The Swedish capability run was submitted after the GPU-hour estimate in `lumi-p1-budget.md` was recorded.

## Immutable inputs

- Repository commit at submission: `9ecbd68db63b762f32e476f91cc5cd4399f79386`
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- Model revision: `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`
- Dataset: `AI-Sweden-Models/Dolci-Instruct-SFT-translated`
- Dataset revision: `e89e3c9f1385f2656fa0a5bed97b45193cfadcbf`
- Staged candidate rows: 300,000
- Candidate JSONL SHA-256: `c4a2cb5649ef6e2a586df3c0710c6bdbc14e6fca4e5929e146697271bbfc9a21`

## Dependency chain

| Stage | Slurm job | Allocation | Initial state |
|---|---:|---|---|
| Token-window data gate | `22151051` | CPU `small`, 32 CPUs, 128 GiB, 2 h | pending priority |
| Swedish LoRA training | `22151052` | `standard-g`, 8 nodes / 64 GCDs, 10.5 h | dependency |
| Base-vs-adapter evaluation | `22151053` | `dev-g`, 1 GCD, 1 h | dependency |

The dependencies use `afterok`: no GPU job is eligible unless the preceding stage exits successfully. At launch time, no GPU-hours had been consumed.
