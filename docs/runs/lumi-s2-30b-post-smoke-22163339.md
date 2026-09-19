# LUMI S2 30B-A3B smoke — job 22163339

## Result

`PASS`. The post-trained 30B-A3B BF16 checkpoint completed a real Swedish
LoRA optimizer update under eight-way FSDP on LUMI and saved an adapter-only
artifact that passed fail-closed validation.

| Measurement | Result |
|---|---:|
| Slurm elapsed | 5 min 44 s |
| Allocated GPU-hours | `8 × 344 / 3600 = 0.764` |
| Training step time | 11.86 s |
| Loss | 6.4322 |
| Gradient norm | 2.0714 |
| Peak memory per observed rank | 15,563,992,576 bytes (14.49 GiB) |
| Trainable LoRA parameters | 4,572,672 |
| Total parameters | 31,582,510,016 |
| Adapter size | 18,309,296 bytes |
| Adapter SHA-256 | `c02b05c665a402561e051437d6b53fa58fdb62ef53be4df1f30cc98ee70a9bfc` |

The fourth attempt used 0.764 GPU-hours versus the 3.20 expected estimate and
4.0 hard ceiling. Across all four attempts, realized allocation was
`0.198 + 0.247 + 0.651 + 0.764 = 1.860 GPU-hours`.

## Qualified system path

- Checkpoint: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at
  `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`
- Repository: `1a03ba172f8395951a65a353216b6daa6be1c666`
- Runtime: PyTorch 2.9.1 + ROCm 6.4, Transformers 4.57.3, PEFT 0.18.1
- Parallelism: FSDP `FULL_SHARD`, eight MI250X GCDs, rank-zero-only CPU load
- ROCm path: eager attention, PyTorch Mamba, hash-pinned RMSNorm and
  MIOpen-free causal depthwise-convolution fallbacks
- Data: eight distinct prevalidated Swedish conversations, global batch 8,
  sequence length 256, final-assistant-turn-only loss
- Adapter: rank 8 on attention `q/k/v/o_proj` and Mamba `in/out_proj`

The run cast 23 frozen base tensors (7,913,472 elements) from FP32 storage to
BF16 so FSDP1 could flatten each Nemotron block. Router/state operations still
promote to FP32 in the model code. This is acceptable for the LoRA
compatibility lane and must not be silently generalized to full-parameter SFT.

## Interpretation

S2 proves the 30B-A3B post-trained checkpoint can be loaded, sharded, updated
and serialized on LUMI. One update on eight examples cannot demonstrate a
Swedish capability gain. The next gate is S3: a pre/post generation comparison
over a 1,024-row run before committing to the 100k P30 capability experiment.

The adapter remains on LUMI at:

`/scratch/project_465002530/users/bmoell/nemotron-swedish-posttraining/runs/22163339/adapter`
