# Swedish continuation plan for Nemotron 3 Nano 30B-A3B

## Starting point

Continue from the post-trained BF16 checkpoint
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, not the base checkpoint. This
preserves NVIDIA's instruction-following, reasoning and chat behavior while
adding Swedish specialization. Quantized inference checkpoints are excluded
from training.

## Experiment ladder

| Gate | Scale | Question | Promotion criterion |
|---|---:|---|---|
| S2 | 8 qualified rows, 1 step | Does the 30B hybrid MoE train and save under ROCm/FSDP? | All systems gates pass |
| S3 | 1,024 rows, 16–32 steps | Is loss stable and is Swedish generation directionally improved? | No format collapse; positive paired Swedish probe |
| E0 | no training | What are the untouched Swedish and English baselines? | Frozen predictions and scores published |
| P30-A | 100k rows | Does attention/Mamba LoRA improve Swedish while retaining English? | Swedish aggregate gain; retention gates pass |
| P30-B | same 100k | Do rank-8 expert-MLP adapters add value over backbone-only LoRA? | Gain exceeds confidence interval and compute cost is justified |
| P30-C | best adapter, 3 seeds | Is the result reproducible? | Same decision under all fixed seeds |

S2 is intentionally too small to change capability. S3 is a debugging run.
Only P30 is designed as a capability update.

S2 passed on LUMI as job `22163339`: one finite update, non-zero gradient,
adapter-only serialization and a validated manifest. The complete result is in
`docs/runs/lumi-s2-30b-post-smoke-22163339.md`.

## P30 training mixture

Use 100,000 quality-filtered conversations for the first measured pilot:

| Share | Rows | Source class | Role |
|---:|---:|---|---|
| 55% | 55,000 | Swedish Dolci translated SFT | broad instruction coverage |
| 15% | 15,000 | human-authored or human-reviewed Swedish | native register and idiom anchor |
| 10% | 10,000 | post-edited NVIDIA-style instruction/format tasks | exact constraint adherence |
| 10% | 10,000 | original English post-training replay | retention of existing behavior |
| 5% | 5,000 | Swedish reasoning/math with verified answers | capability signal, not style only |
| 5% | 5,000 | Swedish safety, refusal and structured-output tasks | behavioral coverage |

Every row must retain source ID, upstream revision, license, transformation
chain and quality scores. Deduplicate against all evaluation prompts before
training. Translation gates cover language ID, named entities, numbers,
placeholders, code blocks and answer schemas; failed rows are post-edited or
excluded.

## Adapter ablation

P30-A tunes attention (`q/k/v/o_proj`) and Mamba (`in/out_proj`) with LoRA rank
16. P30-B adds the routed and shared expert `up/down_proj` layers at rank 8.
Both use identical data order, effective batch, token count and evaluation.
This separates a cheap representation adapter from the much larger MoE-expert
adapter instead of assuming the latter is necessary.

Start at sequence length 2,048, BF16, assistant-only loss, cosine decay, 3%
warmup and one pass over the 100k mixture. Select the learning rate with a
short sweep over `2e-5`, `5e-5` and `1e-4`; do not infer it from the 4B run.

## Evaluation and decision rule

Freeze E0 before training. Compare the untouched checkpoint and every adapter
with identical prompts and decoding.

- Swedish capability: SweSAT, Swedish ScandEval/SuperLim tasks, verified
  Swedish math/reasoning, and a versioned instruction-following suite.
- Format behavior: executable JSON, XML, table, length and lexical constraints.
- Retention: English IFEval, MMLU-Pro and a fixed reasoning/code subset.
- Translation: WMT24++ English-to-Swedish as a diagnostic, not the sole target.
- Safety: paired Swedish benign/refusal cases with blinded human review of
  critical failures.

Promote only with a statistically supported Swedish aggregate gain, at most
two percentage points absolute loss on the English aggregate, at most three
points loss in executable format adherence, and no increase in critical safety
failures. Three fixed-seed runs are required before release.
