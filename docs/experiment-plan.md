# Experiment plan

## Objective

Produce a Swedish-capable Nemotron 3 checkpoint without sacrificing core English instruction following, reasoning, or format compliance. Continue from the 30B-A3B **post-trained** BF16 checkpoint so the experiment adapts an existing instruction model rather than rebuilding alignment from the base checkpoint. Preference optimization and on-policy distillation come only after supervised adaptation is measured.

## Hypotheses

- **H1:** A quality-controlled Swedish SFT mixture materially improves native Swedish reasoning and instruction following over the untouched 30B-A3B post-trained model.
- **H2:** Keeping 10–20% English replay prevents material English regression.
- **H3:** Human-authored Swedish anchors and post-editing improve exact instruction/answer-format compliance more than adding raw translated volume.
- **H4:** Reasoning traces may remain English in the first pilot, but the final answer must follow the requested language and schema.

## Two implementation lanes

NVIDIA's published recipes assume NeMo/Megatron and CUDA. LUMI uses AMD MI250X/ROCm. We therefore keep two explicit lanes instead of claiming backend equivalence:

- **LUMI compatibility lane:** Transformers, the model repository's custom Nemotron-H code with `use_mamba_kernels=false`, PyTorch eager attention, and PEFT/LoRA. This lane discovers data/model/runtime failures cheaply.
- **NVIDIA reference lane:** reproduce the successful mixture and hyperparameters with NeMo/Megatron Bridge on supported NVIDIA hardware before a release-quality full-parameter run.

Run manifests record which lane produced every artifact.

## Stage gates

### S0 — one-GCD systems smoke

- Model: Nemotron 3 Nano 4B BF16, immutable revision in `configs/lumi-smoke.json`
- Data: 64 deterministic Swedish Dolci conversations
- Update: LoRA rank 8, two optimizer steps, sequence length 256
- Pass: finite loss on both steps, non-zero finite gradient norm, at least 16 masked prompt tokens remain in every training window, adapter saved, resolved manifest validates
- Interpretation: plumbing only; no quality claim

### S1 — data/loss smoke

- Same 4B checkpoint, 5k examples, 100–300 optimizer steps
- Stratify by single/multi-turn, code/no-code, response length and domain
- Inspect 100 sampled generations and 100 masked training records
- Pass: no template corruption; at least 99.5% valid rows; Swedish final-answer rate at least 95%; no NaN/OOM

### S2 — target checkpoint compatibility

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- One node, FSDP across eight MI250X GCDs, eight examples, one update
- Use the checkpoint's pinned chat template and verify the assistant-only boundary
- Pass: same systems gates as S0 plus an adapter-only safetensors/config serialization check

### P1 — measured 100k pilot

Proposed starting mixture by sampled conversations:

| Share | Source | Purpose |
|---:|---|---|
| 65% | Swedish Dolci | broad Swedish instruction coverage |
| 10% | Swedish Aya human annotations | human-language anchor |
| 10% | NVIDIA v3 English chat/IF | English capability replay |
| 10% | NVIDIA v3 translated to Swedish and post-edited | exact instruction following |
| 5% | curated Swedish safety/refusal and format tasks | behavioral coverage |

Use temperature-based source sampling so small human sources are not exhausted repeatedly. Deduplicate train-to-train and train-to-eval using normalized exact hashes plus MinHash/embedding similarity. Preserve source ID, upstream ID, license, transformation chain and quality scores on every row.

### P2 — scale decision

Scale toward 500k–1M rows only if P1 clears all gates. Vary one axis at a time:

1. translated-only vs translated + human anchor;
2. no post-edit vs post-edit;
3. 10% vs 20% English replay;
4. LoRA vs full-parameter/reference-lane SFT.

Do not add RL merely because the recipe includes it. First establish a reward/evaluator with high agreement on Swedish format and factuality.

## Translation and post-edit pipeline

Port NVIDIA's staged design rather than translating entire serialized conversations:

1. protect code, XML, URLs, placeholders, numbers and schema literals;
2. translate message content line-by-line;
3. reject wrong-language or missing-content outputs;
4. compare entities, numbers, constraints and answer schema to the source;
5. post-edit failures with a second model;
6. reconstruct the conversation and run template/format validation;
7. retain English reasoning traces where translation would damage verifiability, but require Swedish final answers when requested.

Suggested row-level gates: Swedish language probability, length ratio, entity/number preservation, placeholder equality, code-block equality, round-trip semantic similarity, toxicity/PII scan and exact schema checks. Keep failures and scores; do not silently drop provenance.

## Evaluation design

Evaluate the untouched base, SFT candidate and English-replay ablations with identical prompts and decoding.

**Primary Swedish:** SweSAT-1.0, SuperLim tasks, ScandEval Swedish tasks, WMT24++ `en-sv_SE`, and a versioned Swedish instruction-following set with executable format checks.

**Retention:** English IFEval, MMLU-Pro, GSM8K/MATH subset, code and long-context canaries matching the intended use case.

**Behavioral:** Swedish safety/refusal pairs, hallucination probes, register/locale prompts, JSON/XML/table exactness, and multi-turn constraint retention.

MMLU-ProX is not used as a Swedish metric: its `sw` identifier is Swahili, not Swedish.

## Go/no-go criteria for P1

- statistically reliable gain on the pre-registered Swedish aggregate;
- no more than 2 percentage points absolute regression on the English retention aggregate;
- no regression greater than 3 points on executable format adherence;
- no increase in critical safety failures in the reviewed Swedish set;
- all training rows have resolvable provenance and license metadata;
- three fixed-seed reruns show the result is not a single-run outlier.

## Reproducibility contract

Every run records model and dataset commit SHA, repository commit, container path/hash, package versions, Slurm job/allocation, visible GPU, seeds, resolved config, input JSONL SHA-256, loss trace and saved-artifact hashes. A missing field fails validation rather than being filled by guesswork.
