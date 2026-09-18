# Data-source decisions

The source of truth is `data/sources.yaml`. This note explains why each source is or is not in the first mixture.

## Recommended now

### AI Sweden Swedish Dolci — primary

`AI-Sweden-Models/Dolci-Instruct-SFT-translated` contains 494,841 Swedish multi-turn conversations in messages format under Apache-2.0. It is a deterministic Gemma 3 27B-IT translation of OpenEuroLLM Dolci data and preserves code blocks. It is large, already normalized and owned by the team closest to this experiment, making it the best primary source.

Risk: translation artifacts can preserve semantic meaning while breaking format constraints. Keep the source ID and add line/entity/schema checks before P1.

### Aya Dataset — human anchor

`CohereLabs/aya_dataset` contains 204k human annotations across 65 languages under Apache-2.0. Filter `language_code` for Swedish and retain `annotation_type` so original annotations can be evaluated separately from re-annotations. The Swedish slice is small enough to protect with source-aware sampling.

### NVIDIA SFT v3 — replay and translation source

`nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` supplies current NVIDIA chat and instruction-following data. Use a filtered English slice for retention and translate a controlled instruction-following slice. NVIDIA says only the last assistant turn in chat rows should be trained. Some seed prompts are withheld and require gated upstream access, so unresolved rows must be excluded—not reconstructed heuristically.

## Conditional sources

- `CohereLabs/aya_collection`: useful breadth, but heavy templating means sample only named Swedish sources and deduplicate aggressively.
- `nvidia/Nemotron-Post-Training-Dataset-v2`: valuable as a volume/schema reference and for filtered replay; licenses are per sample and cannot be flattened to a single label.
- `nicher92/magpie_llama70b_200k_filtered_swedish`: potentially useful synthetic ablation, held for Llama 3.3 licensing and quality review.
- `HuggingFaceTB/smoltalk` and `allenai/tulu-3-sft-mixture`: possible English replay after filtering their heterogeneous upstream license terms.

## Excluded or evaluation-only

- Bactrian-X Swedish is CC-BY-NC-4.0: exclude from release-oriented training.
- OASST2 has negligible Swedish yield.
- OPUS-100 `en-sv` is on hold until licenses are resolved at contributing-corpus level.
- `V4ldeLund/scandi-translated-instruct` reports an unclear aggregate license: exclude pending provenance.
- SweSAT, SuperLim and WMT24++ are evaluation assets and must be contamination-screened and held out.

## First curation report

Before P1, produce a versioned report with counts and token counts by source, language confidence, turn count, length bucket, code presence, duplicate cluster, license, PII flag, post-edit disposition and train/eval overlap. The manifest should make every aggregate traceable back to row IDs.

