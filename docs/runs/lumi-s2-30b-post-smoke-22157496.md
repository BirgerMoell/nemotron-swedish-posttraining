# LUMI S2 30B-A3B smoke — attempt 22157496

## Result

`FAIL` during FSDP block construction, before the forward pass.

- Repository commit: `83940a6e4d88cec7933ad065685083b25fd47374`
- Allocation: one node, eight MI250X GCDs, 30-minute limit
- Actual elapsed allocation: 111 seconds
- Consumed allocation: `8 × 111 / 3600 = 0.247 GPU-hours`
- Exit: code 1

The prevalidated 256-token data passed and global rank zero loaded all 13 model
shards. FSDP then rejected a `NemotronHBlock` because it contained both BF16
and FP32 parameters: `Must flatten tensors with uniform dtype`.

## Corrective action

This experiment trains only LoRA parameters. Before FSDP construction, the
third attempt explicitly casts all frozen floating-point base parameters to
BF16 and asserts that every floating-point parameter is then BF16. Nemotron's
router and state-space code still promotes its sensitive operations to FP32 at
runtime, but the stored frozen values have BF16 precision. The manifest records
the number of parameters and elements affected. A future full-parameter run
must revisit mixed-dtype wrapping rather than inherit this LoRA-specific
compatibility choice silently.
