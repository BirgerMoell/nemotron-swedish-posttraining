# LUMI S2 30B-A3B smoke — attempt 22157409

## Result

`FAIL` at the fail-closed token-window gate, before checkpoint loading or an
optimizer step.

- Repository commit: `93f0c2318be4acdf67c9139340dc9e979ba9d584`
- Allocation: one node, eight MI250X GCDs, 30-minute limit
- Actual elapsed allocation: 89 seconds
- Consumed allocation: `8 × 89 / 3600 = 0.198 GPU-hours`
- Exit: code 1

All eight ranks reported that zero of the staged 64 conversations preserved
both the required 16 masked prompt tokens and 16 supervised assistant tokens
inside the 128-token left-truncated window. FSDP construction and model weight
loading had not started, so this attempt provides no model-compatibility result.

## Corrective action

The rerun moves token-window qualification to the login-node staging phase,
stages 1,024 deterministic candidates, selects eight distinct qualifying rows,
and uses a 256-token window. The GPU job now refuses to submit unless the
prevalidated eight-row artifact exists.
