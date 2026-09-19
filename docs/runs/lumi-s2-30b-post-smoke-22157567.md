# LUMI S2 30B-A3B smoke — attempt 22157567

## Result

`FAIL` in the first Mamba layer's depthwise convolution during the forward
pass; no optimizer update occurred.

- Repository commit: `63eb08be681f67d3a4bc832754fdc95d41432acb`
- Allocation: one node, eight MI250X GCDs, 30-minute limit
- Actual elapsed allocation: 293 seconds
- Consumed allocation: `8 × 293 / 3600 = 0.651 GPU-hours`
- Exit: code 1

This attempt passed all earlier gates: token-window validation, rank-zero-only
loading of 13 shards, LoRA injection, the explicit frozen-parameter BF16 cast,
and FSDP construction/synchronization. It cast 23 frozen parameter tensors
(7,913,472 elements) from FP32 storage to BF16. The first forward pass then
failed on every rank with `miopenStatusInternalError` in the depthwise
`torch.nn.Conv1d` used by the pure-PyTorch Mamba fallback.

## Corrective action

The pinned-source compatibility patch now replaces only that training-time
four-tap depthwise convolution with a mathematically equivalent operation:
left padding, `unfold`, elementwise kernel multiplication, FP32 reduction and
conversion back to the input dtype. A local CPU equivalence check matched
PyTorch Conv1d exactly for BF16 test inputs. The replacement remains autograd
safe and its post-patch SHA is pinned for both the 4B and 30B source files.
