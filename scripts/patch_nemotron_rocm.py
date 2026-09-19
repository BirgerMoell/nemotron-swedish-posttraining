#!/usr/bin/env python3
"""Patch pinned Nemotron-H remote code for the LUMI PyTorch fallback.

The upstream file guards Mamba/causal-conv kernels but imports one Triton
RMSNorm helper unconditionally. This exact-source patch lets the already
implemented PyTorch Mamba path run on LUMI without installing CUDA extensions.
It also replaces the four-tap depthwise Conv1d that fails in MIOpen with an
equivalent left-pad/unfold implementation using FP32 accumulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


# NVIDIA currently publishes the same guarded import block in both pinned
# checkpoints used by this repository.  Keep every accepted source hash
# explicit: a new upstream implementation must be reviewed rather than being
# modified by a fuzzy patch.
ORIGINAL_SHA256S = {
    # NVIDIA-Nemotron-3-Nano-4B-BF16@dfaf35d
    "ea982af0b805f181573f919ecb001d5bbc0153459923cf4b2f1ccae194e415a4",
    # NVIDIA-Nemotron-3-Nano-30B-A3B-BF16@bf77c31
    "4d353ce6e8f495d043f2d8c5acd13496a84ba2b2d45da8279797a55f680309d9",
}
RMS_PATCHED_SHA256S = {
    # 4B and 30B sources after the RMSNorm-only compatibility patch.
    "097d41e709bd685b0bbc178a8751f04a8ab6d5f62310b9b9d9b66879ed66410a",
    "89b1575fb0986999e9b851c8d25e131fabf0c962f1487a9236bac2bbfa0a9a29",
}
FULLY_PATCHED_SHA256S = {
    # 4B and 30B sources after both ROCm compatibility patches.
    "0b75b2a263b4e1d3a78f7ad8f1e998ef034d79f5be007d743269e888ca972421",
    "a480c5d1fbf41ed51c0ec071539c783820072828acb756f2934a4e4bf0b5f564",
}
RMS_PATCH_MARKER = "nemotron-swedish-posttraining: pure PyTorch ROCm RMSNorm fallback"
CONV_PATCH_MARKER = "nemotron-swedish-posttraining: MIOpen-free depthwise causal conv1d"

UPSTREAM_BLOCK = """try:
    #from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    raise ImportError(\"mamba-ssm is required by the Mamba model but cannot be imported\")
"""

FALLBACK_BLOCK = f"""try:
    #from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
except ImportError:
    # {RMS_PATCH_MARKER}
    def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
        \"\"\"Autograd-safe equivalent of mamba_ssm's grouped gated RMSNorm.\"\"\"
        if z is not None and not norm_before_gate:
            x = x * torch.nn.functional.silu(z)
        original_dtype = x.dtype
        group_size = group_size or x.shape[-1]
        if x.shape[-1] % group_size:
            raise ValueError(\"RMSNorm hidden size must be divisible by group_size\")
        grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
        grouped = grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
        output = grouped.reshape_as(x).to(original_dtype) * weight
        if bias is not None:
            output = output + bias
        if z is not None and norm_before_gate:
            output = output * torch.nn.functional.silu(z)
        return output
"""

UPSTREAM_CONV_BLOCK = """            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
"""

FALLBACK_CONV_BLOCK = f"""            # {CONV_PATCH_MARKER}
            conv_input = hidden_states_B_C.transpose(1, 2)
            conv_kernel = self.conv1d.weight.squeeze(1)
            conv_windows = nn.functional.pad(
                conv_input, (self.conv_kernel_size - 1, 0)
            ).unfold(-1, self.conv_kernel_size, 1)
            conv_output = (
                conv_windows.float() * conv_kernel.float()[None, :, None, :]
            ).sum(dim=-1)
            if self.use_conv_bias:
                conv_output = conv_output + self.conv1d.bias.float()[None, :, None]
            hidden_states_B_C = self.act(
                conv_output.to(conv_input.dtype).transpose(1, 2)
            )
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_model(model_dir: Path) -> dict[str, Any]:
    target = model_dir / "modeling_nemotron_h.py"
    before = sha256(target)
    text = target.read_text(encoding="utf-8")

    has_rms_patch = RMS_PATCH_MARKER in text
    has_conv_patch = CONV_PATCH_MARKER in text
    if has_rms_patch and has_conv_patch:
        if before not in FULLY_PATCHED_SHA256S:
            raise RuntimeError(f"refusing unexpected fully patched source {before}")
        return {
            "status": "already_patched",
            "accepted_original_sha256s": sorted(ORIGINAL_SHA256S),
            "patched_sha256": before,
            "path": str(target),
        }
    if before not in ORIGINAL_SHA256S | RMS_PATCHED_SHA256S:
        raise RuntimeError(
            f"refusing to patch unexpected source {before}; "
            f"expected one of {sorted(ORIGINAL_SHA256S | RMS_PATCHED_SHA256S)}"
        )
    if not has_rms_patch:
        if text.count(UPSTREAM_BLOCK) != 1:
            raise RuntimeError("expected exactly one upstream RMSNorm import block")
        text = text.replace(UPSTREAM_BLOCK, FALLBACK_BLOCK)
    if not has_conv_patch:
        if text.count(UPSTREAM_CONV_BLOCK) != 1:
            raise RuntimeError("expected exactly one PyTorch Mamba conv1d block")
        text = text.replace(UPSTREAM_CONV_BLOCK, FALLBACK_CONV_BLOCK)

    target.write_text(text, encoding="utf-8")
    after = sha256(target)
    return {
        "status": "patched",
        "original_sha256": before,
        "patched_sha256": after,
        "path": str(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(patch_model(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
