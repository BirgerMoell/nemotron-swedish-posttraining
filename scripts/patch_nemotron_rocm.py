#!/usr/bin/env python3
"""Patch the pinned Nemotron-H remote code with a pure-PyTorch gated RMSNorm.

The upstream file guards Mamba/causal-conv kernels but imports one Triton
RMSNorm helper unconditionally. This exact-source patch lets the already
implemented PyTorch Mamba path run on LUMI without installing CUDA extensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ORIGINAL_SHA256 = "ea982af0b805f181573f919ecb001d5bbc0153459923cf4b2f1ccae194e415a4"
PATCH_MARKER = "nemotron-swedish-posttraining: pure PyTorch ROCm RMSNorm fallback"

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
    # {PATCH_MARKER}
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_model(model_dir: Path) -> dict[str, str]:
    target = model_dir / "modeling_nemotron_h.py"
    before = sha256(target)
    text = target.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        return {
            "status": "already_patched",
            "original_sha256": ORIGINAL_SHA256,
            "patched_sha256": before,
            "path": str(target),
        }
    if before != ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected source {before}; expected {ORIGINAL_SHA256}"
        )
    if text.count(UPSTREAM_BLOCK) != 1:
        raise RuntimeError("expected exactly one upstream RMSNorm import block")

    target.write_text(text.replace(UPSTREAM_BLOCK, FALLBACK_BLOCK), encoding="utf-8")
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

