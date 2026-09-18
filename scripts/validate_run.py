#!/usr/bin/env python3
"""Validate the contract of a completed smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    manifest_path = args.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if manifest.get("status") != "complete":
        failures.append("manifest status is not complete")
    losses = manifest.get("metrics", {}).get("losses", [])
    gradients = manifest.get("metrics", {}).get("gradient_norms", [])
    masked_tokens = manifest.get("metrics", {}).get("masked_prompt_token_counts", [])
    expected_steps = int(manifest.get("training", {}).get("max_steps", -1))
    minimum_masked = int(manifest.get("training", {}).get("min_masked_prompt_tokens", -1))
    if len(losses) != expected_steps or not all(math.isfinite(x) for x in losses):
        failures.append("loss trace is missing or non-finite")
    if len(gradients) != expected_steps or not all(math.isfinite(x) and x > 0 for x in gradients):
        failures.append("gradient trace is missing, zero, or non-finite")
    if len(masked_tokens) != expected_steps or not all(x >= minimum_masked for x in masked_tokens):
        failures.append("assistant-only masking boundary was not exercised on every step")
    adapter = args.run_dir / "adapter" / "adapter_model.safetensors"
    if not adapter.is_file() or adapter.stat().st_size == 0:
        failures.append("adapter_model.safetensors is missing or empty")
    else:
        try:
            from safetensors import safe_open

            with safe_open(adapter, framework="pt", device="cpu") as handle:
                adapter_keys = list(handle.keys())
            if not adapter_keys or not all("lora_" in key for key in adapter_keys):
                failures.append("saved artifact is not an adapter-only LoRA state dict")
        except Exception as exc:  # validation should report, not hide, serialization failures
            failures.append(f"adapter safetensors could not be inspected: {exc}")
    if not (args.run_dir / "adapter" / "adapter_config.json").is_file():
        failures.append("adapter_config.json is missing")
    if manifest.get("model", {}).get("use_mamba_kernels") is not False:
        failures.append("ROCm smoke did not record PyTorch Mamba fallback")
    if manifest.get("repository_commit") in {None, "", "unknown"}:
        failures.append("repository commit is unresolved")

    report = {
        "status": "fail" if failures else "pass",
        "manifest": str(manifest_path),
        "adapter": str(adapter),
        "failures": failures,
    }
    (args.run_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
