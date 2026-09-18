#!/usr/bin/env python3
"""Generate paired Swedish outputs from the base model and an SFT adapter."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def apply_check(text: str, check: dict[str, Any]) -> bool:
    stripped = text.strip()
    kind = check["type"]
    if kind == "regex":
        return re.fullmatch(check["pattern"], stripped) is not None
    if kind == "json_keys":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, dict) and sorted(parsed) == sorted(check["keys"])
    if kind == "bullet_count":
        return sum(line.strip().startswith("- ") for line in stripped.splitlines()) == check["count"]
    if kind == "max_words":
        return 0 < len(stripped.split()) <= check["count"]
    if kind == "contains_all":
        return all(value.casefold() in stripped.casefold() for value in check["values"])
    if kind == "contains_any":
        return any(value.casefold() in stripped.casefold() for value in check["values"])
    if kind == "suffix":
        return stripped.endswith(check["value"])
    raise ValueError(f"unknown check type: {kind}")


def tokenize_prompt(tokenizer: Any, messages: list[dict[str, str]]):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate(model: Any, tokenizer: Any, prompts: list[dict[str, Any]], device: Any):
    import torch

    results = []
    for prompt in prompts:
        inputs = tokenize_prompt(tokenizer, prompt["messages"])
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        completion = tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        results.append(
            {
                "id": prompt["id"],
                "completion": completion,
                "pass": apply_check(completion, prompt["check"]),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA device is required")
    device = torch.device("cuda", 0)
    prompts = load_jsonl(args.prompts)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    config.use_mamba_kernels = False
    config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    started = time.time()
    base_results = generate(model, tokenizer, prompts, device)
    adapted = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    adapted.eval()
    adapter_results = generate(adapted, tokenizer, prompts, device)
    report = {
        "schema_version": 1,
        "prompts": len(prompts),
        "base_passes": sum(result["pass"] for result in base_results),
        "adapter_passes": sum(result["pass"] for result in adapter_results),
        "elapsed_seconds": time.time() - started,
        "base": base_results,
        "adapter": adapter_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("prompts", "base_passes", "adapter_passes", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()

