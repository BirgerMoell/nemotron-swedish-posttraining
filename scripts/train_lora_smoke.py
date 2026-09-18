#!/usr/bin/env python3
"""Minimal fail-closed Nemotron-H LoRA smoke trainer for one ROCm GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    supplied = os.environ.get("REPO_COMMIT")
    if supplied:
        return supplied
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("smoke dataset is empty")
    return records


def _token_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": "pt",
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)[0]
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)[0]


def build_training_example(tokenizer: Any, record: dict[str, Any], max_length: int):
    import torch

    messages = record["messages"]
    assistant_positions = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
    if not assistant_positions:
        raise ValueError(f'{record.get("id", "unknown")}: no assistant turn')

    # Match NVIDIA's chat-data rule: supervise only the final assistant response.
    final_index = assistant_positions[-1]
    conversation = messages[: final_index + 1]
    prompt = conversation[:-1]
    prompt_ids = _token_ids(tokenizer, prompt, add_generation_prompt=True)
    full_ids = _token_ids(tokenizer, conversation, add_generation_prompt=False)

    prompt_list = prompt_ids.tolist()
    full_list = full_ids.tolist()
    if full_list[: len(prompt_list)] != prompt_list:
        raise ValueError(f'{record.get("id", "unknown")}: chat template prefix mismatch')

    labels = torch.full_like(full_ids, -100)
    labels[len(prompt_ids) :] = full_ids[len(prompt_ids) :]
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    if int((labels != -100).sum()) < 2:
        raise ValueError(f'{record.get("id", "unknown")}: fewer than two supervised tokens')
    return full_ids.unsqueeze(0), labels.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    train_cfg = config["training"]

    import peft
    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA device is required")

    seed = int(train_cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    model_config = AutoConfig.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    if not hasattr(model_config, "use_mamba_kernels"):
        raise RuntimeError("expected a Nemotron-H config with use_mamba_kernels")
    model_config.use_mamba_kernels = False
    model_config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to("cuda")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        target_modules=list(train_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("LoRA produced zero trainable parameters")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    records = load_records(args.data)
    losses: list[float] = []
    grad_norms: list[float] = []
    started = time.time()

    optimizer.zero_grad(set_to_none=True)
    for step in range(int(train_cfg["max_steps"])):
        record = records[step % len(records)]
        input_ids, labels = build_training_example(
            tokenizer, record, int(train_cfg["max_length"])
        )
        input_ids = input_ids.to("cuda")
        labels = labels.to("cuda")
        attention_mask = torch.ones_like(input_ids)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            max_norm=float(train_cfg["grad_clip"]),
        )
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
            raise RuntimeError(f"invalid gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(grad_norm.detach().cpu()))
        print(
            json.dumps(
                {
                    "step": step + 1,
                    "id": record["id"],
                    "tokens": int(input_ids.numel()),
                    "supervised_tokens": int((labels != -100).sum()),
                    "loss": losses[-1],
                    "grad_norm": grad_norms[-1],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if not all(math.isfinite(value) for value in losses + grad_norms):
        raise RuntimeError("run contained non-finite metrics")

    adapter_dir = args.output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "lane": "lumi-rocm-transformers-peft",
        "run_name": config["run_name"],
        "repository_commit": git_commit(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "model": {
            "repo_id": config["model"]["repo_id"],
            "revision": config["model"]["revision"],
            "path": str(args.model_dir),
            "use_mamba_kernels": False,
            "attention_implementation": "eager",
        },
        "data": {
            "repo_id": config["data"]["repo_id"],
            "revision": config["data"]["revision"],
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "records": len(records),
            "assistant_only": "final-assistant-turn",
        },
        "training": train_cfg,
        "metrics": {
            "losses": losses,
            "gradient_norms": grad_norms,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "trainable_parameters": trainable,
            "total_parameters": total,
        },
        "environment": {
            "container": os.environ.get("CONTAINER"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()

