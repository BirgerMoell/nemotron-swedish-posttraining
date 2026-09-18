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

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    modules_cache_base = os.environ.get("HF_MODULES_CACHE_BASE")
    if modules_cache_base:
        node_name = os.environ.get("SLURMD_NODENAME", platform.node())
        os.environ["HF_MODULES_CACHE"] = str(
            Path(modules_cache_base) / node_name / f"rank-{local_rank}"
        )

    import peft
    import torch
    import torch.distributed as dist
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA device is required")
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seed = int(train_cfg["seed"])
    expected_world_size = int(train_cfg.get("expected_world_size", world_size))
    if world_size != expected_world_size:
        raise RuntimeError(f"WORLD_SIZE={world_size}, expected {expected_world_size}")
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
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
    model.to(device)

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
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = get_scheduler(
        name=train_cfg.get("lr_scheduler", "constant"),
        optimizer=optimizer,
        num_warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        num_training_steps=int(train_cfg["max_steps"]),
    )
    records = load_records(args.data)
    accepted_target = int(train_cfg.get("accepted_examples", train_cfg["max_steps"] * world_size))
    if accepted_target % world_size:
        raise ValueError("accepted_examples must be divisible by WORLD_SIZE")
    if bool(train_cfg.get("data_prevalidated", False)):
        selected = records[:accepted_target]
        selected_ids = [record["id"] for record in selected]
        if len(selected) != accepted_target or len(set(selected_ids)) != accepted_target:
            raise RuntimeError("prevalidated data count or ID uniqueness contract failed")
        rank_examples = []
        for record in selected[rank::world_size]:
            input_ids, labels = build_training_example(
                tokenizer, record, int(train_cfg["max_length"])
            )
            masked_tokens = int((labels == -100).sum())
            supervised_tokens = int((labels != -100).sum())
            if (
                masked_tokens < int(train_cfg["min_masked_prompt_tokens"])
                or supervised_tokens < int(train_cfg.get("min_supervised_tokens", 2))
            ):
                raise RuntimeError(f'prevalidated record failed token gates: {record["id"]}')
            rank_examples.append((record["id"], input_ids, labels))
        accepted_ids_sha256 = hashlib.sha256(
            ("\n".join(selected_ids) + "\n").encode("utf-8")
        ).hexdigest()
    else:
        accepted: list[tuple[str, Any, Any]] = []
        accepted_ids: set[str] = set()
        for record in records:
            if record["id"] in accepted_ids:
                continue
            input_ids, labels = build_training_example(
                tokenizer, record, int(train_cfg["max_length"])
            )
            masked_tokens = int((labels == -100).sum())
            supervised_tokens = int((labels != -100).sum())
            if (
                masked_tokens >= int(train_cfg["min_masked_prompt_tokens"])
                and supervised_tokens >= int(train_cfg.get("min_supervised_tokens", 2))
            ):
                accepted.append((record["id"], input_ids, labels))
                accepted_ids.add(record["id"])
            if len(accepted) == accepted_target:
                break
        if len(accepted) != accepted_target:
            raise RuntimeError(
                f"only {len(accepted)} distinct records passed token-window gates; "
                f"expected {accepted_target}"
            )
        rank_examples = accepted[rank::world_size]
        accepted_ids_sha256 = hashlib.sha256(
            ("\n".join(item[0] for item in accepted) + "\n").encode("utf-8")
        ).hexdigest()
    if len(rank_examples) != int(train_cfg["max_steps"]):
        raise RuntimeError(
            f"rank {rank} received {len(rank_examples)} examples; "
            f'expected {train_cfg["max_steps"]}'
        )
    losses: list[float] = []
    grad_norms: list[float] = []
    learning_rates: list[float] = []
    supervised_token_counts: list[int] = []
    masked_prompt_token_counts: list[int] = []
    started = time.time()

    optimizer.zero_grad(set_to_none=True)
    for step, (record_id, input_ids, labels) in enumerate(rank_examples):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
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
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        metric_pair = torch.tensor(
            [float(loss.detach()), float(grad_norm.detach())], device=device, dtype=torch.float64
        )
        if world_size > 1:
            dist.all_reduce(metric_pair, op=dist.ReduceOp.SUM)
            metric_pair /= world_size
        losses.append(float(metric_pair[0].cpu()))
        grad_norms.append(float(metric_pair[1].cpu()))
        learning_rates.append(float(scheduler.get_last_lr()[0]))
        supervised_token_counts.append(int((labels != -100).sum()))
        masked_prompt_token_counts.append(int((labels == -100).sum()))
        if rank == 0 and (step == 0 or (step + 1) % 10 == 0 or step + 1 == len(rank_examples)):
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "rank0_id": record_id,
                        "tokens_per_rank": int(input_ids.numel()),
                        "global_batch_size": world_size,
                        "supervised_tokens_rank0": supervised_token_counts[-1],
                        "masked_prompt_tokens_rank0": masked_prompt_token_counts[-1],
                        "mean_loss": losses[-1],
                        "mean_grad_norm": grad_norms[-1],
                        "learning_rate": learning_rates[-1],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        checkpoint_every = int(train_cfg.get("checkpoint_every_steps", 0))
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                checkpoint_dir = args.output_dir / "checkpoints" / f"step-{step + 1:06d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_model = model.module if world_size > 1 else model
                checkpoint_model.save_pretrained(checkpoint_dir / "adapter", safe_serialization=True)
                torch.save(
                    {
                        "step": step + 1,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    checkpoint_dir / "trainer_state.pt",
                )
            if world_size > 1:
                dist.barrier()

    if not all(math.isfinite(value) for value in losses + grad_norms):
        raise RuntimeError("run contained non-finite metrics")

    if world_size > 1:
        dist.barrier()
    adapter_dir = args.output_dir / "adapter"
    if rank == 0:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = model.module if world_size > 1 else model
        unwrapped_model.save_pretrained(adapter_dir, safe_serialization=True)
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
            "modeling_code_sha256": file_sha256(args.model_dir / "modeling_nemotron_h.py"),
        },
        "data": {
            "repo_id": config["data"]["repo_id"],
            "revision": config["data"]["revision"],
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "records": len(records),
            "accepted_examples": accepted_target,
            "accepted_ids_sha256": accepted_ids_sha256,
            "assistant_only": "final-assistant-turn",
        },
        "training": train_cfg,
        "metrics": {
            "losses": losses,
            "gradient_norms": grad_norms,
            "learning_rates": learning_rates,
            "supervised_token_counts": supervised_token_counts,
            "masked_prompt_token_counts": masked_prompt_token_counts,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "world_size": world_size,
            "global_batch_size": world_size,
        },
        "environment": {
            "container": os.environ.get("CONTAINER"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
