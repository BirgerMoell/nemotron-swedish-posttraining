#!/usr/bin/env python3
"""One-step, eight-GCD FSDP/LoRA smoke for Nemotron 3 Nano 30B-A3B.

Only global rank zero reads the 63 GB checkpoint. Other ranks construct the
same model on the meta device; FSDP materializes and synchronizes their shards.
This is a compatibility gate, not a capability experiment.
"""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


class TokenizedDataset:
    def __init__(self, examples: list[dict[str, list[int]]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, list[int]]]):
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        labels = []
        attention_mask = []
        for row in rows:
            padding = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            labels.append(row["labels"] + [-100] * padding)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def select_examples(tokenizer: Any, data_path: Path, train_cfg: dict[str, Any]):
    # Reuse the exact assistant-boundary implementation qualified by S0.
    from train_lora_smoke import build_training_example, load_records

    accepted = []
    accepted_ids: list[str] = []
    masked_counts: list[int] = []
    supervised_counts: list[int] = []
    for record in load_records(data_path):
        input_ids, labels = build_training_example(
            tokenizer, record, int(train_cfg["max_length"])
        )
        masked = int((labels == -100).sum())
        supervised = int((labels != -100).sum())
        if masked < int(train_cfg["min_masked_prompt_tokens"]):
            continue
        if supervised < int(train_cfg["min_supervised_tokens"]):
            continue
        accepted.append(
            {
                "input_ids": input_ids[0].tolist(),
                "labels": labels[0].tolist(),
            }
        )
        accepted_ids.append(record["id"])
        masked_counts.append(masked)
        supervised_counts.append(supervised)
        if len(accepted) == int(train_cfg["accepted_examples"]):
            break
    if len(accepted) != int(train_cfg["accepted_examples"]):
        raise RuntimeError(
            f"only {len(accepted)} records passed token gates; "
            f'expected {train_cfg["accepted_examples"]}'
        )
    ids_sha = hashlib.sha256(("\n".join(accepted_ids) + "\n").encode()).hexdigest()
    return accepted, accepted_ids, ids_sha, masked_counts, supervised_counts


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    train_cfg = config["training"]

    modules_cache_base = os.environ.get("HF_MODULES_CACHE_BASE")
    if modules_cache_base:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        node_name = os.environ.get("SLURMD_NODENAME", platform.node())
        os.environ["HF_MODULES_CACHE"] = str(
            Path(modules_cache_base) / node_name / f"rank-{local_rank}"
        )

    import peft
    import torch
    import torch.distributed as dist
    import transformers
    from accelerate import PartialState
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    state = PartialState()
    rank = state.process_index
    world_size = state.num_processes
    if world_size != int(train_cfg["expected_world_size"]):
        raise RuntimeError(
            f"WORLD_SIZE={world_size}, expected {train_cfg['expected_world_size']}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA device is required")

    seed = int(train_cfg["seed"])
    random.seed(seed + rank)
    set_seed(seed)
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples, accepted_ids, accepted_ids_sha, masked_counts, supervised_counts = (
        select_examples(tokenizer, args.data, train_cfg)
    )

    # This branch is the important CPU-memory contract. Eight full CPU copies
    # would consume roughly 8 * 63 GB before FSDP could shard the model.
    if state.is_main_process:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                model_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager",
            )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        target_modules=list(train_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # PEFT delegates this in current releases, but pin it explicitly because
    # TRANSFORMER_BASED_WRAP must shard complete Nemotron blocks.
    model._no_split_modules = ["NemotronHBlock"]

    # Nemotron-H deliberately stores router, norm and state-space parameters in
    # FP32 alongside BF16 projection weights. FSDP1 requires one dtype per
    # flattened NemotronHBlock. This lane trains only LoRA parameters, so cast
    # the frozen base storage to BF16 before sharding; forward code still
    # promotes numerically sensitive router/state operations to FP32.
    if not bool(train_cfg.get("cast_frozen_parameters_to_bf16", False)):
        raise RuntimeError("30B FSDP lane requires an explicit frozen-parameter BF16 cast")
    cast_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.is_floating_point()
        and parameter.dtype != torch.bfloat16
    ]
    cast_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
        and parameter.is_floating_point()
        and parameter.dtype != torch.bfloat16
    )
    model.to(dtype=torch.bfloat16)
    remaining_dtypes = {
        str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()
    }
    if remaining_dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"FSDP parameters still have mixed dtypes: {remaining_dtypes}")
    if state.is_main_process:
        print(
            json.dumps(
                {
                    "frozen_parameters_cast_to_bf16": len(cast_parameter_names),
                    "frozen_parameter_elements_cast_to_bf16": cast_parameter_count,
                }
            ),
            flush=True,
        )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError("LoRA produced zero trainable parameters")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        max_steps=int(train_cfg["max_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        max_grad_norm=float(train_cfg["grad_clip"]),
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_drop_last=True,
        seed=seed,
        data_seed=seed,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(examples),
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )

    started = time.time()
    train_result = trainer.train()
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(str(args.output_dir / "adapter"))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir / "adapter")
    trainer.accelerator.wait_for_everyone()

    peak_memory = torch.tensor(
        [torch.cuda.max_memory_allocated()], device=torch.device("cuda"), dtype=torch.int64
    )
    if dist.is_initialized():
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)

    history = [entry for entry in trainer.state.log_history if "loss" in entry]
    losses = [float(entry["loss"]) for entry in history]
    grad_norms = [float(entry["grad_norm"]) for entry in history if "grad_norm" in entry]
    if len(losses) != int(train_cfg["max_steps"]):
        raise RuntimeError(f"expected one loss per step, got {losses}")
    if len(grad_norms) != int(train_cfg["max_steps"]):
        raise RuntimeError(f"expected one gradient norm per step, got {grad_norms}")
    if not all(math.isfinite(value) for value in losses + grad_norms):
        raise RuntimeError("run contained non-finite metrics")

    if trainer.is_world_process_zero():
        adapter_path = args.output_dir / "adapter" / "adapter_model.safetensors"
        if not adapter_path.is_file():
            raise RuntimeError("Trainer did not save an adapter-only safetensors file")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "lane": "lumi-rocm-transformers-peft-fsdp",
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
                "accepted_examples": len(accepted_ids),
                "accepted_ids_sha256": accepted_ids_sha,
                "assistant_only": "final-assistant-turn",
            },
            "training": train_cfg,
            "distributed": {
                "backend": "FSDP FULL_SHARD",
                "world_size": world_size,
                "rank0_only_cpu_checkpoint_load": True,
                "sync_module_states": True,
                "use_orig_params": True,
                "frozen_parameters_cast_to_bf16": len(cast_parameter_names),
                "frozen_parameter_elements_cast_to_bf16": cast_parameter_count,
            },
            "metrics": {
                "losses": losses,
                "gradient_norms": grad_norms,
                "masked_prompt_token_counts": [min(masked_counts)] * len(losses),
                "supervised_token_counts": [min(supervised_counts)] * len(losses),
                "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_bytes": int(peak_memory.item()),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "global_batch_size": world_size,
                "trainer_metrics": train_result.metrics,
            },
            "environment": {
                "container": os.environ.get("CONTAINER"),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "gpu": torch.cuda.get_device_name(),
            },
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
