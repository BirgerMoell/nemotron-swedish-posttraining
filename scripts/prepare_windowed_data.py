#!/usr/bin/env python3
"""Create a distinct, token-window-qualified training set from staged candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from train_lora_smoke import build_training_example, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_cfg = config["training"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    candidates = load_records(args.candidates)
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    length_sum = 0
    supervised_sum = 0
    masked_sum = 0

    for record in candidates:
        if record["id"] in seen:
            continue
        input_ids, labels = build_training_example(
            tokenizer, record, int(train_cfg["max_length"])
        )
        masked = int((labels == -100).sum())
        supervised = int((labels != -100).sum())
        if (
            masked < int(train_cfg["min_masked_prompt_tokens"])
            or supervised < int(train_cfg["min_supervised_tokens"])
        ):
            continue
        accepted.append(record)
        seen.add(record["id"])
        length_sum += int(input_ids.numel())
        supervised_sum += supervised
        masked_sum += masked
        if len(accepted) == int(train_cfg["accepted_examples"]):
            break

    expected = int(train_cfg["accepted_examples"])
    if len(accepted) != expected:
        raise RuntimeError(f"accepted {len(accepted)} rows from {len(candidates)}; expected {expected}")
    data_sha256 = write_jsonl(args.output, accepted)
    ids_sha256 = hashlib.sha256(
        ("\n".join(row["id"] for row in accepted) + "\n").encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "run_name": config["run_name"],
        "candidate_records": len(candidates),
        "accepted_records": len(accepted),
        "rejected_records": len(candidates) - len(accepted),
        "mean_tokens": length_sum / len(accepted),
        "mean_supervised_tokens": supervised_sum / len(accepted),
        "mean_masked_prompt_tokens": masked_sum / len(accepted),
        "data_sha256": data_sha256,
        "ids_sha256": ids_sha256,
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

