#!/usr/bin/env python3
"""Stage immutable model assets and a deterministic Swedish smoke sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


ALLOWED_ROLES = {"system", "user", "assistant"}


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_messages(row: dict[str, Any]) -> dict[str, Any]:
    row_id = str(row.get("id", "")).strip()
    messages = row.get("messages")
    if not row_id or not isinstance(messages, list) or not messages:
        raise ValueError("row must contain a non-empty id and messages list")

    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"{row_id}: message is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError(f"{row_id}: invalid role/content")
        cleaned.append({"role": role, "content": content.strip()})

    if not any(m["role"] == "user" for m in cleaned):
        raise ValueError(f"{row_id}: no user turn")
    if not any(m["role"] == "assistant" for m in cleaned):
        raise ValueError(f"{row_id}: no assistant turn")
    return {"id": row_id, "messages": cleaned}


def take_valid(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for row in rows:
        try:
            accepted.append(validate_messages(row))
        except ValueError:
            continue
        if len(accepted) == count:
            break
    if len(accepted) != count:
        raise RuntimeError(f"only found {len(accepted)} valid rows; expected {count}")
    return accepted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    model_cfg = config["model"]
    data_cfg = config["data"]

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from patch_nemotron_rocm import patch_model

    model_dir = args.asset_root / "models" / model_cfg["repo_id"].replace("/", "--")
    data_path = args.asset_root / "data" / f'{config["run_name"]}.jsonl'
    manifest_path = args.asset_root / "manifests" / f'{config["run_name"]}.json'

    snapshot_download(
        repo_id=model_cfg["repo_id"],
        revision=model_cfg["revision"],
        local_dir=model_dir,
    )
    compatibility_patch = patch_model(model_dir)

    stream = load_dataset(
        data_cfg["repo_id"],
        split=data_cfg["split"],
        revision=data_cfg["revision"],
        streaming=True,
    )
    shuffled = stream.shuffle(
        seed=int(data_cfg["seed"]),
        buffer_size=int(data_cfg["shuffle_buffer"]),
    )
    # Request extra rows so malformed inputs can be rejected without changing the seed.
    candidates = islice(shuffled, int(data_cfg["examples"]) * 4)
    rows = take_valid(candidates, int(data_cfg["examples"]))
    data_sha256 = write_jsonl(data_path, rows)

    manifest = {
        "schema_version": 1,
        "run_name": config["run_name"],
        "model": {
            "repo_id": model_cfg["repo_id"],
            "revision": model_cfg["revision"],
            "local_dir": str(model_dir),
            "compatibility_patch": compatibility_patch,
        },
        "data": {
            "repo_id": data_cfg["repo_id"],
            "revision": data_cfg["revision"],
            "split": data_cfg["split"],
            "examples": len(rows),
            "seed": data_cfg["seed"],
            "path": str(data_path),
            "sha256": data_sha256,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
    # The pinned LUMI container can abort while finalizing PyArrow's background
    # filesystem thread after a streaming dataset has already closed. Reaching
    # this point means all downloads, validation, hashes and manifest writes
    # succeeded; bypass only interpreter finalizers, without masking exceptions.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
