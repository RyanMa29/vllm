#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

from tests.utils import VLLM_PATH, RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
HF_OVERRIDES = {
    "architectures": ["Qwen3VLForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}

QUERY = "A cat standing in the snow."
DOCUMENT = "This product was excellent and exceeded my expectations."


def _summarize_and_filter_hook(hook_path: Path) -> tuple[dict, Path]:
    total = 0
    nonzero = 0
    warmup_like = 0
    filtered_path = hook_path.with_name(hook_path.stem + "_nonzero.jsonl")

    if filtered_path.exists():
        filtered_path.unlink()

    with (
        hook_path.open("r", encoding="utf-8") as fin,
        filtered_path.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total += 1
            rec = json.loads(line)
            amax = float(rec.get("amax", 0.0))
            std = float(rec.get("std", 0.0))
            head = rec.get("head", [])
            has_signal = (
                (abs(amax) > 0.0)
                or (abs(std) > 0.0)
                or any(abs(float(x)) > 0.0 for x in head)
            )

            # Many warmup/profile records are exactly all-zero.
            if has_signal:
                nonzero += 1
                fout.write(line)
            else:
                warmup_like += 1

    return (
        {
            "total_records": total,
            "nonzero_records": nonzero,
            "all_zero_records": warmup_like,
        },
        filtered_path,
    )


def run_backend(backend: str, output_dir: Path) -> dict:
    args = [
        "--enforce-eager",
        "--max-model-len",
        "8192",
        "--chat-template",
        str(VLLM_PATH / "examples/pooling/score/template/qwen3_vl_reranker.jinja"),
        "--attention-config",
        json.dumps({"backend": backend}),
    ]

    hook_path = output_dir / f"hook_{backend}.jsonl"
    hook_meta_path = output_dir / f"hook_{backend}.jsonl.meta.json"
    if hook_path.exists():
        hook_path.unlink()
    if hook_meta_path.exists():
        hook_meta_path.unlink()

    env = {
        "VLLM_LAYER_HOOK_DUMP_PATH": str(hook_path),
        # Restrict to decoder transformer layers for concise logs.
        "VLLM_LAYER_HOOK_LAYER_PREFIX": "language_model.model.layers",
        "VLLM_LAYER_HOOK_MAX_RECORDS": "200000",
    }

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        override_hf_configs=HF_OVERRIDES,
        env_dict=env,
    ) as remote_server:
        resp = requests.post(
            remote_server.url_for("score"),
            json={
                "model": MODEL_NAME,
                "queries": QUERY,
                "documents": DOCUMENT,
            },
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()

    if not hook_path.exists() or hook_path.stat().st_size == 0:
        meta_text = ""
        if hook_meta_path.exists():
            meta_text = hook_meta_path.read_text(encoding="utf-8")
        raise RuntimeError(
            f"Hook dump missing for {backend}: {hook_path}. Meta: {meta_text[:2000]}"
        )

    hook_stats, filtered_hook = _summarize_and_filter_hook(hook_path)

    return {
        "backend": backend,
        "score": body["data"][0]["score"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "hook_file": str(hook_path),
        "hook_meta_file": str(hook_meta_path),
        "hook_nonzero_file": str(filtered_hook),
        "hook_stats": hook_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run unittest-style backend serve hook dump for Qwen3-VL reranker."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/.cache/ut_cross_encoder_online_vision_exp/backend_diff_analysis",
        help=(
            "Directory to store hook dumps and summary.json. "
            "Default: ~/.cache/ut_cross_encoder_online_vision_exp/backend_diff_analysis"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] output_dir={output_dir}")

    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "run_mode": "unittest_style_vllm_serve",
        "query": QUERY,
        "document": DOCUMENT,
        "results": results,
    }

    for backend in ("FLASH_ATTN", "TRITON_ATTN"):
        results.append(run_backend(backend, output_dir))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[info] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
