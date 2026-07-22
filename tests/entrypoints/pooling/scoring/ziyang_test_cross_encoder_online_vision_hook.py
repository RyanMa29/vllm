#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re
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
LAYER_INDEX_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.self_attn$")


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


def _load_hook_records(hook_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with hook_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _layer_index(layer_name: str) -> int | None:
    m = LAYER_INDEX_RE.search(layer_name)
    if m is None:
        return None
    return int(m.group(1))


def _write_attention_output_csv(
    output_csv: Path,
    rows: list[dict[str, float | int]],
) -> None:
    fieldnames = [
        "layer",
        "avg_d_std",
        "avg_d_mean",
        "avg_d_amax",
        "max_d_std",
        "max_d_mean",
        "max_d_amax",
    ]

    try:
        import pandas as pd

        pd.DataFrame(rows, columns=fieldnames).to_csv(output_csv, index=False)
        return
    except Exception:
        pass

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _analyze_attention_output_only(
    output_dir: Path,
    flash_hook_path: Path,
    triton_hook_path: Path,
    prompt_tokens: int | None,
) -> dict[str, Any]:
    flash_records = _load_hook_records(flash_hook_path)
    triton_records = _load_hook_records(triton_hook_path)

    def _to_map(
        records: list[dict[str, Any]],
    ) -> dict[tuple[str, int, tuple[int, ...]], dict[str, Any]]:
        out: dict[tuple[str, int, tuple[int, ...]], dict[str, Any]] = {}
        for rec in records:
            layer = str(rec.get("layer", ""))
            call_idx = int(rec.get("call_idx", -1))
            shape_obj = rec.get("shape", [])
            if not isinstance(shape_obj, list):
                continue
            try:
                shape = tuple(int(x) for x in shape_obj)
            except Exception:
                continue
            out[(layer, call_idx, shape)] = rec
        return out

    flash_map = _to_map(flash_records)
    triton_map = _to_map(triton_records)

    matched_keys = sorted(set(flash_map.keys()) & set(triton_map.keys()))
    mismatched = len(set(flash_map.keys()) ^ set(triton_map.keys()))

    layer_stats: dict[int, list[dict[str, float]]] = defaultdict(list)
    first_diff: dict[str, Any] | None = None
    aligned_pairs_used = 0

    for layer_name, call_idx, shape in matched_keys:
        layer_idx = _layer_index(layer_name)
        if layer_idx is None:
            continue
        if len(shape) != 2:
            continue
        if prompt_tokens is not None and shape[0] != prompt_tokens:
            continue

        f = flash_map[(layer_name, call_idx, shape)]
        t = triton_map[(layer_name, call_idx, shape)]

        d_std = abs(float(f.get("std", 0.0)) - float(t.get("std", 0.0)))
        d_mean = abs(float(f.get("mean", 0.0)) - float(t.get("mean", 0.0)))
        d_amax = abs(float(f.get("amax", 0.0)) - float(t.get("amax", 0.0)))

        rec_metrics: dict[str, float] = {
            "d_std": d_std,
            "d_mean": d_mean,
            "d_amax": d_amax,
        }
        rec_detail: dict[str, Any] = {
            "call_idx": int(call_idx),
            "layer_name": layer_name,
            "shape": list(shape),
            "d_std": d_std,
            "d_mean": d_mean,
            "d_amax": d_amax,
        }

        if first_diff is None:
            first_diff = rec_detail

        layer_stats[layer_idx].append(rec_metrics)
        aligned_pairs_used += 1

    if not layer_stats:
        raise RuntimeError(
            "No aligned self_attn output pairs found for analysis. "
            "Please check hook dump contents and prompt_tokens alignment."
        )

    rows: list[dict[str, float | int]] = []
    layer_rank_std: list[tuple[int, float, float]] = []
    layer_rank_amax: list[tuple[int, float, float]] = []

    for layer in sorted(layer_stats.keys()):
        vals = layer_stats[layer]
        n = float(len(vals))
        avg_d_std = sum(v["d_std"] for v in vals) / n
        avg_d_mean = sum(v["d_mean"] for v in vals) / n
        avg_d_amax = sum(v["d_amax"] for v in vals) / n
        max_d_std = max(v["d_std"] for v in vals)
        max_d_mean = max(v["d_mean"] for v in vals)
        max_d_amax = max(v["d_amax"] for v in vals)

        rows.append(
            {
                "layer": layer,
                "avg_d_std": avg_d_std,
                "avg_d_mean": avg_d_mean,
                "avg_d_amax": avg_d_amax,
                "max_d_std": max_d_std,
                "max_d_mean": max_d_mean,
                "max_d_amax": max_d_amax,
            }
        )
        layer_rank_std.append((layer, avg_d_std, avg_d_amax))
        layer_rank_amax.append((layer, avg_d_amax, avg_d_std))

    layer_rank_std.sort(key=lambda x: x[1], reverse=True)
    layer_rank_amax.sort(key=lambda x: x[1], reverse=True)

    csv_path = output_dir / "analysis_attention_output_only_table.csv"

    _write_attention_output_csv(csv_path, rows)

    return {
        "analysis_csv": str(csv_path),
        "analysis_rows": len(rows),
        "analysis_aligned_pairs": aligned_pairs_used,
        "analysis_mismatched_pairs": mismatched,
        "analysis_first_diff": first_diff,
        "top10_by_avg_d_std": [
            {
                "layer": layer,
                "avg_d_std": avg_std,
                "avg_d_amax": avg_amax,
            }
            for layer, avg_std, avg_amax in layer_rank_std[:10]
        ],
        "top10_by_avg_d_amax": [
            {
                "layer": layer,
                "avg_d_amax": avg_amax,
                "avg_d_std": avg_std,
            }
            for layer, avg_amax, avg_std in layer_rank_amax[:10]
        ],
    }


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

    flash_result = next(r for r in results if r["backend"] == "FLASH_ATTN")
    triton_result = next(r for r in results if r["backend"] == "TRITON_ATTN")
    analysis_info = _analyze_attention_output_only(
        output_dir=output_dir,
        flash_hook_path=Path(str(flash_result["hook_file"])),
        triton_hook_path=Path(str(triton_result["hook_file"])),
        prompt_tokens=int(flash_result["prompt_tokens"]),
    )
    summary["analysis"] = analysis_info

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[info] wrote {summary_path}")
    print(f"[info] wrote {analysis_info['analysis_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
