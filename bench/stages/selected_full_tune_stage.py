from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.policies.common import INVALID_SCORE, SelectionResult


def _score(metrics: Dict[str, object]) -> float:
    invalid = int(metrics.get("invalid_config", 0))
    runtime_us = float(metrics.get("runtime_cost_us", 0.0))
    if invalid == 0 and runtime_us > 0.0:
        return runtime_us
    return INVALID_SCORE


def _pick_best_candidate(
    candidates: Sequence[object],
    metrics_seq: Sequence[Dict[str, object]],
) -> tuple[object, Dict[str, object]]:
    if len(candidates) != len(metrics_seq):
        raise RuntimeError(
            f"SELECTED_FULL_TUNE: metrics 长度不匹配，got={len(metrics_seq)} expected={len(candidates)}"
        )
    best_cfg = None
    best_metrics = None
    best_score = INVALID_SCORE
    for cfg, metrics in zip(candidates, metrics_seq):
        score = _score(metrics)
        if score < best_score:
            best_cfg = cfg
            best_metrics = metrics
            best_score = score
    if best_cfg is None or best_metrics is None or best_score >= INVALID_SCORE:
        raise RuntimeError("SELECTED_FULL_TUNE: 某个 shape 的候选全部 invalid")
    return best_cfg, dict(best_metrics)


def select_configs_from_sampled_full_tune(
    config: BenchmarkConfig,
    state: BenchmarkState,
    sample_count: int,
) -> tuple[List[object], List[Dict[str, object]], str]:
    op = config.op
    candidates = list(state.candidates)
    if not candidates:
        raise RuntimeError("C1 为空，无法做 sampled full tune")
    p = max(1, min(int(sample_count), len(config.tune_set)))
    sampled_shapes = list(op.pick_typical_shapes(config.tune_set, p))
    if not sampled_shapes:
        raise RuntimeError("sampled_shapes 为空，无法做 C2 选择")

    entries = [(shape, cfg) for shape in sampled_shapes for cfg in candidates]
    t0 = time.perf_counter()
    metrics = list(config.triton_evaluator.evaluate_batch(entries))
    batch_tune_ms = (time.perf_counter() - t0) * 1000.0
    if len(metrics) != len(entries):
        raise RuntimeError(
            f"sampled full tune 返回长度不匹配，got={len(metrics)} expected={len(entries)}"
        )

    picked: List[object] = []
    picked_ids: set[str] = set()
    report_rows: List[Dict[str, object]] = []
    n_cfg = len(candidates)
    for idx, shape in enumerate(sampled_shapes):
        start = idx * n_cfg
        end = start + n_cfg
        best_cfg, best_metrics = _pick_best_candidate(candidates, metrics[start:end])
        cfg_id = str(getattr(best_cfg, "config_id", ""))
        if cfg_id not in picked_ids:
            picked.append(best_cfg)
            picked_ids.add(cfg_id)
        report_rows.append(
            {
                "shape": f"{shape[0]}x{shape[1]}x{shape[2]}",
                "picked_config_id": cfg_id,
                "BLOCK_M": int(getattr(best_cfg, "BLOCK_M", -1)),
                "BLOCK_N": int(getattr(best_cfg, "BLOCK_N", -1)),
                "BLOCK_K": int(getattr(best_cfg, "BLOCK_K", -1)),
                "runtime_cost_us": float(best_metrics.get("runtime_cost_us", 0.0)),
                "invalid_config": int(best_metrics.get("invalid_config", 0)),
                "notes": str(best_metrics.get("notes", "")),
            }
        )

    detail = (
        f"sampled_full_tune shapes={len(sampled_shapes)} c1={len(candidates)} "
        f"entries={len(entries)} c2={len(picked)} batch_tune_ms={batch_tune_ms:.3f}"
    )
    return picked, report_rows, detail


def write_selected_report_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape",
        "picked_config_id",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "runtime_cost_us",
        "invalid_config",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_full_tune_rows_for_candidates(
    config: BenchmarkConfig,
    candidates: Sequence[object],
) -> tuple[str, List[Dict[str, object]]]:
    op = config.op
    eval_shapes = list(config.eval_set)
    if not eval_shapes:
        return "method=FULL_TUNE eval_rows=0", []
    if not candidates:
        raise RuntimeError("C2 为空，无法验证 full tune 上限")

    entries = [(shape, cfg) for shape in eval_shapes for cfg in candidates]
    t0 = time.perf_counter()
    metrics = list(config.triton_evaluator.evaluate_batch(entries))
    batch_tune_ms = (time.perf_counter() - t0) * 1000.0
    if len(metrics) != len(entries):
        raise RuntimeError(
            f"eval full tune 返回长度不匹配，got={len(metrics)} expected={len(entries)}"
        )

    rows: List[Dict[str, object]] = []
    n_cfg = len(candidates)
    for idx, shape in enumerate(eval_shapes):
        start = idx * n_cfg
        end = start + n_cfg
        best_cfg, best_metrics = _pick_best_candidate(candidates, metrics[start:end])
        sel = SelectionResult(
            config=best_cfg,
            cache_key=f"selected_full_tune:{op.shape_to_str(shape)}",
            tune_time_ms=batch_tune_ms if idx == 0 else 0.0,
            premeasure=dict(best_metrics),
            notes="selected_full_tune_batch_all",
        )
        row = op.make_bucket_record(config, "eval", idx, shape, sel, best_metrics)
        row["method"] = "FULL_TUNE"
        rows.append(row)

    detail = (
        f"method=FULL_TUNE eval_rows={len(rows)} candidates={len(candidates)} "
        f"entries={len(entries)} batch_tune_ms={batch_tune_ms:.3f}"
    )
    return detail, rows


def build_full_tune_cache_key(
    config: BenchmarkConfig,
    candidates: Sequence[object],
) -> str:
    payload = {
        "op": str(config.options.op_name),
        "dtype": str(config.options.dtype),
        "warmup": int(config.options.warmup),
        "repeat": int(config.options.repeat),
        "eval_size": len(config.eval_set),
        "eval_shapes": [list(shape) for shape in config.eval_set],
        "candidates": [
            {
                "config_id": str(getattr(cfg, "config_id", "")),
                "BLOCK_M": int(getattr(cfg, "BLOCK_M", -1)),
                "BLOCK_N": int(getattr(cfg, "BLOCK_N", -1)),
                "BLOCK_K": int(getattr(cfg, "BLOCK_K", -1)),
            }
            for cfg in candidates
        ],
    }
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def save_rows_to_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_rows_from_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]
