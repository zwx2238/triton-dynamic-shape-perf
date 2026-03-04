from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List, Sequence

from bench.bucket_tune.runtime import build_benchmark_config, build_benchmark_state
from bench.bucket_tune.types import BenchmarkOptions
from bench.policies.common import INVALID_SCORE, SelectionResult
from bench.reporting.csv_logger import append_records


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
            f"FULL_TUNE: metrics 长度不匹配，got={len(metrics_seq)} expected={len(candidates)}"
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
        raise RuntimeError("FULL_TUNE: 某个 shape 的候选全部 invalid")
    return best_cfg, dict(best_metrics)


def collect_full_tune_rows(options: BenchmarkOptions) -> tuple[str, List[Dict[str, object]]]:
    local_options = replace(options, reset_results=False, prototype_report_csv="")
    config = build_benchmark_config(local_options)
    state = build_benchmark_state(options.op_name)
    op = config.op

    eval_shapes = list(config.eval_set)
    candidates = list(state.candidates)
    if not eval_shapes:
        return "method=FULL_TUNE eval_rows=0", []
    if not candidates:
        raise RuntimeError("FULL_TUNE: candidates 为空")

    # One maximum-size batch: all eval shapes x all candidates.
    tune_entries = [(shape, cfg) for shape in eval_shapes for cfg in candidates]
    t0 = time.perf_counter()
    tune_metrics = list(config.triton_evaluator.evaluate_batch(tune_entries))
    batch_tune_time_ms = (time.perf_counter() - t0) * 1000.0
    if len(tune_metrics) != len(tune_entries):
        raise RuntimeError(
            f"FULL_TUNE: batch 返回长度不匹配，got={len(tune_metrics)} expected={len(tune_entries)}"
        )

    rows: List[Dict[str, object]] = []
    n_cfg = len(candidates)
    for idx, shape in enumerate(eval_shapes):
        start = idx * n_cfg
        end = start + n_cfg
        shape_metrics = tune_metrics[start:end]
        best_cfg, best_metrics = _pick_best_candidate(candidates, shape_metrics)
        sel = SelectionResult(
            config=best_cfg,
            cache_key=f"full_tune:{op.shape_to_str(shape)}",
            tune_time_ms=batch_tune_time_ms if idx == 0 else 0.0,
            premeasure=dict(best_metrics),
            notes="full_tune_batch_all",
        )
        row = op.make_bucket_record(config, "eval", idx, shape, sel, best_metrics)
        row["method"] = "FULL_TUNE"
        rows.append(row)

    detail = (
        f"method=FULL_TUNE eval_rows={len(rows)} "
        f"candidates={len(candidates)} entries={len(tune_entries)} "
        f"batch_tune_ms={batch_tune_time_ms:.3f}"
    )
    return detail, rows


def run_full_tune(options: BenchmarkOptions, results_csv: str) -> str:
    detail, rows = collect_full_tune_rows(options)
    append_records(results_csv, rows)
    return detail
