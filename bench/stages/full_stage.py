from __future__ import annotations

import time
from typing import Dict, List, Tuple

from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.policies.common import INVALID_SCORE, SelectionResult
from bench.reporting.csv_logger import append_records


def run_full(config: BenchmarkConfig, state: BenchmarkState) -> str:
    options = config.options
    op = config.op
    candidates = list(state.candidates)
    if not candidates:
        raise RuntimeError("FULL tune 失败: candidates 为空")

    row_specs: List[Tuple[str, int, tuple]] = []
    row_specs.extend(("tune", idx, shape) for idx, shape in enumerate(config.tune_set))
    row_specs.extend(("eval", idx, shape) for idx, shape in enumerate(config.eval_set))
    if not row_specs:
        raise RuntimeError("FULL tune 失败: 没有 shape 可评测")

    # Single-shot collection only: one global batch over unique_shape * candidate_config.
    unique_shapes = list(dict.fromkeys(shape for _, _, shape in row_specs))
    tune_entries = [(shape, cfg) for shape in unique_shapes for cfg in candidates]
    t0 = time.perf_counter()
    tune_metrics = list(config.triton_evaluator.evaluate_batch(tune_entries))
    batch_tune_time_ms = (time.perf_counter() - t0) * 1000.0
    expected = len(tune_entries)
    if len(tune_metrics) != expected:
        raise RuntimeError(
            f"FULL tune 失败: batch 返回长度不匹配，got={len(tune_metrics)} expected={expected}"
        )

    best_cfg_by_shape: Dict[tuple, object] = {}
    best_metrics_by_shape: Dict[tuple, Dict[str, object]] = {}
    best_score_by_shape: Dict[tuple, float] = {}
    for (shape, cfg), met in zip(tune_entries, tune_metrics):
        invalid = int(met.get("invalid_config", 0))
        runtime_us = float(met.get("runtime_cost_us", 0.0))
        score = runtime_us if invalid == 0 and runtime_us > 0.0 else INVALID_SCORE
        cur_best = best_score_by_shape.get(shape, INVALID_SCORE)
        if score < cur_best:
            best_score_by_shape[shape] = score
            best_cfg_by_shape[shape] = cfg
            best_metrics_by_shape[shape] = dict(met)

    selection_by_shape: Dict[tuple, SelectionResult[object]] = {}
    for shape in unique_shapes:
        best_cfg = best_cfg_by_shape.get(shape)
        best_metrics = best_metrics_by_shape.get(shape)
        best_score = best_score_by_shape.get(shape, INVALID_SCORE)
        if best_cfg is None or best_metrics is None or best_score >= INVALID_SCORE:
            raise RuntimeError(f"FULL tune 失败: shape={shape} 无有效 config（全部 invalid）")
        selection_by_shape[shape] = SelectionResult(
            config=best_cfg,
            cache_key=f"full_shape_{op.shape_to_str(shape)}",
            tune_time_ms=0.0,
            premeasure=best_metrics,
            notes="full_batch_tune,select_only",
        )

    tune_time_owner = next(
        ((split, idx) for split, idx, _ in row_specs if split == "tune"),
        (row_specs[0][0], row_specs[0][1]),
    )

    rows: List[Dict[str, object]] = []
    for split, idx, shape in row_specs:
        base_sel = selection_by_shape.get(shape)
        if base_sel is None:
            raise RuntimeError(f"FULL tune 失败: shape 没有选择结果, shape={shape}")
        sel = SelectionResult(
            config=base_sel.config,
            cache_key=base_sel.cache_key,
            tune_time_ms=batch_tune_time_ms if (split, idx) == tune_time_owner else 0.0,
            premeasure=dict(base_sel.premeasure) if base_sel.premeasure is not None else None,
            notes=base_sel.notes,
        )
        if sel.premeasure is None:
            raise RuntimeError("FULL tune 失败: 缺少预采集 metrics")
        rows.append(
            op.make_bucket_record(
                config,
                split,
                idx,
                shape,
                sel,
                sel.premeasure,
                method="FULL",
            )
        )

    append_records(options.results_csv, rows)
    return (
        f"method=FULL rows={len(rows)} unique_shapes={len(unique_shapes)} "
        f"candidates={len(candidates)} single_batch=1"
    )
