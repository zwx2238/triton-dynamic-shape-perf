from __future__ import annotations

import time
from typing import Dict, List, Tuple

from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.policies.common import INVALID_SCORE, SelectionResult
from bench.reporting.csv_logger import append_records


def run_bucket(config: BenchmarkConfig, state: BenchmarkState) -> str:
    options = config.options
    op = config.op
    splits = options.bucket_splits
    rows: List[Dict[str, object]] = []
    candidates = list(state.candidates)

    first_shape_by_key: Dict[object, tuple] = {}
    first_index_by_key: Dict[object, int] = {}
    for idx, shape in enumerate(config.tune_set):
        key = op.eval_key(shape, splits)
        if key not in first_shape_by_key:
            first_shape_by_key[key] = shape
            first_index_by_key[key] = idx

    tuned_keys = set(first_shape_by_key.keys())
    if not tuned_keys:
        raise RuntimeError("BUG: tune_set 为空，无法执行 bucket tune")

    # Single-shot collection only: one global batch over bucket_representative_shape * candidate_config.
    tune_entries: List[Tuple[tuple, object]] = []
    key_order = list(first_shape_by_key.keys())
    for key in key_order:
        shape = first_shape_by_key[key]
        for cfg in candidates:
            tune_entries.append((shape, cfg))

    t0 = time.perf_counter()
    tune_metrics = list(config.triton_evaluator.evaluate_batch(tune_entries))
    batch_tune_time_ms = (time.perf_counter() - t0) * 1000.0
    if len(tune_metrics) != len(tune_entries):
        raise RuntimeError(
            f"BUG: tune batch 返回长度不匹配，got={len(tune_metrics)} expected={len(tune_entries)}"
        )

    shape_to_key = {shape: key for key, shape in first_shape_by_key.items()}
    best_cfg_by_key: Dict[object, object] = {}
    best_metrics_by_key: Dict[object, Dict[str, object]] = {}
    best_score_by_key: Dict[object, float] = {}

    for (shape, cfg), met in zip(tune_entries, tune_metrics):
        key = shape_to_key.get(shape)
        if key is None:
            raise RuntimeError(f"BUG: shape 未找到对应 key, shape={shape}")
        invalid = int(met.get("invalid_config", 0))
        runtime_us = float(met.get("runtime_cost_us", 0.0))
        score = runtime_us if invalid == 0 and runtime_us > 0.0 else INVALID_SCORE
        cur_best = best_score_by_key.get(key, INVALID_SCORE)
        if score < cur_best:
            best_score_by_key[key] = score
            best_cfg_by_key[key] = cfg
            best_metrics_by_key[key] = dict(met)

    selection_by_key: Dict[object, SelectionResult[object]] = {}
    for key in key_order:
        best_cfg = best_cfg_by_key.get(key)
        best_metrics = best_metrics_by_key.get(key)
        best_score = best_score_by_key.get(key, INVALID_SCORE)
        if best_cfg is None or best_metrics is None or best_score >= INVALID_SCORE:
            raise RuntimeError(f"BUCKET tune 失败: key={key} 无有效 config（全部 invalid）")
        selection_by_key[key] = SelectionResult(
            config=best_cfg,
            cache_key=op.eval_key_str(key),
            tune_time_ms=0.0,
            premeasure=best_metrics,
            notes="batch_tune",
        )

    # `tune_time_ms` now represents one global batch-tune walltime for this run.
    # Record it once to avoid per-key amortization.
    tune_time_owner_idx = min(first_index_by_key.values())

    for idx, shape in enumerate(config.tune_set):
        key = op.eval_key(shape, splits)
        base_sel = selection_by_key.get(key)
        if base_sel is None:
            raise RuntimeError(f"BUG: tune 阶段缺少 key 对应选择结果: {key}")

        sel = SelectionResult(
            config=base_sel.config,
            cache_key=base_sel.cache_key,
            tune_time_ms=batch_tune_time_ms if idx == tune_time_owner_idx else 0.0,
            premeasure=dict(base_sel.premeasure) if base_sel.premeasure is not None else None,
            notes=base_sel.notes,
        )
        if sel.premeasure is None:
            raise RuntimeError("BUG: tune 阶段缺少 batch premeasure")
        rows.append(op.make_bucket_record(config, "tune", idx, shape, sel, sel.premeasure))

    missing = config.eval_keys - tuned_keys
    if missing:
        raise RuntimeError(f"BUG: 进入 eval 前存在未调过的 key: {sorted(missing)}")

    for idx, shape in enumerate(config.eval_set):
        key = op.eval_key(shape, splits)
        base_sel = selection_by_key.get(key)
        if base_sel is None:
            raise RuntimeError(f"BUG: eval 阶段缺少 key 对应选择结果: {key}")
        sel = SelectionResult(
            config=base_sel.config,
            cache_key=base_sel.cache_key,
            tune_time_ms=0.0,
            premeasure=None,
            notes=base_sel.notes,
        )
        rows.append(op.make_bucket_record(config, "eval", idx, shape, sel, {}))

    eval_rows = [row for row in rows if row["split"] == "eval"]
    cfg_by_id = {str(getattr(cfg, "config_id", "")): cfg for cfg in state.candidates}
    eval_entries = []
    for row in eval_rows:
        cfg = cfg_by_id.get(str(row["config_id"]))
        if cfg is None:
            raise RuntimeError(f"unknown config_id in eval row: {row.get('config_id')}")
        eval_entries.append(op.build_eval_entry(row, cfg))
    eval_metrics = config.triton_evaluator.evaluate_batch(eval_entries)
    for row, met in zip(eval_rows, eval_metrics):
        row.update(
            {
                "compile_time_ms": float(met.get("compile_time_ms", 0.0)),
                "runtime_cost_us": float(met.get("runtime_cost_us", 0.0)),
                "invalid_config": int(met.get("invalid_config", 0)),
                "notes": ";".join(
                    [x for x in [str(row.get("notes", "")), str(met.get("notes", ""))] if x]
                ),
            }
        )

    append_records(options.results_csv, rows)
    return (
        f"method=BUCKET rows={len(rows)} tuned_keys={len(tuned_keys)} "
        f"splits={splits} single_batch=1"
    )
