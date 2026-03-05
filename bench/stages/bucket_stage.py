from __future__ import annotations

import time
from typing import Dict, List, Tuple

from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.policies.common import INVALID_SCORE, SelectionResult
from bench.reporting.csv_logger import append_records


def _build_first_shape_index(config: BenchmarkConfig) -> Tuple[Dict[object, tuple], Dict[object, int]]:
    op = config.op
    splits = config.options.bucket_splits
    first_shape_by_key: Dict[object, tuple] = {}
    first_index_by_key: Dict[object, int] = {}
    for idx, shape in enumerate(config.tune_set):
        key = op.eval_key(shape, splits)
        if key not in first_shape_by_key:
            first_shape_by_key[key] = shape
            first_index_by_key[key] = idx
    return first_shape_by_key, first_index_by_key


def _build_tune_entries(
    first_shape_by_key: Dict[object, tuple],
    candidates: list,
) -> List[Tuple[tuple, object]]:
    tune_entries: List[Tuple[tuple, object]] = []
    key_order = list(first_shape_by_key.keys())
    for key in key_order:
        shape = first_shape_by_key[key]
        for cfg in candidates:
            tune_entries.append((shape, cfg))
    return tune_entries


def _evaluate_tune_batch(
    config: BenchmarkConfig, tune_entries: List[Tuple[tuple, object]]
) -> Tuple[List, float]:
    t0 = time.perf_counter()
    tune_metrics = list(config.triton_evaluator.evaluate_batch(tune_entries))
    batch_tune_time_ms = (time.perf_counter() - t0) * 1000.0
    if len(tune_metrics) != len(tune_entries):
        raise RuntimeError(
            f"BUG: tune batch 返回长度不匹配，got={len(tune_metrics)} expected={len(tune_entries)}"
        )
    return tune_metrics, batch_tune_time_ms


def _score_from_metric(met: dict) -> float:
    invalid = int(met.get("invalid_config", 0))
    runtime_us = float(met.get("runtime_cost_us", 0.0))
    return runtime_us if invalid == 0 and runtime_us > 0.0 else INVALID_SCORE


def _compute_best_per_key(
    tune_entries: List[Tuple[tuple, object]],
    tune_metrics: List,
    first_shape_by_key: Dict[object, tuple],
) -> Tuple[Dict[object, object], Dict[object, Dict[str, object]], Dict[object, float]]:
    shape_to_key = {shape: key for key, shape in first_shape_by_key.items()}
    best_cfg_by_key: Dict[object, object] = {}
    best_metrics_by_key: Dict[object, Dict[str, object]] = {}
    best_score_by_key: Dict[object, float] = {}
    for (shape, cfg), met in zip(tune_entries, tune_metrics):
        key = shape_to_key.get(shape)
        if key is None:
            raise RuntimeError(f"BUG: shape 未找到对应 key, shape={shape}")
        score = _score_from_metric(met)
        cur_best = best_score_by_key.get(key, INVALID_SCORE)
        if score < cur_best:
            best_score_by_key[key] = score
            best_cfg_by_key[key] = cfg
            best_metrics_by_key[key] = dict(met)
    return best_cfg_by_key, best_metrics_by_key, best_score_by_key


def _make_selection_for_key(
    key: object,
    best_cfg: object,
    best_metrics: Dict[str, object],
    best_score: float,
    op,
) -> SelectionResult[object]:
    if best_cfg is None or best_metrics is None or best_score >= INVALID_SCORE:
        raise RuntimeError(f"BUCKET tune 失败: key={key} 无有效 config（全部 invalid）")
    return SelectionResult(
        config=best_cfg,
        cache_key=op.eval_key_str(key),
        tune_time_ms=0.0,
        premeasure=best_metrics,
        notes="batch_tune",
    )


def _build_selection_by_key(
    first_shape_by_key: Dict[object, tuple],
    best_cfg_by_key: Dict[object, object],
    best_metrics_by_key: Dict[object, Dict[str, object]],
    best_score_by_key: Dict[object, float],
    config: BenchmarkConfig,
) -> Dict[object, SelectionResult[object]]:
    op = config.op
    selection_by_key: Dict[object, SelectionResult[object]] = {}
    for key in first_shape_by_key.keys():
        best_cfg = best_cfg_by_key.get(key)
        best_metrics = best_metrics_by_key.get(key)
        best_score = best_score_by_key.get(key, INVALID_SCORE)
        selection_by_key[key] = _make_selection_for_key(
            key, best_cfg, best_metrics, best_score, op
        )
    return selection_by_key


def _build_tune_selection(
    base_sel: SelectionResult[object],
    idx: int,
    tune_time_owner_idx: int,
    batch_tune_time_ms: float,
) -> SelectionResult[object]:
    tms = batch_tune_time_ms if idx == tune_time_owner_idx else 0.0
    pre = dict(base_sel.premeasure) if base_sel.premeasure is not None else None
    return SelectionResult(
        config=base_sel.config,
        cache_key=base_sel.cache_key,
        tune_time_ms=tms,
        premeasure=pre,
        notes=base_sel.notes,
    )


def _make_tune_row(
    config: BenchmarkConfig,
    idx: int,
    shape: tuple,
    selection_by_key: Dict[object, SelectionResult[object]],
    tune_time_owner_idx: int,
    batch_tune_time_ms: float,
) -> Dict[str, object]:
    op = config.op
    splits = config.options.bucket_splits
    key = op.eval_key(shape, splits)
    base_sel = selection_by_key.get(key)
    if base_sel is None:
        raise RuntimeError(f"BUG: tune 阶段缺少 key 对应选择结果: {key}")
    sel = _build_tune_selection(base_sel, idx, tune_time_owner_idx, batch_tune_time_ms)
    if sel.premeasure is None:
        raise RuntimeError("BUG: tune 阶段缺少 batch premeasure")
    return op.make_bucket_record(config, "tune", idx, shape, sel, sel.premeasure)


def _build_tune_rows(
    config: BenchmarkConfig,
    selection_by_key: Dict[object, SelectionResult[object]],
    first_index_by_key: Dict[object, int],
    batch_tune_time_ms: float,
) -> List[Dict[str, object]]:
    tune_time_owner_idx = min(first_index_by_key.values())
    return [
        _make_tune_row(config, idx, shape, selection_by_key, tune_time_owner_idx, batch_tune_time_ms)
        for idx, shape in enumerate(config.tune_set)
    ]


def _make_eval_row(
    config: BenchmarkConfig,
    idx: int,
    shape: tuple,
    selection_by_key: Dict[object, SelectionResult[object]],
) -> Dict[str, object]:
    op = config.op
    splits = config.options.bucket_splits
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
    return op.make_bucket_record(config, "eval", idx, shape, sel, {})


def _build_eval_rows(
    config: BenchmarkConfig,
    selection_by_key: Dict[object, SelectionResult[object]],
    tuned_keys: set,
) -> List[Dict[str, object]]:
    missing = config.eval_keys - tuned_keys
    if missing:
        raise RuntimeError(f"BUG: 进入 eval 前存在未调过的 key: {sorted(missing)}")
    return [
        _make_eval_row(config, idx, shape, selection_by_key)
        for idx, shape in enumerate(config.eval_set)
    ]


def _build_eval_entries(
    config: BenchmarkConfig,
    state: BenchmarkState,
    eval_rows: List[Dict[str, object]],
) -> List:
    op = config.op
    cfg_by_id = {str(getattr(cfg, "config_id", "")): cfg for cfg in state.candidates}
    eval_entries = []
    for row in eval_rows:
        cfg = cfg_by_id.get(str(row["config_id"]))
        if cfg is None:
            raise RuntimeError(f"unknown config_id in eval row: {row.get('config_id')}")
        eval_entries.append(op.build_eval_entry(row, cfg))
    return eval_entries


def _apply_eval_metrics(eval_rows: List[Dict[str, object]], eval_metrics: List) -> None:
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


def _run_eval_metrics(
    config: BenchmarkConfig,
    state: BenchmarkState,
    rows: List[Dict[str, object]],
) -> None:
    eval_rows = [row for row in rows if row["split"] == "eval"]
    eval_entries = _build_eval_entries(config, state, eval_rows)
    eval_metrics = config.triton_evaluator.evaluate_batch(eval_entries)
    _apply_eval_metrics(eval_rows, eval_metrics)


def _build_and_eval_rows(
    config: BenchmarkConfig,
    state: BenchmarkState,
    selection_by_key: Dict[object, SelectionResult[object]],
    first_index_by_key: Dict[object, int],
    batch_tune_time_ms: float,
    tuned_keys: set,
) -> List[Dict[str, object]]:
    tune_rows = _build_tune_rows(
        config, selection_by_key, first_index_by_key, batch_tune_time_ms
    )
    eval_rows = _build_eval_rows(config, selection_by_key, tuned_keys)
    rows = tune_rows + eval_rows
    _run_eval_metrics(config, state, rows)
    return rows


def _run_bucket_core(
    config: BenchmarkConfig,
    state: BenchmarkState,
    first_shape_by_key: Dict[object, tuple],
    first_index_by_key: Dict[object, int],
    tuned_keys: set,
) -> List[Dict[str, object]]:
    tune_entries = _build_tune_entries(first_shape_by_key, list(state.candidates))
    tune_metrics, batch_tune_time_ms = _evaluate_tune_batch(config, tune_entries)
    best_cfg, best_metrics, best_score = _compute_best_per_key(
        tune_entries, tune_metrics, first_shape_by_key
    )
    selection_by_key = _build_selection_by_key(
        first_shape_by_key, best_cfg, best_metrics, best_score, config
    )
    return _build_and_eval_rows(
        config, state, selection_by_key, first_index_by_key, batch_tune_time_ms, tuned_keys
    )


def run_bucket(config: BenchmarkConfig, state: BenchmarkState) -> str:
    options = config.options
    splits = options.bucket_splits
    first_shape_by_key, first_index_by_key = _build_first_shape_index(config)
    tuned_keys = set(first_shape_by_key.keys())
    if not tuned_keys:
        raise RuntimeError("BUG: tune_set 为空，无法执行 bucket tune")
    rows = _run_bucket_core(config, state, first_shape_by_key, first_index_by_key, tuned_keys)
    append_records(options.results_csv, rows)
    return (
        f"method=BUCKET rows={len(rows)} tuned_keys={len(tuned_keys)} "
        f"splits={splits} single_batch=1"
    )
