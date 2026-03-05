from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence, Tuple

from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.policies.common import INVALID_SCORE, SelectionResult, _metrics_score
from bench.reporting.csv_logger import append_records


def _timed_evaluate_batch(
    evaluator, entries: Sequence[tuple],
) -> Tuple[List[Dict[str, object]], float]:
    t0 = time.perf_counter()
    metrics = list(evaluator.evaluate_batch(entries))
    batch_ms = (time.perf_counter() - t0) * 1000.0
    if len(metrics) != len(entries):
        raise RuntimeError(
            f"batch 返回长度不匹配: got={len(metrics)} expected={len(entries)}"
        )
    return metrics, batch_ms


def _select_best_per_group(
    entries: Sequence[Tuple[tuple, object]],
    metrics: Sequence[Dict[str, object]],
    group_fn: Callable[[tuple], object],
) -> Dict[object, Tuple[object, Dict[str, object]]]:
    best_score: Dict[object, float] = {}
    result: Dict[object, Tuple[object, Dict[str, object]]] = {}
    for (shape, cfg), met in zip(entries, metrics):
        group = group_fn(shape)
        score = _metrics_score(met)
        if score < best_score.get(group, INVALID_SCORE):
            best_score[group] = score
            result[group] = (cfg, dict(met))
    return result


def _build_selection_map(
    best: Dict[object, Tuple[object, Dict[str, object]]],
    groups: Sequence[object],
    cache_key_fn: Callable[[object], str],
    notes: str,
    method: str,
) -> Dict[object, SelectionResult]:
    sel: Dict[object, SelectionResult] = {}
    for g in groups:
        pair = best.get(g)
        if pair is None:
            raise RuntimeError(f"{method} 失败: {g} 无有效 config")
        cfg, met = pair
        sel[g] = SelectionResult(
            config=cfg, cache_key=cache_key_fn(g),
            premeasure=met, notes=notes,
        )
    return sel


def _tune_phase(
    evaluator, tune_shapes: Sequence[tuple],
    candidates: Sequence, group_fn: Callable[[tuple], object],
    cache_key_fn: Callable[[object], str], method: str,
) -> Dict[object, SelectionResult]:
    entries = [(s, c) for s in tune_shapes for c in candidates]
    metrics, _ = _timed_evaluate_batch(evaluator, entries)
    groups = list(dict.fromkeys(group_fn(s) for s in tune_shapes))
    best = _select_best_per_group(entries, metrics, group_fn)
    return _build_selection_map(
        best, groups, cache_key_fn,
        notes="batch_tune", method=method,
    )


def _build_eval_entries(
    config: BenchmarkConfig,
    sel_map: Dict[object, SelectionResult],
    group_fn: Callable[[tuple], object],
    method: str,
) -> List[tuple]:
    entries: List[tuple] = []
    for shape in config.eval_set:
        key = group_fn(shape)
        sel = sel_map.get(key)
        if sel is None:
            raise RuntimeError(f"{method}: eval shape {shape} 无匹配 group {key}")
        entries.append((shape, sel.config))
    return entries


def _build_eval_rows(
    config: BenchmarkConfig,
    sel_map: Dict[object, SelectionResult],
    eval_metrics: Sequence[Dict[str, object]],
    group_fn: Callable[[tuple], object],
    method: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, (shape, met) in enumerate(zip(config.eval_set, eval_metrics)):
        sel = sel_map[group_fn(shape)]
        row = config.op.make_bucket_record(
            config, "eval", idx, shape, sel, dict(met), method=method,
        )
        rows.append(row)
    return rows


def _eval_with_profiling(
    config: BenchmarkConfig,
    sel_map: Dict[object, SelectionResult],
    group_fn: Callable[[tuple], object],
    method: str,
) -> List[Dict[str, object]]:
    eval_entries = _build_eval_entries(config, sel_map, group_fn, method)
    eval_metrics, _ = _timed_evaluate_batch(
        config.triton_evaluator, eval_entries,
    )
    return _build_eval_rows(
        config, sel_map, eval_metrics, group_fn, method,
    )


def _eval_from_premeasure(
    config: BenchmarkConfig,
    sel_map: Dict[object, SelectionResult],
    group_fn: Callable[[tuple], object],
    method: str,
) -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []
    for shape in config.eval_set:
        sel = sel_map.get(group_fn(shape))
        if sel is None or sel.premeasure is None:
            raise RuntimeError(f"{method}: shape {shape} 缺少 premeasure")
        metrics.append(sel.premeasure)
    return _build_eval_rows(
        config, sel_map, metrics, group_fn, method,
    )


def _eval_phase(
    config: BenchmarkConfig,
    sel_map: Dict[object, SelectionResult],
    group_fn: Callable[[tuple], object],
    method: str,
    reuse_tune_metrics: bool,
) -> List[Dict[str, object]]:
    if reuse_tune_metrics:
        return _eval_from_premeasure(config, sel_map, group_fn, method)
    return _eval_with_profiling(config, sel_map, group_fn, method)


def _validate_inputs(
    candidates: Sequence, tune_shapes: Sequence[tuple], method: str,
) -> None:
    if not candidates:
        raise RuntimeError(f"{method} 失败: candidates 为空")
    if not tune_shapes:
        raise RuntimeError(f"{method} 失败: tune_shapes 为空")


def _format_summary(
    method: str, rows: list,
    tune_shapes: Sequence, candidates: Sequence,
) -> str:
    return (
        f"method={method} eval_rows={len(rows)} "
        f"tune_shapes={len(tune_shapes)} candidates={len(candidates)}"
    )


def run_tune_eval(
    config: BenchmarkConfig,
    state: BenchmarkState,
    *,
    method: str,
    tune_shapes: Sequence[tuple],
    group_fn: Callable[[tuple], object],
    cache_key_fn: Callable[[object], str],
    reuse_tune_metrics: bool = False,
) -> str:
    candidates = list(state.candidates)
    _validate_inputs(candidates, tune_shapes, method)
    sel_map = _tune_phase(
        config.triton_evaluator, tune_shapes, candidates,
        group_fn, cache_key_fn, method,
    )
    rows = _eval_phase(config, sel_map, group_fn, method, reuse_tune_metrics)
    append_records(config.options.results_csv, rows)
    return _format_summary(method, rows, tune_shapes, candidates)
