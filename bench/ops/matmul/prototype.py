from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .configs import MatmulConfig
from bench.policies.common import _metrics_score, measure_candidates_batch, select_best_candidate

Shape = Tuple[int, int, int]


def _shape_feature(shape: Shape) -> tuple[float, float, float, float]:
    m, n, k = shape
    lm = math.log2(max(1, m))
    ln = math.log2(max(1, n))
    lk = math.log2(max(1, k))
    lr = math.log2(max(1e-9, n / k))
    return lm, ln, lk, lr


def _feature_dist2(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _compute_shape_medians(shapes: Sequence[Shape]) -> tuple[float, float, float, float]:
    log_m = [math.log2(max(1, s[0])) for s in shapes]
    log_n = [math.log2(max(1, s[1])) for s in shapes]
    log_k = [math.log2(max(1, s[2])) for s in shapes]
    log_r = [math.log2(max(1e-9, s[1] / s[2])) for s in shapes]
    return (
        statistics.median(log_m),
        statistics.median(log_n),
        statistics.median(log_k),
        statistics.median(log_r),
    )


def _score_shape(shape: Shape, med_m: float, med_n: float, med_k: float, med_r: float) -> float:
    m, n, k = shape
    lm = math.log2(max(1, m))
    ln = math.log2(max(1, n))
    lk = math.log2(max(1, k))
    lr = math.log2(max(1e-9, n / k))
    return (lm - med_m) ** 2 + (ln - med_n) ** 2 + (lk - med_k) ** 2 + 0.5 * (lr - med_r) ** 2


def _pick_representative_shape(shapes: Sequence[Shape]) -> Shape:
    if not shapes:
        raise RuntimeError("空 shape 集合，无法选择代表 shape")
    med_m, med_n, med_k, med_r = _compute_shape_medians(shapes)
    return min(shapes, key=lambda s: _score_shape(s, med_m, med_n, med_k, med_r))


def _select_next_best_shape(
    unique_shapes: List[Shape],
    feats: Dict[Shape, tuple[float, float, float, float]],
    selected: List[Shape],
    selected_set: set[Shape],
) -> Optional[Shape]:
    best_shape: Optional[Shape] = None
    best_score = -1.0
    for cand in unique_shapes:
        if cand in selected_set:
            continue
        dist_to_selected = min(_feature_dist2(feats[cand], feats[s]) for s in selected)
        if best_shape is None or dist_to_selected > best_score:
            best_shape = cand
            best_score = dist_to_selected
    return best_shape


def _grow_selected_shapes(
    unique_shapes: List[Shape],
    feats: Dict[Shape, tuple[float, float, float, float]],
    count: int,
) -> List[Shape]:
    first = _pick_representative_shape(unique_shapes)
    selected: List[Shape] = [first]
    selected_set: set[Shape] = {first}
    while len(selected) < count:
        best_shape = _select_next_best_shape(unique_shapes, feats, selected, selected_set)
        if best_shape is None:
            break
        selected.append(best_shape)
        selected_set.add(best_shape)
    return selected


def pick_typical_shapes(shapes: Sequence[Shape], count: int) -> List[Shape]:
    if count <= 0:
        return []
    unique_shapes = list(dict.fromkeys(shapes))
    if not unique_shapes:
        raise RuntimeError("空 shape 集合，无法选择典型 shape")
    if count >= len(unique_shapes):
        return unique_shapes
    feats = {s: _shape_feature(s) for s in unique_shapes}
    return _grow_selected_shapes(unique_shapes, feats, count)


def _build_metrics_index(
    typical_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    flat_metrics: Sequence[Dict[str, object]],
) -> Dict[Tuple[Shape, str], Dict[str, object]]:
    idx_map: Dict[Tuple[Shape, str], Dict[str, object]] = {}
    metric_idx = 0
    for shape in typical_shapes:
        for cfg in candidates:
            idx_map[(shape, cfg.config_id)] = dict(flat_metrics[metric_idx])
            metric_idx += 1
    return idx_map


def _build_eval_from_cache(
    typical_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    flat_metrics: Sequence[Dict[str, object]],
) -> Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]]:
    metrics_by_shape_cfg = _build_metrics_index(typical_shapes, candidates, flat_metrics)

    def _eval_from_cache(shape: Shape, cfgs: Sequence[MatmulConfig]) -> Sequence[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for cfg in cfgs:
            key = (shape, cfg.config_id)
            met = metrics_by_shape_cfg.get(key)
            if met is None:
                raise RuntimeError(f"prototype batch 缺少 metrics: shape={shape}, config_id={cfg.config_id}")
            out.append(dict(met))
        return out

    return _eval_from_cache


def _make_report_row(shape: Shape, cfg: MatmulConfig, metrics: Dict[str, object]) -> Dict[str, object]:
    runtime_us = float(metrics.get("runtime_cost_us", 0.0))
    invalid = int(metrics.get("invalid_config", 0))
    score = _metrics_score(metrics)
    return {
        "shape": f"{shape[0]}x{shape[1]}x{shape[2]}",
        "picked_config_id": cfg.config_id,
        "BLOCK_M": cfg.BLOCK_M,
        "BLOCK_N": cfg.BLOCK_N,
        "BLOCK_K": cfg.BLOCK_K,
        "runtime_cost_us": runtime_us,
        "invalid_config": invalid,
        "score": score,
        "notes": str(metrics.get("notes", "")),
    }


def _resolve_eval_for_shape(
    typical_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    eval_batch: Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]],
    eval_batch_all: Optional[Callable[[Sequence[Shape], Sequence[MatmulConfig]], Sequence[Dict[str, object]]]],
) -> Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]]:
    if eval_batch_all is None or not typical_shapes or not candidates:
        return eval_batch
    flat_metrics = list(eval_batch_all(typical_shapes, candidates))
    expected = len(typical_shapes) * len(candidates)
    if len(flat_metrics) != expected:
        raise RuntimeError(
            f"prototype batch 返回长度不匹配，got={len(flat_metrics)} expected={expected}"
        )
    return _build_eval_from_cache(typical_shapes, candidates, flat_metrics)


def _process_single_shape(
    shape: Shape,
    candidates: Sequence[MatmulConfig],
    eval_for_shape: Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]],
) -> tuple[MatmulConfig, Dict[str, object]]:
    metrics_seq = measure_candidates_batch(shape, candidates, eval_for_shape)
    cfg, metrics = select_best_candidate(candidates, metrics_seq)
    return cfg, _make_report_row(shape, cfg, metrics)


def _process_typical_shapes(
    typical_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    eval_for_shape: Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]],
) -> tuple[List[MatmulConfig], List[Dict[str, object]]]:
    chosen: List[MatmulConfig] = []
    chosen_ids: set[str] = set()
    report_rows: List[Dict[str, object]] = []
    for shape in typical_shapes:
        cfg, row = _process_single_shape(shape, candidates, eval_for_shape)
        report_rows.append(row)
        if cfg.config_id not in chosen_ids:
            chosen.append(cfg)
            chosen_ids.add(cfg.config_id)
    return chosen, report_rows


def derive_candidate_pool_from_typical_shapes(
    typical_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    eval_batch: Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]],
    eval_batch_all: Optional[Callable[[Sequence[Shape], Sequence[MatmulConfig]], Sequence[Dict[str, object]]]] = None,
) -> tuple[List[MatmulConfig], List[Dict[str, object]]]:
    eval_for_shape = _resolve_eval_for_shape(typical_shapes, candidates, eval_batch, eval_batch_all)
    return _process_typical_shapes(typical_shapes, candidates, eval_for_shape)


_PROTOTYPE_CSV_FIELDS = [
    "shape",
    "picked_config_id",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "runtime_cost_us",
    "invalid_config",
    "score",
    "notes",
]


def write_prototype_report_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PROTOTYPE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
