from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .configs import MatmulConfig
from bench.policies.common import autotune_best

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


def _pick_representative_shape(shapes: Sequence[Shape]) -> Shape:
    if not shapes:
        raise RuntimeError("空 shape 集合，无法选择代表 shape")

    log_m = [math.log2(max(1, s[0])) for s in shapes]
    log_n = [math.log2(max(1, s[1])) for s in shapes]
    log_k = [math.log2(max(1, s[2])) for s in shapes]
    log_r = [math.log2(max(1e-9, s[1] / s[2])) for s in shapes]

    med_m = statistics.median(log_m)
    med_n = statistics.median(log_n)
    med_k = statistics.median(log_k)
    med_r = statistics.median(log_r)

    def _score(shape: Shape) -> float:
        m, n, k = shape
        lm = math.log2(max(1, m))
        ln = math.log2(max(1, n))
        lk = math.log2(max(1, k))
        lr = math.log2(max(1e-9, n / k))
        return (lm - med_m) ** 2 + (ln - med_n) ** 2 + (lk - med_k) ** 2 + 0.5 * (lr - med_r) ** 2

    return min(shapes, key=_score)


def pick_typical_shapes(shapes: Sequence[Shape], count: int) -> List[Shape]:
    if count <= 0:
        return []

    unique_shapes = list(dict.fromkeys(shapes))
    if not unique_shapes:
        raise RuntimeError("空 shape 集合，无法选择典型 shape")
    if count >= len(unique_shapes):
        return unique_shapes

    feats = {s: _shape_feature(s) for s in unique_shapes}
    first = _pick_representative_shape(unique_shapes)
    selected: List[Shape] = [first]
    selected_set = {first}

    while len(selected) < count:
        best_shape: Optional[Shape] = None
        best_score = -1.0
        for cand in unique_shapes:
            if cand in selected_set:
                continue
            dist_to_selected = min(_feature_dist2(feats[cand], feats[s]) for s in selected)
            if best_shape is None or dist_to_selected > best_score:
                best_shape = cand
                best_score = dist_to_selected
        if best_shape is None:
            break
        selected.append(best_shape)
        selected_set.add(best_shape)
    return selected


def derive_candidate_pool_from_typical_shapes(
    typical_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    eval_batch: Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]],
) -> tuple[List[MatmulConfig], List[Dict[str, object]]]:
    chosen: List[MatmulConfig] = []
    chosen_ids: set[str] = set()
    report_rows: List[Dict[str, object]] = []

    for shape in typical_shapes:
        cfg, metrics, _ = autotune_best(shape, candidates, eval_batch)
        runtime_us = float(metrics.get("runtime_cost_us", 0.0))
        invalid = int(metrics.get("invalid_config", 0))
        score = runtime_us if invalid == 0 and runtime_us > 0.0 else float("inf")
        report_rows.append(
            {
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
        )
        if cfg.config_id not in chosen_ids:
            chosen.append(cfg)
            chosen_ids.add(cfg.config_id)

    return chosen, report_rows


def write_prototype_report_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
