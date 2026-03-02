from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig, get_default_config

Shape = Tuple[int, int, int]
Metrics = Dict[str, object]
EvaluatorFn = Callable[[Shape, MatmulConfig], Metrics]


@dataclass
class SelectionResult:
    config: MatmulConfig
    cache_key: str
    tune_time_ms: float = 0.0
    premeasure: Optional[Metrics] = None
    notes: str = ""


class BudgetTracker:
    def __init__(self, limit_seconds: float) -> None:
        self.limit_seconds = float(limit_seconds)
        self.start_time = time.perf_counter()

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time

    def elapsed_ms(self) -> float:
        return self.elapsed_seconds() * 1000.0

    def remaining_seconds(self) -> float:
        return max(0.0, self.limit_seconds - self.elapsed_seconds())

    def exceeded(self) -> bool:
        return self.elapsed_seconds() >= self.limit_seconds


def autotune_best(
    shape: Shape,
    candidates: Sequence[MatmulConfig],
    evaluator: EvaluatorFn,
    budget: BudgetTracker,
) -> tuple[MatmulConfig, Optional[Metrics], str]:
    best_cfg: Optional[MatmulConfig] = None
    best_metrics: Optional[Metrics] = None
    notes = []

    for cfg in candidates:
        if budget.exceeded():
            notes.append("budget_exceeded")
            break
        metrics = evaluator(shape, cfg)
        if int(metrics.get("invalid_config", 0)) == 1:
            continue
        if best_metrics is None:
            best_cfg = cfg
            best_metrics = metrics
            continue

        best_us = float(best_metrics.get("runtime_cost_us", 1e30))
        curr_us = float(metrics.get("runtime_cost_us", 1e30))
        if curr_us < best_us:
            best_cfg = cfg
            best_metrics = metrics

    if best_cfg is None:
        fallback = get_default_config()
        best_cfg = fallback
        notes.append("fallback_default_config")

    return best_cfg, best_metrics, ";".join(notes)
