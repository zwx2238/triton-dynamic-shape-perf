from __future__ import annotations

import time
from typing import Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.common import BudgetTracker, EvaluatorFn, SelectionResult, autotune_best

Shape = Tuple[int, int, int]


class FixedStaticPolicy:
    """Single-config baseline for dynamic shapes."""

    method = "U"

    def __init__(self, fixed_config: MatmulConfig, cache_key: str, notes: str = "") -> None:
        self.fixed_config = fixed_config
        self.cache_key = cache_key
        self.notes = notes

    def select(self, shape: Shape) -> SelectionResult:
        return SelectionResult(
            config=self.fixed_config,
            cache_key=self.cache_key,
            tune_time_ms=0.0,
            premeasure=None,
            notes=self.notes,
        )


def pick_fixed_config_from_reference(
    reference_shape: Shape,
    candidates: Sequence[MatmulConfig],
    evaluator: EvaluatorFn,
    budget: BudgetTracker,
) -> tuple[MatmulConfig, float, str]:
    t0 = time.perf_counter()
    best_cfg, _, notes = autotune_best(reference_shape, candidates, evaluator, budget)
    tune_time_ms = (time.perf_counter() - t0) * 1000.0
    return best_cfg, tune_time_ms, notes
