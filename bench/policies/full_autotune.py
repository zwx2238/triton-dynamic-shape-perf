from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.common import BudgetTracker, EvaluatorFn, SelectionResult, autotune_best

Shape = Tuple[int, int, int]


class FullAutotunePolicy:
    method = "A"

    def __init__(self, candidates: Sequence[MatmulConfig]) -> None:
        self.candidates = list(candidates)
        self.cache: Dict[Shape, MatmulConfig] = {}

    def select(self, shape: Shape, evaluator: EvaluatorFn, budget: BudgetTracker) -> SelectionResult:
        if shape in self.cache:
            cfg = self.cache[shape]
            M, N, K = shape
            return SelectionResult(config=cfg, cache_key=f"{M}x{N}x{K}")

        t0 = time.perf_counter()
        best_cfg, best_metrics, notes = autotune_best(shape, self.candidates, evaluator, budget)
        tune_time_ms = (time.perf_counter() - t0) * 1000.0

        self.cache[shape] = best_cfg
        M, N, K = shape
        return SelectionResult(
            config=best_cfg,
            cache_key=f"{M}x{N}x{K}",
            tune_time_ms=tune_time_ms,
            premeasure=best_metrics,
            notes=notes,
        )
