from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.buckets import bucket_key, bucket_key_to_str
from bench.policies.common import BudgetTracker, EvaluatorFn, SelectionResult, autotune_best

Shape = Tuple[int, int, int]
BucketKey = Tuple[int, int, int]


class BucketAutotunePolicy:
    def __init__(self, candidates: Sequence[MatmulConfig], method: str = "B") -> None:
        self.candidates = list(candidates)
        self.method = method
        self.cache: Dict[BucketKey, MatmulConfig] = {}

    def select(self, shape: Shape, evaluator: EvaluatorFn, budget: BudgetTracker) -> SelectionResult:
        key = bucket_key(*shape)
        key_str = bucket_key_to_str(key)
        if key in self.cache:
            return SelectionResult(config=self.cache[key], cache_key=key_str)

        t0 = time.perf_counter()
        best_cfg, best_metrics, notes = autotune_best(shape, self.candidates, evaluator, budget)
        tune_time_ms = (time.perf_counter() - t0) * 1000.0
        self.cache[key] = best_cfg
        return SelectionResult(
            config=best_cfg,
            cache_key=key_str,
            tune_time_ms=tune_time_ms,
            premeasure=best_metrics,
            notes=notes,
        )
