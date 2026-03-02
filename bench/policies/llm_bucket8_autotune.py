from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.common import BudgetTracker, EvaluatorFn, SelectionResult, autotune_best

Shape = Tuple[int, int, int]
LlmBucket8Key = Tuple[int, int]


def llm_m_bin4(m: int) -> int:
    if m <= 2:
        return 0
    if m <= 8:
        return 1
    if m <= 32:
        return 2
    return 3


def llm_k_bin2(k: int) -> int:
    if k <= 2048:
        return 0
    return 1


def llm_bucket8_key(M: int, K: int) -> LlmBucket8Key:
    return (llm_m_bin4(M), llm_k_bin2(K))


def llm_bucket8_key_to_str(key: LlmBucket8Key) -> str:
    return f"llm8_m{key[0]}_k{key[1]}"


class LlmBucket8AutotunePolicy:
    method = "B8"

    def __init__(self, candidates: Sequence[MatmulConfig]) -> None:
        self.candidates = list(candidates)
        self.cache: Dict[LlmBucket8Key, MatmulConfig] = {}

    def select(self, shape: Shape, evaluator: EvaluatorFn, budget: BudgetTracker) -> SelectionResult:
        M, _, K = shape
        key = llm_bucket8_key(M, K)
        key_str = llm_bucket8_key_to_str(key)
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
