from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.common import BatchEvaluatorFn, BudgetTracker, EvaluatorFn, SelectionResult, autotune_best

Shape = Tuple[int, int, int]
LlmBucket8Key = int
DEFAULT_M_SPLIT = 16
DEFAULT_N_SPLIT = 4096
DEFAULT_K_SPLIT = 6144

def llm_bucket8_key(
    M: int,
    N: int,
    K: int,
    m_split: int = DEFAULT_M_SPLIT,
    n_split: int = DEFAULT_N_SPLIT,
    k_split: int = DEFAULT_K_SPLIT,
) -> LlmBucket8Key:
    """8-bucket heuristic with 3 binary splits on M/N/K.

    bit2: M <= m_split
    bit1: N <= n_split
    bit0: K <= k_split
    """
    b2 = 1 if M <= m_split else 0
    b1 = 1 if N <= n_split else 0
    b0 = 1 if K <= k_split else 0
    return (b2 << 2) | (b1 << 1) | b0


def llm_bucket8_key_to_str(key: LlmBucket8Key) -> str:
    return f"llm8_g{key}"


class LlmBucket8AutotunePolicy:
    method = "B8"

    def __init__(
        self,
        candidates: Sequence[MatmulConfig],
        m_split: int = DEFAULT_M_SPLIT,
        n_split: int = DEFAULT_N_SPLIT,
        k_split: int = DEFAULT_K_SPLIT,
    ) -> None:
        self.candidates = list(candidates)
        self.cache: Dict[LlmBucket8Key, MatmulConfig] = {}
        self.m_split = int(m_split)
        self.n_split = int(n_split)
        self.k_split = int(k_split)

    def select(
        self,
        shape: Shape,
        evaluator: EvaluatorFn,
        budget: BudgetTracker,
        batch_evaluator: Optional[BatchEvaluatorFn] = None,
    ) -> SelectionResult:
        M, N, K = shape
        key = llm_bucket8_key(M, N, K, self.m_split, self.n_split, self.k_split)
        key_str = llm_bucket8_key_to_str(key)
        if key in self.cache:
            return SelectionResult(config=self.cache[key], cache_key=key_str)

        t0 = time.perf_counter()
        best_cfg, best_metrics, notes = autotune_best(
            shape,
            self.candidates,
            evaluator,
            budget,
            batch_evaluator=batch_evaluator,
        )
        tune_time_ms = (time.perf_counter() - t0) * 1000.0
        self.cache[key] = best_cfg
        return SelectionResult(
            config=best_cfg,
            cache_key=key_str,
            tune_time_ms=tune_time_ms,
            premeasure=best_metrics,
            notes=notes,
        )
