from __future__ import annotations

import bisect
import time
from typing import Dict, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.common import BudgetTracker, EvaluatorFn, SelectionResult, autotune_best

Shape = Tuple[int, int, int]
LlmBucket8Key = int

M_VALUES = [1, 2, 4, 8, 16, 32, 64]
K_VALUES = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
M_TO_INDEX = {v: i for i, v in enumerate(M_VALUES)}
K_TO_INDEX = {v: i for i, v in enumerate(K_VALUES)}

# 固定的 49 -> 8 桶映射
# 行: M_VALUES 顺序, 列: K_VALUES 顺序
LLM_BUCKET8_TABLE = [
    [1, 1, 2, 2, 2, 2, 0],
    [6, 6, 2, 2, 2, 2, 2],
    [6, 6, 2, 2, 2, 2, 2],
    [6, 6, 2, 2, 2, 2, 2],
    [6, 7, 2, 2, 2, 2, 2],
    [6, 6, 2, 2, 2, 2, 2],
    [6, 4, 3, 3, 3, 5, 7],
]


def _value_index(values: Sequence[int], lookup: Dict[int, int], x: int) -> int:
    idx = lookup.get(x)
    if idx is not None:
        return idx
    # 对分布外值，使用 <=x 的最近离散点（最小值以下则取 0）
    pos = bisect.bisect_right(values, x) - 1
    if pos < 0:
        return 0
    if pos >= len(values):
        return len(values) - 1
    return pos


def llm_bucket8_key(M: int, K: int) -> LlmBucket8Key:
    mi = _value_index(M_VALUES, M_TO_INDEX, M)
    ki = _value_index(K_VALUES, K_TO_INDEX, K)
    return int(LLM_BUCKET8_TABLE[mi][ki])


def llm_bucket8_key_to_str(key: LlmBucket8Key) -> str:
    return f"llm8_g{key}"


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
