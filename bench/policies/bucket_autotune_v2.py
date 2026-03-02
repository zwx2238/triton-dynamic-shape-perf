from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig, get_default_config
from bench.policies.common import BudgetTracker, EvaluatorFn, Metrics, SelectionResult, autotune_best

Shape = Tuple[int, int, int]
BucketKeyV2 = Tuple[int, int, int, int]

_REP_SIZE_5 = {
    0: 16,
    1: 64,
    2: 256,
    3: 1024,
    4: 4096,
}


def _bin5(x: int) -> int:
    if x <= 16:
        return 0
    if x <= 64:
        return 1
    if x <= 256:
        return 2
    if x <= 1024:
        return 3
    return 4


def _ratio3(m: int, n: int) -> int:
    if m >= 2 * n:
        return 2
    if n >= 2 * m:
        return 0
    return 1


def bucket_key_v2(M: int, N: int, K: int) -> BucketKeyV2:
    return (_bin5(M), _bin5(N), _bin5(K), _ratio3(M, N))


def bucket_key_v2_to_str(key: BucketKeyV2) -> str:
    return f"{key[0]}_{key[1]}_{key[2]}_r{key[3]}"


def representative_shape_v2(key: BucketKeyV2) -> Shape:
    return (_REP_SIZE_5[key[0]], _REP_SIZE_5[key[1]], _REP_SIZE_5[key[2]])


def _runtime_us(metrics: Metrics) -> float:
    try:
        return float(metrics.get("runtime_cost_us", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_invalid(metrics: Metrics) -> bool:
    return int(metrics.get("invalid_config", 0)) == 1


class BucketAutotuneV2Policy:
    """Improved bucket autotune:
    1) finer bucket key (size bins + aspect ratio),
    2) score each config with current shape + representative shape.
    """

    method = "B2"

    def __init__(
        self,
        candidates: Sequence[MatmulConfig],
        use_anchor: bool = True,
        anchor_alpha: float = 0.3,
        anchor_top_k: int = 4,
        llm_specialization: bool = True,
        llm_candidates: Optional[Sequence[MatmulConfig]] = None,
    ) -> None:
        self.candidates = list(candidates)
        self.use_anchor = use_anchor
        self.anchor_alpha = max(0.0, min(anchor_alpha, 1.0))
        self.anchor_top_k = max(1, int(anchor_top_k))
        self.cache: Dict[BucketKeyV2, MatmulConfig] = {}
        self.llm_specialization = llm_specialization
        self.llm_candidates = list(llm_candidates) if llm_candidates is not None else list(candidates)
        self.llm_cache: Dict[Shape, MatmulConfig] = {}

    def _is_llm_style_shape(self, shape: Shape) -> bool:
        m, n, k = shape
        return m <= 64 and n >= 1024 and k >= 1024

    def _is_extreme_shape(self, shape: Shape) -> bool:
        m, n, _ = shape
        mn_min = max(1, min(m, n))
        mn_max = max(m, n)
        ratio = mn_max / mn_min
        return mn_min <= 8 or ratio >= 8.0

    def select(self, shape: Shape, evaluator: EvaluatorFn, budget: BudgetTracker) -> SelectionResult:
        if self.llm_specialization and self._is_llm_style_shape(shape):
            if shape in self.llm_cache:
                m, n, k = shape
                return SelectionResult(config=self.llm_cache[shape], cache_key=f"llm_{m}_{n}_{k}")

            t0 = time.perf_counter()
            best_cfg, best_metrics, notes = autotune_best(shape, self.llm_candidates, evaluator, budget)
            self.llm_cache[shape] = best_cfg
            tune_time_ms = (time.perf_counter() - t0) * 1000.0
            m, n, k = shape
            ext_notes = "llm_specialized"
            if notes:
                ext_notes = f"{ext_notes};{notes}"
            return SelectionResult(
                config=best_cfg,
                cache_key=f"llm_{m}_{n}_{k}",
                tune_time_ms=tune_time_ms,
                premeasure=best_metrics,
                notes=ext_notes,
            )

        key = bucket_key_v2(*shape)
        key_str = bucket_key_v2_to_str(key)
        if key in self.cache:
            return SelectionResult(config=self.cache[key], cache_key=key_str)

        anchor_shape = representative_shape_v2(key)
        note_list: list[str] = []
        t0 = time.perf_counter()
        rows: list[dict[str, object]] = []

        # Stage-1: 当前 shape 上的 primary 评分（全候选）
        for cfg in self.candidates:
            if budget.exceeded():
                note_list.append("budget_exceeded")
                break
            primary = evaluator(shape, cfg)
            if _is_invalid(primary):
                note_list.append(f"{cfg.config_id}:invalid_primary")
                continue
            primary_us = max(_runtime_us(primary), 1e-12)
            rows.append(
                {
                    "cfg": cfg,
                    "primary_metrics": primary,
                    "primary_us": primary_us,
                    "score": primary_us,
                }
            )

        rows.sort(key=lambda x: float(x["score"]))

        # Stage-2: 仅对前几名做 anchor 校正，避免 tune_time 膨胀与极端形状误导
        use_anchor_now = (
            self.use_anchor and anchor_shape != shape and (not self._is_extreme_shape(shape)) and len(rows) > 1
        )
        if use_anchor_now:
            topn = min(self.anchor_top_k, len(rows))
            for i in range(topn):
                if budget.exceeded():
                    note_list.append("budget_exceeded")
                    break
                entry = rows[i]
                cfg = entry["cfg"]
                anchor = evaluator(anchor_shape, cfg)
                if _is_invalid(anchor):
                    note_list.append(f"{cfg.config_id}:invalid_anchor")
                    continue
                anchor_us = max(_runtime_us(anchor), 1e-12)
                primary_us = float(entry["primary_us"])
                # primary 占主导，anchor 只提供轻量正则（默认 alpha=0.3）
                score = (primary_us ** (1.0 - self.anchor_alpha)) * (anchor_us ** self.anchor_alpha)
                entry["score"] = score

            rows.sort(key=lambda x: float(x["score"]))
        else:
            note_list.append("anchor_skipped")

        best_cfg: Optional[MatmulConfig] = None
        best_primary: Optional[Metrics] = None
        if rows:
            best_cfg = rows[0]["cfg"]
            best_primary = rows[0]["primary_metrics"]

        if best_cfg is None:
            best_cfg = get_default_config()
            note_list.append("fallback_default_config")

        self.cache[key] = best_cfg
        tune_time_ms = (time.perf_counter() - t0) * 1000.0
        return SelectionResult(
            config=best_cfg,
            cache_key=key_str,
            tune_time_ms=tune_time_ms,
            premeasure=best_primary,
            notes=";".join(note_list),
        )
