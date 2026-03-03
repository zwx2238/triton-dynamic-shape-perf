from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, TypeVar

WorkloadT = TypeVar("WorkloadT")
CandidateT = TypeVar("CandidateT")
Metrics = Dict[str, object]
BatchEvaluatorFn = Callable[[WorkloadT, Sequence[CandidateT]], Sequence[Metrics]]
INVALID_SCORE = 1e30


@dataclass
class SelectionResult(Generic[CandidateT]):
    config: CandidateT
    cache_key: str
    tune_time_ms: float = 0.0
    premeasure: Optional[Metrics] = None
    notes: str = ""


def _metrics_score(metrics: Metrics) -> float:
    invalid = int(metrics.get("invalid_config", 0))
    runtime_us = float(metrics.get("runtime_cost_us", 0.0))
    if invalid == 0 and runtime_us > 0.0:
        return runtime_us
    return INVALID_SCORE


def autotune_best(
    workload: WorkloadT,
    candidates: Sequence[CandidateT],
    batch_evaluator: BatchEvaluatorFn[WorkloadT, CandidateT],
) -> tuple[CandidateT, Metrics, str]:
    if not candidates:
        raise RuntimeError("autotune_best: candidates 为空")

    metrics_seq = list(batch_evaluator(workload, candidates))
    if len(metrics_seq) != len(candidates):
        raise RuntimeError(
            f"autotune_best: batch_evaluator 返回长度不匹配，"
            f"got={len(metrics_seq)} expected={len(candidates)}"
        )

    best_cfg: Optional[CandidateT] = None
    best_metrics: Optional[Metrics] = None
    best_score = INVALID_SCORE

    for cfg, metrics in zip(candidates, metrics_seq):
        score = _metrics_score(metrics)
        if score < best_score:
            best_cfg = cfg
            best_metrics = metrics
            best_score = score

    if best_cfg is None or best_score >= INVALID_SCORE:
        raise RuntimeError("autotune_best: 无有效 config（全部 invalid）")
    if best_metrics is None:
        raise RuntimeError("BUG: autotune_best 缺少 best_metrics")
    return best_cfg, best_metrics, "batch_tune"
