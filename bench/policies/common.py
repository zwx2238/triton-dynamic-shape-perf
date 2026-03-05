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


def measure_candidates_batch(
    workload: WorkloadT,
    candidates: Sequence[CandidateT],
    batch_evaluator: BatchEvaluatorFn[WorkloadT, CandidateT],
) -> list[Metrics]:
    if not candidates:
        raise RuntimeError("measure_candidates_batch: candidates 为空")

    metrics_seq = list(batch_evaluator(workload, candidates))
    if len(metrics_seq) != len(candidates):
        raise RuntimeError(
            f"measure_candidates_batch: batch_evaluator 返回长度不匹配，"
            f"got={len(metrics_seq)} expected={len(candidates)}"
        )
    return [dict(met) for met in metrics_seq]


def _find_best_pair(
    candidates: Sequence[CandidateT],
    metrics_seq: Sequence[Metrics],
) -> Optional[tuple[CandidateT, Metrics]]:
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
        return None
    return best_cfg, best_metrics


def select_best_candidate(
    candidates: Sequence[CandidateT],
    metrics_seq: Sequence[Metrics],
) -> tuple[CandidateT, Metrics]:
    if not candidates:
        raise RuntimeError("select_best_candidate: candidates 为空")
    if len(metrics_seq) != len(candidates):
        raise RuntimeError(
            f"select_best_candidate: metrics 数量不匹配，"
            f"got={len(metrics_seq)} expected={len(candidates)}"
        )
    result = _find_best_pair(candidates, metrics_seq)
    if result is None:
        raise RuntimeError("select_best_candidate: 无有效 config（全部 invalid）")
    best_cfg, best_metrics = result
    if best_metrics is None:
        raise RuntimeError("BUG: select_best_candidate 缺少 best_metrics")
    return best_cfg, dict(best_metrics)
