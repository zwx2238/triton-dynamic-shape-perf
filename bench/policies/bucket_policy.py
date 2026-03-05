from __future__ import annotations

import time
from typing import Callable, Dict, Generic, Hashable, Optional, Sequence, TypeVar

from bench.policies.common import (
    BatchEvaluatorFn,
    SelectionResult,
    measure_candidates_batch,
    select_best_candidate,
)

WorkloadT = TypeVar("WorkloadT")
CandidateT = TypeVar("CandidateT")
BucketKeyT = TypeVar("BucketKeyT", bound=Hashable)


class BucketTunePolicy(Generic[WorkloadT, CandidateT, BucketKeyT]):
    method = "BUCKET"

    def __init__(
        self,
        candidates: Sequence[CandidateT],
        key_fn: Callable[[WorkloadT], BucketKeyT],
        key_to_str: Optional[Callable[[BucketKeyT], str]] = None,
    ) -> None:
        self.candidates = list(candidates)
        self.cache: Dict[BucketKeyT, SelectionResult[CandidateT]] = {}
        self._key_fn = key_fn
        self._key_to_str = key_to_str if key_to_str is not None else lambda key: str(key)

    def _return_cached(self, cached: SelectionResult[CandidateT], key_str: str) -> SelectionResult[CandidateT]:
        return SelectionResult(
            config=cached.config,
            cache_key=key_str,
            premeasure=dict(cached.premeasure) if cached.premeasure is not None else None,
            notes=cached.notes,
        )

    def _run_batch_measure(
        self,
        workload: WorkloadT,
        batch_evaluator: BatchEvaluatorFn[WorkloadT, CandidateT],
    ) -> tuple[CandidateT, dict, float]:
        t0 = time.perf_counter()
        metrics_seq = measure_candidates_batch(workload, self.candidates, batch_evaluator)
        best_cfg, best_metrics = select_best_candidate(self.candidates, metrics_seq)
        tune_time_ms = (time.perf_counter() - t0) * 1000.0
        return best_cfg, dict(best_metrics), tune_time_ms

    def _cache_and_return(
        self,
        key: BucketKeyT,
        key_str: str,
        best_cfg: CandidateT,
        premeasure: dict,
        tune_time_ms: float,
    ) -> SelectionResult[CandidateT]:
        notes = "batch_measure_then_select"
        cached = SelectionResult(config=best_cfg, cache_key=key_str, premeasure=premeasure, notes=notes)
        self.cache[key] = cached
        return SelectionResult(
            config=best_cfg,
            cache_key=key_str,
            tune_time_ms=tune_time_ms,
            premeasure=premeasure,
            notes=notes,
        )

    def _measure_and_store(
        self,
        workload: WorkloadT,
        batch_evaluator: BatchEvaluatorFn[WorkloadT, CandidateT],
        key: BucketKeyT,
        key_str: str,
    ) -> SelectionResult[CandidateT]:
        best_cfg, premeasure, tune_time_ms = self._run_batch_measure(workload, batch_evaluator)
        return self._cache_and_return(key, key_str, best_cfg, premeasure, tune_time_ms)

    def select(
        self,
        workload: WorkloadT,
        batch_evaluator: BatchEvaluatorFn[WorkloadT, CandidateT],
    ) -> SelectionResult[CandidateT]:
        key = self._key_fn(workload)
        key_str = self._key_to_str(key)
        cached = self.cache.get(key)
        if cached is not None:
            return self._return_cached(cached, key_str)
        return self._measure_and_store(workload, batch_evaluator, key, key_str)
