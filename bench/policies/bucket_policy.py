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

    def select(
        self,
        workload: WorkloadT,
        batch_evaluator: BatchEvaluatorFn[WorkloadT, CandidateT],
    ) -> SelectionResult[CandidateT]:
        key = self._key_fn(workload)
        key_str = self._key_to_str(key)
        cached = self.cache.get(key)
        if cached is not None:
            return SelectionResult(
                config=cached.config,
                cache_key=key_str,
                premeasure=dict(cached.premeasure) if cached.premeasure is not None else None,
                notes=cached.notes,
            )

        t0 = time.perf_counter()
        metrics_seq = measure_candidates_batch(
            workload,
            self.candidates,
            batch_evaluator,
        )
        best_cfg, best_metrics = select_best_candidate(self.candidates, metrics_seq)
        tune_time_ms = (time.perf_counter() - t0) * 1000.0
        premeasure = dict(best_metrics)
        notes = "batch_measure_then_select"
        self.cache[key] = SelectionResult(
            config=best_cfg,
            cache_key=key_str,
            premeasure=premeasure,
            notes=notes,
        )
        return SelectionResult(
            config=best_cfg,
            cache_key=key_str,
            tune_time_ms=tune_time_ms,
            premeasure=premeasure,
            notes=notes,
        )
