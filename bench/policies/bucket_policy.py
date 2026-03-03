from __future__ import annotations

import time
from typing import Callable, Dict, Generic, Hashable, Optional, Sequence, TypeVar

from bench.policies.common import BatchEvaluatorFn, SelectionResult, autotune_best

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
        self.cache: Dict[BucketKeyT, CandidateT] = {}
        self._key_fn = key_fn
        self._key_to_str = key_to_str if key_to_str is not None else lambda key: str(key)

    def select(
        self,
        workload: WorkloadT,
        batch_evaluator: BatchEvaluatorFn[WorkloadT, CandidateT],
    ) -> SelectionResult[CandidateT]:
        key = self._key_fn(workload)
        key_str = self._key_to_str(key)
        if key in self.cache:
            return SelectionResult(config=self.cache[key], cache_key=key_str)

        t0 = time.perf_counter()
        best_cfg, best_metrics, notes = autotune_best(
            workload,
            self.candidates,
            batch_evaluator,
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
