from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.policies.common import BudgetTracker, EvaluatorFn

Shape = Tuple[int, int, int]


def _geometric_mean(values: List[float]) -> float:
    positives = [max(v, 1e-12) for v in values if v > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def search_top_configs_for_workload(
    workload: str,
    probe_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    evaluator: EvaluatorFn,
    budget: BudgetTracker,
    top_k: int = 6,
) -> Dict[str, object]:
    scores: Dict[str, float] = {}
    notes: List[str] = []

    for cfg in candidates:
        if budget.exceeded():
            notes.append("budget_exceeded_during_probe")
            break

        speeds: List[float] = []
        for shape in probe_shapes:
            if budget.exceeded():
                notes.append("budget_exceeded_during_probe")
                break

            metrics = evaluator(shape, cfg)
            if int(metrics.get("invalid_config", 0)) == 1:
                continue
            perf = float(metrics.get("runtime_perf_tflops", 0.0))
            if perf > 0:
                speeds.append(perf)

        scores[cfg.config_id] = _geometric_mean(speeds)

    ranked = sorted(candidates, key=lambda c: scores.get(c.config_id, 0.0), reverse=True)
    selected_ids = [c.config_id for c in ranked if scores.get(c.config_id, 0.0) > 0][:top_k]

    if len(selected_ids) < top_k:
        for cfg in ranked:
            if cfg.config_id in selected_ids:
                continue
            selected_ids.append(cfg.config_id)
            if len(selected_ids) >= top_k:
                break

    return {
        "workload": workload,
        "scores": scores,
        "selected_ids": selected_ids,
        "notes": ";".join(notes),
    }
