from __future__ import annotations

from typing import Dict, List

from bench.bucket_tune.records import make_torch_record
from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.reporting.csv_logger import append_records


def run_torch(config: BenchmarkConfig, state: BenchmarkState) -> str:
    rows: List[Dict[str, object]] = []
    torch_cfg = config.torch_evaluator.get_torch_config_id()
    eval_shapes = list(config.eval_set)
    if eval_shapes:
        eval_metrics = config.torch_evaluator.evaluate_batch(eval_shapes)
        for idx, (shape, met) in enumerate(zip(eval_shapes, eval_metrics)):
            rows.append(make_torch_record(config, "eval", idx, shape, met, torch_cfg))

    append_records(config.options.results_csv, rows)
    return f"method=TORCH eval_rows={len(rows)}"
