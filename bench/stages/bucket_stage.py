from __future__ import annotations

from typing import Dict

from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.stages._tune_common import run_tune_eval


def _build_first_shape_per_key(tune_set, op, splits) -> Dict[object, tuple]:
    first: Dict[object, tuple] = {}
    for shape in tune_set:
        key = op.eval_key(shape, splits)
        if key not in first:
            first[key] = shape
    return first


def run_bucket(config: BenchmarkConfig, state: BenchmarkState) -> str:
    op, splits = config.op, config.options.bucket_splits
    first_shape = _build_first_shape_per_key(config.tune_set, op, splits)
    if not first_shape:
        raise RuntimeError("BUCKET 失败: tune_set 为空")
    return run_tune_eval(
        config, state,
        method="BUCKET",
        tune_shapes=list(first_shape.values()),
        group_fn=lambda s: op.eval_key(s, splits),
        cache_key_fn=op.eval_key_str,
    )
