from __future__ import annotations

from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.stages._tune_common import run_tune_eval


def run_full(config: BenchmarkConfig, state: BenchmarkState) -> str:
    op = config.op
    eval_shapes = list(dict.fromkeys(config.eval_set))
    if not eval_shapes:
        raise RuntimeError("FULL 失败: 没有 shape 可评测")
    return run_tune_eval(
        config, state,
        method="FULL",
        tune_shapes=eval_shapes,
        group_fn=lambda s: s,
        cache_key_fn=lambda s: f"full_shape_{op.shape_to_str(s)}",
        reuse_tune_metrics=True,
    )
