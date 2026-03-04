from __future__ import annotations

import inspect

from bench.bucket_tune.runtime import eval_triton_tune_batch
from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState


def run_prototype(config: BenchmarkConfig, state: BenchmarkState) -> str:
    proto_count = min(int(config.options.prototype_count), len(config.tune_set))
    op = config.op
    typical_shapes = op.pick_typical_shapes(config.tune_set, proto_count)
    eval_batch = lambda shape, cfgs: eval_triton_tune_batch(config, shape, cfgs)
    eval_batch_all = lambda shapes, cfgs: config.triton_evaluator.evaluate_batch(
        [(shape, cfg) for shape in shapes for cfg in cfgs]
    )

    derive_fn = op.derive_candidate_pool_from_typical_shapes
    try:
        derive_sig = inspect.signature(derive_fn)
        if "eval_batch_all" in derive_sig.parameters:
            proto_candidates, proto_rows = derive_fn(
                typical_shapes,
                state.candidates,
                eval_batch,
                eval_batch_all=eval_batch_all,
            )
        else:
            proto_candidates, proto_rows = derive_fn(
                typical_shapes,
                state.candidates,
                eval_batch,
            )
    except (TypeError, ValueError):
        proto_candidates, proto_rows = derive_fn(
            typical_shapes,
            state.candidates,
            eval_batch,
        )
    if proto_candidates:
        state.candidates = proto_candidates
    op.write_prototype_report_csv(config.prototype_report_csv, proto_rows)

    cfg_desc = ",".join(getattr(c, "config_id", "?") for c in state.candidates)
    shape_desc = ",".join(op.shape_to_str(s) for s in typical_shapes)
    return (
        f"prototype_shapes={len(typical_shapes)}[{shape_desc}] "
        f"candidates={len(state.candidates)}[{cfg_desc}] "
        f"prototype_report_csv={config.prototype_report_csv}"
    )
