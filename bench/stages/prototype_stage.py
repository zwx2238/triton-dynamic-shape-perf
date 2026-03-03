from __future__ import annotations

from bench.bucket_tune.runtime import eval_triton_tune_batch
from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState


def run_prototype(config: BenchmarkConfig, state: BenchmarkState) -> str:
    proto_count = min(int(config.options.prototype_count), len(config.tune_set))
    op = config.op
    typical_shapes = op.pick_typical_shapes(config.tune_set, proto_count)
    proto_candidates, proto_rows = op.derive_candidate_pool_from_typical_shapes(
        typical_shapes,
        state.candidates,
        lambda shape, cfgs: eval_triton_tune_batch(config, shape, cfgs),
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
