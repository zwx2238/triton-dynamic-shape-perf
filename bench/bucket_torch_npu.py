"""Compatibility shim. Prefer importing from `bench.bucket_tune`."""

from bench.bucket_tune import (
    BenchmarkConfig,
    BenchmarkOptions,
    BenchmarkState,
    BucketKey,
    DistributionKey,
    Shape,
    build_benchmark_config,
    build_benchmark_state,
    derive_candidate_pool_from_typical_shapes,
    eval_triton_tune_batch,
    make_bucket_record,
    make_torch_record,
    pick_typical_shapes,
    write_prototype_report_csv,
)

__all__ = [
    "BenchmarkOptions",
    "BenchmarkConfig",
    "BenchmarkState",
    "Shape",
    "BucketKey",
    "DistributionKey",
    "build_benchmark_config",
    "build_benchmark_state",
    "eval_triton_tune_batch",
    "pick_typical_shapes",
    "derive_candidate_pool_from_typical_shapes",
    "write_prototype_report_csv",
    "make_bucket_record",
    "make_torch_record",
]
