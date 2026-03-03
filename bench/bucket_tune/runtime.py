from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

from bench.configs.base_configs import BASE_CONFIGS, MatmulConfig
from bench.kernels.torch_matmul import TorchMatmulEvaluator
from bench.kernels.triton_matmul import TritonMatmulEvaluator
from bench.policies.bucket_autotune import bucket_key
from bench.reporting.csv_logger import reset_csv

from .shape_sets import build_eval_set, build_tune_set
from .types import BenchmarkConfig, BenchmarkOptions, BenchmarkState, Shape


def build_benchmark_config(options: BenchmarkOptions) -> BenchmarkConfig:
    if options.prototype_count <= 0:
        raise ValueError(f"prototype-count 必须 > 0，当前: {options.prototype_count}")

    rng_tune = random.Random(options.seed + 1)
    rng_eval = random.Random(options.seed + 2)
    tune_set = build_tune_set(
        rng_tune,
        options.tune_size,
        options.bucket_m_split,
        options.bucket_n_split,
        options.bucket_k_split,
    )
    eval_set = build_eval_set(rng_eval, options.eval_size)

    eval_keys = {
        bucket_key(s[0], s[1], s[2], options.bucket_m_split, options.bucket_n_split, options.bucket_k_split)
        for s in eval_set
    }
    tune_keys = {
        bucket_key(s[0], s[1], s[2], options.bucket_m_split, options.bucket_n_split, options.bucket_k_split)
        for s in tune_set
    }
    if not eval_keys.issubset(tune_keys):
        raise RuntimeError("BUG: tune key 未覆盖 eval key")

    if options.reset_results:
        reset_csv(options.results_csv)
    Path(options.results_csv).parent.mkdir(parents=True, exist_ok=True)

    triton_evaluator = TritonMatmulEvaluator(
        dtype=options.dtype,
        device="npu",
        warmup=options.warmup,
        repeat=options.repeat,
    )
    torch_evaluator = TorchMatmulEvaluator(
        dtype=options.dtype,
        device="npu",
        warmup=options.warmup,
        repeat=options.repeat,
    )
    gpu = triton_evaluator.get_gpu_name()
    if triton_evaluator.device.type != "npu":
        raise RuntimeError(f"瘦身版仅支持 npu，当前 device_type={triton_evaluator.device.type}")

    if options.prototype_report_csv:
        prototype_report_csv = options.prototype_report_csv
    else:
        p = Path(options.results_csv)
        prototype_report_csv = str(p.with_name(f"{p.stem}_prototype_best.csv"))

    return BenchmarkConfig(
        options=options,
        tune_set=tune_set,
        eval_set=eval_set,
        eval_keys=frozenset(eval_keys),
        triton_evaluator=triton_evaluator,
        torch_evaluator=torch_evaluator,
        gpu=gpu,
        prototype_report_csv=prototype_report_csv,
    )


def build_benchmark_state() -> BenchmarkState:
    return BenchmarkState(candidates=list(BASE_CONFIGS))


def eval_triton_tune_batch(
    config: BenchmarkConfig,
    shape: Shape,
    cfgs: Sequence[MatmulConfig],
) -> List[Dict[str, object]]:
    entries = [(shape, cfg) for cfg in cfgs]
    return list(config.triton_evaluator.evaluate_batch(entries))
