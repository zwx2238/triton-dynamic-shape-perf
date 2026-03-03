from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from bench.configs.base_configs import MatmulConfig
from bench.kernels.torch_matmul import TorchMatmulEvaluator
from bench.kernels.triton_matmul import TritonMatmulEvaluator
from bench.policies.bucket_autotune import DEFAULT_K_SPLIT, DEFAULT_M_SPLIT, DEFAULT_N_SPLIT

Shape = Tuple[int, int, int]
BucketKey = int
DistributionKey = Tuple[int, int]


@dataclass(frozen=True)
class BenchmarkOptions:
    prototype_count: int = 4
    prototype_report_csv: str = ""
    tune_size: int = 64
    eval_size: int = 256
    seed: int = 20260302
    dtype: str = "bf16"
    warmup: int = 5
    repeat: int = 10
    bucket_m_split: int = DEFAULT_M_SPLIT
    bucket_n_split: int = DEFAULT_N_SPLIT
    bucket_k_split: int = DEFAULT_K_SPLIT
    results_csv: str = "results/bucket_torch.csv"
    reset_results: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    options: BenchmarkOptions
    tune_set: tuple[Shape, ...]
    eval_set: tuple[Shape, ...]
    eval_keys: frozenset[BucketKey]
    triton_evaluator: TritonMatmulEvaluator
    torch_evaluator: TorchMatmulEvaluator
    gpu: str
    prototype_report_csv: str


@dataclass
class BenchmarkState:
    candidates: List[MatmulConfig]
