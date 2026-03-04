from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass(frozen=True)
class BenchmarkOptions:
    prototype_count: int = 0
    prototype_report_csv: str = ""
    tune_size: int = 64
    eval_size: int = 256
    seed: int = 20260302
    dtype: str = "bf16"
    warmup: int = 5
    repeat: int = 10
    bucket_splits: Tuple[int, ...] = (16, 4096, 6144)
    op_name: str = "matmul"
    results_csv: str = "results/bucket_torch.csv"
    reset_results: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    options: BenchmarkOptions
    tune_set: tuple
    eval_set: tuple
    eval_keys: frozenset
    triton_evaluator: Any
    torch_evaluator: Any
    gpu: str
    op: Any
    prototype_report_csv: str


@dataclass
class BenchmarkState:
    candidates: List[Any]
