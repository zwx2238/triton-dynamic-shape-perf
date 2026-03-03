from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

Shape = Tuple[int, int, int]


@dataclass
class EvalMetrics:
    compile_time_ms: float
    runtime_cost_us: float
    p99_us: float
    runtime_perf_tflops: float
    invalid_config: int
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "compile_time_ms": self.compile_time_ms,
            "runtime_cost_us": self.runtime_cost_us,
            "p99_us": self.p99_us,
            "runtime_perf_tflops": self.runtime_perf_tflops,
            "invalid_config": self.invalid_config,
            "notes": self.notes,
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


class TorchMatmulEvaluator:
    """cuBLAS baseline via torch.mm."""

    def __init__(
        self,
        dtype: str = "bf16",
        device: str = "cuda",
        warmup: int = 10,
        repeat: int = 50,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法执行 torch matmul benchmark")

        self.dtype_name = dtype
        self.dtype = self._parse_dtype(dtype)
        self.device = torch.device(device)
        self.warmup = int(warmup)
        self.repeat = int(repeat)

        self._first_launch_done = False
        self._cached_shape: Optional[Shape] = None
        self._cached_tensors: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def _parse_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"不支持 dtype: {dtype}")
        return mapping[dtype]

    def _shape_seed(self, shape: Shape) -> int:
        m, n, k = shape
        return ((m * 73856093) ^ (n * 19349663) ^ (k * 83492791)) & 0xFFFFFFFF

    def _get_tensors(self, shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cached_shape == shape and self._cached_tensors is not None:
            return self._cached_tensors

        self._cached_shape = None
        self._cached_tensors = None

        M, N, K = shape
        g = torch.Generator(device="cpu")
        g.manual_seed(self._shape_seed(shape))

        a = torch.randn((M, K), dtype=torch.float32, generator=g).to(device=self.device, dtype=self.dtype)
        b = torch.randn((K, N), dtype=torch.float32, generator=g).to(device=self.device, dtype=self.dtype)
        c = torch.empty((M, N), device=self.device, dtype=self.dtype)

        self._cached_shape = shape
        self._cached_tensors = (a, b, c)
        return a, b, c

    def _launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        torch.mm(a, b, out=c)

    def evaluate(self, shape: Shape) -> Dict[str, object]:
        M, N, K = shape
        compile_time_ms = 0.0

        try:
            a, b, c = self._get_tensors(shape)

            if not self._first_launch_done:
                t0 = time.perf_counter()
                self._launch(a, b, c)
                torch.cuda.synchronize(self.device)
                compile_time_ms = (time.perf_counter() - t0) * 1000.0
                self._first_launch_done = True

            for _ in range(self.warmup):
                self._launch(a, b, c)
            torch.cuda.synchronize(self.device)

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            latencies_us: list[float] = []
            for _ in range(self.repeat):
                start_ev.record()
                self._launch(a, b, c)
                end_ev.record()
                torch.cuda.synchronize(self.device)
                latencies_us.append(start_ev.elapsed_time(end_ev) * 1000.0)

            p50_us = _percentile(latencies_us, 50)
            p99_us = _percentile(latencies_us, 99)
            tflops = 0.0
            if p50_us > 0:
                tflops = (2.0 * M * N * K) / (p50_us / 1_000_000.0) / 1e12

            return EvalMetrics(
                compile_time_ms=compile_time_ms,
                runtime_cost_us=p50_us,
                p99_us=p99_us,
                runtime_perf_tflops=tflops,
                invalid_config=0,
                notes="torch_mm_cublas",
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            return EvalMetrics(
                compile_time_ms=compile_time_ms,
                runtime_cost_us=0.0,
                p99_us=0.0,
                runtime_perf_tflops=0.0,
                invalid_config=1,
                notes=f"{type(exc).__name__}: {exc}",
            ).to_dict()

