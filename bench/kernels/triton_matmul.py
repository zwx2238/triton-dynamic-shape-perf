from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from bench.configs.base_configs import MatmulConfig

Shape = Tuple[int, int, int]


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


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


class TritonMatmulEvaluator:
    def __init__(
        self,
        dtype: str = "bf16",
        device: str = "cuda",
        warmup: int = 10,
        repeat: int = 50,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法执行 Triton matmul benchmark")

        self.dtype_name = dtype
        self.dtype = self._parse_dtype(dtype)
        self.device = torch.device(device)
        self.warmup = int(warmup)
        self.repeat = int(repeat)

        self.compile_cache: set[tuple[str, str, str]] = set()
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

    def get_gpu_name(self) -> str:
        idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(idx)

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

    def _launch(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, cfg: MatmulConfig) -> None:
        M, K = a.shape
        _, N = b.shape

        grid = (
            triton.cdiv(M, cfg.BLOCK_M) * triton.cdiv(N, cfg.BLOCK_N),
        )

        _matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=cfg.BLOCK_M,
            BLOCK_N=cfg.BLOCK_N,
            BLOCK_K=cfg.BLOCK_K,
            GROUP_M=cfg.GROUP_M,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    def evaluate(self, shape: Shape, cfg: MatmulConfig) -> Dict[str, object]:
        M, N, K = shape
        compile_time_ms = 0.0
        compile_key = (
            str(self.device),
            self.dtype_name,
            cfg.config_id,
        )

        try:
            a, b, c = self._get_tensors(shape)

            if compile_key not in self.compile_cache:
                t0 = time.perf_counter()
                self._launch(a, b, c, cfg)
                torch.cuda.synchronize(self.device)
                compile_time_ms = (time.perf_counter() - t0) * 1000.0
                self.compile_cache.add(compile_key)

            for _ in range(self.warmup):
                self._launch(a, b, c, cfg)
            torch.cuda.synchronize(self.device)

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            latencies_us: list[float] = []
            for _ in range(self.repeat):
                start_ev.record()
                self._launch(a, b, c, cfg)
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
                notes="",
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
