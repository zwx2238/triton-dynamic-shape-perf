from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from .configs import MatmulConfig
from bench.kernels.npu_measure import (
    align_ascend_toolchain_env,
    get_device_name,
    measure_latencies_us,
    profile_npu_step_launches_us,
    resolve_device,
    synchronize,
)

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
    SWIZZLE_GROUP: tl.constexpr,
    SWIZZLE_DIRECTION: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_pid_m * num_pid_n

    for block_idx in range(pid, num_blocks, NUM_CORES):
        block_m = block_idx // num_pid_n
        block_n = block_idx % num_pid_n

        if SWIZZLE_DIRECTION == 0:
            pid_m, pid_n = tl.swizzle2d(block_m, block_n, num_pid_m, num_pid_n, SWIZZLE_GROUP)
        else:
            size_gj = SWIZZLE_GROUP * num_pid_m
            group_id = block_idx // size_gj
            off_n = group_id * SWIZZLE_GROUP
            cur_size_g = tl.minimum(num_pid_n - off_n, SWIZZLE_GROUP)
            local_ij = block_idx % size_gj
            pid_m = local_ij // cur_size_g
            pid_n = off_n + (local_ij % cur_size_g)

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
    invalid_config: int
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "compile_time_ms": self.compile_time_ms,
            "runtime_cost_us": self.runtime_cost_us,
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
        device: str = "npu",
        warmup: int = 10,
        repeat: int = 50,
    ) -> None:
        self.dtype_name = dtype
        self.dtype = self._parse_dtype(dtype)
        self.device = resolve_device(device)
        self.warmup = int(warmup)
        self.repeat = int(repeat)
        self.env_notes = ""
        default_cores = 20
        device_name_hint = str(get_device_name(self.device))
        if "910B2" in device_name_hint:
            default_cores = 24
        self.ascend_num_cores = int(os.environ.get("TRITON_ASCEND_NUM_CORES", str(default_cores)))
        self.ascend_swizzle_group = 4

        self.env_notes = align_ascend_toolchain_env()

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
        return get_device_name(self.device)

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

        total_blocks = triton.cdiv(M, cfg.BLOCK_M) * triton.cdiv(N, cfg.BLOCK_N)
        launch_programs = max(1, min(self.ascend_num_cores, total_blocks))
        swizzle_group = self.ascend_swizzle_group
        swizzle_direction = 1 if M < N else 0

        grid = (launch_programs,)

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
            SWIZZLE_GROUP=swizzle_group,
            SWIZZLE_DIRECTION=swizzle_direction,
            NUM_CORES=launch_programs,
        )

    def _kernel_launch_note(self, shape: Shape, cfg: MatmulConfig) -> str:
        M, N, _ = shape
        total_blocks = triton.cdiv(M, cfg.BLOCK_M) * triton.cdiv(N, cfg.BLOCK_N)
        launch_programs = max(1, min(self.ascend_num_cores, total_blocks))
        swizzle_group = self.ascend_swizzle_group
        swizzle_direction = 1 if M < N else 0
        return f"launch=fixed_cores({launch_programs}),swizzle2d_g={swizzle_group},dir={swizzle_direction}"

    def evaluate(self, shape: Shape, cfg: MatmulConfig) -> Dict[str, object]:
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
                synchronize(self.device)
                compile_time_ms = (time.perf_counter() - t0) * 1000.0
                self.compile_cache.add(compile_key)

            for _ in range(self.warmup):
                self._launch(a, b, c, cfg)
            synchronize(self.device)

            latencies_us, timing_note = measure_latencies_us(
                self.device,
                self.repeat,
                lambda: self._launch(a, b, c, cfg),
            )

            p50_us = _percentile(latencies_us, 50)

            note_parts = []
            if self.env_notes:
                note_parts.append(self.env_notes)
            note_parts.append(self._kernel_launch_note(shape, cfg))
            note_parts.append(f"timing={timing_note}")
            return EvalMetrics(
                compile_time_ms=compile_time_ms,
                runtime_cost_us=p50_us,
                invalid_config=0,
                notes=";".join(note_parts),
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            note = f"{type(exc).__name__}: {exc}"
            if self.env_notes:
                note = f"{self.env_notes};{note}"
            return EvalMetrics(
                compile_time_ms=compile_time_ms,
                runtime_cost_us=0.0,
                invalid_config=1,
                notes=note,
            ).to_dict()

    def evaluate_batch(
        self,
        entries: Sequence[tuple[Shape, MatmulConfig]],
    ) -> list[Dict[str, object]]:
        if not entries:
            return []

        try:
            ordered_shapes = list(dict.fromkeys(shape for shape, _ in entries))
            tensor_map: Dict[Shape, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for shape in ordered_shapes:
                tensor_map[shape] = self._get_tensors(shape)

            compile_time_by_index = [0.0 for _ in entries]
            seen_compile: set[tuple[str, str, str]] = set()
            for idx, (shape, cfg) in enumerate(entries):
                compile_key = (str(self.device), self.dtype_name, cfg.config_id)
                if compile_key in self.compile_cache or compile_key in seen_compile:
                    continue
                a, b, c = tensor_map[shape]
                t0 = time.perf_counter()
                self._launch(a, b, c, cfg)
                synchronize(self.device)
                compile_ms = (time.perf_counter() - t0) * 1000.0
                compile_time_by_index[idx] = compile_ms
                seen_compile.add(compile_key)
                self.compile_cache.add(compile_key)

            unique_entries = list(dict.fromkeys((shape, cfg.config_id) for shape, cfg in entries))
            cfg_map = {cfg.config_id: cfg for _, cfg in entries}
            for shape, cfg_id in unique_entries:
                a, b, c = tensor_map[shape]
                cfg = cfg_map[cfg_id]
                for _ in range(self.warmup):
                    self._launch(a, b, c, cfg)
                synchronize(self.device)

            step_launches = []
            for shape, cfg in entries:
                a, b, c = tensor_map[shape]
                for _ in range(self.repeat):
                    step_launches.append(lambda a=a, b=b, c=c, cfg=cfg: self._launch(a, b, c, cfg))

            step_latencies, timing_note = profile_npu_step_launches_us(self.device, step_launches)
            if len(step_latencies) < len(entries) * self.repeat:
                raise RuntimeError(
                    "triton batch profiler steps too short: "
                    f"expected={len(entries) * self.repeat}, actual={len(step_latencies)}, "
                    f"entries={len(entries)}, repeat={self.repeat}, timing_note={timing_note}"
                )

            out: list[Dict[str, object]] = []
            for idx, (shape, _) in enumerate(entries):
                start = idx * self.repeat
                end = start + self.repeat
                latencies = step_latencies[start:end]

                p50_us = _percentile(latencies, 50)

                note_parts = []
                if self.env_notes:
                    note_parts.append(self.env_notes)
                note_parts.append(self._kernel_launch_note(shape, entries[idx][1]))
                note_parts.append(f"timing={timing_note}")

                out.append(
                    EvalMetrics(
                        compile_time_ms=compile_time_by_index[idx],
                        runtime_cost_us=p50_us,
                        invalid_config=0,
                        notes=";".join(note_parts),
                    ).to_dict()
                )
            return out
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "triton evaluate_batch failed: "
                f"entries={len(entries)}, repeat={self.repeat}, warmup={self.warmup}, "
                f"error={type(exc).__name__}: {exc}"
            ) from exc
