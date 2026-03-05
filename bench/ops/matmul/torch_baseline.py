from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from bench.kernels.npu_measure import (
    get_device_name,
    profile_npu_step_launches_us,
    resolve_device,
    synchronize,
)

Shape = Tuple[int, int, int]


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


def _torch_mm_config_id(device: torch.device) -> str:
    return f"torch_mm_{device.type}"


class TorchMatmulEvaluator:
    """Torch baseline via torch.mm."""

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

    def get_gpu_name(self) -> str:
        return get_device_name(self.device)

    def get_torch_config_id(self) -> str:
        return _torch_mm_config_id(self.device)

    def _prepare_tensor_map(self, shapes: Sequence[Shape]) -> Dict[Shape, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ordered_unique = list(dict.fromkeys(shapes))
        tensor_map: Dict[Shape, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for shape in ordered_unique:
            tensor_map[shape] = self._get_tensors(shape)
        return tensor_map

    def _precompile_first_launch(
        self,
        shapes: Sequence[Shape],
        tensor_map: Dict[Shape, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        if self._first_launch_done:
            return 0.0
        first_shape = shapes[0]
        a, b, c = tensor_map[first_shape]
        t0 = time.perf_counter()
        self._launch(a, b, c)
        synchronize(self.device)
        compile_time_ms_first = (time.perf_counter() - t0) * 1000.0
        self._first_launch_done = True
        return compile_time_ms_first

    def _warmup_unique_shapes(
        self,
        tensor_map: Dict[Shape, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        for a, b, c in tensor_map.values():
            for _ in range(self.warmup):
                self._launch(a, b, c)
            synchronize(self.device)

    def _build_step_launches(
        self,
        shapes: Sequence[Shape],
        tensor_map: Dict[Shape, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> list[Callable[[], None]]:
        step_launches: list[Callable[[], None]] = []
        for shape in shapes:
            a, b, c = tensor_map[shape]
            for _ in range(self.repeat):
                step_launches.append(lambda a=a, b=b, c=c: self._launch(a, b, c))
        return step_launches

    def _profile_step_launches(
        self,
        shapes: Sequence[Shape],
        step_launches: list[Callable[[], None]],
    ) -> tuple[list[float], str]:
        step_latencies, timing_note = profile_npu_step_launches_us(self.device, step_launches)
        expected = len(shapes) * self.repeat
        if len(step_latencies) < expected:
            raise RuntimeError(
                f"profile_npu_step_launches_us 返回长度不足: got={len(step_latencies)} expected={expected}"
            )
        return step_latencies, timing_note

    def _make_notes(self, timing_note: str) -> str:
        return f"{self.get_torch_config_id()};timing={timing_note}"

    def _make_single_metric(
        self,
        compile_time_ms: float,
        latencies: list[float],
        timing_note: str,
    ) -> Dict[str, object]:
        p50_us = _percentile(latencies, 50)
        return EvalMetrics(
            compile_time_ms=compile_time_ms,
            runtime_cost_us=p50_us,
            invalid_config=0,
            notes=self._make_notes(timing_note),
        ).to_dict()

    def _build_success_metrics(
        self,
        shapes: Sequence[Shape],
        compile_time_ms_first: float,
        step_latencies: list[float],
        timing_note: str,
    ) -> list[Dict[str, object]]:
        out: list[Dict[str, object]] = []
        for idx, _ in enumerate(shapes):
            start = idx * self.repeat
            end = start + self.repeat
            latencies = step_latencies[start:end]
            compile_time_ms = compile_time_ms_first if idx == 0 else 0.0
            out.append(self._make_single_metric(compile_time_ms, latencies, timing_note))
        return out

    def _build_error_metrics(
        self,
        shapes: Sequence[Shape],
        compile_time_ms_first: float,
        exc: Exception,
    ) -> list[Dict[str, object]]:
        note = f"{type(exc).__name__}: {exc}"
        return [
            EvalMetrics(
                compile_time_ms=compile_time_ms_first if idx == 0 else 0.0,
                runtime_cost_us=0.0,
                invalid_config=1,
                notes=note,
            ).to_dict()
            for idx, _ in enumerate(shapes)
        ]

    def evaluate_batch(self, shapes: Sequence[Shape]) -> list[Dict[str, object]]:
        if not shapes:
            return []

        compile_time_ms_first = 0.0
        try:
            tensor_map = self._prepare_tensor_map(shapes)
            compile_time_ms_first = self._precompile_first_launch(shapes, tensor_map)
            self._warmup_unique_shapes(tensor_map)
            step_launches = self._build_step_launches(shapes, tensor_map)
            step_latencies, timing_note = self._profile_step_launches(shapes, step_launches)
            return self._build_success_metrics(shapes, compile_time_ms_first, step_latencies, timing_note)
        except Exception as exc:  # noqa: BLE001
            return self._build_error_metrics(shapes, compile_time_ms_first, exc)
