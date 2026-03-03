from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from bench.kernels.npu_measure import (
    get_device_name,
    measure_latencies_us,
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

    def evaluate(self, shape: Shape) -> Dict[str, object]:
        compile_time_ms = 0.0

        try:
            a, b, c = self._get_tensors(shape)

            if not self._first_launch_done:
                t0 = time.perf_counter()
                self._launch(a, b, c)
                synchronize(self.device)
                compile_time_ms = (time.perf_counter() - t0) * 1000.0
                self._first_launch_done = True

            for _ in range(self.warmup):
                self._launch(a, b, c)
            synchronize(self.device)

            latencies_us, timing_note = measure_latencies_us(
                self.device,
                self.repeat,
                lambda: self._launch(a, b, c),
            )

            p50_us = _percentile(latencies_us, 50)

            note = f"{self.get_torch_config_id()};timing={timing_note}"
            return EvalMetrics(
                compile_time_ms=compile_time_ms,
                runtime_cost_us=p50_us,
                invalid_config=0,
                notes=note,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            return EvalMetrics(
                compile_time_ms=compile_time_ms,
                runtime_cost_us=0.0,
                invalid_config=1,
                notes=f"{type(exc).__name__}: {exc}",
            ).to_dict()

    def evaluate_batch(self, shapes: Sequence[Shape]) -> list[Dict[str, object]]:
        if not shapes:
            return []

        try:
            ordered_unique = list(dict.fromkeys(shapes))
            tensor_map: Dict[Shape, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for shape in ordered_unique:
                tensor_map[shape] = self._get_tensors(shape)

            compile_time_ms_first = 0.0
            if not self._first_launch_done:
                first_shape = ordered_unique[0]
                a, b, c = tensor_map[first_shape]
                t0 = time.perf_counter()
                self._launch(a, b, c)
                synchronize(self.device)
                compile_time_ms_first = (time.perf_counter() - t0) * 1000.0
                self._first_launch_done = True

            for shape in ordered_unique:
                a, b, c = tensor_map[shape]
                for _ in range(self.warmup):
                    self._launch(a, b, c)
                synchronize(self.device)

            step_launches = []
            for shape in shapes:
                a, b, c = tensor_map[shape]
                for _ in range(self.repeat):
                    step_launches.append(lambda a=a, b=b, c=c: self._launch(a, b, c))

            step_latencies, timing_note = profile_npu_step_launches_us(self.device, step_launches)
            if len(step_latencies) < len(shapes) * self.repeat:
                return [self.evaluate(shape) for shape in shapes]

            out: list[Dict[str, object]] = []
            for idx, shape in enumerate(shapes):
                start = idx * self.repeat
                end = start + self.repeat
                latencies = step_latencies[start:end]

                p50_us = _percentile(latencies, 50)

                note = f"{self.get_torch_config_id()};timing={timing_note}"
                compile_time_ms = compile_time_ms_first if idx == 0 else 0.0
                out.append(
                    EvalMetrics(
                        compile_time_ms=compile_time_ms,
                        runtime_cost_us=p50_us,
                        invalid_config=0,
                        notes=note,
                    ).to_dict()
                )
            return out
        except Exception:  # noqa: BLE001
            return [self.evaluate(shape) for shape in shapes]
