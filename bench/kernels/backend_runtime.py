from __future__ import annotations

from collections import defaultdict
import csv
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import torch

SUPPORTED_DEVICE_TYPES = ("cuda", "npu")


def _backend_module(device_type: str):
    return getattr(torch, device_type, None)


def backend_is_available(device_type: str) -> bool:
    module = _backend_module(device_type)
    if module is None or not hasattr(module, "is_available"):
        return False
    try:
        return bool(module.is_available())
    except Exception:  # noqa: BLE001
        return False


def resolve_device(device: str) -> torch.device:
    text = (device or "auto").strip().lower()
    if text == "auto":
        for device_type in SUPPORTED_DEVICE_TYPES:
            if backend_is_available(device_type):
                return torch.device(f"{device_type}:0")
        raise RuntimeError("未检测到可用加速设备（cuda/npu）")

    out = torch.device(device)
    if out.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(f"不支持 device={device}，仅支持: {SUPPORTED_DEVICE_TYPES} 或 auto")
    if not backend_is_available(out.type):
        raise RuntimeError(f"{out.type} 不可用，无法执行 benchmark")
    return out


def synchronize(device: torch.device) -> None:
    module = _backend_module(device.type)
    if module is None:
        return

    sync_fn = getattr(module, "synchronize", None)
    if sync_fn is None:
        return

    if device.index is not None:
        try:
            sync_fn(device.index)
            return
        except TypeError:
            pass

    try:
        sync_fn(device)
        return
    except TypeError:
        pass

    sync_fn()


def get_device_name(device: torch.device) -> str:
    module = _backend_module(device.type)
    if module is None:
        return str(device)

    getter = getattr(module, "get_device_name", None)
    if getter is None:
        return str(device)

    if device.index is not None:
        try:
            return str(getter(device.index))
        except TypeError:
            pass

    try:
        return str(getter(device))
    except TypeError:
        pass

    try:
        return str(getter())
    except TypeError:
        return str(device)


def _make_event(device: torch.device):
    module = _backend_module(device.type)
    if module is None:
        return None
    event_cls = getattr(module, "Event", None)
    if event_cls is None:
        return None
    try:
        return event_cls(enable_timing=True)
    except TypeError:
        return event_cls()


def _parse_npu_profiler_kernel_latencies_us(prof_root: Path) -> list[float]:
    kernel_files = list(prof_root.rglob("kernel_details.csv"))
    if not kernel_files:
        return []

    latencies_us: list[float] = []
    for kernel_file in kernel_files:
        with kernel_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = str(row.get("Duration(us)", "")).replace("\t", "").strip()
                if not raw:
                    continue
                try:
                    latencies_us.append(float(raw))
                except ValueError:
                    continue
    return latencies_us


def _parse_npu_profiler_kernel_step_durations_us(prof_root: Path) -> dict[int, float]:
    kernel_files = list(prof_root.rglob("kernel_details.csv"))
    if not kernel_files:
        return {}

    step_durations: dict[int, float] = defaultdict(float)
    for kernel_file in kernel_files:
        with kernel_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_raw = str(row.get("Step Id", "")).strip()
                dur_raw = str(row.get("Duration(us)", "")).replace("\t", "").strip()
                if not step_raw or not dur_raw:
                    continue
                try:
                    step_id = int(float(step_raw))
                    dur = float(dur_raw)
                except ValueError:
                    continue
                step_durations[step_id] += dur
    return dict(step_durations)


def _parse_npu_profiler_stage_latencies_us(prof_root: Path) -> list[float]:
    step_files = list(prof_root.rglob("step_trace_time.csv"))
    if not step_files:
        return []

    latencies_us: list[float] = []
    for step_file in step_files:
        with step_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = str(row.get("Computing", "")).strip()
                if not raw:
                    raw = str(row.get("Stage", "")).strip()
                if not raw:
                    continue
                try:
                    latencies_us.append(float(raw))
                except ValueError:
                    continue
    return latencies_us


def _parse_npu_profiler_stage_step_durations_us(prof_root: Path) -> dict[int, float]:
    step_files = list(prof_root.rglob("step_trace_time.csv"))
    if not step_files:
        return {}

    step_durations: dict[int, float] = {}
    for step_file in step_files:
        with step_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_raw = str(row.get("Step", "")).strip()
                dur_raw = str(row.get("Computing", "")).strip()
                if not dur_raw:
                    dur_raw = str(row.get("Stage", "")).strip()
                if not step_raw or not dur_raw:
                    continue
                try:
                    step_id = int(float(step_raw))
                    dur = float(dur_raw)
                except ValueError:
                    continue
                step_durations[step_id] = dur
    return step_durations


def _run_npu_profiler_offline_analyse(profiler_module, prof_root: Path) -> tuple[bool, str]:
    analyse_mod = getattr(profiler_module, "profiler", None)
    analyse_fn = getattr(analyse_mod, "analyse", None) if analyse_mod is not None else None
    if analyse_fn is None:
        return False, "offline_analyse_api_missing"

    try:
        analyse_fn(str(prof_root), export_type=profiler_module.ExportType.Text)
        return True, "offline_analyse_ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"offline_analyse_failed:{type(exc).__name__}"


def _align_step_durations_to_expected(step_durations: dict[int, float], expected_steps: int) -> list[float]:
    if expected_steps <= 0 or not step_durations:
        return []

    cleaned: dict[int, float] = {}
    for step_id, dur in step_durations.items():
        try:
            sid = int(step_id)
            latency = float(dur)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(latency) or latency < 0:
            continue
        cleaned[sid] = latency

    if not cleaned:
        return []

    candidate_starts = {min(cleaned.keys()), 0, 1}
    candidate_starts.update(step_id for step_id, dur in cleaned.items() if dur > 0)

    best_series: list[float] = []
    best_non_zero = -1
    for start in sorted(candidate_starts):
        series = [float(cleaned.get(start + i, 0.0)) for i in range(expected_steps)]
        non_zero = sum(1 for x in series if x > 0)
        if non_zero > best_non_zero:
            best_series = series
            best_non_zero = non_zero

    if best_non_zero <= 0:
        ordered = [float(cleaned[k]) for k in sorted(cleaned.keys())]
        if len(ordered) >= expected_steps:
            return ordered[:expected_steps]
        return ordered + [0.0] * (expected_steps - len(ordered))

    return best_series


def _profile_npu_step_launches_us_single(
    device: torch.device,
    step_launches: list[Callable[[], None]],
) -> tuple[list[float], str]:
    if device.type != "npu":
        raise ValueError("profile_npu_step_launches_us 仅支持 npu 设备")
    if not step_launches:
        return [], "npu_profiler_empty_steps"

    try:
        from torch_npu import profiler  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return [], f"profiler_import_failed:{type(exc).__name__}"

    active_steps = len(step_launches)
    prof_root = Path(tempfile.mkdtemp(prefix="npu_prof_batch_"))
    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
    sched = profiler.schedule(wait=0, warmup=1, active=active_steps, repeat=1)
    trace_cb = profiler.tensorboard_trace_handler(
        dir_name=str(prof_root),
        analyse_flag=False,
        async_mode=False,
    )
    exp_cfg = profiler._ExperimentalConfig(export_type=profiler.ExportType.Text)

    try:
        with profiler.profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=trace_cb,
            experimental_config=exp_cfg,
        ) as prof:
            # schedule warmup step (not counted in active steps)
            step_launches[0]()
            synchronize(device)
            prof.step()
            for launch in step_launches:
                launch()
                synchronize(device)
                prof.step()

        analysed, analyse_note = _run_npu_profiler_offline_analyse(profiler, prof_root)
        if not analysed:
            return [], analyse_note

        kernel_step = _parse_npu_profiler_kernel_step_durations_us(prof_root)
        if kernel_step:
            latencies = _align_step_durations_to_expected(kernel_step, active_steps)
            if latencies and sum(1 for x in latencies if x > 0) >= max(1, int(active_steps * 0.9)):
                return latencies, "npu_profiler_batch_offline_kernel_step"

        stage_step = _parse_npu_profiler_stage_step_durations_us(prof_root)
        if stage_step:
            latencies = _align_step_durations_to_expected(stage_step, active_steps)
            if latencies and sum(1 for x in latencies if x > 0) >= max(1, int(active_steps * 0.8)):
                return latencies, "npu_profiler_batch_offline_step_trace_stage"

        stage_flat = _parse_npu_profiler_stage_latencies_us(prof_root)
        if len(stage_flat) >= active_steps:
            return stage_flat[:active_steps], "npu_profiler_batch_offline_step_trace_flat_fallback"

        flat = _parse_npu_profiler_kernel_latencies_us(prof_root)
        if len(flat) >= active_steps:
            return flat[:active_steps], "npu_profiler_batch_offline_kernel_flat_fallback"
        if flat:
            return flat, "npu_profiler_batch_offline_short_kernel_flat_fallback"
        if stage_flat:
            return stage_flat, "npu_profiler_batch_offline_short_step_trace_flat_fallback"
        return [], f"{analyse_note};npu_profiler_batch_no_csv"
    except Exception as exc:  # noqa: BLE001
        return [], f"npu_profiler_batch_failed:{type(exc).__name__}"
    finally:
        shutil.rmtree(prof_root, ignore_errors=True)


def profile_npu_step_launches_us(
    device: torch.device,
    step_launches: list[Callable[[], None]],
) -> tuple[list[float], str]:
    if device.type != "npu":
        raise ValueError("profile_npu_step_launches_us 仅支持 npu 设备")
    if not step_launches:
        return [], "npu_profiler_empty_steps"

    max_steps_raw = os.environ.get("NPU_PROFILER_BATCH_MAX_STEPS", "2048").strip()
    try:
        max_steps = int(max_steps_raw)
    except ValueError:
        max_steps = 2048
    if max_steps <= 0:
        max_steps = len(step_launches)

    total_steps = len(step_launches)
    if total_steps <= max_steps:
        return _profile_npu_step_launches_us_single(device, step_launches)

    merged_latencies: list[float] = []
    chunk_notes: list[str] = []
    for chunk_start in range(0, total_steps, max_steps):
        chunk_steps = step_launches[chunk_start : chunk_start + max_steps]
        latencies, note = _profile_npu_step_launches_us_single(device, chunk_steps)
        if len(latencies) < len(chunk_steps):
            return [], f"chunked_batch_failed@{chunk_start}:{note}"
        merged_latencies.extend(latencies[: len(chunk_steps)])
        chunk_notes.append(note)

    uniq_notes = sorted(set(chunk_notes))
    note_suffix = "|".join(uniq_notes)
    return merged_latencies, f"npu_profiler_batch_chunked(max_steps={max_steps});{note_suffix}"


def _measure_latencies_us_with_torch_npu_profiler(
    device: torch.device,
    repeat: int,
    launch: Callable[[], None],
) -> tuple[list[float], str]:
    try:
        from torch_npu import profiler  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return [], f"profiler_import_failed:{type(exc).__name__}"

    prof_root = Path(tempfile.mkdtemp(prefix="npu_prof_eval_"))
    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU]
    # Use one warmup step inside schedule to avoid profile() warmup warning.
    sched = profiler.schedule(wait=0, warmup=1, active=repeat, repeat=1)
    trace_cb = profiler.tensorboard_trace_handler(
        dir_name=str(prof_root),
        analyse_flag=False,
        async_mode=False,
    )
    exp_cfg = profiler._ExperimentalConfig(export_type=profiler.ExportType.Text)

    try:
        with profiler.profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=trace_cb,
            experimental_config=exp_cfg,
        ) as prof:
            for _ in range(repeat + 1):
                launch()
                synchronize(device)
                prof.step()

        analysed, analyse_note = _run_npu_profiler_offline_analyse(profiler, prof_root)
        if not analysed:
            return [], analyse_note
        latencies = _parse_npu_profiler_kernel_latencies_us(prof_root)
        if latencies:
            return latencies, "npu_profiler_offline_kernel_details"

        stage_latencies = _parse_npu_profiler_stage_latencies_us(prof_root)
        if stage_latencies:
            return stage_latencies, "npu_profiler_offline_step_trace_stage"
        return [], f"{analyse_note};npu_profiler_no_csv"
    except Exception as exc:  # noqa: BLE001
        return [], f"npu_profiler_failed:{type(exc).__name__}"
    finally:
        shutil.rmtree(prof_root, ignore_errors=True)


def measure_latencies_us(
    device: torch.device,
    repeat: int,
    launch: Callable[[], None],
    *,
    npu_timing: str = "event",
) -> tuple[list[float], str]:
    if repeat <= 0:
        return [], "repeat<=0"

    if device.type == "npu" and npu_timing == "profiler":
        prof_latencies, prof_note = _measure_latencies_us_with_torch_npu_profiler(device, repeat, launch)
        if prof_latencies:
            return prof_latencies, prof_note
        # fallback to backend event timing if profiler path is unavailable.
        fallback_note = f"{prof_note};fallback_event"
    else:
        fallback_note = "event"

    start_ev = _make_event(device)
    end_ev = _make_event(device)
    latencies_us: list[float] = []

    if start_ev is not None and end_ev is not None:
        for _ in range(repeat):
            start_ev.record()
            launch()
            end_ev.record()
            synchronize(device)
            latencies_us.append(float(start_ev.elapsed_time(end_ev)) * 1000.0)
        return latencies_us, fallback_note

    for _ in range(repeat):
        t0 = time.perf_counter()
        launch()
        synchronize(device)
        latencies_us.append((time.perf_counter() - t0) * 1_000_000.0)
    return latencies_us, f"{fallback_note};host_perf_counter"


def torch_mm_config_id(device: torch.device) -> str:
    return f"torch_mm_{device.type}"


def align_ascend_toolchain_env() -> str:
    """Align bishengir-compile and bisheng to the same Ascend toolkit tree."""
    bishengir = shutil.which("bishengir-compile")
    if not bishengir:
        return "missing_bishengir_compile"

    raw_path = Path(bishengir)
    candidate_roots = [raw_path.parent.parent, raw_path.resolve().parent.parent]
    toolkit_root = None
    ccec_bin = None
    bin_dir = None
    for root in candidate_roots:
        this_ccec_bin = root / "compiler" / "ccec_compiler" / "bin"
        this_bin_dir = root / "bin"
        if this_ccec_bin.is_dir() and this_bin_dir.is_dir():
            toolkit_root = root
            ccec_bin = this_ccec_bin
            bin_dir = this_bin_dir
            break

    if toolkit_root is None or ccec_bin is None or bin_dir is None:
        return "toolchain_root_not_found"

    updates: list[str] = []

    preferred = [str(ccec_bin), str(bin_dir)]
    old_entries = [x for x in os.environ.get("PATH", "").split(os.pathsep) if x]

    preferred_norm = {str(Path(x).resolve()) for x in preferred}
    rest = []
    for entry in old_entries:
        try:
            entry_norm = str(Path(entry).resolve())
        except Exception:  # noqa: BLE001
            entry_norm = entry
        if entry_norm in preferred_norm:
            continue
        rest.append(entry)

    new_entries = preferred + rest
    old_path = os.environ.get("PATH", "")
    new_path = os.pathsep.join(new_entries)
    if new_path != old_path:
        os.environ["PATH"] = new_path
        updates.append("path_reordered")

    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        cur = os.environ.get(key, "")
        if cur != str(toolkit_root):
            os.environ[key] = str(toolkit_root)
            updates.append(f"{key}_set")

    return ",".join(updates) if updates else "no_change"
