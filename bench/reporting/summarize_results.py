from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

METHOD_ORDER = {"TORCH": 0, "BUCKET": 1, "FULL_TUNE": 2}


def _to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _shape_sort_key(shape_id: str) -> Tuple[int, str]:
    if not shape_id:
        return (10**9, "")
    tail = shape_id.split("_")[-1]
    if tail.isdigit():
        return (int(tail), shape_id)
    return (10**9, shape_id)


def _method_sort_key(method: str) -> Tuple[int, str]:
    return (METHOD_ORDER.get(method, 99), method)


def _to_float_text(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _power_mean(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    return (sum(v**p for v in values) / len(values)) ** (1.0 / p)


def _print_section(title: str, rows: List[Dict[str, object]], headers: List[str]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return

    widths: Dict[str, int] = {}
    for h in headers:
        widths[h] = len(h)
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def _extract_compare_methods(fieldnames: List[str]) -> List[str]:
    prefix = "runtime_cost_us_"
    methods = [c[len(prefix):] for c in fieldnames if c.startswith(prefix)]
    return sorted(set(methods), key=_method_sort_key)


def _print_case_compare(compare_csv: Path, speed_power: float) -> None:
    if not compare_csv.exists():
        print(f"\n=== CASE COMPARE (EVAL) ===\n(missing file: {compare_csv})")
        return

    with compare_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    methods = _extract_compare_methods(fieldnames)

    display_rows: List[Dict[str, object]] = []
    headers: List[str] = ["shape_id", "shape"]
    if "TORCH" in methods:
        headers.append("runtime_TORCH_us")
    for method in methods:
        if method == "TORCH":
            continue
        headers.extend(
            [
                f"config_{method}",
                f"runtime_{method}_us",
            ]
        )
    for ratio_key in ["speed_BUCKET_over_TORCH", "speed_FULL_TUNE_over_TORCH", "speed_BUCKET_over_FULL_TUNE"]:
        if ratio_key in fieldnames:
            headers.append(ratio_key)

    for row in rows:
        shape = str(row.get("shape", "")).strip()
        if not shape:
            m = str(row.get("M", "")).strip()
            n = str(row.get("N", "")).strip()
            k = str(row.get("K", "")).strip()
            shape = f"{m}x{n}x{k}" if m and n and k else ""
        out: Dict[str, object] = {"shape_id": row.get("shape_id", ""), "shape": shape}
        if "TORCH" in methods:
            out["runtime_TORCH_us"] = row.get("runtime_cost_us_TORCH", "")
        for method in methods:
            if method == "TORCH":
                continue
            out[f"config_{method}"] = row.get(f"config_desc_{method}", "") or row.get(f"config_id_{method}", "")
            out[f"runtime_{method}_us"] = row.get(f"runtime_cost_us_{method}", "")
        out["speed_BUCKET_over_TORCH"] = row.get("speed_BUCKET_over_TORCH", "")
        out["speed_FULL_TUNE_over_TORCH"] = row.get("speed_FULL_TUNE_over_TORCH", "")
        out["speed_BUCKET_over_FULL_TUNE"] = row.get("speed_BUCKET_over_FULL_TUNE", "")
        display_rows.append(out)
    display_rows.sort(key=lambda r: _shape_sort_key(str(r.get("shape_id", ""))))

    summary_row: Dict[str, object] = {
        "shape_id": f"SUMMARY(p={speed_power:g})",
        "shape": f"power_mean_speed, n={len(display_rows)}",
    }
    for ratio_key in ["speed_BUCKET_over_TORCH", "speed_FULL_TUNE_over_TORCH", "speed_BUCKET_over_FULL_TUNE"]:
        if ratio_key not in headers:
            continue
        values = [_to_float_text(str(r.get(ratio_key, ""))) for r in display_rows]
        valid = [v for v in values if v > 0.0]
        summary_row[ratio_key] = f"{_power_mean(valid, speed_power):.6f}" if valid else ""
    display_rows.append(summary_row)

    _print_section("CASE COMPARE (EVAL)", display_rows, headers)


def summarize(
    input_csv: Path,
    out_dir: Path,
    prefix: str,
    compare_csv: Path | None = None,
    speed_power: float = 2.0,
) -> Tuple[Path, Path, Path]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"输入 CSV 为空: {input_csv}")

    eval_by_method: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    tune_by_method: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    eval_by_method_bucket: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)

    for row in rows:
        method = row.get("method", "")
        split = row.get("split", "")
        if split == "eval":
            eval_by_method[method].append(row)
            bk = _to_int(row, "bucket_key", -1)
            eval_by_method_bucket[(method, bk)].append(row)
        elif split == "tune":
            tune_by_method[method].append(row)

    methods = sorted(set(eval_by_method.keys()) | set(tune_by_method.keys()), key=_method_sort_key)

    overall_rows: List[Dict[str, object]] = []
    torch_p50 = None

    for method in methods:
        eval_rows = eval_by_method.get(method, [])
        runtimes = [_to_float(r, "runtime_cost_us") for r in eval_rows]

        if not runtimes:
            row = {
                "method": method,
                "samples": 0,
                "median_runtime_us": "",
                "mean_runtime_us": "",
                "speedup_vs_torch": "",
            }
            overall_rows.append(row)
            continue

        median_runtime = statistics.median(runtimes)
        mean_runtime = statistics.fmean(runtimes)

        row = {
            "method": method,
            "samples": len(runtimes),
            "median_runtime_us": _fmt(median_runtime),
            "mean_runtime_us": _fmt(mean_runtime),
            "speedup_vs_torch": "",
        }
        if method == "TORCH":
            torch_p50 = median_runtime
        overall_rows.append(row)

    if torch_p50 and torch_p50 > 0:
        for row in overall_rows:
            median_raw = row.get("median_runtime_us", "")
            if not median_raw:
                continue
            median_value = float(median_raw)
            row["speedup_vs_torch"] = _fmt(torch_p50 / median_value if median_value > 0 else 0.0)

    tune_rows: List[Dict[str, object]] = []
    for method in methods:
        this_rows = tune_by_method.get(method, [])
        if method == "FULL_TUNE" and not this_rows:
            this_rows = eval_by_method.get(method, [])
        tune_times = [_to_float(r, "tune_time_ms") for r in this_rows]
        compile_times = [_to_float(r, "compile_time_ms") for r in this_rows]
        tune_rows.append(
            {
                "method": method,
                "tune_rows": len(this_rows),
                "tuned_nonzero": sum(1 for v in tune_times if v > 0),
                "total_tune_ms": _fmt(sum(tune_times)),
                "total_compile_ms": _fmt(sum(compile_times)),
            }
        )

    by_bucket_rows: List[Dict[str, object]] = []
    for method, bk in sorted(eval_by_method_bucket.keys(), key=lambda x: (x[0], x[1])):
        bucket_rows = eval_by_method_bucket[(method, bk)]
        runtimes = [_to_float(r, "runtime_cost_us") for r in bucket_rows]
        by_bucket_rows.append(
            {
                "method": method,
                "bucket_key": bk,
                "samples": len(bucket_rows),
                "median_runtime_us": _fmt(statistics.median(runtimes) if runtimes else 0.0),
            }
        )

    overall_path = out_dir / f"{prefix}_summary_overall.csv"
    tune_path = out_dir / f"{prefix}_summary_tune.csv"
    bucket_path = out_dir / f"{prefix}_summary_by_bucket.csv"

    _write_csv(
        overall_path,
        ["method", "samples", "median_runtime_us", "mean_runtime_us", "speedup_vs_torch"],
        overall_rows,
    )
    _write_csv(
        tune_path,
        ["method", "tune_rows", "tuned_nonzero", "total_tune_ms", "total_compile_ms"],
        tune_rows,
    )
    _write_csv(
        bucket_path,
        ["method", "bucket_key", "samples", "median_runtime_us"],
        by_bucket_rows,
    )

    _print_section(
        "OVERALL (EVAL)",
        overall_rows,
        ["method", "samples", "median_runtime_us", "mean_runtime_us", "speedup_vs_torch"],
    )
    _print_section(
        "TUNE COST",
        tune_rows,
        ["method", "tune_rows", "tuned_nonzero", "total_tune_ms", "total_compile_ms"],
    )
    _print_section(
        "BY BUCKET (EVAL)",
        by_bucket_rows,
        ["method", "bucket_key", "samples", "median_runtime_us"],
    )
    if compare_csv is not None:
        if not math.isfinite(float(speed_power)) or float(speed_power) <= 0.0:
            raise ValueError(f"speed_power 必须 > 0 且为有限数，当前: {speed_power}")
        _print_case_compare(compare_csv, float(speed_power))

    return overall_path, tune_path, bucket_path
