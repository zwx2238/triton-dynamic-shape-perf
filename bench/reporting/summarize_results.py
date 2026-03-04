from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def _geometric_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(statistics.fmean(math.log(v) for v in vals))


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


def _print_case_compare(compare_csv: Path) -> None:
    if not compare_csv.exists():
        print(f"\n=== CASE COMPARE (EVAL) ===\n(missing file: {compare_csv})")
        return

    with compare_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    speedup_col = ""
    if rows:
        for col in rows[0].keys():
            if col.startswith("speedup_") and col.endswith("_vs_TORCH"):
                speedup_col = col
                break

    display_rows: List[Dict[str, object]] = []
    speedup_vals: List[float] = []
    for row in rows:
        shape = str(row.get("shape", "")).strip()
        if not shape:
            m = str(row.get("M", "")).strip()
            n = str(row.get("N", "")).strip()
            k = str(row.get("K", "")).strip()
            shape = f"{m}x{n}x{k}" if m and n and k else ""
        speedup_text = row.get(speedup_col, "") if speedup_col else ""
        try:
            speedup_value = float(speedup_text) if speedup_text not in ("", None) else 0.0
        except (TypeError, ValueError):
            speedup_value = 0.0
        if speedup_value > 0:
            speedup_vals.append(speedup_value)
        display_rows.append(
            {
                "shape_id": row.get("shape_id", ""),
                "shape": shape,
                "config_bucket": row.get("config_desc_BUCKET", "") or row.get("config_id_BUCKET", ""),
                "runtime_torch_us": row.get("runtime_cost_us_TORCH", ""),
                "runtime_bucket_us": row.get("runtime_cost_us_BUCKET", ""),
                "speedup": speedup_text,
            }
        )
    display_rows.sort(key=lambda r: _shape_sort_key(str(r.get("shape_id", ""))))
    gm_speedup = _geometric_mean(speedup_vals)
    display_rows.append(
        {
            "shape_id": "speedup_vs_torch",
            "shape": "geomean",
            "config_bucket": "",
            "runtime_torch_us": "",
            "runtime_bucket_us": "",
            "speedup": _fmt(gm_speedup) if gm_speedup > 0 else "",
        }
    )
    _print_section(
        "CASE COMPARE (EVAL)",
        display_rows,
        [
            "shape_id",
            "shape",
            "config_bucket",
            "runtime_torch_us",
            "runtime_bucket_us",
            "speedup",
        ],
    )


def summarize(input_csv: Path, out_dir: Path, prefix: str, compare_csv: Path | None = None) -> Tuple[Path, Path, Path]:
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

    method_order = {"TORCH": 0, "BUCKET": 1}
    methods = sorted(set(eval_by_method.keys()) | set(tune_by_method.keys()), key=lambda x: (method_order.get(x, 99), x))

    overall_rows: List[Dict[str, object]] = []

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
        overall_rows.append(row)

    torch_runtime_by_shape: Dict[str, List[float]] = defaultdict(list)
    for row in eval_by_method.get("TORCH", []):
        shape_id = str(row.get("shape_id", "")).strip()
        runtime = _to_float(row, "runtime_cost_us")
        if shape_id and runtime > 0:
            torch_runtime_by_shape[shape_id].append(runtime)
    torch_median_by_shape = {
        shape_id: statistics.median(vals)
        for shape_id, vals in torch_runtime_by_shape.items()
        if vals
    }

    method_runtime_by_shape: Dict[str, Dict[str, float]] = {}
    for method in methods:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in eval_by_method.get(method, []):
            shape_id = str(row.get("shape_id", "")).strip()
            runtime = _to_float(row, "runtime_cost_us")
            if shape_id and runtime > 0:
                grouped[shape_id].append(runtime)
        method_runtime_by_shape[method] = {
            shape_id: statistics.median(vals)
            for shape_id, vals in grouped.items()
            if vals
        }

    overall_row_by_method = {str(r.get("method", "")): r for r in overall_rows}
    for method in methods:
        row = overall_row_by_method.get(method)
        if row is None:
            continue
        if method == "TORCH":
            row["speedup_vs_torch"] = _fmt(1.0) if row.get("samples", 0) else ""
            continue

        runtime_by_shape = method_runtime_by_shape.get(method, {})
        ratios: List[float] = []
        for shape_id, torch_runtime in torch_median_by_shape.items():
            cur_runtime = runtime_by_shape.get(shape_id, 0.0)
            if torch_runtime > 0 and cur_runtime > 0:
                ratios.append(torch_runtime / cur_runtime)

        row["speedup_vs_torch"] = _fmt(_geometric_mean(ratios)) if ratios else ""

    tune_rows: List[Dict[str, object]] = []
    for method in methods:
        this_rows = tune_by_method.get(method, [])
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
        "TUNE COST",
        tune_rows,
        ["method", "tune_rows", "tuned_nonzero", "total_tune_ms", "total_compile_ms"],
    )
    if compare_csv is not None:
        _print_case_compare(compare_csv)

    return overall_path, tune_path, bucket_path
