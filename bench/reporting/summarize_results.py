from __future__ import annotations

import csv
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


def _write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def summarize(input_csv: Path, out_dir: Path, prefix: str) -> Tuple[Path, Path, Path]:
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

    return overall_path, tune_path, bucket_path
