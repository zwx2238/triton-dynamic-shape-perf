from __future__ import annotations

import argparse
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
            eval_by_method_bucket[(method, _to_int(row, "bucket_m", -1))].append(row)
        elif split == "tune":
            tune_by_method[method].append(row)

    methods = sorted(set(eval_by_method.keys()) | set(tune_by_method.keys()), key=lambda x: (x != "FULL", x != "BUCKET", x))

    overall_rows: List[Dict[str, object]] = []
    full_p50 = None

    for method in methods:
        eval_rows = eval_by_method.get(method, [])
        runtimes = [_to_float(r, "runtime_cost_us") for r in eval_rows]
        p99s = [_to_float(r, "p99_us") for r in eval_rows]
        tflops = [_to_float(r, "runtime_perf_tflops") for r in eval_rows]

        if not runtimes:
            row = {
                "method": method,
                "samples": 0,
                "p50_runtime_us": "",
                "mean_runtime_us": "",
                "p99_us_median": "",
                "mean_tflops": "",
                "speedup_vs_full": "",
            }
            overall_rows.append(row)
            continue

        p50_runtime = statistics.median(runtimes)
        mean_runtime = statistics.fmean(runtimes)
        p99_median = statistics.median(p99s) if p99s else 0.0
        mean_tflops = statistics.fmean(tflops) if tflops else 0.0

        row = {
            "method": method,
            "samples": len(runtimes),
            "p50_runtime_us": _fmt(p50_runtime),
            "mean_runtime_us": _fmt(mean_runtime),
            "p99_us_median": _fmt(p99_median),
            "mean_tflops": _fmt(mean_tflops),
            "speedup_vs_full": "",
        }
        if method == "FULL":
            full_p50 = p50_runtime
        overall_rows.append(row)

    if full_p50 and full_p50 > 0:
        for row in overall_rows:
            p50_raw = row.get("p50_runtime_us", "")
            if not p50_raw:
                continue
            p50_value = float(p50_raw)
            row["speedup_vs_full"] = _fmt(full_p50 / p50_value if p50_value > 0 else 0.0)

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
    for method, bucket in sorted(eval_by_method_bucket.keys(), key=lambda x: (x[0], x[1])):
        bucket_rows = eval_by_method_bucket[(method, bucket)]
        runtimes = [_to_float(r, "runtime_cost_us") for r in bucket_rows]
        tflops = [_to_float(r, "runtime_perf_tflops") for r in bucket_rows]
        by_bucket_rows.append(
            {
                "method": method,
                "bucket_m": bucket,
                "samples": len(bucket_rows),
                "p50_runtime_us": _fmt(statistics.median(runtimes) if runtimes else 0.0),
                "mean_tflops": _fmt(statistics.fmean(tflops) if tflops else 0.0),
            }
        )

    overall_path = out_dir / f"{prefix}_summary_overall.csv"
    tune_path = out_dir / f"{prefix}_summary_tune.csv"
    bucket_path = out_dir / f"{prefix}_summary_by_bucket.csv"

    _write_csv(
        overall_path,
        ["method", "samples", "p50_runtime_us", "mean_runtime_us", "p99_us_median", "mean_tflops", "speedup_vs_full"],
        overall_rows,
    )
    _write_csv(
        tune_path,
        ["method", "tune_rows", "tuned_nonzero", "total_tune_ms", "total_compile_ms"],
        tune_rows,
    )
    _write_csv(
        bucket_path,
        ["method", "bucket_m", "samples", "p50_runtime_us", "mean_tflops"],
        by_bucket_rows,
    )

    _print_section(
        "OVERALL (EVAL)",
        overall_rows,
        ["method", "samples", "p50_runtime_us", "mean_runtime_us", "p99_us_median", "mean_tflops", "speedup_vs_full"],
    )
    _print_section(
        "TUNE COST",
        tune_rows,
        ["method", "tune_rows", "tuned_nonzero", "total_tune_ms", "total_compile_ms"],
    )
    _print_section(
        "BY BUCKET (EVAL)",
        by_bucket_rows,
        ["method", "bucket_m", "samples", "p50_runtime_us", "mean_tflops"],
    )

    return overall_path, tune_path, bucket_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize llm_full_vs_bucket raw CSV results.")
    parser.add_argument("--input-csv", type=str, default="results/llm_full_vs_bucket.csv")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--prefix", type=str, default="llm_full_vs_bucket")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_csv}")
    out_dir = Path(args.out_dir)

    overall_path, tune_path, bucket_path = summarize(input_csv, out_dir, args.prefix)
    print("\nSaved files:")
    print(f"- {overall_path}")
    print(f"- {tune_path}")
    print(f"- {bucket_path}")


if __name__ == "__main__":
    main()
