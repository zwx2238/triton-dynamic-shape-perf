from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict


def _read_table(path: Path, key_col: str) -> Dict[str, Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row[key_col]: row for row in rows if row.get(key_col)}


def _to_float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate benchmark conclusion on target backend. "
            "Default checks: BUCKET significantly reduces tune time while keeping runtime close to FULL."
        )
    )
    parser.add_argument("--overall-csv", type=str, required=True, help="summary_overall.csv")
    parser.add_argument("--tune-csv", type=str, required=True, help="summary_tune.csv")
    parser.add_argument("--max-bucket-slowdown-vs-full", type=float, default=0.10, help="default: 0.10 (10%)")
    parser.add_argument("--min-bucket-tune-reduction-vs-full", type=float, default=0.50, help="default: 0.50 (50%)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overall = _read_table(Path(args.overall_csv), "method")
    tune = _read_table(Path(args.tune_csv), "method")

    if "FULL" not in overall or "BUCKET" not in overall:
        raise RuntimeError("overall summary 需要至少包含 FULL 和 BUCKET")
    if "FULL" not in tune or "BUCKET" not in tune:
        raise RuntimeError("tune summary 需要至少包含 FULL 和 BUCKET")

    full_runtime_us = _to_float(overall["FULL"], "p50_runtime_us")
    bucket_runtime_us = _to_float(overall["BUCKET"], "p50_runtime_us")
    full_tune_ms = _to_float(tune["FULL"], "total_tune_ms")
    bucket_tune_ms = _to_float(tune["BUCKET"], "total_tune_ms")

    if full_runtime_us <= 0 or full_tune_ms <= 0:
        raise RuntimeError("FULL 的 runtime/tune 统计无效，无法验证结论")

    runtime_slowdown = (bucket_runtime_us / full_runtime_us) - 1.0
    tune_reduction = 1.0 - (bucket_tune_ms / full_tune_ms)

    runtime_ok = runtime_slowdown <= args.max_bucket_slowdown_vs_full
    tune_ok = tune_reduction >= args.min_bucket_tune_reduction_vs_full
    passed = runtime_ok and tune_ok

    print("=== Conclusion Check (FULL vs BUCKET) ===")
    print(f"FULL p50_runtime_us   : {full_runtime_us:.6f}")
    print(f"BUCKET p50_runtime_us : {bucket_runtime_us:.6f}")
    print(f"BUCKET slowdown       : {runtime_slowdown * 100:.2f}% (threshold <= {args.max_bucket_slowdown_vs_full * 100:.2f}%)")
    print(f"FULL total_tune_ms    : {full_tune_ms:.6f}")
    print(f"BUCKET total_tune_ms  : {bucket_tune_ms:.6f}")
    print(
        f"BUCKET tune reduction : {tune_reduction * 100:.2f}% "
        f"(threshold >= {args.min_bucket_tune_reduction_vs_full * 100:.2f}%)"
    )
    print(f"VERDICT               : {'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
