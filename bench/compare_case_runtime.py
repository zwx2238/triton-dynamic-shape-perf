from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _to_int(text: str) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _shape_sort_key(shape_id: str) -> Tuple[int, str]:
    if not shape_id:
        return (10**9, "")
    tail = shape_id.split("_")[-1]
    if tail.isdigit():
        return (int(tail), shape_id)
    return (10**9, shape_id)


def _extract_timing(notes: str) -> str:
    if not notes:
        return ""
    for part in str(notes).split(";"):
        p = part.strip()
        if p.startswith("timing="):
            return p[len("timing=") :]
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 case 对齐各方法 runtime_cost_us，生成对比 CSV。")
    parser.add_argument("--input-csv", type=str, required=True, help="llm_full_vs_bucket 原始结果 CSV")
    parser.add_argument("--split", type=str, default="eval", help="默认: eval")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="指定方法列表；默认使用该 split 下出现过的方法",
    )
    parser.add_argument("--baseline", type=str, default="FULL", help="默认: FULL")
    parser.add_argument("--out-csv", type=str, default="", help="默认: <input_stem>_case_compare_<split>.csv")
    parser.add_argument(
        "--allow-mixed-metric",
        action="store_true",
        help="允许 tune/eval 计时口径不一致（默认不允许）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_csv}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = input_csv.with_name(f"{input_csv.stem}_case_compare_{args.split}.csv")

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filtered = [r for r in rows if r.get("split", "") == args.split]
    if not filtered:
        raise RuntimeError(f"在 split={args.split} 下没有数据")

    methods = args.methods or sorted({r.get("method", "") for r in filtered if r.get("method", "")})
    if not methods:
        raise RuntimeError("没有可用方法")

    timing_by_method_split: Dict[Tuple[str, str], set[str]] = {}
    for row in rows:
        method = row.get("method", "")
        split = row.get("split", "")
        if method not in methods:
            continue
        timing = _extract_timing(row.get("notes", ""))
        if not timing:
            continue
        key = (method, split)
        timing_by_method_split.setdefault(key, set()).add(timing)

    if not args.allow_mixed_metric:
        problems: List[str] = []
        for m in methods:
            tune_set = timing_by_method_split.get((m, "tune"), set())
            eval_set = timing_by_method_split.get((m, args.split), set())
            if tune_set and eval_set and tune_set != eval_set:
                problems.append(f"{m}: tune={sorted(tune_set)} eval={sorted(eval_set)}")
        if problems:
            raise RuntimeError(
                "检测到口径混用（tune/eval timing 不一致），拒绝生成 compare CSV。\n"
                + "\n".join(problems)
                + "\n如需强制继续，请加 --allow-mixed-metric"
            )

    key_fields = ("shape_id", "M", "N", "K", "dtype", "gpu", "bucket_m", "bucket_n", "bucket_k")
    table: Dict[Tuple[str, ...], Dict[str, str]] = {}

    for row in filtered:
        method = row.get("method", "")
        if method not in methods:
            continue

        key = tuple(row.get(f, "") for f in key_fields)
        out = table.setdefault(
            key,
            {
                "shape_id": row.get("shape_id", ""),
                "M": row.get("M", ""),
                "N": row.get("N", ""),
                "K": row.get("K", ""),
                "dtype": row.get("dtype", ""),
                "gpu": row.get("gpu", ""),
                "bucket_m": row.get("bucket_m", ""),
                "bucket_n": row.get("bucket_n", ""),
                "bucket_k": row.get("bucket_k", ""),
            },
        )

        out[f"runtime_cost_us_{method}"] = row.get("runtime_cost_us", "")
        out[f"p99_us_{method}"] = row.get("p99_us", "")
        out[f"config_id_{method}"] = row.get("config_id", "")
        out[f"invalid_config_{method}"] = row.get("invalid_config", "")
        out[f"timing_source_{method}"] = _extract_timing(row.get("notes", ""))

    runtime_cols = [f"runtime_cost_us_{m}" for m in methods]
    p99_cols = [f"p99_us_{m}" for m in methods]
    cfg_cols = [f"config_id_{m}" for m in methods]
    invalid_cols = [f"invalid_config_{m}" for m in methods]
    timing_cols = [f"timing_source_{m}" for m in methods]

    ratio_cols: List[str] = []
    delta_cols: List[str] = []
    if args.baseline in methods:
        for m in methods:
            if m == args.baseline:
                continue
            ratio_col = f"ratio_{m}_over_{args.baseline}"
            delta_col = f"delta_us_{m}_minus_{args.baseline}"
            ratio_cols.append(ratio_col)
            delta_cols.append(delta_col)

    out_rows = list(table.values())
    for row in out_rows:
        if args.baseline not in methods:
            continue
        base = _to_float(row.get(f"runtime_cost_us_{args.baseline}", ""))
        for m in methods:
            if m == args.baseline:
                continue
            cur = _to_float(row.get(f"runtime_cost_us_{m}", ""))
            ratio_col = f"ratio_{m}_over_{args.baseline}"
            delta_col = f"delta_us_{m}_minus_{args.baseline}"
            if base > 0 and cur > 0:
                row[ratio_col] = f"{cur / base:.6f}"
                row[delta_col] = f"{cur - base:.6f}"
            else:
                row[ratio_col] = ""
                row[delta_col] = ""

    out_rows.sort(key=lambda r: _shape_sort_key(r.get("shape_id", "")))

    fieldnames = [
        "shape_id",
        "M",
        "N",
        "K",
        "dtype",
        "gpu",
        "bucket_m",
        "bucket_n",
        "bucket_k",
        *runtime_cols,
        *p99_cols,
        *cfg_cols,
        *invalid_cols,
        *timing_cols,
        *ratio_cols,
        *delta_cols,
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"saved: {out_csv}")
    print(f"rows : {len(out_rows)}")
    print(f"split: {args.split}")
    print(f"methods: {methods}")


if __name__ == "__main__":
    main()
