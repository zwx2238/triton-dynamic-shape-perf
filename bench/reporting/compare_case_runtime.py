from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


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
            return p[len("timing="):]
    return ""


def _timing_family(timing: str) -> str:
    t = (timing or "").strip().lower()
    if not t:
        return ""
    if "profiler" in t:
        return "profiler"
    return "unknown"


def _format_shape(row: Dict[str, str]) -> str:
    m = str(row.get("M", "")).strip()
    n = str(row.get("N", "")).strip()
    k = str(row.get("K", "")).strip()
    if m and n and k:
        return f"{m}x{n}x{k}"
    return ""


def _format_config_desc(method: str, row: Dict[str, str]) -> str:
    if method != "TORCH":
        bm = str(row.get("BLOCK_M", "")).strip()
        bn = str(row.get("BLOCK_N", "")).strip()
        bk = str(row.get("BLOCK_K", "")).strip()
        if bm and bn and bk and bm != "-1" and bn != "-1" and bk != "-1":
            return f"BM={bm},BN={bn},BK={bk}"
    if method == "TORCH":
        return ""
    return str(row.get("config_id", "")).strip()


PER_METHOD_COLS = {
    "method", "config_id", "compile_time_ms", "tune_time_ms",
    "runtime_cost_us", "cache_key", "invalid_config", "notes", "timestamp",
}


def compare_case_runtime(
    input_csv: Path,
    *,
    split: str = "eval",
    out_csv: Path | None = None,
    allow_mixed_metric: bool = False,
) -> tuple[Path, int]:
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_csv}")

    if out_csv is None:
        out_csv = input_csv.with_name(f"{input_csv.stem}_case_compare_{split}.csv")

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_columns = list(reader.fieldnames or [])
        rows = list(reader)

    filtered = [r for r in rows if r.get("split", "") == split]
    if not filtered:
        raise RuntimeError(f"在 split={split} 下没有数据")

    method_order = {"TORCH": 0, "BUCKET": 1, "FULL_TUNE": 2}
    methods = sorted(
        {
            str(row.get("method", "")).strip()
            for row in filtered
            if str(row.get("method", "")).strip()
        },
        key=lambda x: (method_order.get(x, 99), x),
    )
    if "TORCH" not in methods:
        raise RuntimeError("compare 需要 TORCH 基线数据（split=eval）")

    timing_by_method_split: Dict[Tuple[str, str], set[str]] = {}
    for row in rows:
        method = row.get("method", "")
        row_split = row.get("split", "")
        if method not in methods:
            continue
        timing = _extract_timing(row.get("notes", ""))
        if not timing:
            continue
        key = (method, row_split)
        timing_by_method_split.setdefault(key, set()).add(timing)

    if not allow_mixed_metric:
        problems: List[str] = []
        for m in methods:
            tune_raw = timing_by_method_split.get((m, "tune"), set())
            eval_raw = timing_by_method_split.get((m, split), set())
            tune_family = {x for x in (_timing_family(t) for t in tune_raw) if x}
            eval_family = {x for x in (_timing_family(t) for t in eval_raw) if x}
            if tune_family and eval_family and tune_family != eval_family:
                problems.append(
                    f"{m}: tune_family={sorted(tune_family)} eval_family={sorted(eval_family)} "
                    f"(raw tune={sorted(tune_raw)} raw eval={sorted(eval_raw)})"
                )
        if problems:
            raise RuntimeError(
                "检测到口径混用（tune/eval timing 不一致），拒绝生成 compare CSV。\n"
                + "\n".join(problems)
                + "\n如需强制继续，请加 --allow-mixed-metric"
            )

    shared_cols = [c for c in all_columns if c not in PER_METHOD_COLS and c != "split"]
    if "shape" not in shared_cols:
        shared_cols.append("shape")

    table: Dict[str, Dict[str, str]] = {}

    for row in filtered:
        method = row.get("method", "")
        if method not in methods:
            continue

        shape_id = row.get("shape_id", "")
        out = table.setdefault(shape_id, {c: row.get(c, "") for c in shared_cols})
        if "shape" not in out or not str(out.get("shape", "")).strip():
            out["shape"] = _format_shape(row)

        out[f"runtime_cost_us_{method}"] = row.get("runtime_cost_us", "")
        out[f"config_id_{method}"] = row.get("config_id", "")
        out[f"config_desc_{method}"] = _format_config_desc(method, row)
        out[f"invalid_config_{method}"] = row.get("invalid_config", "")
        out[f"timing_source_{method}"] = _extract_timing(row.get("notes", ""))

    runtime_cols = [f"runtime_cost_us_{m}" for m in methods]
    cfg_cols = [f"config_id_{m}" for m in methods]
    cfg_desc_cols = [f"config_desc_{m}" for m in methods]
    invalid_cols = [f"invalid_config_{m}" for m in methods]
    timing_cols = [f"timing_source_{m}" for m in methods]
    ratio_cols: List[str] = []
    if "BUCKET" in methods and "TORCH" in methods:
        ratio_cols.append("speed_BUCKET_over_TORCH")
    if "FULL_TUNE" in methods and "TORCH" in methods:
        ratio_cols.append("speed_FULL_TUNE_over_TORCH")
    if "BUCKET" in methods and "FULL_TUNE" in methods:
        ratio_cols.append("speed_BUCKET_over_FULL_TUNE")

    out_rows = list(table.values())
    for row in out_rows:
        runtime_torch = _to_float(row.get("runtime_cost_us_TORCH", ""))
        runtime_bucket = _to_float(row.get("runtime_cost_us_BUCKET", ""))
        runtime_full = _to_float(row.get("runtime_cost_us_FULL_TUNE", ""))
        if "speed_BUCKET_over_TORCH" in ratio_cols:
            row["speed_BUCKET_over_TORCH"] = (
                f"{runtime_torch / runtime_bucket:.6f}" if runtime_torch > 0 and runtime_bucket > 0 else ""
            )
        if "speed_FULL_TUNE_over_TORCH" in ratio_cols:
            row["speed_FULL_TUNE_over_TORCH"] = (
                f"{runtime_torch / runtime_full:.6f}" if runtime_torch > 0 and runtime_full > 0 else ""
            )
        if "speed_BUCKET_over_FULL_TUNE" in ratio_cols:
            row["speed_BUCKET_over_FULL_TUNE"] = (
                f"{runtime_full / runtime_bucket:.6f}" if runtime_full > 0 and runtime_bucket > 0 else ""
            )

    out_rows.sort(key=lambda r: _shape_sort_key(r.get("shape_id", "")))

    fieldnames = [
        *shared_cols,
        *runtime_cols,
        *cfg_cols,
        *cfg_desc_cols,
        *invalid_cols,
        *timing_cols,
        *ratio_cols,
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    return out_csv, len(out_rows)
