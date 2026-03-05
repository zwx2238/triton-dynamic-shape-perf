from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _geometric_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(statistics.fmean(math.log(v) for v in vals))


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
    if method in {"BUCKET", "FULL"}:
        cfg_id = str(row.get("config_id", "")).strip()
        bm = str(row.get("BLOCK_M", "")).strip()
        bn = str(row.get("BLOCK_N", "")).strip()
        bk = str(row.get("BLOCK_K", "")).strip()
        if bm and bn and bk and bm != "-1" and bn != "-1" and bk != "-1":
            core = f"BM={bm},BN={bn},BK={bk}"
            return f"{cfg_id} {core}".strip()
        return cfg_id
    if method == "TORCH":
        dtype = str(row.get("dtype", "")).strip() or "unknown"
        device = "npu"
        config_id = str(row.get("config_id", "")).strip()
        prefix = "torch_mm_"
        if config_id.startswith(prefix) and len(config_id) > len(prefix):
            device = config_id[len(prefix):]
        return f"torch.mm(dtype={dtype},device={device})"
    return str(row.get("config_id", "")).strip()


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
    print_table: bool = True,
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

    known_methods = ["TORCH", "BUCKET", "FULL"]
    present_methods = {str(r.get("method", "")) for r in filtered}
    methods = [m for m in known_methods if m in present_methods]
    if "TORCH" not in methods:
        raise RuntimeError("compare_case_runtime 失败: 缺少 TORCH 基线数据")
    if len(methods) < 2:
        raise RuntimeError(f"compare_case_runtime 失败: 可比较方法不足，present={sorted(present_methods)}")

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
        if "shape_id" not in out or not str(out.get("shape_id", "")).strip():
            out["shape_id"] = shape_id
        if "shape" not in out or not str(out.get("shape", "")).strip():
            out["shape"] = _format_shape(row)

        out[f"runtime_cost_us_{method}"] = row.get("runtime_cost_us", "")
        if method != "TORCH":
            out[f"config_id_{method}"] = row.get("config_id", "")
            out[f"config_desc_{method}"] = _format_config_desc(method, row)
        out[f"invalid_config_{method}"] = row.get("invalid_config", "")
        out[f"timing_source_{method}"] = _extract_timing(row.get("notes", ""))

    target_methods = [m for m in methods if m != "TORCH"]
    cross_speedup_cols: List[str] = []
    if "BUCKET" in methods and "FULL" in methods:
        cross_speedup_cols.append("speedup_BUCKET_vs_FULL")

    out_rows = list(table.values())
    for row in out_rows:
        base_torch = _to_float(row.get("runtime_cost_us_TORCH", ""))
        for m in target_methods:
            cur = _to_float(row.get(f"runtime_cost_us_{m}", ""))
            speedup_col = f"speedup_{m}_vs_TORCH"
            if base_torch > 0 and cur > 0:
                # Torch is baseline: speedup = runtime_torch / runtime_method.
                row[speedup_col] = f"{base_torch / cur:.6f}"
            else:
                row[speedup_col] = ""
        if "speedup_BUCKET_vs_FULL" in cross_speedup_cols:
            runtime_bucket = _to_float(row.get("runtime_cost_us_BUCKET", ""))
            runtime_full = _to_float(row.get("runtime_cost_us_FULL", ""))
            if runtime_bucket > 0 and runtime_full > 0:
                # FULL is baseline here: speedup of bucket over full = runtime_full / runtime_bucket.
                row["speedup_BUCKET_vs_FULL"] = f"{runtime_full / runtime_bucket:.6f}"
            else:
                row["speedup_BUCKET_vs_FULL"] = ""

    out_rows.sort(key=lambda r: _shape_sort_key(str(r.get("shape_id", ""))))

    has_full = ("FULL" in methods)
    display_rows: List[Dict[str, object]] = []
    bucket_speedup_vals: List[float] = []
    full_speedup_vals: List[float] = []
    bucket_over_full_vals: List[float] = []
    for row in out_rows:
        shape = str(row.get("shape", "")).strip()
        if not shape:
            shape = _format_shape(row)

        config_bucket = row.get("config_desc_BUCKET", "") or row.get("config_id_BUCKET", "")
        config_full = row.get("config_desc_FULL", "") or row.get("config_id_FULL", "")
        config_bucket_eq_full = ""
        if has_full and config_bucket and config_full:
            config_bucket_eq_full = str(config_bucket).strip() == str(config_full).strip()

        bucket_speedup_text = row.get("speedup_BUCKET_vs_TORCH", "")
        full_speedup_text = row.get("speedup_FULL_vs_TORCH", "")
        bucket_over_full_text = row.get("speedup_BUCKET_vs_FULL", "")

        bucket_speedup_value = _to_float(bucket_speedup_text)
        if bucket_speedup_value > 0:
            bucket_speedup_vals.append(bucket_speedup_value)
        full_speedup_value = _to_float(full_speedup_text)
        if full_speedup_value > 0:
            full_speedup_vals.append(full_speedup_value)
        bucket_over_full_value = _to_float(bucket_over_full_text)
        if bucket_over_full_value > 0:
            bucket_over_full_vals.append(bucket_over_full_value)
        bucket_over_full_delta_pct_text = (
            f"{(bucket_over_full_value - 1.0) * 100.0:.6f}%"
            if bucket_over_full_value > 0
            else ""
        )

        display_rows.append(
            {
                "shape_id": row.get("shape_id", ""),
                "shape": shape,
                "config_bucket": config_bucket,
                "config_full": config_full,
                "config_bucket_eq_full": config_bucket_eq_full,
                "runtime_torch_us": row.get("runtime_cost_us_TORCH", ""),
                "runtime_bucket_us": row.get("runtime_cost_us_BUCKET", ""),
                "runtime_full_us": row.get("runtime_cost_us_FULL", ""),
                "speedup_bucket_vs_torch": bucket_speedup_text,
                "speedup_full_vs_torch": full_speedup_text,
                "speedup_bucket_vs_full": bucket_over_full_text,
                "speedup_bucket_vs_full_delta_pct": bucket_over_full_delta_pct_text,
            }
        )

    gm_bucket_speedup = _geometric_mean(bucket_speedup_vals)
    gm_full_speedup = _geometric_mean(full_speedup_vals)
    gm_bucket_over_full = _geometric_mean(bucket_over_full_vals)
    display_rows.append(
        {
            "shape_id": "speedup_vs_torch",
            "shape": "geomean",
            "config_bucket": "",
            "config_full": "",
            "config_bucket_eq_full": "",
            "runtime_torch_us": "",
            "runtime_bucket_us": "",
            "runtime_full_us": "",
            "speedup_bucket_vs_torch": _fmt(gm_bucket_speedup) if gm_bucket_speedup > 0 else "",
            "speedup_full_vs_torch": _fmt(gm_full_speedup) if gm_full_speedup > 0 else "",
            "speedup_bucket_vs_full": _fmt(gm_bucket_over_full) if gm_bucket_over_full > 0 else "",
            "speedup_bucket_vs_full_delta_pct": f"{(gm_bucket_over_full - 1.0) * 100.0:.6f}%"
            if gm_bucket_over_full > 0
            else "",
        }
    )

    fieldnames = ["shape_id", "shape", "config_bucket"]
    if has_full:
        fieldnames.extend(["config_full", "config_bucket_eq_full"])
    fieldnames.extend(
        [
            "runtime_torch_us",
            "runtime_bucket_us",
        ]
    )
    if has_full:
        fieldnames.append("runtime_full_us")
    fieldnames.extend(["speedup_bucket_vs_torch"])
    if has_full:
        fieldnames.extend(
            ["speedup_full_vs_torch", "speedup_bucket_vs_full", "speedup_bucket_vs_full_delta_pct"]
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(display_rows)

    if print_table:
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            saved_reader = csv.DictReader(f)
            saved_rows = list(saved_reader)
            saved_headers = list(saved_reader.fieldnames or [])
        _print_section("CASE COMPARE (EVAL)", saved_rows, saved_headers)

    return out_csv, len(display_rows)
