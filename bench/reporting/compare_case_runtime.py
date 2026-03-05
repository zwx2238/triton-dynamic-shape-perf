from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from bench.reporting._common import fmt as _fmt, geometric_mean as _geometric_mean, shape_sort_key as _shape_sort_key, print_section as _print_section


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


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


PER_METHOD_COLS = {
    "method", "config_id", "compile_time_ms", "tune_time_ms",
    "runtime_cost_us", "cache_key", "invalid_config", "notes", "timestamp",
}


def _load_csv_and_resolve_out_path(
    input_csv: Path, split: str, out_csv: Path | None
) -> Tuple[Path, List[str], List[Dict[str, str]]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_csv}")
    if out_csv is None:
        out_csv = input_csv.with_name(f"{input_csv.stem}_case_compare_{split}.csv")
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_columns = list(reader.fieldnames or [])
        rows = list(reader)
    return out_csv, all_columns, rows


def _validate_and_get_methods(
    rows: List[Dict[str, str]], split: str
) -> Tuple[List[Dict[str, str]], List[str]]:
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
    return filtered, methods


def _build_timing_by_method_split(
    rows: List[Dict[str, str]], methods: List[str]
) -> Dict[Tuple[str, str], set[str]]:
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
    return timing_by_method_split


def _get_method_timing_info(
    timing_by_method_split: Dict[Tuple[str, str], set[str]], m: str, split: str
) -> Tuple[set[str], set[str], set[str], set[str]]:
    tune_raw = timing_by_method_split.get((m, "tune"), set())
    eval_raw = timing_by_method_split.get((m, split), set())
    tune_family = {x for x in (_timing_family(t) for t in tune_raw) if x}
    eval_family = {x for x in (_timing_family(t) for t in eval_raw) if x}
    return tune_family, eval_family, tune_raw, eval_raw


def _collect_timing_problems(
    timing_by_method_split: Dict[Tuple[str, str], set[str]],
    methods: List[str],
    split: str,
) -> List[str]:
    problems: List[str] = []
    for m in methods:
        tune_fam, eval_fam, tune_raw, eval_raw = _get_method_timing_info(
            timing_by_method_split, m, split
        )
        if tune_fam and eval_fam and tune_fam != eval_fam:
            problems.append(
                f"{m}: tune_family={sorted(tune_fam)} eval_family={sorted(eval_fam)} "
                f"(raw tune={sorted(tune_raw)} raw eval={sorted(eval_raw)})"
            )
    return problems


def _check_timing_consistency(
    timing_by_method_split: Dict[Tuple[str, str], set[str]],
    methods: List[str],
    split: str,
    allow_mixed_metric: bool,
) -> None:
    if allow_mixed_metric:
        return
    problems = _collect_timing_problems(timing_by_method_split, methods, split)
    if problems:
        raise RuntimeError(
            "检测到口径混用（tune/eval timing 不一致），拒绝生成 compare CSV。\n"
            + "\n".join(problems)
            + "\n如需强制继续，请加 --allow-mixed-metric"
        )


def _get_shared_cols(all_columns: List[str]) -> List[str]:
    shared_cols = [c for c in all_columns if c not in PER_METHOD_COLS and c != "split"]
    if "shape" not in shared_cols:
        shared_cols.append("shape")
    return shared_cols


def _populate_table_row(
    out: Dict[str, str], row: Dict[str, str], method: str, shape_id: str
) -> None:
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


def _build_table(
    filtered: List[Dict[str, str]], methods: List[str], shared_cols: List[str]
) -> Dict[str, Dict[str, str]]:
    table: Dict[str, Dict[str, str]] = {}
    for row in filtered:
        method = row.get("method", "")
        if method not in methods:
            continue
        shape_id = row.get("shape_id", "")
        out = table.setdefault(shape_id, {c: row.get(c, "") for c in shared_cols})
        _populate_table_row(out, row, method, shape_id)
    return table


def _compute_row_speedups(
    row: Dict[str, str], target_methods: List[str], cross_speedup_cols: List[str]
) -> None:
    base_torch = _to_float(row.get("runtime_cost_us_TORCH", ""))
    for m in target_methods:
        cur = _to_float(row.get(f"runtime_cost_us_{m}", ""))
        speedup_col = f"speedup_{m}_vs_TORCH"
        if base_torch > 0 and cur > 0:
            row[speedup_col] = f"{base_torch / cur:.6f}"
        else:
            row[speedup_col] = ""
    if "speedup_BUCKET_vs_FULL" in cross_speedup_cols:
        runtime_bucket = _to_float(row.get("runtime_cost_us_BUCKET", ""))
        runtime_full = _to_float(row.get("runtime_cost_us_FULL", ""))
        if runtime_bucket > 0 and runtime_full > 0:
            row["speedup_BUCKET_vs_FULL"] = f"{runtime_full / runtime_bucket:.6f}"
        else:
            row["speedup_BUCKET_vs_FULL"] = ""


def _compute_speedups(out_rows: List[Dict[str, str]], methods: List[str]) -> None:
    target_methods = [m for m in methods if m != "TORCH"]
    cross_speedup_cols: List[str] = []
    if "BUCKET" in methods and "FULL" in methods:
        cross_speedup_cols.append("speedup_BUCKET_vs_FULL")
    for row in out_rows:
        _compute_row_speedups(row, target_methods, cross_speedup_cols)


def _build_display_row_dict(
    row: Dict[str, str], shape: str, cb: str, cf: str,
    cbeq: str | bool, bs: str, fs: str, bof: str, dp: str,
) -> Dict[str, object]:
    r = row.get
    return {
        "shape_id": r("shape_id", ""), "shape": shape,
        "config_bucket": cb, "config_full": cf, "config_bucket_eq_full": cbeq,
        "runtime_torch_us": r("runtime_cost_us_TORCH", ""),
        "runtime_bucket_us": r("runtime_cost_us_BUCKET", ""),
        "runtime_full_us": r("runtime_cost_us_FULL", ""),
        "speedup_bucket_vs_torch": bs, "speedup_full_vs_torch": fs,
        "speedup_bucket_vs_full": bof, "speedup_bucket_vs_full_delta_pct": dp,
    }


def _make_display_row(
    row: Dict[str, str], has_full: bool
) -> Tuple[Dict[str, object], float, float, float]:
    shape = str(row.get("shape", "")).strip() or _format_shape(row)
    cb = row.get("config_desc_BUCKET", "") or row.get("config_id_BUCKET", "")
    cf = row.get("config_desc_FULL", "") or row.get("config_id_FULL", "")
    cbeq = (str(cb).strip() == str(cf).strip()) if (has_full and cb and cf) else ""
    bs = row.get("speedup_BUCKET_vs_TORCH", "")
    fs = row.get("speedup_FULL_vs_TORCH", "")
    bof = row.get("speedup_BUCKET_vs_FULL", "")
    bv, fv, bofv = _to_float(bs), _to_float(fs), _to_float(bof)
    dp = f"{(bofv - 1.0) * 100.0:.6f}%" if bofv > 0 else ""
    d = _build_display_row_dict(row, shape, cb, cf, cbeq, bs, fs, bof, dp)
    return d, bv, fv, bofv


def _build_display_rows(
    out_rows: List[Dict[str, str]], has_full: bool
) -> Tuple[List[Dict[str, object]], List[float], List[float], List[float]]:
    display_rows: List[Dict[str, object]] = []
    bucket_speedup_vals: List[float] = []
    full_speedup_vals: List[float] = []
    bucket_over_full_vals: List[float] = []
    for row in out_rows:
        d, bv, fv, bofv = _make_display_row(row, has_full)
        display_rows.append(d)
        if bv > 0:
            bucket_speedup_vals.append(bv)
        if fv > 0:
            full_speedup_vals.append(fv)
        if bofv > 0:
            bucket_over_full_vals.append(bofv)
    return display_rows, bucket_speedup_vals, full_speedup_vals, bucket_over_full_vals


def _build_geomean_row(
    gm_bucket: float, gm_full: float, gm_bof: float
) -> Dict[str, object]:
    return {
        "shape_id": "speedup_vs_torch",
        "shape": "geomean",
        "config_bucket": "",
        "config_full": "",
        "config_bucket_eq_full": "",
        "runtime_torch_us": "",
        "runtime_bucket_us": "",
        "runtime_full_us": "",
        "speedup_bucket_vs_torch": _fmt(gm_bucket) if gm_bucket > 0 else "",
        "speedup_full_vs_torch": _fmt(gm_full) if gm_full > 0 else "",
        "speedup_bucket_vs_full": _fmt(gm_bof) if gm_bof > 0 else "",
        "speedup_bucket_vs_full_delta_pct": (
            f"{(gm_bof - 1.0) * 100.0:.6f}%" if gm_bof > 0 else ""
        ),
    }


def _add_geomean_row(
    display_rows: List[Dict[str, object]],
    bucket_vals: List[float],
    full_vals: List[float],
    bucket_over_full_vals: List[float],
) -> None:
    gm_bucket = _geometric_mean(bucket_vals)
    gm_full = _geometric_mean(full_vals)
    gm_bof = _geometric_mean(bucket_over_full_vals)
    display_rows.append(_build_geomean_row(gm_bucket, gm_full, gm_bof))


def _get_display_fieldnames(has_full: bool) -> List[str]:
    fieldnames = ["shape_id", "shape", "config_bucket"]
    if has_full:
        fieldnames.extend(["config_full", "config_bucket_eq_full"])
    fieldnames.extend(["runtime_torch_us", "runtime_bucket_us"])
    if has_full:
        fieldnames.append("runtime_full_us")
    fieldnames.append("speedup_bucket_vs_torch")
    if has_full:
        fieldnames.extend(
            ["speedup_full_vs_torch", "speedup_bucket_vs_full", "speedup_bucket_vs_full_delta_pct"]
        )
    return fieldnames


def _write_output_csv(
    out_csv: Path, display_rows: List[Dict[str, object]], fieldnames: List[str]
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(display_rows)


def _maybe_print_table(out_csv: Path, print_table: bool) -> None:
    if not print_table:
        return
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        saved_reader = csv.DictReader(f)
        saved_rows = list(saved_reader)
        saved_headers = list(saved_reader.fieldnames or [])
    _print_section("CASE COMPARE (EVAL)", saved_rows, saved_headers)


def _compute_and_write_case_compare(
    out_csv: Path,
    filtered: List[Dict[str, str]],
    methods: List[str],
    all_columns: List[str],
) -> int:
    shared_cols = _get_shared_cols(all_columns)
    table = _build_table(filtered, methods, shared_cols)
    out_rows = list(table.values())
    _compute_speedups(out_rows, methods)
    out_rows.sort(key=lambda r: _shape_sort_key(str(r.get("shape_id", ""))))
    has_full = "FULL" in methods
    display_rows, bv, fv, bofv = _build_display_rows(out_rows, has_full)
    _add_geomean_row(display_rows, bv, fv, bofv)
    fieldnames = _get_display_fieldnames(has_full)
    _write_output_csv(out_csv, display_rows, fieldnames)
    return len(display_rows)


def compare_case_runtime(
    input_csv: Path,
    *,
    split: str = "eval",
    out_csv: Path | None = None,
    allow_mixed_metric: bool = False,
    print_table: bool = True,
) -> tuple[Path, int]:
    out_csv, all_columns, rows = _load_csv_and_resolve_out_path(input_csv, split, out_csv)
    filtered, methods = _validate_and_get_methods(rows, split)
    timing_by_method_split = _build_timing_by_method_split(rows, methods)
    _check_timing_consistency(timing_by_method_split, methods, split, allow_mixed_metric)
    n_rows = _compute_and_write_case_compare(out_csv, filtered, methods, all_columns)
    _maybe_print_table(out_csv, print_table)
    return out_csv, n_rows
