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


def _load_case_compare_rows(compare_csv: Path) -> List[Dict[str, str]] | None:
    if not compare_csv.exists():
        print(f"\n=== CASE COMPARE (EVAL) ===\n(missing file: {compare_csv})")
        return None
    with compare_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_case_compare_column_info(headers: List[str]) -> Tuple[bool, str, str, str]:
    has_full = ("runtime_cost_us_FULL" in headers) or ("config_id_FULL" in headers)
    bucket_speedup_col = "speedup_BUCKET_vs_TORCH" if "speedup_BUCKET_vs_TORCH" in headers else ""
    full_speedup_col = "speedup_FULL_vs_TORCH" if "speedup_FULL_vs_TORCH" in headers else ""
    bucket_over_full_col = "speedup_BUCKET_vs_FULL" if "speedup_BUCKET_vs_FULL" in headers else ""
    return has_full, bucket_speedup_col, full_speedup_col, bucket_over_full_col


def _extract_shape_from_row(row: Dict[str, str]) -> str:
    shape = str(row.get("shape", "")).strip()
    if not shape:
        m = str(row.get("M", "")).strip()
        n = str(row.get("N", "")).strip()
        k = str(row.get("K", "")).strip()
        shape = f"{m}x{n}x{k}" if m and n and k else ""
    return shape


def _parse_speedup_float(text: object) -> float:
    if text is None or text == "":
        return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _make_case_compare_display_row(
    row: Dict[str, str], shape: str,
    config_bucket: object, config_full: object, config_bucket_eq_full: object,
    bucket_speedup_text: object, full_speedup_text: object, bucket_over_full_text: object,
    bucket_over_full_delta_pct_text: str,
) -> Dict[str, object]:
    return {
        "shape_id": row.get("shape_id", ""), "shape": shape,
        "config_bucket": config_bucket, "config_full": config_full,
        "config_bucket_eq_full": config_bucket_eq_full,
        "runtime_torch_us": row.get("runtime_cost_us_TORCH", ""),
        "runtime_bucket_us": row.get("runtime_cost_us_BUCKET", ""),
        "runtime_full_us": row.get("runtime_cost_us_FULL", ""),
        "speedup_bucket_vs_torch": bucket_speedup_text,
        "speedup_full_vs_torch": full_speedup_text,
        "speedup_bucket_vs_full": bucket_over_full_text,
        "speedup_bucket_vs_full_delta_pct": bucket_over_full_delta_pct_text,
    }


def _get_case_compare_config_eq(has_full: bool, config_bucket: object, config_full: object) -> object:
    if not (has_full and config_bucket and config_full):
        return ""
    return str(config_bucket).strip() == str(config_full).strip()


def _get_case_compare_speedup_texts(
    row: Dict[str, str], bucket_col: str, full_col: str, bucket_over_full_col: str,
) -> Tuple[object, object, object]:
    bt = row.get(bucket_col, "") if bucket_col else ""
    ft = row.get(full_col, "") if full_col else ""
    bot = row.get(bucket_over_full_col, "") if bucket_over_full_col else ""
    return bt, ft, bot


def _build_case_compare_display_row_with_values(
    row: Dict[str, str],
    shape: str,
    config_bucket: object,
    config_full: object,
    config_bucket_eq_full: object,
    bucket_speedup_text: object,
    full_speedup_text: object,
    bucket_over_full_text: object,
) -> Tuple[Dict[str, object], float, float, float]:
    bv = _parse_speedup_float(bucket_speedup_text)
    fv = _parse_speedup_float(full_speedup_text)
    bov = _parse_speedup_float(bucket_over_full_text)
    delta_pct = f"{(bov - 1.0) * 100.0:.6f}%" if bov > 0 else ""
    drow = _make_case_compare_display_row(
        row, shape, config_bucket, config_full, config_bucket_eq_full,
        bucket_speedup_text, full_speedup_text, bucket_over_full_text, delta_pct,
    )
    return drow, bv, fv, bov


def _process_single_case_compare_row(
    row: Dict[str, str],
    has_full: bool,
    bucket_speedup_col: str,
    full_speedup_col: str,
    bucket_over_full_col: str,
) -> Tuple[Dict[str, object], float, float, float]:
    shape = _extract_shape_from_row(row)
    config_bucket = row.get("config_desc_BUCKET", "") or row.get("config_id_BUCKET", "")
    config_full = row.get("config_desc_FULL", "") or row.get("config_id_FULL", "")
    config_bucket_eq_full = _get_case_compare_config_eq(has_full, config_bucket, config_full)
    bt, ft, bot = _get_case_compare_speedup_texts(
        row, bucket_speedup_col, full_speedup_col, bucket_over_full_col
    )
    return _build_case_compare_display_row_with_values(
        row, shape, config_bucket, config_full, config_bucket_eq_full, bt, ft, bot
    )


def _make_case_compare_geomean_row(
    gm_bucket_speedup: float,
    gm_full_speedup: float,
    gm_bucket_over_full: float,
) -> Dict[str, object]:
    delta_pct = f"{(gm_bucket_over_full - 1.0) * 100.0:.6f}%" if gm_bucket_over_full > 0 else ""
    return {
        "shape_id": "speedup_vs_torch", "shape": "geomean",
        "config_bucket": "", "config_full": "", "config_bucket_eq_full": "",
        "runtime_torch_us": "", "runtime_bucket_us": "", "runtime_full_us": "",
        "speedup_bucket_vs_torch": _fmt(gm_bucket_speedup) if gm_bucket_speedup > 0 else "",
        "speedup_full_vs_torch": _fmt(gm_full_speedup) if gm_full_speedup > 0 else "",
        "speedup_bucket_vs_full": _fmt(gm_bucket_over_full) if gm_bucket_over_full > 0 else "",
        "speedup_bucket_vs_full_delta_pct": delta_pct,
    }


def _append_positive_vals(
    bucket_vals: List[float],
    full_vals: List[float],
    bucket_over_full_vals: List[float],
    bv: float, fv: float, bov: float,
) -> None:
    if bv > 0:
        bucket_vals.append(bv)
    if fv > 0:
        full_vals.append(fv)
    if bov > 0:
        bucket_over_full_vals.append(bov)


def _aggregate_case_compare_rows(
    rows: List[Dict[str, str]],
    has_full: bool,
    bucket_speedup_col: str,
    full_speedup_col: str,
    bucket_over_full_col: str,
) -> Tuple[List[Dict[str, object]], List[float], List[float], List[float]]:
    display_rows, bucket_vals, full_vals, bucket_over_full_vals = [], [], [], []
    for row in rows:
        drow, bv, fv, bov = _process_single_case_compare_row(
            row, has_full, bucket_speedup_col, full_speedup_col, bucket_over_full_col
        )
        display_rows.append(drow)
        _append_positive_vals(bucket_vals, full_vals, bucket_over_full_vals, bv, fv, bov)
    return display_rows, bucket_vals, full_vals, bucket_over_full_vals


def _build_case_compare_display_rows(
    rows: List[Dict[str, str]],
    has_full: bool,
    bucket_speedup_col: str,
    full_speedup_col: str,
    bucket_over_full_col: str,
) -> List[Dict[str, object]]:
    display_rows, bv, fv, bov = _aggregate_case_compare_rows(
        rows, has_full, bucket_speedup_col, full_speedup_col, bucket_over_full_col
    )
    display_rows.sort(key=lambda r: _shape_sort_key(str(r.get("shape_id", ""))))
    display_rows.append(_make_case_compare_geomean_row(
        _geometric_mean(bv), _geometric_mean(fv), _geometric_mean(bov)
    ))
    return display_rows


def _build_case_compare_headers(has_full: bool) -> List[str]:
    headers = ["shape_id", "shape", "config_bucket"]
    if has_full:
        headers.extend(["config_full", "config_bucket_eq_full"])
    headers.extend(["runtime_torch_us", "runtime_bucket_us"])
    if has_full:
        headers.append("runtime_full_us")
    headers.append("speedup_bucket_vs_torch")
    if has_full:
        headers.extend(
            ["speedup_full_vs_torch", "speedup_bucket_vs_full", "speedup_bucket_vs_full_delta_pct"]
        )
    return headers


def _print_case_compare(compare_csv: Path) -> None:
    rows = _load_case_compare_rows(compare_csv)
    if rows is None:
        return
    if not rows:
        _print_section("CASE COMPARE (EVAL)", [], [])
        return

    headers = list(rows[0].keys())
    has_full, bucket_col, full_col, bucket_over_full_col = _get_case_compare_column_info(headers)
    display_rows = _build_case_compare_display_rows(
        rows, has_full, bucket_col, full_col, bucket_over_full_col
    )
    display_headers = _build_case_compare_headers(has_full)
    _print_section("CASE COMPARE (EVAL)", display_rows, display_headers)


def _partition_rows_by_split(
    rows: List[Dict[str, str]],
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], Dict[Tuple[str, int], List[Dict[str, str]]]]:
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
    return eval_by_method, tune_by_method, eval_by_method_bucket


def _make_overall_row(method: str, runtimes: List[float]) -> Dict[str, object]:
    if not runtimes:
        return {"method": method, "samples": 0, "median_runtime_us": "", "mean_runtime_us": "", "speedup_vs_torch": ""}
    return {
        "method": method, "samples": len(runtimes),
        "median_runtime_us": _fmt(statistics.median(runtimes)),
        "mean_runtime_us": _fmt(statistics.fmean(runtimes)),
        "speedup_vs_torch": "",
    }


def _build_overall_rows(
    methods: List[str],
    eval_by_method: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    overall_rows: List[Dict[str, object]] = []
    for method in methods:
        runtimes = [_to_float(r, "runtime_cost_us") for r in eval_by_method.get(method, [])]
        overall_rows.append(_make_overall_row(method, runtimes))
    return overall_rows


def _compute_torch_median_by_shape(
    eval_by_method: Dict[str, List[Dict[str, str]]],
) -> Dict[str, float]:
    torch_runtime_by_shape: Dict[str, List[float]] = defaultdict(list)
    for row in eval_by_method.get("TORCH", []):
        shape_id = str(row.get("shape_id", "")).strip()
        runtime = _to_float(row, "runtime_cost_us")
        if shape_id and runtime > 0:
            torch_runtime_by_shape[shape_id].append(runtime)
    return {
        shape_id: statistics.median(vals)
        for shape_id, vals in torch_runtime_by_shape.items()
        if vals
    }


def _compute_method_runtime_by_shape(
    methods: List[str],
    eval_by_method: Dict[str, List[Dict[str, str]]],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for method in methods:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in eval_by_method.get(method, []):
            shape_id = str(row.get("shape_id", "")).strip()
            runtime = _to_float(row, "runtime_cost_us")
            if shape_id and runtime > 0:
                grouped[shape_id].append(runtime)
        result[method] = {
            shape_id: statistics.median(vals)
            for shape_id, vals in grouped.items()
            if vals
        }
    return result


def _compute_speedup_vs_torch(
    torch_median_by_shape: Dict[str, float],
    runtime_by_shape: Dict[str, float],
) -> str:
    ratios = [
        tr / cr for sh, tr in torch_median_by_shape.items()
        for cr in [runtime_by_shape.get(sh, 0.0)] if tr > 0 and cr > 0
    ]
    return _fmt(_geometric_mean(ratios)) if ratios else ""


def _fill_speedup_vs_torch(
    overall_rows: List[Dict[str, object]],
    methods: List[str],
    torch_median_by_shape: Dict[str, float],
    method_runtime_by_shape: Dict[str, Dict[str, float]],
) -> None:
    overall_row_by_method = {str(r.get("method", "")): r for r in overall_rows}
    for method in methods:
        row = overall_row_by_method.get(method)
        if row is None:
            continue
        if method == "TORCH":
            row["speedup_vs_torch"] = _fmt(1.0) if row.get("samples", 0) else ""
            continue
        row["speedup_vs_torch"] = _compute_speedup_vs_torch(
            torch_median_by_shape, method_runtime_by_shape.get(method, {})
        )


def _build_tune_rows(
    methods: List[str],
    tune_by_method: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    tune_rows: List[Dict[str, object]] = []
    for method in methods:
        this_rows = tune_by_method.get(method, [])
        tune_times = [_to_float(r, "tune_time_ms") for r in this_rows]
        compile_times = [_to_float(r, "compile_time_ms") for r in this_rows]
        tune_rows.append({
            "method": method,
            "tune_rows": len(this_rows),
            "tuned_nonzero": sum(1 for v in tune_times if v > 0),
            "total_tune_ms": _fmt(sum(tune_times)),
            "total_compile_ms": _fmt(sum(compile_times)),
        })
    return tune_rows


def _build_by_bucket_rows(
    eval_by_method_bucket: Dict[Tuple[str, int], List[Dict[str, str]]],
) -> List[Dict[str, object]]:
    by_bucket_rows: List[Dict[str, object]] = []
    for method, bk in sorted(eval_by_method_bucket.keys(), key=lambda x: (x[0], x[1])):
        bucket_rows = eval_by_method_bucket[(method, bk)]
        runtimes = [_to_float(r, "runtime_cost_us") for r in bucket_rows]
        by_bucket_rows.append({
            "method": method,
            "bucket_key": bk,
            "samples": len(bucket_rows),
            "median_runtime_us": _fmt(statistics.median(runtimes) if runtimes else 0.0),
        })
    return by_bucket_rows


def _write_summary_csvs(
    out_dir: Path,
    prefix: str,
    overall_rows: List[Dict[str, object]],
    tune_rows: List[Dict[str, object]],
    by_bucket_rows: List[Dict[str, object]],
) -> Tuple[Path, Path, Path]:
    specs = [
        (f"{prefix}_summary_overall.csv", ["method", "samples", "median_runtime_us", "mean_runtime_us", "speedup_vs_torch"], overall_rows),
        (f"{prefix}_summary_tune.csv", ["method", "tune_rows", "tuned_nonzero", "total_tune_ms", "total_compile_ms"], tune_rows),
        (f"{prefix}_summary_by_bucket.csv", ["method", "bucket_key", "samples", "median_runtime_us"], by_bucket_rows),
    ]
    paths = []
    for name, fieldnames, rows in specs:
        p = out_dir / name
        _write_csv(p, fieldnames, rows)
        paths.append(p)
    return (paths[0], paths[1], paths[2])


def _prepare_summary_data(input_csv: Path) -> Tuple[
    Dict[str, List[Dict[str, str]]],
    Dict[str, List[Dict[str, str]]],
    Dict[Tuple[str, int], List[Dict[str, str]]],
    List[str],
]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"输入 CSV 为空: {input_csv}")

    eval_by_method, tune_by_method, eval_by_method_bucket = _partition_rows_by_split(rows)
    method_order = {"TORCH": 0, "BUCKET": 1, "FULL": 2}
    methods = sorted(
        set(eval_by_method.keys()) | set(tune_by_method.keys()),
        key=lambda x: (method_order.get(x, 99), x),
    )
    return eval_by_method, tune_by_method, eval_by_method_bucket, methods


def _compute_summary_rows(
    eval_by_method: Dict[str, List[Dict[str, str]]],
    tune_by_method: Dict[str, List[Dict[str, str]]],
    eval_by_method_bucket: Dict[Tuple[str, int], List[Dict[str, str]]],
    methods: List[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    overall_rows = _build_overall_rows(methods, eval_by_method)
    torch_median_by_shape = _compute_torch_median_by_shape(eval_by_method)
    method_runtime_by_shape = _compute_method_runtime_by_shape(methods, eval_by_method)
    _fill_speedup_vs_torch(
        overall_rows, methods, torch_median_by_shape, method_runtime_by_shape
    )
    tune_rows = _build_tune_rows(methods, tune_by_method)
    by_bucket_rows = _build_by_bucket_rows(eval_by_method_bucket)
    return overall_rows, tune_rows, by_bucket_rows


def _print_summary_output(tune_rows: List[Dict[str, object]], compare_csv: Path | None) -> None:
    _print_section(
        "TUNE COST", tune_rows,
        ["method", "tune_rows", "tuned_nonzero", "total_tune_ms", "total_compile_ms"],
    )
    if compare_csv is not None:
        _print_case_compare(compare_csv)


def _execute_summary(
    eval_by_method: Dict[str, List[Dict[str, str]]],
    tune_by_method: Dict[str, List[Dict[str, str]]],
    eval_by_method_bucket: Dict[Tuple[str, int], List[Dict[str, str]]],
    methods: List[str],
    out_dir: Path,
    prefix: str,
    compare_csv: Path | None,
) -> Tuple[Path, Path, Path]:
    overall_rows, tune_rows, by_bucket_rows = _compute_summary_rows(
        eval_by_method, tune_by_method, eval_by_method_bucket, methods
    )
    overall_path, tune_path, bucket_path = _write_summary_csvs(
        out_dir, prefix, overall_rows, tune_rows, by_bucket_rows
    )
    _print_summary_output(tune_rows, compare_csv)
    return overall_path, tune_path, bucket_path


def summarize(input_csv: Path, out_dir: Path, prefix: str, compare_csv: Path | None = None) -> Tuple[Path, Path, Path]:
    eval_by_method, tune_by_method, eval_by_method_bucket, methods = _prepare_summary_data(input_csv)
    return _execute_summary(
        eval_by_method, tune_by_method, eval_by_method_bucket, methods,
        out_dir, prefix, compare_csv,
    )
