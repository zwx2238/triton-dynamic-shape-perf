from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple

from .shapes import bucket_key

Shape = Tuple[int, int, int]


def _shape_id(split: str, idx: int) -> str:
    return f"bucket_torch_{split}_{idx:03d}"


def make_bucket_record(
    config,
    split: str,
    idx: int,
    shape: Shape,
    selection,
    metrics: Dict[str, object],
    method: str = "BUCKET",
) -> Dict[str, object]:
    if method not in {"BUCKET", "FULL"}:
        raise ValueError(f"不支持的 triton method: {method}")

    options = config.options
    splits = options.bucket_splits
    m_split, n_split, k_split = splits[0], splits[1], splits[2]
    m, n, k = shape

    cfg = selection.config
    key = bucket_key(m, n, k, m_split, n_split, k_split)
    notes = []
    if selection.notes:
        notes.append(selection.notes)
    metric_note = str(metrics.get("notes", ""))
    if metric_note:
        notes.append(metric_note)

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": method,
        "workload": "bucket_torch_npu",
        "split": split,
        "shape_id": _shape_id(split, idx),
        "M": m,
        "N": n,
        "K": k,
        "dtype": options.dtype,
        "gpu": config.gpu,
        "config_id": cfg.config_id,
        "BLOCK_M": cfg.BLOCK_M,
        "BLOCK_N": cfg.BLOCK_N,
        "BLOCK_K": cfg.BLOCK_K,
        "compile_time_ms": float(metrics.get("compile_time_ms", 0.0)),
        "tune_time_ms": round(selection.tune_time_ms, 3),
        "runtime_cost_us": float(metrics.get("runtime_cost_us", 0.0)),
        "bucket_key": int(key),
        "cache_key": selection.cache_key,
        "invalid_config": int(metrics.get("invalid_config", 0)),
        "notes": ";".join(notes),
    }


def make_torch_record(
    config,
    split: str,
    idx: int,
    shape: Shape,
    metrics: Dict[str, object],
    config_id: str,
) -> Dict[str, object]:
    options = config.options
    splits = options.bucket_splits
    m_split, n_split, k_split = splits[0], splits[1], splits[2]
    m, n, k = shape
    key = bucket_key(m, n, k, m_split, n_split, k_split)

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "TORCH",
        "workload": "bucket_torch_npu",
        "split": split,
        "shape_id": _shape_id(split, idx),
        "M": m,
        "N": n,
        "K": k,
        "dtype": options.dtype,
        "gpu": config.gpu,
        "config_id": config_id,
        "BLOCK_M": -1,
        "BLOCK_N": -1,
        "BLOCK_K": -1,
        "compile_time_ms": float(metrics.get("compile_time_ms", 0.0)),
        "tune_time_ms": 0.0,
        "runtime_cost_us": float(metrics.get("runtime_cost_us", 0.0)),
        "bucket_key": int(key),
        "cache_key": config_id,
        "invalid_config": int(metrics.get("invalid_config", 0)),
        "notes": str(metrics.get("notes", "")),
    }
