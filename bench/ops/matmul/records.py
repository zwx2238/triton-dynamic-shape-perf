from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple

from .shapes import bucket_key

Shape = Tuple[int, int, int]


def _shape_id(split: str, idx: int) -> str:
    return f"bucket_torch_{split}_{idx:03d}"


def _bucket_record_notes(selection, metrics: Dict[str, object]) -> str:
    notes = []
    if selection.notes:
        notes.append(selection.notes)
    metric_note = str(metrics.get("notes", ""))
    if metric_note:
        notes.append(metric_note)
    return ";".join(notes)


def _bucket_record_header(
    config, split: str, idx: int, shape: Shape, method: str,
) -> Dict[str, object]:
    m, n, k = shape
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": method, "workload": "bucket_torch_npu",
        "split": split, "shape_id": _shape_id(split, idx),
        "M": m, "N": n, "K": k,
        "dtype": config.options.dtype, "gpu": config.gpu,
    }


def _bucket_record_tail(
    cfg,
    key: int,
    metrics: Dict[str, object],
    selection,
) -> Dict[str, object]:
    return {
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
        "notes": _bucket_record_notes(selection, metrics),
    }


def _bucket_record_core(
    config,
    split: str,
    idx: int,
    shape: Shape,
    cfg,
    key: int,
    metrics: Dict[str, object],
    method: str,
    selection,
) -> Dict[str, object]:
    header = _bucket_record_header(config, split, idx, shape, method)
    tail = _bucket_record_tail(cfg, key, metrics, selection)
    return {**header, **tail}


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
    return _bucket_record_core(config, split, idx, shape, cfg, key, metrics, method, selection)


def _torch_record_fields(
    config, split: str, idx: int, shape: Shape, key: int,
) -> Dict[str, object]:
    m, n, k = shape
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "TORCH", "workload": "bucket_torch_npu",
        "split": split, "shape_id": _shape_id(split, idx),
        "M": m, "N": n, "K": k,
        "dtype": config.options.dtype, "gpu": config.gpu,
        "bucket_key": int(key),
    }


def _apply_torch_metrics(
    base: Dict[str, object],
    metrics: Dict[str, object],
    config_id: str,
) -> Dict[str, object]:
    base["config_id"] = config_id
    base["BLOCK_M"] = base["BLOCK_N"] = base["BLOCK_K"] = -1
    base["compile_time_ms"] = float(metrics.get("compile_time_ms", 0.0))
    base["tune_time_ms"] = 0.0
    base["runtime_cost_us"] = float(metrics.get("runtime_cost_us", 0.0))
    base["cache_key"] = config_id
    base["invalid_config"] = int(metrics.get("invalid_config", 0))
    base["notes"] = str(metrics.get("notes", ""))
    return base


def _torch_record_core(
    config,
    split: str,
    idx: int,
    shape: Shape,
    metrics: Dict[str, object],
    config_id: str,
) -> Dict[str, object]:
    m, n, k = shape
    splits = config.options.bucket_splits
    key = bucket_key(m, n, k, splits[0], splits[1], splits[2])
    base = _torch_record_fields(config, split, idx, shape, key)
    return _apply_torch_metrics(base, metrics, config_id)


def make_torch_record(
    config,
    split: str,
    idx: int,
    shape: Shape,
    metrics: Dict[str, object],
    config_id: str,
) -> Dict[str, object]:
    return _torch_record_core(config, split, idx, shape, metrics, config_id)
