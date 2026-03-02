from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable

CSV_COLUMNS = [
    "timestamp",
    "method",
    "workload",
    "split",
    "shape_id",
    "M",
    "N",
    "K",
    "dtype",
    "gpu",
    "config_id",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "num_warps",
    "num_stages",
    "GROUP_M",
    "compile_time_ms",
    "tune_time_ms",
    "runtime_cost_us",
    "p99_us",
    "runtime_perf_tflops",
    "bucket_m",
    "bucket_n",
    "bucket_k",
    "cache_key",
    "invalid_config",
    "notes",
]


def _normalize(record: Dict[str, object]) -> Dict[str, object]:
    return {k: record.get(k, "") for k in CSV_COLUMNS}


def append_records(path: str, records: Iterable[Dict[str, object]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0

    count = 0
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(_normalize(record))
            count += 1
    return count


def reset_csv(path: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
