from __future__ import annotations

import csv
import datetime as dt
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

from bench.configs.base_configs import MatmulConfig, get_default_config
from bench.policies.buckets import all_bucket_keys, bucket_key, bucket_key_to_str, representative_shape
from bench.policies.common import BudgetTracker, EvaluatorFn, SelectionResult, autotune_best

Shape = Tuple[int, int, int]
BucketKey = Tuple[int, int, int]

OFFLINE_TABLE_COLUMNS = [
    "timestamp",
    "bucket_m",
    "bucket_n",
    "bucket_k",
    "rep_M",
    "rep_N",
    "rep_K",
    "config_id",
    "tune_time_ms",
    "notes",
]


def build_offline_table(
    path: str,
    candidates: Sequence[MatmulConfig],
    evaluator: EvaluatorFn,
    budget: BudgetTracker,
) -> Dict[BucketKey, str]:
    mapping: Dict[BucketKey, str] = {}
    rows = []

    for key in all_bucket_keys():
        if budget.exceeded():
            break

        rep_shape = representative_shape(key)
        t0 = time.perf_counter()
        best_cfg, _, notes = autotune_best(rep_shape, candidates, evaluator, budget)
        tune_time_ms = (time.perf_counter() - t0) * 1000.0
        mapping[key] = best_cfg.config_id

        rows.append(
            {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "bucket_m": key[0],
                "bucket_n": key[1],
                "bucket_k": key[2],
                "rep_M": rep_shape[0],
                "rep_N": rep_shape[1],
                "rep_K": rep_shape[2],
                "config_id": best_cfg.config_id,
                "tune_time_ms": f"{tune_time_ms:.3f}",
                "notes": notes,
            }
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OFFLINE_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return mapping


def load_offline_table(path: str) -> Dict[BucketKey, str]:
    out: Dict[BucketKey, str] = {}
    p = Path(path)
    if not p.exists():
        return out

    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["bucket_m"]), int(row["bucket_n"]), int(row["bucket_k"]))
            except (KeyError, ValueError):
                continue
            cid = row.get("config_id", "")
            if cid:
                out[key] = cid
    return out


class OfflineTablePolicy:
    method = "D"

    def __init__(self, table: Dict[BucketKey, str], config_map: Dict[str, MatmulConfig]) -> None:
        self.table = table
        self.config_map = config_map

    def select(self, shape: Shape) -> SelectionResult:
        key = bucket_key(*shape)
        key_str = bucket_key_to_str(key)
        cid = self.table.get(key)
        notes = ""

        if cid is None:
            fallback = get_default_config()
            cfg = fallback
            notes = "offline_bucket_miss_fallback"
        else:
            cfg = self.config_map.get(cid, get_default_config())
            if cfg.config_id != cid:
                notes = "offline_config_missing_fallback"

        return SelectionResult(config=cfg, cache_key=key_str, notes=notes)
