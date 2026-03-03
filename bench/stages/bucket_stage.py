from __future__ import annotations

from typing import Dict, List

from bench.bucket_tune.runtime import eval_triton_tune_batch
from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.policies.bucket_policy import BucketTunePolicy
from bench.reporting.csv_logger import append_records


def run_bucket(config: BenchmarkConfig, state: BenchmarkState) -> str:
    options = config.options
    op = config.op
    splits = options.bucket_splits
    rows: List[Dict[str, object]] = []
    policy = BucketTunePolicy(
        candidates=state.candidates,
        key_fn=lambda shape: op.eval_key(shape, splits),
        key_to_str=lambda key: op.eval_key_str(key),
    )

    tuned_keys: set = set()
    for idx, shape in enumerate(config.tune_set):
        sel = policy.select(shape, lambda s, cfgs: eval_triton_tune_batch(config, s, cfgs))
        if sel.premeasure is None:
            raise RuntimeError("BUG: tune 阶段缺少 batch premeasure")
        rows.append(op.make_bucket_record(config, "tune", idx, shape, sel, sel.premeasure))
        tuned_keys.add(op.eval_key(shape, splits))

    missing = config.eval_keys - tuned_keys
    if missing:
        raise RuntimeError(f"BUG: 进入 eval 前存在未调过的 key: {sorted(missing)}")

    for idx, shape in enumerate(config.eval_set):
        sel = policy.select(shape, lambda s, cfgs: eval_triton_tune_batch(config, s, cfgs))
        if sel.tune_time_ms > 0:
            raise RuntimeError("BUG: bucket policy 在 eval 阶段发生了调参")
        rows.append(op.make_bucket_record(config, "eval", idx, shape, sel, {}))

    eval_rows = [row for row in rows if row["split"] == "eval"]
    cfg_by_id = {str(getattr(cfg, "config_id", "")): cfg for cfg in state.candidates}
    eval_entries = []
    for row in eval_rows:
        cfg = cfg_by_id.get(str(row["config_id"]))
        if cfg is None:
            raise RuntimeError(f"unknown config_id in eval row: {row.get('config_id')}")
        eval_entries.append(op.build_eval_entry(row, cfg))
    eval_metrics = config.triton_evaluator.evaluate_batch(eval_entries)
    for row, met in zip(eval_rows, eval_metrics):
        row.update(
            {
                "compile_time_ms": float(met.get("compile_time_ms", 0.0)),
                "runtime_cost_us": float(met.get("runtime_cost_us", 0.0)),
                "invalid_config": int(met.get("invalid_config", 0)),
                "notes": ";".join(
                    [x for x in [str(row.get("notes", "")), str(met.get("notes", ""))] if x]
                ),
            }
        )

    append_records(options.results_csv, rows)
    return f"method=BUCKET rows={len(rows)} tuned_keys={len(tuned_keys)} splits={splits}"
