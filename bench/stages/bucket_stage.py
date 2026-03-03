from __future__ import annotations

from typing import Dict, List

from bench.bucket_tune.records import make_bucket_record
from bench.bucket_tune.runtime import eval_triton_tune_batch
from bench.bucket_tune.types import BenchmarkConfig, BenchmarkState
from bench.configs.base_configs import CONFIG_MAP
from bench.policies.bucket_autotune import BucketAutotunePolicy, bucket_key
from bench.reporting.csv_logger import append_records


def run_bucket(config: BenchmarkConfig, state: BenchmarkState) -> str:
    options = config.options
    rows: List[Dict[str, object]] = []
    policy = BucketAutotunePolicy(
        state.candidates,
        m_split=options.bucket_m_split,
        n_split=options.bucket_n_split,
        k_split=options.bucket_k_split,
    )

    tuned_keys: set[int] = set()
    for idx, shape in enumerate(config.tune_set):
        sel = policy.select(shape, lambda s, cfgs: eval_triton_tune_batch(config, s, cfgs))
        if sel.premeasure is None:
            raise RuntimeError("BUG: tune 阶段缺少 batch premeasure")
        rows.append(make_bucket_record(config, "tune", idx, shape, sel, sel.premeasure))
        tuned_keys.add(
            bucket_key(
                shape[0],
                shape[1],
                shape[2],
                options.bucket_m_split,
                options.bucket_n_split,
                options.bucket_k_split,
            )
        )

    missing = config.eval_keys - tuned_keys
    if missing:
        raise RuntimeError(f"BUG: 进入 eval 前存在未调过的 key: {sorted(missing)}")

    for idx, shape in enumerate(config.eval_set):
        sel = policy.select(shape, lambda s, cfgs: eval_triton_tune_batch(config, s, cfgs))
        if sel.tune_time_ms > 0:
            raise RuntimeError("BUG: bucket policy 在 eval 阶段发生了调参")
        rows.append(make_bucket_record(config, "eval", idx, shape, sel, {}))

    eval_rows = [row for row in rows if row["split"] == "eval"]
    eval_entries = [
        ((int(row["M"]), int(row["N"]), int(row["K"])), CONFIG_MAP[str(row["config_id"])])
        for row in eval_rows
    ]
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
    return (
        f"method=BUCKET rows={len(rows)} tuned_keys={len(tuned_keys)} "
        f"splits=(M<={options.bucket_m_split},N<={options.bucket_n_split},K<={options.bucket_k_split})"
    )
