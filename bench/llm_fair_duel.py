from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bench.configs.base_configs import CONFIG_MAP, MatmulConfig
from bench.csv_logger import append_records, reset_csv
from bench.kernels.triton_matmul import TritonMatmulEvaluator
from bench.policies.common import BudgetTracker, SelectionResult
from bench.policies.full_autotune import FullAutotunePolicy
from bench.policies.llm_bucket8_autotune import (
    LlmBucket8AutotunePolicy,
    llm_bucket8_key,
    llm_bucket8_key_to_str,
    llm_k_bin2,
    llm_m_bin4,
)

Shape = Tuple[int, int, int]

# 16 个 llm_style case：覆盖 8 个 bucket key，每个 key 2 个 shape
LLM_CASES_16: List[Shape] = [
    # m_bin=0, k_bin=0
    (1, 1024, 1024),
    (2, 4096, 2048),
    # m_bin=1, k_bin=0
    (4, 1024, 1024),
    (8, 4096, 2048),
    # m_bin=2, k_bin=0
    (16, 2048, 1024),
    (32, 4096, 2048),
    # m_bin=3, k_bin=0
    (64, 1024, 1024),
    (64, 4096, 2048),
    # m_bin=0, k_bin=1
    (1, 2048, 4096),
    (2, 8192, 8192),
    # m_bin=1, k_bin=1
    (4, 2048, 4096),
    (8, 8192, 8192),
    # m_bin=2, k_bin=1
    (16, 4096, 4096),
    (32, 8192, 8192),
    # m_bin=3, k_bin=1
    (64, 2048, 4096),
    (64, 8192, 8192),
]


def _parse_config_ids(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _resolve_configs(ids: Sequence[str]) -> List[MatmulConfig]:
    out: List[MatmulConfig] = []
    for cid in ids:
        cfg = CONFIG_MAP.get(cid)
        if cfg is None:
            raise ValueError(f"未知 config_id: {cid}")
        out.append(cfg)
    return out


def _make_record(
    method: str,
    shape_idx: int,
    shape: Shape,
    selection: SelectionResult,
    metrics: Dict[str, object],
    dtype: str,
    gpu: str,
) -> Dict[str, object]:
    M, N, K = shape
    cfg = selection.config
    m_bin = llm_m_bin4(M)
    k_bin = llm_k_bin2(K)
    notes = []
    if selection.notes:
        notes.append(selection.notes)
    metric_note = str(metrics.get("notes", ""))
    if metric_note:
        notes.append(metric_note)

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": method,
        "workload": "llm_style",
        "split": "eval",
        "shape_id": f"llm16_eval_{shape_idx:02d}",
        "M": M,
        "N": N,
        "K": K,
        "dtype": dtype,
        "gpu": gpu,
        "config_id": cfg.config_id,
        "BLOCK_M": cfg.BLOCK_M,
        "BLOCK_N": cfg.BLOCK_N,
        "BLOCK_K": cfg.BLOCK_K,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
        "GROUP_M": cfg.GROUP_M,
        "compile_time_ms": float(metrics.get("compile_time_ms", 0.0)),
        "tune_time_ms": round(selection.tune_time_ms, 3),
        "runtime_cost_us": float(metrics.get("runtime_cost_us", 0.0)),
        "p99_us": float(metrics.get("p99_us", 0.0)),
        "runtime_perf_tflops": float(metrics.get("runtime_perf_tflops", 0.0)),
        "bucket_m": m_bin,
        "bucket_n": k_bin,
        "bucket_k": -1,
        "cache_key": selection.cache_key,
        "invalid_config": int(metrics.get("invalid_config", 0)),
        "notes": ";".join(notes),
    }


def _validate_cases() -> None:
    if len(LLM_CASES_16) != 16:
        raise RuntimeError(f"case 数量错误: 期望 16，实际 {len(LLM_CASES_16)}")
    keys = {llm_bucket8_key(m, k) for (m, _, k) in LLM_CASES_16}
    if len(keys) != 8:
        raise RuntimeError(f"bucket key 数量错误: 期望 8，实际 {len(keys)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair duel for llm_style: A vs B8")
    parser.add_argument("--methods", nargs="+", default=["A", "B8"], choices=["A", "B8"])
    parser.add_argument("--candidate-ids", type=str, default="c11,c00,c04,c02")
    parser.add_argument("--expect-config-count", type=int, default=4)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--results-csv", type=str, default="results/phase2_llm_fair_duel.csv")
    parser.add_argument("--reset-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_cases()

    cfg_ids = _parse_config_ids(args.candidate_ids)
    candidates = _resolve_configs(cfg_ids)
    if len(candidates) != args.expect_config_count:
        raise RuntimeError(
            f"配置数量不匹配: expect={args.expect_config_count}, actual={len(candidates)}, ids={cfg_ids}"
        )

    if args.reset_results:
        reset_csv(args.results_csv)

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    evaluator = TritonMatmulEvaluator(
        dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    gpu = evaluator.get_gpu_name()

    for method in args.methods:
        budget = BudgetTracker(args.budget_seconds)
        if method == "A":
            policy = FullAutotunePolicy(candidates)
            select_fn = lambda s: policy.select(s, evaluator.evaluate, budget)
        elif method == "B8":
            policy = LlmBucket8AutotunePolicy(candidates)
            select_fn = lambda s: policy.select(s, evaluator.evaluate, budget)
        else:
            raise ValueError(method)

        records: List[Dict[str, object]] = []
        for i, shape in enumerate(LLM_CASES_16):
            if budget.exceeded():
                break
            selection = select_fn(shape)
            metrics = selection.premeasure if selection.premeasure is not None else evaluator.evaluate(shape, selection.config)
            records.append(_make_record(method, i, shape, selection, metrics, args.dtype, gpu))

        append_records(args.results_csv, records)
        keys = sorted({llm_bucket8_key_to_str(llm_bucket8_key(s[0], s[2])) for s in LLM_CASES_16})
        print(
            f"[method={method}] rows={len(records)} budget_s={args.budget_seconds:.2f} "
            f"elapsed_s={budget.elapsed_seconds():.2f} keys={len(keys)}"
        )

    print(f"done. results_csv={args.results_csv}")


if __name__ == "__main__":
    main()
