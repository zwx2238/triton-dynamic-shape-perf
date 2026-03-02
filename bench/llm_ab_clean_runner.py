from __future__ import annotations

import argparse
import datetime as dt
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from bench.configs.base_configs import BASE_CONFIGS, CONFIG_MAP, MatmulConfig
from bench.csv_logger import append_records, reset_csv
from bench.kernels.triton_matmul import TritonMatmulEvaluator
from bench.policies.common import BudgetTracker, SelectionResult
from bench.policies.full_autotune import FullAutotunePolicy
from bench.policies.llm_bucket8_autotune import (
    LlmBucket8AutotunePolicy,
    llm_bucket8_key,
    llm_bucket8_key_to_str,
)

Shape = Tuple[int, int, int]
BucketKey8 = Tuple[int, int]

M_BY_BIN = {
    0: [1, 2],
    1: [4, 8],
    2: [16, 32],
    3: [64],
}
K_BY_BIN = {
    0: [1024, 1536, 2048],
    1: [3072, 4096, 6144, 8192],
}
N_CHOICES = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
ALL_BUCKET8_KEYS: List[BucketKey8] = [(m_bin, k_bin) for m_bin in range(4) for k_bin in range(2)]


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


def _sample_shape_for_key(rng: random.Random, key: BucketKey8) -> Shape:
    m_bin, k_bin = key
    m = rng.choice(M_BY_BIN[m_bin])
    k = rng.choice(K_BY_BIN[k_bin])
    n = rng.choice(N_CHOICES)
    return (m, n, k)


def _build_tune_set(rng: random.Random, tune_size: int) -> List[Shape]:
    if tune_size < len(ALL_BUCKET8_KEYS):
        raise RuntimeError(f"tune_size={tune_size} 太小，必须 >= {len(ALL_BUCKET8_KEYS)} 以覆盖全部 8 个 key")

    base = tune_size // len(ALL_BUCKET8_KEYS)
    rem = tune_size % len(ALL_BUCKET8_KEYS)
    out: List[Shape] = []
    for i, key in enumerate(ALL_BUCKET8_KEYS):
        cnt = base + (1 if i < rem else 0)
        for _ in range(cnt):
            out.append(_sample_shape_for_key(rng, key))
    rng.shuffle(out)
    return out


def _build_eval_set(rng: random.Random, eval_size: int) -> List[Shape]:
    out: List[Shape] = []
    for _ in range(eval_size):
        key = rng.choice(ALL_BUCKET8_KEYS)
        out.append(_sample_shape_for_key(rng, key))
    return out


def _shape_id(split: str, idx: int) -> str:
    return f"llm_style_{split}_{idx:03d}"


def _make_record(
    method: str,
    split: str,
    idx: int,
    shape: Shape,
    selection: SelectionResult,
    metrics: Dict[str, object],
    dtype: str,
    gpu: str,
) -> Dict[str, object]:
    M, N, K = shape
    cfg = selection.config
    key = llm_bucket8_key(M, K)
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
        "split": split,
        "shape_id": _shape_id(split, idx),
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
        "bucket_m": key[0],
        "bucket_n": key[1],
        "bucket_k": -1,
        "cache_key": selection.cache_key,
        "invalid_config": int(metrics.get("invalid_config", 0)),
        "notes": ";".join(notes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean llm_style-only benchmark: Plan A vs Plan B")
    parser.add_argument("--methods", nargs="+", default=["A", "B"], choices=["A", "B"])
    parser.add_argument("--candidate-ids", type=str, default=",".join(c.config_id for c in BASE_CONFIGS))
    parser.add_argument("--tune-size", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260302)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--results-csv", type=str, default="results/phase2_llm_clean_ab.csv")
    parser.add_argument("--reset-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng_tune = random.Random(args.seed + 1)
    rng_eval = random.Random(args.seed + 2)

    tune_set = _build_tune_set(rng_tune, args.tune_size)
    eval_set = _build_eval_set(rng_eval, args.eval_size)
    eval_keys = {llm_bucket8_key(s[0], s[2]) for s in eval_set}
    tune_keys = {llm_bucket8_key(s[0], s[2]) for s in tune_set}
    if not eval_keys.issubset(tune_keys):
        raise RuntimeError("BUG: tune key 未覆盖 eval key")

    cfg_ids = _parse_config_ids(args.candidate_ids)
    candidates = _resolve_configs(cfg_ids)

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
        rows: List[Dict[str, object]] = []

        if method == "A":
            policy = FullAutotunePolicy(candidates)
            for i, shape in enumerate(tune_set):
                if budget.exceeded():
                    break
                sel = policy.select(shape, evaluator.evaluate, budget)
                met = sel.premeasure if sel.premeasure is not None else evaluator.evaluate(shape, sel.config)
                rows.append(_make_record("A", "tune", i, shape, sel, met, args.dtype, gpu))
            for i, shape in enumerate(eval_set):
                if budget.exceeded():
                    break
                sel = policy.select(shape, evaluator.evaluate, budget)
                met = sel.premeasure if sel.premeasure is not None else evaluator.evaluate(shape, sel.config)
                rows.append(_make_record("A", "eval", i, shape, sel, met, args.dtype, gpu))
            append_records(args.results_csv, rows)
            print(
                f"[method=A] rows={len(rows)} elapsed_s={budget.elapsed_seconds():.2f} budget_s={args.budget_seconds:.2f}"
            )
            continue

        if method == "B":
            policy = LlmBucket8AutotunePolicy(candidates)
            tuned_keys = set()
            for i, shape in enumerate(tune_set):
                if budget.exceeded():
                    break
                sel = policy.select(shape, evaluator.evaluate, budget)
                met = sel.premeasure if sel.premeasure is not None else evaluator.evaluate(shape, sel.config)
                rows.append(_make_record("B", "tune", i, shape, sel, met, args.dtype, gpu))
                tuned_keys.add(llm_bucket8_key(shape[0], shape[2]))

            missing = eval_keys - tuned_keys
            if missing:
                raise RuntimeError(f"BUG: 进入 eval 前存在未调过的 key: {sorted(missing)}")

            for i, shape in enumerate(eval_set):
                if budget.exceeded():
                    break
                sel = policy.select(shape, evaluator.evaluate, budget)
                if sel.tune_time_ms > 0:
                    raise RuntimeError("BUG: bucket policy 在 eval 阶段发生了调参")
                met = evaluator.evaluate(shape, sel.config)
                rows.append(_make_record("B", "eval", i, shape, sel, met, args.dtype, gpu))

            append_records(args.results_csv, rows)
            print(
                f"[method=B] rows={len(rows)} elapsed_s={budget.elapsed_seconds():.2f} "
                f"budget_s={args.budget_seconds:.2f} tuned_keys={len(tuned_keys)}"
            )
            continue

    print(f"done. results_csv={args.results_csv}")


if __name__ == "__main__":
    main()
