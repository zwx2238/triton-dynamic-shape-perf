from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bench.configs.base_configs import BASE_CONFIGS, CONFIG_MAP, MatmulConfig
from bench.csv_logger import append_records, reset_csv
from bench.kernels.torch_matmul import TorchMatmulEvaluator
from bench.kernels.triton_matmul import TritonMatmulEvaluator
from bench.policies.common import BudgetTracker, SelectionResult, autotune_best
from bench.policies.full_autotune import FullAutotunePolicy
from bench.policies.llm_bucket8_autotune import (
    DEFAULT_K_SPLIT,
    DEFAULT_M_SPLIT,
    DEFAULT_N_SPLIT,
    LlmBucket8AutotunePolicy,
    llm_bucket8_key,
)

Shape = Tuple[int, int, int]
BucketKey8 = int
DistributionKey = Tuple[int, int]

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
M_CHOICES = [1, 2, 4, 8, 16, 32, 64]
N_CHOICES = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
K_CHOICES = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
ALL_BUCKET8_KEYS: List[BucketKey8] = list(range(8))
ALL_DISTRIBUTION_KEYS: List[DistributionKey] = [(m_bin, k_bin) for m_bin in range(4) for k_bin in range(2)]


def _build_bucket_candidates(m_split: int, n_split: int, k_split: int) -> Dict[BucketKey8, List[Shape]]:
    out: Dict[BucketKey8, List[Shape]] = defaultdict(list)
    for m in M_CHOICES:
        for n in N_CHOICES:
            for k in K_CHOICES:
                out[llm_bucket8_key(m, n, k, m_split, n_split, k_split)].append((m, n, k))
    missing = [key for key in ALL_BUCKET8_KEYS if not out.get(key)]
    if missing:
        raise RuntimeError(f"BUG: 新分桶存在不可达 key: {missing}")
    return dict(out)


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


def _sample_shape_from_distribution_key(rng: random.Random, key: DistributionKey) -> Shape:
    m_bin, k_bin = key
    m = rng.choice(M_BY_BIN[m_bin])
    n = rng.choice(N_CHOICES)
    k = rng.choice(K_BY_BIN[k_bin])
    return (m, n, k)


def _sample_shape_for_bucket_key(
    rng: random.Random,
    key: BucketKey8,
    bucket_candidates: Dict[BucketKey8, List[Shape]],
) -> Shape:
    return rng.choice(bucket_candidates[key])


def _ensure_bucket_key_coverage(
    rng: random.Random,
    shapes: List[Shape],
    bucket_candidates: Dict[BucketKey8, List[Shape]],
    m_split: int,
    n_split: int,
    k_split: int,
) -> List[Shape]:
    out = list(shapes)
    positions_by_key: Dict[int, List[int]] = defaultdict(list)
    for idx, (m, n, k) in enumerate(out):
        key = llm_bucket8_key(m, n, k, m_split, n_split, k_split)
        positions_by_key[key].append(idx)

    missing = [key for key in ALL_BUCKET8_KEYS if key not in positions_by_key]
    if not missing:
        return out

    # 只替换“冗余 key”的样本，避免把唯一覆盖的 key 换掉。
    for miss_key in missing:
        donor_key = None
        donor_count = -1
        for key, positions in positions_by_key.items():
            if len(positions) > 1 and len(positions) > donor_count:
                donor_key = key
                donor_count = len(positions)

        if donor_key is None:
            raise RuntimeError("BUG: 无法在不破坏已有覆盖的前提下补齐 bucket key 覆盖")

        replace_pos = positions_by_key[donor_key].pop()
        out[replace_pos] = _sample_shape_for_bucket_key(rng, miss_key, bucket_candidates)
        positions_by_key[miss_key].append(replace_pos)

    return out


def _build_tune_set(
    rng: random.Random,
    tune_size: int,
    bucket_candidates: Dict[BucketKey8, List[Shape]],
    m_split: int,
    n_split: int,
    k_split: int,
) -> List[Shape]:
    if tune_size < len(ALL_BUCKET8_KEYS):
        raise RuntimeError(f"tune_size={tune_size} 太小，必须 >= {len(ALL_BUCKET8_KEYS)} 以覆盖全部 8 个 key")

    base = tune_size // len(ALL_DISTRIBUTION_KEYS)
    rem = tune_size % len(ALL_DISTRIBUTION_KEYS)
    out: List[Shape] = []
    for i, key in enumerate(ALL_DISTRIBUTION_KEYS):
        cnt = base + (1 if i < rem else 0)
        for _ in range(cnt):
            out.append(_sample_shape_from_distribution_key(rng, key))
    out = _ensure_bucket_key_coverage(rng, out, bucket_candidates, m_split, n_split, k_split)
    rng.shuffle(out)
    return out


def _build_eval_set(rng: random.Random, eval_size: int) -> List[Shape]:
    out: List[Shape] = []
    for _ in range(eval_size):
        key = rng.choice(ALL_DISTRIBUTION_KEYS)
        out.append(_sample_shape_from_distribution_key(rng, key))
    return out


def _pick_representative_shape(shapes: Sequence[Shape]) -> Shape:
    if not shapes:
        raise RuntimeError("空 shape 集合，无法选择代表 shape")

    log_m = [math.log2(max(1, s[0])) for s in shapes]
    log_n = [math.log2(max(1, s[1])) for s in shapes]
    log_k = [math.log2(max(1, s[2])) for s in shapes]
    log_r = [math.log2(max(1e-9, s[1] / s[2])) for s in shapes]

    med_m = statistics.median(log_m)
    med_n = statistics.median(log_n)
    med_k = statistics.median(log_k)
    med_r = statistics.median(log_r)

    def _score(shape: Shape) -> float:
        m, n, k = shape
        lm = math.log2(max(1, m))
        ln = math.log2(max(1, n))
        lk = math.log2(max(1, k))
        lr = math.log2(max(1e-9, n / k))
        return (lm - med_m) ** 2 + (ln - med_n) ** 2 + (lk - med_k) ** 2 + 0.5 * (lr - med_r) ** 2

    return min(shapes, key=_score)


def _shape_feature(shape: Shape) -> tuple[float, float, float, float]:
    m, n, k = shape
    lm = math.log2(max(1, m))
    ln = math.log2(max(1, n))
    lk = math.log2(max(1, k))
    lr = math.log2(max(1e-9, n / k))
    return lm, ln, lk, lr


def _feature_dist2(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _pick_typical_shapes(shapes: Sequence[Shape], count: int) -> List[Shape]:
    if count <= 0:
        return []
    unique_shapes = list(dict.fromkeys(shapes))
    if not unique_shapes:
        raise RuntimeError("空 shape 集合，无法选择典型 shape")
    if count >= len(unique_shapes):
        return unique_shapes

    feats = {s: _shape_feature(s) for s in unique_shapes}
    first = _pick_representative_shape(unique_shapes)
    selected: List[Shape] = [first]
    selected_set = {first}

    while len(selected) < count:
        best_shape: Optional[Shape] = None
        best_score = -1.0
        for cand in unique_shapes:
            if cand in selected_set:
                continue
            dist_to_selected = min(_feature_dist2(feats[cand], feats[s]) for s in selected)
            if best_shape is None or dist_to_selected > best_score:
                best_shape = cand
                best_score = dist_to_selected
        if best_shape is None:
            break
        selected.append(best_shape)
        selected_set.add(best_shape)
    return selected


def _rank_configs_by_metrics(
    candidates: Sequence[MatmulConfig],
    metrics_seq: Sequence[Dict[str, object]],
) -> List[tuple[MatmulConfig, Dict[str, object], float]]:
    ranked: List[tuple[MatmulConfig, Dict[str, object], float]] = []
    for cfg, met in zip(candidates, metrics_seq):
        invalid = int(met.get("invalid_config", 0))
        us = float(met.get("runtime_cost_us", 0.0))
        score = us if invalid == 0 and us > 0.0 else 1e30
        ranked.append((cfg, met, score))
    ranked.sort(key=lambda x: x[2])
    return ranked


def _derive_candidate_pool_from_typical_shapes(
    typical_shapes: Sequence[Shape],
    candidates: Sequence[MatmulConfig],
    eval_one: Callable[[Shape, MatmulConfig], Dict[str, object]],
    eval_batch: Optional[Callable[[Shape, Sequence[MatmulConfig]], Sequence[Dict[str, object]]]] = None,
) -> tuple[List[MatmulConfig], List[Dict[str, object]]]:
    per_shape_ranked: List[List[tuple[MatmulConfig, Dict[str, object], float]]] = []
    report_rows: List[Dict[str, object]] = []

    for shape in typical_shapes:
        if eval_batch is not None:
            metrics_seq = list(eval_batch(shape, candidates))
            if len(metrics_seq) != len(candidates):
                metrics_seq = [eval_one(shape, cfg) for cfg in candidates]
        else:
            metrics_seq = [eval_one(shape, cfg) for cfg in candidates]
        ranked = _rank_configs_by_metrics(candidates, metrics_seq)
        per_shape_ranked.append(ranked)

        if ranked:
            top_cfg, top_met, top_score = ranked[0]
            report_rows.append(
                {
                    "shape": f"{shape[0]}x{shape[1]}x{shape[2]}",
                    "picked_config_id": top_cfg.config_id,
                    "BLOCK_M": top_cfg.BLOCK_M,
                    "BLOCK_N": top_cfg.BLOCK_N,
                    "BLOCK_K": top_cfg.BLOCK_K,
                    "runtime_cost_us": float(top_met.get("runtime_cost_us", 0.0)),
                    "invalid_config": int(top_met.get("invalid_config", 0)),
                    "score": top_score,
                    "notes": str(top_met.get("notes", "")),
                }
            )

    chosen: List[MatmulConfig] = []
    chosen_ids: set[str] = set()

    for ranked in per_shape_ranked:
        if not ranked:
            continue
        cfg = ranked[0][0]
        if cfg.config_id not in chosen_ids:
            chosen.append(cfg)
            chosen_ids.add(cfg.config_id)

    return chosen, report_rows


def _write_prototype_report_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape",
        "picked_config_id",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "runtime_cost_us",
        "invalid_config",
        "score",
        "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    m_split: int,
    n_split: int,
    k_split: int,
) -> Dict[str, object]:
    M, N, K = shape
    cfg = selection.config
    key = llm_bucket8_key(M, N, K, m_split, n_split, k_split)
    bucket_m = int(key)
    bucket_n = -1
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
        "bucket_m": bucket_m,
        "bucket_n": bucket_n,
        "bucket_k": -1,
        "cache_key": selection.cache_key,
        "invalid_config": int(metrics.get("invalid_config", 0)),
        "notes": ";".join(notes),
    }


def _make_torch_record(
    split: str,
    idx: int,
    shape: Shape,
    metrics: Dict[str, object],
    dtype: str,
    gpu: str,
    config_id: str,
    m_split: int,
    n_split: int,
    k_split: int,
) -> Dict[str, object]:
    M, N, K = shape
    key = llm_bucket8_key(M, N, K, m_split, n_split, k_split)
    metric_note = str(metrics.get("notes", ""))

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "TORCH",
        "workload": "llm_style",
        "split": split,
        "shape_id": _shape_id(split, idx),
        "M": M,
        "N": N,
        "K": K,
        "dtype": dtype,
        "gpu": gpu,
        "config_id": config_id,
        "BLOCK_M": -1,
        "BLOCK_N": -1,
        "BLOCK_K": -1,
        "num_warps": -1,
        "num_stages": -1,
        "GROUP_M": -1,
        "compile_time_ms": float(metrics.get("compile_time_ms", 0.0)),
        "tune_time_ms": 0.0,
        "runtime_cost_us": float(metrics.get("runtime_cost_us", 0.0)),
        "p99_us": float(metrics.get("p99_us", 0.0)),
        "runtime_perf_tflops": float(metrics.get("runtime_perf_tflops", 0.0)),
        "bucket_m": int(key),
        "bucket_n": -1,
        "bucket_k": -1,
        "cache_key": config_id,
        "invalid_config": int(metrics.get("invalid_config", 0)),
        "notes": metric_note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="llm_style benchmark: FULL/BUCKET/ONEKEY(Triton) + TORCH(torch.mm)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["BUCKET", "TORCH"],
        choices=["FULL", "BUCKET", "TORCH", "ONEKEY"],
    )
    parser.add_argument("--candidate-ids", type=str, default=",".join(c.config_id for c in BASE_CONFIGS))
    parser.add_argument(
        "--prototype-count",
        type=int,
        default=4,
        help="先选 N 个典型 shape，并为每个 shape 找到最优 config，作为后续 tune 的候选池；0 表示关闭",
    )
    parser.add_argument(
        "--prototype-report-csv",
        type=str,
        default="",
        help="典型 shape -> 最优 config 报告 CSV 路径；为空时自动放到 results 旁边",
    )
    parser.add_argument(
        "--prototype-only",
        action="store_true",
        help="只运行 prototype 选配置并输出报告，不执行 BUCKET/TORCH/FULL/ONEKEY 主流程",
    )
    parser.add_argument("--tune-size", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260302)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="auto", help="auto / cuda[:id] / npu[:id]")
    parser.add_argument(
        "--npu-timing",
        type=str,
        default="profiler_eval",
        choices=["event", "profiler_eval", "profiler_all"],
        help="NPU 计时方式: event / profiler_eval(仅 eval 用 profiler) / profiler_all",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--bucket-m-split", type=int, default=DEFAULT_M_SPLIT)
    parser.add_argument("--bucket-n-split", type=int, default=DEFAULT_N_SPLIT)
    parser.add_argument("--bucket-k-split", type=int, default=DEFAULT_K_SPLIT)
    parser.add_argument(
        "--disable-onekey-torch-guard",
        action="store_true",
        help="禁用 ONEKEY 代表 shape 必须优于 torch 的门槛（默认启用）",
    )
    parser.add_argument(
        "--onekey-min-speedup-vs-torch",
        type=float,
        default=0.0,
        help="ONEKEY 代表 shape 门槛: torch_us / triton_us >= 该值（默认 0：仅记录不拦截）",
    )
    parser.add_argument("--results-csv", type=str, default="results/llm_full_vs_bucket.csv")
    parser.add_argument("--reset-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng_tune = random.Random(args.seed + 1)
    rng_eval = random.Random(args.seed + 2)
    bucket_candidates = _build_bucket_candidates(args.bucket_m_split, args.bucket_n_split, args.bucket_k_split)

    tune_set = _build_tune_set(
        rng_tune,
        args.tune_size,
        bucket_candidates,
        args.bucket_m_split,
        args.bucket_n_split,
        args.bucket_k_split,
    )
    eval_set = _build_eval_set(rng_eval, args.eval_size)
    eval_keys = {llm_bucket8_key(s[0], s[1], s[2], args.bucket_m_split, args.bucket_n_split, args.bucket_k_split) for s in eval_set}
    tune_keys = {llm_bucket8_key(s[0], s[1], s[2], args.bucket_m_split, args.bucket_n_split, args.bucket_k_split) for s in tune_set}
    if not eval_keys.issubset(tune_keys):
        raise RuntimeError("BUG: tune key 未覆盖 eval key")

    cfg_ids = _parse_config_ids(args.candidate_ids)
    candidates = _resolve_configs(cfg_ids)

    if args.reset_results:
        reset_csv(args.results_csv)
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)

    need_triton = any(method in {"FULL", "BUCKET", "ONEKEY"} for method in args.methods)
    onekey_guard_enabled = ("ONEKEY" in args.methods) and (not args.disable_onekey_torch_guard)
    need_torch = ("TORCH" in args.methods) or onekey_guard_enabled

    triton_evaluator: Optional[TritonMatmulEvaluator] = None
    torch_evaluator: Optional[TorchMatmulEvaluator] = None

    if need_triton:
        triton_evaluator = TritonMatmulEvaluator(
            dtype=args.dtype,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            npu_timing="event",
        )

    if need_torch:
        torch_evaluator = TorchMatmulEvaluator(
            dtype=args.dtype,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            npu_timing="event",
        )

    if triton_evaluator is not None:
        gpu = triton_evaluator.get_gpu_name()
    elif torch_evaluator is not None:
        gpu = torch_evaluator.get_gpu_name()
    else:
        raise RuntimeError("methods 为空，至少需要一个方法")

    device_type = triton_evaluator.device.type if triton_evaluator is not None else torch_evaluator.device.type

    if device_type == "npu":
        if args.npu_timing == "event":
            tune_timing_mode = "event"
            eval_timing_mode = "event"
        elif args.npu_timing == "profiler_all":
            tune_timing_mode = "profiler"
            eval_timing_mode = "profiler"
        else:
            tune_timing_mode = "event"
            eval_timing_mode = "profiler"
    else:
        tune_timing_mode = "event"
        eval_timing_mode = "event"
    use_profiler_batch_eval = device_type == "npu" and eval_timing_mode == "profiler"
    use_profiler_batch_tune = device_type == "npu" and tune_timing_mode == "profiler"

    def triton_eval_tune(shape: Shape, cfg: MatmulConfig) -> Dict[str, object]:
        if triton_evaluator is None:
            raise RuntimeError("内部错误：缺少 triton evaluator")
        return triton_evaluator.evaluate(shape, cfg, timing_mode=tune_timing_mode)

    def triton_eval_runtime(shape: Shape, cfg: MatmulConfig) -> Dict[str, object]:
        if triton_evaluator is None:
            raise RuntimeError("内部错误：缺少 triton evaluator")
        return triton_evaluator.evaluate(shape, cfg, timing_mode=eval_timing_mode)

    def triton_eval_tune_batch(shape: Shape, cfgs: Sequence[MatmulConfig]) -> list[Dict[str, object]]:
        if triton_evaluator is None:
            raise RuntimeError("内部错误：缺少 triton evaluator")
        entries = [(shape, cfg) for cfg in cfgs]
        return triton_evaluator.evaluate_batch(entries, timing_mode=tune_timing_mode)

    def torch_eval_tune(shape: Shape) -> Dict[str, object]:
        if torch_evaluator is None:
            raise RuntimeError("内部错误：缺少 torch evaluator")
        return torch_evaluator.evaluate(shape, timing_mode=tune_timing_mode)

    def torch_eval_runtime(shape: Shape) -> Dict[str, object]:
        if torch_evaluator is None:
            raise RuntimeError("内部错误：缺少 torch evaluator")
        return torch_evaluator.evaluate(shape, timing_mode=eval_timing_mode)

    if need_triton and args.prototype_count > 0:
        if triton_evaluator is None:
            raise RuntimeError("内部错误：原型 shape 选优缺少 triton evaluator")
        proto_count = min(int(args.prototype_count), len(tune_set))
        typical_shapes = _pick_typical_shapes(tune_set, proto_count)
        proto_candidates, proto_rows = _derive_candidate_pool_from_typical_shapes(
            typical_shapes,
            candidates,
            triton_eval_tune,
            eval_batch=triton_eval_tune_batch if use_profiler_batch_tune else None,
        )
        if proto_candidates:
            candidates = proto_candidates
        cfg_desc = ", ".join(
            f"{c.config_id}(BM={c.BLOCK_M},BN={c.BLOCK_N},BK={c.BLOCK_K})" for c in candidates
        )
        shape_desc = ", ".join(f"{m}x{n}x{k}" for m, n, k in typical_shapes)
        print(
            f"[prototype] shapes={len(typical_shapes)} [{shape_desc}] "
            f"=> candidates={len(candidates)} [{cfg_desc}]"
        )

        report_csv = args.prototype_report_csv
        if not report_csv:
            p = Path(args.results_csv)
            report_csv = str(p.with_name(f"{p.stem}_prototype_best.csv"))
        _write_prototype_report_csv(report_csv, proto_rows)
        print(f"[prototype] report_csv={report_csv}")
        if args.prototype_only:
            print("prototype-only done.")
            return

    for method in args.methods:
        budget = BudgetTracker(args.budget_seconds)
        rows: List[Dict[str, object]] = []

        if method == "FULL":
            if triton_evaluator is None:
                raise RuntimeError("内部错误：FULL 需要 Triton evaluator")
            policy = FullAutotunePolicy(candidates)
            for i, shape in enumerate(tune_set):
                if budget.exceeded():
                    break
                sel = policy.select(
                    shape,
                    triton_eval_tune,
                    budget,
                    batch_evaluator=triton_eval_tune_batch if use_profiler_batch_tune else None,
                )
                met = sel.premeasure if sel.premeasure is not None else triton_eval_tune(shape, sel.config)
                rows.append(
                    _make_record(
                        "FULL",
                        "tune",
                        i,
                        shape,
                        sel,
                        met,
                        args.dtype,
                        gpu,
                        args.bucket_m_split,
                        args.bucket_n_split,
                        args.bucket_k_split,
                    )
                )
            for i, shape in enumerate(eval_set):
                if budget.exceeded():
                    break
                sel = policy.select(
                    shape,
                    triton_eval_tune,
                    budget,
                    batch_evaluator=triton_eval_tune_batch if use_profiler_batch_tune else None,
                )
                if use_profiler_batch_eval:
                    rows.append(
                        _make_record(
                            "FULL",
                            "eval",
                            i,
                            shape,
                            sel,
                            {},
                            args.dtype,
                            gpu,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )
                else:
                    met = triton_eval_runtime(shape, sel.config)
                    rows.append(
                        _make_record(
                            "FULL",
                            "eval",
                            i,
                            shape,
                            sel,
                            met,
                            args.dtype,
                            gpu,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )

            if use_profiler_batch_eval:
                eval_rows = [r for r in rows if r["split"] == "eval"]
                eval_entries = [((int(r["M"]), int(r["N"]), int(r["K"])), CONFIG_MAP[str(r["config_id"])]) for r in eval_rows]
                eval_metrics = triton_evaluator.evaluate_batch(eval_entries, timing_mode=eval_timing_mode)
                for row, met in zip(eval_rows, eval_metrics):
                    row.update(
                        {
                            "compile_time_ms": float(met.get("compile_time_ms", 0.0)),
                            "runtime_cost_us": float(met.get("runtime_cost_us", 0.0)),
                            "p99_us": float(met.get("p99_us", 0.0)),
                            "runtime_perf_tflops": float(met.get("runtime_perf_tflops", 0.0)),
                            "invalid_config": int(met.get("invalid_config", 0)),
                            "notes": ";".join([x for x in [str(row.get("notes", "")), str(met.get("notes", ""))] if x]),
                        }
                    )
            append_records(args.results_csv, rows)
            print(
                f"[method=FULL] rows={len(rows)} elapsed_s={budget.elapsed_seconds():.2f} "
                f"budget_s={args.budget_seconds:.2f}"
            )
            continue

        if method == "BUCKET":
            if triton_evaluator is None:
                raise RuntimeError("内部错误：BUCKET 需要 Triton evaluator")
            policy = LlmBucket8AutotunePolicy(
                candidates,
                m_split=args.bucket_m_split,
                n_split=args.bucket_n_split,
                k_split=args.bucket_k_split,
            )
            tuned_keys = set()
            for i, shape in enumerate(tune_set):
                if budget.exceeded():
                    break
                sel = policy.select(
                    shape,
                    triton_eval_tune,
                    budget,
                    batch_evaluator=triton_eval_tune_batch if use_profiler_batch_tune else None,
                )
                met = sel.premeasure if sel.premeasure is not None else triton_eval_tune(shape, sel.config)
                rows.append(
                    _make_record(
                        "BUCKET",
                        "tune",
                        i,
                        shape,
                        sel,
                        met,
                        args.dtype,
                        gpu,
                        args.bucket_m_split,
                        args.bucket_n_split,
                        args.bucket_k_split,
                    )
                )
                tuned_keys.add(
                    llm_bucket8_key(shape[0], shape[1], shape[2], args.bucket_m_split, args.bucket_n_split, args.bucket_k_split)
                )

            missing = eval_keys - tuned_keys
            if missing:
                raise RuntimeError(f"BUG: 进入 eval 前存在未调过的 key: {sorted(missing)}")

            for i, shape in enumerate(eval_set):
                if budget.exceeded():
                    break
                sel = policy.select(
                    shape,
                    triton_eval_tune,
                    budget,
                    batch_evaluator=triton_eval_tune_batch if use_profiler_batch_tune else None,
                )
                if sel.tune_time_ms > 0:
                    raise RuntimeError("BUG: bucket policy 在 eval 阶段发生了调参")
                if use_profiler_batch_eval:
                    rows.append(
                        _make_record(
                            "BUCKET",
                            "eval",
                            i,
                            shape,
                            sel,
                            {},
                            args.dtype,
                            gpu,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )
                else:
                    met = triton_eval_runtime(shape, sel.config)
                    rows.append(
                        _make_record(
                            "BUCKET",
                            "eval",
                            i,
                            shape,
                            sel,
                            met,
                            args.dtype,
                            gpu,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )

            if use_profiler_batch_eval:
                eval_rows = [r for r in rows if r["split"] == "eval"]
                eval_entries = [((int(r["M"]), int(r["N"]), int(r["K"])), CONFIG_MAP[str(r["config_id"])]) for r in eval_rows]
                eval_metrics = triton_evaluator.evaluate_batch(eval_entries, timing_mode=eval_timing_mode)
                for row, met in zip(eval_rows, eval_metrics):
                    row.update(
                        {
                            "compile_time_ms": float(met.get("compile_time_ms", 0.0)),
                            "runtime_cost_us": float(met.get("runtime_cost_us", 0.0)),
                            "p99_us": float(met.get("p99_us", 0.0)),
                            "runtime_perf_tflops": float(met.get("runtime_perf_tflops", 0.0)),
                            "invalid_config": int(met.get("invalid_config", 0)),
                            "notes": ";".join([x for x in [str(row.get("notes", "")), str(met.get("notes", ""))] if x]),
                        }
                    )

            append_records(args.results_csv, rows)
            print(
                f"[method=BUCKET] rows={len(rows)} elapsed_s={budget.elapsed_seconds():.2f} "
                f"budget_s={args.budget_seconds:.2f} tuned_keys={len(tuned_keys)} "
                f"splits=(M<={args.bucket_m_split},N<={args.bucket_n_split},K<={args.bucket_k_split})"
            )
            continue

        if method == "TORCH":
            if torch_evaluator is None:
                raise RuntimeError("内部错误：TORCH 需要 Torch evaluator")
            torch_cfg = torch_evaluator.get_torch_config_id()
            tune_items: List[tuple[int, Shape]] = []
            for i, shape in enumerate(tune_set):
                if budget.exceeded():
                    break
                if use_profiler_batch_tune:
                    tune_items.append((i, shape))
                else:
                    met = torch_eval_tune(shape)
                    rows.append(
                        _make_torch_record(
                            "tune",
                            i,
                            shape,
                            met,
                            args.dtype,
                            gpu,
                            torch_cfg,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )

            if use_profiler_batch_tune and tune_items:
                tune_shapes = [shape for _, shape in tune_items]
                tune_metrics = torch_evaluator.evaluate_batch(tune_shapes, timing_mode=tune_timing_mode)
                for (i, shape), met in zip(tune_items, tune_metrics):
                    rows.append(
                        _make_torch_record(
                            "tune",
                            i,
                            shape,
                            met,
                            args.dtype,
                            gpu,
                            torch_cfg,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )

            eval_items: List[tuple[int, Shape]] = []
            for i, shape in enumerate(eval_set):
                if budget.exceeded():
                    break
                if use_profiler_batch_eval:
                    eval_items.append((i, shape))
                else:
                    met = torch_eval_runtime(shape)
                    rows.append(
                        _make_torch_record(
                            "eval",
                            i,
                            shape,
                            met,
                            args.dtype,
                            gpu,
                            torch_cfg,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )

            if use_profiler_batch_eval and eval_items:
                eval_shapes = [shape for _, shape in eval_items]
                eval_metrics = torch_evaluator.evaluate_batch(eval_shapes, timing_mode=eval_timing_mode)
                for (i, shape), met in zip(eval_items, eval_metrics):
                    rows.append(
                        _make_torch_record(
                            "eval",
                            i,
                            shape,
                            met,
                            args.dtype,
                            gpu,
                            torch_cfg,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )

            append_records(args.results_csv, rows)
            print(
                f"[method=TORCH] rows={len(rows)} elapsed_s={budget.elapsed_seconds():.2f} "
                f"budget_s={args.budget_seconds:.2f}"
            )
            continue

        if method == "ONEKEY":
            if triton_evaluator is None:
                raise RuntimeError("内部错误：ONEKEY 需要 Triton evaluator")
            rep_shape = _pick_representative_shape(tune_set)
            t0 = time.perf_counter()
            best_cfg, best_metrics, notes = autotune_best(
                rep_shape,
                candidates,
                triton_eval_tune,
                budget,
                batch_evaluator=triton_eval_tune_batch if use_profiler_batch_tune else None,
            )
            rep_tune_ms = (time.perf_counter() - t0) * 1000.0
            if best_metrics is None:
                best_metrics = triton_eval_tune(rep_shape, best_cfg)

            rep_cache_key = f"onekey:{rep_shape[0]}x{rep_shape[1]}x{rep_shape[2]}"
            rep_notes = f"onekey_rep_shape={rep_shape[0]}x{rep_shape[1]}x{rep_shape[2]}"

            if onekey_guard_enabled:
                if torch_evaluator is None:
                    raise RuntimeError("内部错误：ONEKEY torch 门槛检查缺少 torch evaluator")
                if use_profiler_batch_tune:
                    torch_rep_metrics = torch_evaluator.evaluate_batch([rep_shape], timing_mode=tune_timing_mode)[0]
                else:
                    torch_rep_metrics = torch_eval_tune(rep_shape)

                triton_invalid = int(best_metrics.get("invalid_config", 0)) == 1
                torch_invalid = int(torch_rep_metrics.get("invalid_config", 0)) == 1
                triton_us = float(best_metrics.get("runtime_cost_us", 0.0))
                torch_us = float(torch_rep_metrics.get("runtime_cost_us", 0.0))
                speedup_vs_torch = 0.0
                if triton_us > 0:
                    speedup_vs_torch = torch_us / triton_us

                guard_note = (
                    "onekey_rep_vs_torch="
                    f"{speedup_vs_torch:.6f}x(torch={torch_us:.3f}us,triton={triton_us:.3f}us,"
                    f"min={args.onekey_min_speedup_vs_torch:.3f})"
                )
                rep_notes = f"{rep_notes};{guard_note}"

                min_speedup = float(args.onekey_min_speedup_vs_torch)
                guard_pass = (
                    (not triton_invalid)
                    and (not torch_invalid)
                    and triton_us > 0.0
                    and torch_us > 0.0
                    and (min_speedup <= 0.0 or speedup_vs_torch >= min_speedup)
                )
                if not guard_pass:
                    raise RuntimeError(
                        "ONEKEY 门槛失败: "
                        f"rep_shape={rep_shape[0]}x{rep_shape[1]}x{rep_shape[2]}, "
                        f"cfg={best_cfg.config_id}, "
                        f"torch_us={torch_us:.3f}, triton_us={triton_us:.3f}, "
                        f"speedup={speedup_vs_torch:.6f}x, "
                        f"min_required={min_speedup:.3f}. "
                        "请先调整 --candidate-ids 或代表 shape 选择策略。"
                    )
            if notes:
                rep_notes = f"{rep_notes};{notes}"

            rep_idx = next((i for i, s in enumerate(tune_set) if s == rep_shape), 0)

            for i, shape in enumerate(tune_set):
                if budget.exceeded():
                    break
                if i == rep_idx:
                    sel = SelectionResult(
                        config=best_cfg,
                        cache_key=rep_cache_key,
                        tune_time_ms=rep_tune_ms,
                        premeasure=best_metrics,
                        notes=rep_notes,
                    )
                    met = best_metrics
                else:
                    sel = SelectionResult(
                        config=best_cfg,
                        cache_key=rep_cache_key,
                        tune_time_ms=0.0,
                        notes=rep_notes,
                    )
                    met = triton_eval_tune(shape, best_cfg)

                rows.append(
                    _make_record(
                        "ONEKEY",
                        "tune",
                        i,
                        shape,
                        sel,
                        met,
                        args.dtype,
                        gpu,
                        args.bucket_m_split,
                        args.bucket_n_split,
                        args.bucket_k_split,
                    )
                )

            for i, shape in enumerate(eval_set):
                if budget.exceeded():
                    break
                sel = SelectionResult(
                    config=best_cfg,
                    cache_key=rep_cache_key,
                    tune_time_ms=0.0,
                    notes=rep_notes,
                )
                if use_profiler_batch_eval:
                    rows.append(
                        _make_record(
                            "ONEKEY",
                            "eval",
                            i,
                            shape,
                            sel,
                            {},
                            args.dtype,
                            gpu,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )
                else:
                    met = triton_eval_runtime(shape, best_cfg)
                    rows.append(
                        _make_record(
                            "ONEKEY",
                            "eval",
                            i,
                            shape,
                            sel,
                            met,
                            args.dtype,
                            gpu,
                            args.bucket_m_split,
                            args.bucket_n_split,
                            args.bucket_k_split,
                        )
                    )

            if use_profiler_batch_eval:
                eval_rows = [r for r in rows if r["split"] == "eval"]
                eval_entries = [((int(r["M"]), int(r["N"]), int(r["K"])), CONFIG_MAP[str(r["config_id"])]) for r in eval_rows]
                eval_metrics = triton_evaluator.evaluate_batch(eval_entries, timing_mode=eval_timing_mode)
                for row, met in zip(eval_rows, eval_metrics):
                    row.update(
                        {
                            "compile_time_ms": float(met.get("compile_time_ms", 0.0)),
                            "runtime_cost_us": float(met.get("runtime_cost_us", 0.0)),
                            "p99_us": float(met.get("p99_us", 0.0)),
                            "runtime_perf_tflops": float(met.get("runtime_perf_tflops", 0.0)),
                            "invalid_config": int(met.get("invalid_config", 0)),
                            "notes": ";".join([x for x in [str(row.get("notes", "")), str(met.get("notes", ""))] if x]),
                        }
                    )

            append_records(args.results_csv, rows)
            print(
                f"[method=ONEKEY] rows={len(rows)} elapsed_s={budget.elapsed_seconds():.2f} "
                f"budget_s={args.budget_seconds:.2f} rep_shape={rep_shape[0]}x{rep_shape[1]}x{rep_shape[2]} "
                f"cfg={best_cfg.config_id}"
            )
            continue

    print(f"done. results_csv={args.results_csv}")


if __name__ == "__main__":
    main()
