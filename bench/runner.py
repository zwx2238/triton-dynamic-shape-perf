from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from bench.configs.base_configs import BASE_CONFIGS, CONFIG_MAP, MatmulConfig
from bench.configs.searched_configs import save_searched_configs
from bench.csv_logger import append_records, reset_csv
from bench.kernels.triton_matmul import TritonMatmulEvaluator
from bench.policies.bucket_autotune import BucketAutotunePolicy
from bench.policies.bucket_autotune_v2 import BucketAutotuneV2Policy
from bench.policies.buckets import bucket_key
from bench.policies.common import BudgetTracker, SelectionResult
from bench.policies.fixed_static import FixedStaticPolicy, pick_fixed_config_from_reference
from bench.policies.full_autotune import FullAutotunePolicy
from bench.policies.heuristic import HeuristicPolicy
from bench.policies.offline_table import OfflineTablePolicy, build_offline_table
from bench.policies.script_search import search_top_configs_for_workload
from bench.workloads.synthetic import Shape, WorkloadSplits, build_synthetic_workloads

ALL_METHODS = ["A", "B", "B2", "C", "D", "F", "U"]
ALL_WORKLOADS = ["uniform", "llm_style", "training_style", "adversarial"]


def _parse_config_id_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _shape_id(workload: str, split: str, idx: int) -> str:
    return f"{workload}_{split}_{idx:03d}"


def _trim_splits(splits: WorkloadSplits, max_tune: int, max_eval: int, max_probe: int) -> WorkloadSplits:
    return {
        "tune": splits["tune"][:max_tune],
        "eval": splits["eval"][:max_eval],
        "probe": splits["probe"][:max_probe],
    }


def _float_ms(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _make_record(
    method: str,
    workload: str,
    split: str,
    shape_idx: int,
    shape: Shape,
    dtype: str,
    gpu_name: str,
    selection: SelectionResult,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    M, N, K = shape
    b_m, b_n, b_k = bucket_key(M, N, K)
    cfg = selection.config

    notes = []
    if selection.notes:
        notes.append(selection.notes)
    metric_note = str(metrics.get("notes", ""))
    if metric_note:
        notes.append(metric_note)

    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": method,
        "workload": workload,
        "split": split,
        "shape_id": _shape_id(workload, split, shape_idx),
        "M": M,
        "N": N,
        "K": K,
        "dtype": dtype,
        "gpu": gpu_name,
        "config_id": cfg.config_id,
        "BLOCK_M": cfg.BLOCK_M,
        "BLOCK_N": cfg.BLOCK_N,
        "BLOCK_K": cfg.BLOCK_K,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
        "GROUP_M": cfg.GROUP_M,
        "compile_time_ms": _float_ms(metrics.get("compile_time_ms", 0.0)),
        "tune_time_ms": round(selection.tune_time_ms, 3),
        "runtime_cost_us": _float_ms(metrics.get("runtime_cost_us", 0.0)),
        "p99_us": _float_ms(metrics.get("p99_us", 0.0)),
        "runtime_perf_tflops": _float_ms(metrics.get("runtime_perf_tflops", 0.0)),
        "bucket_m": b_m,
        "bucket_n": b_n,
        "bucket_k": b_k,
        "cache_key": selection.cache_key,
        "invalid_config": int(metrics.get("invalid_config", 0)),
        "notes": ";".join(notes),
    }


def _run_shapes(
    method: str,
    workload: str,
    split: str,
    shapes: Sequence[Shape],
    select_fn: Callable[[Shape], SelectionResult],
    evaluator: TritonMatmulEvaluator,
    budget: BudgetTracker,
    dtype: str,
    gpu_name: str,
    results_csv: str,
) -> int:
    rows: List[Dict[str, object]] = []

    for idx, shape in enumerate(shapes):
        if budget.exceeded():
            break

        selection = select_fn(shape)
        if selection.premeasure is not None:
            metrics = selection.premeasure
        else:
            metrics = evaluator.evaluate(shape, selection.config)

        rows.append(
            _make_record(
                method=method,
                workload=workload,
                split=split,
                shape_idx=idx,
                shape=shape,
                dtype=dtype,
                gpu_name=gpu_name,
                selection=selection,
                metrics=metrics,
            )
        )

    return append_records(results_csv, rows)


def _run_method_c(
    workloads: Dict[str, WorkloadSplits],
    workload_order: Sequence[str],
    evaluator: TritonMatmulEvaluator,
    budget: BudgetTracker,
    args: argparse.Namespace,
) -> int:
    policy = HeuristicPolicy(CONFIG_MAP)
    rows = 0

    for workload in workload_order:
        for split in ["tune", "eval"]:
            rows += _run_shapes(
                method="C",
                workload=workload,
                split=split,
                shapes=workloads[workload][split],
                select_fn=lambda shape: policy.select(shape),
                evaluator=evaluator,
                budget=budget,
                dtype=args.dtype,
                gpu_name=evaluator.get_gpu_name(),
                results_csv=args.results_csv,
            )
            if budget.exceeded():
                return rows
    return rows


def _run_method_a_or_b(
    method: str,
    workloads: Dict[str, WorkloadSplits],
    workload_order: Sequence[str],
    evaluator: TritonMatmulEvaluator,
    budget: BudgetTracker,
    args: argparse.Namespace,
) -> int:
    if method == "A":
        policy = FullAutotunePolicy(BASE_CONFIGS)
    elif method == "B":
        policy = BucketAutotunePolicy(BASE_CONFIGS, method="B")
    elif method == "B2":
        llm_ids = _parse_config_id_list(args.b2_llm_candidates)
        llm_cfgs = [CONFIG_MAP[cid] for cid in llm_ids if cid in CONFIG_MAP]
        if not llm_cfgs:
            llm_cfgs = BASE_CONFIGS
        policy = BucketAutotuneV2Policy(
            BASE_CONFIGS,
            use_anchor=not args.b2_disable_anchor,
            anchor_alpha=args.b2_anchor_alpha,
            anchor_top_k=args.b2_anchor_top_k,
            llm_specialization=not args.b2_disable_llm_specialization,
            llm_candidates=llm_cfgs,
        )
    else:
        raise ValueError(method)

    rows = 0

    for workload in workload_order:
        for split in ["tune", "eval"]:
            select_fn = lambda shape: policy.select(shape, evaluator.evaluate, budget)

            rows += _run_shapes(
                method=method,
                workload=workload,
                split=split,
                shapes=workloads[workload][split],
                select_fn=select_fn,
                evaluator=evaluator,
                budget=budget,
                dtype=args.dtype,
                gpu_name=evaluator.get_gpu_name(),
                results_csv=args.results_csv,
            )
            if budget.exceeded():
                return rows
    return rows


def _run_method_d(
    workloads: Dict[str, WorkloadSplits],
    workload_order: Sequence[str],
    evaluator: TritonMatmulEvaluator,
    budget: BudgetTracker,
    args: argparse.Namespace,
) -> int:
    table = build_offline_table(
        path=args.offline_table_csv,
        candidates=BASE_CONFIGS,
        evaluator=evaluator.evaluate,
        budget=budget,
    )
    policy = OfflineTablePolicy(table=table, config_map=CONFIG_MAP)

    rows = 0
    for workload in workload_order:
        for split in ["tune", "eval"]:
            rows += _run_shapes(
                method="D",
                workload=workload,
                split=split,
                shapes=workloads[workload][split],
                select_fn=lambda shape: policy.select(shape),
                evaluator=evaluator,
                budget=budget,
                dtype=args.dtype,
                gpu_name=evaluator.get_gpu_name(),
                results_csv=args.results_csv,
            )
            if budget.exceeded():
                return rows
    return rows


def _run_method_f(
    workloads: Dict[str, WorkloadSplits],
    workload_order: Sequence[str],
    evaluator: TritonMatmulEvaluator,
    budget: BudgetTracker,
    args: argparse.Namespace,
) -> int:
    searched_mapping: Dict[str, List[str]] = {}

    # Search 阶段（budget 与在线阶段共享）
    for workload in workload_order:
        if budget.exceeded():
            break
        search_res = search_top_configs_for_workload(
            workload=workload,
            probe_shapes=workloads[workload]["probe"],
            candidates=BASE_CONFIGS,
            evaluator=evaluator.evaluate,
            budget=budget,
            top_k=6,
        )
        searched_mapping[workload] = list(search_res["selected_ids"])

    save_searched_configs(args.searched_configs_json, searched_mapping)

    rows = 0
    for workload in workload_order:
        if budget.exceeded():
            break

        selected_ids = searched_mapping.get(workload, [])
        selected_cfgs = [CONFIG_MAP[cid] for cid in selected_ids if cid in CONFIG_MAP]
        if not selected_cfgs:
            selected_cfgs = BASE_CONFIGS[:6]

        policy = BucketAutotunePolicy(selected_cfgs, method="F")
        for split in ["tune", "eval"]:
            rows += _run_shapes(
                method="F",
                workload=workload,
                split=split,
                shapes=workloads[workload][split],
                select_fn=lambda shape: policy.select(shape, evaluator.evaluate, budget),
                evaluator=evaluator,
                budget=budget,
                dtype=args.dtype,
                gpu_name=evaluator.get_gpu_name(),
                results_csv=args.results_csv,
            )
            if budget.exceeded():
                return rows

    return rows


def _run_method_u(
    workloads: Dict[str, WorkloadSplits],
    workload_order: Sequence[str],
    evaluator: TritonMatmulEvaluator,
    budget: BudgetTracker,
    args: argparse.Namespace,
) -> int:
    ref_shape = tuple(args.unoptimized_ref_shape)
    fixed_cfg, tune_time_ms, pick_notes = pick_fixed_config_from_reference(
        reference_shape=ref_shape, candidates=BASE_CONFIGS, evaluator=evaluator.evaluate, budget=budget
    )
    notes = f"fixed_static_cfg;ref_shape={ref_shape[0]}x{ref_shape[1]}x{ref_shape[2]};pick={pick_notes}"
    policy = FixedStaticPolicy(fixed_config=fixed_cfg, cache_key=f"fixed_{fixed_cfg.config_id}", notes=notes)

    rows = 0
    first_row = True
    for workload in workload_order:
        for split in ["tune", "eval"]:
            shaped_rows: List[Dict[str, object]] = []
            for idx, shape in enumerate(workloads[workload][split]):
                if budget.exceeded():
                    break
                selection = policy.select(shape)
                if first_row:
                    # 把一次性“选固定 config”的开销记录到首条数据，避免丢失方法内调参成本。
                    selection.tune_time_ms = tune_time_ms
                    first_row = False
                metrics = evaluator.evaluate(shape, selection.config)
                shaped_rows.append(
                    _make_record(
                        method="U",
                        workload=workload,
                        split=split,
                        shape_idx=idx,
                        shape=shape,
                        dtype=args.dtype,
                        gpu_name=evaluator.get_gpu_name(),
                        selection=selection,
                        metrics=metrics,
                    )
                )
            rows += append_records(args.results_csv, shaped_rows)
            if budget.exceeded():
                return rows
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 Triton dynamic-shape benchmark runner")
    parser.add_argument("--methods", nargs="+", default=["C", "D", "B", "A", "F"], choices=ALL_METHODS)
    parser.add_argument("--workloads", nargs="+", default=ALL_WORKLOADS, choices=ALL_WORKLOADS)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--results-csv", type=str, default="results/phase1_raw.csv")
    parser.add_argument("--offline-table-csv", type=str, default="results/offline_table.csv")
    parser.add_argument("--searched-configs-json", type=str, default="results/searched_configs.json")
    parser.add_argument("--reset-results", action="store_true")
    parser.add_argument("--max-tune-shapes", type=int, default=32)
    parser.add_argument("--max-eval-shapes", type=int, default=64)
    parser.add_argument("--max-probe-shapes", type=int, default=8)
    parser.add_argument(
        "--b2-disable-anchor",
        action="store_true",
        help="关闭 B2 的 representative-anchor 联合评分，仅用当前 shape 评分",
    )
    parser.add_argument(
        "--b2-anchor-alpha",
        type=float,
        default=0.1,
        help="B2 anchor 评分权重，范围 [0,1]，值越大越依赖 representative shape",
    )
    parser.add_argument(
        "--b2-anchor-top-k",
        type=int,
        default=4,
        help="B2 仅对当前 shape 前 K 名候选做 anchor 校正",
    )
    parser.add_argument(
        "--b2-disable-llm-specialization",
        action="store_true",
        help="关闭 B2 的 llm_style 形状专项通道",
    )
    parser.add_argument(
        "--b2-llm-candidates",
        type=str,
        default="c11,c00,c04,c02,c10,c01",
        help="B2 llm 专项通道候选 config 列表，逗号分隔",
    )
    parser.add_argument(
        "--unoptimized-ref-shape",
        nargs=3,
        type=int,
        default=[2048, 2048, 2048],
        metavar=("M", "N", "K"),
        help="U 方法用于选固定 config 的静态参考 shape",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reset_results:
        reset_csv(args.results_csv)

    workloads = build_synthetic_workloads(
        seed=20260302,
        tune_count=args.max_tune_shapes,
        eval_count=args.max_eval_shapes,
        probe_count=args.max_probe_shapes,
    )
    for name in list(workloads.keys()):
        workloads[name] = _trim_splits(
            workloads[name],
            max_tune=args.max_tune_shapes,
            max_eval=args.max_eval_shapes,
            max_probe=args.max_probe_shapes,
        )

    # 运行前确保结果目录存在
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)

    evaluator = TritonMatmulEvaluator(
        dtype=args.dtype,
        device=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    total_rows = 0
    for method in args.methods:
        method_budget = BudgetTracker(args.budget_seconds)

        if method == "C":
            rows = _run_method_c(workloads, args.workloads, evaluator, method_budget, args)
        elif method in {"A", "B", "B2"}:
            rows = _run_method_a_or_b(method, workloads, args.workloads, evaluator, method_budget, args)
        elif method == "D":
            rows = _run_method_d(workloads, args.workloads, evaluator, method_budget, args)
        elif method == "F":
            rows = _run_method_f(workloads, args.workloads, evaluator, method_budget, args)
        elif method == "U":
            rows = _run_method_u(workloads, args.workloads, evaluator, method_budget, args)
        else:
            raise ValueError(f"未知方法: {method}")

        total_rows += rows
        print(
            f"[method={method}] rows={rows}, elapsed_s={method_budget.elapsed_seconds():.2f}, "
            f"budget_s={args.budget_seconds:.2f}"
        )

    print(f"done. total_rows={total_rows}, results_csv={args.results_csv}")


if __name__ == "__main__":
    main()
