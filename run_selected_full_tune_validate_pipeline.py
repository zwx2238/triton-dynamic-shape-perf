#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from bench.bucket_tune.runtime import build_benchmark_config, build_benchmark_state
from bench.bucket_tune.types import BenchmarkConfig, BenchmarkOptions, BenchmarkState
from bench.ops import list_operators
from bench.reporting.compare_case_runtime import compare_case_runtime
from bench.reporting.csv_logger import append_records
from bench.reporting.summarize_results import summarize
from bench.stages.selected_full_tune_stage import (
    build_full_tune_cache_key,
    collect_full_tune_rows_for_candidates,
    load_rows_from_csv,
    save_rows_to_csv,
    select_configs_from_sampled_full_tune,
    write_selected_report_csv,
)
from bench.stages.torch_stage import run_torch


def utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("selected_full_tune_validate_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt_date = "%Y-%m-%dT%H:%M:%SZ"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt=fmt_date))
    file_handler.formatter.converter = time.gmtime

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt=fmt_date))
    stream_handler.formatter.converter = time.gmtime

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate selected configs by FULL_TUNE upper bound vs TORCH baseline."
    )
    parser.add_argument("--op", type=str, default="matmul", choices=list_operators())
    parser.add_argument("--dtype", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tune-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=8)
    parser.add_argument("--sample-p", type=int, default=8, help="阶段2: sampled full tune 的 shape 数")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--speed-p", type=float, default=2.0, help="speed ratio 汇总的 power-mean 参数，默认 2")
    parser.add_argument("--target-speed-full-over-torch", type=float, default=0.0, help="若 > 0，则做达标校验")
    parser.add_argument("--seed", type=int, default=20260302)
    parser.add_argument("--bucket-splits", type=int, nargs="+", default=[4, 2048, 3072])
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="results/full_tune_cache")
    parser.add_argument("--disable-full-tune-cache", action="store_true")
    return parser.parse_args()


def _power_mean(values: list[float], p: float) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0.0]
    if not vals:
        return 0.0
    return (sum(v**p for v in vals) / len(vals)) ** (1.0 / p)


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.run_dir:
            self.run_dir = Path(args.run_dir)
        else:
            self.run_dir = Path("results") / f"selected_validate_{utc_ts()}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / "pipeline.log"
        self.status_file = self.run_dir / "status.txt"
        self.stage_times_csv = self.run_dir / "stage_times.csv"
        self.main_results_csv = self.run_dir / "full_tune_torch.csv"
        self.case_compare_csv = self.run_dir / "full_tune_torch_case_compare_eval.csv"
        self.sample_report_csv = self.run_dir / "sampled_full_tune_picks.csv"
        self.selected_configs_csv = self.run_dir / "selected_c2_configs.csv"
        self.summary_prefix = "full_tune_torch"
        self.cache_dir = Path(args.cache_dir)

        self.logger = build_logger(self.log_file)
        self.stage_elapsed_sec: list[tuple[str, int]] = []

        with self.stage_times_csv.open("w", newline="", encoding="utf-8") as f:
            f.write("stage,start_utc,end_utc,elapsed_sec,detail\n")

    def update_status(self, status: str, stage: str) -> None:
        text = "\n".join(
            [
                f"status={status}",
                f"stage={stage}",
                f"updated_at={utc_iso()}",
                f"run_dir={self.run_dir}",
                f"log_file={self.log_file}",
            ]
        )
        self.status_file.write_text(text + "\n", encoding="utf-8")

    def append_stage_time(self, stage: str, start_utc: str, end_utc: str, elapsed: int, detail: str = "") -> None:
        with self.stage_times_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([stage, start_utc, end_utc, elapsed, detail])

    def run_stage(self, stage: str, fn: Callable[[], Optional[str]]) -> None:
        start_utc = utc_iso()
        t0 = time.time()
        self.update_status("running", stage)
        self.logger.info("START stage=%s", stage)
        detail = ""
        try:
            out = fn()
            if out:
                detail = out
        except Exception:
            self.update_status("failed", stage)
            self.logger.exception("FAILED stage=%s", stage)
            raise
        elapsed = int(time.time() - t0)
        end_utc = utc_iso()
        self.logger.info("END stage=%s elapsed_sec=%s", stage, elapsed)
        self.stage_elapsed_sec.append((stage, elapsed))
        self.append_stage_time(stage, start_utc, end_utc, elapsed, detail)

    def run(self) -> None:
        args = self.args
        speed_p = float(args.speed_p)
        if int(args.sample_p) <= 0:
            raise ValueError(f"要求 --sample-p > 0（当前: {args.sample_p}）")
        if not math.isfinite(speed_p) or speed_p <= 0.0:
            raise ValueError(f"要求 --speed-p > 0 且为有限数（当前: {args.speed_p}）")

        pipeline_start_utc = utc_iso()
        pipeline_t0 = time.time()
        self.logger.info("run_dir=%s", self.run_dir)
        self.logger.info(
            "params: op=%s dtype=%s tune_size=%s eval_size=%s sample_p=%s warmup=%s repeat=%s bucket_splits=%s",
            args.op,
            args.dtype,
            args.tune_size,
            args.eval_size,
            args.sample_p,
            args.warmup,
            args.repeat,
            args.bucket_splits,
        )
        self.logger.info("speed_ratio_power_p=%s", speed_p)

        options = BenchmarkOptions(
            prototype_count=max(1, int(args.sample_p)),
            prototype_report_csv="",
            tune_size=int(args.tune_size),
            eval_size=int(args.eval_size),
            seed=int(args.seed),
            op_name=str(args.op),
            dtype=str(args.dtype),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            bucket_splits=tuple(args.bucket_splits),
            results_csv=str(self.main_results_csv),
            reset_results=True,
        )
        benchmark_config: Optional[BenchmarkConfig] = None
        benchmark_state: Optional[BenchmarkState] = None
        selected_candidates: list[object] = []

        def require_context() -> tuple[BenchmarkConfig, BenchmarkState]:
            if benchmark_config is None or benchmark_state is None:
                raise RuntimeError("benchmark 尚未初始化")
            return benchmark_config, benchmark_state

        def setup_stage() -> str:
            nonlocal benchmark_config
            nonlocal benchmark_state
            benchmark_config = build_benchmark_config(options)
            benchmark_state = build_benchmark_state(options.op_name)
            c1 = list(benchmark_state.candidates)
            c1_desc = ",".join(str(getattr(x, "config_id", "?")) for x in c1)
            return f"results_csv={self.main_results_csv},c1={len(c1)}[{c1_desc}]"

        def select_stage() -> str:
            nonlocal selected_candidates
            config, state = require_context()
            selected_candidates, rows, detail = select_configs_from_sampled_full_tune(config, state, int(args.sample_p))
            write_selected_report_csv(self.sample_report_csv, rows)
            with self.selected_configs_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["config_id", "BLOCK_M", "BLOCK_N", "BLOCK_K"])
                for cfg in selected_candidates:
                    writer.writerow(
                        [
                            str(getattr(cfg, "config_id", "")),
                            int(getattr(cfg, "BLOCK_M", -1)),
                            int(getattr(cfg, "BLOCK_N", -1)),
                            int(getattr(cfg, "BLOCK_K", -1)),
                        ]
                    )
            c2_desc = ",".join(str(getattr(x, "config_id", "?")) for x in selected_candidates)
            return f"{detail},c2_ids=[{c2_desc}],sample_report={self.sample_report_csv}"

        def full_tune_validate_stage() -> str:
            config, _ = require_context()
            if not selected_candidates:
                raise RuntimeError("C2 为空，无法执行验证")
            if not args.disable_full_tune_cache:
                cache_key = build_full_tune_cache_key(config, selected_candidates)
                cache_csv = self.cache_dir / f"{cache_key}.csv"
                cached_rows = load_rows_from_csv(cache_csv)
                if cached_rows:
                    append_records(config.options.results_csv, cached_rows)
                    return (
                        f"method=FULL_TUNE eval_rows={len(cached_rows)} "
                        f"cache_hit=1 cache_key={cache_key} cache_csv={cache_csv}"
                    )
                detail, rows = collect_full_tune_rows_for_candidates(config, selected_candidates)
                append_records(config.options.results_csv, rows)
                save_rows_to_csv(cache_csv, rows)
                return f"{detail},cache_hit=0 cache_key={cache_key} cache_csv={cache_csv}"

            detail, rows = collect_full_tune_rows_for_candidates(config, selected_candidates)
            append_records(config.options.results_csv, rows)
            return f"{detail},cache_disabled=1"

        def torch_stage() -> str:
            config, state = require_context()
            return run_torch(config, state)

        def compare_stage() -> str:
            out_csv, rows = compare_case_runtime(
                self.main_results_csv,
                split="eval",
                out_csv=self.case_compare_csv,
                allow_mixed_metric=False,
            )
            return f"case_compare_csv={out_csv},rows={rows}"

        def summary_stage() -> str:
            overall_path, tune_path, bucket_path = summarize(
                self.main_results_csv,
                self.run_dir,
                self.summary_prefix,
                compare_csv=self.case_compare_csv,
                speed_power=speed_p,
            )
            return f"overall={overall_path},tune={tune_path},bucket={bucket_path},speed_p={speed_p}"

        def metric_gate_stage() -> str:
            with self.case_compare_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            vals: list[float] = []
            for row in rows:
                raw = str(row.get("speed_FULL_TUNE_over_TORCH", "")).strip()
                try:
                    v = float(raw)
                except ValueError:
                    continue
                if math.isfinite(v) and v > 0.0:
                    vals.append(v)
            metric = _power_mean(vals, speed_p)
            if metric <= 0.0:
                raise RuntimeError("未能从 compare 结果中计算 speed_FULL_TUNE_over_TORCH")
            target = float(args.target_speed_full_over_torch)
            if target > 0.0 and metric < target:
                raise RuntimeError(
                    f"speed_FULL_TUNE_over_TORCH(p={speed_p})={metric:.6f} < target={target:.6f}"
                )
            return f"speed_FULL_TUNE_over_TORCH_p{speed_p:g}={metric:.6f},target={target:.6f}"

        self.run_stage("setup_benchmark", setup_stage)
        self.run_stage("select_c2_from_sampled_full_tune", select_stage)
        self.run_stage("benchmark_full_tune_validate", full_tune_validate_stage)
        self.run_stage("benchmark_torch", torch_stage)
        self.run_stage("case_compare", compare_stage)
        self.run_stage("summary", summary_stage)
        self.run_stage("metric_gate", metric_gate_stage)

        total_elapsed_sec = int(time.time() - pipeline_t0)
        pipeline_end_utc = utc_iso()
        self.append_stage_time("all_total", pipeline_start_utc, pipeline_end_utc, total_elapsed_sec, "pipeline_total")
        self.logger.info(
            "stage_elapsed_sec=%s",
            ",".join(f"{name}:{sec}" for name, sec in self.stage_elapsed_sec),
        )
        self.logger.info("TOTAL elapsed_sec=%s", total_elapsed_sec)
        self.update_status("done", "all_done")
        self.logger.info("DONE")
        self.logger.info("outputs:")
        self.logger.info("  - %s", self.main_results_csv)
        self.logger.info("  - %s", self.sample_report_csv)
        self.logger.info("  - %s", self.selected_configs_csv)
        self.logger.info("  - %s", self.case_compare_csv)
        self.logger.info("  - %s", self.run_dir / f"{self.summary_prefix}_summary_overall.csv")
        self.logger.info("  - %s", self.run_dir / f"{self.summary_prefix}_summary_tune.csv")
        self.logger.info("  - %s", self.run_dir / f"{self.summary_prefix}_summary_by_bucket.csv")


def main() -> None:
    args = parse_args()
    try:
        Pipeline(args=args).run()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
