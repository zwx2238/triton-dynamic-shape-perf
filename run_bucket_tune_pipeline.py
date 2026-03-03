#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from bench.bucket_tune.runtime import build_benchmark_config, build_benchmark_state
from bench.bucket_tune.types import BenchmarkConfig, BenchmarkOptions, BenchmarkState
from bench.ops import list_operators
from bench.reporting.compare_case_runtime import compare_case_runtime
from bench.reporting.summarize_results import summarize
from bench.stages.bucket_stage import run_bucket
from bench.stages.prototype_stage import run_prototype
from bench.stages.torch_stage import run_torch


def utc_ts() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("bucket_tune_pipeline")
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
    parser = argparse.ArgumentParser(description="NPU BUCKET vs TORCH pipeline entry (prototype mandatory).")
    parser.add_argument("--op", type=str, default="matmul", choices=list_operators())
    parser.add_argument("--dtype", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tune-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--prototype-count", type=int, default=4, help="必须 > 0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260302)
    parser.add_argument("--bucket-splits", type=int, nargs="+", default=[4, 2048, 3072])
    parser.add_argument("--run-dir", type=str, default="")
    return parser.parse_args()


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.run_dir:
            self.run_dir = Path(args.run_dir)
        else:
            self.run_dir = Path("results") / f"pipeline_repro_{utc_ts()}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / "pipeline.log"
        self.status_file = self.run_dir / "status.txt"
        self.stage_times_csv = self.run_dir / "stage_times.csv"
        self.proto_report_csv = self.run_dir / "prototype_best.csv"
        self.main_results_csv = self.run_dir / "bucket_torch.csv"
        self.case_compare_csv = self.run_dir / "bucket_torch_case_compare_eval.csv"
        self.summary_prefix = "bucket_torch"

        self.logger = build_logger(self.log_file)

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
        self.append_stage_time(stage, start_utc, end_utc, elapsed, detail)

    def run(self) -> None:
        args = self.args
        if int(args.prototype_count) <= 0:
            raise ValueError(f"要求 --prototype-count > 0（当前: {args.prototype_count}）")

        self.logger.info("run_dir=%s", self.run_dir)
        self.logger.info("op=%s", args.op)
        self.logger.info(
            "params: dtype=%s tune_size=%s eval_size=%s prototype_count=%s warmup=%s repeat=%s bucket_splits=%s",
            args.dtype,
            args.tune_size,
            args.eval_size,
            args.prototype_count,
            args.warmup,
            args.repeat,
            args.bucket_splits,
        )

        options = BenchmarkOptions(
            prototype_count=int(args.prototype_count),
            prototype_report_csv=str(self.proto_report_csv),
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

        def require_context() -> tuple[BenchmarkConfig, BenchmarkState]:
            if benchmark_config is None or benchmark_state is None:
                raise RuntimeError("benchmark 尚未初始化")
            return benchmark_config, benchmark_state

        def setup_benchmark_stage() -> str:
            nonlocal benchmark_config
            nonlocal benchmark_state
            benchmark_config = build_benchmark_config(options)
            benchmark_state = build_benchmark_state(options.op_name)
            return f"results_csv={self.main_results_csv}"

        def prototype_stage() -> str:
            config, state = require_context()
            detail = run_prototype(config, state)
            candidate_ids = ",".join(getattr(cfg, "config_id", "?") for cfg in state.candidates)
            self.logger.info("prototype_candidate_ids=%s", candidate_ids)
            return detail

        def bucket_stage() -> str:
            config, state = require_context()
            return run_bucket(config, state)

        def torch_stage() -> str:
            config, state = require_context()
            return run_torch(config, state)

        def case_compare_stage() -> str:
            out_csv, rows = compare_case_runtime(
                self.main_results_csv,
                split="eval",
                out_csv=self.case_compare_csv,
                allow_mixed_metric=False,
            )
            return f"case_compare_csv={out_csv},rows={rows}"

        def summary_stage() -> str:
            overall_path, tune_path, bucket_path = summarize(self.main_results_csv, self.run_dir, self.summary_prefix)
            return f"overall={overall_path},tune={tune_path},bucket={bucket_path}"

        self.run_stage("setup_benchmark", setup_benchmark_stage)
        self.run_stage("prototype", prototype_stage)
        self.run_stage("benchmark_bucket", bucket_stage)
        self.run_stage("benchmark_torch", torch_stage)
        self.run_stage("case_compare", case_compare_stage)
        self.run_stage("summary", summary_stage)

        self.update_status("done", "all_done")
        self.logger.info("DONE")
        self.logger.info("outputs:")
        self.logger.info("  - %s", self.proto_report_csv)
        self.logger.info("  - %s", self.main_results_csv)
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
