#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

T0="$(date +%s)"

# 与刚刚那轮一致的固定参数：
# device=npu, dtype=fp16, npu_timing=profiler_all,
# prototype_count=4, tune_size=16, eval_size=16, warmup=1, repeat=5
RUN_DIR="results/pipeline_repro_$(date -u +%Y%m%dT%H%M%SZ)"

echo "run_dir=${RUN_DIR}"

bench/run_bucket_torch_pipeline.sh \
  --device npu \
  --dtype fp16 \
  --npu-timing profiler_all \
  --prototype-count 4 \
  --tune-size 16 \
  --eval-size 16 \
  --warmup 1 \
  --repeat 5 \
  --run-dir "${RUN_DIR}"

T1="$(date +%s)"
ELAPSED="$((T1 - T0))"
ELAPSED_MIN="$((ELAPSED / 60))"
ELAPSED_SEC="$((ELAPSED % 60))"
echo "done. outputs in ${RUN_DIR}"
echo "total_elapsed_sec=${ELAPSED} (${ELAPSED_MIN}m${ELAPSED_SEC}s)"
