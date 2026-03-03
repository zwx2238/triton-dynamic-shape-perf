#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="npu"
DTYPE="fp16"
NPU_TIMING="profiler_all"
TUNE_SIZE=16
EVAL_SIZE=16
PROTOTYPE_COUNT=4
WARMUP=1
REPEAT=5
SEED=20260302
BUDGET_SECONDS=300
BUCKET_M_SPLIT=4
BUCKET_N_SPLIT=2048
BUCKET_K_SPLIT=3072
RUN_DIR=""
CANDIDATE_IDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --npu-timing) NPU_TIMING="$2"; shift 2 ;;
        --tune-size) TUNE_SIZE="$2"; shift 2 ;;
        --eval-size) EVAL_SIZE="$2"; shift 2 ;;
        --prototype-count) PROTOTYPE_COUNT="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --budget-seconds) BUDGET_SECONDS="$2"; shift 2 ;;
        --bucket-m-split) BUCKET_M_SPLIT="$2"; shift 2 ;;
        --bucket-n-split) BUCKET_N_SPLIT="$2"; shift 2 ;;
        --bucket-k-split) BUCKET_K_SPLIT="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --candidate-ids) CANDIDATE_IDS="$2"; shift 2 ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${RUN_DIR}" ]]; then
    RUN_DIR="results/pipeline_${RUN_TS}"
fi
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/pipeline.log"
STATUS_FILE="${RUN_DIR}/status.txt"
STAGE_TIMES_CSV="${RUN_DIR}/stage_times.csv"
PROTO_REPORT_CSV="${RUN_DIR}/prototype_best.csv"
PROTO_DUMMY_CSV="${RUN_DIR}/prototype_dummy.csv"
MAIN_RESULTS_CSV="${RUN_DIR}/bucket_torch.csv"
CASE_COMPARE_CSV="${RUN_DIR}/bucket_torch_case_compare_eval.csv"
SUMMARY_PREFIX="bucket_torch"

echo "stage,start_utc,end_utc,elapsed_sec,detail" > "${STAGE_TIMES_CSV}"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG_FILE}"
}

update_status() {
    local status="$1"
    local stage="$2"
    cat > "${STATUS_FILE}" <<EOF
status=${status}
stage=${stage}
updated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
run_dir=${RUN_DIR}
log_file=${LOG_FILE}
EOF
}

run_stage() {
    local stage="$1"
    shift
    local start_utc end_utc t0 t1 elapsed
    start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    t0="$(date +%s)"
    update_status "running" "${stage}"
    log "START stage=${stage}"
    "$@" >> "${LOG_FILE}" 2>&1
    t1="$(date +%s)"
    end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    elapsed="$((t1 - t0))"
    log "END stage=${stage} elapsed_sec=${elapsed}"
    echo "${stage},${start_utc},${end_utc},${elapsed}," >> "${STAGE_TIMES_CSV}"
}

on_error() {
    local ec=$?
    update_status "failed" "${CURRENT_STAGE:-unknown}"
    log "FAILED stage=${CURRENT_STAGE:-unknown} exit_code=${ec}"
    exit "${ec}"
}
trap on_error ERR

log "run_dir=${RUN_DIR}"
log "log_file=${LOG_FILE}"
log "status_file=${STATUS_FILE}"
log "stage_times_csv=${STAGE_TIMES_CSV}"
log "params: device=${DEVICE} dtype=${DTYPE} npu_timing=${NPU_TIMING} tune_size=${TUNE_SIZE} eval_size=${EVAL_SIZE} prototype_count=${PROTOTYPE_COUNT}"

BASE_ARGS=(
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --npu-timing "${NPU_TIMING}"
    --warmup "${WARMUP}"
    --repeat "${REPEAT}"
    --seed "${SEED}"
    --budget-seconds "${BUDGET_SECONDS}"
    --bucket-m-split "${BUCKET_M_SPLIT}"
    --bucket-n-split "${BUCKET_N_SPLIT}"
    --bucket-k-split "${BUCKET_K_SPLIT}"
)

if [[ -n "${CANDIDATE_IDS}" ]]; then
    BASE_ARGS+=(--candidate-ids "${CANDIDATE_IDS}")
fi

if [[ "${PROTOTYPE_COUNT}" -gt 0 ]]; then
    CURRENT_STAGE="prototype_select"
    run_stage "${CURRENT_STAGE}" \
        python bench/llm_full_vs_fixed_bucket.py \
        --methods BUCKET \
        "${BASE_ARGS[@]}" \
        --tune-size "${TUNE_SIZE}" \
        --eval-size 0 \
        --prototype-count "${PROTOTYPE_COUNT}" \
        --prototype-only \
        --prototype-report-csv "${PROTO_REPORT_CSV}" \
        --results-csv "${PROTO_DUMMY_CSV}" \
        --reset-results

    CURRENT_STAGE="prototype_dedup"
    run_stage "${CURRENT_STAGE}" bash -lc "
        set -euo pipefail
        if [[ ! -f '${PROTO_REPORT_CSV}' ]]; then
            echo 'prototype report missing: ${PROTO_REPORT_CSV}' >&2
            exit 1
        fi
        ids=\$(tail -n +2 '${PROTO_REPORT_CSV}' | cut -d',' -f2 | awk 'NF>0 && !seen[\$0]++' | paste -sd, -)
        if [[ -z \"\${ids}\" ]]; then
            echo 'empty prototype candidate ids' >&2
            exit 1
        fi
        echo \"\${ids}\" > '${RUN_DIR}/prototype_candidate_ids.txt'
        echo \"prototype_candidate_ids=\${ids}\"
    "
    CANDIDATE_IDS_EFFECTIVE="$(cat "${RUN_DIR}/prototype_candidate_ids.txt")"
else
    if [[ -z "${CANDIDATE_IDS}" ]]; then
        echo "prototype_count=0 时必须显式传 --candidate-ids" >&2
        exit 2
    fi
    CANDIDATE_IDS_EFFECTIVE="${CANDIDATE_IDS}"
fi

log "candidate_ids_effective=${CANDIDATE_IDS_EFFECTIVE}"

CURRENT_STAGE="benchmark_bucket_torch"
run_stage "${CURRENT_STAGE}" \
    python bench/llm_full_vs_fixed_bucket.py \
    --methods BUCKET TORCH \
    "${BASE_ARGS[@]}" \
    --candidate-ids "${CANDIDATE_IDS_EFFECTIVE}" \
    --prototype-count 0 \
    --tune-size "${TUNE_SIZE}" \
    --eval-size "${EVAL_SIZE}" \
    --results-csv "${MAIN_RESULTS_CSV}" \
    --reset-results

CURRENT_STAGE="case_compare"
run_stage "${CURRENT_STAGE}" \
    python bench/compare_case_runtime.py \
    --input-csv "${MAIN_RESULTS_CSV}" \
    --split eval \
    --methods TORCH BUCKET \
    --baseline TORCH \
    --out-csv "${CASE_COMPARE_CSV}"

CURRENT_STAGE="summary"
run_stage "${CURRENT_STAGE}" \
    python bench/summarize_results.py \
    --input-csv "${MAIN_RESULTS_CSV}" \
    --out-dir "${RUN_DIR}" \
    --prefix "${SUMMARY_PREFIX}"

update_status "done" "all_done"
log "DONE"
log "outputs:"
log "  - ${PROTO_REPORT_CSV}"
log "  - ${RUN_DIR}/prototype_candidate_ids.txt"
log "  - ${MAIN_RESULTS_CSV}"
log "  - ${CASE_COMPARE_CSV}"
log "  - ${RUN_DIR}/${SUMMARY_PREFIX}_summary_overall.csv"
log "  - ${RUN_DIR}/${SUMMARY_PREFIX}_summary_tune.csv"
log "  - ${RUN_DIR}/${SUMMARY_PREFIX}_summary_by_bucket.csv"

