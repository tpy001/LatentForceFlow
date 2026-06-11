#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-outputs/libero_eval_logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_DIR}/${TIMESTAMP}"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
RESULTS_FILE="${RUN_DIR}/suite_results.tsv"
VIDEO_DIR="${RUN_DIR}/videos"

TASK_SUITES=(
  # libero_spatial
  # libero_object
  # libero_goal
  libero_10
)

mkdir -p "${RUN_DIR}"
: > "${RESULTS_FILE}"
overall_status=0

{
  echo "LIBERO evaluation started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Python: ${PYTHON_BIN}"
  echo "Output directory: ${RUN_DIR}"
  echo "Video directory: ${VIDEO_DIR}"
  echo "Task suites: ${TASK_SUITES[*]}"
  echo
} | tee "${SUMMARY_FILE}"

for suite in "${TASK_SUITES[@]}"; do
  LOG_FILE="${RUN_DIR}/${suite}.txt"

  {
    echo "============================================================"
    echo "Running ${suite}"
    echo "Log file: ${LOG_FILE}"
    echo "Started at $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Command: ${PYTHON_BIN} libero_test.py --args.task-suite-name ${suite} --args.video-out-path ${VIDEO_DIR} $*"
    echo "============================================================"
  } | tee -a "${SUMMARY_FILE}" | tee "${LOG_FILE}"

  "${PYTHON_BIN}" libero_test.py --args.task-suite-name "${suite}" --args.video-out-path "${VIDEO_DIR}" "$@" 2>&1 | tee -a "${LOG_FILE}"
  status=${PIPESTATUS[0]}
  success_rate="$("${PYTHON_BIN}" - "${LOG_FILE}" <<'PY'
import re
import sys

log_path = sys.argv[1]
rate = None
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        match = re.search(r"Total success rate:\s*([0-9.]+)", line)
        if match:
            rate = float(match.group(1)) * 100.0

if rate is None:
    print("nan")
else:
    print(f"{rate:.1f}")
PY
)"
  printf "%s\t%s\t%s\n" "${suite}" "${success_rate}" "${status}" >> "${RESULTS_FILE}"

  {
    echo
    echo "Finished ${suite} at $(date '+%Y-%m-%d %H:%M:%S') with exit code ${status}"
    echo "Success rate: ${success_rate}%"
    echo
  } | tee -a "${SUMMARY_FILE}" | tee -a "${LOG_FILE}"

  if [[ "${status}" -ne 0 ]]; then
    overall_status="${status}"
    echo "${suite} failed. Continuing with the next suite. See ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"
  fi
done

{
  echo "============================================================"
  echo "LIBERO evaluation results"
  "${PYTHON_BIN}" - "${RESULTS_FILE}" <<'PY'
import math
import sys

results_path = sys.argv[1]
rates = []
with open(results_path, "r", encoding="utf-8") as f:
    for line in f:
        suite, rate_text, status = line.rstrip("\n").split("\t")
        try:
            rate = float(rate_text)
        except ValueError:
            print(f"{suite}: N/A (exit code {status})")
            continue
        if math.isnan(rate):
            print(f"{suite}: N/A (exit code {status})")
            continue
        rates.append(rate)
        print(f"{suite}: {rate:.1f}%")

if rates:
    print(f"Average: {sum(rates) / len(rates):.1f}%")
else:
    print("Average: N/A")
PY
  echo "All LIBERO evaluations finished at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Run directory: ${RUN_DIR}"
  echo "Video directory: ${VIDEO_DIR}"
  echo "Summary: ${SUMMARY_FILE}"
} | tee -a "${SUMMARY_FILE}"
exit "${overall_status}"
