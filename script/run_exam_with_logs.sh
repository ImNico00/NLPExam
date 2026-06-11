#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="run_${RUN_TIMESTAMP}"
RUNS_DIR="${RUNS_DIR:-pipeline_exam/run_logs/pipeline_step_runs}"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/pipeline_step.log"
META_FILE="${RUN_DIR}/run_metadata.txt"
COMMAND_FILE="${RUN_DIR}/command.txt"

mkdir -p "${RUN_DIR}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

COMMAND=(
  "poetry"
  "run"
  "${PYTHON_BIN}"
  -m
  pipeline_exam.src.start_exam
  "$@"
)

{
  echo "run_id=${RUN_ID}"
  echo "run_timestamp=${RUN_TIMESTAMP}"
  echo "repo_root=."
  echo "python_bin=${PYTHON_BIN}"
  echo "host=$(hostname)"
  echo "user=${USER:-unknown}"
  echo "started_at=$(date +"%Y-%m-%d %H:%M:%S %Z")"
} > "${META_FILE}"

printf '%q ' "${COMMAND[@]}" > "${COMMAND_FILE}"
printf '\n' >> "${COMMAND_FILE}"

echo "[${RUN_ID}] starting Exam Pipeline Steps"
echo "[${RUN_ID}] logs: ${LOG_FILE}"
echo "[${RUN_ID}] metadata: ${META_FILE}"

"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
COMMAND_EXIT_CODE=${PIPESTATUS[0]}

{
  echo "finished_at=$(date +"%Y-%m-%d %H:%M:%S %Z")"
  echo "exit_code=${COMMAND_EXIT_CODE}"
} >> "${META_FILE}"

echo "[${RUN_ID}] completed with exit code ${COMMAND_EXIT_CODE}"
exit "${COMMAND_EXIT_CODE}"