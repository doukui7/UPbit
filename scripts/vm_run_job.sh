#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: bash scripts/vm_run_job.sh <upbit|health_check|daily_status|kiwoom_gold|kis_isa|kis_pension>"
  exit 1
fi

case "${MODE}" in
  upbit|health_check|daily_status|kiwoom_gold|kis_isa|kis_pension) ;;
  *)
    echo "[error] unsupported mode: ${MODE}"
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
LOCK_FILE="/tmp/upbit_${MODE}.lock"
AUTO_UPDATE="${AUTO_UPDATE:-0}"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-${REPO_DIR}/.vm_runtime_env}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
LOG_FILE="${LOG_DIR}/${MODE}.log"

if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${RUNTIME_ENV_FILE}"
  set +a
fi

if [[ -f "${REPO_DIR}/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/venv/bin/activate"
fi

if [[ "${AUTO_UPDATE}" == "1" ]]; then
  git fetch origin --quiet 2>/dev/null || true
  git reset --hard origin/master --quiet 2>/dev/null || true
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[error] python command not found"
  exit 3
fi

exec 9>"${LOCK_FILE}"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    msg="[$(date '+%F %T')] skip mode=${MODE} reason=already_running"
    echo "${msg}" | tee -a "${LOG_FILE}"
    exit 0
  fi
fi

echo "[$(date '+%F %T')] start mode=${MODE}" >> "${LOG_FILE}"
# python script loads .env via python-dotenv internally.
if TRADING_MODE="${MODE}" python scripts/github_action_trade.py >> "${LOG_FILE}" 2>&1; then
  echo "[$(date '+%F %T')] done mode=${MODE}" >> "${LOG_FILE}"
else
  code=$?
  echo "[$(date '+%F %T')] fail mode=${MODE} code=${code}" >> "${LOG_FILE}"
  {
    echo "[error] vm_run_job failed: mode=${MODE} code=${code}"
    echo "[error] recent ${MODE}.log tail:"
    tail -n 40 "${LOG_FILE}" || true
  } >&2
  exit "${code}"
fi
