#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
PID_FILE="${LOG_DIR}/vm_scheduler.pid"
STATE_FILE="${LOG_DIR}/vm_scheduler_state.json"
MAX_HEARTBEAT_AGE_SEC="${MAX_HEARTBEAT_AGE_SEC:-180}"

mkdir -p "${LOG_DIR}"

choose_python() {
  if [[ -x "${REPO_DIR}/venv/bin/python" ]]; then
    echo "${REPO_DIR}/venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

is_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

heartbeat_age_sec() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo ""
    return 0
  fi

  local py
  if ! py="$(choose_python)"; then
    echo ""
    return 0
  fi

  "${py}" - "${STATE_FILE}" <<'PY'
import json
import sys
import time

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("__heartbeat_epoch")
    if raw in (None, ""):
        print("")
        raise SystemExit(0)
    ts = float(raw)
    age = int(max(0, time.time() - ts))
    print(age)
except Exception:
    print("")
PY
}

start_scheduler() {
  if is_running; then
    echo "[info] scheduler already running (pid=$(cat "${PID_FILE}"))"
    return 0
  fi

  local py
  if ! py="$(choose_python)"; then
    echo "[error] python executable not found"
    exit 1
  fi

  cd "${REPO_DIR}"
  chmod +x scripts/vm_run_job.sh scripts/vm_scheduler_manager.sh || true
  nohup "${py}" scripts/vm_scheduler.py >> "${LOG_DIR}/vm_scheduler.out.log" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  sleep 1

  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[ok] scheduler started (pid=${pid})"
  else
    echo "[error] scheduler start failed"
    exit 1
  fi
}

stop_scheduler() {
  if ! is_running; then
    echo "[info] scheduler is not running"
    rm -f "${PID_FILE}" || true
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  kill "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      rm -f "${PID_FILE}" || true
      echo "[ok] scheduler stopped"
      return 0
    fi
    sleep 0.5
  done
  echo "[warn] graceful stop timeout, sending SIGKILL"
  kill -9 "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}" || true
  echo "[ok] scheduler killed"
}

show_status() {
  local runtime_env_file="${REPO_DIR}/.vm_runtime_env"
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    echo "[ok] scheduler running (pid=${pid})"
    if [[ -f "${LOG_DIR}/vm_scheduler.log" ]]; then
      echo "[info] latest scheduler log:"
      tail -n 20 "${LOG_DIR}/vm_scheduler.log" || true
    fi
    if [[ -f "${LOG_DIR}/vm_scheduler_state.json" ]]; then
      echo "[info] scheduler state:"
      tail -n 40 "${LOG_DIR}/vm_scheduler_state.json" || true
      echo ""
      local hb_age
      hb_age="$(heartbeat_age_sec)"
      if [[ -n "${hb_age}" ]]; then
        echo "[info] scheduler heartbeat age: ${hb_age}s (threshold=${MAX_HEARTBEAT_AGE_SEC}s)"
      else
        echo "[warn] scheduler heartbeat not found in state file"
      fi
    fi
    if [[ -f "${LOG_DIR}/upbit.log" ]]; then
      echo "[info] latest upbit log:"
      tail -n 40 "${LOG_DIR}/upbit.log" || true
    fi
    if [[ -f "${LOG_DIR}/health_check.log" ]]; then
      echo "[info] latest health_check log:"
      tail -n 20 "${LOG_DIR}/health_check.log" || true
    fi
    if [[ -f "${runtime_env_file}" ]]; then
      echo "[info] runtime env file: ${runtime_env_file}"
      echo "[info] runtime env keys:"
      awk -F= '/^[A-Z0-9_]+=/{print "  - " $1}' "${runtime_env_file}" || true
    else
      echo "[warn] runtime env file not found: ${runtime_env_file}"
    fi
  else
    echo "[info] scheduler not running"
  fi
}

last_trade_age_sec() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo ""
    return 0
  fi
  local py
  if ! py="$(choose_python)"; then
    echo ""
    return 0
  fi
  "${py}" - "${STATE_FILE}" <<'PY'
import json, sys, time
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("__last_trade_epoch")
    if raw in (None, ""):
        print("")
        raise SystemExit(0)
    ts = float(raw)
    age = int(max(0, time.time() - ts))
    print(age)
except Exception:
    print("")
PY
}

consecutive_failures() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo "0"
    return 0
  fi
  local py
  if ! py="$(choose_python)"; then
    echo "0"
    return 0
  fi
  "${py}" - "${STATE_FILE}" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(data.get("__consecutive_failures", "0"))
except Exception:
    print("0")
PY
}

ensure_scheduler() {
  if is_running; then
    local pid
    pid="$(cat "${PID_FILE}")"
    local hb_age
    hb_age="$(heartbeat_age_sec)"

    # 1) heartbeat가 stale이면 재시작
    if [[ -n "${hb_age}" ]] && [[ "${hb_age}" =~ ^[0-9]+$ ]] && (( hb_age > MAX_HEARTBEAT_AGE_SEC )); then
      echo "[warn] scheduler process is alive but heartbeat is stale (${hb_age}s > ${MAX_HEARTBEAT_AGE_SEC}s). restarting..."
      stop_scheduler
      start_scheduler
      return 0
    fi

    # 2) 연속 실패 5회 이상이면 재시작
    local cons_fail
    cons_fail="$(consecutive_failures)"
    if [[ "${cons_fail}" =~ ^[0-9]+$ ]] && (( cons_fail >= 5 )); then
      echo "[warn] ${cons_fail} consecutive failures detected. restarting..."
      stop_scheduler
      start_scheduler
      return 0
    fi

    # 3) 마지막 trade 성공이 너무 오래됐으면 재시작 (6시간 = 21600초)
    local trade_age
    trade_age="$(last_trade_age_sec)"
    if [[ -n "${trade_age}" ]] && [[ "${trade_age}" =~ ^[0-9]+$ ]] && (( trade_age > 21600 )); then
      echo "[warn] last trade success was ${trade_age}s ago (>6h). restarting..."
      stop_scheduler
      start_scheduler
      return 0
    fi

    echo "[ok] scheduler already running (pid=${pid})"
    if [[ -n "${hb_age}" ]]; then
      echo "[info] heartbeat age: ${hb_age}s"
    fi
    if [[ -n "${trade_age}" ]]; then
      echo "[info] last trade age: ${trade_age}s"
    fi
    if [[ -n "${cons_fail}" ]] && [[ "${cons_fail}" != "0" ]]; then
      echo "[warn] consecutive failures: ${cons_fail}"
    fi
    return 0
  fi
  echo "[warn] scheduler not running. starting now..."
  start_scheduler
}

case "${ACTION}" in
  start)
    start_scheduler
    show_status
    ;;
  ensure)
    ensure_scheduler
    show_status
    ;;
  stop)
    stop_scheduler
    ;;
  restart)
    stop_scheduler
    start_scheduler
    show_status
    ;;
  status)
    show_status
    ;;
  *)
    echo "usage: bash scripts/vm_scheduler_manager.sh <start|ensure|stop|restart|status>"
    exit 2
    ;;
esac
