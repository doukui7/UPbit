#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
PID_FILE="${LOG_DIR}/vm_scheduler.pid"

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
    fi
    if [[ -f "${LOG_DIR}/upbit.log" ]]; then
      echo "[info] latest upbit log:"
      tail -n 40 "${LOG_DIR}/upbit.log" || true
    fi
    if [[ -f "${LOG_DIR}/health_check.log" ]]; then
      echo "[info] latest health_check log:"
      tail -n 20 "${LOG_DIR}/health_check.log" || true
    fi
  else
    echo "[info] scheduler not running"
  fi
}

ensure_scheduler() {
  if is_running; then
    echo "[ok] scheduler already running (pid=$(cat "${PID_FILE}"))"
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
