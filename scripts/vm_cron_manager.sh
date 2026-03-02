#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-show}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

BLOCK_START="# >>> UPBIT_VM_CRON >>>"
BLOCK_END="# <<< UPBIT_VM_CRON <<<"

if [[ ! -f "${REPO_DIR}/scripts/vm_run_job.sh" ]]; then
  echo "[error] scripts/vm_run_job.sh not found in ${REPO_DIR}"
  exit 1
fi

read_crontab() {
  crontab -l 2>/dev/null || true
}

strip_block() {
  awk -v s="${BLOCK_START}" -v e="${BLOCK_END}" '
    $0 == s {skip = 1; next}
    $0 == e {skip = 0; next}
    !skip {print}
  '
}

render_block() {
  cat <<EOF
${BLOCK_START}
CRON_TZ=Asia/Seoul
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

0 1,5,9,13,17,21 * * * cd ${REPO_DIR} && bash scripts/vm_run_job.sh upbit
5 1,5,9,13,17,21 * * * cd ${REPO_DIR} && bash scripts/vm_run_job.sh health_check
0 9 * * 1-5 cd ${REPO_DIR} && bash scripts/vm_run_job.sh daily_status
5 15 * * 1-5 cd ${REPO_DIR} && bash scripts/vm_run_job.sh kiwoom_gold
10 15 * * 5 cd ${REPO_DIR} && bash scripts/vm_run_job.sh kis_isa
10 15 25-31 * 1-5 cd ${REPO_DIR} && bash scripts/vm_run_job.sh kis_pension
${BLOCK_END}
EOF
}

install_block() {
  chmod +x "${REPO_DIR}/scripts/vm_run_job.sh" "${REPO_DIR}/scripts/vm_cron_manager.sh" || true
  local tmp
  tmp="$(mktemp)"
  {
    read_crontab | strip_block
    echo
    render_block
  } > "${tmp}"
  crontab "${tmp}"
  rm -f "${tmp}"
  echo "[ok] VM cron installed."
}

remove_block() {
  local tmp
  tmp="$(mktemp)"
  read_crontab | strip_block > "${tmp}"
  crontab "${tmp}"
  rm -f "${tmp}"
  echo "[ok] VM cron removed."
}

show_block() {
  local out
  out="$(read_crontab | awk -v s="${BLOCK_START}" -v e="${BLOCK_END}" '
    $0 == s {in_block = 1; print; next}
    in_block {print}
    $0 == e {in_block = 0}
  ')"
  if [[ -z "${out}" ]]; then
    echo "[info] VM cron block not found."
  else
    printf '%s\n' "${out}"
  fi
}

case "${ACTION}" in
  install)
    install_block
    show_block
    ;;
  remove)
    remove_block
    ;;
  show|list)
    show_block
    ;;
  *)
    echo "usage: bash scripts/vm_cron_manager.sh <install|show|remove>"
    exit 2
    ;;
esac
