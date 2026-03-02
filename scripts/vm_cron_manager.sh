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

ensure_crontab() {
  if command -v crontab >/dev/null 2>&1; then
    return 0
  fi

  echo "[info] crontab not found. trying to install cron package..."
  echo "[info] user=$(id -un) uid=$(id -u)"

  _run_root() {
    if [[ "$(id -u)" == "0" ]]; then
      "$@"
      return $?
    fi
    if command -v sudo >/dev/null 2>&1; then
      sudo -n "$@" 2>/dev/null
      return $?
    fi
    return 127
  }

  _root_ok=true
  if ! _run_root true; then
    _root_ok=false
    echo "[warn] root 권한이 없어 패키지 설치를 수행할 수 없습니다."
    echo "[warn] VM에서 수동으로 cron 설치 후 다시 vm_cron_install을 실행해 주세요."
  fi

  if [[ "${_root_ok}" == "true" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "[info] package manager: apt-get"
      _run_root apt-get update -y || true
      _run_root apt-get install -y cron || _run_root apt-get install -y cronie || true
      _run_root systemctl enable --now cron || _run_root service cron start || _run_root cron || true
    elif command -v dnf >/dev/null 2>&1; then
      echo "[info] package manager: dnf"
      _run_root dnf install -y cronie || true
      _run_root systemctl enable --now crond || _run_root service crond start || true
    elif command -v yum >/dev/null 2>&1; then
      echo "[info] package manager: yum"
      _run_root yum install -y cronie || true
      _run_root systemctl enable --now crond || _run_root service crond start || true
    elif command -v apk >/dev/null 2>&1; then
      echo "[info] package manager: apk"
      _run_root apk add --no-cache dcron || _run_root apk add --no-cache cronie || true
      _run_root rc-update add crond || true
      _run_root service crond start || true
    elif command -v microdnf >/dev/null 2>&1; then
      echo "[info] package manager: microdnf"
      _run_root microdnf install -y cronie || true
      _run_root systemctl enable --now crond || _run_root service crond start || true
    else
      echo "[warn] 지원되는 패키지 매니저를 찾지 못했습니다. (apt/dnf/yum/apk/microdnf)"
    fi
  fi

  if ! command -v crontab >/dev/null 2>&1; then
    echo "[error] crontab command is still unavailable. install cron on VM first."
    exit 1
  fi
}

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
  ensure_crontab
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
  ensure_crontab
  local tmp
  tmp="$(mktemp)"
  read_crontab | strip_block > "${tmp}"
  crontab "${tmp}"
  rm -f "${tmp}"
  echo "[ok] VM cron removed."
}

show_block() {
  ensure_crontab
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
