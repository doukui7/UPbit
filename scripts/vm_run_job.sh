#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: bash scripts/vm_run_job.sh <upbit|health_check|daily_status|kiwoom_gold|kis_isa|kis_pension>"
  exit 1
fi

case "${MODE}" in
  upbit|health_check|daily_status|kiwoom_gold|kis_isa|kis_pension|account_sync|manual_order) ;;
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
  # 로컬 상태 파일 보존 (git reset 덮어쓰기 방지)
  cp signal_state.json /tmp/_signal_state_backup.json 2>/dev/null || true
  cp balance_cache.json /tmp/_balance_cache_backup.json 2>/dev/null || true
  cp trade_log.json /tmp/_trade_log_backup.json 2>/dev/null || true
  cp signal_test_orders.json /tmp/_signal_test_orders_backup.json 2>/dev/null || true
  cp logs/vm_scheduler_state.json /tmp/_vm_scheduler_state_backup.json 2>/dev/null || true
  git fetch origin --quiet 2>/dev/null || true
  git reset --hard origin/master --quiet 2>/dev/null || true
  cp /tmp/_signal_state_backup.json signal_state.json 2>/dev/null || true
  cp /tmp/_balance_cache_backup.json balance_cache.json 2>/dev/null || true
  cp /tmp/_trade_log_backup.json trade_log.json 2>/dev/null || true
  cp /tmp/_signal_test_orders_backup.json signal_test_orders.json 2>/dev/null || true
  cp /tmp/_vm_scheduler_state_backup.json logs/vm_scheduler_state.json 2>/dev/null || true
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
if TRADING_MODE="${MODE}" python scripts/github_action_trade.py 2>&1 | tee -a "${LOG_FILE}"; then
  echo "[$(date '+%F %T')] done mode=${MODE}" >> "${LOG_FILE}"

  # balance_cache / signal_state → GitHub Contents API push (git 충돌 완전 회피)
  if [[ "${MODE}" == "upbit" || "${MODE}" == "manual_order" || "${MODE}" == "account_sync" || "${MODE}" == "kis_pension" ]]; then
    if [[ -n "${GH_PAT:-}" ]]; then
      python -c "
import json, base64, os, sys
try:
    import urllib.request, urllib.error
except ImportError:
    sys.exit(0)

PAT = os.environ.get('GH_PAT', '')
if not PAT:
    sys.exit(0)

REPO = 'doukui7/UPbit'
MODE = '${MODE}'
FILES = [
    'balance_cache.json', 'signal_state.json', 'trade_log.json',
    'signal_test_orders.json', 'config/pension_orders.json',
    'logs/vm_scheduler_state.json',
]
ok = 0
for fpath in FILES:
    if not os.path.exists(fpath):
        continue
    api = f'https://api.github.com/repos/{REPO}/contents/{fpath}'
    hdrs = {'Authorization': f'token {PAT}', 'Accept': 'application/vnd.github+json', 'User-Agent': 'auto-trade-bot'}
    content = open(fpath, 'rb').read()
    encoded = base64.b64encode(content).decode()
    # GET sha
    sha = ''
    try:
        req = urllib.request.Request(api, headers=hdrs)
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode())
            sha = d.get('sha', '')
            remote = base64.b64decode(d.get('content', '').replace('\n', ''))
            if remote == content:
                continue  # 변경 없음
    except urllib.error.HTTPError as e:
        if e.code != 404:
            continue
    # PUT
    try:
        body = json.dumps({'message': f'auto: VM {MODE} sync {fpath}', 'content': encoded, 'sha': sha, 'committer': {'name': 'auto-trade-bot', 'email': 'bot@auto-trade'}}).encode()
        req = urllib.request.Request(api, data=body, headers=hdrs, method='PUT')
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status in (200, 201):
                ok += 1
    except Exception:
        pass
print(f'API push: {ok}/{len(FILES)} files')
" 2>&1 | tee -a "${LOG_FILE}" && echo "[$(date '+%F %T')] API push done" >> "${LOG_FILE}" \
        || echo "[$(date '+%F %T')] API push failed" >> "${LOG_FILE}"
    fi
  fi

  # 상태 파일 업데이트 — 스케줄러 외부 실행(수동/GitHub Actions)도 헬스체크에서 인식
  STATE_FILE="${LOG_DIR}/vm_scheduler_state.json"
  if command -v python >/dev/null 2>&1; then
    python -c "
import json, os, time
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('Asia/Seoul'))
except Exception:
    now = datetime.now()
mk = now.strftime('%Y%m%d%H')
path = '${STATE_FILE}'
state = {}
if os.path.exists(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except Exception:
        pass
state['${MODE}'] = mk
state['__heartbeat_epoch'] = f'{time.time():.3f}'
state['__heartbeat_kst'] = now.strftime('%Y-%m-%d %H:%M:%S')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(state, f, ensure_ascii=False, indent=2)
" 2>/dev/null || true
  fi
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
