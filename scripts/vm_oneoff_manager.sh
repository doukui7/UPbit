#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-show}"
MODE="${2:-}"
RUN_AT_KST="${3:-}"
NOTE="${4:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
ONEOFF_FILE="${LOG_DIR}/vm_oneoff_jobs.json"

mkdir -p "${LOG_DIR}"

is_valid_mode() {
  case "${1:-}" in
    upbit|health_check|daily_status|kiwoom_gold|kis_isa|kis_pension)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

add_job() {
  if [[ -z "${MODE}" || -z "${RUN_AT_KST}" ]]; then
    echo "[error] add requires: <mode> <run_at_kst>"
    echo "example: bash scripts/vm_oneoff_manager.sh add kis_pension \"2026-03-05 15:20\" \"453540 테스트\""
    exit 2
  fi
  if ! is_valid_mode "${MODE}"; then
    echo "[error] unsupported mode: ${MODE}"
    exit 2
  fi

  python - "${ONEOFF_FILE}" "${MODE}" "${RUN_AT_KST}" "${NOTE}" <<'PY'
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

path, mode, run_at_raw, note = sys.argv[1:5]

def parse_run_at(raw: str):
    txt = str(raw or "").strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(txt, fmt)
            return dt.replace(tzinfo=timezone(timedelta(hours=9)))
        except Exception:
            pass
    return None

run_at = parse_run_at(run_at_raw)
if run_at is None:
    print("[error] invalid run_at_kst format. use YYYY-MM-DD HH:MM[:SS]")
    raise SystemExit(2)

jobs = []
if os.path.exists(path):
    try:
        loaded = json.loads(open(path, "r", encoding="utf-8").read())
        if isinstance(loaded, list):
            jobs = [x for x in loaded if isinstance(x, dict)]
    except Exception:
        jobs = []

job_id = f"once-{mode}-{run_at.strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
job = {
    "id": job_id,
    "mode": mode,
    "run_at_kst": run_at.strftime("%Y-%m-%d %H:%M:%S"),
    "run_at_epoch": run_at.timestamp(),
    "status": "pending",
    "note": str(note or "").strip(),
    "created_at_kst": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"),
}
jobs.append(job)

with open(path, "w", encoding="utf-8") as f:
    json.dump(jobs, f, ensure_ascii=False, indent=2)

print(f"[ok] oneoff added: id={job_id} mode={mode} run_at={job['run_at_kst']}")
if job["note"]:
    print(f"[info] note: {job['note']}")
PY
}

show_jobs() {
  python - "${ONEOFF_FILE}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    print("[info] oneoff file not found")
    raise SystemExit(0)

try:
    jobs = json.loads(open(path, "r", encoding="utf-8").read())
except Exception as e:
    print(f"[error] failed to read oneoff file: {e}")
    raise SystemExit(1)

if not isinstance(jobs, list) or len(jobs) == 0:
    print("[info] oneoff jobs empty")
    raise SystemExit(0)

print(f"[info] oneoff jobs ({len(jobs)}):")
for j in jobs:
    if not isinstance(j, dict):
        continue
    jid = str(j.get("id", ""))
    mode = str(j.get("mode", ""))
    run_at = str(j.get("run_at_kst", ""))
    status = str(j.get("status", "pending"))
    executed = str(j.get("executed_at_kst", ""))
    note = str(j.get("note", ""))
    line = f"- {jid} | {mode} | {run_at} | {status}"
    if executed:
        line += f" | executed={executed}"
    if note:
        line += f" | note={note}"
    print(line)
PY
}

remove_job() {
  local job_id="${MODE}"
  if [[ -z "${job_id}" ]]; then
    echo "[error] remove requires: <job_id>"
    exit 2
  fi

  python - "${ONEOFF_FILE}" "${job_id}" <<'PY'
import json
import os
import sys

path, job_id = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print("[info] oneoff file not found")
    raise SystemExit(0)

jobs = []
try:
    loaded = json.loads(open(path, "r", encoding="utf-8").read())
    if isinstance(loaded, list):
        jobs = [x for x in loaded if isinstance(x, dict)]
except Exception:
    pass

before = len(jobs)
jobs = [j for j in jobs if str(j.get("id", "")) != job_id]
after = len(jobs)

with open(path, "w", encoding="utf-8") as f:
    json.dump(jobs, f, ensure_ascii=False, indent=2)

if after < before:
    print(f"[ok] oneoff removed: {job_id}")
else:
    print(f"[warn] oneoff id not found: {job_id}")
PY
}

clear_jobs() {
  python - "${ONEOFF_FILE}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
if not os.path.exists(path):
    print("[info] oneoff file not found")
    raise SystemExit(0)
with open(path, "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=2)
print("[ok] oneoff jobs cleared")
PY
}

case "${ACTION}" in
  add)
    add_job
    show_jobs
    ;;
  show)
    show_jobs
    ;;
  remove)
    remove_job
    show_jobs
    ;;
  clear)
    clear_jobs
    ;;
  *)
    echo "usage: bash scripts/vm_oneoff_manager.sh <add|show|remove|clear> [mode|job_id] [run_at_kst] [note]"
    exit 2
    ;;
esac

