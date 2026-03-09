#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


KST = ZoneInfo("Asia/Seoul") if ZoneInfo else None


@dataclass(frozen=True)
class ScheduleRule:
    mode: str
    label: str
    is_due: Callable[[datetime], bool]


RUNNING = True
REPO_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_DIR / "logs"
STATE_FILE = LOG_DIR / "vm_scheduler_state.json"
LOCK_FILE = LOG_DIR / "vm_scheduler.lock"
RESERVED_ORDER_FILE = LOG_DIR / "vm_reserved_orders.json"
LEGACY_ONEOFF_FILE = LOG_DIR / "vm_oneoff_jobs.json"
BALANCE_CACHE_FILE = REPO_DIR / "balance_cache.json"
CHECK_INTERVAL_SEC = 5
HEARTBEAT_INTERVAL_SEC = 30
BALANCE_SYNC_INTERVAL_SEC = 30       # 잔고 조회 주기 (초)
BALANCE_PUSH_INTERVAL_SEC = 5 * 60   # 잔고 git push 주기 (초) — 5분마다
HEARTBEAT_EPOCH_KEY = "__heartbeat_epoch"
HEARTBEAT_KST_KEY = "__heartbeat_kst"
STARTED_AT_KEY = "__started_at_kst"
LAST_ERROR_KEY = "__last_error"
LAST_TRADE_EPOCH_KEY = "__last_trade_epoch"
LAST_TRADE_KST_KEY = "__last_trade_kst"
LAST_TRADE_MODE_KEY = "__last_trade_mode"
CONSECUTIVE_FAIL_KEY = "__consecutive_failures"
_LAST_BALANCE_SYNC = 0.0  # 마지막 잔고 조회 시각 (epoch)
_LAST_BALANCE_PUSH = 0.0  # 마지막 잔고 push 시각 (epoch)


def _is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5


RULES: list[ScheduleRule] = [
    ScheduleRule(
        mode="upbit",
        label="코인 자동매매",
        is_due=lambda dt: dt.minute <= 10 and dt.hour in (1, 5, 9, 13, 17, 21),
    ),
    ScheduleRule(
        mode="health_check",
        label="헬스체크",
        is_due=lambda dt: 5 <= dt.minute <= 15 and dt.hour in (1, 5, 9, 13, 17, 21),
    ),
    ScheduleRule(
        mode="daily_status",
        label="일일 자산 현황",
        is_due=lambda dt: dt.minute <= 10 and dt.hour == 9 and _is_weekday(dt),
    ),
    ScheduleRule(
        mode="kiwoom_gold",
        label="키움 금현물",
        is_due=lambda dt: 5 <= dt.minute <= 15 and dt.hour == 15 and _is_weekday(dt),
    ),
    ScheduleRule(
        mode="kis_isa",
        label="KIS ISA",
        is_due=lambda dt: 10 <= dt.minute <= 20 and dt.hour == 15 and dt.weekday() == 4,
    ),
    ScheduleRule(
        mode="kis_pension",
        label="KIS 연금저축",
        is_due=lambda dt: 20 <= dt.minute <= 30
        and dt.hour == 15
        and _is_weekday(dt)
        and 25 <= dt.day <= 31,
    ),
]
RULE_LABELS = {rule.mode: rule.label for rule in RULES}


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "vm_scheduler.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _handle_signal(signum, _frame) -> None:
    global RUNNING
    logging.info("종료 시그널 수신: %s", signum)
    RUNNING = False


def _load_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        logging.warning("상태 파일 로드 실패: %s", e)
    return {}


def _save_state(state: dict[str, str]) -> None:
    try:
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logging.error("상태 파일 저장 실패: %s", e)


def _parse_oneoff_run_at(raw: str) -> datetime | None:
    txt = str(raw or "").strip().replace("T", " ")
    if not txt:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(txt, fmt)
            if KST is not None:
                return dt.replace(tzinfo=KST)
            return dt
        except Exception:
            continue
    return None


def _load_oneoff_jobs() -> list[dict]:
    target = RESERVED_ORDER_FILE
    source = target
    if not target.exists() and LEGACY_ONEOFF_FILE.exists():
        source = LEGACY_ONEOFF_FILE
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            jobs = [x for x in raw if isinstance(x, dict)]
            if source == LEGACY_ONEOFF_FILE and jobs:
                # 구버전 파일을 새 예약주문 파일로 마이그레이션
                _save_oneoff_jobs(jobs)
            return jobs
    except Exception as e:
        logging.warning("1회성 작업 파일 로드 실패: %s", e)
    return []


def _save_oneoff_jobs(jobs: list[dict]) -> None:
    try:
        RESERVED_ORDER_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESERVED_ORDER_FILE.write_text(
            json.dumps(jobs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logging.error("1회성 작업 파일 저장 실패: %s", e)


def _acquire_lock() -> object:
    import fcntl

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fp = open(LOCK_FILE, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logging.error("이미 스케줄러가 실행 중입니다. lock=%s", LOCK_FILE)
        sys.exit(2)
    return lock_fp


def _now_kst() -> datetime:
    if KST is None:
        # Fallback: VM timezone assumed KST.
        return datetime.now()
    return datetime.now(KST)


def _self_heal(state: dict[str, str]) -> None:
    """연속 실패 시 git pull → 코드 갱신 → 상태 초기화."""
    logging.info("자가 복구 시작: git fetch + reset (상태 파일 보존)")
    cwd = str(REPO_DIR)
    try:
        # 상태 파일 백업
        for fn in ("signal_state.json", "balance_cache.json", "trade_log.json", "signal_test_orders.json"):
            src = REPO_DIR / fn
            if src.exists():
                subprocess.run(
                    ["cp", str(src), f"/tmp/_{fn.replace('.', '_')}_heal.json"],
                    capture_output=True, timeout=5,
                )
        # git reset --hard (코드 갱신)
        subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=cwd, capture_output=True, timeout=15)
        subprocess.run(["git", "reset", "--hard", "origin/master", "--quiet"], cwd=cwd, capture_output=True, timeout=15)
        # 상태 파일 복원
        for fn in ("signal_state.json", "balance_cache.json", "trade_log.json", "signal_test_orders.json"):
            bak = f"/tmp/_{fn.replace('.', '_')}_heal.json"
            subprocess.run(["cp", bak, str(REPO_DIR / fn)], capture_output=True, timeout=5)
        state[CONSECUTIVE_FAIL_KEY] = "0"
        state[LAST_ERROR_KEY] = f"{_now_kst().strftime('%Y-%m-%d %H:%M:%S')} | self-heal: git reset 완료"
        logging.info("자가 복구 완료: 코드 갱신됨")
    except Exception as e:
        logging.error("자가 복구 실패: %s", e)


def _send_scheduler_telegram(message: str) -> None:
    """스케줄러 이벤트 텔레그램 알림."""
    try:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not bot_token or not chat_id:
            # .vm_runtime_env에서 로드 시도
            bot_token = bot_token or _read_env_file_value("TELEGRAM_BOT_TOKEN")
            chat_id = chat_id or _read_env_file_value("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            return
        import urllib.request
        import urllib.parse
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": chat_id, "text": message, "parse_mode": "HTML",
        }).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        logging.debug("텔레그램 알림 실패: %s", e)


def _run_mode(mode: str, label: str, state: dict[str, str] | None = None) -> bool:
    cmd = ["bash", "scripts/vm_run_job.sh", mode]
    logging.info("실행 시작: %s (%s)", label, mode)
    started = time.time()
    timeout_sec = 60 * 80 if mode == "kis_pension" else 60 * 40
    try:
        res = subprocess.run(
            cmd,
            cwd=str(REPO_DIR),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        elapsed = time.time() - started
        if res.returncode == 0:
            logging.info("실행 완료: %s (%s) %.1fs", label, mode, elapsed)
            if state is not None:
                now = _now_kst()
                state[LAST_TRADE_EPOCH_KEY] = f"{time.time():.3f}"
                state[LAST_TRADE_KST_KEY] = now.strftime("%Y-%m-%d %H:%M:%S")
                state[LAST_TRADE_MODE_KEY] = mode
                state[CONSECUTIVE_FAIL_KEY] = "0"
            return True
        else:
            logging.error(
                "실행 실패: %s (%s) code=%s %.1fs stdout=%s stderr=%s",
                label,
                mode,
                res.returncode,
                elapsed,
                (res.stdout or "").strip()[:600],
                (res.stderr or "").strip()[:600],
            )
            if state is not None:
                prev = int(state.get(CONSECUTIVE_FAIL_KEY, "0") or "0")
                state[CONSECUTIVE_FAIL_KEY] = str(prev + 1)
            return False
    except subprocess.TimeoutExpired:
        logging.error("실행 타임아웃: %s (%s)", label, mode)
        if state is not None:
            prev = int(state.get(CONSECUTIVE_FAIL_KEY, "0") or "0")
            state[CONSECUTIVE_FAIL_KEY] = str(prev + 1)
        return False
    except Exception as e:
        logging.exception("실행 예외: %s (%s): %s", label, mode, e)
        if state is not None:
            prev = int(state.get(CONSECUTIVE_FAIL_KEY, "0") or "0")
            state[CONSECUTIVE_FAIL_KEY] = str(prev + 1)
        return False


def _run_due_oneoff_jobs(now: datetime, state: dict[str, str]) -> bool:
    jobs = _load_oneoff_jobs()
    if not jobs:
        return False

    jobs_dirty = False
    now_epoch = now.timestamp()
    for job in jobs:
        status = str(job.get("status", "pending")).strip().lower()
        if status != "pending":
            continue

        mode = str(job.get("mode", "")).strip()
        if mode not in RULE_LABELS:
            job["status"] = "invalid"
            job["error"] = f"지원하지 않는 mode: {mode}"
            jobs_dirty = True
            continue

        run_at_dt = _parse_oneoff_run_at(str(job.get("run_at_kst", "")))
        if run_at_dt is None:
            job["status"] = "invalid"
            job["error"] = "run_at_kst 형식 오류 (YYYY-MM-DD HH:MM[:SS])"
            jobs_dirty = True
            continue

        normalized_run_at = run_at_dt.strftime("%Y-%m-%d %H:%M:%S")
        if str(job.get("run_at_kst", "")) != normalized_run_at:
            job["run_at_kst"] = normalized_run_at
            jobs_dirty = True

        run_epoch = run_at_dt.timestamp()
        prev_epoch = None
        try:
            prev_epoch = float(job.get("run_at_epoch"))
        except Exception:
            prev_epoch = None
        if prev_epoch is None or abs(prev_epoch - run_epoch) > 0.001:
            job["run_at_epoch"] = run_epoch
            jobs_dirty = True

        if now_epoch + 0.001 < run_epoch:
            continue

        job_id = str(job.get("id", "")).strip() or f"once-{mode}-{int(run_epoch)}"
        dedupe_key = f"once::{job_id}"
        if state.get(dedupe_key):
            job["status"] = "done"
            if not str(job.get("executed_at_kst", "")).strip():
                job["executed_at_kst"] = now.strftime("%Y-%m-%d %H:%M:%S")
            jobs_dirty = True
            continue

        label = RULE_LABELS.get(mode, mode)
        ok = _run_mode(mode, f"1회성 예약: {label}", state)
        if ok:
            job["status"] = "done"
        else:
            job["status"] = "failed"
        job["executed_at_kst"] = now.strftime("%Y-%m-%d %H:%M:%S")
        state[dedupe_key] = _minute_key(now)
        jobs_dirty = True

    if jobs_dirty:
        _save_oneoff_jobs(jobs)
    return jobs_dirty


def _minute_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def _slot_key(dt: datetime) -> str:
    """시간 슬롯 키 — 같은 시간대 재실행 방지 (10분 윈도우 대응)."""
    return dt.strftime("%Y%m%d%H")


# ═══════════════════════════════════════════
# 잔고 실시간 동기화 (30초마다 조회, 10분마다 push)
# ═══════════════════════════════════════════
def _sync_balance() -> bool:
    """잔고+현재가 조회 후 balance_cache.json 로컬 저장. 성공 시 True."""
    global _LAST_BALANCE_SYNC
    now_epoch = time.time()
    if (now_epoch - _LAST_BALANCE_SYNC) < BALANCE_SYNC_INTERVAL_SEC:
        return False

    _LAST_BALANCE_SYNC = now_epoch

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    ak = os.environ.get("UPBIT_ACCESS_KEY", "")
    sk = os.environ.get("UPBIT_SECRET_KEY", "")
    if not ak or not sk:
        return False

    try:
        sys.path.insert(0, str(REPO_DIR))
        from src.trading.upbit_trader import UpbitTrader
        trader = UpbitTrader(ak, sk)
        balances = trader.get_all_balances()
        if not isinstance(balances, dict) or not balances:
            return False

        # 보유 코인 현재가 조회
        prices = {}
        for sym in list(balances.keys()):
            if sym == "KRW":
                continue
            ticker = f"KRW-{sym}"
            try:
                p = float(trader.get_current_price(ticker) or 0)
                if p > 0:
                    prices[ticker] = p
            except Exception:
                pass

        now_kst = _now_kst()
        cache = {
            "updated_at": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "balances": {str(k): float(v) for k, v in balances.items()},
        }
        if prices:
            cache["prices"] = prices

        BALANCE_CACHE_FILE.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    except Exception as e:
        logging.warning("잔고 동기화 실패: %s", e)
        return False


def _read_env_file_value(key: str) -> str:
    """`.vm_runtime_env` 파일에서 key 값을 읽는다 (실행 중 프로세스 env에 없을 때 fallback)."""
    env_file = REPO_DIR / ".vm_runtime_env"
    if not env_file.exists():
        return ""
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}="):
                val = line[len(key) + 1:]
                # printf %q 포맷 해제: $'...' 또는 그대로
                if val.startswith("$'") and val.endswith("'"):
                    val = val[2:-1].encode().decode("unicode_escape")
                return val.strip()
    except Exception:
        pass
    return ""


def _get_gh_pat() -> str:
    """GH_PAT를 os.environ → .vm_runtime_env 순서로 조회."""
    pat = os.environ.get("GH_PAT", "")
    if not pat:
        pat = _read_env_file_value("GH_PAT")
    return pat


def _push_file_via_api(gh_pat: str, filepath: str, commit_msg: str) -> bool:
    """GitHub Contents API로 단일 파일 업데이트. git 작업 없이 atomic PUT."""
    import base64
    try:
        import urllib.request
        import urllib.error

        repo = "doukui7/UPbit"
        api_url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
        headers = {
            "Authorization": f"token {gh_pat}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "auto-trade-bot",
        }

        local_path = REPO_DIR / filepath
        if not local_path.exists():
            return False
        content = local_path.read_bytes()
        encoded = base64.b64encode(content).decode()

        # 1) GET: 현재 SHA 조회
        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                sha = data.get("sha", "")
                # 내용 동일하면 스킵
                remote_content = base64.b64decode(data.get("content", "").replace("\n", ""))
                if remote_content == content:
                    return False  # 변경 없음
        except urllib.error.HTTPError as e:
            if e.code == 404:
                sha = ""  # 파일 미존재 → 새 생성
            else:
                logging.warning("Contents API GET 실패 (%s): %s", filepath, e)
                return False

        # 2) PUT: 파일 업데이트 (atomic)
        body = json.dumps({
            "message": commit_msg,
            "content": encoded,
            "sha": sha,
            "committer": {"name": "auto-trade-bot", "email": "bot@auto-trade"},
        }).encode()
        req = urllib.request.Request(api_url, data=body, headers=headers, method="PUT")
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status in (200, 201):
                return True
        return False
    except Exception as e:
        logging.warning("Contents API PUT 실패 (%s): %s", filepath, e)
        return False


def _push_balance_cache() -> bool:
    """GitHub Contents API로 상태 파일 동기화. git 충돌 완전 회피."""
    global _LAST_BALANCE_PUSH
    now_epoch = time.time()
    if (now_epoch - _LAST_BALANCE_PUSH) < BALANCE_PUSH_INTERVAL_SEC:
        return False

    gh_pat = _get_gh_pat()
    if not gh_pat:
        return False

    _files = [
        "balance_cache.json",
        "signal_state.json",
        "trade_log.json",
        "logs/vm_scheduler_state.json",
    ]
    ok_count = 0
    for f in _files:
        try:
            if _push_file_via_api(gh_pat, f, f"auto: sync {f}"):
                ok_count += 1
        except Exception as e:
            logging.debug("API push 예외 (%s): %s", f, e)

    if ok_count > 0:
        _LAST_BALANCE_PUSH = now_epoch
        logging.info("Contents API push 완료 (%d/%d 파일)", ok_count, len(_files))
    else:
        # 전체 실패 시 60초 후 재시도 허용 (5분 대기 방지)
        _LAST_BALANCE_PUSH = now_epoch - BALANCE_PUSH_INTERVAL_SEC + 60
        logging.warning("Contents API push 전체 실패 (%d 파일) — 60초 후 재시도", len(_files))
        # 로컬 git도 원격과 동기화 (코드 업데이트 수신)
        try:
            subprocess.run(
                ["git", "fetch", "origin"], cwd=str(REPO_DIR),
                capture_output=True, timeout=15,
            )
            subprocess.run(
                ["git", "reset", "--hard", "origin/master"], cwd=str(REPO_DIR),
                capture_output=True, timeout=15,
            )
        except Exception:
            pass
        return True
    return False


def _touch_heartbeat(state: dict[str, str], now: datetime, *, force: bool = False) -> bool:
    """
    상태 파일에 heartbeat를 기록한다.
    - force=False일 때는 디스크 쓰기 과다를 피하려고 일정 주기(HEARTBEAT_INTERVAL_SEC)로만 갱신
    """
    cur_epoch = time.time()
    prev_epoch_raw = state.get(HEARTBEAT_EPOCH_KEY, "0")
    try:
        prev_epoch = float(prev_epoch_raw)
    except Exception:
        prev_epoch = 0.0
    if not force and (cur_epoch - prev_epoch) < HEARTBEAT_INTERVAL_SEC:
        return False

    state[HEARTBEAT_EPOCH_KEY] = f"{cur_epoch:.3f}"
    state[HEARTBEAT_KST_KEY] = now.strftime("%Y-%m-%d %H:%M:%S")
    return True


def main() -> int:
    _setup_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _lock_fp = _acquire_lock()

    state = _load_state()
    state[STARTED_AT_KEY] = _now_kst().strftime("%Y-%m-%d %H:%M:%S")
    _touch_heartbeat(state, _now_kst(), force=True)
    _save_state(state)
    logging.info("VM 파이썬 스케줄러 시작 (repo=%s)", REPO_DIR)
    logging.info(
        "스케줄: upbit[01/05/09/13/17/21:00], health[+5m], daily[평일09:00], "
        "gold[평일15:05], isa[금15:10], pension[25~31평일15:20], oneoff[vm_reserved_orders.json]"
    )
    logging.info(
        "잔고 동기화: 조회=%ds, push=%ds (다른 작업 실행 중에는 건너뜀)",
        BALANCE_SYNC_INTERVAL_SEC, BALANCE_PUSH_INTERVAL_SEC,
    )

    try:
        while RUNNING:
            now = _now_kst()
            sk = _slot_key(now)
            state_dirty = False
            try:
                if _run_due_oneoff_jobs(now, state):
                    state_dirty = True
                    if LAST_ERROR_KEY in state:
                        state.pop(LAST_ERROR_KEY, None)

                for rule in RULES:
                    if not rule.is_due(now):
                        continue
                    if state.get(rule.mode) == sk:
                        continue
                    ok = _run_mode(rule.mode, rule.label, state)
                    state[rule.mode] = sk
                    state_dirty = True
                    if ok and LAST_ERROR_KEY in state:
                        state.pop(LAST_ERROR_KEY, None)
                    # 연속 실패 시 텔레그램 알림 + 코드 갱신
                    if not ok:
                        cons_fail = int(state.get(CONSECUTIVE_FAIL_KEY, "0") or "0")
                        if cons_fail == 2:
                            _send_scheduler_telegram(
                                f"⚠️ <b>VM 스케줄러 연속 {cons_fail}회 실패</b>\n"
                                f"모드: {rule.mode} ({rule.label})\n"
                                f"시각: {now.strftime('%m-%d %H:%M')}"
                            )
                        if cons_fail >= 3:
                            logging.warning(
                                "연속 %d회 실패 — git pull 후 재시도", cons_fail
                            )
                            _send_scheduler_telegram(
                                f"🔄 <b>VM 스케줄러 자가 복구 시작</b>\n"
                                f"연속 {cons_fail}회 실패 → git reset + 코드 갱신"
                            )
                            _self_heal(state)
            except Exception as e:
                # 루프 전체가 죽지 않도록 방어하고 다음 tick에서 계속 진행한다.
                logging.exception("스케줄 루프 예외: %s", e)
                state[LAST_ERROR_KEY] = f"{now.strftime('%Y-%m-%d %H:%M:%S')} | {type(e).__name__}: {e}"
                state_dirty = True

            # ── 잔고 실시간 동기화 (항상 실행 — heavy job 여부 무관) ──
            try:
                _sync_balance()
                _push_balance_cache()
            except Exception as e:
                logging.debug("잔고 동기화 예외 (무시): %s", e)

            if _touch_heartbeat(state, now, force=state_dirty):
                state_dirty = True
            if state_dirty:
                _save_state(state)

            # Keep loop lightweight and stable.
            time.sleep(CHECK_INTERVAL_SEC)
    finally:
        logging.info("VM 파이썬 스케줄러 종료")
        _save_state(state)
        try:
            _lock_fp.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
