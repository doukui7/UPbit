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
BALANCE_PUSH_INTERVAL_SEC = 10 * 60  # 잔고 git push 주기 (초)
HEARTBEAT_EPOCH_KEY = "__heartbeat_epoch"
HEARTBEAT_KST_KEY = "__heartbeat_kst"
STARTED_AT_KEY = "__started_at_kst"
LAST_ERROR_KEY = "__last_error"
_LAST_BALANCE_SYNC = 0.0  # 마지막 잔고 조회 시각 (epoch)
_LAST_BALANCE_PUSH = 0.0  # 마지막 잔고 push 시각 (epoch)


def _is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5


RULES: list[ScheduleRule] = [
    ScheduleRule(
        mode="upbit",
        label="코인 자동매매",
        is_due=lambda dt: dt.minute == 0 and dt.hour in (1, 5, 9, 13, 17, 21),
    ),
    ScheduleRule(
        mode="health_check",
        label="헬스체크",
        is_due=lambda dt: dt.minute == 5 and dt.hour in (1, 5, 9, 13, 17, 21),
    ),
    ScheduleRule(
        mode="daily_status",
        label="일일 자산 현황",
        is_due=lambda dt: dt.minute == 0 and dt.hour == 9 and _is_weekday(dt),
    ),
    ScheduleRule(
        mode="kiwoom_gold",
        label="키움 금현물",
        is_due=lambda dt: dt.minute == 5 and dt.hour == 15 and _is_weekday(dt),
    ),
    ScheduleRule(
        mode="kis_isa",
        label="KIS ISA",
        is_due=lambda dt: dt.minute == 10 and dt.hour == 15 and dt.weekday() == 4,
    ),
    ScheduleRule(
        mode="kis_pension",
        label="KIS 연금저축",
        is_due=lambda dt: dt.minute == 20
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


def _run_mode(mode: str, label: str) -> bool:
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
            return False
    except subprocess.TimeoutExpired:
        logging.error("실행 타임아웃: %s (%s)", label, mode)
        return False
    except Exception as e:
        logging.exception("실행 예외: %s (%s): %s", label, mode, e)
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
        ok = _run_mode(mode, f"1회성 예약: {label}")
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


def _push_balance_cache() -> bool:
    """balance_cache.json을 git commit+push. 성공 시 True."""
    global _LAST_BALANCE_PUSH
    now_epoch = time.time()
    if (now_epoch - _LAST_BALANCE_PUSH) < BALANCE_PUSH_INTERVAL_SEC:
        return False

    _LAST_BALANCE_PUSH = now_epoch

    if not BALANCE_CACHE_FILE.exists():
        return False

    try:
        subprocess.run(
            ["git", "remote", "set-url", "origin",
             "git@github.com:doukui7/UPbit.git"],
            cwd=str(REPO_DIR), capture_output=True, timeout=5,
        )
        subprocess.run(
            ["git", "add", "-f", "balance_cache.json"],
            cwd=str(REPO_DIR), capture_output=True, timeout=5,
        )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(REPO_DIR), capture_output=True, timeout=5,
        )
        if diff.returncode == 0:
            # 변경 없음
            subprocess.run(
                ["git", "remote", "set-url", "origin",
                 "https://github.com/doukui7/UPbit.git"],
                cwd=str(REPO_DIR), capture_output=True, timeout=5,
            )
            return False

        subprocess.run(
            ["git", "-c", "user.name=auto-trade-bot",
             "-c", "user.email=bot@auto-trade",
             "commit", "-m", "auto: 잔고 캐시 동기화"],
            cwd=str(REPO_DIR), capture_output=True, timeout=10,
        )
        push_res = subprocess.run(
            ["git", "push", "origin", "master"],
            cwd=str(REPO_DIR), capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "remote", "set-url", "origin",
             "https://github.com/doukui7/UPbit.git"],
            cwd=str(REPO_DIR), capture_output=True, timeout=5,
        )
        if push_res.returncode == 0:
            logging.info("잔고 캐시 push 완료")
            return True
        else:
            logging.warning("잔고 캐시 push 실패: %s", push_res.stderr[:200] if push_res.stderr else "")
            return False
    except Exception as e:
        logging.warning("잔고 캐시 push 예외: %s", e)
        # remote URL 복원
        subprocess.run(
            ["git", "remote", "set-url", "origin",
             "https://github.com/doukui7/UPbit.git"],
            cwd=str(REPO_DIR), capture_output=True, timeout=5,
        )
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
            mk = _minute_key(now)
            state_dirty = False
            has_heavy_job = False

            try:
                if _run_due_oneoff_jobs(now, state):
                    state_dirty = True
                    has_heavy_job = True
                    if LAST_ERROR_KEY in state:
                        state.pop(LAST_ERROR_KEY, None)

                for rule in RULES:
                    if not rule.is_due(now):
                        continue
                    if state.get(rule.mode) == mk:
                        continue
                    _run_mode(rule.mode, rule.label)
                    state[rule.mode] = mk
                    state_dirty = True
                    has_heavy_job = True
                    if LAST_ERROR_KEY in state:
                        state.pop(LAST_ERROR_KEY, None)
            except Exception as e:
                # 루프 전체가 죽지 않도록 방어하고 다음 tick에서 계속 진행한다.
                logging.exception("스케줄 루프 예외: %s", e)
                state[LAST_ERROR_KEY] = f"{now.strftime('%Y-%m-%d %H:%M:%S')} | {type(e).__name__}: {e}"
                state_dirty = True

            # ── 잔고 실시간 동기화 (다른 작업이 없을 때) ──
            if not has_heavy_job:
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
