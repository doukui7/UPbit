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
CHECK_INTERVAL_SEC = 5


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
        is_due=lambda dt: dt.minute == 10
        and dt.hour == 15
        and _is_weekday(dt)
        and 25 <= dt.day <= 31,
    ),
]


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


def _run_mode(mode: str, label: str) -> None:
    cmd = ["bash", "scripts/vm_run_job.sh", mode]
    logging.info("실행 시작: %s (%s)", label, mode)
    started = time.time()
    try:
        res = subprocess.run(
            cmd,
            cwd=str(REPO_DIR),
            text=True,
            capture_output=True,
            timeout=60 * 40,
        )
        elapsed = time.time() - started
        if res.returncode == 0:
            logging.info("실행 완료: %s (%s) %.1fs", label, mode, elapsed)
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
    except subprocess.TimeoutExpired:
        logging.error("실행 타임아웃: %s (%s)", label, mode)
    except Exception as e:
        logging.exception("실행 예외: %s (%s): %s", label, mode, e)


def _minute_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def main() -> int:
    _setup_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _lock_fp = _acquire_lock()

    state = _load_state()
    logging.info("VM 파이썬 스케줄러 시작 (repo=%s)", REPO_DIR)
    logging.info(
        "스케줄: upbit[01/05/09/13/17/21:00], health[+5m], daily[평일09:00], "
        "gold[평일15:05], isa[금15:10], pension[25~31평일15:10]"
    )

    try:
        while RUNNING:
            now = _now_kst()
            mk = _minute_key(now)

            for rule in RULES:
                if not rule.is_due(now):
                    continue
                if state.get(rule.mode) == mk:
                    continue
                _run_mode(rule.mode, rule.label)
                state[rule.mode] = mk
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
