"""시그널 상태, 잔고 캐시, 거래 로그, 테스트 주문 관리."""
import os
import json
import logging
from datetime import datetime

from .constants import (
    KST,
    SIGNAL_STATE_FILE, BALANCE_CACHE_FILE, TRADE_LOG_FILE,
    SIGNAL_TEST_ORDERS_FILE, SIGNAL_TEST_CANCEL_AFTER_MIN,
    TRADE_LOG_MAX_ENTRIES,
)

logger = logging.getLogger(__name__)


# ── 거래 로그 ────────────────────────────────────────────

def append_trade_log(entry: dict):
    """주문 시도/결과를 trade_log.json에 기록."""
    entry.setdefault("time", datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"))
    logs = []
    try:
        if os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = []
    except Exception:
        logs = []
    logs.insert(0, entry)
    logs = logs[:TRADE_LOG_MAX_ENTRIES]
    content = json.dumps(logs, indent=2, ensure_ascii=False, default=str)
    try:
        with open(TRADE_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logger.warning(f"trade_log 저장 실패: {e}")


# ── 시그널 상태 ──────────────────────────────────────────

def load_signal_state() -> dict:
    """전략별 이전 포지션 상태 로드."""
    try:
        if os.path.exists(SIGNAL_STATE_FILE):
            with open(SIGNAL_STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            non_meta = {k: v for k, v in data.items() if not k.startswith('__')}
            logger.info(f"signal_state 로드: {SIGNAL_STATE_FILE} ({len(non_meta)}개 전략키)")
            return data
        else:
            logger.info(f"signal_state 파일 없음(최초): {SIGNAL_STATE_FILE}")
    except Exception as e:
        logger.warning(f"signal_state 로드 실패: {e} (경로={SIGNAL_STATE_FILE})")
    return {}


def save_signal_state(state: dict):
    """전략별 포지션 상태 저장."""
    try:
        content = json.dumps(state, indent=2, ensure_ascii=False)
        with open(SIGNAL_STATE_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"signal_state 저장 완료: {SIGNAL_STATE_FILE}")
    except Exception as e:
        logger.error(f"signal_state 저장 실패: {e}")


def get_prev_state(signal_state: dict, key: str):
    """signal_state에서 이전 상태값 추출 (string/dict 호환)."""
    raw = signal_state.get(key)
    if isinstance(raw, dict):
        s = str(raw.get("state", "")).upper()
        return s if s in {"BUY", "SELL", "HOLD"} else None
    if isinstance(raw, str):
        s = raw.upper()
        return s if s in {"BUY", "SELL", "HOLD"} else None
    return None


def build_signal_entry(state: str, analysis: dict) -> dict:
    """signal_state 저장용 확장 dict 생성 (목표가/이격도 포함)."""
    return {
        "state": state,
        "buy_target": analysis.get("buy_level", 0),
        "sell_target": analysis.get("sell_level", 0),
        "strategy_type": str(analysis.get("strategy_name", "")).lower(),
        "current_price": analysis.get("current_price", 0),
        "buy_dist": analysis.get("buy_gap_pct", 0),
        "sell_dist": analysis.get("sell_gap_pct", 0),
        "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── 테스트 주문 관리 ─────────────────────────────────────

def save_signal_test_orders(orders: list):
    """Signal 모드 테스트 주문 UUID 저장."""
    try:
        with open(SIGNAL_TEST_ORDERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(orders, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"signal_test_orders 저장 실패: {e}")


def load_signal_test_orders() -> list:
    """Signal 모드 테스트 주문 UUID 로드."""
    try:
        if os.path.exists(SIGNAL_TEST_ORDERS_FILE):
            with open(SIGNAL_TEST_ORDERS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def cancel_old_signal_test_orders(trader):
    """SIGNAL_TEST_CANCEL_AFTER_MIN분 이상 경과한 테스트 주문 취소."""
    orders = load_signal_test_orders()
    if not orders:
        return
    now = datetime.now(KST)
    remaining = []
    cancelled = 0
    for o in orders:
        created_str = o.get("created_at", "")
        try:
            created = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S KST")
            created = created.replace(tzinfo=KST)
        except Exception:
            created = now - __import__("datetime").timedelta(hours=1)
        age_min = (now - created).total_seconds() / 60
        if age_min >= SIGNAL_TEST_CANCEL_AFTER_MIN:
            uuid = o.get("uuid")
            if uuid:
                result = trader.cancel_order(uuid)
                logger.info(f"테스트 주문 취소 (경과 {age_min:.0f}분): {o.get('ticker')} {o.get('side')} → {result}")
                append_trade_log({
                    "mode": "signal", "ticker": o.get("ticker", ""),
                    "side": f"{o.get('side', '')}_CANCEL",
                    "result": "cancelled", "detail": f"테스트 주문 {age_min:.0f}분 경과 취소 (uuid={uuid})",
                })
                cancelled += 1
        else:
            remaining.append(o)
    if cancelled > 0:
        logger.info(f"테스트 주문 {cancelled}건 취소, {len(remaining)}건 유지")
    save_signal_test_orders(remaining)


# ── 잔고 캐시 ────────────────────────────────────────────

def save_balance_cache(balances: dict, prices: dict = None):
    """잔고+시세를 캐시 파일로 저장."""
    try:
        cache = {
            "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
            "balances": {str(k): float(v) for k, v in balances.items()},
        }
        if prices:
            cache["prices"] = {str(k): float(v) for k, v in prices.items()}
        content = json.dumps(cache, indent=2, ensure_ascii=False)
        with open(BALANCE_CACHE_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"잔고 캐시 저장 완료: {BALANCE_CACHE_FILE}")
    except Exception as e:
        logger.error(f"잔고 캐시 저장 실패: {e}")
