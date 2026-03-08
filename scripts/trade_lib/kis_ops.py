"""KIS 증권 공통 함수 (시장시간, 주문, 잔고 등)."""
import time
import logging
from datetime import datetime

from .constants import KST, GOLD_KRX_ETF_CODE, GOLD_LEGACY_ETF_CODES, ISA_WDR_TRADE_ETF_CODES
from .data import get_kis_price_local_first

from src.engine.kis_trader import KISTrader

logger = logging.getLogger(__name__)


# ── 시장 시간 / 주문 구간 ───────────────────────────────

def is_kr_market_hours() -> bool:
    """KST 09:00~16:00 국내 장 시간 확인 (동시호가 + 시간외 포함)."""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_kr_order_phase(now_kst: datetime | None = None) -> str:
    """국내 주문 구간 판별.

    regular: 09:00~15:20
    closing_auction: 15:20~15:30
    after_hours: 16:00~18:00 (시간외종가)
    closed: 장 외 시간/주말
    """
    now = now_kst or datetime.now(KST)
    if now.weekday() >= 5:
        return "closed"
    t = now.time()
    if datetime.strptime("09:00", "%H:%M").time() <= t < datetime.strptime("15:20", "%H:%M").time():
        return "regular"
    if datetime.strptime("15:20", "%H:%M").time() <= t < datetime.strptime("15:30", "%H:%M").time():
        return "closing_auction"
    if datetime.strptime("16:00", "%H:%M").time() <= t < datetime.strptime("18:00", "%H:%M").time():
        return "after_hours"
    return "closed"


def is_kr_order_window() -> bool:
    return get_kr_order_phase() != "closed"


def kr_order_phase_label(phase: str) -> str:
    return {
        "regular": "정규장",
        "closing_auction": "동시호가",
        "after_hours": "시간외종가",
        "closed": "장외시간",
    }.get(str(phase), str(phase))


# ── 주문 성공 / 실패 판별 ─────────────────────────────

def order_success(result) -> bool:
    if not result:
        return False
    if isinstance(result, dict):
        if "success" in result:
            return bool(result.get("success"))
        if "rt_cd" in result:
            return str(result.get("rt_cd")) == "0"
    return bool(result)


def extract_order_fail_msg(order_result) -> str:
    """주문 실패 응답에서 사용자 메시지를 최대한 추출."""
    if order_result is None:
        return ""
    if isinstance(order_result, dict):
        for k in ("msg", "message", "msg1", "error_message"):
            v = str(order_result.get(k, "")).strip()
            if v:
                return v
    txt = str(order_result).strip()
    return "" if txt.lower() == "none" else txt


def extract_kis_balance_fail_detail(balance_result) -> str:
    """KIS 잔고 조회 실패 응답(dict)에서 상세 실패 사유를 추출."""
    if not isinstance(balance_result, dict):
        return ""
    if not bool(balance_result.get("error")):
        return ""
    parts = []
    msg_cd = str(balance_result.get("msg_cd", "")).strip()
    msg1 = str(balance_result.get("msg1", "")).strip()
    rt_cd = str(balance_result.get("rt_cd", "")).strip()
    if msg_cd:
        parts.append(msg_cd)
    if msg1:
        parts.append(msg1)
    if rt_cd and rt_cd != "0":
        parts.append(f"rt_cd={rt_cd}")
    return " / ".join(parts)


def is_non_actionable_kis_order_failure(msg: str) -> bool:
    """헬스체크 FAIL로 집계하지 않아도 되는 업무 제약성 주문 실패."""
    text = str(msg or "").strip()
    if not text:
        return False
    patterns = [
        "만기일부터 주문이 불가",
        "주문이 불가",
        "주문가격이 하한가 미만",
        "주문가격이 상한가 초과",
    ]
    return any(p in text for p in patterns)


# ── 주문 가능 금액 ───────────────────────────────────

def get_kis_orderable_cash(balance: dict | None, fallback: float = 0.0) -> float:
    """KIS 잔고 응답에서 주문 가능 금액을 우선으로 반환."""
    if not isinstance(balance, dict):
        return float(fallback)
    try:
        buyable = float(balance.get("buyable_cash", 0.0) or 0.0)
    except Exception:
        buyable = 0.0
    if buyable > 0:
        return buyable
    try:
        return float(balance.get("cash", fallback) or fallback)
    except Exception:
        return float(fallback)


def get_kis_orderable_cash_precise(
    trader: KISTrader,
    balance: dict | None,
    base_code: str = "",
    fallback: float = 0.0,
) -> float:
    """KIS 주문가능금액 조회 API를 우선 사용하고 실패 시 잔고 응답값으로 폴백."""
    base_cash = get_kis_orderable_cash(balance, fallback=fallback)
    code = str(base_code or "").strip()
    if not code:
        return base_cash

    # 1) 지정가 기준 조회 (현재가 사용)
    price = get_kis_price_local_first(trader, code)
    if price and price > 0:
        amt = trader.get_orderable_cash(code, price=int(price), ord_dvsn="00")
        if amt is not None and float(amt) > 0:
            return float(amt)

    # 2) 시장가 기준 조회
    amt = trader.get_orderable_cash(code, price=0, ord_dvsn="01")
    if amt is not None and float(amt) > 0:
        return float(amt)

    return base_cash


# ── 잔고 조회 (재시도) ───────────────────────────────

def get_kis_balance_with_retry(
    trader: KISTrader,
    retries: int = 3,
    delay_sec: float = 0.8,
    log_prefix: str = "KIS",
) -> dict | None:
    """KIS 잔고 조회를 짧게 재시도해 일시적 API 실패를 완화."""
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            bal = trader.get_balance()
        except Exception as e:
            bal = None
            logger.warning(f"{log_prefix} 잔고 조회 예외 ({attempt}/{attempts}): {e}")
        if isinstance(bal, dict) and not bool(bal.get("error")):
            return bal
        if isinstance(bal, dict) and bool(bal.get("error")):
            logger.warning(
                f"{log_prefix} 잔고 API 오류 ({attempt}/{attempts}): "
                f"{bal.get('msg_cd', '')} {bal.get('msg1', '')}"
            )
        if attempt < attempts:
            logger.warning(
                f"{log_prefix} 잔고 조회 실패 ({attempt}/{attempts}) - {delay_sec:.1f}초 후 재시도"
            )
            time.sleep(delay_sec)
    return None


# ── 구간별 주문 실행 ─────────────────────────────────

def execute_kis_sell_qty_by_phase(trader: KISTrader, code: str, qty: int, phase: str):
    qty = int(qty or 0)
    if qty <= 0:
        return None
    if phase == "regular":
        return trader.smart_sell_qty(code, qty)
    if phase == "closing_auction":
        return trader.smart_sell_qty_closing(code, qty)
    if phase == "after_hours":
        if hasattr(trader, "smart_sell_qty_after_hours"):
            return trader.smart_sell_qty_after_hours(code, qty)
        return trader.send_order("SELL", code, qty, price=0, ord_dvsn="06")
    return None


def execute_kis_buy_qty_by_phase(trader: KISTrader, code: str, qty: int, phase: str):
    qty = int(qty or 0)
    if qty <= 0:
        return None
    if phase == "regular":
        return trader.send_order("BUY", code, qty, price=0, ord_dvsn="01")
    if phase == "closing_auction":
        return trader.execute_closing_auction_buy(code, qty)
    if phase == "after_hours":
        return trader.send_order("BUY", code, qty, price=0, ord_dvsn="06")
    return None


def execute_kis_buy_amount_by_phase(trader: KISTrader, code: str, krw_amount: float, phase: str):
    amount = float(krw_amount or 0.0)
    if amount <= 0:
        return None
    if phase == "regular":
        return trader.smart_buy_krw(code, amount)
    if phase == "closing_auction":
        return trader.smart_buy_krw_closing(code, amount)
    if phase == "after_hours":
        if hasattr(trader, "smart_buy_krw_after_hours"):
            return trader.smart_buy_krw_after_hours(code, amount)
        price = get_kis_price_local_first(trader, code)
        qty = int(amount / price) if price > 0 else 0
        return trader.send_order("BUY", code, qty, price=0, ord_dvsn="06") if qty > 0 else None
    return None


# ── 보유 종목 요약 ───────────────────────────────────

def format_holdings_brief(holdings, max_items=3):
    if not holdings:
        return "미보유"
    parts = []
    for h in holdings[:max_items]:
        code = str(h.get("code") or "").strip()
        name = str(h.get("name") or "").strip()
        qty = int(float(h.get("qty", 0) or 0))
        if code and name:
            label = f"{code}({name})"
        elif name:
            label = name
        elif code:
            label = code
        else:
            label = "UNKNOWN"
        parts.append(f"{label} {qty}주")
    if len(holdings) > max_items:
        parts.append(f"... +{len(holdings) - max_items}종")
    return ", ".join(parts)


# ── KIS 계좌 필드 정규화 ────────────────────────────

def normalize_kis_account_fields(account_raw, prdt_raw):
    """계좌번호 + 상품코드 정규화.

    '50123456-01' → ('50123456', '01')
    '5012345601' → ('50123456', '01') (입력 prdt가 '01'이면)
    """
    acct = str(account_raw or "").strip()
    prdt = str(prdt_raw or "01").strip()
    if "-" in acct:
        parts = acct.split("-", 1)
        acct = parts[0].strip()
        if len(parts) > 1 and parts[1].strip():
            prdt = parts[1].strip()
    elif len(acct) == 10 and acct.isdigit():
        prdt = acct[8:]
        acct = acct[:8]
    return acct, prdt


# ── ISA ETF 코드 보정 ───────────────────────────────

def sanitize_isa_trade_etf(code_raw: str, default: str = "418660") -> str:
    """ISA 매매 ETF 코드: 2배 전용 코드 아니면 기본값으로 보정."""
    code = str(code_raw or "").strip()
    if code in ISA_WDR_TRADE_ETF_CODES:
        return code
    return default


# ── 금 ETF 코드 정규화 ──────────────────────────────

def normalize_gold_kr_etf(code_raw: str) -> str:
    """레거시 금 ETF 코드(132030) → 411060 변환."""
    code = str(code_raw or "").strip()
    if code in GOLD_LEGACY_ETF_CODES:
        return GOLD_KRX_ETF_CODE
    return code
