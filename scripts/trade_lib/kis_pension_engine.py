"""KIS 연금저축 예약주문 실행 엔진."""
import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

from .constants import PROJECT_ROOT
from .utils import get_env_any
from .notifier import send_telegram
from .kis_ops import normalize_kis_account_fields

from src.engine.kis_trader import KISTrader

logger = logging.getLogger(__name__)


def run_kis_pension_trade():
    """
    한국투자증권 연금저축 계좌 - 예약주문 실행.

    config/pension_orders.json의 '대기' 상태 주문을 모두 실행한다.
    주문 목록은 Streamlit UI에서 관리하고, 이 함수는 실행만 담당.
    """
    load_dotenv()
    logger.info("=== KIS 연금저축 예약주문 실행 시작 ===")

    # 예약주문 파일 로드
    orders_file = os.path.join(PROJECT_ROOT, "config", "pension_orders.json")
    try:
        if os.path.exists(orders_file):
            with open(orders_file, "r", encoding="utf-8") as f:
                orders = json.load(f)
            if not isinstance(orders, list):
                orders = []
        else:
            orders = []
    except Exception as e:
        logger.error(f"pension_orders.json 로드 실패: {e}")
        orders = []

    pending = [o for o in orders if o.get("status") == "대기"]
    if not pending:
        logger.info("대기 중인 예약주문 없음. 종료.")
        send_telegram("<b>KIS 연금저축</b>\n대기 중인 예약주문 없음")
        return

    logger.info(f"대기 주문 {len(pending)}건 발견")

    # KIS 인증
    pension_key = get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
    pension_secret = get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
    pension_acct_raw = get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    pension_prdt_raw = get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    pension_acct, pension_prdt = normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)

    trader = KISTrader(is_mock=False)
    if pension_key:
        trader.app_key = pension_key
    if pension_secret:
        trader.app_secret = pension_secret
    if pension_acct:
        trader.account_no = pension_acct
        trader.acnt_prdt_cd = pension_prdt

    if not trader.auth():
        logger.error("KIS 인증 실패. 종료.")
        send_telegram("<b>KIS 연금저축</b>\nKIS 인증 실패")
        return

    # 대기 주문 실행
    results = []
    changed = False
    for o in orders:
        if o.get("status") != "대기":
            continue

        code = str(o.get("etf_code", "")).strip()
        side = o.get("side", "")
        qty = int(o.get("qty", 0))
        method = o.get("method", "")
        oid = o.get("id", "")
        etf_name = o.get("etf_name", code)

        if qty <= 0 or not code:
            o["status"] = "실패"
            o["result"] = "수량 또는 ETF코드 누락"
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{etf_name}({code}): 실패 (수량/코드 누락)")
            logger.warning(f"주문 {oid}: 수량/코드 누락")
            continue

        try:
            res = None
            if "동시호가" in method:
                if "매수" in side:
                    res = trader.execute_closing_auction_buy(code, qty)
                else:
                    res = trader.execute_closing_auction_sell(code, qty)
                # 동시호가 실패 시 시간외 종가로 자동 재주문
                if not (isinstance(res, dict) and res.get("success", False)):
                    logger.warning(f"동시호가 실패 → 시간외 종가 재주문: {code}")
                    ord_side = "BUY" if "매수" in side else "SELL"
                    res2 = trader.send_order(ord_side, code, qty, price=0, ord_dvsn="06")
                    success2 = isinstance(res2, dict) and res2.get("success", False)
                    res = {
                        "success": success2,
                        "method": "동시호가실패→시간외종가",
                        "closing_result": str(res)[:100],
                        "after_hours_result": str(res2)[:100],
                    }
            elif "시간외" in method:
                ord_side = "BUY" if "매수" in side else "SELL"
                res = trader.send_order(ord_side, code, qty, price=0, ord_dvsn="06")
            elif "시장가" in method:
                if "매수" in side:
                    price_val = float(o.get("price", 0)) or 0
                    res = trader.smart_buy_krw(code, price_val)
                else:
                    res = trader.smart_sell_all(code)
            else:
                # 지정가 fallback
                price_val = int(o.get("price", 0))
                ord_side = "BUY" if "매수" in side else "SELL"
                if price_val > 0:
                    res = trader.send_order(ord_side, code, qty, price=price_val, ord_dvsn="00")
                else:
                    res = {"success": False, "msg": "지정가 0"}

            success = isinstance(res, dict) and res.get("success", False)
            o["status"] = "완료" if success else "실패"
            o["result"] = str(res)[:200] if res else "응답 없음"
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            status_str = "완료" if success else "실패"
            results.append(f"{side} {etf_name}({code}) x{qty}: {status_str}")
            logger.info(f"주문 {oid}: {side} {code} x{qty} → {status_str}")
            time.sleep(0.5)
        except Exception as e:
            o["status"] = "실패"
            o["result"] = str(e)[:200]
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{side} {etf_name}({code}) x{qty}: 실패 ({e})")
            logger.error(f"주문 {oid} 예외: {e}")

    # 결과 저장
    if changed:
        try:
            os.makedirs(os.path.dirname(orders_file), exist_ok=True)
            with open(orders_file, "w", encoding="utf-8") as f:
                json.dump(orders, f, ensure_ascii=False, indent=2)
            logger.info("pension_orders.json 업데이트 완료")
        except Exception as e:
            logger.error(f"pension_orders.json 저장 실패: {e}")

    logger.info("=== KIS 연금저축 예약주문 실행 완료 ===")
    tg = [f"<b>KIS 연금저축 예약주문</b>"]
    tg.append(f"처리: {len(results)}건")
    tg.extend([f"- {r}" for r in results[:10]])
    send_telegram("\n".join(tg))
