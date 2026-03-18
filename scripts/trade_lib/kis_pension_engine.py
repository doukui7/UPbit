"""KIS 연금저축 예약주문 실행 엔진."""
import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

from .constants import PROJECT_ROOT, KST
from .utils import get_env_any
from .notifier import send_telegram
from .kis_ops import normalize_kis_account_fields

from src.engine.kis_trader import KISTrader

logger = logging.getLogger(__name__)


def _append_attempt(order: dict, at: str, action: str):
    """주문 기록에 실행 시도 로그를 추가한다."""
    if "attempts" not in order:
        order["attempts"] = []
    order["attempts"].append({"at": at, "action": action})
    # 최근 20건만 유지
    if len(order["attempts"]) > 20:
        order["attempts"] = order["attempts"][-20:]


def _save_orders(orders_file: str, orders: list):
    """pension_orders.json 저장."""
    try:
        os.makedirs(os.path.dirname(orders_file), exist_ok=True)
        with open(orders_file, "w", encoding="utf-8") as f:
            json.dump(orders, f, ensure_ascii=False, indent=2)
        logger.info("pension_orders.json 업데이트 완료")
    except Exception as e:
        logger.error(f"pension_orders.json 저장 실패: {e}")


def _resolve_exec_method(method: str, hour: int, minute: int) -> str | None:
    """주문 방식 + 현재 시각 → 실제 실행 방식 결정.

    Returns:
        실행 방식 문자열, 또는 None (시간 외 → 건너뜀)
    """
    if "동시호가" in method:
        if hour == 15 and 18 <= minute <= 32:
            return "동시호가"
        if hour == 15 and minute >= 38:
            return "시간외종가"  # 동시호가 시간 초과 → 자동 전환
        return None
    if "시간외" in method:
        if hour == 15 and minute >= 38:
            return "시간외종가"
        return None
    if "시장가" in method:
        if 9 <= hour < 15 or (hour == 15 and minute < 20):
            return "시장가"
        return None
    if "분산" in method:
        # 장중 시간만 허용 (09:00~15:20)
        if 9 <= hour < 15 or (hour == 15 and minute < 20):
            return "분산"
        return None
    if "지정가" in method:
        # 장중 시간만 허용 (09:00~15:20)
        if 9 <= hour < 15 or (hour == 15 and minute < 20):
            return "지정가"
        return None
    # 기타: 시간 제한 없음
    return method


def _execute_split_order(trader, order: dict, orders: list, orders_file: str, now_str: str) -> str:
    """분산 주문 실행: N분할로 간격을 두고 순차 주문.

    Returns: 결과 요약 문자열
    """
    code = str(order.get("etf_code", "")).strip()
    side = order.get("side", "")
    total_qty = int(order.get("qty", 0))
    etf_name = order.get("etf_name", code)
    splits = int(order.get("splits", 10))
    interval_sec = int(order.get("interval_sec", 60))
    sub_method = order.get("sub_method", "시장가")
    offset_pct = float(order.get("price_offset_pct", 0))
    completed = int(order.get("completed_splits", 0))
    ord_side = "BUY" if "매수" in side else "SELL"

    # 분할 수량 계산
    base_qty = total_qty // splits
    remainder = total_qty % splits
    if base_qty <= 0:
        order["status"] = "실패"
        order["result"] = f"분할 수량 부족 (총 {total_qty}주 / {splits}분할)"
        order["executed_at"] = now_str
        _append_attempt(order, now_str, f"실패: 분할 수량 부족")
        return f"{side} {etf_name}({code}) x{total_qty}: 실패 (분할 수량 부족)"

    # 시작 알림
    if completed == 0:
        _sub_label = sub_method
        if sub_method == "지정가":
            _sub_label += f" {offset_pct:+.1f}%"
        send_telegram(
            f"<b>연금저축 분산 주문 시작</b>\n"
            f"{side} {etf_name}({code}) x{total_qty}\n"
            f"{_sub_label}, {splits}분할, {interval_sec}초 간격"
        )

    success_count = completed  # 이전 성공 포함
    fail_count = 0

    for i in range(completed, splits):
        sub_qty = base_qty + (1 if i < remainder else 0)
        _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            if sub_method == "지정가":
                cur_price = trader.get_current_price(code)
                if not cur_price or cur_price <= 0:
                    _append_attempt(order, _now, f"분산 {i+1}/{splits}: 현재가 조회 실패")
                    fail_count += 1
                    order["completed_splits"] = i + 1
                    _save_orders(orders_file, orders)
                    if i < splits - 1:
                        time.sleep(interval_sec)
                    continue
                limit_price = int(round(cur_price * (1 + offset_pct / 100)))
                if limit_price <= 0:
                    limit_price = 1
                res = trader.send_order(ord_side, code, sub_qty, price=limit_price, ord_dvsn="00")
                _price_info = f"@{limit_price:,}"
            else:
                res = trader.send_order(ord_side, code, sub_qty, price=0, ord_dvsn="01")
                _price_info = "시장가"

            ok = isinstance(res, dict) and res.get("success", False)
            _ord_no = res.get("ord_no", "") if isinstance(res, dict) else ""
            if ok:
                success_count += 1
                _append_attempt(order, _now, f"분산 {i+1}/{splits}: 완료 {sub_qty}주 {_price_info} (ord_no:{_ord_no})")
            else:
                fail_count += 1
                _fail_msg = str(res)[:60] if res else "응답 없음"
                _append_attempt(order, _now, f"분산 {i+1}/{splits}: 실패 {sub_qty}주 {_price_info} ({_fail_msg})")

        except Exception as e:
            fail_count += 1
            _append_attempt(order, _now, f"분산 {i+1}/{splits}: 예외 ({str(e)[:60]})")
            logger.error(f"분산 주문 {i+1}/{splits} 예외: {e}")

        order["completed_splits"] = i + 1
        _save_orders(orders_file, orders)

        # 마지막 회차가 아니면 대기
        if i < splits - 1:
            time.sleep(interval_sec)

    # ── 최종 주문: 미체결 취소 + 잔량 일괄 주문 ──
    final_method = order.get("final_method", "")
    final_offset = float(order.get("final_offset_pct", 0))
    final_filled = 0

    if final_method and sub_method == "지정가":
        # 지정가 분산은 미체결 가능 → 미체결 조회 후 취소 + 재주문
        _now_f = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time.sleep(2)  # 체결 반영 대기

        try:
            pending = trader.get_pending_orders(code)
            unfilled_qty = sum(p.get("remaining_qty", 0) for p in pending
                               if p.get("side") == ord_side)

            if unfilled_qty > 0:
                # 미체결 전량 취소
                cancelled = 0
                for p in pending:
                    if p.get("side") != ord_side:
                        continue
                    _ord_no = p.get("ord_no", "")
                    if _ord_no:
                        cres = trader.cancel_order(_ord_no, code)
                        c_ok = isinstance(cres, dict) and cres.get("success", False)
                        if c_ok:
                            cancelled += p.get("remaining_qty", 0)
                        _append_attempt(order, _now_f,
                                        f"최종: 취소 ord_no={_ord_no} {'OK' if c_ok else 'FAIL'}")
                        time.sleep(0.3)

                if cancelled > 0:
                    time.sleep(1)
                    # 최종 일괄 주문
                    if final_method == "시장가":
                        fres = trader.send_order(ord_side, code, cancelled, price=0, ord_dvsn="01")
                        _fp_info = "시장가"
                    else:
                        cur_price = trader.get_current_price(code)
                        fp = int(round((cur_price or 0) * (1 + final_offset / 100))) if cur_price else 0
                        if fp <= 0:
                            fp = 1
                        fres = trader.send_order(ord_side, code, cancelled, price=fp, ord_dvsn="00")
                        _fp_info = f"@{fp:,} ({final_offset:+.1f}%)"

                    f_ok = isinstance(fres, dict) and fres.get("success", False)
                    _f_ord = fres.get("ord_no", "") if isinstance(fres, dict) else ""
                    _append_attempt(order, _now_f,
                                    f"최종: {cancelled}주 {_fp_info} → {'완료' if f_ok else '실패'} (ord_no:{_f_ord})")
                    if f_ok:
                        final_filled = cancelled
                    send_telegram(
                        f"<b>연금저축 최종 주문</b>\n"
                        f"{side} {etf_name}({code}) x{cancelled} {_fp_info}\n"
                        f"{'완료' if f_ok else '실패'}"
                    )
            else:
                _append_attempt(order, _now_f, "최종: 미체결 없음 (전량 체결)")
        except Exception as e:
            _append_attempt(order, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            f"최종: 예외 ({str(e)[:60]})")
            logger.error(f"최종 주문 예외: {e}")

    # 최종 결과
    _now_final = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if fail_count == 0 or final_filled > 0:
        order["status"] = "완료"
    else:
        order["status"] = "실패" if success_count == 0 else "완료"
    _result_parts = [f"분산 {success_count}/{splits}"]
    if final_method:
        _result_parts.append(f"최종 {final_filled}주")
    order["result"] = " + ".join(_result_parts)
    order["executed_at"] = _now_final
    _save_orders(orders_file, orders)

    _summary = f"{side} {etf_name}({code}) x{total_qty}: {order['result']}"
    send_telegram(f"<b>연금저축 분산 주문 {'완료' if order['status'] == '완료' else '종료'}</b>\n{_summary}")
    return _summary


def run_kis_pension_trade():
    """
    한국투자증권 연금저축 계좌 - 예약주문 실행.

    config/pension_orders.json의 '대기' 상태 주문을 모두 실행한다.
    주문 목록은 Streamlit UI에서 관리하고, 이 함수는 실행만 담당.

    시간대별 실행 규칙:
      동시호가  → 15:18~15:32 (동시호가), 15:38~ (시간외종가 자동전환)
      시간외종가 → 15:38~16:00
      시장가    → 09:00~15:20
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

    now_kst = datetime.now(KST)
    today_str = now_kst.strftime("%Y-%m-%d")
    _hour = now_kst.hour
    _minute = now_kst.minute

    _now_str = now_kst.strftime("%Y-%m-%d %H:%M:%S")

    pending = [
        o for o in orders
        if o.get("status") == "대기"
        and o.get("scheduled_kst", "9999")[:10] <= today_str
    ]
    if not pending:
        logger.info("대기 중인 예약주문 없음. 종료.")
        return

    logger.info(f"대기 주문 {len(pending)}건 발견 (오늘 이전 예정분)")

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
        # 인증 실패도 시도 기록 남기기
        for o in pending:
            _append_attempt(o, _now_str, "인증 실패로 건너뜀")
        _save_orders(orders_file, orders)
        return

    # 대기 주문 실행
    results = []
    changed = False
    for o in orders:
        if o.get("status") != "대기":
            continue

        # 날짜 필터: 미래 예정 주문은 건너뜀
        sched_date = o.get("scheduled_kst", "")[:10]
        if sched_date > today_str:
            continue

        code = str(o.get("etf_code", "")).strip()
        side = o.get("side", "")
        qty = int(o.get("qty", 0))
        method = o.get("method", "")
        oid = o.get("id", "")
        etf_name = o.get("etf_name", code)

        # ── 시간대 안전장치 & 방식 자동 전환 ──
        exec_method = _resolve_exec_method(method, _hour, _minute)
        if exec_method is None:
            _skip_reason = f"시간외 건너뜀 ({_hour}:{_minute:02d}, {method})"
            logger.warning(f"주문 {oid}: {_skip_reason}")
            _append_attempt(o, _now_str, _skip_reason)
            changed = True
            continue

        if qty <= 0 or not code:
            o["status"] = "실패"
            o["result"] = "수량 또는 ETF코드 누락"
            o["executed_at"] = _now_str
            _append_attempt(o, _now_str, "실패: 수량/코드 누락")
            changed = True
            results.append(f"{etf_name}({code}): 실패 (수량/코드 누락)")
            logger.warning(f"주문 {oid}: 수량/코드 누락")
            continue

        # ── 분산 주문: 별도 함수로 처리 ──
        if exec_method == "분산":
            _split_result = _execute_split_order(trader, o, orders, orders_file, _now_str)
            changed = True
            results.append(_split_result)
            continue

        # 실행 방식 안내
        if exec_method != method.split("(")[0].strip():
            _label = f"{exec_method} (원래: {method.split('(')[0].strip()})"
        else:
            _label = exec_method
        send_telegram(
            f"<b>연금저축 주문 실행</b>\n"
            f"{side} {etf_name}({code}) x{qty} ({_label})"
        )

        try:
            res = None
            if exec_method == "동시호가":
                if "매수" in side:
                    res = trader.execute_closing_auction_buy(code, qty)
                else:
                    res = trader.execute_closing_auction_sell(code, qty)
                # 동시호가 실패 시 처리
                if not (isinstance(res, dict) and res.get("success", False)):
                    if _minute < 35:
                        # 시간외종가 시간 전 → "대기" 유지, 15:40에 시간외종가로 재시도
                        _fail_msg = str(res)[:80]
                        o["result"] = f"동시호가 실패→시간외 대기: {_fail_msg}"
                        o["executed_at"] = _now_str
                        _append_attempt(o, _now_str, f"동시호가 실패→시간외 대기: {_fail_msg}")
                        changed = True
                        results.append(
                            f"{side} {etf_name}({code}) x{qty}: 동시호가 실패→시간외 대기"
                        )
                        send_telegram(
                            f"<b>연금저축 동시호가 실패</b>\n"
                            f"{side} {etf_name}({code}) x{qty}\n"
                            f"15:40 시간외종가 자동 재시도\n{_fail_msg}"
                        )
                        continue  # status stays "대기"
                    else:
                        # 시간외종가 시간 → 즉시 fallback
                        logger.warning(f"동시호가 실패 → 시간외 종가 재주문: {code}")
                        ord_side = "BUY" if "매수" in side else "SELL"
                        res2 = trader.send_order(
                            ord_side, code, qty, price=0, ord_dvsn="06"
                        )
                        success2 = isinstance(res2, dict) and res2.get(
                            "success", False
                        )
                        res = {
                            "success": success2,
                            "method": "동시호가실패→시간외종가",
                            "closing_result": str(res)[:100],
                            "after_hours_result": str(res2)[:100],
                        }
            elif exec_method == "시간외종가":
                ord_side = "BUY" if "매수" in side else "SELL"
                res = trader.send_order(ord_side, code, qty, price=0, ord_dvsn="06")
            elif exec_method == "시장가":
                ord_side = "BUY" if "매수" in side else "SELL"
                res = trader.send_order(ord_side, code, qty, price=0, ord_dvsn="01")
            else:
                # 지정가 (고정가 또는 오프셋)
                price_val = int(o.get("price", 0))
                offset_pct = float(o.get("price_offset_pct", 0))
                ord_side = "BUY" if "매수" in side else "SELL"
                if price_val > 0:
                    res = trader.send_order(
                        ord_side, code, qty, price=price_val, ord_dvsn="00"
                    )
                elif offset_pct != 0:
                    cur_price = trader.get_current_price(code)
                    if cur_price and cur_price > 0:
                        limit_price = int(round(cur_price * (1 + offset_pct / 100)))
                        if limit_price <= 0:
                            limit_price = 1
                        res = trader.send_order(
                            ord_side, code, qty, price=limit_price, ord_dvsn="00"
                        )
                    else:
                        res = {"success": False, "msg": "현재가 조회 실패"}
                else:
                    res = {"success": False, "msg": "지정가 0"}

            success = isinstance(res, dict) and res.get("success", False)
            o["status"] = "완료" if success else "실패"
            o["result"] = str(res)[:200] if res else "응답 없음"
            o["executed_at"] = _now_str
            _exec_label = exec_method if exec_method == method.split("(")[0].strip() else f"{exec_method}(원래:{method.split('(')[0].strip()})"
            _append_attempt(o, _now_str, f"{'완료' if success else '실패'}: {_exec_label}")
            changed = True
            status_str = "완료" if success else "실패"
            results.append(f"{side} {etf_name}({code}) x{qty}: {status_str}")
            logger.info(f"주문 {oid}: {side} {code} x{qty} → {status_str}")
            # 개별 결과 알림
            if success:
                _ord_no = res.get("ord_no", "") if isinstance(res, dict) else ""
                _detail = f" (ord_no: {_ord_no})" if _ord_no else ""
                send_telegram(
                    f"<b>연금저축 주문 완료</b>\n"
                    f"{side} {etf_name}({code}) x{qty}{_detail}"
                )
            else:
                _fail_msg = str(res)[:80] if res else "응답 없음"
                send_telegram(
                    f"<b>연금저축 주문 실패</b>\n"
                    f"{side} {etf_name}({code}) x{qty}\n{_fail_msg}"
                )
            time.sleep(0.5)
        except Exception as e:
            o["status"] = "실패"
            o["result"] = str(e)[:200]
            o["executed_at"] = _now_str
            _append_attempt(o, _now_str, f"예외: {str(e)[:80]}")
            changed = True
            results.append(f"{side} {etf_name}({code}) x{qty}: 실패 ({e})")
            logger.error(f"주문 {oid} 예외: {e}")
            send_telegram(
                f"<b>연금저축 주문 실패</b>\n"
                f"{side} {etf_name}({code}) x{qty}\n{str(e)[:80]}"
            )

    # 결과 저장
    if changed:
        _save_orders(orders_file, orders)

    if not results:
        logger.info("실행 대상 주문 없음 (시간 외 건너뜀). 종료.")
        return

    logger.info("=== KIS 연금저축 예약주문 실행 완료 ===")
    tg = [f"<b>KIS 연금저축 예약주문</b>"]
    tg.append(f"처리: {len(results)}건")
    tg.extend([f"- {r}" for r in results[:10]])
    send_telegram("\n".join(tg))
