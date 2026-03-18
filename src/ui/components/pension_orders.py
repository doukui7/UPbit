"""연금저축 예약 주문 — 데이터 관리 + 탭 UI."""
import json
import os
import random
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from src.utils.formatting import _etf_name_kr, _fmt_etf_code_name


def _try_send_telegram(msg: str):
    """텔레그램 전송 시도 (실패 시 무시)."""
    try:
        from scripts.trade_lib.notifier import send_telegram
        send_telegram(msg)
    except Exception:
        pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PEN_ORDERS_FILE = os.path.join(_PROJECT_ROOT, "config", "pension_orders.json")


# ═══════════════════════════════════════════
# 데이터 관리
# ═══════════════════════════════════════════
def load_pen_orders() -> list[dict]:
    """예약 주문 내역 로드 (영구 보존)."""
    try:
        if os.path.exists(_PEN_ORDERS_FILE):
            with open(_PEN_ORDERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass
    return []


def _save_pen_orders(orders: list[dict]):
    """예약 주문 내역 저장."""
    os.makedirs(os.path.dirname(_PEN_ORDERS_FILE), exist_ok=True)
    with open(_PEN_ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)


def add_pen_order(etf_code: str, side: str, qty: int, method: str,
                  price: int, scheduled_kst: str, note: str,
                  sub_method: str = "", price_offset_pct: float = 0.0,
                  splits: int = 0, interval_sec: int = 60,
                  final_method: str = "", final_offset_pct: float = 0.0) -> dict:
    """예약 주문 추가."""
    orders = load_pen_orders()
    now_kst = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    order = {
        "id": f"pen-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}",
        "etf_code": str(etf_code),
        "etf_name": _etf_name_kr(str(etf_code)),
        "side": side,
        "qty": qty,
        "method": method,
        "price": price,
        "scheduled_kst": scheduled_kst,
        "status": "대기",
        "created_at": now_kst,
        "executed_at": "",
        "result": "",
        "note": note,
    }
    # 분산 주문 추가 필드
    if "분산" in method:
        order["sub_method"] = sub_method or "시장가"
        order["price_offset_pct"] = price_offset_pct
        order["splits"] = max(splits, 2)
        order["interval_sec"] = max(interval_sec, 10)
        order["completed_splits"] = 0
        # 최종 주문 (미체결 잔량 일괄 처리)
        if final_method and final_method != "없음":
            order["final_method"] = final_method
            order["final_offset_pct"] = final_offset_pct
    elif "지정가" in method and price_offset_pct != 0:
        order["price_offset_pct"] = price_offset_pct
    orders.append(order)
    _save_pen_orders(orders)
    return order


def update_pen_order_status(order_id: str, status: str, result: str = ""):
    """예약 주문 상태 업데이트 (삭제하지 않고 상태만 변경)."""
    orders = load_pen_orders()
    for o in orders:
        if o.get("id") == order_id:
            o["status"] = status
            if result:
                o["result"] = result
            if status in ("완료", "실패", "취소"):
                o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            break
    _save_pen_orders(orders)


def execute_pending_pen_orders(trader) -> list[str]:
    """예정 시간이 지난 대기 주문을 실행하고 상태를 업데이트한다."""
    orders = load_pen_orders()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    results = []
    changed = False

    for o in orders:
        if o.get("status") != "대기":
            continue
        sched = o.get("scheduled_kst", "")
        if not sched or sched > now_str:
            continue

        code = str(o.get("etf_code", "")).strip()
        side = o.get("side", "")
        qty = int(o.get("qty", 0))
        method = o.get("method", "")
        oid = o.get("id", "")

        if qty <= 0 or not code:
            o["status"] = "실패"
            o["result"] = "수량 또는 ETF코드 누락"
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{oid}: 실패 (수량/코드 누락)")
            continue

        etf_name = o.get("etf_name", code)
        _method_short = method.split("(")[0].strip() if method else method
        _try_send_telegram(f"<b>연금저축 주문 실행</b>\n{side} {etf_name}({code}) x{qty} ({_method_short})")

        try:
            res = None
            if "동시호가" in method:
                if "매수" in side:
                    res = trader.execute_closing_auction_buy(code, qty)
                else:
                    res = trader.execute_closing_auction_sell(code, qty)
                # 동시호가 실패 시 시간외 종가로 자동 재주문
                if not (isinstance(res, dict) and res.get("success", False)):
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
                ord_side = "BUY" if "매수" in side else "SELL"
                res = trader.send_order(ord_side, code, qty, price=0, ord_dvsn="01")
            else:
                # 지정가 fallback
                price = int(o.get("price", 0))
                ord_side = "BUY" if "매수" in side else "SELL"
                if price > 0:
                    res = trader.send_order(ord_side, code, qty, price=price, ord_dvsn="00")
                else:
                    res = {"success": False, "msg": "지정가 0"}

            success = isinstance(res, dict) and res.get("success", False)
            o["status"] = "완료" if success else "실패"
            o["result"] = str(res)[:200] if res else "응답 없음"
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{oid}: {'완료' if success else '실패'}")
            # 개별 결과 알림
            if success:
                _ord_no = res.get("ord_no", "") if isinstance(res, dict) else ""
                _detail = f" (ord_no: {_ord_no})" if _ord_no else ""
                _try_send_telegram(f"<b>연금저축 주문 완료</b>\n{side} {etf_name}({code}) x{qty}{_detail}")
            else:
                _fail_msg = str(res)[:80] if res else "응답 없음"
                _try_send_telegram(f"<b>연금저축 주문 실패</b>\n{side} {etf_name}({code}) x{qty}\n{_fail_msg}")
        except Exception as e:
            o["status"] = "실패"
            o["result"] = str(e)[:200]
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{oid}: 실패 ({e})")
            _try_send_telegram(f"<b>연금저축 주문 실패</b>\n{side} {etf_name}({code}) x{qty}\n{str(e)[:80]}")

    if changed:
        _save_pen_orders(orders)
    return results


# ═══════════════════════════════════════════
# 탭 UI 렌더링
# ═══════════════════════════════════════════
def render_pension_orders_tab(etf_options: dict, current_price: float = 0):
    """연금저축 예약 주문 탭 메인 렌더러.

    Args:
        etf_options: {표시명: ETF코드} 딕셔너리
        current_price: 선택된 ETF 현재가
    """
    st.header("예약 주문")
    st.caption("주문을 예약하고 이력을 영구 보관합니다. 실행된 주문도 삭제되지 않습니다.")

    # ── 리밸런싱 주문 미리보기 ──
    _combo_rebal = st.session_state.get("pen_combined_rebal_data")
    if _combo_rebal:
        _rebal_buy = [r for r in _combo_rebal if r.get("주문 상태") == "매수" and int(r.get("매수 예정(주)", 0)) > 0]
        _rebal_sell = [r for r in _combo_rebal if r.get("주문 상태") == "매도" and int(r.get("매도 예정(주)", 0)) > 0]
        _has_rebal = bool(_rebal_buy or _rebal_sell)

        # 리밸런싱 예정일 계산
        from datetime import date as _rb_date
        import calendar as _rb_cal
        _rb_today = _rb_date.today()
        _rb_y, _rb_m = _rb_today.year, _rb_today.month
        _rb_last = _rb_cal.monthrange(_rb_y, _rb_m)[1]
        _rb_dt = _rb_date(_rb_y, _rb_m, _rb_last)
        while _rb_dt.weekday() >= 5:
            _rb_dt -= timedelta(days=1)
        _is_rebal_day = (_rb_today == _rb_dt)
        _rb_days_left = (_rb_dt - _rb_today).days

        if _has_rebal:
            _rb_title = "리밸런싱 주문 미리보기"
            if _is_rebal_day:
                _rb_title += " (오늘 리밸런싱일)"
            with st.expander(_rb_title, expanded=_is_rebal_day):
                if _is_rebal_day:
                    st.success("오늘은 리밸런싱 예정일입니다. 아래 주문을 확인 후 일괄 예약 등록하세요.")
                else:
                    st.info(f"다음 리밸런싱: {_rb_dt.strftime('%Y-%m-%d')} (D-{_rb_days_left})")

                # 매도 주문 먼저
                if _rebal_sell:
                    st.markdown("**매도 예정**")
                    _sell_rows = []
                    for _rs in _rebal_sell:
                        _sell_rows.append({
                            "ETF": _rs.get("ETF", ""),
                            "현재수량(주)": _rs.get("현재수량(주)", 0),
                            "합산 목표(주)": _rs.get("합산 목표(주)", 0),
                            "매도수량(주)": _rs.get("매도 예정(주)", 0),
                            "현재 비중(%)": _rs.get("현재 비중(%)", 0),
                            "목표 비중(%)": _rs.get("목표 비중(%)", 0),
                        })
                    st.dataframe(pd.DataFrame(_sell_rows), use_container_width=True, hide_index=True)

                # 매수 주문
                if _rebal_buy:
                    st.markdown("**매수 예정**")
                    _buy_rows = []
                    for _rb in _rebal_buy:
                        _buy_rows.append({
                            "ETF": _rb.get("ETF", ""),
                            "현재수량(주)": _rb.get("현재수량(주)", 0),
                            "합산 목표(주)": _rb.get("합산 목표(주)", 0),
                            "매수수량(주)": _rb.get("매수 예정(주)", 0),
                            "현재 비중(%)": _rb.get("현재 비중(%)", 0),
                            "목표 비중(%)": _rb.get("목표 비중(%)", 0),
                        })
                    st.dataframe(pd.DataFrame(_buy_rows), use_container_width=True, hide_index=True)

                # 일괄 예약 등록
                st.markdown("---")
                _rbc1, _rbc2, _rbc3 = st.columns(3)
                with _rbc1:
                    _rb_method = st.selectbox(
                        "주문 방식",
                        ["동시호가 (장마감)", "시간외 종가", "시장가", "지정가 (오프셋)", "분산 주문"],
                        key="pen_rebal_bulk_method",
                    )
                with _rbc2:
                    _rb_exec_date = st.date_input("실행 예정일", value=_rb_dt, key="pen_rebal_bulk_date")
                with _rbc3:
                    if _rb_method == "동시호가 (장마감)":
                        _rb_def_time = pd.Timestamp("15:20").time()
                    elif _rb_method == "시간외 종가":
                        _rb_def_time = pd.Timestamp("15:40").time()
                    elif "분산" in _rb_method or "지정가" in _rb_method:
                        _rb_def_time = pd.Timestamp("09:30").time()
                    else:
                        _rb_def_time = pd.Timestamp("15:10").time()
                    _rb_time = st.time_input("실행 시각", value=_rb_def_time, key="pen_rebal_bulk_time")

                # ── 지정가 (오프셋) 옵션 ──
                _rb_is_limit = "지정가" in _rb_method and "분산" not in _rb_method
                _rb_offset_pct = 0.0
                if _rb_is_limit:
                    _rb_offset_pct = st.number_input(
                        "가격 오프셋 (%)", value=1.0, step=0.5, format="%.1f",
                        key="pen_rebal_limit_offset",
                        help="현재가 대비 +면 비싸게(매수 유리), -면 싸게(매도 유리)",
                    )

                # ── 분산 주문 옵션 ──
                _rb_is_split = "분산" in _rb_method
                _rb_sub_method = ""
                _rb_split_offset = 0.0
                _rb_splits = 10
                _rb_interval = 60
                _rb_final_method = ""
                _rb_final_offset = 0.0
                if _rb_is_split:
                    sp1, sp2, sp3, sp4 = st.columns(4)
                    with sp1:
                        _rb_sub_method = st.selectbox("하위 방식", ["시장가", "지정가"], key="pen_rebal_sub_method")
                    with sp2:
                        _rb_split_offset = st.number_input(
                            "가격 오프셋 (%)", value=1.0, step=0.5, format="%.1f",
                            key="pen_rebal_split_offset",
                            disabled=(_rb_sub_method != "지정가"),
                        )
                    with sp3:
                        _rb_splits = st.number_input("분할 수", min_value=2, max_value=20, value=10, step=1, key="pen_rebal_splits")
                    with sp4:
                        _rb_total_min = st.number_input("총 시간 (분)", min_value=2, max_value=120, value=20, step=1, key="pen_rebal_total_min")
                        _rb_interval = max(int(_rb_total_min * 60 / _rb_splits), 10)
                        st.caption(f"{_rb_interval}초 간격")

                    st.markdown("**최종 주문** (분산 완료 후 미체결 잔량 일괄 처리)")
                    fp1, fp2 = st.columns(2)
                    with fp1:
                        _rb_final_method = st.selectbox("최종 주문 방식", ["없음", "시장가", "지정가"], key="pen_rebal_final_method")
                    with fp2:
                        _rb_final_offset = st.number_input(
                            "최종 오프셋 (%)", value=2.0, step=0.5, format="%.1f",
                            key="pen_rebal_final_offset",
                            disabled=(_rb_final_method != "지정가"),
                        )

                if st.button("일괄 예약 등록", key="pen_rebal_bulk_register", type="primary"):
                    _rb_sched = f"{_rb_exec_date.strftime('%Y-%m-%d')} {_rb_time.strftime('%H:%M')}"
                    _rb_count = 0
                    # 공통 kwargs
                    _rb_kwargs = {}
                    if _rb_is_split:
                        _rb_kwargs = dict(
                            sub_method=_rb_sub_method,
                            price_offset_pct=_rb_split_offset,
                            splits=_rb_splits,
                            interval_sec=_rb_interval,
                            final_method=_rb_final_method,
                            final_offset_pct=_rb_final_offset,
                        )
                    elif _rb_is_limit:
                        _rb_kwargs = dict(price_offset_pct=_rb_offset_pct)
                    # 매도 먼저 등록
                    for _rs in _rebal_sell:
                        _etf_code = str(_rs.get("ETF코드", "")).strip()
                        _qty = int(_rs.get("매도 예정(주)", 0))
                        if _etf_code and _qty > 0:
                            add_pen_order(
                                etf_code=_etf_code, side="매도", qty=_qty,
                                method=_rb_method, price=0,
                                scheduled_kst=_rb_sched, note="리밸런싱",
                                **_rb_kwargs,
                            )
                            _rb_count += 1
                    # 매수
                    for _rb in _rebal_buy:
                        _etf_code = str(_rb.get("ETF코드", "")).strip()
                        _qty = int(_rb.get("매수 예정(주)", 0))
                        if _etf_code and _qty > 0:
                            add_pen_order(
                                etf_code=_etf_code, side="매수", qty=_qty,
                                method=_rb_method, price=0,
                                scheduled_kst=_rb_sched, note="리밸런싱",
                                **_rb_kwargs,
                            )
                            _rb_count += 1
                    if _rb_count > 0:
                        st.success(f"리밸런싱 주문 {_rb_count}건 예약 등록 완료")
                        st.rerun()
                    else:
                        st.warning("등록할 주문이 없습니다. (ETF 코드 매핑 확인 필요)")

    with st.expander("새 예약 주문 등록", expanded=False):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            rsv_etf_label = st.selectbox("ETF 종목", list(etf_options.keys()), key="pen_rsv_etf")
            rsv_etf_code = etf_options[rsv_etf_label]
        with rc2:
            rsv_side = st.selectbox("매매 방향", ["매수", "매도"], key="pen_rsv_side")
        with rc3:
            rsv_method = st.selectbox("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가", "분산 주문"], key="pen_rsv_method")

        # ── 분산 주문 추가 옵션 ──
        _is_split = "분산" in rsv_method
        rsv_sub_method = ""
        rsv_offset_pct = 0.0
        rsv_splits = 10
        rsv_interval = 60
        rsv_final_method = ""
        rsv_final_offset = 0.0
        if _is_split:
            sp1, sp2, sp3, sp4 = st.columns(4)
            with sp1:
                rsv_sub_method = st.selectbox("하위 방식", ["시장가", "지정가"], key="pen_rsv_sub_method")
            with sp2:
                rsv_offset_pct = st.number_input(
                    "가격 오프셋 (%)", value=1.0, step=0.5, format="%.1f",
                    key="pen_rsv_offset", help="지정가: 현재가 대비 +면 비싸게, -면 싸게",
                    disabled=(rsv_sub_method != "지정가"),
                )
            with sp3:
                rsv_splits = st.number_input("분할 수", min_value=2, max_value=20, value=10, step=1, key="pen_rsv_splits")
            with sp4:
                rsv_total_min = st.number_input("총 시간 (분)", min_value=2, max_value=120, value=20, step=1, key="pen_rsv_total_min")
                rsv_interval = max(int(rsv_total_min * 60 / rsv_splits), 10)
                st.caption(f"{rsv_interval}초 간격")

            # ── 최종 주문 옵션 ──
            st.markdown("**최종 주문** (분산 완료 후 미체결 잔량 일괄 처리)")
            fp1, fp2 = st.columns(2)
            with fp1:
                rsv_final_method = st.selectbox("최종 주문 방식", ["없음", "시장가", "지정가"], key="pen_rsv_final_method")
            with fp2:
                rsv_final_offset = st.number_input(
                    "최종 오프셋 (%)", value=2.0, step=0.5, format="%.1f",
                    key="pen_rsv_final_offset",
                    disabled=(rsv_final_method != "지정가"),
                )

        rc4, rc5, rc6 = st.columns(3)
        with rc4:
            rsv_qty = st.number_input("주문 수량 (주)", min_value=1, value=1, step=1, key="pen_rsv_qty")
        with rc5:
            if _is_split:
                st.markdown("**가격**")
                if rsv_sub_method == "지정가":
                    st.info(f"현재가 {rsv_offset_pct:+.1f}%")
                else:
                    st.info("시장가 (자동)")
                rsv_price = 0
            else:
                rsv_price = st.number_input(
                    "지정가 (원, 시장가=0)",
                    min_value=0,
                    value=int(current_price) if rsv_method == "지정가" and current_price > 0 else 0,
                    step=50,
                    key="pen_rsv_price",
                )
        with rc6:
            rsv_date = st.date_input("실행 예정일", key="pen_rsv_date")
            # 주문 방식에 따라 시간 자동 결정
            if rsv_method == "동시호가 (장마감)":
                rsv_time = pd.Timestamp("15:20").time()
                st.markdown("**실행 예정 시각**")
                st.info("15:20 (동시호가 고정)")
            elif rsv_method == "시간외 종가":
                rsv_time = pd.Timestamp("15:40").time()
                st.markdown("**실행 예정 시각**")
                st.info("15:40 (시간외 종가 고정)")
            else:
                _default_time = pd.Timestamp("09:30").time() if _is_split else pd.Timestamp("15:10").time()
                rsv_time = st.time_input("실행 예정 시각", value=_default_time, key="pen_rsv_time")

        rsv_note = st.text_input("메모 (선택)", key="pen_rsv_note", placeholder="예: 채권 리밸런싱")

        # ── 요약 미리보기 ──
        _rsv_unit = rsv_price if rsv_price > 0 else (current_price if current_price > 0 else 0)
        _rsv_total = int(rsv_qty * _rsv_unit) if _rsv_unit > 0 else 0
        _method_label = rsv_method
        if _is_split:
            _sub_label = rsv_sub_method
            if rsv_sub_method == "지정가":
                _sub_label += f" {rsv_offset_pct:+.1f}%"
            _method_label = f"분산({_sub_label}, {rsv_splits}분할, {rsv_interval}초)"
            if rsv_final_method and rsv_final_method != "없음":
                _fl = rsv_final_method
                if rsv_final_method == "지정가":
                    _fl += f" {rsv_final_offset:+.1f}%"
                _method_label += f" + 최종:{_fl}"
        st.markdown(
            f"<div style='background:#fffde7;border:1px solid #fff9c4;border-radius:8px;padding:10px 14px;margin:8px 0'>"
            f"<b>종목:</b> {_fmt_etf_code_name(rsv_etf_code)} &nbsp;|&nbsp; "
            f"<b>방향:</b> {rsv_side} &nbsp;|&nbsp; "
            f"<b>수량:</b> {rsv_qty}주 &nbsp;|&nbsp; "
            f"<b>예상 금액:</b> <span style='font-weight:bold'>{_rsv_total:,}원</span> &nbsp;|&nbsp; "
            f"<b>방식:</b> {_method_label}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("예약 등록", key="pen_rsv_add", type="primary"):
            sched_str = f"{rsv_date.strftime('%Y-%m-%d')} {rsv_time.strftime('%H:%M')}"
            new_order = add_pen_order(
                etf_code=rsv_etf_code,
                side=rsv_side,
                qty=rsv_qty,
                method=rsv_method,
                price=rsv_price,
                scheduled_kst=sched_str,
                note=rsv_note,
                sub_method=rsv_sub_method if _is_split else "",
                price_offset_pct=rsv_offset_pct if _is_split else 0.0,
                splits=rsv_splits if _is_split else 0,
                interval_sec=rsv_interval if _is_split else 60,
                final_method=rsv_final_method if _is_split else "",
                final_offset_pct=rsv_final_offset if _is_split else 0.0,
            )
            st.success(f"예약 등록 완료: {new_order['id']} — {_fmt_etf_code_name(rsv_etf_code)} {rsv_side} {rsv_qty}주 @ {sched_str}")
            st.rerun()

    # ── 예약 주문 내역 표시 ──
    pen_orders = load_pen_orders()
    if pen_orders:
        # 대기 → 완료 → 실패 → 취소 순 정렬, 최신 먼저
        _status_priority = {"대기": 0, "완료": 1, "실패": 2, "취소": 3}
        pen_orders_sorted = sorted(
            pen_orders,
            key=lambda o: (_status_priority.get(o.get("status", ""), 9), -(o.get("created_at", "") or "").__hash__()),
        )

        pending_orders = [o for o in pen_orders_sorted if o.get("status") == "대기"]
        done_orders = [o for o in pen_orders_sorted if o.get("status") != "대기"]

        if pending_orders:
            st.markdown(f"**대기 중 ({len(pending_orders)}건)**")
            for o in pending_orders:
                oid = o.get("id", "")
                _m_label = o.get("method", "")
                if "지정가" in _m_label and o.get("price_offset_pct") and "분산" not in _m_label:
                    _off = o.get("price_offset_pct", 0)
                    _m_label = f"지정가 (오프셋 {_off:+.1f}%)"
                elif "분산" in _m_label:
                    _sm = o.get("sub_method", "시장가")
                    _sp = o.get("splits", 0)
                    _iv = o.get("interval_sec", 60)
                    _cs = o.get("completed_splits", 0)
                    _off = o.get("price_offset_pct", 0)
                    _sm_str = f"{_sm} {_off:+.1f}%" if _sm == "지정가" else _sm
                    _m_label = f"분산({_sm_str}, {_sp}분할, {_iv}초)"
                    if _cs > 0:
                        _m_label += f" [{_cs}/{_sp} 완료]"
                label = f"{o.get('side', '')} {_fmt_etf_code_name(o.get('etf_code', ''))} {o.get('qty', 0)}주 | {_m_label} | 예정: {o.get('scheduled_kst', '')}"
                if o.get("note"):
                    label += f" | {o['note']}"
                # 최근 시도 기록이 있으면 표시
                _attempts = o.get("attempts", [])
                if _attempts:
                    _last = _attempts[-1]
                    label += f"\n  └ 최근 시도: {_last.get('at', '')} — {_last.get('action', '')}"
                _pc1, _pc2 = st.columns([6, 1])
                _pc1.markdown(label)
                if _pc2.button("취소", key=f"pen_rsv_cancel_{oid}"):
                    update_pen_order_status(oid, "취소")
                    st.rerun()

        # ── 전체 이력 테이블 ──
        with st.expander(f"주문 이력 ({len(pen_orders)}건)", expanded=bool(done_orders)):
            _rows = []
            for o in reversed(pen_orders):
                _status_icon = {"대기": "", "완료": "", "실패": "", "취소": ""}.get(o.get("status", ""), "?")
                _atts = o.get("attempts", [])
                _last_att = _atts[-1].get("action", "") if _atts else ""
                _hist_method = o.get("method", "")
                if "지정가" in _hist_method and o.get("price_offset_pct") and "분산" not in _hist_method:
                    _hoff = o.get("price_offset_pct", 0)
                    _hist_method = f"지정가({_hoff:+.1f}%)"
                elif "분산" in _hist_method:
                    _hsm = o.get("sub_method", "시장가")
                    _hsp = o.get("splits", 0)
                    _hiv = o.get("interval_sec", 60)
                    _hcs = o.get("completed_splits", 0)
                    _hoff = o.get("price_offset_pct", 0)
                    _hsm_str = f"{_hsm} {_hoff:+.1f}%" if _hsm == "지정가" else _hsm
                    _hist_method = f"분산({_hsm_str} {_hsp}×{_hiv}초)"
                    if _hcs > 0:
                        _hist_method += f" {_hcs}/{_hsp}"
                _rows.append({
                    "상태": f"{_status_icon} {o.get('status', '')}",
                    "방향": o.get("side", ""),
                    "종목": _fmt_etf_code_name(o.get("etf_code", "")),
                    "수량": o.get("qty", 0),
                    "방식": _hist_method,
                    "지정가": f"{o.get('price', 0):,}" if o.get("price", 0) > 0 else "-",
                    "예정일시": o.get("scheduled_kst", ""),
                    "등록일시": o.get("created_at", ""),
                    "실행일시": o.get("executed_at", "") or "-",
                    "시도": len(_atts),
                    "최근시도": _last_att[:40] if _last_att else "-",
                    "결과": (o.get("result", "") or "-")[:60],
                })
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    else:
        st.info("등록된 예약 주문이 없습니다.")
