"""미국주식 모드 — 예약주문 관리 UI."""
import json
import os
import streamlit as st
from datetime import datetime

ORDERS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "us_stock_orders.json")


def _load_orders() -> list:
    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_orders(orders: list):
    os.makedirs(os.path.dirname(ORDERS_PATH), exist_ok=True)
    with open(ORDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)


def _gen_order_id() -> str:
    return f"us-{datetime.now().strftime('%Y%m%d%H%M%S')}-{id(object()) % 10000:04d}"


def render_us_stock_orders_tab(trader=None):
    """예약주문 등록/관리 탭."""
    orders = _load_orders()

    st.subheader("예약주문 등록")
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("종목코드 (예: TSLA)", key="us_ord_symbol").upper().strip()
    with c2:
        side = st.selectbox("매수/매도", ["매수", "매도"], key="us_ord_side")
    with c3:
        market = st.selectbox("거래소", ["NASDAQ (82)", "NYSE (81)"], key="us_ord_market")
    market_code = "82" if "82" in market else "81"

    c4, c5 = st.columns(2)
    with c4:
        qty = st.number_input("수량", min_value=1, value=1, step=1, key="us_ord_qty")
    with c5:
        price_type = st.selectbox("주문유형", ["지정가", "시장가"], key="us_ord_ptype")

    price = 0.0
    if price_type == "지정가":
        price = st.number_input("주문가격 (USD)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="us_ord_price")

    scheduled_kst = st.text_input(
        "예약 시각 (KST, 예: 2026-03-16 23:30)",
        value=datetime.now().strftime("%Y-%m-%d 23:30"),
        key="us_ord_schedule",
    )
    note = st.text_input("메모 (선택)", key="us_ord_note")

    if st.button("예약주문 등록", key="us_ord_add"):
        if not symbol:
            st.error("종목코드를 입력하세요.")
        elif price_type == "지정가" and price <= 0:
            st.error("지정가 주문은 가격을 입력해야 합니다.")
        else:
            new_order = {
                "id": _gen_order_id(),
                "symbol": symbol,
                "market": market_code,
                "side": side,
                "qty": int(qty),
                "price": float(price),
                "price_type": price_type,
                "scheduled_kst": scheduled_kst.strip(),
                "status": "대기",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "executed_at": None,
                "result": None,
                "note": note.strip(),
                "attempts": [],
            }
            orders.append(new_order)
            _save_orders(orders)
            st.success(f"예약주문 등록: {side} {symbol} x{qty}")
            st.rerun()

    # ── 주문 목록 ──
    st.divider()
    st.subheader("예약주문 목록")

    if not orders:
        st.info("등록된 예약주문이 없습니다.")
        return

    # 상태별 필터
    _statuses = sorted(set(o.get("status", "") for o in orders))
    _filter = st.multiselect("상태 필터", _statuses, default=["대기"] if "대기" in _statuses else _statuses, key="us_ord_filter")
    filtered = [o for o in orders if o.get("status", "") in _filter] if _filter else orders

    for i, o in enumerate(filtered):
        _status = o.get("status", "")
        _icon = {"대기": "🟡", "완료": "🟢", "실패": "🔴", "취소": "⚪"}.get(_status, "⚫")
        _label = f"{_icon} {o.get('side','')} {o.get('symbol','')} x{o.get('qty',0)}"
        if o.get("price_type") == "지정가":
            _label += f" @${o.get('price', 0):.2f}"
        _label += f" | {o.get('scheduled_kst', '')} | {_status}"

        with st.expander(_label, expanded=(_status == "대기")):
            st.json(o)
            c_a, c_b, c_c = st.columns(3)
            if _status == "대기":
                with c_a:
                    if st.button("즉시 실행", key=f"us_exec_{o['id']}"):
                        _execute_order(o, trader)
                        _save_orders(orders)
                        st.rerun()
                with c_b:
                    if st.button("취소", key=f"us_cancel_{o['id']}"):
                        o["status"] = "취소"
                        _save_orders(orders)
                        st.rerun()
            with c_c:
                if st.button("삭제", key=f"us_del_{o['id']}"):
                    orders.remove(o)
                    _save_orders(orders)
                    st.rerun()


def _execute_order(order: dict, trader=None):
    """단건 예약주문 실행."""
    if trader is None:
        order["status"] = "실패"
        order["result"] = "트레이더 없음"
        return

    symbol = order.get("symbol", "")
    side = "BUY" if order.get("side") == "매수" else "SELL"
    qty = int(order.get("qty", 0))
    price = float(order.get("price", 0))
    market = str(order.get("market", "82"))
    price_type = "00" if order.get("price_type") == "지정가" else "03"

    attempt = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        result = trader.send_order(
            side=side, symbol=symbol, qty=qty,
            price=price, market=market, price_type=price_type,
        )
        attempt["result"] = result
        if result and result.get("success"):
            order["status"] = "완료"
            order["executed_at"] = attempt["time"]
            order["result"] = result.get("msg", "OK")
        else:
            order["status"] = "실패"
            order["result"] = (result or {}).get("msg", "주문 실패")
    except Exception as e:
        attempt["error"] = str(e)
        order["status"] = "실패"
        order["result"] = str(e)

    order.setdefault("attempts", []).append(attempt)


def execute_pending_us_orders(trader) -> list[str]:
    """예정 시간이 지난 대기 주문을 자동 실행. 결과 메시지 리스트 반환."""
    orders = _load_orders()
    now = datetime.now()
    results = []

    changed = False
    for o in orders:
        if o.get("status") != "대기":
            continue
        _sched = o.get("scheduled_kst", "")
        try:
            sched_dt = datetime.strptime(_sched, "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            continue
        if now < sched_dt:
            continue

        _execute_order(o, trader)
        changed = True
        _msg = f"[US] {o.get('side','')} {o.get('symbol','')} x{o.get('qty',0)} → {o.get('status','')}: {o.get('result','')}"
        results.append(_msg)

    if changed:
        _save_orders(orders)
    return results
