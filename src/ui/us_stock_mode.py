"""미국주식(LS증권) 모드 — 메인 UI."""
import streamlit as st
import time
import logging
from datetime import datetime

from src.ui.components.us_stock_sidebar import render_us_stock_sidebar
from src.ui.components.us_stock_orders import (
    render_us_stock_orders_tab,
    execute_pending_us_orders,
)

logger = logging.getLogger(__name__)


def render_us_stock_mode(config, save_config):
    """LS증권 미국주식 모드 렌더링."""
    from src.engine.ls_trader import LSTrader

    # ── 사이드바 ──
    _sb = render_us_stock_sidebar(config, save_config)
    if _sb is None:
        st.info("사이드바에서 LS증권 API 설정을 완료하세요.")
        return

    # ── 트레이더 초기화 ──
    trader = LSTrader()
    trader.app_key = _sb["ls_app_key"]
    trader.secret_key = _sb["ls_secret_key"]
    trader.account_no = _sb["ls_account_no"]
    trader.account_pwd = _sb["ls_account_pwd"]

    # 세션 토큰 캐싱
    _tok_key = f"ls_token_{_sb['ls_account_no']}"
    _cached = st.session_state.get(_tok_key)
    if _cached and _cached.get("api") is not None:
        trader.api = _cached["api"]
        trader._authenticated = True
    else:
        if not trader.auth():
            st.error("LS증권 로그인에 실패했습니다. API 키를 확인해 주세요.")
            return
        st.session_state[_tok_key] = {"api": trader.api, "ts": time.time()}

    # ── 예약주문 자동 실행 ──
    _us_exec_key = "_us_orders_last_check"
    _us_last = st.session_state.get(_us_exec_key, "")
    _us_now_min = datetime.now().strftime("%Y-%m-%d %H:%M")
    if _us_last != _us_now_min:
        st.session_state[_us_exec_key] = _us_now_min
        _exec_results = execute_pending_us_orders(trader)
        for _er in _exec_results:
            st.toast(_er)

    # ── 탭 구성 ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 잔고/시세",
        "🛒 수동주문",
        "📋 예약주문",
        "📜 미체결",
    ])

    with tab1:
        _render_balance_tab(trader)
    with tab2:
        _render_manual_order_tab(trader)
    with tab3:
        render_us_stock_orders_tab(trader)
    with tab4:
        _render_pending_orders_tab(trader)


# ═════════════════════════════════════════════════════════════
# Tab 1: 잔고 / 시세
# ═════════════════════════════════════════════════════════════

def _render_balance_tab(trader):
    st.subheader("보유종목")

    if st.button("잔고 새로고침", key="us_refresh_bal"):
        st.session_state.pop("_us_balance", None)

    bal = st.session_state.get("_us_balance")
    if bal is None:
        with st.spinner("잔고 조회 중..."):
            bal = trader.get_balance()
        if bal:
            st.session_state["_us_balance"] = bal

    if not bal:
        st.warning("잔고 조회에 실패했습니다.")
        return

    # 예수금
    dep = trader.get_deposit()
    if dep:
        c1, c2, c3 = st.columns(3)
        c1.metric("USD 가용", f"${dep.get('usd_available', 0):,.2f}")
        c2.metric("KRW 예수금", f"₩{dep.get('krw_deposit', 0):,.0f}")
        c3.metric("환율", f"{dep.get('exchange_rate', 0):,.2f}")

    # 보유 종목 테이블
    holdings = bal.get("holdings", [])
    if not holdings:
        st.info("보유 종목이 없습니다.")
    else:
        import pandas as pd
        rows = []
        for h in holdings:
            rows.append({
                "종목": h.get("code", ""),
                "수량": int(h.get("qty", 0)),
                "매도가능": int(h.get("sellable_qty", 0)),
                "평균단가": f"${h.get('avg_price', 0):,.2f}",
                "현재가": f"${h.get('cur_price', 0):,.2f}",
                "수익률": f"{h.get('pnl_rate', 0):+.2f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 시세 조회
    st.divider()
    st.subheader("시세 조회")
    c1, c2 = st.columns([3, 1])
    with c1:
        _sym = st.text_input("종목코드", value="TSLA", key="us_quote_sym").upper().strip()
    with c2:
        _mkt = st.selectbox("거래소", ["NASDAQ (82)", "NYSE (81)"], key="us_quote_mkt")
    _mkt_code = "82" if "82" in _mkt else "81"

    if st.button("시세 조회", key="us_get_quote") and _sym:
        with st.spinner(f"{_sym} 시세 조회 중..."):
            q = trader.get_price(_sym, market=_mkt_code)
        if q:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(q.get("name", _sym), f"${q.get('price', 0):,.2f}")
            c2.metric("등락", f"${q.get('change', 0):+,.2f}", f"{q.get('change_rate', 0):+.2f}%")
            c3.metric("고가/저가", f"${q.get('high', 0):,.2f} / ${q.get('low', 0):,.2f}")
            c4.metric("거래량", f"{q.get('volume', 0):,.0f}")
        else:
            st.warning(f"{_sym} 시세를 조회할 수 없습니다.")


# ═════════════════════════════════════════════════════════════
# Tab 2: 수동 주문
# ═════════════════════════════════════════════════════════════

def _render_manual_order_tab(trader):
    st.subheader("수동 주문")

    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("종목코드", value="", key="us_manual_sym").upper().strip()
    with c2:
        side = st.selectbox("매수/매도", ["매수", "매도"], key="us_manual_side")
    with c3:
        market = st.selectbox("거래소", ["NASDAQ (82)", "NYSE (81)"], key="us_manual_mkt")
    market_code = "82" if "82" in market else "81"

    c4, c5, c6 = st.columns(3)
    with c4:
        qty = st.number_input("수량", min_value=1, value=1, step=1, key="us_manual_qty")
    with c5:
        price_type = st.selectbox("주문유형", ["지정가", "시장가"], key="us_manual_ptype")
    with c6:
        price = 0.0
        if price_type == "지정가":
            price = st.number_input("가격 (USD)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="us_manual_price")

    if st.button("주문 전송", key="us_manual_send", type="primary"):
        if not symbol:
            st.error("종목코드를 입력하세요.")
        elif price_type == "지정가" and price <= 0:
            st.error("지정가 주문은 가격을 입력하세요.")
        else:
            _side = "BUY" if side == "매수" else "SELL"
            _pt = "00" if price_type == "지정가" else "03"
            with st.spinner(f"{side} {symbol} x{qty} 주문 전송 중..."):
                result = trader.send_order(
                    side=_side, symbol=symbol, qty=int(qty),
                    price=float(price), market=market_code, price_type=_pt,
                )
            if result and result.get("success"):
                st.success(f"주문 성공: {result.get('msg', '')}")
                st.session_state.pop("_us_balance", None)
            else:
                st.error(f"주문 실패: {(result or {}).get('msg', '알 수 없는 오류')}")


# ═════════════════════════════════════════════════════════════
# Tab 4: 미체결 주문
# ═════════════════════════════════════════════════════════════

def _render_pending_orders_tab(trader):
    st.subheader("미체결 주문")

    _mkt = st.selectbox("거래소", ["NASDAQ (82)", "NYSE (81)"], key="us_pending_mkt")
    _mkt_code = "82" if "82" in _mkt else "81"

    if st.button("미체결 조회", key="us_refresh_pending"):
        st.session_state.pop("_us_pending", None)

    pending = st.session_state.get("_us_pending")
    if pending is None:
        with st.spinner("미체결 조회 중..."):
            pending = trader.get_pending_orders(market=_mkt_code)
        st.session_state["_us_pending"] = pending

    if not pending:
        st.info("미체결 주문이 없습니다.")
        return

    import pandas as pd
    rows = []
    for p in pending:
        rows.append({
            "주문번호": p.get("ord_no", ""),
            "종목": p.get("symbol", ""),
            "구분": p.get("side", ""),
            "주문수량": p.get("qty", 0),
            "미체결": p.get("unfilled_qty", 0),
            "주문가": f"${p.get('price', 0):,.2f}",
            "주문시각": p.get("ord_time", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 취소 기능
    st.divider()
    st.subheader("주문 취소")
    _ord_nos = [str(p.get("ord_no", "")) for p in pending]
    _sel = st.selectbox("취소할 주문번호", _ord_nos, key="us_cancel_sel")
    if st.button("선택 주문 취소", key="us_cancel_btn"):
        with st.spinner("주문 취소 중..."):
            result = trader.cancel_order(int(_sel), market=_mkt_code)
        if result and result.get("success"):
            st.success(f"취소 성공: {result.get('msg', '')}")
            st.session_state.pop("_us_pending", None)
            st.session_state.pop("_us_balance", None)
            st.rerun()
        else:
            st.error(f"취소 실패: {(result or {}).get('msg', '')}")
