"""연금저축 수동 주문 탭."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.constants import IS_CLOUD
from src.utils.formatting import _fmt_etf_code_name, _code_only
from src.utils.kis import _compute_kis_balance_summary


def render_pension_manual_tab(
    trader,
    pen_bal_key: str,
    etf_codes: list[str],
    resolve_quote_code,
    get_current_price,
    get_daily_chart,
    get_orderbook,
):
    """수동 주문 탭 렌더링.

    Args:
        trader: KISTrader 인스턴스
        pen_bal_key: 잔고 세션 키
        etf_codes: 매매 대상 ETF 코드 목록
        resolve_quote_code: 시세 코드 변환 함수 (code) -> (quote_code, note)
        get_current_price: 현재가 조회 함수 (code) -> float
        get_daily_chart: 일봉 차트 조회 함수 (code, count) -> DataFrame
        get_orderbook: 호가 조회 함수 (code) -> dict
    """
    st.header("수동 주문")

    bal = st.session_state.get(pen_bal_key)
    if not bal:
        st.warning("잔고를 먼저 조회해 주세요.")
        return

    _bal_sum = _compute_kis_balance_summary(bal)
    cash = _bal_sum["cash"]
    buyable_cash = _bal_sum["buyable_cash"]
    holdings = _bal_sum["holdings"]

    if not etf_codes:
        etf_codes = ["360750"]

    etf_options = {_fmt_etf_code_name(c): c for c in etf_codes}
    selected_pen_label = st.selectbox("매매 ETF 선택", list(etf_options.keys()), key="pen_trade_etf_label")
    selected_pen_etf = etf_options[selected_pen_label]
    selected_pen_quote_etf, _pen_quote_note = resolve_quote_code(str(selected_pen_etf))
    _pen_quote_substituted = str(selected_pen_quote_etf) != str(selected_pen_etf)
    if _pen_quote_substituted and _pen_quote_note:
        st.warning(_pen_quote_note)
        st.caption(
            f"주문 코드는 {_fmt_etf_code_name(selected_pen_etf)} 그대로 유지됩니다. "
            f"시세/차트/호가만 {_fmt_etf_code_name(selected_pen_quote_etf)} 기준으로 표시합니다."
        )
    _pen_order_disabled = bool(IS_CLOUD or _pen_quote_substituted)

    pen_holding = next((h for h in holdings if str(h.get("code", "")) == str(selected_pen_etf)), None)
    pen_qty = int(pen_holding.get("qty", 0)) if pen_holding else 0

    # ── 상단 정보 바 ──
    cur_price = get_current_price(str(selected_pen_quote_etf))
    _pen_cur = float(cur_price) if cur_price and cur_price > 0 else 0
    _pen_eval = _pen_cur * pen_qty if _pen_cur > 0 else 0

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("현재가", f"{_pen_cur:,.0f}원" if _pen_cur > 0 else "-")
    pc2.metric(f"{_fmt_etf_code_name(selected_pen_etf)} 보유", f"{pen_qty}주")
    pc3.metric("평가금액", f"{_pen_eval:,.0f}원")
    pc4.metric("매수 가능금액", f"{buyable_cash:,.0f}원")

    # ═══ 일봉 차트 ═══
    _render_daily_chart(selected_pen_quote_etf, selected_pen_etf, get_daily_chart)

    st.divider()

    ob_col, order_col = st.columns([2, 3])

    # ── 좌: 호가창 ──
    with ob_col:
        _render_orderbook(selected_pen_quote_etf, _pen_cur, get_orderbook)

    # ── 우: 주문 패널 ──
    with order_col:
        if _pen_quote_substituted:
            st.info(
                f"{_fmt_etf_code_name(selected_pen_etf)} 주문은 비활성화했습니다. "
                f"설정에서 종목코드를 {_fmt_etf_code_name(selected_pen_quote_etf)}로 변경한 뒤 주문해 주세요."
            )
        buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

        with buy_tab:
            _render_buy_panel(trader, selected_pen_etf, _pen_cur, buyable_cash,
                              pen_bal_key, _pen_order_disabled)

        with sell_tab:
            _render_sell_panel(trader, selected_pen_etf, _pen_cur, pen_qty,
                               pen_bal_key, _pen_order_disabled)


def _render_daily_chart(quote_etf: str, order_etf: str, get_daily_chart):
    """일봉 캔들스틱 + 이동평균선 + 거래량 차트."""
    _pen_chart_df = get_daily_chart(str(quote_etf), count=260)
    if _pen_chart_df is None or len(_pen_chart_df) == 0:
        st.info("차트 데이터 로딩 중...")
        return

    _pen_chart_df = _pen_chart_df.copy().sort_index()
    if "close" not in _pen_chart_df.columns and "Close" in _pen_chart_df.columns:
        _pen_chart_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
    st.markdown(f"**{_fmt_etf_code_name(quote_etf)} 일봉 차트**")
    try:
        _chart_last_date = pd.to_datetime(_pen_chart_df.index[-1]).date()
        _chart_age_days = int((pd.Timestamp.now(tz="Asia/Seoul").date() - _chart_last_date).days)
        if _chart_age_days > 7:
            st.warning(
                f"일봉 최신일이 {_chart_last_date} (KST)로 오래되었습니다. "
                "실시간 조회 가능 종목코드인지 확인해 주세요."
            )
    except Exception:
        pass
    try:
        _chart_start = pd.to_datetime(_pen_chart_df.index[0]).date()
        _chart_end = pd.to_datetime(_pen_chart_df.index[-1]).date()
        st.caption(f"차트 구간: {_chart_start} ~ {_chart_end} | 캔들: {len(_pen_chart_df):,}개")
    except Exception:
        pass

    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
    _fig.add_trace(go.Candlestick(
        x=_pen_chart_df.index, open=_pen_chart_df['open'], high=_pen_chart_df['high'],
        low=_pen_chart_df['low'], close=_pen_chart_df['close'], name='일봉',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    ), row=1, col=1)

    _sma_colors = {5: "#FF9800", 20: "#2196F3", 60: "#00897B", 120: "#8D6E63", 200: "#455A64"}
    for _p, _color in _sma_colors.items():
        _sma = _pen_chart_df["close"].rolling(_p).mean()
        _fig.add_trace(
            go.Scatter(x=_pen_chart_df.index, y=_sma, name=f"SMA{_p}",
                       line=dict(color=_color, width=1.2 if _p < 200 else 1.5)),
            row=1, col=1,
        )

    if 'volume' in _pen_chart_df.columns:
        _vol_colors = ['#26a69a' if c >= o else '#ef5350'
                       for c, o in zip(_pen_chart_df['close'], _pen_chart_df['open'])]
        _fig.add_trace(go.Bar(x=_pen_chart_df.index, y=_pen_chart_df['volume'],
                              marker_color=_vol_colors, name='거래량', showlegend=False), row=2, col=1)

    _fig.update_layout(
        height=450, margin=dict(l=0, r=0, t=10, b=30),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0),
        xaxis2=dict(showticklabels=True, tickformat='%m/%d', tickangle=-45),
        yaxis=dict(title="", side="right"),
        yaxis2=dict(title="", side="right"),
    )
    st.plotly_chart(_fig, use_container_width=True, key=f"pen_manual_chart_{order_etf}")


def _render_orderbook(quote_etf: str, cur_price: float, get_orderbook):
    """호가창 렌더링."""
    ob = get_orderbook(str(quote_etf))
    if not ob or not ob.get("asks") or not ob.get("bids"):
        st.info("호가 데이터를 불러오는 중...")
        return

    asks = ob["asks"]
    bids = ob["bids"]
    all_qtys = [a["qty"] for a in asks] + [b["qty"] for b in bids]
    max_qty = max(all_qtys) if all_qtys else 1

    html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
    html.append(
        '<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666">'
        '<td>구분</td><td style="text-align:right">잔량</td>'
        '<td style="text-align:right">가격(원)</td>'
        '<td style="text-align:right">등락</td><td>비율</td></tr>'
    )
    for a in reversed(asks):
        ap, aq = a["price"], a["qty"]
        diff = ((ap / cur_price) - 1) * 100 if cur_price > 0 else 0
        bar_w = int(aq / max_qty * 100) if max_qty > 0 else 0
        html.append(
            f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
            f'<td>매도</td>'
            f'<td style="text-align:right">{aq:,}</td>'
            f'<td style="text-align:right;font-weight:bold">{ap:,.0f}</td>'
            f'<td style="text-align:right">{diff:+.2f}%</td>'
            f'<td><div style="background:#1976D233;height:14px;width:{bar_w}%"></div></td></tr>'
        )
    html.append(
        f'<tr style="background:#FFF3E0;border:2px solid #FF9800;height:32px;font-weight:bold">'
        f'<td colspan="2" style="color:#E65100">현재가</td>'
        f'<td style="text-align:right;color:#E65100;font-size:15px">{cur_price:,.0f}</td>'
        f'<td colspan="2"></td></tr>'
    )
    for b in bids:
        bp, bq = b["price"], b["qty"]
        diff = ((bp / cur_price) - 1) * 100 if cur_price > 0 else 0
        bar_w = int(bq / max_qty * 100) if max_qty > 0 else 0
        html.append(
            f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
            f'<td>매수</td>'
            f'<td style="text-align:right">{bq:,}</td>'
            f'<td style="text-align:right;font-weight:bold">{bp:,.0f}</td>'
            f'<td style="text-align:right">{diff:+.2f}%</td>'
            f'<td><div style="background:#D32F2F33;height:14px;width:{bar_w}%"></div></td></tr>'
        )
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)

    if asks and bids:
        spread = asks[0]["price"] - bids[0]["price"]
        spread_pct = (spread / cur_price * 100) if cur_price > 0 else 0
        total_ask_q = sum(a["qty"] for a in asks)
        total_bid_q = sum(b["qty"] for b in bids)
        st.caption(
            f"스프레드: {spread:,.0f}원 ({spread_pct:.3f}%) | "
            f"매도잔량: {total_ask_q:,} | 매수잔량: {total_bid_q:,}"
        )


def _render_buy_panel(trader, etf_code: str, cur_price: float, buyable_cash: float,
                      pen_bal_key: str, disabled: bool):
    """매수 주문 패널."""
    pen_buy_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"],
                              key="pen_buy_method", horizontal=True)

    pen_buy_price = 0
    if pen_buy_method == "지정가":
        pen_buy_price = st.number_input("매수 지정가 (원)", min_value=0,
                                        value=int(cur_price) if cur_price > 0 else 0,
                                        step=50, key="pen_buy_price")
    else:
        pen_buy_price = cur_price

    pen_buy_qty = st.number_input("매수 수량 (주)", min_value=0, value=0, step=1, key="pen_buy_qty")

    _pen_buy_unit = pen_buy_price if pen_buy_price > 0 else cur_price
    _pen_total = int(pen_buy_qty * _pen_buy_unit) if pen_buy_qty > 0 and _pen_buy_unit > 0 else 0
    st.markdown(
        f"<div style='background:#fff3f3;border:1px solid #ffcdd2;border-radius:8px;padding:10px 14px;margin:8px 0'>"
        f"<b>단가:</b> {_pen_buy_unit:,.0f}원 &nbsp;|&nbsp; "
        f"<b>수량:</b> {pen_buy_qty:,}주 &nbsp;|&nbsp; "
        f"<b>총 금액:</b> <span style='color:#D32F2F;font-weight:bold'>{_pen_total:,}원</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("매수 실행", key="pen_exec_buy", type="primary", disabled=disabled):
        if _pen_total <= 0:
            st.error("매수 금액을 입력해 주세요.")
        elif _pen_total > buyable_cash:
            st.error(f"매수 가능금액 부족 (필요: {_pen_total:,}원 / 가능: {buyable_cash:,.0f}원)")
        else:
            with st.spinner("매수 주문 실행 중..."):
                if pen_buy_method == "동시호가 (장마감)":
                    result = trader.execute_closing_auction_buy(str(etf_code), pen_buy_qty) if pen_buy_qty > 0 else None
                elif pen_buy_method == "지정가" and pen_buy_price > 0:
                    result = trader.send_order("BUY", str(etf_code), pen_buy_qty, price=pen_buy_price, ord_dvsn="00") if pen_buy_qty > 0 else None
                elif pen_buy_method == "시간외 종가":
                    result = trader.send_order("BUY", str(etf_code), pen_buy_qty, price=0, ord_dvsn="06") if pen_buy_qty > 0 else None
                else:
                    result = trader.send_order("BUY", str(etf_code), pen_buy_qty, ord_dvsn="01") if pen_buy_qty > 0 else None
                if result and (isinstance(result, dict) and result.get("success")):
                    st.success(f"매수 완료: {result}")
                    st.session_state[pen_bal_key] = trader.get_balance()
                else:
                    st.error(f"매수 실패: {result}")


def _render_sell_panel(trader, etf_code: str, cur_price: float, pen_qty: int,
                       pen_bal_key: str, disabled: bool):
    """매도 주문 패널."""
    pen_sell_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"],
                               key="pen_sell_method", horizontal=True)

    pen_sell_price = 0
    if pen_sell_method == "지정가":
        pen_sell_price = st.number_input("매도 지정가 (원)", min_value=0,
                                         value=int(cur_price) if cur_price > 0 else 0,
                                         step=50, key="pen_sell_price")
    else:
        pen_sell_price = cur_price

    pen_sell_qty = st.number_input("매도 수량 (주)", min_value=0, max_value=max(pen_qty, 1),
                                    value=pen_qty, step=1, key="pen_sell_qty")
    pen_sell_all = st.checkbox("전량 매도", value=True, key="pen_sell_all")

    _pen_sell_unit = pen_sell_price if pen_sell_price > 0 else cur_price
    _pen_sell_qty_final = pen_qty if pen_sell_all else pen_sell_qty
    _pen_sell_total = int(_pen_sell_qty_final * _pen_sell_unit) if _pen_sell_qty_final > 0 and _pen_sell_unit > 0 else 0
    st.markdown(
        f"<div style='background:#f3f8ff;border:1px solid #bbdefb;border-radius:8px;padding:10px 14px;margin:8px 0'>"
        f"<b>단가:</b> {_pen_sell_unit:,.0f}원 &nbsp;|&nbsp; "
        f"<b>수량:</b> {_pen_sell_qty_final:,}주 &nbsp;|&nbsp; "
        f"<b>총 금액:</b> <span style='color:#1976D2;font-weight:bold'>{_pen_sell_total:,}원</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button("매도 실행", key="pen_exec_sell", type="primary", disabled=disabled):
        _sq = pen_qty if pen_sell_all else pen_sell_qty
        if _sq <= 0:
            st.error("매도할 수량이 없습니다.")
        else:
            with st.spinner("매도 주문 실행 중..."):
                if pen_sell_method == "동시호가 (장마감)":
                    result = trader.smart_sell_all_closing(str(etf_code)) if pen_sell_all else trader.smart_sell_qty_closing(str(etf_code), _sq)
                elif pen_sell_method == "지정가" and pen_sell_price > 0:
                    result = trader.send_order("SELL", str(etf_code), _sq, price=pen_sell_price, ord_dvsn="00")
                elif pen_sell_method == "시간외 종가":
                    result = trader.send_order("SELL", str(etf_code), _sq, price=0, ord_dvsn="06")
                else:
                    result = trader.smart_sell_all(str(etf_code)) if pen_sell_all else trader.smart_sell_qty(str(etf_code), _sq)
                if result and (isinstance(result, dict) and result.get("success")):
                    st.success(f"매도 완료: {result}")
                    st.session_state[pen_bal_key] = trader.get_balance()
                else:
                    st.error(f"매도 실패: {result}")
