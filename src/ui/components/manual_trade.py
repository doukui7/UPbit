"""수동 주문 탭 — 업비트 스타일 호가창, 시장가/지정가 매수·매도, 미체결/체결 주문."""
import json
import streamlit as st
import streamlit.components.v1 as st_components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.ui.coin_utils import (
    ttl_cache, clear_cache, load_account_cache,
    trigger_and_wait_gh, TOP_20_TICKERS,
)


def _build_orderbook_html(units, cur_price, max_size):
    """업비트 스타일 호가창 HTML 생성."""
    rows = []
    # 매도 호가 (위쪽, 높은가→낮은가, 파란색)
    for u in reversed(units):
        ask_p = u.get('ask_price', 0)
        ask_s = u.get('ask_size', 0)
        pct = ((ask_p / cur_price) - 1) * 100 if cur_price > 0 else 0
        bar_w = (ask_s / max_size * 100) if max_size > 0 else 0
        pct_color = '#1976d2'
        rows.append(
            f'<div class="ob-row ob-ask" data-price="{int(ask_p)}">'
            f'<div class="ob-bar" style="width:{bar_w:.1f}%;background:rgba(30,136,229,0.15);"></div>'
            f'<span class="ob-qty">{ask_s:.4f}</span>'
            f'<span class="ob-price" style="color:{pct_color};">{int(ask_p):,}</span>'
            f'<span class="ob-pct" style="color:{pct_color};">{pct:+.2f}%</span>'
            f'</div>'
        )

    # 현재가 중앙 표시
    rows.append(
        f'<div class="ob-current">'
        f'<span style="font-size:15px;font-weight:700;">{int(cur_price):,}</span>'
        f'<span style="font-size:11px;margin-left:8px;opacity:0.7;">현재가</span>'
        f'</div>'
    )

    # 매수 호가 (아래쪽, 높은가→낮은가, 빨간색)
    for u in units:
        bid_p = u.get('bid_price', 0)
        bid_s = u.get('bid_size', 0)
        pct = ((bid_p / cur_price) - 1) * 100 if cur_price > 0 else 0
        bar_w = (bid_s / max_size * 100) if max_size > 0 else 0
        pct_color = '#d32f2f'
        rows.append(
            f'<div class="ob-row ob-bid" data-price="{int(bid_p)}">'
            f'<div class="ob-bar" style="width:{bar_w:.1f}%;background:rgba(211,47,47,0.15);"></div>'
            f'<span class="ob-qty">{bid_s:.4f}</span>'
            f'<span class="ob-price" style="color:{pct_color};">{int(bid_p):,}</span>'
            f'<span class="ob-pct" style="color:{pct_color};">{pct:+.2f}%</span>'
            f'</div>'
        )

    style = """
<style>
.ob-wrap{border:1px solid #e0e0e0;border-radius:6px;overflow:hidden;font-family:-apple-system,sans-serif;font-size:12px;}
.ob-hdr{display:flex;background:#f5f5f5;padding:4px 10px;border-bottom:1px solid #e0e0e0;font-weight:600;color:#666;}
.ob-hdr span{flex:1;text-align:center;}
.ob-row{display:flex;align-items:center;height:26px;padding:0 10px;position:relative;border-bottom:1px solid #f5f5f5;cursor:pointer;transition:background .1s;}
.ob-row:hover{background:#f0f0f0 !important;}
.ob-bar{position:absolute;right:0;top:0;bottom:0;z-index:0;pointer-events:none;}
.ob-qty{flex:1;text-align:left;z-index:1;font-variant-numeric:tabular-nums;}
.ob-price{flex:1;text-align:center;font-weight:600;z-index:1;font-variant-numeric:tabular-nums;}
.ob-pct{flex:1;text-align:right;z-index:1;font-variant-numeric:tabular-nums;}
.ob-current{display:flex;align-items:center;justify-content:center;height:32px;background:#1a1a2e;color:#fff;gap:4px;}
</style>
"""
    header = (
        '<div class="ob-hdr">'
        '<span>잔량</span><span>가격</span><span>등락</span>'
        '</div>'
    )
    return style + '<div class="ob-wrap">' + header + '\n'.join(rows) + '</div>'


def _get_tick_size(price):
    """업비트 KRW 마켓 호가 단위 반환."""
    if price >= 2_000_000: return 1000
    if price >= 1_000_000: return 500
    if price >= 500_000: return 100
    if price >= 100_000: return 50
    if price >= 10_000: return 10
    if price >= 1_000: return 5
    if price >= 100: return 1
    if price >= 10: return 0.1
    if price >= 1: return 0.01
    return 0.001


def _round_to_tick(price):
    """호가 단위에 맞게 내림 (업비트 규칙)."""
    tick = _get_tick_size(price)
    if tick >= 1:
        return int(price // tick * tick)
    return round(price // tick * tick, 4)


def render_manual_trade_tab(portfolio_list, trader=None):
    # 호가 클릭 처리 (iframe JS → query param → session_state)
    if 'ob_price' in st.query_params:
        try:
            _click_p = int(st.query_params['ob_price'])
            st.session_state["mt_sell_price"] = _click_p
            st.session_state["mt_buy_price"] = _click_p
        except (ValueError, TypeError):
            pass
        del st.query_params['ob_price']
        st.rerun()

    st.header("수동 주문")

    # ── 잔고 표시 ──
    _acct = load_account_cache()
    bals = _acct.get("balances", {}) if _acct.get("updated_at") else {}
    krw_balance = float(bals.get("KRW", 0) or 0)
    if bals:
        bal_parts = []
        if krw_balance > 0:
            bal_parts.append(f"KRW: {krw_balance:,.0f}")
        for k, v in bals.items():
            if k != "KRW" and float(v or 0) > 0:
                bal_parts.append(f"{k}: {float(v):.8f}")
        if bal_parts:
            st.caption(f"잔고 ({_acct['updated_at']}): " + " | ".join(bal_parts))

    # ── 코인 선택 + 30분봉 차트 ──
    port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
    manual_options = list(dict.fromkeys(port_tickers + TOP_20_TICKERS))
    mt_ticker = st.selectbox("코인 선택", manual_options, key="mt_ticker_chart")
    import pyupbit as _pyupbit

    df_30m = ttl_cache(
        f"m30_{mt_ticker}",
        lambda: _pyupbit.get_ohlcv(mt_ticker, interval="minute30", count=48),
        ttl=10,
    )
    if df_30m is not None and len(df_30m) > 0:
        last_dt = df_30m.index[-1]
        refresh_text = last_dt.strftime('%Y-%m-%d %H:%M') if hasattr(last_dt, 'strftime') else str(last_dt)
        _c_title, _c_time = st.columns([3, 1])
        _c_title.markdown("**30분봉 차트**")
        _c_time.caption(f"최종 봉: {refresh_text}")
        fig_30m = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
        fig_30m.add_trace(go.Candlestick(
            x=df_30m.index, open=df_30m['open'], high=df_30m['high'],
            low=df_30m['low'], close=df_30m['close'], name='30분봉',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        ), row=1, col=1)
        ma5 = df_30m['close'].rolling(5).mean()
        ma20 = df_30m['close'].rolling(20).mean()
        fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma5, name='MA5', line=dict(color='#FF9800', width=1)), row=1, col=1)
        fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma20, name='MA20', line=dict(color='#2196F3', width=1)), row=1, col=1)
        colors_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_30m['close'], df_30m['open'])]
        fig_30m.add_trace(go.Bar(
            x=df_30m.index, y=df_30m['volume'], marker_color=colors_vol, name='거래량', showlegend=False
        ), row=2, col=1)
        fig_30m.update_layout(
            height=520, margin=dict(l=0, r=0, t=10, b=30),
            xaxis_rangeslider_visible=False, showlegend=True,
            legend=dict(orientation="h", y=1.06, x=0),
            xaxis2=dict(showticklabels=True, tickformat='%H:%M', tickangle=-45),
            yaxis=dict(title="", side="right"),
            yaxis2=dict(title="", side="right"),
        )
        st.plotly_chart(fig_30m, use_container_width=True, key=f"chart30m_{mt_ticker}")
    else:
        st.info("차트 데이터 로딩 중...")

    st.divider()

    # ═══ 호가창 + 주문 패널 ═══
    mt_coin = mt_ticker.split("-")[1] if "-" in mt_ticker else mt_ticker

    # 보유 수량/금액 조회
    coin_holding = float(bals.get(mt_coin, 0) or 0)
    ob_data = ttl_cache(f"ob_{mt_ticker}", lambda: _pyupbit.get_orderbook(mt_ticker), ttl=5)
    cur_price = ttl_cache(f"price_{mt_ticker}", lambda: _pyupbit.get_current_price(mt_ticker) or 0, ttl=5)
    coin_value = coin_holding * cur_price if cur_price > 0 else 0

    # 주문 가능 정보 (수수료율 등)
    chance_info = None
    if trader:
        chance_info = ttl_cache(
            f"chance_{mt_ticker}",
            lambda: trader.upbit.get_chance(mt_ticker),
            ttl=30,
        )

    # 수수료율 추출
    bid_fee = 0.0005  # 기본값 0.05%
    ask_fee = 0.0005
    if chance_info and isinstance(chance_info, dict):
        try:
            bid_fee = float(chance_info.get('bid_fee', 0.0005) or 0.0005)
            ask_fee = float(chance_info.get('ask_fee', 0.0005) or 0.0005)
        except (ValueError, TypeError):
            pass

    ob_col, order_col = st.columns([2, 3])

    # ── 좌: 호가창 (업비트 스타일) ──
    with ob_col:
        st.markdown("**호가창**")
        try:
            if ob_data and len(ob_data) > 0:
                ob = ob_data[0] if isinstance(ob_data, list) else ob_data
                units = ob.get('orderbook_units', [])[:15]

                if units:
                    max_size = max(
                        max(u.get('ask_size', 0) for u in units),
                        max(u.get('bid_size', 0) for u in units),
                    )
                    # HTML 호가창 + 클릭 JS
                    html = _build_orderbook_html(units, cur_price, max_size)
                    click_js = """<script>
document.querySelectorAll('.ob-row').forEach(row => {
    row.addEventListener('click', function() {
        const price = this.dataset.price;
        const url = new URL(window.parent.location);
        url.searchParams.set('ob_price', price);
        window.parent.location.href = url.toString();
    });
});
</script>"""
                    ob_height = 28 + len(units) * 52 + 37
                    st_components.html(html + click_js, height=ob_height, scrolling=False)

                    # 스프레드 및 매수비율
                    best_ask = units[0].get('ask_price', 0)
                    best_bid = units[0].get('bid_price', 0)
                    spread = best_ask - best_bid
                    spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
                    total_ask = ob.get('total_ask_size', 0)
                    total_bid = ob.get('total_bid_size', 0)
                    ob_ratio = total_bid / (total_ask + total_bid) * 100 if (total_ask + total_bid) > 0 else 50
                    st.caption(
                        f"스프레드 **{spread:,.0f}** ({spread_pct:.3f}%) | "
                        f"매도 {total_ask:.2f} | 매수 {total_bid:.2f} | 매수비율 {ob_ratio:.0f}%"
                    )
                else:
                    st.info("호가 데이터가 없습니다.")
            else:
                st.info("호가 데이터를 불러올 수 없습니다.")
        except Exception as e:
            st.warning(f"호가 조회 실패: {e}")

    # ── 우: 주문 패널 ──
    with order_col:
        st.markdown("**주문 실행 (VM 경유)**")

        # 수수료 정보 표시
        _fee_parts = [f"매수 수수료 **{bid_fee*100:.3f}%**", f"매도 수수료 **{ask_fee*100:.3f}%**"]
        if chance_info and isinstance(chance_info, dict):
            _max_total = chance_info.get('max_total')
            if _max_total:
                _fee_parts.append(f"최대 주문 **{float(_max_total):,.0f} KRW**")
        tick = _get_tick_size(cur_price)
        _fee_parts.append(f"호가단위 **{tick:g}**")
        st.caption(" | ".join(_fee_parts))

        buy_tab, sell_tab = st.tabs(["매수", "매도"])

        def _order_and_sync(order_json, status_el, label):
            ok, msg = trigger_and_wait_gh(
                "manual_order", status_el,
                extra_inputs={"manual_order_params": order_json},
            )
            if ok:
                status_el.success(f"{label} 완료 ({msg})")
                status_el.info("잔고 동기화 중...")
                trigger_and_wait_gh("account_sync", status_el)
                clear_cache("krw_bal_t1", "balances_t1", "prices_t1")
                status_el.success(f"{label} 완료 · 잔고 갱신됨")
            else:
                status_el.error(f"{label} 실패: {msg}")

        with buy_tab:
            # 가용 KRW 표시 (수수료 고려)
            if krw_balance > 0:
                usable_krw = krw_balance / (1 + bid_fee)
                st.caption(f"주문 가능: **{krw_balance:,.0f} KRW** (수수료 제외 실주문: {usable_krw:,.0f})")

            buy_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_buy_type")

            if buy_type == "시장가":
                def _sync_buy_slider(): st.session_state["mt_buy_pct"] = st.session_state["mt_buy_pct_slider"]
                def _sync_buy_input(): st.session_state["mt_buy_pct_slider"] = st.session_state["mt_buy_pct"]
                _bp1, _bp2 = st.columns([1, 2])
                buy_pct = _bp1.number_input("매수 비율 (%)", min_value=1, max_value=100, value=50, step=1,
                                            key="mt_buy_pct", on_change=_sync_buy_input)
                _bp2.slider("비율 조절", 1, 100, value=50, step=1, key="mt_buy_pct_slider",
                            label_visibility="collapsed", on_change=_sync_buy_slider)
                est_amount = krw_balance * buy_pct / 100
                est_after_fee = est_amount / (1 + bid_fee)
                st.caption(
                    f"가용 KRW의 {buy_pct}% = **{est_amount:,.0f} KRW** "
                    f"(수수료 {est_amount - est_after_fee:,.0f} → 실주문 {est_after_fee:,.0f})"
                )

                if st.button(
                    f"{mt_coin} {buy_pct}% 시장가 매수",
                    key="mt_buy_vm", type="primary", use_container_width=True,
                ):
                    _buy_status = st.empty()
                    _buy_status.info("3분할 체결 진행 중 (~3분 소요)...")
                    _buy_json = json.dumps({"coin": mt_coin, "side": "buy", "pct": buy_pct})
                    _order_and_sync(_buy_json, _buy_status, "시장가 매수")

            else:  # 지정가
                def _on_buy_vol():
                    _p = st.session_state.get("mt_buy_price", 0)
                    _v = st.session_state.get("mt_buy_vol", 0)
                    if _p > 0 and _v > 0:
                        st.session_state["mt_buy_amount"] = int(_p * _v)

                def _on_buy_amount():
                    _p = st.session_state.get("mt_buy_price", 0)
                    _a = st.session_state.get("mt_buy_amount", 0)
                    if _p > 0 and _a > 0:
                        st.session_state["mt_buy_vol"] = _a / _p

                _bc1, _bc2 = st.columns(2)
                _default_buy = _round_to_tick(cur_price * 0.99) if cur_price > 0 else 1
                buy_price = _bc1.number_input(
                    "매수 가격 (KRW)", min_value=1,
                    value=_default_buy,
                    step=max(1, int(_get_tick_size(cur_price))),
                    key="mt_buy_price",
                )
                # 호가 단위 보정 안내
                tick_adj = _round_to_tick(buy_price)
                if tick_adj != buy_price:
                    _bc1.caption(f"호가 보정: {tick_adj:,}")

                buy_vol = _bc2.number_input(
                    f"매수 수량 ({mt_coin})", min_value=0.00000001, value=0.001,
                    format="%.8f", key="mt_buy_vol", on_change=_on_buy_vol,
                )
                _buy_amount = st.number_input(
                    "매수 금액 (KRW)", min_value=0, value=int(buy_price * buy_vol),
                    step=10000, key="mt_buy_amount", on_change=_on_buy_amount,
                )
                buy_total = buy_price * buy_vol
                fee_amount = buy_total * bid_fee
                st.caption(f"총액: **{buy_total:,.0f} KRW** (수수료 {fee_amount:,.0f} → 합계 {buy_total + fee_amount:,.0f})")

                if st.button(
                    f"{mt_coin} 지정가 매수 ({buy_price:,.0f} × {buy_vol:.8g})",
                    key="mt_lbuy_vm", type="primary", use_container_width=True,
                ):
                    # 호가 단위 보정 적용
                    final_price = _round_to_tick(buy_price)
                    final_total = final_price * buy_vol
                    if final_total < 5000:
                        st.error("최소 주문금액: 5,000 KRW")
                    else:
                        _buy_json = json.dumps({
                            "coin": mt_coin, "side": "buy",
                            "order_type": "limit", "price": final_price, "volume": buy_vol,
                        })
                        _order_and_sync(_buy_json, st.empty(), "지정가 매수")

        with sell_tab:
            # 보유 수량/금액 표시
            if coin_holding > 0:
                est_sell_fee = coin_value * ask_fee
                st.markdown(
                    f"**보유 {mt_coin}**: {coin_holding:.8g} "
                    f"(≈ **{coin_value:,.0f} KRW**, 매도 시 수수료 ~{est_sell_fee:,.0f})"
                )
            else:
                st.caption(f"{mt_coin} 보유 없음")

            sell_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_sell_type")

            if sell_type == "시장가":
                def _sync_sell_slider(): st.session_state["mt_sell_pct"] = st.session_state["mt_sell_pct_slider"]
                def _sync_sell_input(): st.session_state["mt_sell_pct_slider"] = st.session_state["mt_sell_pct"]
                _sp1, _sp2 = st.columns([1, 2])
                sell_pct = _sp1.number_input("매도 비율 (%)", min_value=1, max_value=100, value=100, step=1,
                                             key="mt_sell_pct", on_change=_sync_sell_input)
                _sp2.slider("비율 조절", 1, 100, value=100, step=1, key="mt_sell_pct_slider",
                            label_visibility="collapsed", on_change=_sync_sell_slider)
                est_sell_qty = coin_holding * sell_pct / 100
                est_sell_val = est_sell_qty * cur_price if cur_price > 0 else 0
                est_sell_fee_amt = est_sell_val * ask_fee
                st.caption(
                    f"보유 {mt_coin}의 {sell_pct}% = "
                    f"**{est_sell_qty:.8g} {mt_coin}** (≈ {est_sell_val:,.0f} KRW, "
                    f"수수료 {est_sell_fee_amt:,.0f} → 수령 {est_sell_val - est_sell_fee_amt:,.0f})"
                )

                if st.button(
                    f"{mt_coin} {sell_pct}% 시장가 매도",
                    key="mt_sell_vm", type="primary", use_container_width=True,
                ):
                    _sell_status = st.empty()
                    _sell_status.info("3분할 체결 진행 중 (~3분 소요)...")
                    _sell_json = json.dumps({"coin": mt_coin, "side": "sell", "pct": sell_pct})
                    _order_and_sync(_sell_json, _sell_status, "시장가 매도")

            else:  # 지정가
                def _on_sell_vol():
                    _p = st.session_state.get("mt_sell_price", 0)
                    _v = st.session_state.get("mt_sell_vol", 0)
                    if _p > 0 and _v > 0:
                        st.session_state["mt_sell_amount"] = int(_p * _v)

                def _on_sell_amount():
                    _p = st.session_state.get("mt_sell_price", 0)
                    _a = st.session_state.get("mt_sell_amount", 0)
                    if _p > 0 and _a > 0:
                        st.session_state["mt_sell_vol"] = _a / _p

                # 비율 슬라이더 → 수량 자동 계산
                if coin_holding > 0:
                    def _on_sell_pct_limit():
                        _pct_val = st.session_state.get("mt_sell_pct_limit", 100)
                        _fill_qty = coin_holding * _pct_val / 100
                        st.session_state["mt_sell_vol"] = _fill_qty
                        _sp = st.session_state.get("mt_sell_price", cur_price)
                        if _sp > 0:
                            st.session_state["mt_sell_amount"] = int(_sp * _fill_qty)
                    st.slider("보유량 비율 (%)", 1, 100, value=100, step=1,
                              key="mt_sell_pct_limit", on_change=_on_sell_pct_limit)

                _sc1, _sc2 = st.columns(2)
                _default_sell = _round_to_tick(cur_price * 1.01) if cur_price > 0 else 1
                sell_price = _sc1.number_input(
                    "매도 가격 (KRW)", min_value=1,
                    value=_default_sell,
                    step=max(1, int(_get_tick_size(cur_price))),
                    key="mt_sell_price",
                )
                # 호가 단위 보정 안내
                tick_adj_s = _round_to_tick(sell_price)
                if tick_adj_s != sell_price:
                    _sc1.caption(f"호가 보정: {tick_adj_s:,}")

                sell_vol = _sc2.number_input(
                    f"매도 수량 ({mt_coin})", min_value=0.00000001, value=0.001,
                    format="%.8f", key="mt_sell_vol", on_change=_on_sell_vol,
                )
                _sell_amount = st.number_input(
                    "매도 금액 (KRW)", min_value=0, value=int(sell_price * sell_vol),
                    step=10000, key="mt_sell_amount", on_change=_on_sell_amount,
                )
                sell_total = sell_price * sell_vol
                sell_fee_amt = sell_total * ask_fee
                st.caption(f"총액: **{sell_total:,.0f} KRW** (수수료 {sell_fee_amt:,.0f} → 수령 {sell_total - sell_fee_amt:,.0f})")

                if st.button(
                    f"{mt_coin} 지정가 매도 ({sell_price:,.0f} × {sell_vol:.8g})",
                    key="mt_lsell_vm", type="primary", use_container_width=True,
                ):
                    final_sell_price = _round_to_tick(sell_price)
                    final_sell_total = final_sell_price * sell_vol
                    if final_sell_total < 5000:
                        st.error("최소 주문금액: 5,000 KRW")
                    else:
                        _sell_json = json.dumps({
                            "coin": mt_coin, "side": "sell",
                            "order_type": "limit", "price": final_sell_price, "volume": sell_vol,
                        })
                        _order_and_sync(_sell_json, st.empty(), "지정가 매도")

    # ── 미체결 주문 + 체결 내역 ──
    st.divider()
    pend_col, done_col = st.columns(2)

    with pend_col:
        st.markdown("**미체결 주문**")
        if st.button("조회/동기화", key="mt_pending_btn", use_container_width=True):
            _pend_status = st.empty()
            _pend_status.info("미체결 주문 조회 중...")
            ok, msg = trigger_and_wait_gh("account_sync", _pend_status)
            if ok:
                _pend_status.success("잔고 동기화 완료")
            else:
                _pend_status.warning(f"동기화 실패: {msg}")

        _acct2 = load_account_cache()
        pending_orders = _acct2.get("pending_orders", [])
        if pending_orders:
            for order in pending_orders:
                side_kr = "매수" if order.get('side') == 'bid' else "매도"
                side_color = "red" if order.get('side') == 'bid' else "blue"
                market = order.get('market', '')
                price = float(order.get('price', 0) or 0)
                remaining = float(order.get('remaining_volume', 0) or 0)
                created = order.get('created_at', '')
                if created:
                    try:
                        created = pd.to_datetime(created).strftime('%m/%d %H:%M')
                    except Exception:
                        pass
                st.markdown(
                    f"**:{side_color}[{side_kr}]** {market}\n\n"
                    f"{price:,.0f} × {remaining:.8f} | {created}"
                )
        else:
            st.caption("미체결 주문이 없습니다.")

    with done_col:
        st.markdown("**최근 체결 내역**")
        # account_cache에서 자동 로드 (잔고 동기화 시 함께 업데이트)
        _acct3 = load_account_cache()
        cached_orders = _acct3.get("orders", [])
        # 선택된 코인 필터링
        done_list = [
            o for o in cached_orders
            if mt_ticker in str(o.get('market', '')) or mt_coin in str(o.get('currency', ''))
        ][:10]

        if _acct3.get("updated_at"):
            st.caption(f"동기화: {_acct3['updated_at']}")

        if done_list:
            for order in done_list:
                # account_cache의 orders 형식 (get_history 결과)
                side_raw = order.get('side', order.get('type', ''))
                if side_raw in ('bid', 'buy'):
                    side_kr, side_color = "매수", "red"
                elif side_raw in ('ask', 'sell'):
                    side_kr, side_color = "매도", "blue"
                else:
                    side_kr, side_color = side_raw, "gray"
                price = float(order.get('price', 0) or order.get('avg_price', 0) or 0)
                vol = float(order.get('executed_volume', 0) or order.get('volume', 0) or 0)
                fee = float(order.get('paid_fee', 0) or 0)
                created = order.get('created_at', order.get('done_at', ''))
                if created:
                    try:
                        created = pd.to_datetime(created).strftime('%m/%d %H:%M')
                    except Exception:
                        pass
                total_krw = price * vol if price > 0 else 0
                st.markdown(
                    f"**:{side_color}[{side_kr}]** {price:,.0f} × {vol:.8g}\n\n"
                    f"금액 {total_krw:,.0f} | 수수료 {fee:,.0f} | {created}"
                )
        else:
            st.caption("체결 내역이 없습니다.")
