"""개별 자산 백테스트 탭."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.constants import TOP_20_TICKERS, INTERVAL_MAP, INTERVAL_REV_MAP, CANDLES_PER_DAY
import src.engine.data_cache as data_cache
from src.ui.components.performance import _render_performance_analysis, _apply_dd_hover_format

_DEFAULT_SLIPPAGE = {
    "major": {"day": 0.03, "minute240": 0.05, "minute60": 0.08, "minute30": 0.08, "minute15": 0.10, "minute5": 0.15, "minute1": 0.20},
    "mid":   {"day": 0.05, "minute240": 0.08, "minute60": 0.10, "minute30": 0.10, "minute15": 0.15, "minute5": 0.20, "minute1": 0.30},
    "alt":   {"day": 0.10, "minute240": 0.15, "minute60": 0.20, "minute30": 0.20, "minute15": 0.25, "minute5": 0.35, "minute1": 0.50},
}
_MAJOR_COINS = {"BTC", "ETH"}
_MID_COINS = {"XRP", "SOL", "DOGE", "ADA", "TRX", "AVAX", "LINK", "BCH", "DOT", "ETC"}


def _get_default_slippage(ticker, interval):
    coin = ticker.split("-")[-1].upper() if "-" in ticker else ticker.upper()
    if coin in _MAJOR_COINS:
        tier = "major"
    elif coin in _MID_COINS:
        tier = "mid"
    else:
        tier = "alt"
    return _DEFAULT_SLIPPAGE[tier].get(interval, 0.10)


def render_backtest_tab(portfolio_list, initial_cap, backtest_engine):
    """개별 자산 백테스트 서브탭."""
    st.header("개별 자산 백테스트")

    port_tickers_bt = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
    base_options_bt = list(dict.fromkeys(port_tickers_bt + TOP_20_TICKERS))

    selected_ticker_bt = st.selectbox("백테스트 대상", base_options_bt + ["직접입력"], key="bt_target")

    bt_ticker = ""
    bt_sma = 0
    bt_buy_period = 20
    bt_sell_period = 10

    if selected_ticker_bt == "직접입력":
        bt_custom = st.text_input("코인", "BTC", key="bt_c")
        bt_ticker = f"KRW-{bt_custom.upper()}"
    else:
        bt_ticker = selected_ticker_bt

    port_match = next((item for item in portfolio_list if f"{item['market']}-{item['coin'].upper()}" == bt_ticker), None)

    default_strat_idx = 0
    if port_match and port_match.get('strategy') == 'Donchian':
        default_strat_idx = 1

    bt_strategy = st.selectbox("전략 선택", ["SMA 전략", "돈키안 전략"], index=default_strat_idx, key="bt_strategy_sel")

    # 기본값 초기화
    bt_sma = 60
    bt_buy_period = 20
    bt_sell_period = 10
    bt_sell_mode = "하단선 (Lower)"

    if bt_strategy == "SMA 전략":
        default_sma = port_match.get('parameter', 60) if port_match else 60
        bt_sma = st.number_input("단기 SMA (추세)", value=default_sma, key="bt_sma_select", min_value=5, step=1)
    else:
        default_buy = int(port_match.get('parameter', 20)) if port_match and port_match.get('strategy') == 'Donchian' else 20
        default_sell = int(port_match.get('sell_parameter', 10)) if port_match and port_match.get('strategy') == 'Donchian' else 10
        if default_sell == 0:
            default_sell = max(5, default_buy // 2)
        dc_col1, dc_col2 = st.columns(2)
        with dc_col1:
            bt_buy_period = st.number_input("매수 채널 기간", value=default_buy, min_value=5, max_value=300, step=1, key="bt_dc_buy")
        with dc_col2:
            bt_sell_period = st.number_input("매도 채널 기간", value=default_sell, min_value=5, max_value=300, step=1, key="bt_dc_sell")

        st.divider()
        st.caption("📌 매도 기준 선택")
        bt_sell_mode = st.radio(
            "매도 라인",
            ["하단선 (Lower)", "중심선 (Midline)", "두 방법 비교"],
            horizontal=True,
            key="bt_sell_mode",
            help="하단선: 저가 채널 이탈 시 매도 / 중심선: (상단+하단)/2 이탈 시 매도"
        )

    default_interval_idx = 0
    if port_match:
        port_iv_label = INTERVAL_REV_MAP.get(port_match.get('interval', 'day'), '일봉')
        interval_keys = list(INTERVAL_MAP.keys())
        if port_iv_label in interval_keys:
            default_interval_idx = interval_keys.index(port_iv_label)

    bt_interval_label = st.selectbox("시간봉 선택", options=list(INTERVAL_MAP.keys()), index=default_interval_idx, key="bt_interval_sel")
    bt_interval = INTERVAL_MAP[bt_interval_label]

    default_slip = _get_default_slippage(bt_ticker, bt_interval)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.caption("백테스트 기간")
        d_col1, d_col2 = st.columns(2)
        try:
            default_start_bt = datetime(2020, 1, 1).date()
        except:
            default_start_bt = datetime.now().date() - timedelta(days=365)
        default_end_bt = datetime.now().date()

        bt_start = d_col1.date_input("시작일", value=default_start_bt, max_value=datetime.now().date(), format="YYYY.MM.DD", key="bt_start")
        bt_end = d_col2.date_input("종료일", value=default_end_bt, max_value=datetime.now().date(), format="YYYY.MM.DD", key="bt_end")

        if bt_start > bt_end:
            st.error("시작일은 종료일보다 빨라야 합니다.")
            bt_end = bt_start

        days_diff = (bt_end - bt_start).days
        st.caption(f"기간: {days_diff}일")

        fee = st.number_input("매매 수수료 (%)", value=0.05, format="%.2f", key="bt_fee") / 100
        bt_slippage = st.number_input("슬리피지 (%)", value=default_slip, min_value=0.0, max_value=2.0, step=0.01, format="%.2f", key="bt_slip")

        fee_pct = fee * 100
        cost_per_trade = fee_pct + bt_slippage
        cost_round_trip = (fee_pct * 2) + (bt_slippage * 2)
        st.caption(f"편도: {cost_per_trade:.2f}% | 왕복: {cost_round_trip:.2f}%")

        run_btn = st.button("백테스트 실행", type="primary", key="bt_run")

    if run_btn:
        if bt_strategy == "돈키안 전략":
            req_period = max(bt_buy_period, bt_sell_period)
            bt_strategy_mode = "Donchian"
            bt_sell_ratio = bt_sell_period / bt_buy_period if bt_buy_period > 0 else 0.5
            _smode_raw = bt_sell_mode if bt_strategy == "돈키안 전략" else "하단선 (Lower)"
            _compare_mode = _smode_raw == "두 방법 비교"
            _sell_mode_api = "midline" if _smode_raw == "중심선 (Midline)" else "lower"
        else:
            req_period = bt_sma
            bt_strategy_mode = "SMA 전략"
            bt_sell_ratio = 0.5
            _compare_mode = False
            _sell_mode_api = "lower"

        to_date = bt_end + timedelta(days=1)
        to_str = to_date.strftime("%Y-%m-%d 09:00:00")
        cpd = CANDLES_PER_DAY.get(bt_interval, 1)
        req_count = days_diff * cpd + req_period + 300
        fetch_count = max(req_count, req_period + 300)

        with st.spinner(f"백테스트 실행 중 ({bt_start} ~ {bt_end}, {bt_interval_label}, {bt_strategy})..."):
            df_bt = data_cache.get_ohlcv_local_first(
                bt_ticker,
                interval=bt_interval,
                to=to_str,
                count=fetch_count,
                allow_api_fallback=True,
            )
            if df_bt is None or df_bt.empty:
                st.error("데이터를 가져올 수 없습니다.")
                st.stop()

            st.caption(f"조회된 캔들: {len(df_bt)}개 ({df_bt.index[0].strftime('%Y-%m-%d')} ~ {df_bt.index[-1].strftime('%Y-%m-%d')})")

            result = backtest_engine.run_backtest(
                bt_ticker, period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                interval=bt_interval, count=fetch_count, fee=fee,
                start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=bt_slippage,
                sell_mode="lower" if _compare_mode else _sell_mode_api
            )
            # 비교 모드: 중심선 결과도 실행
            if _compare_mode and bt_strategy_mode == "Donchian":
                result_mid = backtest_engine.run_backtest(
                    bt_ticker, period=bt_buy_period,
                    interval=bt_interval, count=fetch_count, fee=fee,
                    start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                    strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=bt_slippage,
                    sell_mode="midline"
                )
            else:
                result_mid = None

            if "error" in result:
                st.error(result["error"])
            else:
                df = result["df"]
                res = result["performance"]

                # ── 비교 요약 테이블 (비교모드) ──
                if _compare_mode and result_mid and "error" not in result_mid:
                    res_mid = result_mid["performance"]
                    st.subheader("📊 하단선 vs 중심선 비교")
                    cmp_data = {
                        "항목": ["총 수익률", "CAGR", "MDD", "샤프비율", "승률", "거래 횟수", "최종 자산"],
                        f"하단선 Lower({bt_sell_period})": [
                            f"{res['total_return']:,.2f}%",
                            f"{res.get('cagr', 0):,.2f}%",
                            f"{res['mdd']:,.2f}%",
                            f"{res.get('sharpe', 0):.2f}",
                            f"{res['win_rate']:,.2f}%",
                            f"{res['trade_count']}회",
                            f"{res['final_equity']:,.0f} KRW",
                        ],
                        f"중심선 Midline": [
                            f"{res_mid['total_return']:,.2f}%",
                            f"{res_mid.get('cagr', 0):,.2f}%",
                            f"{res_mid['mdd']:,.2f}%",
                            f"{res_mid.get('sharpe', 0):.2f}",
                            f"{res_mid['win_rate']:,.2f}%",
                            f"{res_mid['trade_count']}회",
                            f"{res_mid['final_equity']:,.0f} KRW",
                        ],
                    }
                    st.dataframe(pd.DataFrame(cmp_data).set_index("항목"), use_container_width=True)

                    if res['total_return'] > res_mid['total_return']:
                        st.success(f"✅ 하단선(Lower) 방식이 수익률 {res['total_return']:.2f}% vs {res_mid['total_return']:.2f}% 로 우수")
                    elif res_mid['total_return'] > res['total_return']:
                        st.success(f"✅ 중심선(Midline) 방식이 수익률 {res_mid['total_return']:.2f}% vs {res['total_return']:.2f}% 로 우수")
                    else:
                        st.info("두 방식의 수익률이 동일합니다.")
                    st.divider()

                sell_mode_label = "중심선(Midline)" if _sell_mode_api == "midline" and not _compare_mode else ("하단선(Lower)" if not _compare_mode else "하단선(Lower) [기준]")
                if not _compare_mode:
                    st.caption(f"매도방식: **{sell_mode_label}**")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("총 수익률", f"{res['total_return']:,.2f}%")
                m2.metric("연평균(CAGR)", f"{res.get('cagr', 0):,.2f}%")
                m3.metric("승률", f"{res['win_rate']:,.2f}%")
                m4.metric("최대낙폭(MDD)", f"{res['mdd']:,.2f}%")
                m5.metric("샤프비율", f"{res['sharpe']:.2f}")

                trade_count = res['trade_count']
                total_cost_pct = cost_round_trip * trade_count
                st.success(
                    f"최종 잔고: **{res['final_equity']:,.0f} KRW** (초기 {initial_cap:,.0f} KRW) | "
                    f"거래 {trade_count}회 | 왕복비용 {cost_round_trip:.2f}% | 누적 약 {total_cost_pct:.1f}%"
                )

                if bt_slippage > 0:
                    result_no_slip = backtest_engine.run_backtest(
                        bt_ticker, period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                        interval=bt_interval, count=fetch_count, fee=fee,
                        start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                        strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=0.0
                    )
                    if "error" not in result_no_slip:
                        res_ns = result_no_slip['performance']
                        slip_ret_diff = res_ns['total_return'] - res['total_return']
                        slip_cost = res_ns['final_equity'] - res['final_equity']
                        st.info(f"슬리피지 영향: 수익률 차이 **{slip_ret_diff:,.2f}%p**, 금액 차이 **{slip_cost:,.0f} KRW**")

                st.subheader("가격 & 전략 성과")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

                fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name='가격'), row=1, col=1, secondary_y=False)

                if bt_strategy_mode == "Donchian":
                    upper_col = f'Donchian_Upper_{bt_buy_period}'
                    lower_col = f'Donchian_Lower_{bt_sell_period}'
                    if upper_col in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df[upper_col], line=dict(color='green', width=1.5, dash='dash'), name=f'상단 ({bt_buy_period})'), row=1, col=1, secondary_y=False)
                    if lower_col in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df[lower_col], line=dict(color='red', width=1.5, dash='dash'), name=f'하단 ({bt_sell_period})'), row=1, col=1, secondary_y=False)
                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{bt_sma}'], line=dict(color='orange', width=2), name=f'SMA {bt_sma}'), row=1, col=1, secondary_y=False)

                fig.add_trace(go.Scatter(x=df.index, y=df['equity'], line=dict(color='blue', width=2), name='전략 자산'), row=1, col=1, secondary_y=True)

                buy_dates = [t['date'] for t in res['trades'] if t['type'] == 'buy']
                buy_prices = [t['price'] for t in res['trades'] if t['type'] == 'buy']
                sell_dates = [t['date'] for t in res['trades'] if t['type'] == 'sell']
                sell_prices = [t['price'] for t in res['trades'] if t['type'] == 'sell']
                if buy_dates:
                    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='매수'), row=1, col=1, secondary_y=False)
                if sell_dates:
                    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='매도'), row=1, col=1, secondary_y=False)

                fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], name='낙폭 (%)', fill='tozeroy', line=dict(color='red', width=1)), row=2, col=1)
                fig.update_layout(height=800, title_text="백테스트 결과", xaxis_rangeslider_visible=False, margin=dict(t=80),
                    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                fig.update_yaxes(title_text="가격 (KRW)", row=1, col=1, secondary_y=False)
                fig.update_yaxes(title_text="자산 (KRW)", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
                fig = _apply_dd_hover_format(fig)
                st.plotly_chart(fig, use_container_width=True)

                if 'yearly_stats' in res:
                    st.subheader("연도별 성과")
                    st.dataframe(res['yearly_stats'].style.format("{:.2f}%"))

                _render_performance_analysis(
                    equity_series=df.get("equity"),
                    benchmark_series=df.get("close"),
                    strategy_metrics=res,
                    strategy_label="백테스트 전략",
                    benchmark_label=f"{bt_ticker} 단순보유",
                )

                st.info(f"전략 상태: **{res['final_status']}** | 다음 행동: **{res['next_action'] if res['next_action'] else '없음'}**")

                with st.expander("거래 내역"):
                    if res['trades']:
                        trades_df = pd.DataFrame(res['trades'])
                        st.dataframe(trades_df.style.format({"price": "{:,.2f}", "amount": "{:,.6f}", "balance": "{:,.2f}", "profit": "{:,.2f}%"}))
                    else:
                        st.info("실행된 거래가 없습니다.")

                csv_data = df.to_csv(index=True).encode('utf-8')
                st.download_button(label="일별 로그 다운로드", data=csv_data, file_name=f"{bt_ticker}_{bt_start}_daily_log.csv", mime="text/csv")
