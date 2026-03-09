"""연금저축 백테스트 탭."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import src.engine.data_cache as data_cache
from src.utils.formatting import _code_only, _fmt_etf_code_name
from src.ui.components.performance import _render_performance_analysis


def _normalize_numeric_series(series_obj, preferred_cols=("equity", "close", "Close")) -> pd.Series:
    """Series/DataFrame/array를 숫자 Series로 정규화."""
    if series_obj is None:
        return pd.Series(dtype=float)
    if isinstance(series_obj, pd.DataFrame):
        if series_obj.empty:
            return pd.Series(dtype=float)
        pick_col = None
        for col in preferred_cols:
            if col in series_obj.columns:
                pick_col = col
                break
        s = series_obj[pick_col] if pick_col else series_obj.iloc[:, 0]
    elif isinstance(series_obj, pd.Series):
        s = series_obj.copy()
    else:
        try:
            s = pd.Series(series_obj)
        except Exception:
            return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _render_trade_history_table(bt_result: dict, key_prefix: str = ""):
    """백테스트 매매 내역 테이블."""
    if not isinstance(bt_result, dict):
        return
    trades_df = bt_result.get("trades")
    if not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
        return
    st.subheader("백테스트 매매 내역")
    display = trades_df.copy()
    col_map = {"date": "날짜", "action": "매매", "ticker": "종목",
               "price": "가격", "shares": "수량", "equity": "자산"}
    display.rename(columns=col_map, inplace=True)
    if "날짜" in display.columns:
        display["날짜"] = pd.to_datetime(display["날짜"]).dt.strftime("%Y-%m-%d")
    if "가격" in display.columns:
        display["가격"] = display["가격"].apply(lambda x: f"{x:,.2f}")
    if "수량" in display.columns:
        display["수량"] = display["수량"].apply(lambda x: f"{x:,.4f}")
    if "자산" in display.columns:
        display["자산"] = display["자산"].apply(lambda x: f"{x:,.0f}")
    cols = [c for c in ["날짜", "매매", "종목", "가격", "수량", "자산"] if c in display.columns]
    st.dataframe(display[cols], use_container_width=True, hide_index=True,
                 key=f"trades_table_{key_prefix}" if key_prefix else None)
    _buy_cnt = int((trades_df["action"] == "매수").sum()) if "action" in trades_df.columns else 0
    _sell_cnt = int((trades_df["action"] == "매도").sum()) if "action" in trades_df.columns else 0
    st.caption(f"총 {len(trades_df)}건 (매수 {_buy_cnt} / 매도 {_sell_cnt})")


def _render_bt_metrics(metrics: dict):
    """백테스트 지표 표시."""
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("총 수익률", f"{metrics['total_return']:.2f}%")
    mc2.metric("CAGR", f"{metrics['cagr']:.2f}%")
    mc3.metric("MDD", f"{metrics['mdd']:.2f}%")
    mc4.metric("샤프", f"{metrics['sharpe']:.2f}")
    mc5.metric("최종 자산", f"{metrics['final_equity']:,.0f}")


def render_pension_backtest_tab(
    active_strategies: list[str],
    pen_bt_start_ts,
    pen_bt_cap: int,
    kr_etf_map: dict,
    kr_spy: str, kr_iwd: str, kr_gld: str,
    kr_ief: str, kr_qqq: str, kr_shy: str,
    dm_settings: dict | None,
    vaa_settings: dict | None,
    cdm_settings: dict | None,
    get_pen_daily_chart,
    pen_local_first: bool,
):
    """백테스트 탭 렌더링."""
    from src.strategy.laa import LAAStrategy
    from src.strategy.dual_momentum import DualMomentumStrategy

    if pen_local_first:
        st.info("백테스트 가격 데이터: 로컬 파일(cache/data) 우선, 부족 시 API 보강 모드입니다.")

    _bt_candidates = [s for s in ["LAA", "듀얼모멘텀", "VAA", "CDM"] if s in active_strategies]
    for _s in ["듀얼모멘텀", "VAA", "CDM"]:
        if _s not in _bt_candidates:
            _bt_candidates.append(_s)

    if not _bt_candidates:
        st.warning("포트폴리오에 활성화된 전략이 없어 백테스트를 실행할 수 없습니다.")
        return

    _bt_strategy = st.selectbox("백테스트 전략", _bt_candidates, key="pen_bt_strategy_select")
    pen_bt_start = st.date_input("시작일", value=pen_bt_start_ts.date(), key="pen_bt_start_date")
    pen_bt_cap_input = st.number_input("초기 자본 (KRW)", value=int(pen_bt_cap), step=1_000_000, key="pen_bt_cap")
    pen_bt_fee = st.number_input("수수료 (%)", value=0.02, format="%.2f", key="pen_bt_fee") / 100.0
    _filter_ts = pd.Timestamp(pen_bt_start).normalize()

    if _bt_strategy == "LAA":
        _run_laa_backtest(LAAStrategy, kr_etf_map, pen_bt_cap_input, pen_bt_fee, _filter_ts)
    elif _bt_strategy == "듀얼모멘텀":
        _run_dm_backtest(DualMomentumStrategy, dm_settings, pen_bt_cap_input,
                         pen_bt_fee, _filter_ts, get_pen_daily_chart)
    elif _bt_strategy == "VAA":
        _run_vaa_backtest(vaa_settings, pen_bt_cap_input, pen_bt_fee, _filter_ts)
    elif _bt_strategy == "CDM":
        _run_cdm_backtest(cdm_settings, pen_bt_cap_input, pen_bt_fee, _filter_ts)
    else:
        st.info("선택된 전략의 백테스트를 준비 중입니다.")


def _run_laa_backtest(LAAStrategy, kr_etf_map, cap, fee, filter_ts):
    """LAA 백테스트."""
    st.header("LAA 백테스트")
    st.caption("국내 ETF 매핑(SPY/IWD/GLD/IEF/QQQ/SHY) 기반 월간 리밸런싱 시뮬레이션")

    if st.button("LAA 백테스트 실행", key="pen_bt_run_laa", type="primary"):
        with st.spinner("LAA 백테스트 실행 중... (국내 데이터 조회)"):
            tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
            price_data = {}
            for ticker in tickers:
                df_t = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
                if df_t is None or df_t.empty:
                    st.error(f"{ticker} yfinance 데이터 조회 실패")
                    price_data = None
                    break
                df_t = df_t.copy().sort_index()
                df_t = df_t[df_t.index >= filter_ts]
                if df_t.empty:
                    st.error(f"{ticker} 시작일 이후 데이터가 없습니다. 시작일을 조정해 주세요.")
                    price_data = None
                    break
                price_data[ticker] = df_t

            if price_data:
                strategy = LAAStrategy(settings={"kr_etf_map": kr_etf_map})
                bt_result = strategy.run_backtest(price_data, initial_balance=float(cap), fee=float(fee))
                if bt_result:
                    st.session_state["pen_bt_laa_result"] = bt_result
                    st.session_state["pen_bt_result"] = bt_result
                    _bm_series = _normalize_numeric_series(price_data.get("SPY"), preferred_cols=("close", "Close"))
                    if not _bm_series.empty:
                        st.session_state["pen_bt_laa_benchmark_series"] = _bm_series
                        st.session_state["pen_bt_laa_benchmark_label"] = "SPY Buy & Hold"
                else:
                    st.error("백테스트 실행 실패 (데이터 부족)")

    bt_res = st.session_state.get("pen_bt_laa_result") or st.session_state.get("pen_bt_result")
    if bt_res:
        metrics = bt_res["metrics"]
        _render_bt_metrics(metrics)

        eq_df = bt_res["equity_df"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq_df.index, y=eq_df["equity"], name="포트폴리오", line=dict(color="royalblue")))
        fig.update_layout(title="LAA 백테스트 자산 곡선", xaxis_title="날짜", yaxis_title="자산 (KRW)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        if "equity" in eq_df.columns:
            yearly = eq_df["equity"].resample("YE").last()
            if len(yearly) > 1:
                yr_ret = yearly.pct_change().dropna() * 100
                yr_data = [{"연도": str(d.year), "수익률(%)": f"{r:.2f}"} for d, r in yr_ret.items()]
                st.subheader("연도별 수익률")
                st.dataframe(pd.DataFrame(yr_data), use_container_width=True, hide_index=True)

        if "allocations" in bt_res:
            alloc_df = bt_res["allocations"]
            if not alloc_df.empty:
                st.subheader("월별 자산 배분 이력")
                st.dataframe(alloc_df.tail(24), use_container_width=True, hide_index=True)

        _render_trade_history_table(bt_res, key_prefix="bt_laa")

        _bm_series = st.session_state.get("pen_bt_laa_benchmark_series")
        _bm_label = str(st.session_state.get("pen_bt_laa_benchmark_label", "SPY Buy & Hold"))
        _render_performance_analysis(
            equity_series=eq_df.get("equity"),
            benchmark_series=_bm_series,
            strategy_metrics=metrics,
            strategy_label="LAA 전략",
            benchmark_label=_bm_label,
            show_drawdown=True, show_weight=True, equity_df=eq_df,
        )


def _run_dm_backtest(DualMomentumStrategy, dm_settings, cap, fee, filter_ts, get_daily_chart):
    """듀얼모멘텀 백테스트."""
    st.header("듀얼모멘텀 백테스트")
    st.caption("사이드바의 국내 ETF 설정(공격 2종/방어/카나리아) 기반 월간 리밸런싱 시뮬레이션")

    if not dm_settings:
        st.warning("사이드바에서 듀얼모멘텀 설정을 먼저 입력해 주세요.")
        return

    if st.button("듀얼모멘텀 백테스트 실행", key="pen_bt_run_dm", type="primary"):
        with st.spinner("듀얼모멘텀 백테스트 실행 중... (국내 데이터 조회)"):
            dm_tickers = []
            for tk in (dm_settings.get("offensive", []) + dm_settings.get("defensive", []) + dm_settings.get("canary", [])):
                tku = str(tk).strip().upper()
                if tku and tku not in dm_tickers:
                    dm_tickers.append(tku)

            dm_price_data = {}
            dm_kr_map = dm_settings.get("kr_etf_map", {}) or {}
            for ticker in dm_tickers:
                kr_code = str(dm_kr_map.get(ticker, "")).strip()
                if not kr_code:
                    st.error(f"{ticker} 국내 ETF 매핑이 없습니다.")
                    dm_price_data = None
                    break
                df_t = get_daily_chart(kr_code, count=3000, use_disk_cache=True)
                if df_t is None or df_t.empty:
                    st.error(f"{ticker} ({kr_code}) 로컬 데이터가 없습니다.")
                    dm_price_data = None
                    break
                df_t = df_t.copy().sort_index()
                if "close" not in df_t.columns and "Close" in df_t.columns:
                    df_t["close"] = df_t["Close"]
                if "close" not in df_t.columns:
                    st.error(f"{ticker} ({kr_code}) 종가 컬럼이 없습니다.")
                    dm_price_data = None
                    break
                df_t = df_t[df_t.index >= filter_ts]
                if df_t.empty:
                    st.error(f"{ticker} ({kr_code}) 시작일 이후 데이터가 없습니다.")
                    dm_price_data = None
                    break
                dm_price_data[ticker] = df_t

            if dm_price_data:
                dm_strategy = DualMomentumStrategy(settings=dm_settings)
                dm_bt_result = dm_strategy.run_backtest(dm_price_data, initial_balance=float(cap), fee=float(fee))
                if dm_bt_result:
                    st.session_state["pen_bt_dm_result"] = dm_bt_result
                    _bm_ticker = ""
                    for _t in (dm_settings.get("offensive", []) or []):
                        if _t in dm_price_data:
                            _bm_ticker = str(_t)
                            break
                    if not _bm_ticker:
                        for _t in dm_tickers:
                            if _t in dm_price_data:
                                _bm_ticker = str(_t)
                                break
                    _bm_series = _normalize_numeric_series(dm_price_data.get(_bm_ticker), preferred_cols=("close", "Close"))
                    if not _bm_series.empty:
                        st.session_state["pen_bt_dm_benchmark_series"] = _bm_series
                        _bm_code = _code_only(str(dm_kr_map.get(_bm_ticker, "")))
                        if _bm_code:
                            st.session_state["pen_bt_dm_benchmark_label"] = f"{_fmt_etf_code_name(_bm_code)} Buy & Hold"
                        else:
                            st.session_state["pen_bt_dm_benchmark_label"] = f"{_bm_ticker} Buy & Hold"
                else:
                    st.error("백테스트 실행 실패 (데이터 부족)")

    dm_res = st.session_state.get("pen_bt_dm_result")
    if dm_res:
        _dm_bt_eq = dm_res.get("equity_df")
        metrics = dm_res["metrics"]
        _render_bt_metrics(metrics)

        _bm_series = st.session_state.get("pen_bt_dm_benchmark_series")
        _bm_label = str(st.session_state.get("pen_bt_dm_benchmark_label", "SPY Buy & Hold"))
        if isinstance(_dm_bt_eq, pd.DataFrame) and not _dm_bt_eq.empty:
            _render_performance_analysis(
                equity_series=_dm_bt_eq.get("equity"),
                benchmark_series=_bm_series,
                strategy_metrics=metrics,
                strategy_label="듀얼모멘텀 전략",
                benchmark_label=_bm_label,
                show_drawdown=True, show_weight=True, equity_df=_dm_bt_eq,
            )
        _render_trade_history_table(dm_res, key_prefix="bt_dm")


def _run_vaa_backtest(vaa_settings, cap, fee, filter_ts):
    """VAA 백테스트."""
    st.header("VAA 백테스트")
    st.caption("13612W 모멘텀 기반 공격/방어 전환, yfinance 미국 원본 데이터 사용")

    if st.button("VAA 백테스트 실행", key="pen_bt_run_vaa", type="primary"):
        with st.spinner("VAA 백테스트 실행 중..."):
            vaa_tickers = list(set(vaa_settings.get("offensive", []) + vaa_settings.get("defensive", [])))
            vaa_price_data = {}
            for ticker in vaa_tickers:
                df_t = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
                if df_t is None or df_t.empty:
                    st.error(f"{ticker} yfinance 데이터 조회 실패")
                    vaa_price_data = None
                    break
                df_t = df_t.copy().sort_index()
                df_t = df_t[df_t.index >= filter_ts]
                if df_t.empty:
                    st.error(f"{ticker} 시작일 이후 데이터가 없습니다.")
                    vaa_price_data = None
                    break
                vaa_price_data[ticker] = df_t

            if vaa_price_data:
                from src.strategy.vaa import VAAStrategy
                vaa_strategy = VAAStrategy(settings=vaa_settings)
                vaa_bt_result = vaa_strategy.run_backtest(vaa_price_data, initial_balance=float(cap), fee=float(fee))
                if vaa_bt_result:
                    st.session_state["pen_bt_vaa_result"] = vaa_bt_result
                    if "SPY" in vaa_price_data:
                        st.session_state["pen_bt_vaa_benchmark_series"] = _normalize_numeric_series(
                            vaa_price_data.get("SPY"), preferred_cols=("close", "Close"))
                        st.session_state["pen_bt_vaa_benchmark_label"] = "SPY Buy & Hold"
                else:
                    st.error("VAA 백테스트 실행 실패 (데이터 부족)")

    vaa_bt_res = st.session_state.get("pen_bt_vaa_result")
    if vaa_bt_res:
        _eq = vaa_bt_res.get("equity_df")
        metrics = vaa_bt_res["metrics"]
        _render_bt_metrics(metrics)

        _bm_series = st.session_state.get("pen_bt_vaa_benchmark_series")
        _bm_label = str(st.session_state.get("pen_bt_vaa_benchmark_label", "SPY Buy & Hold"))
        if isinstance(_eq, pd.DataFrame) and not _eq.empty:
            _render_performance_analysis(
                equity_series=_eq.get("equity"),
                benchmark_series=_bm_series,
                strategy_metrics=metrics,
                strategy_label="VAA 전략",
                benchmark_label=_bm_label,
                show_drawdown=True, show_weight=True, equity_df=_eq,
            )

        pos_df = vaa_bt_res.get("positions")
        if isinstance(pos_df, pd.DataFrame) and not pos_df.empty:
            st.subheader("월별 포지션 이력")
            st.dataframe(pos_df.tail(36), use_container_width=True, hide_index=True)

        _render_trade_history_table(vaa_bt_res, key_prefix="bt_vaa")


def _run_cdm_backtest(cdm_settings, cap, fee, filter_ts):
    """CDM 백테스트."""
    st.header("CDM 백테스트")
    st.caption("4모듈 듀얼모멘텀, yfinance 미국 원본 데이터 사용")

    if st.button("CDM 백테스트 실행", key="pen_bt_run_cdm", type="primary"):
        with st.spinner("CDM 백테스트 실행 중..."):
            cdm_tickers = list(set(cdm_settings.get("offensive", []) + cdm_settings.get("defensive", [])))
            cdm_price_data = {}
            for ticker in cdm_tickers:
                df_t = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
                if df_t is None or df_t.empty:
                    st.error(f"{ticker} yfinance 데이터 조회 실패")
                    cdm_price_data = None
                    break
                df_t = df_t.copy().sort_index()
                df_t = df_t[df_t.index >= filter_ts]
                if df_t.empty:
                    st.error(f"{ticker} 시작일 이후 데이터가 없습니다.")
                    cdm_price_data = None
                    break
                cdm_price_data[ticker] = df_t

            if cdm_price_data:
                from src.strategy.cdm import CDMStrategy
                cdm_strategy = CDMStrategy(settings=cdm_settings)
                cdm_bt_result = cdm_strategy.run_backtest(cdm_price_data, initial_balance=float(cap), fee=float(fee))
                if cdm_bt_result:
                    st.session_state["pen_bt_cdm_result"] = cdm_bt_result
                    if "SPY" in cdm_price_data:
                        st.session_state["pen_bt_cdm_benchmark_series"] = _normalize_numeric_series(
                            cdm_price_data.get("SPY"), preferred_cols=("close", "Close"))
                        st.session_state["pen_bt_cdm_benchmark_label"] = "SPY Buy & Hold"
                else:
                    st.error("CDM 백테스트 실행 실패 (데이터 부족)")

    cdm_bt_res = st.session_state.get("pen_bt_cdm_result")
    if cdm_bt_res:
        _eq = cdm_bt_res.get("equity_df")
        metrics = cdm_bt_res["metrics"]
        _render_bt_metrics(metrics)

        _bm_series = st.session_state.get("pen_bt_cdm_benchmark_series")
        _bm_label = str(st.session_state.get("pen_bt_cdm_benchmark_label", "SPY Buy & Hold"))
        if isinstance(_eq, pd.DataFrame) and not _eq.empty:
            _render_performance_analysis(
                equity_series=_eq.get("equity"),
                benchmark_series=_bm_series,
                strategy_metrics=metrics,
                strategy_label="CDM 전략",
                benchmark_label=_bm_label,
                show_drawdown=True, show_weight=True, equity_df=_eq,
            )

        alloc_df = cdm_bt_res.get("allocations")
        if isinstance(alloc_df, pd.DataFrame) and not alloc_df.empty:
            st.subheader("월별 배분 이력")
            st.dataframe(alloc_df.tail(36), use_container_width=True, hide_index=True)

        _render_trade_history_table(cdm_bt_res, key_prefix="bt_cdm")
