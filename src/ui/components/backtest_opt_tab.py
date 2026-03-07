"""파라미터 최적화 + 전체 종목 스캔 탭."""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.constants import TOP_20_TICKERS, INTERVAL_MAP, INTERVAL_REV_MAP, CANDLES_PER_DAY
from src.backtest.engine import BacktestEngine
from src.strategy.donchian import DonchianStrategy
import src.engine.data_cache as data_cache


def _add_robustness(df_in, neighbor_steps=2):
    """각 파라미터 조합에 대해 인접 ±neighbor_steps 단계 Calmar 평균 = Robustness."""
    if "Robustness" in df_in.columns:
        return df_in
    df_out = df_in.copy()

    # Donchian (2D)
    if "Buy Period" in df_out.columns and "Sell Period" in df_out.columns:
        buy_vals = sorted(df_out["Buy Period"].unique())
        sell_vals = sorted(df_out["Sell Period"].unique())
        buy_idx = {v: i for i, v in enumerate(buy_vals)}
        sell_idx = {v: i for i, v in enumerate(sell_vals)}
        max_bi, max_si = len(buy_vals) - 1, len(sell_vals) - 1

        calmar_lookup = {}
        for _, row in df_out.iterrows():
            calmar_lookup[(row["Buy Period"], row["Sell Period"])] = row["Calmar"]

        def _rob(bp, sp):
            bi, si = buy_idx.get(bp, -1), sell_idx.get(sp, -1)
            if bi == -1 or si == -1:
                return 0.0
            vals = []
            for b_i in range(max(0, bi - neighbor_steps), min(max_bi, bi + neighbor_steps) + 1):
                for s_i in range(max(0, si - neighbor_steps), min(max_si, si + neighbor_steps) + 1):
                    k = (buy_vals[b_i], sell_vals[s_i])
                    if k in calmar_lookup:
                        vals.append(calmar_lookup[k])
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        df_out["Robustness"] = df_out.apply(lambda r: _rob(r["Buy Period"], r["Sell Period"]), axis=1)

    # SMA (1D)
    elif "SMA Period" in df_out.columns:
        vals = sorted(df_out["SMA Period"].unique())
        v_idx = {v: i for i, v in enumerate(vals)}
        max_i = len(vals) - 1
        lookup = {}
        for _, row in df_out.iterrows():
            lookup[row["SMA Period"]] = row["Calmar"]

        def _rob_sma(val):
            idx = v_idx.get(val, -1)
            if idx == -1:
                return 0.0
            n_vals = []
            for i in range(max(0, idx - neighbor_steps), min(max_i, idx + neighbor_steps) + 1):
                nv = vals[i]
                if nv in lookup:
                    n_vals.append(lookup[nv])
            return round(sum(n_vals) / len(n_vals), 2) if n_vals else 0.0

        df_out["Robustness"] = df_out["SMA Period"].apply(_rob_sma)

    return df_out


def render_optimization_tab(portfolio_list, initial_cap, backtest_engine):
    """파라미터 최적화 서브탭."""
    st.header("파라미터 최적화")

    with st.expander("데이터 캐시 관리", expanded=False):
        cache_list = data_cache.list_cache()
        if cache_list:
            st.dataframe(pd.DataFrame(cache_list), use_container_width=True, hide_index=True)
        else:
            st.info("캐시된 데이터가 없습니다.")

        if st.button("캐시 전체 삭제", key="opt_clear_cache"):
            data_cache.clear_cache()
            st.success("캐시가 삭제되었습니다.")
            st.rerun()

    # 최적화 대상 설정
    opt_port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
    opt_base_options = list(dict.fromkeys(opt_port_tickers + TOP_20_TICKERS))
    opt_ticker_sel = st.selectbox("최적화 대상", opt_base_options, key="opt_ticker")
    opt_strategy_sel = st.selectbox("전략", ["SMA 전략", "돈키안 전략"], key="opt_strat_sel")

    with st.form("optimization_form"):
        opt_method = st.radio("최적화 방법", ["Grid Search (전수 탐색)", "Optuna (베이지안 최적화)"], horizontal=True, key="opt_method")
        use_optuna = "Optuna" in opt_method

        opt_interval_label = st.selectbox("시간봉", options=list(INTERVAL_MAP.keys()), index=0, key="opt_interval_sel")
        opt_interval = INTERVAL_MAP[opt_interval_label]

        if opt_strategy_sel == "돈키안 전략":
            st.caption("돈치안 채널의 매수/매도 기간을 최적화합니다.")
            st.markdown("##### 매수 채널 기간")
            oc1, oc2, oc3 = st.columns(3)
            opt_buy_start = oc1.number_input("시작", 5, 200, 10, key="opt_dc_buy_start")
            opt_buy_end = oc2.number_input("끝", 5, 200, 60, key="opt_dc_buy_end")
            opt_buy_step = oc3.number_input("간격", 1, 50, 5, key="opt_dc_buy_step")
            st.markdown("##### 매도 채널 기간")
            oc4, oc5, oc6 = st.columns(3)
            opt_sell_start = oc4.number_input("시작", 5, 200, 5, key="opt_dc_sell_start")
            opt_sell_end = oc5.number_input("끝", 5, 200, 30, key="opt_dc_sell_end")
            opt_sell_step = oc6.number_input("간격", 1, 50, 5, key="opt_dc_sell_step")
            st.markdown("##### 매도 방식")
            st.caption("하단선: 저가 채널 이탈 시 매도 | 중심선: (상단+하단)/2 이탈 시 매도")
            opt_dc_sell_mode = st.radio(
                "매도 라인",
                ["하단선 (Lower)", "중심선 (Midline)", "두 방법 비교"],
                horizontal=True,
                key="opt_dc_sell_mode",
            )
        else:
            st.caption("SMA 이동평균 기간을 최적화합니다.")
            st.markdown("##### SMA 기간")
            oc1, oc2, oc3 = st.columns(3)
            opt_s_start = oc1.number_input("시작", 5, 200, 20, key="opt_s_start")
            opt_s_end = oc2.number_input("끝", 5, 200, 60, key="opt_s_end")
            opt_s_step = oc3.number_input("간격", 1, 50, 5, key="opt_s_step")

        if use_optuna:
            st.divider()
            st.markdown("##### Optuna 설정")
            opc1, opc2 = st.columns(2)
            optuna_n_trials = opc1.number_input("탐색 횟수", 50, 2000, 200, step=50, key="optuna_trials")
            optuna_objective = opc2.selectbox("목적함수", ["Calmar (CAGR/|MDD|)", "Sharpe", "수익률 (Return)", "MDD 최소"], key="optuna_obj")

        # 기간 설정
        st.divider()
        opt_d1, opt_d2 = st.columns(2)
        opt_start = opt_d1.date_input("시작일", value=datetime(2020, 1, 1).date(), key="opt_start_date")
        opt_end = opt_d2.date_input("종료일", value=datetime.now().date(), key="opt_end_date")
        opt_fee = st.number_input("수수료 (%)", value=0.05, format="%.2f", key="opt_fee") / 100
        opt_slippage = st.number_input("슬리피지 (%)", value=0.05, min_value=0.0, max_value=2.0, step=0.01, format="%.2f", key="opt_slippage")

        opt_submitted = st.form_submit_button("최적화 시작", type="primary")

    if opt_strategy_sel != "돈키안 전략":
        opt_dc_sell_mode = "하단선 (Lower)"

    if opt_submitted:
        import plotly.express as px
        opt_results = []
        opt_days_diff = (opt_end - opt_start).days

        with st.status("최적화 진행 중...", expanded=True) as status:
            progress_bar = st.progress(0)
            log_area = st.empty()

            try:
                import time as _time
                opt_cpd = CANDLES_PER_DAY.get(opt_interval, 1)
                to_str_opt = (opt_end + timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")

                if opt_strategy_sel == "돈키안 전략":
                    buy_range = range(opt_buy_start, opt_buy_end + 1, opt_buy_step)
                    sell_range = range(opt_sell_start, opt_sell_end + 1, opt_sell_step)
                    total_iter = len(buy_range) * len(sell_range)
                    max_req_p = max(opt_buy_end, opt_sell_end)
                    fetch_count_opt = opt_days_diff * opt_cpd + max_req_p + 300
                else:
                    sma_range = range(opt_s_start, opt_s_end + 1, opt_s_step)
                    total_iter = len(sma_range)
                    fetch_count_opt = opt_days_diff * opt_cpd + opt_s_end + 300

                def dl_progress(fetched, total):
                    pct = min(fetched / total, 1.0) if total > 0 else 0
                    progress_bar.progress(pct * 0.3)
                    log_area.text(f"다운로드: {fetched:,}/{total:,} candles ({pct*100:.0f}%)")

                t0 = _time.time()
                full_df = data_cache.get_ohlcv_cached(opt_ticker_sel, interval=opt_interval, to=to_str_opt, count=fetch_count_opt, progress_callback=dl_progress)
                dl_elapsed = _time.time() - t0

                if full_df is None or full_df.empty:
                    status.update(label="데이터 로드 실패", state="error")
                else:
                    st.write(f"데이터 준비: {len(full_df):,} candles ({dl_elapsed:.1f}초)")

                    def opt_progress(idx, total, msg):
                        pct = 0.3 + (idx / total) * 0.7
                        progress_bar.progress(min(pct, 1.0))
                        log_area.text(f"{msg} ({idx}/{total})")

                    t1 = _time.time()
                    optuna_result = None

                    if use_optuna:
                        obj_map = {"Calmar (CAGR/|MDD|)": "calmar", "Sharpe": "sharpe", "수익률 (Return)": "return", "MDD 최소": "mdd"}
                        obj_key = obj_map.get(optuna_objective, "calmar")

                        if opt_strategy_sel == "돈키안 전략":
                            optuna_result = backtest_engine.optuna_optimize(
                                full_df, strategy_mode="Donchian", buy_range=(opt_buy_start, opt_buy_end),
                                sell_range=(opt_sell_start, opt_sell_end), fee=opt_fee, slippage=opt_slippage,
                                start_date=opt_start, initial_balance=initial_cap, n_trials=optuna_n_trials,
                                objective_metric=obj_key, progress_callback=opt_progress)
                        else:
                            optuna_result = backtest_engine.optuna_optimize(
                                full_df, strategy_mode="SMA 전략", buy_range=(opt_s_start, opt_s_end),
                                fee=opt_fee, slippage=opt_slippage, start_date=opt_start,
                                initial_balance=initial_cap, n_trials=optuna_n_trials,
                                objective_metric=obj_key, progress_callback=opt_progress)

                        for r in optuna_result['trials']:
                            row = {"Total Return (%)": r["total_return"], "CAGR (%)": r["cagr"], "MDD (%)": r["mdd"],
                                   "Calmar": r["calmar"], "Win Rate (%)": r["win_rate"], "Sharpe": r["sharpe"], "Trades": r["trade_count"]}
                            if opt_strategy_sel == "돈키안 전략":
                                row["Buy Period"] = r["buy_period"]
                                row["Sell Period"] = r["sell_period"]
                            else:
                                row["SMA Period"] = r["sma_period"]
                            opt_results.append(row)
                        total_iter = optuna_n_trials
                    else:
                        if opt_strategy_sel == "돈키안 전략":
                            buy_range = range(opt_buy_start, opt_buy_end + 1, opt_buy_step)
                            sell_range = range(opt_sell_start, opt_sell_end + 1, opt_sell_step)

                            _modes_to_run = []
                            if opt_dc_sell_mode == "두 방법 비교":
                                _modes_to_run = [("lower", "하단선"), ("midline", "중심선")]
                            elif opt_dc_sell_mode == "중심선 (Midline)":
                                _modes_to_run = [("midline", "중심선")]
                            else:
                                _modes_to_run = [("lower", "하단선")]

                            _all_mode_results = {}
                            for _sm, _sm_label in _modes_to_run:
                                _raw = backtest_engine.optimize_donchian(
                                    full_df, buy_range, sell_range, fee=opt_fee, slippage=opt_slippage,
                                    start_date=opt_start, initial_balance=initial_cap, progress_callback=opt_progress,
                                    sell_mode=_sm)
                                _mode_rows = []
                                for r in _raw:
                                    _mode_rows.append({"Buy Period": r["Buy Period"], "Sell Period": r["Sell Period"],
                                        "Total Return (%)": r["total_return"], "CAGR (%)": r["cagr"], "MDD (%)": r["mdd"],
                                        "Calmar": abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0,
                                        "Win Rate (%)": r["win_rate"], "Sharpe": r["sharpe"], "Trades": r["trade_count"]})
                                _all_mode_results[_sm_label] = _mode_rows
                                if len(_modes_to_run) == 1:
                                    opt_results = _mode_rows
                            if len(_modes_to_run) == 2:
                                st.session_state["opt_compare_results"] = _all_mode_results
                                st.session_state.pop("opt_single_results", None)
                            else:
                                st.session_state["opt_single_results"] = {
                                    "rows": opt_results,
                                    "strategy": opt_strategy_sel,
                                    "use_optuna": use_optuna,
                                    "ticker": opt_ticker_sel,
                                    "interval": opt_interval,
                                    "start_date": str(opt_start),
                                    "end_date": str(opt_end),
                                    "fee": opt_fee,
                                    "slippage": opt_slippage,
                                    "initial_balance": initial_cap,
                                }
                                st.session_state.pop("opt_compare_results", None)
                        else:
                            raw_results = backtest_engine.optimize_sma(
                                full_df, sma_range, fee=opt_fee, slippage=opt_slippage,
                                start_date=opt_start, initial_balance=initial_cap, progress_callback=opt_progress)
                            for r in raw_results:
                                opt_results.append({"SMA Period": r["SMA Period"],
                                    "Total Return (%)": r["total_return"], "CAGR (%)": r["cagr"], "MDD (%)": r["mdd"],
                                    "Calmar": abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0,
                                    "Win Rate (%)": r["win_rate"], "Sharpe": r["sharpe"], "Trades": r["trade_count"]})

                    opt_elapsed = _time.time() - t1
                    status.update(label=f"완료! ({total_iter}건, {dl_elapsed:.1f}초 + {opt_elapsed:.1f}초)", state="complete")

                    if opt_results and "opt_single_results" not in st.session_state:
                        st.session_state["opt_single_results"] = {
                            "rows": opt_results,
                            "strategy": opt_strategy_sel,
                            "use_optuna": use_optuna,
                            "ticker": opt_ticker_sel,
                            "interval": opt_interval,
                            "start_date": str(opt_start),
                            "end_date": str(opt_end),
                            "fee": opt_fee,
                            "slippage": opt_slippage,
                            "initial_balance": initial_cap,
                        }

            except Exception as e:
                status.update(label=f"오류: {e}", state="error")
                import traceback
                st.code(traceback.format_exc())

    # ── 결과 표시 (세션 상태에서 유지) ──
    _saved_compare = st.session_state.get("opt_compare_results", {})
    _saved_single = st.session_state.get("opt_single_results", {})

    if _saved_compare:
        st.subheader("🔀 매도방식 비교 결과")
        tab_labels = list(_saved_compare.keys())
        cmp_tabs = st.tabs([f"📊 {lbl}" for lbl in tab_labels])
        for _tab, _lbl in zip(cmp_tabs, tab_labels):
            with _tab:
                _rows = _saved_compare[_lbl]
                if _rows:
                    _df = pd.DataFrame(_rows).sort_values("Total Return (%)", ascending=False).reset_index(drop=True)
                    _df = _add_robustness(_df)
                    _df.index = _df.index + 1
                    _df.index.name = "순위"
                    _best = _df.iloc[0]
                    st.success(f"【{_lbl}】 최적: 매수 **{int(_best['Buy Period'])}**, 매도 **{int(_best['Sell Period'])}** → 수익률 {_best['Total Return (%)']:.2f}%, Calmar {_best['Calmar']:.2f}, Robustness {_best['Robustness']:.2f}")
                    st.dataframe(
                        _df.style.background_gradient(cmap='RdYlGn', subset=['Total Return (%)', 'Calmar', 'Sharpe', 'Robustness'])
                            .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)']).format("{:,.2f}"),
                        use_container_width=True, height=400)
                    import plotly.express as _px
                    _fig = _px.density_heatmap(_df.reset_index(), x="Buy Period", y="Sell Period", z="Total Return (%)",
                                               histfunc="avg", title=f"[{_lbl}] 히트맵", text_auto=".0f", color_continuous_scale="RdYlGn")
                    st.plotly_chart(_fig, use_container_width=True)
                else:
                    st.info(f"{_lbl} 결과 없음")

        # 핵심 지표 비교 테이블
        if len(tab_labels) == 2:
            st.subheader("📋 핵심 지표 비교")
            _compare_rows = []
            for _lbl in tab_labels:
                _rows = _saved_compare[_lbl]
                if _rows:
                    _df2 = pd.DataFrame(_rows)
                    _df2 = _add_robustness(_df2)
                    _best2 = _df2.sort_values("Total Return (%)", ascending=False).iloc[0]
                    _compare_rows.append({
                        "매도방식": _lbl,
                        "최적 매수": int(_best2["Buy Period"]),
                        "최적 매도": int(_best2["Sell Period"]),
                        "수익률 (%)": round(_best2["Total Return (%)"], 2),
                        "CAGR (%)": round(_best2["CAGR (%)"], 2),
                        "MDD (%)": round(_best2["MDD (%)"], 2),
                        "Calmar": round(_best2["Calmar"], 2),
                        "Robustness": round(_best2["Robustness"], 2),
                        "Sharpe": round(_best2["Sharpe"], 2),
                        "거래횟수": int(_best2["Trades"]),
                    })
            if _compare_rows:
                _cmp_df = pd.DataFrame(_compare_rows).set_index("매도방식")
                st.dataframe(_cmp_df.style.highlight_max(axis=0, color="#d4edda", subset=["수익률 (%)", "Calmar", "Robustness", "Sharpe"]).highlight_min(axis=0, color="#f8d7da", subset=["MDD (%)"]), use_container_width=True)

    elif _saved_single:
        import plotly.express as px
        opt_results = _saved_single["rows"]
        _s_strategy = _saved_single["strategy"]
        _s_optuna = _saved_single["use_optuna"]
        if opt_results:
            opt_df = pd.DataFrame(opt_results)
            opt_df = _add_robustness(opt_df)
            _total_combos = len(opt_df)

            # ── 결과 필터 & 정렬 ──
            _fc1, _fc2, _fc3 = st.columns(3)
            _SORT_OPTIONS = ["Calmar (CAGR/MDD)", "수익률 (높은순)", "CAGR (높은순)", "MDD (낮은순)", "Sharpe (높은순)", "Robustness (높은순)"]
            _opt_sort = _fc1.selectbox("정렬 기준", _SORT_OPTIONS, key="opt_sort_by")
            _opt_mdd_filter = _fc2.number_input("최대 MDD (%)", -100.0, 0.0, -50.0, 5.0, format="%.1f", key="opt_max_mdd", help="이 값보다 MDD가 나쁜 조합은 제외")
            _opt_top_n = int(_fc3.number_input("상위 N개", 5, 200, 30, 5, key="opt_top_n"))

            _sort_map = {"Calmar (CAGR/MDD)": ("Calmar", False), "수익률 (높은순)": ("Total Return (%)", False),
                         "CAGR (높은순)": ("CAGR (%)", False), "MDD (낮은순)": ("MDD (%)", True),
                         "Sharpe (높은순)": ("Sharpe", False), "Robustness (높은순)": ("Robustness", False)}
            _scol, _sasc = _sort_map.get(_opt_sort, ("Calmar", False))
            if _scol in opt_df.columns:
                opt_df = opt_df.sort_values(_scol, ascending=_sasc).reset_index(drop=True)
            best_row = opt_df.iloc[0]

            # MDD 필터
            _filtered_df = opt_df[opt_df["MDD (%)"] >= _opt_mdd_filter].reset_index(drop=True)
            _n_filtered = len(_filtered_df)

            # 최적 결과 요약
            if _s_strategy == "돈키안 전략":
                st.success(f"최적: 매수 **{int(best_row['Buy Period'])}**, 매도 **{int(best_row['Sell Period'])}** → 수익률 {best_row['Total Return (%)']:.2f}%, CAGR {best_row['CAGR (%)']:.2f}%, MDD {best_row['MDD (%)']:.2f}%, Calmar {best_row['Calmar']:.2f}, Robustness {best_row.get('Robustness', 0):.2f}")
            else:
                st.success(f"최적: SMA **{int(best_row['SMA Period'])}** → 수익률 {best_row['Total Return (%)']:.2f}%, CAGR {best_row['CAGR (%)']:.2f}%, MDD {best_row['MDD (%)']:.2f}%, Calmar {best_row['Calmar']:.2f}, Robustness {best_row.get('Robustness', 0):.2f}")
            if _n_filtered < _total_combos:
                st.caption(f"총 {_total_combos}개 중 {_n_filtered}개 통과 (MDD ≥ {_opt_mdd_filter:.1f}%) | {_opt_sort} 기준 상위 {min(_opt_top_n, _n_filtered)}개 표시")
            else:
                st.caption(f"총 {_total_combos}개 | {_opt_sort} 기준 상위 {min(_opt_top_n, _total_combos)}개 표시")

            # 결과 테이블
            _display_src = _filtered_df if _n_filtered > 0 else opt_df
            _display_df = _display_src.head(_opt_top_n).copy()
            _display_df.index = _display_df.index + 1
            _display_df.index.name = "순위"
            _grad_cols = [c for c in ['Total Return (%)', 'Calmar', 'Sharpe', 'Robustness'] if c in _display_df.columns]
            st.dataframe(
                _display_df.style.background_gradient(cmap='RdYlGn', subset=_grad_cols)
                    .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)']).format("{:,.2f}"),
                use_container_width=True, height=500)

            # 상세 분석 Expander (Robustness)
            with st.expander("🔍 최적 파라미터 주변 상세 분석 (Robustness)", expanded=False):
                try:
                    if _s_strategy == "돈키안 전략" and "Buy Period" in opt_df.columns:
                        st.caption("최적 (Buy, Sell) 파라미터 기준 ±2단계 이웃들의 성과를 분석합니다.")
                        b_val, s_val = int(best_row["Buy Period"]), int(best_row["Sell Period"])
                        b_uniq = sorted(opt_df["Buy Period"].unique())
                        s_uniq = sorted(opt_df["Sell Period"].unique())

                        if b_val in b_uniq and s_val in s_uniq:
                            b_idx, s_idx = b_uniq.index(b_val), s_uniq.index(s_val)
                            nb_vals = b_uniq[max(0, b_idx - 2): min(len(b_uniq), b_idx + 3)]
                            ns_vals = s_uniq[max(0, s_idx - 2): min(len(s_uniq), s_idx + 3)]

                            sub_df = opt_df[
                                (opt_df["Buy Period"].isin(nb_vals)) &
                                (opt_df["Sell Period"].isin(ns_vals))
                            ].copy()

                            c1, c2, c3 = st.columns(3)
                            c1.metric("이웃 평균 수익률", f"{sub_df['Total Return (%)'].mean():.2f}%")
                            c2.metric("이웃 평균 Calmar", f"{sub_df['Calmar'].mean():.2f}")
                            c3.metric("이웃 최소 MDD", f"{sub_df['MDD (%)'].min():.2f}%")

                            st.dataframe(sub_df.style.background_gradient(cmap='RdYlGn', subset=['Calmar']), use_container_width=True)
                        else:
                            st.warning("파라미터 인덱스 조회 실패")

                    elif _s_strategy != "돈키안 전략" and "SMA Period" in opt_df.columns:
                        st.caption("최적 SMA Period 기준 ±2단계 이웃들의 성과를 분석합니다.")
                        p_val = int(best_row["SMA Period"])
                        p_uniq = sorted(opt_df["SMA Period"].unique())

                        if p_val in p_uniq:
                            p_idx = p_uniq.index(p_val)
                            np_vals = p_uniq[max(0, p_idx - 2): min(len(p_uniq), p_idx + 3)]

                            sub_df = opt_df[opt_df["SMA Period"].isin(np_vals)].copy()

                            c1, c2 = st.columns(2)
                            c1.metric("이웃 평균 수익률", f"{sub_df['Total Return (%)'].mean():.2f}%")
                            c2.metric("이웃 평균 Calmar", f"{sub_df['Calmar'].mean():.2f}")

                            st.bar_chart(sub_df.set_index("SMA Period")[["Calmar", "Total Return (%)"]])
                        else:
                            st.warning("파라미터 인덱스 조회 실패")
                except Exception as e:
                    st.error(f"상세 분석 중 오류 발생: {e}")

            if _s_strategy == "돈키안 전략" and not _s_optuna:
                fig_opt = px.density_heatmap(opt_df.reset_index(), x="Buy Period", y="Sell Period", z="Total Return (%)", histfunc="avg", title="돈키안 최적화 히트맵", text_auto=".0f", color_continuous_scale="RdYlGn")
                st.plotly_chart(fig_opt, use_container_width=True)
            elif _s_strategy != "돈키안 전략" and not _s_optuna:
                st.line_chart(opt_df.reset_index().set_index("SMA Period")[['Total Return (%)', 'MDD (%)']])


def render_scan_tab():
    """전체 종목 스캔 서브탭."""
    st.header("전체 종목 스캔")
    st.caption("상위 종목을 전 시간대/전략으로 백테스트하여 Calmar 순으로 정렬합니다.")

    # 스캔 설정
    scan_col1, scan_col2, scan_col3 = st.columns(3)
    scan_strategy = scan_col1.selectbox("전략", ["SMA", "Donchian"], key="scan_strat")
    scan_period = scan_col2.number_input("기간 (Period)", 5, 300, 20, key="scan_period")
    scan_count = scan_col3.number_input("백테스트 캔들 수", 200, 10000, 2000, step=200, key="scan_count")

    scan_col4, scan_col5 = st.columns(2)
    _scan_interval_alias = {
        "일봉": "1D", "4시간": "4H", "1시간": "1H",
        "30분": "30m", "15분": "15m", "5분": "5m", "1분": "1m",
    }
    _scan_default_raw = st.session_state.get("scan_intervals", ["1D", "4H", "1H"])
    if not isinstance(_scan_default_raw, (list, tuple)):
        _scan_default_raw = ["1D", "4H", "1H"]
    _scan_defaults = []
    for _v in _scan_default_raw:
        _k = _scan_interval_alias.get(str(_v), str(_v))
        if _k in INTERVAL_MAP and _k not in _scan_defaults:
            _scan_defaults.append(_k)
    if not _scan_defaults:
        _scan_defaults = [k for k in ["1D", "4H", "1H"] if k in INTERVAL_MAP]

    scan_intervals = scan_col4.multiselect(
        "시간봉", list(INTERVAL_MAP.keys()),
        default=_scan_defaults,
        key="scan_intervals"
    )
    sell_ratio = 0.5
    if scan_strategy == "Donchian":
        sell_ratio = st.slider("매도 채널 비율", 0.1, 1.0, 0.5, 0.1, key="scan_sell_ratio")

    st.caption(f"대상: 시가총액 상위 {len(TOP_20_TICKERS)}개 — {', '.join(t.replace('KRW-', '') for t in TOP_20_TICKERS)}")

    if st.button("🔍 스캔 시작", key="scan_run", type="primary"):
        engine = BacktestEngine()
        top_tickers = TOP_20_TICKERS

        if top_tickers:
            interval_apis = [INTERVAL_MAP[k] for k in scan_intervals]
            total_jobs = len(top_tickers) * len(interval_apis)
            st.write(f"종목 {len(top_tickers)}개 × 시간봉 {len(interval_apis)}개 = 총 **{total_jobs}건** 백테스트")

            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            done = 0

            for ticker in top_tickers:
                for interval_api in interval_apis:
                    done += 1
                    interval_label = INTERVAL_REV_MAP.get(interval_api, interval_api)
                    status_text.text(f"[{done}/{total_jobs}] {ticker} ({interval_label})")
                    progress_bar.progress(done / total_jobs)

                    try:
                        df = data_cache.get_ohlcv_cached(ticker, interval=interval_api, count=scan_count)
                        if df is None or len(df) < scan_period + 10:
                            continue

                        df = df.copy()

                        if scan_strategy == "Donchian":
                            strat = DonchianStrategy()
                            sell_p = max(5, int(scan_period * sell_ratio))
                            df = strat.create_features(df, buy_period=scan_period, sell_period=sell_p)
                            signal_arr = np.zeros(len(df), dtype=np.int8)
                            upper_col = f'Donchian_Upper_{scan_period}'
                            lower_col = f'Donchian_Lower_{sell_p}'
                            if upper_col in df.columns and lower_col in df.columns:
                                signal_arr[df['close'].values > df[upper_col].values] = 1
                                signal_arr[df['close'].values < df[lower_col].values] = -1
                            else:
                                continue
                        else:
                            sma_vals = df['close'].rolling(window=scan_period).mean().values
                            close_vals = df['close'].values
                            signal_arr = np.zeros(len(df), dtype=np.int8)
                            valid = ~np.isnan(sma_vals)
                            signal_arr[valid & (close_vals > sma_vals)] = 1
                            signal_arr[valid & (close_vals <= sma_vals)] = -1

                        open_arr = df['open'].values
                        close_arr = df['close'].values

                        res = engine._fast_simulate(open_arr, close_arr, signal_arr, fee=0.0005, slippage=0.0, initial_balance=1000000)

                        bnh_return = (close_arr[-1] / close_arr[0] - 1) * 100
                        calmar = abs(res['cagr'] / res['mdd']) if res['mdd'] != 0 else 0

                        results.append({
                            '종목': ticker,
                            '시간봉': interval_label,
                            'CAGR (%)': round(res['cagr'], 2),
                            'MDD (%)': round(res['mdd'], 2),
                            'Calmar': round(calmar, 2),
                            '수익률 (%)': round(res['total_return'], 2),
                            'B&H (%)': round(bnh_return, 2),
                            '초과수익 (%)': round(res['total_return'] - bnh_return, 2),
                            '승률 (%)': round(res['win_rate'], 1),
                            '거래수': res['trade_count'],
                            'Sharpe': round(res['sharpe'], 2),
                            '캔들수': len(df),
                        })
                    except Exception:
                        continue

            progress_bar.progress(1.0)
            status_text.text(f"완료! {len(results)}건 결과")

            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values('Calmar', ascending=False).reset_index(drop=True)
                df_results.index = df_results.index + 1
                df_results.index.name = "순위"

                st.success(f"스캔 완료: {len(results)}건 중 수익 {len(df_results[df_results['수익률 (%)'] > 0])}건, 손실 {len(df_results[df_results['수익률 (%)'] <= 0])}건")

                st.dataframe(
                    df_results.style.format({
                        'CAGR (%)': '{:.2f}',
                        'MDD (%)': '{:.2f}',
                        'Calmar': '{:.2f}',
                        '수익률 (%)': '{:.2f}',
                        'B&H (%)': '{:.2f}',
                        '초과수익 (%)': '{:.2f}',
                        '승률 (%)': '{:.1f}',
                        'Sharpe': '{:.2f}',
                    }).background_gradient(cmap='RdYlGn', subset=['Calmar', '초과수익 (%)'])
                    .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)']),
                    use_container_width=True,
                    height=700,
                )

                st.divider()
                sum_col1, sum_col2 = st.columns(2)
                with sum_col1:
                    st.caption("시간봉별 평균 Calmar")
                    interval_summary = df_results.groupby('시간봉').agg(
                        Calmar_평균=('Calmar', 'mean'),
                        수익률_평균=('수익률 (%)', 'mean'),
                        종목수=('종목', 'count')
                    ).sort_values('Calmar_평균', ascending=False)
                    st.dataframe(interval_summary.style.format({'Calmar_평균': '{:.2f}', '수익률_평균': '{:.2f}'}), use_container_width=True)

                with sum_col2:
                    st.caption("종목별 최고 Calmar 시간봉")
                    best_per_ticker = df_results.loc[df_results.groupby('종목')['Calmar'].idxmax()][['종목', '시간봉', 'Calmar', '수익률 (%)', 'MDD (%)']].reset_index(drop=True)
                    best_per_ticker.index = best_per_ticker.index + 1
                    st.dataframe(best_per_ticker.style.format({'Calmar': '{:.2f}', '수익률 (%)': '{:.2f}', 'MDD (%)': '{:.2f}'}), use_container_width=True)
            else:
                st.warning("결과가 없습니다. 데이터 다운로드가 필요할 수 있습니다.")
