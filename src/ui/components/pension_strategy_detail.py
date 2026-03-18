"""연금저축 전략 상세 탭 (Tab 1 하단 LAA/DM/VAA/CDM 서브탭)."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.utils.formatting import _fmt_etf_code_name
from src.ui.components.performance import _render_performance_analysis


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


def render_strategy_detail_tabs(
    active_strategies: list[str],
    auto_signal_strategies: list[str],
    laa_res: dict | None,
    dm_res: dict | None,
    vaa_res: dict | None,
    bal: dict | None,
    pen_bt_start_raw: str,
    pen_bt_cap: float,
    dm_settings: dict | None,
):
    """전략별 상세 서브탭 렌더링 (Tab 1 하단)."""
    import streamlit.components.v1 as components

    st.divider()

    _detail_tab_names = []
    _detail_tab_keys = []
    if "LAA" in active_strategies:
        _detail_tab_names.append("LAA 전략")
        _detail_tab_keys.append("LAA")
    if "듀얼모멘텀" in active_strategies:
        _detail_tab_names.append("듀얼모멘텀 전략")
        _detail_tab_keys.append("DM")
    if "VAA" in active_strategies:
        _detail_tab_names.append("VAA 전략")
        _detail_tab_keys.append("VAA")
    if _detail_tab_names:
        _detail_tabs = st.tabs(_detail_tab_names)
        _detail_tab_map = dict(zip(_detail_tab_keys, _detail_tabs))
    else:
        _detail_tab_map = {}

    if "LAA" in _detail_tab_map:
        with _detail_tab_map["LAA"]:
            _render_laa_detail(laa_res, bal, pen_bt_start_raw, pen_bt_cap,
                               auto_signal_strategies, components)

    if "DM" in _detail_tab_map:
        with _detail_tab_map["DM"]:
            _render_dm_detail(dm_res, bal, pen_bt_start_raw, pen_bt_cap,
                              dm_settings, auto_signal_strategies, components)

    if "VAA" in _detail_tab_map:
        with _detail_tab_map["VAA"]:
            _render_vaa_detail(vaa_res, auto_signal_strategies)


def _guide_button_js(components, key: str):
    """전략 가이드 탭 이동 JS 버튼."""
    if st.button("📖 전략 가이드", key=key):
        components.html("""
        <script>
        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        for (let t of tabs) {
            if (t.textContent.includes('전략 가이드')) { t.click(); break; }
        }
        </script>
        """, height=0)


def _render_laa_detail(res, bal, pen_bt_start_raw, pen_bt_cap, auto_signal_strategies, components):
    """LAA 전략 상세."""
    _laa_hdr_col1, _laa_hdr_col2 = st.columns([4, 1])
    with _laa_hdr_col1:
        st.subheader("LAA 전략 포트폴리오")
    with _laa_hdr_col2:
        _guide_button_js(components, "pen_laa_guide_btn")

    if not res:
        if "LAA" not in auto_signal_strategies:
            st.info("LAA 비중이 0%라 자동 시그널 계산을 생략했습니다.")
        else:
            st.info("LAA 시그널이 계산되지 않았습니다.")
        return

    if res.get("error"):
        st.error(res["error"])
        return

    sig = res["signal"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("리스크 상태", "공격 (Risk-On)" if sig.get("risk_on") else "방어 (Risk-Off)")
    c2.metric("리스크 자산", sig["selected_risk_asset"])
    c3.metric("국내 ETF", _fmt_etf_code_name(sig["selected_risk_kr_code"]))
    c4.metric("권장 동작", res["action"])
    st.info(sig.get("reason", ""))

    if "source_map" in res:
        st.caption("시그널 데이터(국내): " + ", ".join([f"{k}→{v}" for k, v in res["source_map"].items()]))

    # 리스크 판단 차트
    _risk_chart_df = res.get("risk_chart_df")
    _risk_chart_code = str(res.get("risk_chart_code", "")).strip() or "360750"
    if isinstance(_risk_chart_df, pd.DataFrame) and not _risk_chart_df.empty and "ma200" in _risk_chart_df.columns:
        _plot_df = _risk_chart_df.dropna(subset=["ma200"]).copy()
        if not _plot_df.empty:
            _last = _plot_df.iloc[-1]
            _last_close = float(_last.get("close", 0.0))
            _last_ma200 = float(_last.get("ma200", 0.0))
            _last_div = float(_last.get("divergence", 0.0))
            _mode_label = "공격 (Risk-On)" if sig.get("risk_on") else "방어 (Risk-Off)"
            _mode_color = "#16a34a" if sig.get("risk_on") else "#dc2626"

            _is_us_spy = "US" in str(_risk_chart_code)
            _chart_title = "SPY (미국 S&P500) + 200일선" if _is_us_spy else "TIGER 미국S&P500 + 200일선"
            _unit = "USD" if _is_us_spy else "KRW"
            st.subheader(_chart_title)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("현재가", f"{_last_close:,.2f} {_unit}" if _is_us_spy else f"{_last_close:,.0f} {_unit}")
            cc2.metric("200일선", f"{_last_ma200:,.2f} {_unit}" if _is_us_spy else f"{_last_ma200:,.0f} {_unit}")
            cc3.metric("이격도(200일)", f"{_last_div:+.2f}%")
            _chart_label = "SPY (미국 원본)" if _is_us_spy else _fmt_etf_code_name(_risk_chart_code)
            st.caption(f"현재 모드: {_mode_label} | 기준: {_chart_label}")

            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=_plot_df.index, y=_plot_df["close"],
                name=f"{'SPY (미국 원본)' if _is_us_spy else _fmt_etf_code_name(_risk_chart_code)} 종가",
                line=dict(color="royalblue", width=2),
            ))
            fig_risk.add_trace(go.Scatter(
                x=_plot_df.index, y=_plot_df["ma200"],
                name="200일 이동평균",
                line=dict(color="orange", width=2, dash="dash"),
            ))
            fig_risk.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.98,
                text=f"현재 모드: {_mode_label}", showarrow=False,
                bgcolor="rgba(255,255,255,0.85)", bordercolor=_mode_color,
                font=dict(color=_mode_color, size=13),
            )
            fig_risk.update_layout(
                height=430, xaxis_title="날짜", yaxis_title=f"가격 ({_unit})",
                legend=dict(orientation="h", yanchor="bottom", y=1.06),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            st.subheader("이격도 차트")
            fig_div = go.Figure()
            fig_div.add_trace(go.Scatter(
                x=_plot_df.index, y=_plot_df["divergence"],
                name="이격도(%)", line=dict(color="#7c3aed", width=2),
            ))
            fig_div.add_hline(y=0, line_dash="dash", line_color="#374151")
            _max_div = float(np.nanmax(_plot_df["divergence"].values)) if len(_plot_df) else 0.0
            _min_div = float(np.nanmin(_plot_df["divergence"].values)) if len(_plot_df) else 0.0
            if _max_div > 0:
                fig_div.add_hrect(y0=0, y1=_max_div, fillcolor="rgba(22,163,74,0.10)", line_width=0)
            if _min_div < 0:
                fig_div.add_hrect(y0=_min_div, y1=0, fillcolor="rgba(220,38,38,0.10)", line_width=0)
            fig_div.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.98,
                text=f"현재 모드: {_mode_label} ({_last_div:+.2f}%)", showarrow=False,
                bgcolor="rgba(255,255,255,0.85)", bordercolor=_mode_color,
                font=dict(color=_mode_color, size=13),
            )
            fig_div.update_layout(
                height=320, xaxis_title="날짜", yaxis_title="이격도 (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.06),
            )
            st.plotly_chart(fig_div, use_container_width=True)

    if "alloc_df" in res and isinstance(res["alloc_df"], pd.DataFrame) and not res["alloc_df"].empty:
        st.subheader("목표 배분 vs 현재 보유")
        st.caption("매매수량은 비중 차이 기준 자동 계산되며, 소수점 주식은 모두 버림 처리합니다.")
        st.dataframe(res["alloc_df"], use_container_width=True, hide_index=True)

    _laa_bt = res.get("bt_result")
    if isinstance(_laa_bt, dict):
        _laa_eq = _laa_bt.get("equity_df")
        if isinstance(_laa_eq, pd.DataFrame) and not _laa_eq.empty and "equity" in _laa_eq.columns:
            _laa_m = _laa_bt.get("metrics", {}) or {}
            _laa_start = str(res.get("bt_start_date", pen_bt_start_raw))
            _laa_bt_cap = float(res.get("bt_initial_cap", pen_bt_cap) or pen_bt_cap)
            if _laa_bt_cap <= 0:
                _laa_bt_cap = 1.0
            _laa_final_eq = float(_laa_m.get("final_equity", _laa_eq["equity"].iloc[-1]))
            _laa_total_ret = float(_laa_m.get("total_return", 0.0))
            _laa_mdd = float(_laa_m.get("mdd", 0.0))
            _laa_cagr = float(_laa_m.get("cagr", 0.0))

            _bal_valid = isinstance(bal, dict) and not bal.get("error")
            _actual_cash_v = float(bal.get("cash", 0.0)) if _bal_valid else 0.0
            _actual_hlds = (bal.get("holdings", []) or []) if _bal_valid else []
            _actual_eval_v = sum(float(h.get("eval_amt", 0.0) or 0.0) for h in _actual_hlds)
            _actual_total_v = _actual_cash_v + _actual_eval_v

            _alloc_df_sync = res.get("alloc_df")
            _max_gap_pct = 0.0
            _sync_ok = True
            if isinstance(_alloc_df_sync, pd.DataFrame) and not _alloc_df_sync.empty:
                if "비중 차이(%p)" in _alloc_df_sync.columns:
                    _max_gap_pct = float(
                        pd.to_numeric(_alloc_df_sync["비중 차이(%p)"], errors="coerce").abs().max()
                    )
                if "주문" in _alloc_df_sync.columns:
                    _orders = _alloc_df_sync["주문"].astype(str)
                    _sync_ok = not _orders.isin(["매수", "매도"]).any()

            st.write(f"**전략 성과 ({_laa_start} ~ 현재)**")
            st.write(f"수익률: **{_laa_total_ret:+.2f}%** | MDD: **{_laa_mdd:.2f}%** | CAGR: **{_laa_cagr:.2f}%**")
            st.write(f"최종자산: {_laa_final_eq:,.0f}원 (초기자본 {_laa_bt_cap:,.0f}원 기준)")

            st.divider()
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("백테스트 자산", f"{_laa_final_eq:,.0f}원", delta=f"{_laa_total_ret:+.2f}%")
            ac1.caption(f"초기자본 {_laa_bt_cap:,.0f}원 기준")
            ac2.metric("실제 총자산", f"{_actual_total_v:,.0f}원" if _bal_valid else "조회 불가")
            ac3.metric("최대 비중 차이", f"{_max_gap_pct:.2f}%p")
            ac4.metric("포지션 동기화", "일치" if _sync_ok else "불일치")
            ac4.caption("주문 상태가 모두 '유지'이면 일치로 판단합니다.")

            _laa_bm_series = res.get("bt_benchmark_series")
            _laa_bm_label = str(res.get("bt_benchmark_label", "SPY Buy & Hold"))
            if isinstance(_laa_bm_series, pd.Series):
                _laa_bm_series = _laa_bm_series.dropna()
                _laa_bm_series = _laa_bm_series[_laa_bm_series.index >= pd.Timestamp(_laa_start)]

            _render_performance_analysis(
                equity_series=_laa_eq["equity"],
                benchmark_series=_laa_bm_series if isinstance(_laa_bm_series, pd.Series) else None,
                strategy_metrics=_laa_m,
                strategy_label="LAA 전략",
                benchmark_label=_laa_bm_label,
            )
            _render_trade_history_table(_laa_bt, key_prefix="mon_laa")


def _render_dm_detail(dm_res, bal, pen_bt_start_raw, pen_bt_cap, dm_settings, auto_signal_strategies, components):
    """듀얼모멘텀 전략 상세."""
    _dm_hdr_col1, _dm_hdr_col2 = st.columns([4, 1])
    with _dm_hdr_col1:
        st.subheader("듀얼모멘텀 전략 포트폴리오")
    with _dm_hdr_col2:
        _guide_button_js(components, "pen_dm_guide_btn")

    st.caption("공격 ETF 2종 상대모멘텀 + 카나리아 절대모멘텀 기반으로 공격/방어 1종을 선택합니다.")

    if not dm_res:
        if "듀얼모멘텀" not in auto_signal_strategies:
            st.info("듀얼모멘텀 비중이 0%라 자동 시그널 계산을 생략했습니다.")
        else:
            st.info("듀얼모멘텀 시그널이 계산되지 않았습니다.")
        return

    if dm_res.get("error"):
        st.error(dm_res["error"])
        return

    _dm_sig = dm_res["signal"]
    _d1, _d2, _d3, _d4 = st.columns(4)
    _d1.metric("선택 자산", str(_dm_sig.get("target_ticker", "-")))
    _d2.metric("국내 ETF", _fmt_etf_code_name(_dm_sig.get("target_kr_code", "")))
    _d3.metric("카나리아 수익률", f"{float(_dm_sig.get('canary_return', 0.0)) * 100:+.2f}%")
    _d4.metric("권장 동작", str(dm_res.get("action", "HOLD")))
    st.info(str(_dm_sig.get("reason", "")))

    if "source_map" in dm_res:
        st.caption("시그널 데이터(국내): " + ", ".join([f"{k}→{v}" for k, v in dm_res["source_map"].items()]))

    _dm_w = (dm_settings or {}).get("momentum_weights", {}) or {}
    _w1v = float(_dm_w.get("m1", 12.0))
    _w3v = float(_dm_w.get("m3", 4.0))
    _w6v = float(_dm_w.get("m6", 2.0))
    _w12v = float(_dm_w.get("m12", 1.0))
    _lbv = int((dm_settings or {}).get("lookback", 12))
    _tdv = int((dm_settings or {}).get("trading_days_per_month", 22))
    st.subheader("시그널/전략 선택 로직 설명")
    st.markdown(
        f"""
1. 공격 자산(공격 ETF 1, 공격 ETF 2)의 상대 모멘텀 점수를 계산합니다.
   점수식: `((1개월수익률 × {_w1v:g}) + (3개월수익률 × {_w3v:g}) + (6개월수익률 × {_w6v:g}) + (12개월수익률 × {_w12v:g})) ÷ 4`
2. 카나리아 ETF의 절대 모멘텀을 계산합니다.
   절대모멘텀: `룩백 {_lbv}개월 수익률` (월 환산 거래일 `{_tdv}`일 기준)
3. 전략 선택 규칙
   공격 자산 점수 1위가 카나리아 룩백 수익률보다 크면 공격 자산 1위를 선택하고, 아니면 방어 ETF를 선택합니다.
4. 오늘이 리밸런싱일이라고 가정하고, 최근 종가(전일/휴일은 마지막 거래일 종가) 기준으로 목표 비중과 목표 수량(버림)을 계산합니다.
"""
    )

    _score_df = dm_res.get("score_df")
    if isinstance(_score_df, pd.DataFrame) and not _score_df.empty:
        st.subheader("요약 점수")
        st.dataframe(_score_df, use_container_width=True, hide_index=True)

    _mom_df = dm_res.get("momentum_detail_df")
    if isinstance(_mom_df, pd.DataFrame) and not _mom_df.empty:
        st.subheader("모멘텀 계산 과정")
        st.dataframe(_mom_df, use_container_width=True, hide_index=True)

    _exp_df = dm_res.get("expected_rebalance_df")
    _meta = dm_res.get("expected_meta", {}) or {}
    if isinstance(_exp_df, pd.DataFrame) and not _exp_df.empty:
        st.subheader("오늘 리밸런싱 가정 예상 포트폴리오")
        _ref_date = str(_meta.get("ref_date", "") or "-")
        _lag_days = _meta.get("lag_days", None)
        _lag_text = "기준일 정보 없음"
        try:
            _lag_i = int(_lag_days)
            if _lag_i <= 0:
                _lag_text = "당일 종가 기준"
            elif _lag_i == 1:
                _lag_text = "전일 종가 기준"
            else:
                _lag_text = f"휴일 반영 최근 거래일 기준 (오늘 대비 {_lag_i}일 전)"
        except Exception:
            pass
        st.caption(f"기준 종가일: {_ref_date} | {_lag_text}")
        em1, em2, em3 = st.columns(3)
        em1.metric("듀얼모멘텀 전략 비중", f"{float(_meta.get('dm_weight_pct', 0.0)):.1f}%")
        em2.metric("전략 배정 평가금액", f"{float(_meta.get('sleeve_eval', 0.0)):,.0f} KRW")
        em3.metric("버림 후 잔여 현금(예상)", f"{float(_meta.get('sleeve_cash_est', 0.0)):,.0f} KRW")
        st.dataframe(_exp_df, use_container_width=True, hide_index=True)

    # 백테스트 성과
    _dm_bt = dm_res.get("bt_result")
    if isinstance(_dm_bt, dict):
        _dm_eq = _dm_bt.get("equity_df")
        if isinstance(_dm_eq, pd.DataFrame) and not _dm_eq.empty and "equity" in _dm_eq.columns:
            _dm_m = _dm_bt.get("metrics", {}) or {}
            _dm_start = str(dm_res.get("bt_start_date", pen_bt_start_raw))
            _dm_bt_cap = float(dm_res.get("bt_initial_cap", pen_bt_cap) or pen_bt_cap)
            if _dm_bt_cap <= 0:
                _dm_bt_cap = 1.0
            _dm_final_eq = float(_dm_m.get("final_equity", _dm_eq["equity"].iloc[-1]))
            _dm_total_ret = float(_dm_m.get("total_return", 0.0))
            _dm_mdd = float(_dm_m.get("mdd", 0.0))
            _dm_cagr = float(_dm_m.get("cagr", 0.0))

            _bal_valid = isinstance(bal, dict) and not bal.get("error")
            _actual_cash_v = float(bal.get("cash", 0.0)) if _bal_valid else 0.0
            _actual_hlds = (bal.get("holdings", []) or []) if _bal_valid else []
            _actual_eval_v = sum(float(h.get("eval_amt", 0.0) or 0.0) for h in _actual_hlds)
            _actual_total_v = _actual_cash_v + _actual_eval_v

            _dm_kr_map_local = (dm_settings or {}).get("kr_etf_map", {}) or {}
            _dm_codes = [str(v).strip() for v in _dm_kr_map_local.values() if str(v).strip()]
            _actual_dm_codes = []
            for _h in _actual_hlds:
                _code = str(_h.get("code", "")).strip()
                _qty = float(_h.get("qty", 0.0) or 0.0)
                if _code in _dm_codes and _qty > 0:
                    _actual_dm_codes.append(_code)
            _actual_dm_codes = sorted(set(_actual_dm_codes))

            _bt_last_ticker = ""
            if "ticker" in _dm_eq.columns and len(_dm_eq) > 0:
                _bt_last_ticker = str(_dm_eq["ticker"].iloc[-1]).strip().upper()
            if not _bt_last_ticker:
                _bt_last_ticker = str(_dm_sig.get("target_ticker", "")).strip().upper()
            _bt_last_code = str(_dm_kr_map_local.get(_bt_last_ticker, "")).strip()
            _sync_dm = (len(_actual_dm_codes) == 1 and _actual_dm_codes[0] == _bt_last_code)
            _bt_pos_label = _fmt_etf_code_name(_bt_last_code) if _bt_last_code else _bt_last_ticker or "-"
            _actual_pos_label = ", ".join([_fmt_etf_code_name(c) for c in _actual_dm_codes]) if _actual_dm_codes else "CASH"

            st.write(f"**전략 성과 ({_dm_start} ~ 현재)**")
            st.write(f"수익률: **{_dm_total_ret:+.2f}%** | MDD: **{_dm_mdd:.2f}%** | CAGR: **{_dm_cagr:.2f}%**")
            st.write(f"최종자산: {_dm_final_eq:,.0f}원 (초기자본 {_dm_bt_cap:,.0f}원 기준)")

            st.divider()
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("백테스트 자산", f"{_dm_final_eq:,.0f}원", delta=f"{_dm_total_ret:+.2f}%")
            dc1.caption(f"초기자본 {_dm_bt_cap:,.0f}원 기준")
            dc2.metric("실제 총자산", f"{_actual_total_v:,.0f}원" if _bal_valid else "조회 불가")
            dc3.metric("백테/실제 포지션", f"{_bt_pos_label} / {_actual_pos_label}")
            dc4.metric("포지션 동기화", "일치" if _sync_dm else "불일치")

            _dm_bm_series = dm_res.get("bt_benchmark_series")
            _dm_bm_label = str(dm_res.get("bt_benchmark_label", "SPY Buy & Hold"))
            if isinstance(_dm_bm_series, pd.Series):
                _dm_bm_series = _dm_bm_series.dropna()
                _dm_bm_series = _dm_bm_series[_dm_bm_series.index >= pd.Timestamp(_dm_start)]

            _render_performance_analysis(
                equity_series=_dm_eq["equity"],
                benchmark_series=_dm_bm_series if isinstance(_dm_bm_series, pd.Series) else None,
                strategy_metrics=_dm_m,
                strategy_label="듀얼모멘텀 전략",
                benchmark_label=_dm_bm_label,
            )
            _render_trade_history_table(_dm_bt, key_prefix="mon_dm")


def _render_vaa_detail(vaa_res, auto_signal_strategies):
    """VAA 전략 상세."""
    st.subheader("VAA 전략 포트폴리오")
    st.caption("공격자산 4종 13612W 모멘텀 → 양수 최고 1개 선택, 전부 음수 시 방어자산 최고 1개")

    if not vaa_res:
        if "VAA" not in auto_signal_strategies:
            st.info("VAA 비중이 0%라 자동 시그널 계산을 생략했습니다.")
        else:
            st.info("VAA 시그널이 계산되지 않았습니다.")
        return

    if vaa_res.get("error"):
        st.error(str(vaa_res["error"]))
        return

    _vs = vaa_res.get("signal", {})
    _v1, _v2, _v3, _v4 = st.columns(4)
    _tgt_tickers = _vs.get("target_tickers", [])
    _v1.metric("선택 자산", ", ".join(_tgt_tickers) if _tgt_tickers else "-")
    _v2.metric("포지션", "공격" if _vs.get("is_offensive") else "방어")
    _v3.metric("선택 ETF", ", ".join([_fmt_etf_code_name(c) for c in _vs.get("target_kr_codes", [])]))
    _v4.metric("권장 동작", str(vaa_res.get("action", "HOLD")))
    st.info(str(_vs.get("reason", "")))

    _off_sc = _vs.get("offensive_scores", {})
    _def_sc = _vs.get("defensive_scores", {})
    _vaa_kr_map = vaa_res.get("kr_etf_map", {}) or {}
    if _off_sc or _def_sc:
        _score_rows = []
        for t, s in _off_sc.items():
            _kr_code = str(_vaa_kr_map.get(t, "")).strip()
            _score_rows.append({"티커": t, "국내 ETF": _fmt_etf_code_name(_kr_code) if _kr_code else "", "유형": "공격", "모멘텀": round(s * 100, 2)})
        for t, s in _def_sc.items():
            _kr_code = str(_vaa_kr_map.get(t, "")).strip()
            _score_rows.append({"티커": t, "국내 ETF": _fmt_etf_code_name(_kr_code) if _kr_code else "", "유형": "방어", "모멘텀": round(s * 100, 2)})
        st.dataframe(pd.DataFrame(_score_rows), use_container_width=True, hide_index=True)

    _vaa_alloc = vaa_res.get("alloc_df")
    if isinstance(_vaa_alloc, pd.DataFrame) and not _vaa_alloc.empty:
        st.markdown("**목표 배분 vs 현재 보유**")
        st.dataframe(_vaa_alloc, use_container_width=True, hide_index=True)


