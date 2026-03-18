"""연금저축 Tab1: 실시간 모니터링 + 포트폴리오 합산.

잔고 표시, 시그널 캐싱/트리거, 전략별 합산 리밸런싱 테이블을 렌더링한다.
"""

import calendar as _cal
from datetime import date as _d_date, timedelta as _d_td

import streamlit as st
import pandas as pd

from src.utils.formatting import _fmt_etf_code_name, _code_only, _format_kis_holdings_df
from src.utils.kis import _compute_kis_balance_summary
from src.ui.components.pension_strategy_detail import render_strategy_detail_tabs
from src.ui.components.pension_signals import (
    compute_laa_signal,
    compute_dm_signal,
    compute_vaa_signal,
)


def render_monitoring_tab(
    *,
    pen_port_edited,
    active_strategies,
    auto_signal_strategies,
    pen_bal_key,
    pen_bal_ts_key,
    pen_bal_err_key,
    kis_acct,
    kis_prdt,
    refresh_pension_balance,
    trader,
    kr_spy, kr_iwd, kr_gld, kr_ief, kr_qqq, kr_shy,
    kr_etf_map,
    dm_settings,
    vaa_settings,
    pen_bt_start_raw,
    pen_bt_cap,
    pen_bt_start_ts,
    pen_live_auto_backtest,
    pen_api_fallback,
    get_daily_chart,
    get_current_price,
):
    """Tab1 실시간 모니터링 전체 렌더링."""

    # ── 잔고 표시 ──
    _strat_summary = " + ".join(
        [f"{r['strategy']} {r['weight']}%" for _, r in pen_port_edited.iterrows()]
    ) if not pen_port_edited.empty else "LAA 100%"
    st.header("포트폴리오 모니터링")
    st.caption(f"구성: {_strat_summary}")

    refresh_pension_balance(force=False, show_spinner=(pen_bal_key not in st.session_state))

    rcol1, rcol2 = st.columns([1, 5])
    with rcol1:
        if st.button("잔고 새로고침", key="pen_refresh_balance"):
            refresh_pension_balance(force=True, show_spinner=True)
            st.rerun()

    bal = st.session_state.get(pen_bal_key)
    _bal_err = str(st.session_state.get(pen_bal_err_key, "") or "").strip()
    _bal_ts = float(st.session_state.get(pen_bal_ts_key, 0.0) or 0.0)
    if _bal_ts > 0:
        try:
            _bal_ts_text = pd.to_datetime(_bal_ts, unit="s", utc=True).tz_convert("Asia/Seoul").strftime("%m-%d %H:%M:%S")
        except Exception:
            _bal_ts_text = pd.to_datetime(_bal_ts, unit="s").strftime("%m-%d %H:%M:%S")
        st.caption(f"잔고 기준시각: {_bal_ts_text} KST")
    if _bal_err:
        st.info(f"최신 조회 상태: {_bal_err}")

    if not bal:
        st.warning("잔고 조회에 실패했습니다. (응답 None — 네트워크 또는 인증 오류)")
        st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt}")
    elif bal.get("error"):
        st.error(f"잔고 조회 API 오류: [{bal.get('msg_cd', '')}] {bal.get('msg1', '알 수 없는 오류')}")
        st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt} / rt_cd: {bal.get('rt_cd', '')}")
    else:
        _bal_sum = _compute_kis_balance_summary(bal)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("매수 가능금액", f"{_bal_sum['buyable_cash']:,.0f} KRW")
        m2.metric("주식 평가", f"{_bal_sum['stock_eval']:,.0f} KRW")
        m3.metric("총 평가", f"{_bal_sum['total_eval']:,.0f} KRW")
        m4.metric("보유 종목 수", f"{len(_bal_sum['holdings'])}")
        # 자산관리 투입원금 연동
        _pen_invested = float(st.session_state.get("_asset_mgmt_total_invested_pension", 0) or 0)
        if _pen_invested > 0:
            _pen_pnl = _bal_sum['total_eval'] - _pen_invested
            _pen_pnl_pct = (_pen_pnl / _pen_invested * 100) if _pen_invested > 0 else 0.0
            im1, im2, im3 = st.columns(3)
            im1.metric("투입원금", f"{_pen_invested:,.0f} KRW")
            im2.metric("수익금", f"{_pen_pnl:+,.0f} KRW")
            im3.metric("수익률", f"{_pen_pnl_pct:+.2f}%")
        if _bal_sum["holdings"]:
            st.dataframe(_format_kis_holdings_df(_bal_sum["holdings"]), use_container_width=True, hide_index=True)

    st.divider()

    # ── 시그널 캐싱 + 트리거 ──
    res, _dm_res, _vaa_res = _run_signal_caching(
        auto_signal_strategies=auto_signal_strategies,
        kis_acct=kis_acct, kis_prdt=kis_prdt,
        kr_spy=kr_spy, kr_iwd=kr_iwd, kr_gld=kr_gld,
        kr_ief=kr_ief, kr_qqq=kr_qqq, kr_shy=kr_shy,
        kr_etf_map=kr_etf_map,
        dm_settings=dm_settings, vaa_settings=vaa_settings,
        pen_bt_start_raw=pen_bt_start_raw, pen_bt_cap=pen_bt_cap, pen_bt_start_ts=pen_bt_start_ts,
        pen_live_auto_backtest=pen_live_auto_backtest, pen_api_fallback=pen_api_fallback,
        pen_bal_key=pen_bal_key,
        trader=trader,
        get_daily_chart=get_daily_chart, get_current_price=get_current_price,
        pen_port_edited=pen_port_edited,
    )

    # ── 전체 포트폴리오 합산 ──
    _render_portfolio_aggregation(
        active_strategies=active_strategies,
        res=res, dm_res=_dm_res, vaa_res=_vaa_res,
        pen_bal_key=pen_bal_key,
        get_current_price=get_current_price,
        get_daily_chart=get_daily_chart,
    )

    # ── 전략별 상세 하위탭 ──
    render_strategy_detail_tabs(
        active_strategies=active_strategies,
        auto_signal_strategies=auto_signal_strategies,
        laa_res=res, dm_res=_dm_res, vaa_res=_vaa_res,
        bal=bal, pen_bt_start_raw=pen_bt_start_raw, pen_bt_cap=pen_bt_cap,
        dm_settings=dm_settings,
    )


# ---------------------------------------------------------------------------
# 시그널 캐싱 + 트리거
# ---------------------------------------------------------------------------

def _run_signal_caching(
    *,
    auto_signal_strategies,
    kis_acct, kis_prdt,
    kr_spy, kr_iwd, kr_gld, kr_ief, kr_qqq, kr_shy,
    kr_etf_map, dm_settings, vaa_settings,
    pen_bt_start_raw, pen_bt_cap, pen_bt_start_ts,
    pen_live_auto_backtest, pen_api_fallback,
    pen_bal_key, trader,
    get_daily_chart, get_current_price,
    pen_port_edited,
):
    """시그널 계산 결과를 session_state에 캐싱하고, 캐시 미스 시 재계산한다."""

    # ── LAA ──
    pen_sig_params = {
        "acct": str(kis_acct), "prdt": str(kis_prdt),
        "kr_spy": str(kr_spy), "kr_iwd": str(kr_iwd),
        "kr_gld": str(kr_gld), "kr_ief": str(kr_ief),
        "kr_qqq": str(kr_qqq), "kr_shy": str(kr_shy),
        "bt_start": str(pen_bt_start_raw), "bt_cap": float(pen_bt_cap),
    }
    res = None
    if "LAA" in auto_signal_strategies:
        if st.session_state.get("pen_signal_result") is None or st.session_state.get("pen_signal_params") != pen_sig_params:
            with st.spinner("LAA 시그널을 자동 계산하는 중입니다..."):
                st.session_state["pen_signal_result"] = compute_laa_signal(
                    source_codes={
                        "SPY": _code_only(kr_spy or kr_iwd or "360750"),
                        "IWD": _code_only(kr_iwd), "GLD": _code_only(kr_gld),
                        "IEF": _code_only(kr_ief), "QQQ": _code_only(kr_qqq),
                        "SHY": _code_only(kr_shy),
                    },
                    kr_etf_map=kr_etf_map,
                    get_daily_chart=get_daily_chart,
                    get_current_price=get_current_price,
                    bal=st.session_state.get(pen_bal_key) or trader.get_balance() or {},
                    pen_bt_start_raw=pen_bt_start_raw, pen_bt_cap=pen_bt_cap,
                    pen_bt_start_ts=pen_bt_start_ts,
                    pen_live_auto_backtest=pen_live_auto_backtest,
                    pen_api_fallback=pen_api_fallback,
                )
                st.session_state["pen_signal_params"] = pen_sig_params
                _pen_res = st.session_state["pen_signal_result"]
                if isinstance(_pen_res, dict) and _pen_res.get("balance"):
                    st.session_state[pen_bal_key] = _pen_res["balance"]
        res = st.session_state.get("pen_signal_result")
    else:
        st.session_state.pop("pen_signal_result", None)
        st.session_state.pop("pen_signal_params", None)

    # ── 듀얼모멘텀 ──
    _dm_res = None
    if "듀얼모멘텀" in auto_signal_strategies and dm_settings:
        _dm_data_source = st.radio(
            "듀얼모멘텀 데이터 소스",
            options=["US (미국 원본)", "KR (국내 ETF)"],
            index=0,
            horizontal=True,
            key="pen_dm_data_source",
        )
        _dm_src_key = "US" if "US" in _dm_data_source else "KR"
        _dm_weights = dm_settings.get("momentum_weights", {})
        _dm_params = {
            "acct": str(kis_acct), "prdt": str(kis_prdt),
            "off": tuple(dm_settings.get("offensive", [])),
            "def": tuple(dm_settings.get("defensive", [])),
            "canary": tuple(dm_settings.get("canary", [])),
            "lookback": int(dm_settings.get("lookback", 12)),
            "td": int(dm_settings.get("trading_days_per_month", 22)),
            "w1": float(_dm_weights.get("m1", 12.0)),
            "w3": float(_dm_weights.get("m3", 4.0)),
            "w6": float(_dm_weights.get("m6", 2.0)),
            "w12": float(_dm_weights.get("m12", 1.0)),
            "kr_spy": str((dm_settings.get("kr_etf_map", {}) or {}).get("SPY", "")),
            "kr_efa": str((dm_settings.get("kr_etf_map", {}) or {}).get("EFA", "")),
            "kr_agg": str((dm_settings.get("kr_etf_map", {}) or {}).get("AGG", "")),
            "kr_bil": str((dm_settings.get("kr_etf_map", {}) or {}).get("BIL", "")),
            "bt_start": str(pen_bt_start_raw), "bt_cap": float(pen_bt_cap),
            "data_source": _dm_src_key,
        }
        if st.session_state.get("pen_dm_signal_result") is None or st.session_state.get("pen_dm_signal_params") != _dm_params:
            with st.spinner("듀얼모멘텀 시그널을 자동 계산하는 중입니다..."):
                st.session_state["pen_dm_signal_result"] = compute_dm_signal(
                    dm_settings=dm_settings,
                    get_daily_chart=get_daily_chart,
                    get_current_price=get_current_price,
                    bal=st.session_state.get(pen_bal_key) or trader.get_balance() or {},
                    pen_port_edited=pen_port_edited,
                    pen_bt_start_raw=pen_bt_start_raw, pen_bt_cap=pen_bt_cap,
                    pen_bt_start_ts=pen_bt_start_ts,
                    pen_live_auto_backtest=pen_live_auto_backtest,
                    data_source=_dm_src_key,
                )
                st.session_state["pen_dm_signal_params"] = _dm_params
                _dm_res_cache = st.session_state["pen_dm_signal_result"]
                if isinstance(_dm_res_cache, dict) and _dm_res_cache.get("balance"):
                    st.session_state[pen_bal_key] = _dm_res_cache["balance"]
        _dm_res = st.session_state.get("pen_dm_signal_result")
    else:
        st.session_state.pop("pen_dm_signal_result", None)
        st.session_state.pop("pen_dm_signal_params", None)

    # ── VAA ──
    _vaa_res = None
    if "VAA" in auto_signal_strategies:
        _vaa_sig_params = {"acct": str(kis_acct), "vaa_settings": str(vaa_settings)}
        if (st.session_state.get("pen_vaa_signal_params") != _vaa_sig_params
                or "pen_vaa_signal_result" not in st.session_state):
            with st.spinner("VAA 시그널 계산 중..."):
                st.session_state["pen_vaa_signal_result"] = compute_vaa_signal(
                    vaa_settings=vaa_settings,
                    get_current_price=get_current_price,
                    bal=st.session_state.get(pen_bal_key) or {},
                    pen_port_edited=pen_port_edited,
                )
                st.session_state["pen_vaa_signal_params"] = _vaa_sig_params
        _vaa_res = st.session_state.get("pen_vaa_signal_result")
    else:
        st.session_state.pop("pen_vaa_signal_result", None)
        st.session_state.pop("pen_vaa_signal_params", None)

    return res, _dm_res, _vaa_res


# ---------------------------------------------------------------------------
# 전체 포트폴리오 합산
# ---------------------------------------------------------------------------

def _render_portfolio_aggregation(
    *,
    active_strategies,
    res, dm_res, vaa_res,
    pen_bal_key,
    get_current_price,
    get_daily_chart,
):
    """전략별 시그널 결과를 합산하여 리밸런싱 테이블을 표시한다."""
    st.divider()
    st.subheader("전체 포트폴리오 합산")

    _strat_col_map = {}
    if "LAA" in active_strategies:
        _strat_col_map["LAA"] = "LAA 목표(주)"
    if "듀얼모멘텀" in active_strategies:
        _strat_col_map["DM"] = "DM 목표(주)"
    if "VAA" in active_strategies:
        _strat_col_map["VAA"] = "VAA 목표(주)"
    _combined_rows = {}

    def _ensure_combined_row(code, etf_name, cur_qty):
        if code not in _combined_rows:
            _combined_rows[code] = {"ETF": etf_name, "현재수량(주)": int(cur_qty)}
            for _col in _strat_col_map.values():
                _combined_rows[code][_col] = 0
        _combined_rows[code]["현재수량(주)"] = max(_combined_rows[code]["현재수량(주)"], int(cur_qty))

    # LAA
    _laa_action = None
    if res and not res.get("error"):
        _laa_action = res.get("action")
        _laa_df = res.get("alloc_df")
        if isinstance(_laa_df, pd.DataFrame) and not _laa_df.empty:
            for _, row in _laa_df.iterrows():
                code = str(row.get("ETF 코드", "")).strip()
                if not code:
                    continue
                _ensure_combined_row(code, row.get("ETF", _fmt_etf_code_name(code)), row.get("현재수량(주)", 0))
                if "LAA 목표(주)" in _combined_rows[code]:
                    _combined_rows[code]["LAA 목표(주)"] = int(row.get("목표수량(주)", 0))

    # DM
    _dm_action_val = None
    if dm_res and not dm_res.get("error"):
        _dm_action_val = dm_res.get("action")
        _dm_df = dm_res.get("expected_rebalance_df")
        if isinstance(_dm_df, pd.DataFrame) and not _dm_df.empty:
            for _, row in _dm_df.iterrows():
                code = str(row.get("ETF 코드", "")).strip()
                if not code:
                    continue
                _ensure_combined_row(code, row.get("국내 ETF", _fmt_etf_code_name(code)), row.get("현재수량(주)", 0))
                if "DM 목표(주)" in _combined_rows[code]:
                    _combined_rows[code]["DM 목표(주)"] = int(row.get("목표수량(주,버림)", 0))

    # VAA
    _vaa_action_val = None
    if vaa_res and not vaa_res.get("error"):
        _vaa_action_val = vaa_res.get("action")
        _vaa_df = vaa_res.get("alloc_df")
        if isinstance(_vaa_df, pd.DataFrame) and not _vaa_df.empty:
            for _, row in _vaa_df.iterrows():
                code = str(row.get("ETF 코드", "")).strip()
                if not code:
                    continue
                _ensure_combined_row(code, row.get("ETF", _fmt_etf_code_name(code)), row.get("현재수량(주)", 0))
                if "VAA 목표(주)" in _combined_rows[code]:
                    _combined_rows[code]["VAA 목표(주)"] = int(row.get("목표수량(주)", 0))

    if not _combined_rows:
        st.info("시그널 계산 결과가 없습니다. 잔고를 새로고침해주세요.")
        st.session_state.pop("pen_combined_rebal_data", None)
        return

    # 합산 테이블 생성
    _combo_list = []
    _total_buy_count = 0
    _total_sell_count = 0
    _rebal_strategies = 0
    if _laa_action == "REBALANCE":
        _rebal_strategies += 1
    if _dm_action_val in ("REBALANCE", "BUY"):
        _rebal_strategies += 1
    if _vaa_action_val == "REBALANCE":
        _rebal_strategies += 1

    _bal_combo = st.session_state.get(pen_bal_key) or {}
    _holdings_combo = _bal_combo.get("holdings", []) or []
    _cash_combo = float(_bal_combo.get("cash", 0.0) or 0.0)
    _total_eval_combo = float(_bal_combo.get("total_eval", 0.0) or 0.0)
    if _total_eval_combo <= 0:
        _total_eval_combo = _cash_combo + sum(float(_h.get("eval_amt", 0.0) or 0.0) for _h in _holdings_combo)
    _total_eval_combo = max(_total_eval_combo, 1.0)

    _hold_eval_map = {}
    _hold_px_map = {}
    _hold_qty_map = {}
    for _h in _holdings_combo:
        _code = str(_h.get("code", "")).strip()
        if not _code:
            continue
        _hold_eval_map[_code] = _hold_eval_map.get(_code, 0.0) + float(_h.get("eval_amt", 0.0) or 0.0)
        _hold_qty_map[_code] = _hold_qty_map.get(_code, 0) + int(float(_h.get("qty", 0) or 0))
        _cur_px = float(_h.get("cur_price", 0.0) or 0.0)
        if _cur_px > 0:
            _hold_px_map[_code] = _cur_px

    # 레거시 ETF 평가금액 합산 (132030 → 411060)
    from src.constants import ETF_LEGACY_MERGE
    _legacy_sell_qty = {}  # {legacy_code: qty} — 매도 우선순위용
    for _leg_code, _pri_code in ETF_LEGACY_MERGE.items():
        _leg_eval = _hold_eval_map.pop(_leg_code, 0.0)
        _leg_qty = _hold_qty_map.pop(_leg_code, 0)
        if _leg_eval > 0 or _leg_qty > 0:
            _hold_eval_map[_pri_code] = _hold_eval_map.get(_pri_code, 0.0) + _leg_eval
            _hold_qty_map[_pri_code] = _hold_qty_map.get(_pri_code, 0) + _leg_qty
            _legacy_sell_qty[_leg_code] = _leg_qty

    # 레거시 합산 수량으로 현재수량 보정
    for _code in _combined_rows:
        _merged_qty = _hold_qty_map.get(_code, 0)
        if _merged_qty > _combined_rows[_code]["현재수량(주)"]:
            _combined_rows[_code]["현재수량(주)"] = _merged_qty

    _combo_price_cache = {}

    def _resolve_combo_price(_code_str, _cur_qty, _cur_eval):
        _k = str(_code_str).strip()
        if not _k:
            return 0.0
        if _k in _combo_price_cache:
            return float(_combo_price_cache[_k])
        _p = float(_hold_px_map.get(_k, 0.0) or 0.0)
        if _p <= 0:
            try:
                _p = float(get_current_price(_k) or 0.0)
            except Exception:
                _p = 0.0
        if _p <= 0 and _cur_qty > 0:
            _p = float(_cur_eval) / float(_cur_qty)
        if _p <= 0:
            try:
                _ch = get_daily_chart(_k, count=5)
                if _ch is not None and not _ch.empty:
                    if "close" in _ch.columns:
                        _p = float(_ch["close"].iloc[-1])
                    elif "Close" in _ch.columns:
                        _p = float(_ch["Close"].iloc[-1])
            except Exception:
                _p = 0.0
        _combo_price_cache[_k] = float(_p if _p > 0 else 0.0)
        return float(_combo_price_cache[_k])

    for code, info in _combined_rows.items():
        _total_target = sum(info.get(col, 0) for col in _strat_col_map.values())
        _cur = info["현재수량(주)"]
        _buy = max(_total_target - _cur, 0)
        _sell = max(_cur - _total_target, 0)
        _cur_eval = float(_hold_eval_map.get(code, 0.0) or 0.0)
        _px = float(_resolve_combo_price(code, _cur, _cur_eval))
        if _cur_eval <= 0 and _px > 0 and _cur > 0:
            _cur_eval = _px * _cur
        _target_eval = float(_total_target) * _px if _px > 0 else 0.0
        _cur_weight = (_cur_eval / _total_eval_combo) * 100.0
        _target_weight = (_target_eval / _total_eval_combo) * 100.0
        if _buy > 0:
            _order_status = "매수"
            _total_buy_count += 1
        elif _sell > 0:
            _order_status = "매도"
            _total_sell_count += 1
        else:
            _order_status = "유지"
        _row_data = {"ETF": info["ETF"], "ETF코드": str(code), "현재수량(주)": _cur, "현재 비중(%)": round(_cur_weight, 2)}
        for _col in _strat_col_map.values():
            _row_data[_col] = info.get(_col, 0)
        _row_data["합산 목표(주)"] = _total_target
        _row_data["목표 비중(%)"] = round(_target_weight, 2)
        _row_data["매수 예정(주)"] = _buy if _buy > 0 else 0
        _row_data["매도 예정(주)"] = _sell if _sell > 0 else 0
        _row_data["주문 상태"] = _order_status
        _combo_list.append(_row_data)

    # 다음 리밸런싱 예정일 계산 (매월 마지막 영업일)
    _today = _d_date.today()

    def _next_rebal_date(_ref):
        y, m = _ref.year, _ref.month
        last_day = _cal.monthrange(y, m)[1]
        dt = _d_date(y, m, last_day)
        while dt.weekday() >= 5:
            dt -= _d_td(days=1)
        if dt >= _ref:
            return dt
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        last_day = _cal.monthrange(y, m)[1]
        dt = _d_date(y, m, last_day)
        while dt.weekday() >= 5:
            dt -= _d_td(days=1)
        return dt

    _next_rebal = _next_rebal_date(_today)
    _days_left = (_next_rebal - _today).days

    _sm1, _sm2, _sm3, _sm4 = st.columns(4)
    _sm1.metric("다음 리밸런싱", f"{_next_rebal.strftime('%Y-%m-%d')}", delta=f"D-{_days_left}" if _days_left > 0 else "오늘")
    _sm2.metric("리밸런싱 필요 전략", f"{_rebal_strategies}개")
    _sm3.metric("매수 예정 종목", f"{_total_buy_count}개")
    _sm4.metric("매도 예정 종목", f"{_total_sell_count}개")
    # 레거시 ETF 우선 매도 분리 (132030 보유 시 411060 매도를 132030 우선으로 분할)
    _final_combo = []
    for _row in _combo_list:
        _rc = str(_row.get("ETF코드", "")).strip()
        _sell_q = int(_row.get("매도 예정(주)", 0))
        if _rc in ETF_LEGACY_MERGE.values() and _sell_q > 0 and _legacy_sell_qty:
            # 현행 ETF 매도 → 레거시 우선 매도 분할
            for _leg_c, _leg_q in list(_legacy_sell_qty.items()):
                if ETF_LEGACY_MERGE.get(_leg_c) != _rc or _leg_q <= 0:
                    continue
                _leg_sell = min(_leg_q, _sell_q)
                if _leg_sell > 0:
                    _final_combo.append({
                        **_row,
                        "ETF": _fmt_etf_code_name(_leg_c) + " (우선매도)",
                        "ETF코드": _leg_c,
                        "매도 예정(주)": _leg_sell,
                        "매수 예정(주)": 0,
                        "주문 상태": "매도",
                    })
                    _sell_q -= _leg_sell
            if _sell_q > 0:
                _final_combo.append({**_row, "매도 예정(주)": _sell_q})
            else:
                _final_combo.append({**_row, "매도 예정(주)": 0, "주문 상태": "유지"})
        else:
            _final_combo.append(_row)

    st.dataframe(pd.DataFrame(_final_combo), use_container_width=True, hide_index=True)
    st.session_state["pen_combined_rebal_data"] = _final_combo
