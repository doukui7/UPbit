"""ISA 라이브 백테스트 탭 (tab_i7).

render_isa_live_tab(ctx) 함수 하나를 외부에 노출한다.
ctx keys:
  isa_etf_code, isa_trend_etf_code, wdr_eval_mode, get_daily_chart
"""
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import pickle
import zlib
import logging
import os
import itertools
import time
import plotly.graph_objects as go

from concurrent.futures import ThreadPoolExecutor, as_completed
from src.strategy.widaeri import WDRStrategy
from src.utils.db_manager import DBManager
import src.engine.data_cache as _dc
from src.utils.formatting import _fmt_etf_code_name
from src.ui.components.performance import _render_performance_analysis, _apply_return_hover_format


def render_isa_live_tab(ctx: dict):
    isa_etf_code = ctx["isa_etf_code"]
    isa_trend_etf_code = ctx["isa_trend_etf_code"]
    wdr_eval_mode = ctx["wdr_eval_mode"]
    _get_isa_daily_chart = ctx["get_daily_chart"]

    st.header("라이브 백테스트")
    st.caption("상단은 사전 계산 범위, 하단은 전체 지표 다이얼 적용값입니다.")

    def _ss(_k, _d=None):
        _v = st.session_state.get(_k, _d)
        return _d if _v is None else _v

    def _to_date(_v, _fallback):
        try:
            return pd.to_datetime(_v).date()
        except Exception:
            return _fallback

    def _snap_step(_v: float, _step: float, _vmin: float, _vmax: float) -> float:
        _s = float(_step) if float(_step) > 0 else 1.0
        _x = round(round(float(_v) / _s) * _s, 4)
        return float(min(float(_vmax), max(float(_vmin), _x)))

    def _grid_values(_vmin: float, _vmax: float, _step: float) -> list[float]:
        _step = max(float(_step), 1e-9)
        if float(_vmax) < float(_vmin):
            _vmin, _vmax = _vmax, _vmin
        _n = int(np.floor((float(_vmax) - float(_vmin)) / _step + 1e-9)) + 1
        vals = [round(float(_vmin) + i * _step, 4) for i in range(max(1, _n))]
        if vals[-1] < float(_vmax) - 1e-9:
            vals.append(round(float(_vmax), 4))
        return vals

    # 탭6에서 전달된 적용값 반영 (위젯 생성 전)
    _apply_token = st.session_state.get("isa_live_apply_token")
    if _apply_token and st.session_state.get("_isa_live_applied_token") != _apply_token:
        _payload = st.session_state.get("isa_live_apply_payload")
        if isinstance(_payload, dict):
            for _k, _v in _payload.items():
                st.session_state[_k] = _v
        st.session_state["_isa_live_applied_token"] = _apply_token

    _live_listing_date = _dc.get_wdr_trade_listing_date(str(isa_etf_code))
    _live_min_start = pd.to_datetime(_live_listing_date).date() if _live_listing_date else pd.to_datetime("2012-01-01").date()

    # ── 1) 사이드바 시작일(isa_start_date)과 연동 (강제 동기화) ──
    def _sync_ratio_from_start_date(_date):
        """DB에서 시작일 기준 추천 비중을 가져와 세션에 저장"""
        _rec = _dc.get_wdr_v10_stock_ratio(str(isa_etf_code), _date)
        if _rec:
            st.session_state["isa_live_ratio"] = round(float(_rec["stock_ratio"]) * 100.0, 2)
        else:
            _default_ratio = _dc.get_wdr_default_initial_stock_ratio(str(isa_etf_code))
            if _default_ratio is not None:
                st.session_state["isa_live_ratio"] = round(float(_default_ratio) * 100.0, 2)

    def _on_live_start_change():
        """시작일 위젯 직접 변경 시 실행될 콜백"""
        _new_date = st.session_state.get("isa_live_start")
        if _new_date:
            _sync_ratio_from_start_date(_new_date)

    _sidebar_start = ctx.get("isa_start_date")
    if _sidebar_start and st.session_state.get("_isa_sidebar_start_sync") != str(_sidebar_start):
        st.session_state["_isa_sidebar_start_sync"] = str(_sidebar_start)
        _new_start = pd.to_datetime(_sidebar_start).date()
        st.session_state["isa_live_start"] = _new_start
        _sync_ratio_from_start_date(_new_start)

    # 2. ETF가 바뀌면 다시 갱신
    _live_etf_sync_key = "isa_live_etf_sync"
    if st.session_state.get(_live_etf_sync_key) != str(isa_etf_code):
        st.session_state[_live_etf_sync_key] = str(isa_etf_code)
        _start_to_use = pd.to_datetime(_sidebar_start).date() if _sidebar_start else _live_min_start
        st.session_state["isa_live_start"] = _start_to_use
        
        # 범위 초기화 및 비중 연동
        st.session_state["isa_live_ratio_min"] = 0.0
        st.session_state["isa_live_ratio_max"] = 100.0
        st.session_state["isa_live_ratio_step"] = 2.0
        
        _rec = _dc.get_wdr_v10_stock_ratio(str(isa_etf_code), _start_to_use)
        if _rec:
            st.session_state["isa_live_ratio"] = round(float(_rec["stock_ratio"]) * 100.0, 2)
        else:
            _default_ratio = _dc.get_wdr_default_initial_stock_ratio(str(isa_etf_code))
            if _default_ratio is not None:
                st.session_state["isa_live_ratio"] = round(float(_default_ratio) * 100.0, 2)

    # 3) 사이드바 초기 주식 비중(isa_initial_ratio)과 연동 (변경 시 실시간 반영)
    _sidebar_ratio = st.session_state.get("isa_initial_ratio")
    if _sidebar_ratio is not None:
        _sync_key = "_isa_sidebar_ratio_last_sync"
        if st.session_state.get(_sync_key) != float(_sidebar_ratio):
            st.session_state[_sync_key] = float(_sidebar_ratio)
            st.session_state["isa_live_ratio"] = float(_sidebar_ratio)


    _live_default_start = pd.to_datetime(_ss("isa_live_start", _live_min_start)).date()
    if _live_default_start < _live_min_start:
        _live_default_start = _live_min_start
        st.session_state["isa_live_start"] = _live_min_start

    _live_defaults = {
        "isa_live_search_mode": "그리드 탐색",
        "isa_live_random_trials": 2000,
        "isa_live_pre_max_combos": 120000,
        "isa_live_pre_workers": max(2, min((os.cpu_count() or 4), 16)),
        "isa_live_ov_min": 0.0,
        "isa_live_ov_max": 20.0,
        "isa_live_ov_step": 0.5,
        "isa_live_un_min": -20.0,
        "isa_live_un_max": 0.0,
        "isa_live_un_step": 0.5,
        "isa_live_ratio_min": 0.0,
        "isa_live_ratio_max": 100.0,
        "isa_live_ratio_step": 2.0,

        "isa_live_use_anim": True,
        "isa_live_anim_step": 24,
        "isa_live_anim_delay_ms": 60,
        "isa_live_sell_ov": 100.0,
        "isa_live_sell_neu": 66.7,
        "isa_live_sell_un": 60.0,
        "isa_live_buy_ov": 66.7,
        "isa_live_buy_neu": 66.7,
        "isa_live_buy_un": 120.0,
        "isa_live_sell_sov": 150.0,
        "isa_live_sell_sun": 33.0,
        "isa_live_buy_sov": 33.0,
        "isa_live_buy_sun": 200.0,
        "isa_live_sell_ov_min": 60.0,
        "isa_live_sell_ov_max": 150.0,
        "isa_live_sell_ov_step": 10.0,
        "isa_live_sell_neu_min": 30.0,
        "isa_live_sell_neu_max": 100.0,
        "isa_live_sell_neu_step": 10.0,
        "isa_live_sell_un_min": 30.0,
        "isa_live_sell_un_max": 100.0,
        "isa_live_sell_un_step": 10.0,
        "isa_live_buy_ov_min": 30.0,
        "isa_live_buy_ov_max": 100.0,
        "isa_live_buy_ov_step": 10.0,
        "isa_live_buy_neu_min": 30.0,
        "isa_live_buy_neu_max": 100.0,
        "isa_live_buy_neu_step": 10.0,
        "isa_live_buy_un_min": 60.0,
        "isa_live_buy_un_max": 200.0,
        "isa_live_buy_un_step": 10.0,
        "isa_live_sell_sov_min": 80.0,
        "isa_live_sell_sov_max": 200.0,
        "isa_live_sell_sov_step": 10.0,
        "isa_live_sell_sun_min": 10.0,
        "isa_live_sell_sun_max": 100.0,
        "isa_live_sell_sun_step": 10.0,
        "isa_live_buy_sov_min": 10.0,
        "isa_live_buy_sov_max": 100.0,
        "isa_live_buy_sov_step": 10.0,
        "isa_live_buy_sun_min": 100.0,
        "isa_live_buy_sun_max": 300.0,
        "isa_live_buy_sun_step": 10.0,
        "isa_live_min_stock_ratio": 10.0,
        "isa_live_max_stock_ratio": 100.0,
    }
    for _k, _v in _live_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    st.caption("상단은 사전 계산 범위 설정입니다. 실제 단일값은 아래 '라이브 다이얼 적용값'에서 조절합니다.")
    top_l, top_r = st.columns(2)
    with top_l:
        _eval_options = [3, 5]
        _eval_default = int(_ss("isa_live_eval_mode", int(wdr_eval_mode)))
        if _eval_default not in _eval_options:
            _eval_default = int(wdr_eval_mode)
        live_eval_mode = st.selectbox(
            "평가 시스템",
            _eval_options,
            format_func=lambda x: f"{x}단계",
            key="isa_live_eval_mode",
        )
        live_start = st.date_input(
            "시작일",
            min_value=_live_min_start,
            key="isa_live_start",
            on_change=_on_live_start_change,
        )
        live_end = st.date_input(
            "종료일",
            key="isa_live_end",
        )
        if _live_listing_date:
            st.caption(f"매매 ETF 첫 거래일(상장 후 첫 금요일): {_live_listing_date}")

        live_cap = int(st.number_input(
            "초기 자본(원)",
            min_value=1_000_000,
            max_value=1_000_000_000,
            step=1_000_000,
            key="isa_live_cap",
        ))
        live_fee = float(st.number_input(
            "매매 수수료(%)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            key="isa_live_fee",
        ))
        live_search_mode = st.selectbox("탐색 방식", ["그리드 탐색", "랜덤 탐색", "Optuna (베이지안)"], key="isa_live_search_mode")
        _trial_label = "랜덤 탐색 횟수" if live_search_mode == "랜덤 탐색" else "탐색 횟수"
        live_random_trials = int(st.number_input(
            _trial_label,
            min_value=100,
            max_value=10_000_000,
            step=100,
            key="isa_live_random_trials",
            disabled=(live_search_mode == "그리드 탐색"),
        ))
        live_pre_max = int(st.number_input(
            "사전 계산 최대 조합",
            min_value=100,
            max_value=10_000_000,
            step=100,
            key="isa_live_pre_max_combos",
        ))

    with top_r:
        live_pre_workers = int(st.number_input(
            "사전 계산 워커 수",
            min_value=1,
            max_value=64,
            step=1,
            key="isa_live_pre_workers",
        ))
        live_auto = False  # 자동 계산 비활성화 (수동 실행 버튼으로만 실행)
        live_use_anim = st.checkbox("라이브 애니메이션 사용", key="isa_live_use_anim")
        live_anim_step = int(st.number_input("재생 프레임 단위", min_value=1, max_value=200, step=1, key="isa_live_anim_step"))
        live_anim_delay_ms = int(st.number_input("프레임 지연(ms)", min_value=0, max_value=2000, step=10, key="isa_live_anim_delay_ms"))
        live_run_btn = st.button("라이브 계산 실행", type="primary", key="isa_live_run")

    st.subheader("사전 계산 범위")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.number_input("고평가 최소 (%)", value=float(_ss("isa_live_ov_min", 0.0)), step=0.5, key="isa_live_ov_min")
        st.number_input("고평가 최대 (%)", value=float(_ss("isa_live_ov_max", 20.0)), step=0.5, key="isa_live_ov_max")
        st.number_input("고평가 스텝", min_value=0.1, value=float(_ss("isa_live_ov_step", 1.0)), step=0.1, key="isa_live_ov_step")
    with g2:
        st.number_input("저평가 최소 (%)", value=float(_ss("isa_live_un_min", -20.0)), step=0.5, key="isa_live_un_min")
        st.number_input("저평가 최대 (%)", value=float(_ss("isa_live_un_max", 0.0)), step=0.5, key="isa_live_un_max")
        st.number_input("저평가 스텝", min_value=0.1, value=float(_ss("isa_live_un_step", 1.0)), step=0.1, key="isa_live_un_step")
    with g3:
        st.number_input("초기 주식 비중 최소 (%)", min_value=0.0, max_value=100.0, value=float(_ss("isa_live_ratio_min", 0.0)), step=1.0, key="isa_live_ratio_min")
        st.number_input("초기 주식 비중 최대 (%)", min_value=0.0, max_value=100.0, value=float(_ss("isa_live_ratio_max", 100.0)), step=1.0, key="isa_live_ratio_max")
        st.number_input("초기 주식 비중 스텝", min_value=0.1, max_value=100.0, value=float(_ss("isa_live_ratio_step", 2.0)), step=0.1, key="isa_live_ratio_step")

    st.subheader("평가별 매수/매도 비율 사전 계산 범위(%)")

    def _range_triplet(_title: str, _key_min: str, _key_max: str, _key_step: str, _dmin: float, _dmax: float, _dstep: float):
        st.markdown(f"**{_title}**")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.number_input("최소(%)", min_value=0.0, max_value=300.0, step=1.0, key=_key_min, value=float(_ss(_key_min, _dmin)))
        with t2:
            st.number_input("최대(%)", min_value=0.0, max_value=300.0, step=1.0, key=_key_max, value=float(_ss(_key_max, _dmax)))
        with t3:
            st.number_input("스텝", min_value=0.1, max_value=100.0, step=0.1, key=_key_step, value=float(_ss(_key_step, _dstep)))

    r_ov_l, r_ov_r = st.columns(2)
    with r_ov_l:
        _range_triplet("고평가 (매도)", "isa_live_sell_ov_min", "isa_live_sell_ov_max", "isa_live_sell_ov_step", 60.0, 150.0, 10.0)
    with r_ov_r:
        _range_triplet("고평가 (매수)", "isa_live_buy_ov_min", "isa_live_buy_ov_max", "isa_live_buy_ov_step", 30.0, 100.0, 10.0)

    r_n_l, r_n_r = st.columns(2)
    with r_n_l:
        _range_triplet("중립 (매도)", "isa_live_sell_neu_min", "isa_live_sell_neu_max", "isa_live_sell_neu_step", 30.0, 100.0, 10.0)
    with r_n_r:
        _range_triplet("중립 (매수)", "isa_live_buy_neu_min", "isa_live_buy_neu_max", "isa_live_buy_neu_step", 30.0, 100.0, 10.0)

    r_un_l, r_un_r = st.columns(2)
    with r_un_l:
        _range_triplet("저평가 (매도)", "isa_live_sell_un_min", "isa_live_sell_un_max", "isa_live_sell_un_step", 30.0, 100.0, 10.0)
    with r_un_r:
        _range_triplet("저평가 (매수)", "isa_live_buy_un_min", "isa_live_buy_un_max", "isa_live_buy_un_step", 60.0, 200.0, 10.0)

    if int(live_eval_mode) == 5:
        r_sov_l, r_sov_r = st.columns(2)
        with r_sov_l:
            _range_triplet("초고평가 (매도)", "isa_live_sell_sov_min", "isa_live_sell_sov_max", "isa_live_sell_sov_step", 80.0, 200.0, 10.0)
        with r_sov_r:
            _range_triplet("초고평가 (매수)", "isa_live_buy_sov_min", "isa_live_buy_sov_max", "isa_live_buy_sov_step", 10.0, 100.0, 10.0)

        r_sun_l, r_sun_r = st.columns(2)
        with r_sun_l:
            _range_triplet("초저평가 (매도)", "isa_live_sell_sun_min", "isa_live_sell_sun_max", "isa_live_sell_sun_step", 10.0, 100.0, 10.0)
        with r_sun_r:
            _range_triplet("초저평가 (매수)", "isa_live_buy_sun_min", "isa_live_buy_sun_max", "isa_live_buy_sun_step", 100.0, 300.0, 10.0)
    else:
        st.caption("3단계에서는 초고평가/초저평가 비율 범위를 사용하지 않습니다.")

    def _dial_slider(_label: str, _value_key: str, _min_key: str, _max_key: str, _step_key: str) -> float:
        _mn_raw = float(_ss(_min_key, 0.0))
        _mx_raw = float(_ss(_max_key, 100.0))
        _mn = float(min(_mn_raw, _mx_raw))
        _mx = float(max(_mn_raw, _mx_raw))
        
        # Streamlit 에러 방지: min과 max가 같거나 min이 max보다 크면 한쪽을 살짝 조정
        if _mn >= _mx:
            _mx = _mn + 0.1
            
        _st = float(_ss(_step_key, 1.0))
        _v0 = _snap_step(float(_ss(_value_key, _mn)), max(0.1, _st), _mn, _mx)
        st.session_state[_value_key] = _v0  # 슬라이더 렌더링 전 강제 클램프
        
        return float(st.slider(
            _label,
            min_value=_mn,
            max_value=_mx,
            step=float(max(0.1, _st)),
            key=_value_key,
        ))

    st.subheader("라이브 다이얼 적용값")
    # --- Row 1: General Thresholds ---
    tc1, tc2, tc3 = st.columns(3)
    with tc1: _dial_slider("고평가 임계값 (%)", "isa_live_ov", "isa_live_ov_min", "isa_live_ov_max", "isa_live_ov_step")
    with tc2: _dial_slider("저평가 임계값 (%)", "isa_live_un", "isa_live_un_min", "isa_live_un_max", "isa_live_un_step")
    with tc3: _dial_slider("초기 주식 비중 (%)", "isa_live_ratio", "isa_live_ratio_min", "isa_live_ratio_max", "isa_live_ratio_step")

    st.markdown("---")

    # --- Row 2~N: paired Sell/Buy Ratios ---
    live_sell_sov = live_sell_sun = live_buy_sov = live_buy_sun = None

    # Overvalue Pair
    rc1, rc2, rc3 = st.columns(3)
    with rc1: live_sell_ov = _dial_slider("매도 고평가(%)", "isa_live_sell_ov", "isa_live_sell_ov_min", "isa_live_sell_ov_max", "isa_live_sell_ov_step")
    with rc2: live_buy_ov = _dial_slider("매수 고평가(%)", "isa_live_buy_ov", "isa_live_buy_ov_min", "isa_live_buy_ov_max", "isa_live_buy_ov_step")
    with rc3:
        if int(live_eval_mode) == 5:
            live_sell_sov = _dial_slider("매도 초고평가(%)", "isa_live_sell_sov", "isa_live_sell_sov_min", "isa_live_sell_sov_max", "isa_live_sell_sov_step")
        else:
            st.empty()

    # Neutral Pair
    rc1, rc2, rc3 = st.columns(3)
    with rc1: live_sell_neu = _dial_slider("매도 중립(%)", "isa_live_sell_neu", "isa_live_sell_neu_min", "isa_live_sell_neu_max", "isa_live_sell_neu_step")
    with rc2: live_buy_neu = _dial_slider("매수 중립(%)", "isa_live_buy_neu", "isa_live_buy_neu_min", "isa_live_buy_neu_max", "isa_live_buy_neu_step")
    with rc3:
        if int(live_eval_mode) == 5:
            live_buy_sov = _dial_slider("매수 초고평가(%)", "isa_live_buy_sov", "isa_live_buy_sov_min", "isa_live_buy_sov_max", "isa_live_buy_sov_step")
        else:
            st.empty()

    # Undervalue Pair
    rc1, rc2, rc3 = st.columns(3)
    with rc1: live_sell_un = _dial_slider("매도 저평가(%)", "isa_live_sell_un", "isa_live_sell_un_min", "isa_live_sell_un_max", "isa_live_sell_un_step")
    with rc2: live_buy_un = _dial_slider("매수 저평가(%)", "isa_live_buy_un", "isa_live_buy_un_min", "isa_live_buy_un_max", "isa_live_buy_un_step")
    with rc3:
        if int(live_eval_mode) == 5:
            live_sell_sun = _dial_slider("매도 초저평가(%)", "isa_live_sell_sun", "isa_live_sell_sun_min", "isa_live_sell_sun_max", "isa_live_sell_sun_step")
        else:
            st.empty()

        if int(live_eval_mode) == 5:
            rc1, rc2, rc3 = st.columns(3)
            with rc1: st.empty()
            with rc2: live_buy_sun = _dial_slider("매수 초저평가(%)", "isa_live_buy_sun", "isa_live_buy_sun_min", "isa_live_buy_sun_max", "isa_live_buy_sun_step")
            with rc3: st.empty()
        else:
            st.caption("3단계에서는 초고평가/초저평가 다이얼이 비활성화됩니다.")

        st.markdown("---")
        st.markdown("**주식 비중 제한 설정**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1: live_min_stock = st.slider("최소 주식 비중 (%)", 0.0, 100.0, float(_ss("isa_live_min_stock_ratio", 10.0)), step=1.0, key="isa_live_min_stock_ratio")
        with sc2: live_max_stock = st.slider("최대 주식 비중 (%)", 0.0, 100.0, float(_ss("isa_live_max_stock_ratio", 100.0)), step=1.0, key="isa_live_max_stock_ratio")
        with sc3: st.caption("전략이 매수/매도할 때 이 범위를 벗어나지 않도록 강제 제한합니다.")

    live_ratio_settings = {
        "sell_ratio_overvalue": float(live_sell_ov),
        "sell_ratio_neutral": float(live_sell_neu),
        "sell_ratio_undervalue": float(live_sell_un),
        "buy_ratio_overvalue": float(live_buy_ov),
        "buy_ratio_neutral": float(live_buy_neu),
        "buy_ratio_undervalue": float(live_buy_un),
        "min_stock_ratio": float(live_min_stock) / 100.0,
        "max_stock_ratio": float(live_max_stock) / 100.0,
    }
    if int(live_eval_mode) == 5:
        live_ratio_settings.update({
            "sell_ratio_super_overvalue": float(live_sell_sov if live_sell_sov is not None else 150.0),
            "sell_ratio_super_undervalue": float(live_sell_sun if live_sell_sun is not None else 33.0),
            "buy_ratio_super_overvalue": float(live_buy_sov if live_buy_sov is not None else 33.0),
            "buy_ratio_super_undervalue": float(live_buy_sun if live_buy_sun is not None else 200.0),
        })

    _param_specs = [
        ("ov", "overvalue_threshold", "isa_live_ov", "isa_live_ov_min", "isa_live_ov_max", "isa_live_ov_step"),
        ("un", "undervalue_threshold", "isa_live_un", "isa_live_un_min", "isa_live_un_max", "isa_live_un_step"),
        ("ir", "initial_stock_ratio", "isa_live_ratio", "isa_live_ratio_min", "isa_live_ratio_max", "isa_live_ratio_step"),
        ("sov", "sell_ratio_overvalue", "isa_live_sell_ov", "isa_live_sell_ov_min", "isa_live_sell_ov_max", "isa_live_sell_ov_step"),
        ("sn", "sell_ratio_neutral", "isa_live_sell_neu", "isa_live_sell_neu_min", "isa_live_sell_neu_max", "isa_live_sell_neu_step"),
        ("sun", "sell_ratio_undervalue", "isa_live_sell_un", "isa_live_sell_un_min", "isa_live_sell_un_max", "isa_live_sell_un_step"),
        ("bov", "buy_ratio_overvalue", "isa_live_buy_ov", "isa_live_buy_ov_min", "isa_live_buy_ov_max", "isa_live_buy_ov_step"),
        ("bn", "buy_ratio_neutral", "isa_live_buy_neu", "isa_live_buy_neu_min", "isa_live_buy_neu_max", "isa_live_buy_neu_step"),
        ("bun", "buy_ratio_undervalue", "isa_live_buy_un", "isa_live_buy_un_min", "isa_live_buy_un_max", "isa_live_buy_un_step"),
    ]
    if int(live_eval_mode) == 5:
        _param_specs.extend([
            ("ssov", "sell_ratio_super_overvalue", "isa_live_sell_sov", "isa_live_sell_sov_min", "isa_live_sell_sov_max", "isa_live_sell_sov_step"),
            ("ssun", "sell_ratio_super_undervalue", "isa_live_sell_sun", "isa_live_sell_sun_min", "isa_live_sell_sun_max", "isa_live_sell_sun_step"),
            ("bsov", "buy_ratio_super_overvalue", "isa_live_buy_sov", "isa_live_buy_sov_min", "isa_live_buy_sov_max", "isa_live_buy_sov_step"),
            ("bsun", "buy_ratio_super_undervalue", "isa_live_buy_sun", "isa_live_buy_sun_min", "isa_live_buy_sun_max", "isa_live_buy_sun_step"),
        ])

    _param_grids = {}
    for _short, _st_key, _dial_key, _min_key, _max_key, _step_key in _param_specs:
        _param_grids[_short] = _grid_values(float(_ss(_min_key, 0.0)), float(_ss(_max_key, 100.0)), float(_ss(_step_key, 1.0)))

    _est_combos = 1
    for _vals in _param_grids.values():
        _est_combos *= int(len(_vals))

    _core_sig_payload = {
        "trend": str(isa_trend_etf_code),
        "trade": str(isa_etf_code),
        "mode": int(live_eval_mode),
        "start": str(live_start),
        "end": str(live_end),
        "cap": int(live_cap),
        "fee": round(float(live_fee), 6),
    }
    _core_sig = hashlib.md5(json.dumps(_core_sig_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]

    _task_sig_payload = {
        **_core_sig_payload,
        "search": str(live_search_mode),
        "search_trials": int(live_random_trials),
        "max_combos": int(live_pre_max),
        "param_ranges": {
            _short: [float(_ss(_min_key, 0.0)), float(_ss(_max_key, 100.0)), float(_ss(_step_key, 1.0))]
            for _short, _st_key, _dial_key, _min_key, _max_key, _step_key in _param_specs
        },
    }
    _pre_sig = hashlib.md5(json.dumps(_task_sig_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    _db_table = f"isa_live_store_{_core_sig}"
    db = DBManager()

    def _db_pack_obj(_obj):
        try:
            return zlib.compress(pickle.dumps(_obj))
        except Exception:
            return None

    def _db_unpack_obj(_blob):
        if _blob is None:
            return None
        try:
            return pickle.loads(zlib.decompress(_blob))
        except Exception:
            return None

    def _db_get_result(_signature: str, _combo_key: str):
        _target_table = f"isa_live_store_{_core_sig}"
        return _db_unpack_obj(db.get_compute_result(_target_table, _combo_key))

    def _db_get_existing_keys(_signature: str, _keys: list[str]) -> set[str]:
        return db.get_existing_keys(f"isa_live_store_{_core_sig}")

    if st.session_state.get("isa_live_pre_sig") != _pre_sig:
        st.session_state["isa_live_pre_sig"] = _pre_sig
        st.session_state["isa_live_pre_pack"] = None
        _run_row = db.get_isa_run(_pre_sig)
        if isinstance(_run_row, dict):
            st.session_state["isa_live_pre_pack"] = {
                "signature": _pre_sig,
                "created_at": _run_row.get("created_at"),
                "updated_at": _run_row.get("updated_at"),
                "status": _run_row.get("status"),
                "target_count": int(_run_row.get("target_count", 0)),
                "done_count": int(_run_row.get("done_count", 0)),
                "results": {},
                "storage": "db",
            }
    elif st.session_state.get("isa_live_pre_pack") is None:
        _run_row = db.get_isa_run(_pre_sig)
        if isinstance(_run_row, dict):
            st.session_state["isa_live_pre_pack"] = {
                "signature": _pre_sig,
                "created_at": _run_row.get("created_at"),
                "updated_at": _run_row.get("updated_at"),
                "status": _run_row.get("status"),
                "target_count": int(_run_row.get("target_count", 0)),
                "done_count": int(_run_row.get("done_count", 0)),
                "results": {},
                "storage": "db",
            }

    pc1, pc2, pc3, pc4 = st.columns([1.1, 1.1, 1.1, 2.2])
    with pc1:
        if live_search_mode == "그리드 탐색":
            precompute_label = "사전 계산 시작"
        elif live_search_mode == "랜덤 탐색":
            precompute_label = "랜덤 탐색 시작"
        else:
            precompute_label = "Optuna 탐색 시작"
        precompute_btn = st.button(precompute_label, key="isa_live_precompute", type="primary")
    with pc2:
        reset_cache_btn = st.button("세션 캐시 초기화", key="isa_live_reset_cache")
    with pc3:
        clear_disk_btn = st.button("디스크 캐시 삭제", key="isa_live_clear_disk")
    with pc4:
        st.info(f"예상 조합 {_est_combos:,}개 | 사전 계산 최대 {int(live_pre_max):,}개")

    if reset_cache_btn:
        st.session_state.pop("isa_live_pre_pack", None)
        st.session_state.pop("isa_live_bt_result", None)
        st.session_state.pop("isa_live_bt_sig", None)
        st.success("세션 캐시를 초기화했습니다.")

    if clear_disk_btn:
        db.clear_compute_cache()
        st.success("DB 계산 캐시를 초기화했습니다.")

    if precompute_btn and live_start <= live_end:
        with st.spinner("사전 계산 중..."):
            _sig_df = _get_isa_daily_chart(str(isa_trend_etf_code), count=5000, end_date=str(live_end))
            _trade_df = _get_isa_daily_chart(str(isa_etf_code), count=5000, end_date=str(live_end))
            if _sig_df is None or _trade_df is None or len(_sig_df) < 200 or len(_trade_df) < 60:
                st.warning(f"사전 계산용 데이터를 불러오지 못했습니다. ({isa_trend_etf_code}/{isa_etf_code})")
            else:
                _start_eff = str(live_start)
                if _live_listing_date and _start_eff < _live_listing_date:
                    _start_eff = _live_listing_date
                _sig_df = _sig_df[_sig_df.index <= pd.Timestamp(live_end)]
                _trade_df = _trade_df[_trade_df.index <= pd.Timestamp(live_end)]

                _grid_keys = [_s[0] for _s in _param_specs]
                _grid_settings_key = {_s[0]: _s[1] for _s in _param_specs}
                _grid_lists = [_param_grids[_k] for _k in _grid_keys]

                _rng = np.random.default_rng()
                _combos = []
                _target_total = 0
                _exists = db.get_existing_keys(_db_table)

                if live_search_mode == "랜덤 탐색":
                    _max_unique = 1
                    for _arr in _grid_lists:
                        _max_unique *= int(len(_arr))
                    _target_n = min(int(live_random_trials), int(live_pre_max), max(1, _max_unique))

                    _combo_set = set()
                    _attempts = 0
                    _max_attempts = _target_n * 10  # 좀 더 넉넉하게 시도

                    while len(_combo_set) < _target_n and _attempts < _max_attempts:
                        _attempts += 1
                        _vals = []
                        for _arr in _grid_lists:
                            _vals.append(float(_arr[int(_rng.integers(0, len(_arr)))]))
                        _c = tuple(_vals)
                        _k = "|".join(f"{float(x):.4f}" for x in _c)
                        if _k not in _exists:
                            _combo_set.add(_c)
                    _combos = list(_combo_set)
                    _target_total = int(len(_combos))
                elif live_search_mode == "그리드 탐색":
                    _total_grid = 1
                    for _arr in _grid_lists:
                        _total_grid *= int(len(_arr))
                    _target_n = min(int(live_pre_max), max(1, int(_total_grid)))

                    def _idx_to_combo(_idx: int):
                        _out = [0.0] * len(_grid_lists)
                        _x = int(_idx)
                        for _i in range(len(_grid_lists) - 1, -1, -1):
                            _arr = _grid_lists[_i]
                            _base = int(len(_arr))
                            _out[_i] = float(_arr[_x % _base])
                            _x //= _base
                        return tuple(_out)

                    if int(_total_grid) <= int(_target_n):
                        _combos = [tuple(float(x) for x in _c) for _c in itertools.product(*_grid_lists)]
                    else:
                        # 랜덤 샘플링 로직으로 변경 (분할 저장 지원)
                        _combos = []
                        _seen_indices = set()

                        # 너무 많이 시도하지 않도록 루프 제한 (이미 꽉 찬 경우 대비)
                        _max_attempts = _target_n * 3
                        _attempts = 0

                        while len(_combos) < _target_n and _attempts < _max_attempts:
                            _attempts += 1
                            _idx = int(_rng.integers(0, _total_grid))
                            if _idx in _seen_indices:
                                continue
                            _seen_indices.add(_idx)
                            _c = _idx_to_combo(_idx)
                            _k = "|".join(f"{float(x):.4f}" for x in _c)
                            if _k not in _exists:
                                _combos.append(_c)

                        # 만약 랜덤으로 새로운 것을 못 찾았다면 (거의 다 찼다면) 정방향으로 빈틈 찾기
                        if len(_combos) < (_target_n // 2) and len(_exists) < _total_grid:
                            for _idx in range(min(100000, _total_grid)):
                                if len(_combos) >= _target_n:
                                    break
                                if _idx not in _seen_indices:
                                    _c = _idx_to_combo(_idx)
                                    _k = "|".join(f"{float(x):.4f}" for x in _c)
                                    if _k not in _exists:
                                        _combos.append(_c)

                    _combo_keys = ["|".join(f"{float(x):.4f}" for x in _c) for _c in _combos]
                    if _exists:
                        _combos = [_c for _c, _k in zip(_combos, _combo_keys) if _k not in _exists]
                    _target_total = int(len(_combos))
                else:
                    _target_total = int(min(int(live_random_trials), int(live_pre_max)))

                # ── 트렌드/이격도 사전 계산 (단 1회, 워커 전체 공유) ──
                _shared_merged = None
                try:
                    _tmp_strat = WDRStrategy(evaluation_mode=int(live_eval_mode))
                    _weekly_sig = WDRStrategy.daily_to_weekly(_sig_df)
                    _weekly_trd = WDRStrategy.daily_to_weekly(_trade_df)
                    if _weekly_sig is not None and _weekly_trd is not None:
                        _trend_arr = _tmp_strat.calc_growth_trend(_weekly_sig)
                        _wsig = _weekly_sig.copy()
                        _wsig["trend"] = _trend_arr
                        _wsig = _wsig.dropna(subset=["trend"])
                        _wsig["divergence"] = (_wsig["close"] - _wsig["trend"]) / _wsig["trend"] * 100.0
                        _shared_merged = _wsig[["close", "divergence"]].rename(columns={"close": "signal_close"}).join(
                            _weekly_trd.rename(columns={"close": "trade_close"}), how="inner"
                        ).dropna(subset=["trade_close"])
                        if _start_eff:
                            _shared_merged = _shared_merged[_shared_merged.index >= pd.Timestamp(_start_eff)]
                        if len(_shared_merged) < 5:
                            _shared_merged = None
                except Exception as _e:
                    st.warning(f"공유 트렌드 사전 계산 실패 (개별 계산으로 fallback): {_e}")
                    _shared_merged = None

                def _pre_task(_combo):
                    _settings = {}
                    _ir = 0.0
                    for _i, _short in enumerate(_grid_keys):
                        _v = float(_combo[_i])
                        if _short == "ir":
                            _ir = _v
                        else:
                            _settings[_grid_settings_key[_short]] = _v
                    _strat = WDRStrategy(settings=_settings, evaluation_mode=int(live_eval_mode))
                    _bt = _strat.run_backtest(
                        signal_daily_df=_sig_df,
                        trade_daily_df=_trade_df,
                        initial_balance=float(live_cap),
                        start_date=_start_eff,
                        fee_rate=float(live_fee) / 100.0,
                        initial_stock_ratio=float(_ir) / 100.0,
                        precomputed_merged=_shared_merged,
                    )
                    return "|".join(f"{float(x):.4f}" for x in _combo), _bt

                _workers = max(1, int(live_pre_workers))
                _res_map = {}
                _prog = st.progress(0.0)
                _done = 0
                _saved = 0
                db.save_isa_run(_pre_sig, "running", _target_total, 0, _task_sig_payload)
                if live_search_mode == "Optuna (베이지안)":
                    try:
                        import optuna
                        optuna.logging.set_verbosity(optuna.logging.WARNING)

                        _existing_keys = set(db.get_existing_keys(_db_table))
                        _best_score = float("-inf")

                        def _trial_to_combo(trial):
                            _vals = []
                            for _short, _st_key, _dial_key, _min_key, _max_key, _step_key in _param_specs:
                                _mn_raw = float(_ss(_min_key, 0.0))
                                _mx_raw = float(_ss(_max_key, 100.0))
                                _mn = float(min(_mn_raw, _mx_raw))
                                _mx = float(max(_mn_raw, _mx_raw))
                                _st = float(max(0.1, float(_ss(_step_key, 1.0))))
                                if _mx <= _mn:
                                    _v = _mn
                                else:
                                    _v = trial.suggest_float(f"p_{_short}", _mn, _mx, step=_st)
                                _vals.append(float(_snap_step(_v, _st, _mn, _mx)))
                            return tuple(_vals)

                        def objective(trial):
                            nonlocal _done, _saved, _best_score
                            _score = float("-inf")
                            try:
                                _combo = _trial_to_combo(trial)
                                _k, _v = _pre_task(_combo)
                                if isinstance(_v, dict):
                                    _metrics = _v.get("metrics", {}) or {}
                                    _score = float(_metrics.get("cagr", float("-inf")) or float("-inf"))
                                    if _k not in _existing_keys:
                                        _blob = _db_pack_obj(_v)
                                        if _blob is not None:
                                            db.save_compute_result(_db_table, str(_k), _blob)
                                            _saved += 1
                                            _existing_keys.add(_k)
                                            if len(_res_map) < 256:
                                                _res_map[_k] = _v
                            except Exception as _e:
                                logging.error(f"Optuna objective error: {_e}")
                                _score = float("-inf")

                            if _score > _best_score:
                                _best_score = _score

                            _done += 1
                            if _target_total > 0:
                                if _done == _target_total or _done % max(1, _target_total // 100) == 0:
                                    _prog.progress(min(1.0, _done / _target_total))
                                    if _done % 20 == 0:
                                        db.save_isa_run(_pre_sig, "running", _target_total, _saved)
                            return _score

                        _study = optuna.create_study(
                            direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                        )
                        _study.optimize(objective, n_trials=max(1, int(_target_total)))
                        _workers = 1
                        _prog.empty()
                        if _study.best_trial is not None:
                            st.caption(f"Optuna 최고 CAGR: {_study.best_value:+.2f}%")
                    except Exception as _e:
                        _prog.empty()
                        st.error(f"Optuna 탐색 실패: {_e}")
                else:
                    with ThreadPoolExecutor(max_workers=_workers) as _pool:
                        _futs = [_pool.submit(_pre_task, c) for c in _combos]
                        _total = max(1, len(_futs))
                        for _ft in as_completed(_futs):
                            try:
                                _k, _v = _ft.result()
                                if isinstance(_v, dict):
                                    _blob = _db_pack_obj(_v)
                                    if _blob is not None:
                                        db.save_compute_result(_db_table, str(_k), _blob)
                                        _saved += 1
                                        if len(_res_map) < 256:
                                            _res_map[_k] = _v
                            except Exception as _e:
                                logging.error(f"Error in pre_task: {_e}")
                                pass
                            _done += 1
                            if _done == _total or _done % max(1, _total // 100) == 0:
                                _prog.progress(_done / _total)
                                if _done % 20 == 0:
                                    db.save_isa_run(_pre_sig, "running", _target_total, _saved)
                    _prog.empty()
                _total_saved_db = len(db.get_existing_keys(_db_table))
                _pack = {
                    "signature": _pre_sig,
                    "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "done",
                    "target_count": int(max(_target_total, _total_saved_db)),
                    "done_count": int(_total_saved_db),
                    "results": _res_map,
                    "storage": "db",
                }
                db.save_isa_run(_pre_sig, "done", int(max(_target_total, _total_saved_db)), int(_total_saved_db), _task_sig_payload)
                st.session_state["isa_live_pre_pack"] = _pack
                st.session_state.pop("isa_live_top30_cache_key", None)  # 캐시 무효화
                st.success(f"사전 계산 완료: 신규 {_saved:,}개 저장 | DB 누적 {_total_saved_db:,}개 (워커 {_workers}개)")

    _pack_now = st.session_state.get("isa_live_pre_pack")
    if isinstance(_pack_now, dict) and _pack_now.get("signature") == _pre_sig:
        _done_count = int(_pack_now.get("done_count", 0))
        _target_count = int(_pack_now.get("target_count", 0))
        _status = str(_pack_now.get("status", "ready"))

        if _status == "running":
            st.warning("⚠️ 이전 계산 작업이 중단된 상태입니다. 버튼을 누르면 이어서 계산합니다.")

        st.caption(
            f"사전 계산 상태: {_done_count:,}/{_target_count:,}개 완료 ({_status}) | 생성: {_pack_now.get('created_at', '-')}"
        )
    else:
        # Top30 결과 캐시: _db_table이 바뀔 때만 DB 재조회
        _top30_cache_key = st.session_state.get("isa_live_top30_cache_key")
        if _top30_cache_key != _db_table:
            # DB에서 키 목록 조회 및 결과 로드
            _ek = db.get_existing_keys(_db_table)
            _param_map_ko = {
                "ov": "고평가(%)", "un": "저평가(%)", "ir": "초기비중(%)",
                "sov": "매도_고평가", "sn": "매도_중립", "sun": "매도_저평가",
                "bov": "매수_고평가", "bn": "매수_중립", "bun": "매수_저평가",
                "ssov": "매도_초고평가", "ssun": "매도_초저평가",
                "bsov": "매수_초고평가", "bsun": "매수_초저평가"
            }
            _res_list_cache = []
            if _ek:
                for _k in list(_ek)[:2000]:
                    _v = _db_get_result(_pre_sig, _k)
                    if _v and "metrics" in _v:
                        _m = _v["metrics"]
                        _row = {"Key": _k,
                            "연복리(%)": _m.get("cagr", 0), "낙폭(%)": _m.get("mdd", 0),
                            "샤프": _m.get("sharpe", 0), "칼마": _m.get("calmar", 0),
                            "승률(%)": _m.get("win_rate", 0)}
                        for _idx, _spec in enumerate(_param_specs):
                            _ko = _param_map_ko.get(_spec[0], _spec[0])
                            _kp = _k.split("|")
                            if _idx < len(_kp):
                                _row[_ko] = float(_kp[_idx])
                        _res_list_cache.append(_row)
            st.session_state["isa_live_top30_cache_key"] = _db_table
            st.session_state["isa_live_top30_ek_count"] = len(_ek) if _ek else 0
            st.session_state["isa_live_top30_res_list"] = _res_list_cache

        _ek_count = st.session_state.get("isa_live_top30_ek_count", 0)
        _res_list = st.session_state.get("isa_live_top30_res_list", [])

        if _ek_count > 0:
            with st.expander(f"🏆 최적화 성과 Top 30 (누적 {_ek_count:,}개 중)", expanded=True):
                if _res_list:
                    _res_df = pd.DataFrame(_res_list).sort_values("연복리(%)", ascending=False).head(30)
                    st.dataframe(
                        _res_df, use_container_width=True, hide_index=True,
                        column_config={
                            "Key": None,
                            "연복리(%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "낙폭(%)": st.column_config.NumberColumn(format="%.2f%%"),
                            "샤프": st.column_config.NumberColumn(format="%.2f"),
                            "칼마": st.column_config.NumberColumn(format="%.2f"),
                            "승률(%)": st.column_config.NumberColumn(format="%.1f%%"),
                        }
                    )
                    st.caption("※ 초기 2,000개 샘플 중 상위 30개 결과입니다.")
                    st.markdown("---")
                    sel_col1, sel_col2 = st.columns([3, 1])
                    with sel_col1:
                        _rank_options = [
                            f"{_i+1}위: 연복리 {_res_df.iloc[_i]['연복리(%)']:.2f}%, 낙폭 {_res_df.iloc[_i]['낙폭(%)']:.2f}%"
                            for _i in range(len(_res_df))
                        ]
                        _selected_rank = st.selectbox("다이얼에 적용할 순위 선택", _rank_options, label_visibility="collapsed")
                    with sel_col2:
                        if st.button("🏆 설정 적용", use_container_width=True, type="primary"):
                            _idx = _rank_options.index(_selected_rank)
                            _target_key = _res_df.iloc[_idx]['Key']
                            _parts = _target_key.split("|")
                            _apply_payload = {}
                            for _i, _spec in enumerate(_param_specs):
                                _dial_key = _spec[2]
                                if _i < len(_parts):
                                    _apply_payload[_dial_key] = float(_parts[_i])
                            if _apply_payload:
                                st.session_state["isa_live_apply_payload"] = _apply_payload
                                st.session_state["isa_live_apply_token"] = time.time()
                            st.rerun()
                else:
                    st.info("결과 데이터 로드 중 오류가 발생했습니다.")

    if live_start > live_end:
        st.error("기간을 확인해 주세요. (시작일 <= 종료일)")
    else:
        _dial_combo_vals = []
        _live_settings_by_combo = {}
        _ir_snap = 0.0
        for _short, _st_key, _dial_key, _min_key, _max_key, _step_key in _param_specs:
            _mn = float(_ss(_min_key, 0.0))
            _mx = float(_ss(_max_key, 100.0))
            _st = float(_ss(_step_key, 1.0))
            _dv = _snap_step(float(_ss(_dial_key, _mn)), max(0.1, _st), min(_mn, _mx), max(_mn, _mx))
            _dial_combo_vals.append(float(_dv))
            if _short == "ir":
                _ir_snap = float(_dv)
            else:
                _live_settings_by_combo[_st_key] = float(_dv)
        _live_key = "|".join(f"{float(_x):.4f}" for _x in _dial_combo_vals)

        # 트렌드 차트(가격/추세/임계밴드 + 이격도) 캐시
        _live_start_effective = str(live_start)
        if _live_listing_date and _live_start_effective < _live_listing_date:
            _live_start_effective = _live_listing_date
        _trend_sig = (str(isa_trend_etf_code), str(live_end))
        if st.session_state.get("isa_live_trend_sig") != _trend_sig:
            _trend_pack = None
            _trend_daily = _get_isa_daily_chart(str(isa_trend_etf_code), count=5000, end_date=str(live_end))
            if _trend_daily is not None and len(_trend_daily) >= 200:
                _trend_daily = _trend_daily[_trend_daily.index <= pd.Timestamp(live_end)]
                _trend_strat = WDRStrategy(evaluation_mode=int(live_eval_mode))
                _weekly = _trend_strat.daily_to_weekly(_trend_daily)
                if _weekly is not None and len(_weekly) > 2:
                    _trend_arr = _trend_strat.calc_growth_trend(_weekly)
                    _trend_pack = {"weekly_df": _weekly, "trend": _trend_arr}
            st.session_state["isa_live_trend_sig"] = _trend_sig
            st.session_state["isa_live_trend_pack"] = _trend_pack

        _trend_pack = st.session_state.get("isa_live_trend_pack")
        st.subheader("트렌드 차트")
        if isinstance(_trend_pack, dict) and _trend_pack.get("weekly_df") is not None:
            _weekly_df = _trend_pack["weekly_df"]
            _trend_arr = np.asarray(_trend_pack.get("trend"), dtype=float)
            _mask_date = _weekly_df.index >= pd.Timestamp(_live_start_effective)
            _wplot = _weekly_df.loc[_mask_date]
            _tplot = _trend_arr[_mask_date]
            _valid = ~np.isnan(_tplot)
            _wplot = _wplot.loc[_valid]
            _tplot = _tplot[_valid]

            if len(_wplot) >= 2:
                _ov_plot = float(_live_settings_by_combo.get("overvalue_threshold", float(_ss("isa_live_ov", 5.0))))
                _un_plot = float(_live_settings_by_combo.get("undervalue_threshold", float(_ss("isa_live_un", -6.0))))

                fig_tr = go.Figure()
                fig_tr.add_trace(go.Scatter(x=_wplot.index, y=_wplot["close"], name="시그널 가격", line=dict(color="royalblue", width=2)))
                fig_tr.add_trace(go.Scatter(x=_wplot.index, y=_tplot, name="성장 추세", line=dict(color="orange", width=2, dash="dash")))
                fig_tr.add_trace(go.Scatter(
                    x=_wplot.index,
                    y=_tplot * (1.0 + _ov_plot / 100.0),
                    name=f"고평가(+{_ov_plot:.2f}%)",
                    line=dict(color="red", width=1.2, dash="dot"),
                ))
                fig_tr.add_trace(go.Scatter(
                    x=_wplot.index,
                    y=_tplot * (1.0 + _un_plot / 100.0),
                    name=f"저평가({_un_plot:.2f}%)",
                    line=dict(color="green", width=1.2, dash="dot"),
                ))
                fig_tr.update_layout(
                    title="가격 vs 성장 추세 vs 임계 밴드",
                    height=360,
                    margin=dict(l=0, r=0, t=60, b=20),
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig_tr, use_container_width=True)

                _div = ((_wplot["close"].values - _tplot) / _tplot) * 100.0
                fig_div = go.Figure()
                fig_div.add_trace(go.Scatter(
                    x=_wplot.index,
                    y=_div,
                    name="이격도(%)",
                    line=dict(color="#1f77b4", width=2),
                ))
                fig_div.add_hline(y=float(_ov_plot), line_dash="dot", line_color="red")
                fig_div.add_hline(y=float(_un_plot), line_dash="dot", line_color="green")
                fig_div.add_hline(y=0.0, line_dash="dash", line_color="gray")
                fig_div.update_layout(
                    title="이격도(%)와 고/저평가 임계값",
                    yaxis_title="이격도 (%)",
                    height=280,
                    margin=dict(l=0, r=0, t=50, b=20),
                )
                st.plotly_chart(fig_div, use_container_width=True)
            else:
                st.info("트렌드 차트 표시를 위한 유효 데이터가 부족합니다.")
        else:
            st.info("트렌드 차트를 불러오지 못했습니다.")
        live_sig = (
            str(isa_trend_etf_code),
            str(isa_etf_code),
            int(live_eval_mode),
            str(live_start),
            str(live_end),
            int(live_cap),
            float(round(float(live_fee), 4)),
            _live_key,
            tuple(sorted((k, round(float(v), 4)) for k, v in live_ratio_settings.items())),
            _pre_sig,
        )
        # ── 데이터 캐시: ETF/종료일이 바뀔 때만 재로드 ──────────────────
        _live_data_key = (str(isa_trend_etf_code), str(isa_etf_code), str(live_end))
        if st.session_state.get("isa_live_data_key") != _live_data_key:
            with st.spinner("데이터 로딩 중..."):
                _live_sig_df_raw = _get_isa_daily_chart(str(isa_trend_etf_code), count=5000, end_date=str(live_end))
                _live_trade_df_raw = _get_isa_daily_chart(str(isa_etf_code), count=5000, end_date=str(live_end))
                if _live_sig_df_raw is not None:
                    _live_sig_df_raw = _live_sig_df_raw[_live_sig_df_raw.index <= pd.Timestamp(live_end)]
                if _live_trade_df_raw is not None:
                    _live_trade_df_raw = _live_trade_df_raw[_live_trade_df_raw.index <= pd.Timestamp(live_end)]
                st.session_state["isa_live_data_key"] = _live_data_key
                st.session_state["isa_live_sig_df"] = _live_sig_df_raw
                st.session_state["isa_live_trade_df"] = _live_trade_df_raw
                # merged도 사전 계산해 둠 (백테스트 호출마다 trend 재계산 방지)
                _live_shared_merged = None
                if _live_sig_df_raw is not None and _live_trade_df_raw is not None and len(_live_sig_df_raw) >= 200 and len(_live_trade_df_raw) >= 60:
                    try:
                        _tmp = WDRStrategy(evaluation_mode=int(live_eval_mode))
                        _ws = WDRStrategy.daily_to_weekly(_live_sig_df_raw)
                        _wt = WDRStrategy.daily_to_weekly(_live_trade_df_raw)
                        if _ws is not None and _wt is not None:
                            _tr = _tmp.calc_growth_trend(_ws)
                            _ws2 = _ws.copy()
                            _ws2["trend"] = _tr
                            _ws2 = _ws2.dropna(subset=["trend"])
                            _ws2["divergence"] = (_ws2["close"] - _ws2["trend"]) / _ws2["trend"] * 100.0
                            _live_shared_merged = _ws2[["close", "divergence"]].rename(columns={"close": "signal_close"}).join(
                                _wt.rename(columns={"close": "trade_close"}), how="inner"
                            ).dropna(subset=["trade_close"])
                    except Exception:
                        _live_shared_merged = None
                st.session_state["isa_live_shared_merged"] = _live_shared_merged

        _live_sig_df = st.session_state.get("isa_live_sig_df")
        _live_trade_df = st.session_state.get("isa_live_trade_df")
        _live_shared_merged = st.session_state.get("isa_live_shared_merged")

        # ─────────────────────────────────────────────────────────────────
        _sig_changed = st.session_state.get("isa_live_bt_sig") != live_sig
        need_db_lookup = _sig_changed
        need_full_run = bool(live_run_btn)  # 버튼: 강제 재계산

        if need_db_lookup or need_full_run:
            t0 = time.perf_counter()
            _cached_bt = None
            _cached_pack = st.session_state.get("isa_live_pre_pack")
            # 1) 세션 메모리 캐시 먼저
            if isinstance(_cached_pack, dict) and _cached_pack.get("signature") == _pre_sig:
                _cached_bt = (_cached_pack.get("results") or {}).get(_live_key)
            # 2) DB 조회
            if _cached_bt is None:
                _cached_bt = _db_get_result(_pre_sig, _live_key)
                if isinstance(_cached_bt, dict) and isinstance(_cached_pack, dict) and _cached_pack.get("signature") == _pre_sig:
                    _mem_results = _cached_pack.setdefault("results", {})
                    if len(_mem_results) < 256:
                        _mem_results[_live_key] = _cached_bt

            if isinstance(_cached_bt, dict):
                # DB 히트 → 즉시 표시
                st.session_state["isa_live_bt_result"] = _cached_bt
                st.session_state["isa_live_bt_source"] = "사전 계산 DB 캐시"
            else:
                # DB 미스 → 캐시 데이터로 즉시 계산 후 DB 저장
                if _live_sig_df is None or _live_trade_df is None or len(_live_sig_df) < 200 or len(_live_trade_df) < 60:
                    st.warning(f"데이터 없음: {isa_trend_etf_code}/{isa_etf_code}")
                    st.session_state["isa_live_bt_result"] = None
                else:
                    _lse = str(live_start)
                    if _live_listing_date and _lse < _live_listing_date:
                        _lse = _live_listing_date

                    live_settings = dict(_live_settings_by_combo)
                    live_strat = WDRStrategy(settings=live_settings, evaluation_mode=int(live_eval_mode))
                    live_bt = live_strat.run_backtest(
                        signal_daily_df=_live_sig_df,
                        trade_daily_df=_live_trade_df,
                        initial_balance=float(live_cap),
                        start_date=_lse,
                        fee_rate=float(live_fee) / 100.0,
                        initial_stock_ratio=float(_ir_snap) / 100.0,
                        precomputed_merged=_live_shared_merged,
                    )
                    st.session_state["isa_live_bt_result"] = live_bt
                    st.session_state["isa_live_bt_source"] = "실시간 계산 → DB 저장"
                    # 결과를 DB에 즉시 저장 (다음 조회 시 캐시 히트)
                    if isinstance(live_bt, dict):
                        _blob = _db_pack_obj(live_bt)
                        if _blob is not None:
                            db.save_compute_result(_db_table, _live_key, _blob)

            st.session_state["isa_live_bt_sig"] = live_sig
            st.session_state["isa_live_bt_updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["isa_live_bt_elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

        live_bt_res = st.session_state.get("isa_live_bt_result")
        _bt_source = st.session_state.get("isa_live_bt_source", "")
        if not isinstance(live_bt_res, dict):
            st.info("다이얼을 조정하면 자동으로 계산됩니다.")
        else:
            live_eq = live_bt_res.get("equity_df")
            if live_eq is None or len(live_eq) < 2:
                st.info("라이브 백테스트 결과가 없습니다.")
            else:
                live_metrics = live_bt_res.get("metrics", {}) or {}
                live_bm = live_bt_res.get("benchmark_df")
                live_updated_at = str(st.session_state.get("isa_live_bt_updated_at", ""))
                live_elapsed_ms = st.session_state.get("isa_live_bt_elapsed_ms")
                live_source = str(st.session_state.get("isa_live_bt_source", "실시간 계산"))

                if live_updated_at:
                    st.caption(f"최근 계산 시각: {live_updated_at}")
                if live_elapsed_ms is not None:
                    st.caption(f"응답 시간: {float(live_elapsed_ms):,.1f} ms | 결과 소스: {live_source}")

                _plot_idx = live_eq.index
                if bool(live_use_anim) and len(_plot_idx) > 2:
                    ap_col1, ap_col2 = st.columns([1, 4])
                    with ap_col1:
                        live_autoplay = st.checkbox("자동 재생", value=False, key="isa_live_autoplay")
                    with ap_col2:
                        _view_n = int(st.slider(
                            "라이브 프레임",
                            min_value=1,
                            max_value=len(_plot_idx),
                            value=int(st.session_state.get("isa_live_frame_pos", len(_plot_idx))),
                            step=max(1, int(live_anim_step)),
                            key="isa_live_frame_pos",
                        ))
                    
                    if live_autoplay and _view_n < len(_plot_idx):
                        # 프레임 증가 후 재실행
                        st.session_state["isa_live_frame_pos"] = min(len(_plot_idx), _view_n + max(1, int(live_anim_step)))
                        time.sleep(float(live_anim_delay_ms) / 1000.0)
                        st.rerun()
                    elif live_autoplay and _view_n >= len(_plot_idx):
                        st.session_state["isa_live_autoplay"] = False # 정지
                else:
                    _view_n = len(_plot_idx)

                _leq_sub = live_eq.iloc[:_view_n]
                _lbm_sub = live_bm.iloc[:_view_n] if live_bm is not None else None
                
                # 실시간 상태 표시 (마지막 포인트 기준)
                if len(_leq_sub) > 0:
                    _last = _leq_sub.iloc[-1]
                    _l_date = _leq_sub.index[-1].strftime("%Y-%m-%d")
                    _l_state = _last.get("action", "HOLD")
                    _l_weight = (_last["shares"] * _last["price"] / _last["equity"]) * 100.0
                    
                    st.markdown(f"### [ {_l_date} ] 현재 상태: `{_l_state}` | 보유 비중: `{_l_weight:.1f}%`")

                # --- Performance Analysis (Charts & Metrics) ---
                _render_performance_analysis(
                    equity_series=_leq_sub["equity"],
                    benchmark_series=_lbm_sub,
                    strategy_metrics=live_metrics,
                    strategy_label="위대리(전략)",
                    benchmark_label="나스닥 지수",
                    show_drawdown=True,
                    show_weight=True,
                    equity_df=_leq_sub
                )

                with st.expander("상세 성과 지표"):
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("매매 횟수", f"{live_metrics.get('trade_count', 0)}회")
                    sc2.metric("승률", f"{live_metrics.get('win_rate', 0):.1f}%")
                    sc3.metric("최종 자산", f"{int(live_metrics.get('final_equity', 0)):,}원")
                    
                    if "trades" in live_bt_res:
                        st.dataframe(pd.DataFrame(live_bt_res["trades"]).tail(100), use_container_width=True)
