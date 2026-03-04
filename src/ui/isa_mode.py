import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import os
import json
import pickle
import hashlib
import itertools
import sqlite3
import zlib
import plotly.graph_objects as go
from pathlib import Path
from src.utils.db_manager import DBManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.engine.kis_trader import KISTrader
from src.strategy.widaeri import WDRStrategy, _wdr_opt_task
import src.engine.data_cache as _dc
from src.constants import IS_CLOUD, ISA_WDR_TRADE_ETF_CODES
from src.utils.formatting import (
    _etf_name_kr, _fmt_etf_code_name, _sanitize_isa_trade_etf, 
    _code_only, _format_kis_holdings_df
)
from src.utils.helpers import _get_runtime_value
from src.utils.kis import _get_kis_token, _compute_kis_balance_summary
from src.ui.components.performance import (
    _render_performance_analysis, _apply_return_hover_format, _apply_dd_hover_format
)
from src.ui.components.triggers import render_strategy_trigger_tab
from src.ui.isa_opt_tab import render_isa_opt_tab
from src.ui.isa_live_tab import render_isa_live_tab

def render_kis_isa_mode(config, save_config):
    """KIS ISA 위대리(WDR) 전략 모드 - 4탭 구성."""
    from src.utils.helpers import load_mode_config, save_mode_config
    _isa_cfg = load_mode_config("isa")
    st.title("ISA 위대리(WDR) 전략")

    # ── 사이드바 설정 ──
    st.sidebar.header("ISA 설정")
    kis_ak = _get_runtime_value(("KIS_ISA_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_ISA_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(_isa_cfg.get("kis_isa_account_no", "") or _isa_cfg.get("kis_isa_account_no", "") or _get_runtime_value(("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(_isa_cfg.get("kis_isa_prdt_cd", "") or _isa_cfg.get("kis_isa_prdt_cd", "") or _get_runtime_value(("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))

    # 계좌번호 10자리 → 앞8(CANO) + 뒤2(상품코드) 자동 분리
    _raw_acct = kis_acct.replace("-", "").strip()
    if len(_raw_acct) == 10 and kis_prdt in ("01", ""):
        kis_acct = _raw_acct[:8]
        kis_prdt = _raw_acct[8:]

    if not IS_CLOUD:
        with st.sidebar.expander("KIS API 키", expanded=False):
            kis_ak = st.text_input("앱 키", value=kis_ak, type="password", key="isa_app_key")
            kis_sk = st.text_input("시크릿 키", value=kis_sk, type="password", key="isa_app_secret")
            kis_acct = st.text_input("계좌번호 (앞 8자리)", value=kis_acct, key="isa_account_no", help="10자리 입력 시 자동으로 앞8+뒤2 분리")
            kis_prdt = st.text_input("상품코드 (뒤 2자리)", value=kis_prdt, key="isa_prdt_cd")
            _raw2 = kis_acct.replace("-", "").strip()
            if len(_raw2) == 10:
                kis_acct = _raw2[:8]
                kis_prdt = _raw2[8:]

    # ETF 선택: 매매 ETF / TREND ETF(시그널용)
    def _build_etf_options(code_list):
        out = {}
        for _code in code_list:
            _c = str(_code).strip()
            if _c:
                out[f"{_c} {_etf_name_kr(_c)}"] = _c
        return out

    _isa_trade_options = _build_etf_options(list(ISA_WDR_TRADE_ETF_CODES))
    _isa_trend_options = _build_etf_options(["133690", "360750", "453850", "251350", "418660", "409820", "423920", "465610", "461910"])

    _saved_trade_raw = str(_isa_cfg.get("kis_isa_etf_code", _get_runtime_value("KIS_ISA_ETF_CODE", "418660")))
    _saved_trade_etf = _sanitize_isa_trade_etf(_saved_trade_raw, default="418660")
    _saved_trend_etf = str(_isa_cfg.get("kis_isa_trend_etf_code", _get_runtime_value("KIS_ISA_TREND_ETF_CODE", "133690")))

    if _saved_trend_etf and _saved_trend_etf not in _isa_trend_options.values():
        _isa_trend_options[f"{_saved_trend_etf} {_etf_name_kr(_saved_trend_etf)}"] = _saved_trend_etf

    _trade_default_label = next((k for k, v in _isa_trade_options.items() if v == _saved_trade_etf), list(_isa_trade_options.keys())[0])
    _trend_default_label = next((k for k, v in _isa_trend_options.items() if v == _saved_trend_etf), list(_isa_trend_options.keys())[0])

    selected_trend_etf_label = st.sidebar.selectbox(
        "TREND ETF (시그널)",
        list(_isa_trend_options.keys()),
        index=list(_isa_trend_options.keys()).index(_trend_default_label),
        key="isa_trend_etf_select",
        disabled=IS_CLOUD,
    )
    selected_etf_label = st.sidebar.selectbox(
        "매매 ETF",
        list(_isa_trade_options.keys()),
        index=list(_isa_trade_options.keys()).index(_trade_default_label),
        key="isa_etf_select",
        disabled=IS_CLOUD,
    )
    isa_etf_code = _sanitize_isa_trade_etf(_isa_trade_options[selected_etf_label], default="418660")
    isa_trend_etf_code = _isa_trend_options[selected_trend_etf_label]
    if _saved_trade_raw != _saved_trade_etf:
        st.sidebar.caption("1배 ETF는 매매 ETF에서 제외됩니다. 기존 설정값을 2배 ETF로 보정했습니다.")

    wdr_eval_mode = st.sidebar.selectbox(
        "평가 시스템", [3, 5], index=[3, 5].index(int(_isa_cfg.get("kis_isa_wdr_mode", 5))),
        format_func=lambda x: f"{x}단계", key="isa_wdr_mode", disabled=IS_CLOUD,
    )
    wdr_ov = st.sidebar.number_input(
        "고평가 임계값 (%)", min_value=0.0, max_value=30.0,
        value=float(_isa_cfg.get("kis_isa_wdr_ov", 5.0)), step=0.5,
        key="isa_wdr_ov", disabled=IS_CLOUD,
    )
    wdr_un = st.sidebar.number_input(
        "저평가 임계값 (%)", min_value=-30.0, max_value=0.0,
        value=float(_isa_cfg.get("kis_isa_wdr_un", -6.0)), step=0.5,
        key="isa_wdr_un", disabled=IS_CLOUD,
    )

    with st.sidebar.expander("💼 상세 매수/매도 비중 설정", expanded=False):
        st.caption("각 상태별 리밸런싱 시 적용할 비중 (%)")
        col_s, col_b = st.columns(2)
        with col_s:
            st.markdown("**매도 비중**")
            wdr_s_ov = st.number_input("고평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_s_ov", 100.0)), 1.0, key="isa_wdr_s_ov")
            wdr_s_neu = st.number_input("중립", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_s_neu", 66.7)), 1.0, key="isa_wdr_s_neu")
            wdr_s_un = st.number_input("저평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_s_un", 60.0)), 1.0, key="isa_wdr_s_un")
            if int(wdr_eval_mode) == 5:
                wdr_s_sov = st.number_input("초고평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_s_sov", 150.0)), 1.0, key="isa_wdr_s_sov")
                wdr_s_sun = st.number_input("초저평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_s_sun", 33.0)), 1.0, key="isa_wdr_s_sun")
        with col_b:
            st.markdown("**매수 비중**")
            wdr_b_ov = st.number_input("고평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_b_ov", 66.7)), 1.0, key="isa_wdr_b_ov")
            wdr_b_neu = st.number_input("중립", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_b_neu", 66.7)), 1.0, key="isa_wdr_b_neu")
            wdr_b_un = st.number_input("저평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_b_un", 120.0)), 1.0, key="isa_wdr_b_un")
            if int(wdr_eval_mode) == 5:
                wdr_b_sov = st.number_input("초고평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_b_sov", 33.0)), 1.0, key="isa_wdr_b_sov")
                wdr_b_sun = st.number_input("초저평가", 0.0, 300.0, float(_isa_cfg.get("kis_isa_wdr_b_sun", 200.0)), 1.0, key="isa_wdr_b_sun")
    isa_seed = st.sidebar.number_input(
        "초기자본 (시드)", min_value=1_000_000, max_value=1_000_000_000,
        value=int(_isa_cfg.get("kis_isa_seed", 10_000_000)), step=1_000_000,
        key="isa_seed", disabled=IS_CLOUD,
    )
    
    # 초기 주식 비중 (Sidebar)
    wdr_start_ratio = float(_isa_cfg.get("kis_isa_initial_ratio", 50.0))
    # ETF나 날짜 변경 시 자동 추천 비중 연동 로직
    if st.session_state.get("isa_sidebar_ratio_sync_etf") != str(isa_etf_code):
        st.session_state["isa_sidebar_ratio_sync_etf"] = str(isa_etf_code)
        _rec = _dc.get_wdr_default_initial_stock_ratio(str(isa_etf_code))
        if _rec is not None:
            wdr_start_ratio = round(float(_rec) * 100.0, 1)
            st.session_state["isa_initial_ratio"] = wdr_start_ratio

    isa_initial_ratio = st.sidebar.slider(
        "초기 주식 비중 (%)", min_value=0.0, max_value=100.0,
        value=float(st.session_state.get("isa_initial_ratio", wdr_start_ratio)),
        step=2.0, key="isa_initial_ratio", disabled=IS_CLOUD
    )
    _isa_start_default = _isa_cfg.get("kis_isa_start_date", "2022-03-08")
    _isa_listing_date = _dc.get_wdr_trade_listing_date(str(isa_etf_code))
    if _isa_listing_date and str(_isa_start_default) < _isa_listing_date:
        _isa_start_default = _isa_listing_date
    # 이전 계산에서 매매 ETF 상장일이 감지되었으면 자동 보정
    _prev_res = st.session_state.get("isa_signal_result")
    if isinstance(_prev_res, dict) and _prev_res.get("trade_first_date"):
        _tfd = _prev_res["trade_first_date"]
        if str(_isa_start_default) < _tfd:
            _isa_start_default = _tfd

    _isa_start_sync_key = "isa_start_sync_trade_code"
    _isa_trade_sync_val = str(isa_etf_code)
    if st.session_state.get(_isa_start_sync_key) != _isa_trade_sync_val or "isa_start_date" not in st.session_state:
        st.session_state["isa_start_date"] = pd.to_datetime(_isa_start_default).date()
        st.session_state[_isa_start_sync_key] = _isa_trade_sync_val
    elif str(st.session_state.get("isa_start_date")) < str(_isa_start_default):
        # 상장일 등 자동 보정된 날짜가 현재 세션값보다 미래인 경우 강제 업데이트
        st.session_state["isa_start_date"] = pd.to_datetime(_isa_start_default).date()

    isa_start_date = st.sidebar.date_input(
        "시작일",
        key="isa_start_date",
        disabled=IS_CLOUD,
    )
    if _isa_listing_date:
        st.sidebar.caption(f"시작일 자동기준: 매매 ETF 첫 거래일({_isa_listing_date})")

    if not IS_CLOUD and st.sidebar.button("ISA 설정 저장", key="isa_save_cfg"):
        isa_data = {
            "kis_isa_account_no": str(kis_acct).strip(),
            "kis_isa_prdt_cd": str(kis_prdt).strip() or "01",
            "kis_isa_etf_code": isa_etf_code,
            "kis_isa_trend_etf_code": isa_trend_etf_code,
            "kis_isa_wdr_mode": int(wdr_eval_mode),
            "kis_isa_wdr_ov": float(wdr_ov),
            "kis_isa_wdr_un": float(wdr_un),
            "kis_isa_start_date": str(isa_start_date),
            "kis_isa_seed": int(isa_seed),
            "kis_isa_initial_ratio": float(isa_initial_ratio),
            "kis_isa_wdr_s_ov": float(st.session_state.isa_wdr_s_ov),
            "kis_isa_wdr_s_neu": float(st.session_state.isa_wdr_s_neu),
            "kis_isa_wdr_s_un": float(st.session_state.isa_wdr_s_un),
            "kis_isa_wdr_b_ov": float(st.session_state.isa_wdr_b_ov),
            "kis_isa_wdr_b_neu": float(st.session_state.isa_wdr_b_neu),
            "kis_isa_wdr_b_un": float(st.session_state.isa_wdr_b_un),
        }
        if int(wdr_eval_mode) == 5:
            isa_data["kis_isa_wdr_s_sov"] = float(st.session_state.isa_wdr_s_sov)
            isa_data["kis_isa_wdr_s_sun"] = float(st.session_state.isa_wdr_s_sun)
            isa_data["kis_isa_wdr_b_sov"] = float(st.session_state.isa_wdr_b_sov)
            isa_data["kis_isa_wdr_b_sun"] = float(st.session_state.isa_wdr_b_sun)
        # 모드별 설정 파일 저장 (config/isa.json)
        save_mode_config("isa", isa_data)
        # 전역 config에도 반영 (하위호환)
        new_cfg = config.copy()
        new_cfg.update(isa_data)
        save_config(new_cfg)
        st.sidebar.success("ISA 설정을 저장했습니다.")

    # 라이브 백테스트 결과 요약 (Sidebar)
    _live_bt = st.session_state.get("isa_live_bt_result")
    if isinstance(_live_bt, dict) and "metrics" in _live_bt:
        _m = _live_bt["metrics"]
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 라이브 전략 요약")
        sc1, sc2 = st.sidebar.columns(2)
        sc1.metric("총 수익률", f"{_m.get('total_return', 0):+.1f}%")
        sc1.metric("CAGR", f"{_m.get('cagr', 0):.1f}%")
        sc2.metric("MDD", f"{_m.get('mdd', 0):.1f}%", delta_color="inverse")
        sc2.metric("Calmar", f"{_m.get('calmar', 0):.2f}")


    st.sidebar.divider()
    with st.sidebar.expander("📌 ETF별 권장 시작일/비중", expanded=True):
        _ref_df_data = []
        from src.engine.data_cache import WDR_TICKER_START_RATIO
        for _ref_code, _ref_val in WDR_TICKER_START_RATIO.items():
            _ref_df_data.append({
                "종목명": _etf_name_kr(_ref_code),
                "시작일": _ref_val["ref_date"],
                "비중": f"{_ref_val['stock_ratio']*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(_ref_df_data), hide_index=True, use_container_width=True)



    # ── 탭 렌더 전 날짜 위젯 세션 사전 클램프 ──────────────────────────
    # opt 탭: opt_wdr_start 가 min_value 보다 작으면 미리 보정
    _pre_opt_listing = _dc.get_wdr_trade_listing_date(str(isa_etf_code))
    _pre_opt_min = pd.to_datetime(_pre_opt_listing).date() if _pre_opt_listing else pd.to_datetime("2012-01-01").date()
    # ETF 변경 감지 → 시작일 리셋
    if st.session_state.get("isa_opt_etf_sync") != str(isa_etf_code):
        st.session_state["isa_opt_etf_sync"] = str(isa_etf_code)
        st.session_state["opt_wdr_start"] = _pre_opt_min
    # 저장값이 min보다 작으면 클램프
    _pre_opt_cur = st.session_state.get("opt_wdr_start")
    if _pre_opt_cur is not None:
        try:
            if pd.to_datetime(_pre_opt_cur).date() < _pre_opt_min:
                st.session_state["opt_wdr_start"] = _pre_opt_min
        except Exception:
            st.session_state["opt_wdr_start"] = _pre_opt_min

    # live 탭: isa_live_start 가 min_value 보다 작으면 미리 보정
    _pre_live_listing = _dc.get_wdr_trade_listing_date(str(isa_etf_code))
    _pre_live_min = pd.to_datetime(_pre_live_listing).date() if _pre_live_listing else pd.to_datetime("2012-01-01").date()
    if st.session_state.get("isa_live_etf_sync") != str(isa_etf_code):
        st.session_state["isa_live_etf_sync"] = str(isa_etf_code)
        st.session_state["isa_live_start"] = _pre_live_min
        _default_ratio = _dc.get_wdr_default_initial_stock_ratio(str(isa_etf_code))
        if _default_ratio is not None:
            st.session_state["isa_live_ratio"] = round(float(_default_ratio) * 100.0, 2)
    _pre_live_cur = st.session_state.get("isa_live_start")
    if _pre_live_cur is not None:
        try:
            if pd.to_datetime(_pre_live_cur).date() < _pre_live_min:
                st.session_state["isa_live_start"] = _pre_live_min
        except Exception:
            st.session_state["isa_live_start"] = _pre_live_min
    # ─────────────────────────────────────────────────────────────────

    # ── 탭 정의 분리 (공통) ──
    tab_i1, tab_i2, tab_i3, tab_i4, tab_i5, tab_i6, tab_i7, tab_i8 = st.tabs([
        "🚀 실시간 모니터링", "🛒 수동 주문", "📋 주문방식", "💳 수수료/세금",
        "📊 미국 위대리 백테스트", "🔧 위대리 최적화", "🧪 라이브 백테스트", "⏰ 트리거"
    ])

    # ══════════════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링 (잔고 + WDR 시그널)
    # ══════════════════════════════════════════════════════════════
    with tab_i1:
        if not (kis_ak and kis_sk and kis_acct):
            st.warning("KIS ISA API 키와 계좌번호를 설정해 주세요.")
            return

        trader = KISTrader(is_mock=False)
        trader.app_key = kis_ak
        trader.app_secret = kis_sk
        trader.account_no = kis_acct
        trader.acnt_prdt_cd = kis_prdt

        # 공통 토큰 관리 사용
        if not _get_kis_token(trader, kis_acct, kis_ak):
            st.error("KIS 인증에 실패했습니다. API 키/계좌를 확인해 주세요.")
            return

        isa_bal_key = f"isa_balance_cache_{kis_acct}_{kis_prdt}"

        # ISA 데이터 로딩 정책: 로컬 파일 우선 + 부족 시 API 보강
        _isa_local_first_raw = str(config.get("isa_local_first", "1")).strip().lower()
        _isa_local_first = _isa_local_first_raw not in ("0", "false", "no", "off")
        _isa_api_fallback_raw = str(config.get("isa_api_fallback", "1")).strip().lower()
        _isa_api_fallback = _isa_api_fallback_raw not in ("0", "false", "no", "off")

        _isa_px_cache_key = f"isa_price_cache_{kis_acct}_{kis_prdt}"
        _isa_px_cache = st.session_state.get(_isa_px_cache_key)
        if not isinstance(_isa_px_cache, dict):
            _isa_px_cache = {}
            st.session_state[_isa_px_cache_key] = _isa_px_cache

        # ISA 진입 시(QQQ/TQQQ 사용 화면) 미국 일봉 CSV를 세션당 1회 최신 거래일까지 동기화
        _isa_us_sync_key = f"isa_us_wtr_yf_synced_{pd.Timestamp.now().strftime('%Y%m%d')}"
        if not st.session_state.get(_isa_us_sync_key):
            try:
                _dc.fetch_and_cache_yf("QQQ", start="1999-03-10", force_refresh=True)
                _dc.fetch_and_cache_yf("TQQQ", start="2010-02-12", force_refresh=True)
            except Exception as _e:
                logging.warning(f"ISA 미국 데이터 자동 동기화 실패: {_e}")
            finally:
                st.session_state[_isa_us_sync_key] = True

        _isa_chart_cache = st.session_state.setdefault("isa_daily_chart_cache", {})

        def _get_isa_daily_chart(_code: str, count: int = 120, end_date: str | None = None):
            _c = _code_only(_code)
            if not _c:
                return None
            
            _cache_key = f"{_c}_{count}_{end_date}"
            if _cache_key in _isa_chart_cache:
                return _isa_chart_cache[_cache_key]

            if _isa_local_first:
                _df = _dc.get_kis_domestic_local_first(
                    trader,
                    _c,
                    count=int(max(1, count)),
                    end_date=end_date,
                    allow_api_fallback=bool(_isa_api_fallback),
                )
            else:
                try:
                    _df = trader.get_daily_chart(_c, count=int(max(1, count)), end_date=end_date or None)
                except Exception:
                    _df = None
            
            if _df is None or _df.empty:
                return _df
            
            _df = _df.copy().sort_index()
            if "close" not in _df.columns and "Close" in _df.columns:
                _df = _df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            
            _isa_chart_cache[_cache_key] = _df
            return _df

        def _get_isa_current_price(_code: str, ttl_sec: float = 10.0) -> float:
            _c = _code_only(_code)
            if not _c:
                return 0.0
            _now = float(time.time())
            _hit = _isa_px_cache.get(_c)
            if isinstance(_hit, dict) and (_now - float(_hit.get("ts", 0.0))) <= float(ttl_sec):
                return float(_hit.get("val", 0.0) or 0.0)

            _p = 0.0
            _df = _get_isa_daily_chart(_c, count=3)
            if _df is not None and not _df.empty:
                if "close" in _df.columns:
                    _p = float(_df["close"].iloc[-1] or 0.0)
                elif "Close" in _df.columns:
                    _p = float(_df["Close"].iloc[-1] or 0.0)
            if _p <= 0 and not _isa_local_first:
                try:
                    _p = float(trader.get_current_price(_c) or 0.0)
                except Exception:
                    _p = 0.0
            if _p <= 0 and _isa_api_fallback:
                try:
                    _p = float(trader.get_current_price(_c) or 0.0)
                except Exception:
                    _p = 0.0
            _isa_px_cache[_c] = {"ts": _now, "val": float(_p if _p > 0 else 0.0)}
            return float(_p if _p > 0 else 0.0)

        # ── 탭 호출용 공통 컨텍스트 (Optimization, Live Backtest 등에서 공유) ──
        isa_ctx = {
            "trader": trader,
            "isa_etf_code": isa_etf_code,
            "isa_trend_etf_code": isa_trend_etf_code,
            "wdr_eval_mode": wdr_eval_mode,
            "isa_start_date": isa_start_date,
            "isa_seed": isa_seed,
            "get_daily_chart": _get_isa_daily_chart,
            "wdr_ov": wdr_ov,
            "wdr_un": wdr_un,
        }

        st.header("WDR 시그널 모니터링")
        st.caption(f"매매 ETF: {_fmt_etf_code_name(isa_etf_code)} | TREND ETF: {_fmt_etf_code_name(isa_trend_etf_code)}")

        # 잔고 표시
        if isa_bal_key not in st.session_state:
            with st.spinner("ISA 잔고를 조회하는 중..."):
                st.session_state[isa_bal_key] = trader.get_balance()

        rcol1, rcol2 = st.columns([1, 5])
        with rcol1:
            if st.button("잔고 새로고침", key="isa_refresh_balance"):
                with st.spinner("ISA 잔고를 다시 조회하는 중..."):
                    st.session_state[isa_bal_key] = trader.get_balance()
                st.session_state.pop("isa_signal_result", None)
                st.session_state.pop("isa_signal_params", None)
                st.rerun()

        bal = st.session_state.get(isa_bal_key)
        if not bal:
            st.warning("잔고 조회에 실패했습니다. (응답 None — 네트워크 또는 인증 오류)")
        elif bal.get("error"):
            st.error(f"잔고 조회 API 오류: [{bal.get('msg_cd', '')}] {bal.get('msg1', '알 수 없는 오류')}")
        else:
            _bal_sum = _compute_kis_balance_summary(bal)
            buyable_cash = _bal_sum["buyable_cash"]
            holdings = _bal_sum["holdings"]
            stock_eval = _bal_sum["stock_eval"]
            total_eval = _bal_sum["total_eval"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("매수 가능금액", f"{buyable_cash:,.0f} KRW")
            m2.metric("주식 평가", f"{stock_eval:,.0f} KRW")
            m3.metric("총 평가", f"{total_eval:,.0f} KRW")
            m4.metric("보유 종목 수", f"{len(holdings)}")

            if holdings:
                st.dataframe(_format_kis_holdings_df(holdings), use_container_width=True, hide_index=True)

        st.divider()

        # WDR 시그널 자동 계산
        isa_sig_params = {
            "trade_etf": str(isa_etf_code),
            "trend_etf": str(isa_trend_etf_code),
            "ov": float(wdr_ov),
            "un": float(wdr_un),
            "eval_mode": int(wdr_eval_mode),
            "start_date": str(isa_start_date),
            "seed": int(isa_seed),
            "acct": str(kis_acct),
            "prdt": str(kis_prdt),
        }

        def _compute_isa_signal_result():
            sig_df = _get_isa_daily_chart(str(isa_trend_etf_code), count=5000)
            if sig_df is None or len(sig_df) < 260:
                return {"error": f"시그널 데이터가 부족합니다. ({isa_trend_etf_code})"}

            strategy = WDRStrategy(settings={
                "overvalue_threshold": float(wdr_ov),
                "undervalue_threshold": float(wdr_un),
            }, evaluation_mode=int(wdr_eval_mode))
            signal = strategy.analyze(sig_df)
            if not signal:
                return {"error": "WDR 분석에 실패했습니다."}

            bal_local = trader.get_balance() or {"cash": 0.0, "holdings": []}
            cash_local = float(bal_local.get("cash", 0.0))
            holdings_local = bal_local.get("holdings", []) or []
            etf_holding = next((h for h in holdings_local if str(h.get("code", "")) == str(isa_etf_code)), None)
            qty = int(etf_holding.get("qty", 0)) if etf_holding else 0
            cur = _get_isa_current_price(str(isa_etf_code)) or 0.0

            weekly_pnl = 0.0
            ch = _get_isa_daily_chart(str(isa_etf_code), count=10)
            if ch is not None and len(ch) >= 5 and qty > 0 and cur > 0:
                p5 = float(ch["close"].iloc[-5])
                weekly_pnl = (cur - p5) * qty

            action = strategy.get_rebalance_action(
                weekly_pnl=weekly_pnl,
                divergence=float(signal["divergence"]),
                current_shares=qty,
                current_price=float(cur) if cur else 1.0,
                cash=cash_local,
            )

            bt_trade_df = _get_isa_daily_chart(str(isa_etf_code), count=5000)
            bt_res = None
            _trade_first_date = None
            if bt_trade_df is not None and len(bt_trade_df) > 0:
                _trade_first_date = str(bt_trade_df.index[0].date())

            _effective_start = str(isa_start_date)
            if _trade_first_date and str(isa_start_date) < _trade_first_date:
                _effective_start = _trade_first_date

            _ref_stock_ratio = None
            _ref_info = _dc.get_wdr_v10_stock_ratio(str(isa_etf_code), _effective_start)
            if _ref_info:
                _ref_stock_ratio = float(_ref_info.get("stock_ratio", 0.0))
            elif _effective_start > "2022-03-08":
                _ref_sig_df = _get_isa_daily_chart(str(isa_trend_etf_code), count=5000)
                _ref_trade_df = _get_isa_daily_chart(str(isa_etf_code), count=5000)
                if _ref_sig_df is not None and _ref_trade_df is not None:
                    _ref_strat = WDRStrategy(settings={
                        "overvalue_threshold": float(wdr_ov),
                        "undervalue_threshold": float(wdr_un),
                    }, evaluation_mode=int(wdr_eval_mode))
                    _ref_bt = _ref_strat.run_backtest(
                        signal_daily_df=_ref_sig_df,
                        trade_daily_df=_ref_trade_df,
                        initial_balance=float(isa_seed),
                        start_date="2022-03-08",
                    )
                    if _ref_bt and _ref_bt.get("equity_df") is not None:
                        _ref_eq = _ref_bt["equity_df"]
                        _eff_ts = pd.Timestamp(_effective_start)
                        _ref_mask = _ref_eq.index <= _eff_ts
                        if _ref_mask.any():
                            _ref_row = _ref_eq.loc[_ref_mask].iloc[-1]
                            _ref_equity = float(_ref_row["equity"])
                            _ref_sv = float(_ref_row["shares"]) * float(_ref_row["price"])
                            if _ref_equity > 0:
                                _ref_stock_ratio = _ref_sv / _ref_equity
                                _ref_info = {
                                    "source": "backtest",
                                    "signal_etf_code": str(isa_trend_etf_code),
                                    "trade_etf_code": str(isa_etf_code),
                                    "ref_date": str(_ref_eq.loc[_ref_mask].index[-1].date()),
                                    "ref_shares": int(_ref_row["shares"]),
                                    "ref_price": float(_ref_row["price"]),
                                    "ref_cash": float(_ref_row["cash"]),
                                    "ref_equity": _ref_equity,
                                    "stock_ratio": round(_ref_stock_ratio, 4),
                                    "cash_ratio": round(1.0 - _ref_stock_ratio, 4),
                                }

            if sig_df is not None and bt_trade_df is not None:
                bt_res = strategy.run_backtest(
                    signal_daily_df=sig_df,
                    trade_daily_df=bt_trade_df,
                    initial_balance=float(isa_seed),
                    start_date=_effective_start,
                    initial_stock_ratio=_ref_stock_ratio,
                )

            weekly = strategy.daily_to_weekly(sig_df)
            trend = strategy.calc_growth_trend(weekly)

            return {
                "signal": signal,
                "action": action,
                "weekly_df": weekly,
                "trend": trend,
                "balance": bal_local,
                "bt_res": bt_res,
                "cur_price": cur,
                "trade_first_date": _trade_first_date,
                "effective_start": _effective_start,
                "ref_info": _ref_info,
            }

        if st.session_state.get("isa_signal_result") is None or st.session_state.get("isa_signal_params") != isa_sig_params:
            with st.spinner("WDR 시그널 및 백테스트 계산 중..."):
                st.session_state["isa_signal_result"] = _compute_isa_signal_result()
                st.session_state["isa_signal_params"] = isa_sig_params
                if isinstance(st.session_state["isa_signal_result"], dict) and st.session_state["isa_signal_result"].get("balance"):
                    st.session_state[isa_bal_key] = st.session_state["isa_signal_result"]["balance"]
                _computed = st.session_state["isa_signal_result"]
                if isinstance(_computed, dict) and _computed.get("trade_first_date"):
                    _tfd = _computed["trade_first_date"]
                    if str(isa_start_date) < _tfd:
                        st.rerun()

        res = st.session_state.get("isa_signal_result")
        if res:
            if res.get("error"):
                st.error(res["error"])
            else:
                sig = res["signal"]
                bt = res.get("bt_res")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("이격도", f"{float(sig['divergence']):+.2f}%")
                c2.metric("시장 상태", str(sig["state"]))
                c3.metric("매도 비율", f"{float(sig['sell_ratio']):.1f}%")
                c4.metric("매수 비율", f"{float(sig['buy_ratio']):.1f}%")

                _bal_valid = bal and not bal.get("error")
                actual_cash_v = float(bal.get("cash", 0.0)) if _bal_valid else 0.0
                actual_hlds = (bal.get("holdings", []) or []) if _bal_valid else []
                actual_etf = next((h for h in actual_hlds if str(h.get("code", "")) == str(isa_etf_code)), None)
                actual_shares = int(actual_etf.get("qty", 0)) if actual_etf else 0
                actual_price = res.get("cur_price", 0)
                actual_eval = actual_shares * actual_price
                actual_total = actual_cash_v + actual_eval
                actual_stock_pct = actual_eval / actual_total * 100 if actual_total > 0 else 0

                bt_shares = 0
                bt_equity = 0.0
                bt_stock_ratio_pct = 0.0
                if bt:
                    eq_df = bt["equity_df"]
                    bt_last = eq_df.iloc[-1]
                    bt_shares = int(bt_last["shares"])
                    bt_equity = float(bt_last["equity"])
                    bt_stock_ratio_pct = (bt_shares * float(bt_last["price"])) / bt_equity * 100 if bt_equity > 0 else 0

                _sync_action = None
                _sync_qty = 0
                _sync_reason = ""
                if bt and _bal_valid and actual_price > 0:
                    _bt_sr = bt_stock_ratio_pct / 100
                    _target_stock_val = actual_total * _bt_sr
                    _target_shares = int(_target_stock_val / actual_price)
                    _diff = _target_shares - actual_shares
                    
                    if _diff > 0:
                        _sync_action = "BUY"
                        _sync_qty = _diff
                        if (_diff * actual_price) > actual_cash_v:
                            _sync_qty = int(actual_cash_v * 0.95 / actual_price)
                            _sync_reason = f"예수금 부족으로 {_sync_qty}주만 매수 가능"
                        else:
                            _sync_reason = f"목표 비중 {bt_stock_ratio_pct:.1f}% 맞추기"
                    elif _diff < 0:
                        _sync_action = "SELL"
                        _sync_qty = abs(_diff)
                        _sync_reason = f"목표 비중 {bt_stock_ratio_pct:.1f}% 맞추기"
                    else:
                        _sync_reason = "백테스트와 동기화 완료"

                st.info(f"**백테스트 목표 주식비율**: {bt_stock_ratio_pct:.1f}% | **실제 주식비율**: {actual_stock_pct:.1f}% | **현재가**: {actual_price:,.0f}원")

                sc1, sc2 = st.columns(2)
                with sc1:
                    _qty_display = "HOLD"
                    if _sync_action == "BUY" and _sync_qty > 0: _qty_display = f"+{_sync_qty}주 매수"
                    elif _sync_action == "SELL" and _sync_qty > 0: _qty_display = f"-{_sync_qty}주 매도"
                    
                    if _sync_qty > 0:
                        getattr(st, "error" if _sync_action == "SELL" else "success")(f"### 권장 주문: **{_qty_display}**")
                    else:
                        st.success(f"### 권장 주문: **{_qty_display}**")
                    st.caption(f"실제 {actual_shares}주 보유 · 사유: {_sync_reason}")

                if bt:
                    m = bt["metrics"]
                    with sc2:
                        st.write(f"**백테스트 성과 ({res.get('effective_start')} ~)**")
                        st.write(f"수익률: **{m['total_return']:+.2f}%** | MDD: **{m['mdd']:.2f}%**")
                        st.write(f"최종자산: {bt_equity:,.0f}원 ({isa_seed/10000:.0f}만원 기준)")

                # 차트 렌더링
                _chart_start_ts = pd.Timestamp(isa_start_date)
                # 시그널 차트는 최소 52주(1년) 분량을 보여주도록 설정 (오늘 기준 1년 전과 설정 시작일 중 더 과거 시점부터)
                _1year_ago = pd.Timestamp.now() - pd.Timedelta(weeks=52)
                _display_start_ts = min(_chart_start_ts, _1year_ago)

                weekly_df = res.get("weekly_df")
                trend = res.get("trend")
                if weekly_df is not None and trend is not None:
                    mask = weekly_df.index >= _display_start_ts
                    w_plot = weekly_df.loc[mask]
                    t_plot = np.asarray(trend)[mask]

                    if len(w_plot) > 0:
                        fig = go.Figure()
                        
                        # 현재 시각 기준 미래 데이터(가상 포인트) 유무 확인
                        _now_ts = pd.Timestamp.now().normalize()
                        _is_virtual = w_plot.index[-1] > _now_ts
                        
                        if _is_virtual:
                            # 마지막 포인트(가상)와 이전 포인트들(확정) 분리
                            w_confirmed = w_plot.iloc[:-1]
                            w_virtual = w_plot.iloc[-1:]
                            
                            fig.add_trace(go.Scatter(
                                x=w_confirmed.index, y=w_confirmed["close"], 
                                name="시그널 가격(확정)", line=dict(color="royalblue")
                            ))
                            fig.add_trace(go.Scatter(
                                x=w_virtual.index, y=w_virtual["close"], 
                                name="시그널 가격(실시간/가상)", 
                                mode='lines+markers',
                                marker=dict(symbol='star', size=12, color='royalblue'),
                                line=dict(color="royalblue", dash='dot')
                            ))
                        else:
                            fig.add_trace(go.Scatter(x=w_plot.index, y=w_plot["close"], name="시그널 가격", line=dict(color="royalblue")))

                        fig.add_trace(go.Scatter(x=w_plot.index, y=t_plot, name="성장 추세", line=dict(color="orange", dash="dash")))
                        
                        ov_th = float(wdr_ov) / 100.0
                        un_th = float(wdr_un) / 100.0
                        fig.add_trace(go.Scatter(x=w_plot.index, y=t_plot*(1+ov_th), name=f"고평가(+{wdr_ov}%)", line=dict(color="red", dash="dot", width=1)))
                        fig.add_trace(go.Scatter(x=w_plot.index, y=t_plot*(1+un_th), name=f"저평가({wdr_un}%)", line=dict(color="green", dash="dot", width=1)))
                        
                        fig.update_layout(
                            height=400, 
                            margin=dict(l=0,r=0,t=40,b=0), 
                            legend=dict(orientation="h", y=1.1),
                            xaxis=dict(range=[_display_start_ts, w_plot.index[-1]])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        if _is_virtual:
                            st.caption(f"💡 현재 시각이 금요일 장 마감 전이므로, 가장 최근 데이터({pd.Timestamp.now().strftime('%m/%d')})를 이번 주 금요일({w_plot.index[-1].strftime('%m/%d')})의 가상 데이터로 표시하고 있습니다. (별표로 표시)")


                if bt:
                    eq_df = bt["equity_df"]
                    _eq_ret = (eq_df["equity"] / float(isa_seed) - 1) * 100
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(x=eq_df.index, y=_eq_ret.values, name="전략 수익률", line=dict(color="gold", width=2)))
                    fig_eq.update_layout(
                        height=300, 
                        margin=dict(l=0,r=0,t=40,b=0),
                        xaxis=dict(range=[_display_start_ts, eq_df.index[-1]]), # X축 범위 통일
                        hovermode="x unified"
                    )
                    fig_eq = _apply_return_hover_format(fig_eq, apply_all=True)
                    st.plotly_chart(fig_eq, use_container_width=True)


                    _render_performance_analysis(
                        equity_series=eq_df["equity"],
                        benchmark_series=bt.get("benchmark_df"),
                        strategy_metrics=bt.get("metrics"),
                        strategy_label="전략",
                        benchmark_label=f"{_fmt_etf_code_name(str(isa_trend_etf_code))} (시그널)",
                        show_drawdown=True,
                        show_weight=True,
                        equity_df=eq_df
                    )

    with tab_i2: st.info("수동 주문 기능 준비 중")
    with tab_i3: st.info("주문 방식 가이드 준비 중")
    with tab_i4: st.info("수수료/세금 정보 준비 중")
    with tab_i5: st.info("미국 백테스트 상세 뷰 준비 중")
    with tab_i6:
        render_isa_opt_tab(isa_ctx)

    with tab_i7:
        render_isa_live_tab(isa_ctx)
    with tab_i8:
        render_strategy_trigger_tab("ISA")
