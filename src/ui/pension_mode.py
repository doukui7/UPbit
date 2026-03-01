import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import plotly.graph_objects as go
from src.engine.kis_trader import KISTrader
from src.strategy.laa import LAAStrategy
from src.strategy.dual_momentum import DualMomentumStrategy
from src.strategy.vaa import VAAStrategy
from src.strategy.cdm import CDMStrategy
import src.engine.data_cache as _dc
from src.constants import IS_CLOUD
from src.utils.formatting import (
    _etf_name_kr, _fmt_etf_code_name, _code_only, 
    _format_kis_holdings_df, _safe_float, _safe_int
)
from src.utils.helpers import _get_runtime_value
from src.utils.kis import _get_kis_token, _compute_kis_balance_summary
from src.ui.components.performance import (
    _render_performance_analysis, _apply_return_hover_format, _apply_dd_hover_format
)

# ── Helpers ──

def _sidebar_etf_code_input(title: str, code_value: str, key: str, disabled: bool = False) -> str:
    code = _code_only(code_value)
    if not st.session_state.get("_etf_code_input_css_loaded", False):
        st.sidebar.markdown(
            """
            <style>
            section[data-testid="stSidebar"] input[aria-label="종목번호"]{
                font-size:0.90rem !important;
                font-weight:500 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_etf_code_input_css_loaded"] = True

    st.sidebar.markdown(
        f"<div style='font-size:1.02rem; font-weight:700; margin:0.05rem 0 0.28rem 0;'>{_etf_name_kr(code)}"
        f" <span style='font-size:0.92rem; font-weight:600; color:#6b7280;'>({title})</span></div>",
        unsafe_allow_html=True,
    )
    code_col, _spacer_col = st.sidebar.columns([0.9, 2.1])
    with code_col:
        typed_code = st.text_input(
            "종목번호",
            value=code,
            key=key,
            max_chars=6,
            disabled=disabled,
            label_visibility="collapsed",
        )
    return _code_only(typed_code)

def _normalize_numeric_series(df: pd.DataFrame | None, preferred_cols=("close", "Close")):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for c in preferred_cols:
        if c in df.columns:
            return df[c].astype(float)
    return pd.Series(dtype=float)

# ── Main Entry ──

def render_kis_pension_mode(config, save_config):
    """KIS 연금저축 포트폴리오 모드 - 다중 전략 지원."""
    
    PEN_STRATEGIES = ["LAA", "듀얼모멘텀", "VAA", "CDM", "정적배분"]

    st.title("연금저축 포트폴리오")
    st.sidebar.header("연금저축 설정")
    
    # ── 기초 설정 ──
    _pen_bt_start_raw = str(config.get("kis_pension_start_date", config.get("start_date", "2020-01-01")) or "2020-01-01")
    try:
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    except Exception:
        _pen_bt_start_raw = "2020-01-01"
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    _pen_bt_cap_default = int(float(config.get("kis_pension_initial_cap", 10_000_000) or 10_000_000))

    kis_ak = _get_runtime_value(("KIS_PENSION_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(config.get("kis_pension_account_no", "") or _get_runtime_value(("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(config.get("kis_pension_prdt_cd", "") or _get_runtime_value(("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))

    _raw_acct = kis_acct.replace("-", "").strip()
    if len(_raw_acct) == 10 and (not kis_prdt or kis_prdt == "01"):
        kis_acct = _raw_acct[:8]
        kis_prdt = _raw_acct[8:]

    if not IS_CLOUD:
        with st.sidebar.expander("KIS API 키", expanded=False):
            kis_ak = st.text_input("앱 키", value=kis_ak, type="password", key="pen_app_key")
            kis_sk = st.text_input("시크릿 키", value=kis_sk, type="password", key="pen_app_secret")
            kis_acct = st.text_input("계좌번호 (앞 8자리)", value=kis_acct, key="pen_account_no")
            kis_prdt = st.text_input("상품코드 (뒤 2자리)", value=kis_prdt, key="pen_prdt_cd")

    st.sidebar.subheader("공통 설정")
    _pen_bt_start_date = st.sidebar.date_input("기준 시작일", value=_pen_bt_start_ts.date(), key="pen_common_start_date")
    _pen_bt_cap = st.sidebar.number_input("초기 자본금", min_value=100_000, value=int(_pen_bt_cap_default), step=100_000, key="pen_common_initial_cap")
    _pen_bt_start_raw = str(_pen_bt_start_date)
    _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()

    # ── 포트폴리오 에디터 ──
    def _normalize_pen_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            df = pd.DataFrame([{"strategy": "LAA", "weight": 100}, {"strategy": "듀얼모멘텀", "weight": 0}])
        if "strategy" not in df.columns: df["strategy"] = "LAA"
        if "weight" not in df.columns: df["weight"] = 0
        out = df[["strategy", "weight"]].copy()
        out["strategy"] = out["strategy"].astype(str).str.strip()
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0)
        
        # Ensure default strategies exist
        for s in ["LAA", "듀얼모멘텀"]:
            if not (out["strategy"] == s).any():
                out = pd.concat([out, pd.DataFrame([{"strategy": s, "weight": 0}])], ignore_index=True)
        return out

    _saved_portfolio = config.get("pension_portfolio", [{"strategy": "LAA", "weight": 100}, {"strategy": "듀얼모멘텀", "weight": 0}])
    _pen_port_state_key = "pen_portfolio_editor_df"
    if _pen_port_state_key not in st.session_state:
        st.session_state[_pen_port_state_key] = _normalize_pen_portfolio_df(pd.DataFrame(_saved_portfolio))

    _pen_port_edited = st.sidebar.data_editor(
        st.session_state[_pen_port_state_key],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="pen_portfolio_editor",
        column_config={
            "strategy": st.column_config.SelectboxColumn("전략", options=PEN_STRATEGIES, required=True),
            "weight": st.column_config.NumberColumn("비중(%)", min_value=0, max_value=100, step=5, required=True),
        }
    )
    _pen_port_edited = _normalize_pen_portfolio_df(_pen_port_edited)
    st.session_state[_pen_port_state_key] = _pen_port_edited.copy()

    _total_w = float(_pen_port_edited["weight"].sum())
    if _total_w > 100:
        st.sidebar.error(f"비중 합계: {_total_w:.0f}% (100% 이하여야 합니다)")
    else:
        st.sidebar.caption(f"투자 비중: {_total_w:.0f}% | 현금: {100-_total_w:.0f}%")

    _active_strategies = list(_pen_port_edited["strategy"].unique())

    # ── 전략별 상세 설정 패널 ──
    _panel_options = ["접기"]
    for s in _active_strategies:
        _panel_options.append(f"{s} 설정")
    
    _panel_key = "pen_strategy_settings_panel"
    if st.session_state.get(_panel_key) not in _panel_options:
        st.session_state[_panel_key] = _panel_options[0]
    _selected_panel = st.sidebar.radio("전략 상세 설정", _panel_options, key=_panel_key)

    # LAA Settings
    kr_spy_laa = _sidebar_etf_code_input("LAA SPY", config.get("pen_laa_spy", "360750"), "pen_laa_spy") if _selected_panel == "LAA 설정" else config.get("pen_laa_spy", "360750")
    
    # Dual Momentum Settings
    dm_lookback = 12
    if _selected_panel == "듀얼모멘텀 설정":
        dm_lookback = st.sidebar.number_input("DM 룩백(개월)", 1, 24, 12, key="pen_dm_lookback")

    if st.sidebar.button("설정 저장"):
        new_cfg = config.copy()
        new_cfg["pension_portfolio"] = _pen_port_edited.to_dict("records")
        # 추가 설정들 저장...
        save_config(new_cfg)
        st.sidebar.success("저장 완료")

    # ── 탭 렌더링 ──
    tab1, tab2, tab3 = st.tabs(["🚀 실시간 모니터링", "🧪 백테스트", "📖 가이드"])

    with tab1:
        # ── KIS 인증 (모니터링 탭 내부로 격리) ──
        if not (kis_ak and kis_sk and kis_acct):
            st.warning("KIS API 설정을 완료해주세요.")
            return

        trader = KISTrader(is_mock=False)
        trader.app_key, trader.app_secret, trader.account_no, trader.acnt_prdt_cd = kis_ak, kis_sk, kis_acct, kis_prdt

        if not _get_kis_token(trader, kis_acct, kis_ak):
            st.error("KIS 인증 실패")
            return

        # ── 데이터 로딩 헬퍼 ──
        def _get_pen_daily_chart(code, count=420):
            c = _code_only(code)
            if not c: return None
            return _dc.get_kis_domestic_local_first(trader, c, count=count, allow_api_fallback=True)

        pen_bal_key = f"pension_balance_cache_{kis_acct}"
        if pen_bal_key not in st.session_state:
            st.session_state[pen_bal_key] = trader.get_balance()
        
        bal = st.session_state[pen_bal_key]
        bal_sum = _compute_kis_balance_summary(bal)

        st.subheader("계좌 현황")
        m1, m2, m3 = st.columns(3)
        m1.metric("총 평가금액", f"{bal_sum['total_eval']:,.0f}원")
        m2.metric("매수 가능금액", f"{bal_sum['buyable_cash']:,.0f}원")
        m3.metric("보유 종목수", f"{len(bal_sum['holdings'])}개")
        
        if bal_sum["holdings"]:
            st.dataframe(_format_kis_holdings_df(bal_sum["holdings"]), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("전략 통합 시그널")
        st.info("시그널 계산 및 리밸런싱 제안 기능 구현 중...")

    with tab2:
        st.info("포트폴리오 통합 백테스트 기능 준비 중")

    with tab3:
        st.markdown("""
        ### 연금저축 포트폴리오 운용 가이드
        1. **LAA (Lethargic Asset Allocation)**: 동적 자산배분 전략으로 리스크에 따라 자산을 변경합니다.
        2. **듀얼모멘텀**: 상대 모멘텀과 절대 모멘텀을 결합하여 강한 자산에 집중합니다.
        3. **VAA/CDM**: 가속 모멘텀 및 복합 모멘텀을 활용한 고도화된 전략입니다.
        """)
