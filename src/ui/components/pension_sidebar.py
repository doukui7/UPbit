"""연금저축 사이드바 설정 (포트폴리오 구성, ETF 매핑, 전략 설정, 저장)."""
import streamlit as st
import pandas as pd

from src.constants import IS_CLOUD
from src.utils.formatting import _code_only, _etf_name_kr, _fmt_etf_code_name
from src.utils.helpers import _get_runtime_value

PEN_STRATEGIES = ["LAA", "듀얼모멘텀", "VAA"]


def _sidebar_etf_code_input(title: str, code_value: str, key: str, disabled: bool = False) -> str:
    """사이드바 ETF 코드 입력 위젯."""
    _raw = str(code_value or "").strip()
    _name = _etf_name_kr(_raw)
    _label = f"{title}"
    if _name:
        _label = f"{title} ({_name})"
    val = st.text_input(
        _label,
        value=_raw,
        key=key,
        disabled=disabled,
        help=f"현재: {_raw} ({_name})" if _name else f"현재: {_raw}",
    )
    val = _code_only(val)
    if val and val != _raw:
        _new_name = _etf_name_kr(val)
        if _new_name:
            st.caption(f"→ {_new_name}")
        else:
            st.caption(f"→ {val} (이름 미확인)")
    return val


def _normalize_pen_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame([{"strategy": "LAA", "weight": 100}])
    if "strategy" not in df.columns:
        df["strategy"] = "LAA"
    if "weight" not in df.columns:
        df["weight"] = 0
    out = df[["strategy", "weight"]].copy()
    out["strategy"] = out["strategy"].astype(str).str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0)

    if not (out["strategy"] == "LAA").any():
        out = pd.concat([pd.DataFrame([{"strategy": "LAA", "weight": 0}]), out], ignore_index=True)
    if not (out["strategy"] == "듀얼모멘텀").any():
        out = pd.concat([out, pd.DataFrame([{"strategy": "듀얼모멘텀", "weight": 0}])], ignore_index=True)
    if not (out["strategy"] == "VAA").any():
        out = pd.concat([out, pd.DataFrame([{"strategy": "VAA", "weight": 0}])], ignore_index=True)

    out = out[~out["strategy"].isin(["정적배분", "CDM"])].reset_index(drop=True)
    return out


def render_pension_sidebar(config: dict, save_config) -> dict | None:
    """사이드바 설정을 렌더링하고 설정값 dict를 반환.

    Returns None if KIS API keys are missing (caller should return early).
    """
    from src.utils.helpers import load_mode_config, save_mode_config
    pen_cfg = load_mode_config("pension")

    st.title("연금저축 포트폴리오")
    st.sidebar.header("연금저축 설정")

    from src.ui.components.asset_mgmt import get_earliest_start_date
    _am_pen_earliest = get_earliest_start_date("pension")
    _pen_bt_start_raw = str(
        _am_pen_earliest
        or pen_cfg.get("kis_pension_start_date", config.get("start_date", "2020-01-01"))
        or "2020-01-01"
    )
    try:
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    except Exception:
        _pen_bt_start_raw = "2020-01-01"
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    _pen_bt_cap_default = int(float(pen_cfg.get("kis_pension_initial_cap", 10_000_000) or 10_000_000))

    kis_ak = _get_runtime_value(("KIS_PENSION_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(pen_cfg.get("kis_pension_account_no", "") or _get_runtime_value(("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(pen_cfg.get("kis_pension_prdt_cd", "") or _get_runtime_value(("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))
    _raw_acct = kis_acct.replace("-", "").strip()
    if len(_raw_acct) == 10 and kis_prdt in ("01", ""):
        kis_acct = _raw_acct[:8]
        kis_prdt = _raw_acct[8:]

    if not IS_CLOUD:
        with st.sidebar.expander("KIS API 키", expanded=False):
            kis_ak = st.text_input("앱 키", value=kis_ak, type="password", key="pen_app_key")
            kis_sk = st.text_input("시크릿 키", value=kis_sk, type="password", key="pen_app_secret")
            kis_acct = st.text_input("계좌번호 (앞 8자리)", value=kis_acct, key="pen_account_no", help="10자리 입력 시 자동으로 앞8+뒤2 분리")
            kis_prdt = st.text_input("상품코드 (뒤 2자리)", value=kis_prdt, key="pen_prdt_cd")
            _raw2 = kis_acct.replace("-", "").strip()
            if len(_raw2) == 10:
                kis_acct = _raw2[:8]
                kis_prdt = _raw2[8:]

    st.sidebar.subheader("공통 설정")
    _pen_bt_start_date = st.sidebar.date_input(
        "기준 시작일",
        value=_pen_bt_start_ts.date(),
        key="pen_common_start_date",
        help="연금저축 전략 성과/차트/백테스트 공통 기준 시작일",
    )
    _pen_bt_cap = st.sidebar.number_input(
        "초기 자본금 (KRW - 원 단위)",
        min_value=100_000,
        value=int(_pen_bt_cap_default),
        step=100_000,
        format="%d",
        key="pen_common_initial_cap",
        help="연금저축 전략 성과/차트/백테스트 공통 기준 초기자본",
    )
    _pen_bt_start_raw = str(_pen_bt_start_date)
    _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    st.sidebar.caption(f"설정: 시작일 {_pen_bt_start_raw} | 초기 자본 {_pen_bt_cap:,.0f} KRW")

    # ── 포트폴리오 편집기 ──
    st.sidebar.subheader("포트폴리오")
    _saved_portfolio = pen_cfg.get("pension_portfolio", [{"strategy": "LAA", "weight": 100}])
    if not _saved_portfolio:
        _saved_portfolio = [{"strategy": "LAA", "weight": 100}]
    _pen_port_df = _normalize_pen_portfolio_df(pd.DataFrame(_saved_portfolio))

    _pen_port_state_key = "pen_portfolio_editor_df"
    if _pen_port_state_key not in st.session_state or not isinstance(st.session_state.get(_pen_port_state_key), pd.DataFrame):
        st.session_state[_pen_port_state_key] = _normalize_pen_portfolio_df(_pen_port_df)
    else:
        _state_df = st.session_state[_pen_port_state_key].copy()
        st.session_state[_pen_port_state_key] = _normalize_pen_portfolio_df(_state_df)

    def _with_no(df):
        _view = df.copy()
        _view.insert(0, "no", range(1, len(_view) + 1))
        return _view

    _pen_port_edited = st.sidebar.data_editor(
        _with_no(st.session_state[_pen_port_state_key]),
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="pen_portfolio_editor",
        column_config={
            "no": st.column_config.NumberColumn("번호", min_value=1, step=1),
            "strategy": st.column_config.SelectboxColumn("전략", options=PEN_STRATEGIES, required=True),
            "weight": st.column_config.NumberColumn("비중(%)", min_value=0, max_value=100, step=5, required=True),
        },
        disabled=["no"],
    )
    if "no" in _pen_port_edited.columns:
        _pen_port_edited = _pen_port_edited.drop(columns=["no"], errors="ignore")
    if "strategy" not in _pen_port_edited.columns:
        _pen_port_edited["strategy"] = "LAA"
    if "weight" not in _pen_port_edited.columns:
        _pen_port_edited["weight"] = 0
    _pen_port_edited["weight"] = pd.to_numeric(_pen_port_edited["weight"], errors="coerce").fillna(0)

    _normalized_edited = _normalize_pen_portfolio_df(_pen_port_edited)
    _need_editor_resync = (
        len(_normalized_edited) != len(_pen_port_edited)
        or not (_normalized_edited["strategy"] == "LAA").any()
        or not (_normalized_edited["strategy"] == "듀얼모멘텀").any()
    )
    st.session_state[_pen_port_state_key] = _normalized_edited[["strategy", "weight"]].copy()
    _pen_port_edited = _normalized_edited.copy()
    if _need_editor_resync:
        if "pen_portfolio_editor" in st.session_state:
            del st.session_state["pen_portfolio_editor"]
        st.rerun()

    _total_w = float(pd.to_numeric(_pen_port_edited["weight"], errors="coerce").fillna(0).sum()) if not _pen_port_edited.empty else 0.0
    if _total_w > 100:
        st.sidebar.error(f"비중 합계: {_total_w:.0f}% (100% 이하여야 합니다)")
    else:
        _cash_w = max(0.0, 100.0 - _total_w)
        st.sidebar.caption(f"투자 비중: {_total_w:.0f}% | 현금: {_cash_w:.0f}%")

    _active_strategies = list(_pen_port_edited["strategy"].unique()) if not _pen_port_edited.empty else []
    _auto_signal_strategies = list(_pen_port_edited["strategy"].unique()) if not _pen_port_edited.empty else []

    _panel_options = ["접기", "LAA 전략 설정", "듀얼모멘텀 설정", "VAA 전략 설정"]
    _panel_key = "pen_strategy_settings_panel"
    if st.session_state.get(_panel_key) not in _panel_options:
        st.session_state[_panel_key] = _panel_options[0]
    st.sidebar.caption("전략 상세 설정 (하나만 펼치기)")
    _selected_panel = st.sidebar.radio(
        "전략 상세 설정", _panel_options, key=_panel_key, label_visibility="collapsed",
    )

    # ── deprecated ETF 코드 자동 교체 (config 변경 반영) ──
    _deprecated_etf_map = {
        "114470": "329750",  # KODEX 국고채3년 → TIGER 미국달러단기채권액티브
        "453540": "305080",  # TIGER 회사채 → TIGER 미국채10년선물
        "453850": "251350",  # ACE 미국30년국채 → KODEX 선진국MSCI World
        "295820": "195980",  # PLUS 200동일가중 → PLUS 신흥국MSCI
        "132030": "411060",  # KODEX Gold선물 → ACE KRX금현물
    }
    _etf_session_keys = [
        "pen_laa_spy", "pen_laa_iwd", "pen_laa_gld", "pen_laa_ief", "pen_laa_qqq", "pen_laa_shy",
        "pen_dm_kr_spy", "pen_dm_kr_efa", "pen_dm_kr_agg", "pen_dm_kr_bil",
        "pen_vaa_kr_spy", "pen_vaa_kr_efa", "pen_vaa_kr_eem", "pen_vaa_kr_agg",
        "pen_vaa_kr_lqd", "pen_vaa_kr_ief", "pen_vaa_kr_shy",
    ]
    for _sk in _etf_session_keys:
        _sv = str(st.session_state.get(_sk, "")).strip()
        if _sv in _deprecated_etf_map:
            # del로 삭제해야 Streamlit 위젯이 config 기본값으로 재초기화됨
            # (재할당은 위젯 렌더 시 브라우저 캐시 값으로 다시 덮어써짐)
            del st.session_state[_sk]

    # ── LAA 전략 설정 ──
    _kr_iwd_default = _code_only(pen_cfg.get("kr_etf_laa_iwd", _get_runtime_value("KR_ETF_LAA_IWD", _get_runtime_value("KR_ETF_SPY", "360750"))))
    _kr_spy_default = _code_only(pen_cfg.get("kr_etf_laa_spy", _get_runtime_value("KR_ETF_LAA_SPY", _get_runtime_value("KR_ETF_SPY", _kr_iwd_default or "360750"))))
    _kr_gld_default = _code_only(pen_cfg.get("kr_etf_laa_gld", _get_runtime_value("KR_ETF_LAA_GLD", "411060")))
    _kr_ief_default = _code_only(pen_cfg.get("kr_etf_laa_ief", _get_runtime_value("KR_ETF_LAA_IEF", _get_runtime_value("KR_ETF_AGG", "305080"))))
    _kr_qqq_default = _code_only(pen_cfg.get("kr_etf_laa_qqq", _get_runtime_value("KR_ETF_LAA_QQQ", "133690")))
    _kr_shy_default = _code_only(pen_cfg.get("kr_etf_laa_shy", _get_runtime_value("KR_ETF_LAA_SHY", "329750")))
    def _safe_etf(key, default):
        v = _code_only(st.session_state.get(key, default))
        return _deprecated_etf_map.get(v, v)

    kr_spy = _safe_etf("pen_laa_spy", _kr_spy_default)
    kr_iwd = _safe_etf("pen_laa_iwd", _kr_iwd_default)
    kr_gld = _safe_etf("pen_laa_gld", _kr_gld_default)
    kr_ief = _safe_etf("pen_laa_ief", _kr_ief_default)
    kr_qqq = _safe_etf("pen_laa_qqq", _kr_qqq_default)
    kr_shy = _safe_etf("pen_laa_shy", _kr_shy_default)

    if ("LAA" in _active_strategies) and (_selected_panel == "LAA 전략 설정"):
        with st.sidebar.expander("LAA 전략 설정", expanded=True):
            st.caption("LAA 전략 전용 설정")
            kr_spy = _sidebar_etf_code_input("SPY 신호 ETF", kr_spy, key="pen_laa_spy", disabled=IS_CLOUD)
            kr_iwd = _sidebar_etf_code_input("IWD 대체 ETF", kr_iwd, key="pen_laa_iwd", disabled=IS_CLOUD)
            kr_gld = _sidebar_etf_code_input("GLD 대체 ETF", kr_gld, key="pen_laa_gld", disabled=IS_CLOUD)
            kr_ief = _sidebar_etf_code_input("IEF 대체 ETF", kr_ief, key="pen_laa_ief", disabled=IS_CLOUD)
            kr_qqq = _sidebar_etf_code_input("QQQ 대체 ETF", kr_qqq, key="pen_laa_qqq", disabled=IS_CLOUD)
            kr_shy = _sidebar_etf_code_input("SHY 대체 ETF", kr_shy, key="pen_laa_shy", disabled=IS_CLOUD)

    _kr_etf_map = {"SPY": str(kr_spy), "IWD": str(kr_iwd), "GLD": str(kr_gld), "IEF": str(kr_ief), "QQQ": str(kr_qqq), "SHY": str(kr_shy)}
    _kr_etf_map = {k: _deprecated_etf_map.get(v, v) for k, v in _kr_etf_map.items()}

    # ── 듀얼모멘텀 전략 설정 ──
    _dm_offensive = ["SPY", "EFA"]
    _dm_defensive = ["AGG"]
    _dm_canary = ["BIL"]
    _dm_lookback = int(st.session_state.get("pen_dm_lookback", pen_cfg.get("pen_dm_lookback", 12)))
    _dm_td = int(st.session_state.get("pen_dm_trading_days", pen_cfg.get("pen_dm_trading_days", 22)))
    _dm_w1 = float(st.session_state.get("pen_dm_w1", pen_cfg.get("pen_dm_w1", 12.0)))
    _dm_w3 = float(st.session_state.get("pen_dm_w3", pen_cfg.get("pen_dm_w3", 4.0)))
    _dm_w6 = float(st.session_state.get("pen_dm_w6", pen_cfg.get("pen_dm_w6", 2.0)))
    _dm_w12 = float(st.session_state.get("pen_dm_w12", pen_cfg.get("pen_dm_w12", 1.0)))

    _dm_kr_spy_default = _code_only(pen_cfg.get("pen_dm_kr_spy", pen_cfg.get("pen_dm_agg_etf", _get_runtime_value("KR_ETF_SPY", "360750"))))
    _dm_kr_efa_default = _code_only(pen_cfg.get("pen_dm_kr_efa", _get_runtime_value("KR_ETF_EFA", "251350")))
    _dm_kr_agg_default = _code_only(pen_cfg.get("pen_dm_kr_agg", pen_cfg.get("pen_dm_def_etf", _get_runtime_value("KR_ETF_AGG", "305080"))))
    _dm_kr_bil_default = _code_only(pen_cfg.get("pen_dm_kr_bil", _get_runtime_value("KR_ETF_BIL", _get_runtime_value("KR_ETF_SHY", "329750"))))
    _dm_kr_spy = _safe_etf("pen_dm_kr_spy", _dm_kr_spy_default)
    _dm_kr_efa = _safe_etf("pen_dm_kr_efa", _dm_kr_efa_default)
    _dm_kr_agg = _safe_etf("pen_dm_kr_agg", _dm_kr_agg_default)
    _dm_kr_bil = _safe_etf("pen_dm_kr_bil", _dm_kr_bil_default)

    if ("듀얼모멘텀" in _active_strategies) and (_selected_panel == "듀얼모멘텀 설정"):
        with st.sidebar.expander("듀얼모멘텀 설정", expanded=True):
            st.caption("듀얼모멘텀 전략 전용 설정")
            dmc1, dmc2 = st.columns(2)
            _dm_lookback = dmc1.number_input(
                "카나리아 룩백(개월)", min_value=1, max_value=24,
                value=int(_dm_lookback), step=1, key="pen_dm_lookback",
                help="기본 12개월 (절대 모멘텀 기준)",
            )
            _dm_td = dmc2.number_input(
                "월 환산 거래일", min_value=18, max_value=24,
                value=int(_dm_td), step=1, key="pen_dm_trading_days", help="기본 22일",
            )
            st.markdown("**모멘텀 가중치 (1M/3M/6M/12M)**")
            w1c, w3c, w6c, w12c = st.columns(4)
            _dm_w1 = w1c.number_input("1M", min_value=0.0, max_value=30.0, value=float(_dm_w1), step=0.5, key="pen_dm_w1")
            _dm_w3 = w3c.number_input("3M", min_value=0.0, max_value=30.0, value=float(_dm_w3), step=0.5, key="pen_dm_w3")
            _dm_w6 = w6c.number_input("6M", min_value=0.0, max_value=30.0, value=float(_dm_w6), step=0.5, key="pen_dm_w6")
            _dm_w12 = w12c.number_input("12M", min_value=0.0, max_value=30.0, value=float(_dm_w12), step=0.5, key="pen_dm_w12")
            st.markdown("**국내 ETF 설정 (시그널/실매매 공용)**")
            _dm_kr_spy = _sidebar_etf_code_input("공격 ETF 1", _dm_kr_spy, key="pen_dm_kr_spy", disabled=IS_CLOUD)
            _dm_kr_efa = _sidebar_etf_code_input("공격 ETF 2", _dm_kr_efa, key="pen_dm_kr_efa", disabled=IS_CLOUD)
            _dm_kr_agg = _sidebar_etf_code_input("방어 ETF", _dm_kr_agg, key="pen_dm_kr_agg", disabled=IS_CLOUD)
            _dm_kr_bil = _sidebar_etf_code_input("카나리아 ETF", _dm_kr_bil, key="pen_dm_kr_bil", disabled=IS_CLOUD)

    _dm_settings = {
        "offensive": _dm_offensive, "defensive": _dm_defensive, "canary": _dm_canary,
        "lookback": int(_dm_lookback), "trading_days_per_month": int(_dm_td),
        "momentum_weights": {"m1": float(_dm_w1), "m3": float(_dm_w3), "m6": float(_dm_w6), "m12": float(_dm_w12)},
        "kr_etf_map": {k: _deprecated_etf_map.get(v, v) for k, v in {"SPY": str(_dm_kr_spy), "EFA": str(_dm_kr_efa), "AGG": str(_dm_kr_agg), "BIL": str(_dm_kr_bil)}.items()},
    }

    # ── VAA 전략 설정 ──
    from src.strategy.vaa import VAAStrategy as _VAAStrategy
    _vaa_defaults = _VAAStrategy.DEFAULT_SETTINGS
    _vaa_kr_spy = _safe_etf("pen_vaa_kr_spy", pen_cfg.get("pen_vaa_kr_spy", _vaa_defaults['kr_etf_map']['SPY']))
    _vaa_kr_efa = _safe_etf("pen_vaa_kr_efa", pen_cfg.get("pen_vaa_kr_efa", _vaa_defaults['kr_etf_map']['EFA']))
    _vaa_kr_eem = _safe_etf("pen_vaa_kr_eem", pen_cfg.get("pen_vaa_kr_eem", _vaa_defaults['kr_etf_map']['EEM']))
    _vaa_kr_agg = _safe_etf("pen_vaa_kr_agg", pen_cfg.get("pen_vaa_kr_agg", _vaa_defaults['kr_etf_map']['AGG']))
    _vaa_kr_lqd = _safe_etf("pen_vaa_kr_lqd", pen_cfg.get("pen_vaa_kr_lqd", _vaa_defaults['kr_etf_map']['LQD']))
    _vaa_kr_ief = _safe_etf("pen_vaa_kr_ief", pen_cfg.get("pen_vaa_kr_ief", _vaa_defaults['kr_etf_map']['IEF']))
    _vaa_kr_shy = _safe_etf("pen_vaa_kr_shy", pen_cfg.get("pen_vaa_kr_shy", _vaa_defaults['kr_etf_map']['SHY']))

    if ("VAA" in _active_strategies) and (_selected_panel == "VAA 전략 설정"):
        with st.sidebar.expander("VAA 전략 설정", expanded=True):
            st.caption("VAA (Vigilant Asset Allocation) 전용 설정")
            st.markdown("**공격자산 국내 ETF**")
            _vaa_kr_spy = _sidebar_etf_code_input("SPY 대체", _vaa_kr_spy, key="pen_vaa_kr_spy", disabled=IS_CLOUD)
            _vaa_kr_efa = _sidebar_etf_code_input("EFA 대체", _vaa_kr_efa, key="pen_vaa_kr_efa", disabled=IS_CLOUD)
            _vaa_kr_eem = _sidebar_etf_code_input("EEM 대체", _vaa_kr_eem, key="pen_vaa_kr_eem", disabled=IS_CLOUD)
            _vaa_kr_agg = _sidebar_etf_code_input("AGG 대체", _vaa_kr_agg, key="pen_vaa_kr_agg", disabled=IS_CLOUD)
            st.markdown("**방어자산 국내 ETF**")
            _vaa_kr_lqd = _sidebar_etf_code_input("LQD 대체", _vaa_kr_lqd, key="pen_vaa_kr_lqd", disabled=IS_CLOUD)
            _vaa_kr_ief = _sidebar_etf_code_input("IEF 대체", _vaa_kr_ief, key="pen_vaa_kr_ief", disabled=IS_CLOUD)
            _vaa_kr_shy = _sidebar_etf_code_input("SHY 대체", _vaa_kr_shy, key="pen_vaa_kr_shy", disabled=IS_CLOUD)

    _vaa_settings = {
        'offensive': ['SPY', 'EFA', 'EEM', 'AGG'], 'defensive': ['LQD', 'IEF', 'SHY'],
        'lookback': 12, 'top_n': 1, 'trading_days_per_month': 22,
        'momentum_weights': {'m1': 12.0, 'm3': 4.0, 'm6': 2.0, 'm12': 1.0},
        'kr_etf_map': {k: _deprecated_etf_map.get(v, v) for k, v in {
            'SPY': str(_vaa_kr_spy), 'EFA': str(_vaa_kr_efa), 'EEM': str(_vaa_kr_eem),
            'AGG': str(_vaa_kr_agg), 'LQD': str(_vaa_kr_lqd), 'IEF': str(_vaa_kr_ief), 'SHY': str(_vaa_kr_shy),
        }.items()},
    }

    # ── 설정 저장 ──
    if not IS_CLOUD and st.sidebar.button("연금저축 설정 저장", key="pen_save_cfg"):
        pen_data = {
            "kis_pension_account_no": str(kis_acct).strip(),
            "kis_pension_prdt_cd": str(kis_prdt).strip() or "01",
            "kis_pension_start_date": str(_pen_bt_start_raw),
            "kis_pension_initial_cap": int(_pen_bt_cap),
            "pension_portfolio": _pen_port_edited.to_dict("records"),
        }
        if kr_spy: pen_data["kr_etf_laa_spy"] = kr_spy
        if kr_iwd: pen_data["kr_etf_laa_iwd"] = kr_iwd
        if kr_gld: pen_data["kr_etf_laa_gld"] = kr_gld
        if kr_ief: pen_data["kr_etf_laa_ief"] = kr_ief
        if kr_qqq: pen_data["kr_etf_laa_qqq"] = kr_qqq
        if kr_shy: pen_data["kr_etf_laa_shy"] = kr_shy
        if _dm_settings:
            _dmw = _dm_settings.get("momentum_weights", {})
            _dm_kr_map = _dm_settings.get("kr_etf_map", {})
            pen_data.update({
                "pen_dm_lookback": int(_dm_settings.get("lookback", 12)),
                "pen_dm_trading_days": int(_dm_settings.get("trading_days_per_month", 22)),
                "pen_dm_offensive": ",".join(_dm_settings.get("offensive", ["SPY", "EFA"])),
                "pen_dm_defensive": ",".join(_dm_settings.get("defensive", ["AGG"])),
                "pen_dm_canary": ",".join(_dm_settings.get("canary", ["BIL"])),
                "pen_dm_w1": float(_dmw.get("m1", 12.0)), "pen_dm_w3": float(_dmw.get("m3", 4.0)),
                "pen_dm_w6": float(_dmw.get("m6", 2.0)), "pen_dm_w12": float(_dmw.get("m12", 1.0)),
                "pen_dm_kr_spy": str(_dm_kr_map.get("SPY", "360750")),
                "pen_dm_kr_efa": str(_dm_kr_map.get("EFA", "251350")),
                "pen_dm_kr_agg": str(_dm_kr_map.get("AGG", "305080")),
                "pen_dm_kr_bil": str(_dm_kr_map.get("BIL", "329750")),
                "pen_dm_agg_etf": str(_dm_kr_map.get("SPY", "360750")),
                "pen_dm_def_etf": str(_dm_kr_map.get("AGG", "305080")),
            })
        _vaa_kr = _vaa_settings.get("kr_etf_map", {})
        pen_data.update({
            "pen_vaa_kr_spy": str(_vaa_kr.get("SPY", "379800")), "pen_vaa_kr_efa": str(_vaa_kr.get("EFA", "195930")),
            "pen_vaa_kr_eem": str(_vaa_kr.get("EEM", "195980")), "pen_vaa_kr_agg": str(_vaa_kr.get("AGG", "305080")),
            "pen_vaa_kr_lqd": str(_vaa_kr.get("LQD", "329750")), "pen_vaa_kr_ief": str(_vaa_kr.get("IEF", "305080")),
            "pen_vaa_kr_shy": str(_vaa_kr.get("SHY", "329750")),
        })
        save_mode_config("pension", pen_data)
        new_cfg = config.copy()
        new_cfg.update(pen_data)
        save_config(new_cfg)
        st.sidebar.success("연금저축 설정을 저장했습니다.")

    # ── 검증 ──
    if not (kis_ak and kis_sk and kis_acct):
        st.warning("KIS 연금저축 API 키와 계좌번호를 설정해 주세요.")
        return None

    return {
        "pen_cfg": pen_cfg,
        "pen_bt_start_raw": _pen_bt_start_raw,
        "pen_bt_start_ts": _pen_bt_start_ts,
        "pen_bt_cap": _pen_bt_cap,
        "kis_ak": kis_ak, "kis_sk": kis_sk,
        "kis_acct": kis_acct, "kis_prdt": kis_prdt,
        "pen_port_edited": _pen_port_edited,
        "active_strategies": _active_strategies,
        "auto_signal_strategies": _auto_signal_strategies,
        "kr_spy": kr_spy, "kr_iwd": kr_iwd, "kr_gld": kr_gld,
        "kr_ief": kr_ief, "kr_qqq": kr_qqq, "kr_shy": kr_shy,
        "kr_etf_map": _kr_etf_map,
        "dm_settings": _dm_settings,
        "vaa_settings": _vaa_settings,
    }
