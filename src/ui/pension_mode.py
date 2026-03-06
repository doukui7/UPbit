import json
import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import plotly.graph_objects as go
from datetime import datetime, timedelta

import src.engine.data_cache as data_cache
from src.constants import IS_CLOUD
from src.utils.formatting import (
    _etf_name_kr,
    _fmt_etf_code_name,
    _code_only,
    _format_kis_holdings_df,
    _safe_float,
    _safe_int,
)
from src.utils.helpers import _get_runtime_value
from src.utils.kis import _get_kis_token, _compute_kis_balance_summary
from src.ui.components.performance import (
    _render_performance_analysis,
    _apply_return_hover_format,
    _apply_dd_hover_format,
)
from src.ui.components.triggers import render_strategy_trigger_tab


# ─── 예약 주문 데이터 관리 ───
_PEN_ORDERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "pension_orders.json",
)


def _load_pen_orders() -> list[dict]:
    """예약 주문 내역 로드 (영구 보존)."""
    try:
        if os.path.exists(_PEN_ORDERS_FILE):
            with open(_PEN_ORDERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass
    return []


def _save_pen_orders(orders: list[dict]):
    """예약 주문 내역 저장."""
    os.makedirs(os.path.dirname(_PEN_ORDERS_FILE), exist_ok=True)
    with open(_PEN_ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)


def _add_pen_order(etf_code: str, side: str, qty: int, method: str,
                   price: int, scheduled_kst: str, note: str) -> dict:
    """예약 주문 추가."""
    orders = _load_pen_orders()
    now_kst = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    order = {
        "id": f"pen-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000,9999)}",
        "etf_code": str(etf_code),
        "etf_name": _etf_name_kr(str(etf_code)),
        "side": side,
        "qty": qty,
        "method": method,
        "price": price,
        "scheduled_kst": scheduled_kst,
        "status": "대기",
        "created_at": now_kst,
        "executed_at": "",
        "result": "",
        "note": note,
    }
    orders.append(order)
    _save_pen_orders(orders)
    return order


def _update_pen_order_status(order_id: str, status: str, result: str = ""):
    """예약 주문 상태 업데이트 (삭제하지 않고 상태만 변경)."""
    orders = _load_pen_orders()
    for o in orders:
        if o.get("id") == order_id:
            o["status"] = status
            if result:
                o["result"] = result
            if status in ("완료", "실패", "취소"):
                o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            break
    _save_pen_orders(orders)


def _execute_pending_pen_orders(trader) -> list[str]:
    """예정 시간이 지난 대기 주문을 실행하고 상태를 업데이트한다."""
    orders = _load_pen_orders()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    results = []
    changed = False

    for o in orders:
        if o.get("status") != "대기":
            continue
        sched = o.get("scheduled_kst", "")
        if not sched or sched > now_str:
            continue

        code = str(o.get("etf_code", "")).strip()
        side = o.get("side", "")
        qty = int(o.get("qty", 0))
        method = o.get("method", "")
        oid = o.get("id", "")

        if qty <= 0 or not code:
            o["status"] = "실패"
            o["result"] = "수량 또는 ETF코드 누락"
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{oid}: 실패 (수량/코드 누락)")
            continue

        try:
            res = None
            if "동시호가" in method:
                if "매수" in side:
                    res = trader.execute_closing_auction_buy(code, qty)
                else:
                    res = trader.execute_closing_auction_sell(code, qty)
            elif "시간외" in method:
                ord_side = "BUY" if "매수" in side else "SELL"
                res = trader.send_order(ord_side, code, qty, price=0, ord_dvsn="06")
            elif "시장가" in method:
                if "매수" in side:
                    res = trader.smart_buy_krw(code, float(o.get("price", 0)) or 0)
                else:
                    res = trader.smart_sell_all(code)
            else:
                # 지정가 fallback
                price = int(o.get("price", 0))
                ord_side = "BUY" if "매수" in side else "SELL"
                if price > 0:
                    res = trader.send_order(ord_side, code, qty, price=price, ord_dvsn="00")
                else:
                    res = {"success": False, "msg": "지정가 0"}

            success = isinstance(res, dict) and res.get("success", False)
            o["status"] = "완료" if success else "실패"
            o["result"] = str(res)[:200] if res else "응답 없음"
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{oid}: {'완료' if success else '실패'}")
        except Exception as e:
            o["status"] = "실패"
            o["result"] = str(e)[:200]
            o["executed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            changed = True
            results.append(f"{oid}: 실패 ({e})")

    if changed:
        _save_pen_orders(orders)
    return results


def _sidebar_etf_code_input(title: str, code_value: str, key: str, disabled: bool = False) -> str:
    code = _code_only(code_value)
    if not st.session_state.get("_etf_code_input_css_loaded", False):
        st.sidebar.markdown(
            """
            <style>
            section[data-testid="stSidebar"] input[aria-label="????"]{
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
            "????",
            value=code,
            key=key,
            max_chars=6,
            disabled=disabled,
            label_visibility="collapsed",
        )
    return _code_only(typed_code)


def _normalize_numeric_series(series_obj, preferred_cols=("equity", "close", "Close")) -> pd.Series:
    """Series/DataFrame/array를 숫자 Series로 정규화한다."""
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

    if isinstance(s.index, pd.DatetimeIndex):
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _infer_periods_per_year(index_like) -> int:
    """DatetimeIndex 간격으로 연환산 주기를 추정한다."""
    if not isinstance(index_like, pd.DatetimeIndex) or len(index_like) < 2:
        return 252
    try:
        deltas = index_like.to_series().diff().dropna().dt.total_seconds()
        if deltas.empty:
            return 252
        med = float(deltas.median())
        if med <= 0:
            return 252
        annual = int(round((365.25 * 24 * 3600) / med))
        return max(1, min(annual, 365))
    except Exception:
        return 252


def _calc_equity_metrics(equity_series: pd.Series, periods_per_year: int = 252) -> dict:
    """equity Series로 기본 성과 지표를 계산한다."""
    eq = _normalize_numeric_series(equity_series, preferred_cols=("equity", "close", "Close"))
    if len(eq) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "avg_yearly_mdd": 0.0,
            "final_equity": float(eq.iloc[-1]) if len(eq) else 0.0,
        }

    init_val = float(eq.iloc[0])
    final_val = float(eq.iloc[-1])
    total_return = ((final_val / init_val) - 1.0) * 100.0 if init_val > 0 else 0.0

    if isinstance(eq.index, pd.DatetimeIndex):
        days = max((eq.index[-1] - eq.index[0]).days, 1)
        cagr = ((final_val / init_val) ** (365.0 / days) - 1.0) * 100.0 if init_val > 0 and final_val > 0 else 0.0
    else:
        years = max((len(eq) - 1) / max(periods_per_year, 1), 1 / max(periods_per_year, 1))
        cagr = ((final_val / init_val) ** (1.0 / years) - 1.0) * 100.0 if init_val > 0 and final_val > 0 else 0.0

    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    mdd = float(dd.min()) if len(dd) else 0.0

    rets = eq.pct_change().dropna()
    if len(rets) > 1 and float(rets.std()) > 0:
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(max(periods_per_year, 1)))
    else:
        sharpe = 0.0

    if isinstance(dd.index, pd.DatetimeIndex) and len(dd) > 0:
        yearly_mdd = dd.groupby(dd.index.year).min()
        avg_yearly_mdd = float(yearly_mdd.mean()) if len(yearly_mdd) > 0 else mdd
    else:
        avg_yearly_mdd = mdd

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "sharpe": float(sharpe),
        "avg_yearly_mdd": float(avg_yearly_mdd),
        "final_equity": float(final_val),
    }

def render_kis_pension_mode(config, save_config):
    """KIS 연금저축 포트폴리오 모드 - 다중 전략 지원."""
    from src.utils.helpers import load_mode_config, save_mode_config
    _pen_cfg = load_mode_config("pension")
    from src.engine.kis_trader import KISTrader
    from src.strategy.laa import LAAStrategy
    from src.strategy.dual_momentum import DualMomentumStrategy
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── 사용 가능한 전략 목록 ──
    PEN_STRATEGIES = ["LAA", "듀얼모멘텀", "VAA", "CDM"]

    st.title("연금저축 포트폴리오")
    st.sidebar.header("연금저축 설정")
    _pen_bt_start_raw = str(
        _pen_cfg.get("kis_pension_start_date", config.get("start_date", "2020-01-01"))
        or "2020-01-01"
    )
    try:
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    except Exception:
        _pen_bt_start_raw = "2020-01-01"
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    _pen_bt_cap_default = int(float(_pen_cfg.get("kis_pension_initial_cap", 10_000_000) or 10_000_000))

    kis_ak = _get_runtime_value(("KIS_PENSION_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(_pen_cfg.get("kis_pension_account_no", "") or _get_runtime_value(("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(_pen_cfg.get("kis_pension_prdt_cd", "") or _get_runtime_value(("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))
    # 계좌번호 10자리 → 앞8(CANO) + 뒤2(상품코드) 자동 분리
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
    _saved_portfolio = _pen_cfg.get("pension_portfolio", [{"strategy": "LAA", "weight": 100}])
    if not _saved_portfolio:
        _saved_portfolio = [{"strategy": "LAA", "weight": 100}]
    _pen_port_df = pd.DataFrame(_saved_portfolio)

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

        # 연금저축 화면에서는 LAA/듀얼모멘텀/VAA/CDM 행이 항상 보이도록 유지
        if not (out["strategy"] == "LAA").any():
            out = pd.concat([pd.DataFrame([{"strategy": "LAA", "weight": 0}]), out], ignore_index=True)
        if not (out["strategy"] == "듀얼모멘텀").any():
            out = pd.concat([out, pd.DataFrame([{"strategy": "듀얼모멘텀", "weight": 0}])], ignore_index=True)
        if not (out["strategy"] == "VAA").any():
            out = pd.concat([out, pd.DataFrame([{"strategy": "VAA", "weight": 0}])], ignore_index=True)
        if not (out["strategy"] == "CDM").any():
            out = pd.concat([out, pd.DataFrame([{"strategy": "CDM", "weight": 0}])], ignore_index=True)
        
        # 정적배분 삭제 보정
        out = out[out["strategy"] != "정적배분"].reset_index(drop=True)
        return out

    def _with_pen_portfolio_no(df: pd.DataFrame) -> pd.DataFrame:
        _base = _normalize_pen_portfolio_df(df)
        _view = _base.copy()
        _view.insert(0, "no", range(1, len(_view) + 1))
        return _view

    _pen_port_df = _normalize_pen_portfolio_df(_pen_port_df)

    _pen_port_state_key = "pen_portfolio_editor_df"
    if _pen_port_state_key not in st.session_state or not isinstance(st.session_state.get(_pen_port_state_key), pd.DataFrame):
        st.session_state[_pen_port_state_key] = _normalize_pen_portfolio_df(_pen_port_df)
    else:
        _state_df = st.session_state[_pen_port_state_key].copy()
        st.session_state[_pen_port_state_key] = _normalize_pen_portfolio_df(_state_df)

    _pen_port_edited = st.sidebar.data_editor(
        _with_pen_portfolio_no(st.session_state[_pen_port_state_key]),
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

    # data_editor 내부 상태가 이전 값(예: LAA 단일행)으로 고정되는 경우를 보정
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

    # 현재 포트폴리오에 포함된 전략 목록(표시용)
    _active_strategies = list(_pen_port_edited["strategy"].unique()) if not _pen_port_edited.empty else []
    # 자동 시그널 계산 대상: 비중이 0%를 초과하는 전략만
    if not _pen_port_edited.empty:
        _weighted_port = _pen_port_edited[
            pd.to_numeric(_pen_port_edited["weight"], errors="coerce").fillna(0) > 0
        ]
        _auto_signal_strategies = list(_weighted_port["strategy"].unique()) if not _weighted_port.empty else []
    else:
        _auto_signal_strategies = []

    _panel_options = ["접기", "LAA 전략 설정", "듀얼모멘텀 설정", "VAA 전략 설정", "CDM 전략 설정"]
    
    _panel_key = "pen_strategy_settings_panel"
    if st.session_state.get(_panel_key) not in _panel_options:
        st.session_state[_panel_key] = _panel_options[0]
    st.sidebar.caption("전략 상세 설정 (하나만 펼치기)")
    _selected_panel = st.sidebar.radio(
        "전략 상세 설정",
        _panel_options,
        key=_panel_key,
        label_visibility="collapsed",
    )

    # ── LAA 전략 설정 ──
    _kr_iwd_default = _code_only(_pen_cfg.get("kr_etf_laa_iwd", _get_runtime_value("KR_ETF_LAA_IWD", _get_runtime_value("KR_ETF_SPY", "360750"))))
    _kr_spy_default = _code_only(_pen_cfg.get("kr_etf_laa_spy", _get_runtime_value("KR_ETF_LAA_SPY", _get_runtime_value("KR_ETF_SPY", _kr_iwd_default or "360750"))))
    _kr_gld_default = _code_only(_pen_cfg.get("kr_etf_laa_gld", _get_runtime_value("KR_ETF_LAA_GLD", "132030")))
    _kr_ief_default = _code_only(_pen_cfg.get("kr_etf_laa_ief", _get_runtime_value("KR_ETF_LAA_IEF", _get_runtime_value("KR_ETF_AGG", "453540"))))
    _kr_qqq_default = _code_only(_pen_cfg.get("kr_etf_laa_qqq", _get_runtime_value("KR_ETF_LAA_QQQ", "133690")))
    _kr_shy_default = _code_only(_pen_cfg.get("kr_etf_laa_shy", _get_runtime_value("KR_ETF_LAA_SHY", "114470")))
    kr_spy = _code_only(st.session_state.get("pen_laa_spy", _kr_spy_default))
    kr_iwd = _code_only(st.session_state.get("pen_laa_iwd", _kr_iwd_default))
    kr_gld = _code_only(st.session_state.get("pen_laa_gld", _kr_gld_default))
    kr_ief = _code_only(st.session_state.get("pen_laa_ief", _kr_ief_default))
    kr_qqq = _code_only(st.session_state.get("pen_laa_qqq", _kr_qqq_default))
    kr_shy = _code_only(st.session_state.get("pen_laa_shy", _kr_shy_default))

    _show_laa_panel = ("LAA" in _active_strategies) and (_selected_panel == "LAA 전략 설정")
    if _show_laa_panel:
        with st.sidebar.expander("LAA 전략 설정", expanded=True):
            st.caption("LAA 전략 전용 설정")
            kr_spy = _sidebar_etf_code_input("SPY 신호 ETF", kr_spy, key="pen_laa_spy", disabled=IS_CLOUD)
            kr_iwd = _sidebar_etf_code_input("IWD 대체 ETF", kr_iwd, key="pen_laa_iwd", disabled=IS_CLOUD)
            kr_gld = _sidebar_etf_code_input("GLD 대체 ETF", kr_gld, key="pen_laa_gld", disabled=IS_CLOUD)
            kr_ief = _sidebar_etf_code_input("IEF 대체 ETF", kr_ief, key="pen_laa_ief", disabled=IS_CLOUD)
            kr_qqq = _sidebar_etf_code_input("QQQ 대체 ETF", kr_qqq, key="pen_laa_qqq", disabled=IS_CLOUD)
            kr_shy = _sidebar_etf_code_input("SHY 대체 ETF", kr_shy, key="pen_laa_shy", disabled=IS_CLOUD)

    _kr_etf_map = {"SPY": str(kr_spy), "IWD": str(kr_iwd), "GLD": str(kr_gld), "IEF": str(kr_ief), "QQQ": str(kr_qqq), "SHY": str(kr_shy)}

    # ── 듀얼모멘텀 전략 설정 ──
    # 백테스트에서 항상 사용 가능하도록 기본 설정은 전략 활성 여부와 무관하게 준비한다.
    _dm_offensive = ["SPY", "EFA"]
    _dm_defensive = ["AGG"]
    _dm_canary = ["BIL"]
    _dm_lookback = int(st.session_state.get("pen_dm_lookback", _pen_cfg.get("pen_dm_lookback", 12)))
    _dm_td = int(st.session_state.get("pen_dm_trading_days", _pen_cfg.get("pen_dm_trading_days", 22)))
    _dm_w1 = float(st.session_state.get("pen_dm_w1", _pen_cfg.get("pen_dm_w1", 12.0)))
    _dm_w3 = float(st.session_state.get("pen_dm_w3", _pen_cfg.get("pen_dm_w3", 4.0)))
    _dm_w6 = float(st.session_state.get("pen_dm_w6", _pen_cfg.get("pen_dm_w6", 2.0)))
    _dm_w12 = float(st.session_state.get("pen_dm_w12", _pen_cfg.get("pen_dm_w12", 1.0)))

    _dm_kr_spy_default = _code_only(_pen_cfg.get("pen_dm_kr_spy", _pen_cfg.get("pen_dm_agg_etf", _get_runtime_value("KR_ETF_SPY", "360750"))))
    _dm_kr_efa_default = _code_only(_pen_cfg.get("pen_dm_kr_efa", _get_runtime_value("KR_ETF_EFA", "453850")))
    _dm_kr_agg_default = _code_only(_pen_cfg.get("pen_dm_kr_agg", _pen_cfg.get("pen_dm_def_etf", _get_runtime_value("KR_ETF_AGG", "453540"))))
    _dm_kr_bil_default = _code_only(_pen_cfg.get("pen_dm_kr_bil", _get_runtime_value("KR_ETF_BIL", _get_runtime_value("KR_ETF_SHY", "114470"))))
    _dm_kr_spy = _code_only(st.session_state.get("pen_dm_kr_spy", _dm_kr_spy_default))
    _dm_kr_efa = _code_only(st.session_state.get("pen_dm_kr_efa", _dm_kr_efa_default))
    _dm_kr_agg = _code_only(st.session_state.get("pen_dm_kr_agg", _dm_kr_agg_default))
    _dm_kr_bil = _code_only(st.session_state.get("pen_dm_kr_bil", _dm_kr_bil_default))

    _show_dm_panel = ("듀얼모멘텀" in _active_strategies) and (_selected_panel == "듀얼모멘텀 설정")
    if _show_dm_panel:
        with st.sidebar.expander("듀얼모멘텀 설정", expanded=True):
            st.caption("듀얼모멘텀 전략 전용 설정")
            dmc1, dmc2 = st.columns(2)
            _dm_lookback = dmc1.number_input(
                "카나리아 룩백(개월)",
                min_value=1,
                max_value=24,
                value=int(_dm_lookback),
                step=1,
                key="pen_dm_lookback",
                help="기본 12개월 (절대 모멘텀 기준)",
            )
            _dm_td = dmc2.number_input(
                "월 환산 거래일",
                min_value=18,
                max_value=24,
                value=int(_dm_td),
                step=1,
                key="pen_dm_trading_days",
                help="기본 22일",
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
        "offensive": _dm_offensive,
        "defensive": _dm_defensive,
        "canary": _dm_canary,
        "lookback": int(_dm_lookback),
        "trading_days_per_month": int(_dm_td),
        "momentum_weights": {
            "m1": float(_dm_w1),
            "m3": float(_dm_w3),
            "m6": float(_dm_w6),
            "m12": float(_dm_w12),
        },
        "kr_etf_map": {
            "SPY": str(_dm_kr_spy),
            "EFA": str(_dm_kr_efa),
            "AGG": str(_dm_kr_agg),
            "BIL": str(_dm_kr_bil),
        },
    }

    # ── VAA 전략 설정 ──
    from src.strategy.vaa import VAAStrategy as _VAAStrategy
    _vaa_defaults = _VAAStrategy.DEFAULT_SETTINGS
    _vaa_kr_spy = _code_only(st.session_state.get("pen_vaa_kr_spy", _pen_cfg.get("pen_vaa_kr_spy", _vaa_defaults['kr_etf_map']['SPY'])))
    _vaa_kr_efa = _code_only(st.session_state.get("pen_vaa_kr_efa", _pen_cfg.get("pen_vaa_kr_efa", _vaa_defaults['kr_etf_map']['EFA'])))
    _vaa_kr_eem = _code_only(st.session_state.get("pen_vaa_kr_eem", _pen_cfg.get("pen_vaa_kr_eem", _vaa_defaults['kr_etf_map']['EEM'])))
    _vaa_kr_agg = _code_only(st.session_state.get("pen_vaa_kr_agg", _pen_cfg.get("pen_vaa_kr_agg", _vaa_defaults['kr_etf_map']['AGG'])))
    _vaa_kr_lqd = _code_only(st.session_state.get("pen_vaa_kr_lqd", _pen_cfg.get("pen_vaa_kr_lqd", _vaa_defaults['kr_etf_map']['LQD'])))
    _vaa_kr_ief = _code_only(st.session_state.get("pen_vaa_kr_ief", _pen_cfg.get("pen_vaa_kr_ief", _vaa_defaults['kr_etf_map']['IEF'])))
    _vaa_kr_shy = _code_only(st.session_state.get("pen_vaa_kr_shy", _pen_cfg.get("pen_vaa_kr_shy", _vaa_defaults['kr_etf_map']['SHY'])))

    _show_vaa_panel = ("VAA" in _active_strategies) and (_selected_panel == "VAA 전략 설정")
    if _show_vaa_panel:
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
        'offensive': ['SPY', 'EFA', 'EEM', 'AGG'],
        'defensive': ['LQD', 'IEF', 'SHY'],
        'lookback': 12,
        'top_n': 1,
        'trading_days_per_month': 22,
        'momentum_weights': {'m1': 12.0, 'm3': 4.0, 'm6': 2.0, 'm12': 1.0},
        'kr_etf_map': {
            'SPY': str(_vaa_kr_spy), 'EFA': str(_vaa_kr_efa),
            'EEM': str(_vaa_kr_eem), 'AGG': str(_vaa_kr_agg),
            'LQD': str(_vaa_kr_lqd), 'IEF': str(_vaa_kr_ief),
            'SHY': str(_vaa_kr_shy),
        },
    }

    # ── CDM 전략 설정 ──
    from src.strategy.cdm import CDMStrategy as _CDMStrategy
    _cdm_defaults = _CDMStrategy.DEFAULT_SETTINGS
    _cdm_kr_spy = _code_only(st.session_state.get("pen_cdm_kr_spy", _pen_cfg.get("pen_cdm_kr_spy", _cdm_defaults['kr_etf_map']['SPY'])))
    _cdm_kr_veu = _code_only(st.session_state.get("pen_cdm_kr_veu", _pen_cfg.get("pen_cdm_kr_veu", _cdm_defaults['kr_etf_map']['VEU'])))
    _cdm_kr_vnq = _code_only(st.session_state.get("pen_cdm_kr_vnq", _pen_cfg.get("pen_cdm_kr_vnq", _cdm_defaults['kr_etf_map']['VNQ'])))
    _cdm_kr_rem = _code_only(st.session_state.get("pen_cdm_kr_rem", _pen_cfg.get("pen_cdm_kr_rem", _cdm_defaults['kr_etf_map']['REM'])))
    _cdm_kr_lqd = _code_only(st.session_state.get("pen_cdm_kr_lqd", _pen_cfg.get("pen_cdm_kr_lqd", _cdm_defaults['kr_etf_map']['LQD'])))
    _cdm_kr_hyg = _code_only(st.session_state.get("pen_cdm_kr_hyg", _pen_cfg.get("pen_cdm_kr_hyg", _cdm_defaults['kr_etf_map']['HYG'])))
    _cdm_kr_tlt = _code_only(st.session_state.get("pen_cdm_kr_tlt", _pen_cfg.get("pen_cdm_kr_tlt", _cdm_defaults['kr_etf_map']['TLT'])))
    _cdm_kr_gld = _code_only(st.session_state.get("pen_cdm_kr_gld", _pen_cfg.get("pen_cdm_kr_gld", _cdm_defaults['kr_etf_map']['GLD'])))
    _cdm_kr_bil = _code_only(st.session_state.get("pen_cdm_kr_bil", _pen_cfg.get("pen_cdm_kr_bil", _cdm_defaults['kr_etf_map']['BIL'])))

    _show_cdm_panel = ("CDM" in _active_strategies) and (_selected_panel == "CDM 전략 설정")
    if _show_cdm_panel:
        with st.sidebar.expander("CDM 전략 설정", expanded=True):
            st.caption("CDM (Composite Dual Momentum) 4모듈 전략")
            st.markdown("**모듈 1: 미국 vs 해외**")
            _cdm_kr_spy = _sidebar_etf_code_input("SPY 대체", _cdm_kr_spy, key="pen_cdm_kr_spy", disabled=IS_CLOUD)
            _cdm_kr_veu = _sidebar_etf_code_input("VEU 대체", _cdm_kr_veu, key="pen_cdm_kr_veu", disabled=IS_CLOUD)
            st.markdown("**모듈 2: 부동산**")
            _cdm_kr_vnq = _sidebar_etf_code_input("VNQ 대체", _cdm_kr_vnq, key="pen_cdm_kr_vnq", disabled=IS_CLOUD)
            _cdm_kr_rem = _sidebar_etf_code_input("REM 대체", _cdm_kr_rem, key="pen_cdm_kr_rem", disabled=IS_CLOUD)
            st.markdown("**모듈 3: 채권**")
            _cdm_kr_lqd = _sidebar_etf_code_input("LQD 대체", _cdm_kr_lqd, key="pen_cdm_kr_lqd", disabled=IS_CLOUD)
            _cdm_kr_hyg = _sidebar_etf_code_input("HYG 대체", _cdm_kr_hyg, key="pen_cdm_kr_hyg", disabled=IS_CLOUD)
            st.markdown("**모듈 4: 장기채 vs 금**")
            _cdm_kr_tlt = _sidebar_etf_code_input("TLT 대체", _cdm_kr_tlt, key="pen_cdm_kr_tlt", disabled=IS_CLOUD)
            _cdm_kr_gld = _sidebar_etf_code_input("GLD 대체", _cdm_kr_gld, key="pen_cdm_kr_gld", disabled=IS_CLOUD)
            st.markdown("**방어자산**")
            _cdm_kr_bil = _sidebar_etf_code_input("BIL 대체", _cdm_kr_bil, key="pen_cdm_kr_bil", disabled=IS_CLOUD)

    _cdm_settings = {
        'offensive': ['SPY', 'VEU', 'VNQ', 'REM', 'LQD', 'HYG', 'TLT', 'GLD'],
        'defensive': ['BIL'],
        'lookback': 12,
        'trading_days_per_month': 22,
        'kr_etf_map': {
            'SPY': str(_cdm_kr_spy), 'VEU': str(_cdm_kr_veu),
            'VNQ': str(_cdm_kr_vnq), 'REM': str(_cdm_kr_rem),
            'LQD': str(_cdm_kr_lqd), 'HYG': str(_cdm_kr_hyg),
            'TLT': str(_cdm_kr_tlt), 'GLD': str(_cdm_kr_gld),
            'BIL': str(_cdm_kr_bil),
        },
    }

    # (정적배분 설정 부분 삭제됨)

    # ── 설정 저장 ──
    if not IS_CLOUD and st.sidebar.button("연금저축 설정 저장", key="pen_save_cfg"):
        pen_data = {
            "kis_pension_account_no": str(kis_acct).strip(),
            "kis_pension_prdt_cd": str(kis_prdt).strip() or "01",
            "kis_pension_start_date": str(_pen_bt_start_raw),
            "kis_pension_initial_cap": int(_pen_bt_cap),
            "pension_portfolio": _pen_port_edited.to_dict("records"),
        }
        # LAA 설정
        if kr_spy: pen_data["kr_etf_laa_spy"] = kr_spy
        if kr_iwd: pen_data["kr_etf_laa_iwd"] = kr_iwd
        if kr_gld: pen_data["kr_etf_laa_gld"] = kr_gld
        if kr_ief: pen_data["kr_etf_laa_ief"] = kr_ief
        if kr_qqq: pen_data["kr_etf_laa_qqq"] = kr_qqq
        if kr_shy: pen_data["kr_etf_laa_shy"] = kr_shy
        # 듀얼모멘텀 설정
        if _dm_settings:
            _dmw = _dm_settings.get("momentum_weights", {})
            _dm_kr_map = _dm_settings.get("kr_etf_map", {})
            pen_data.update({
                "pen_dm_lookback": int(_dm_settings.get("lookback", 12)),
                "pen_dm_trading_days": int(_dm_settings.get("trading_days_per_month", 22)),
                "pen_dm_offensive": ",".join(_dm_settings.get("offensive", ["SPY", "EFA"])),
                "pen_dm_defensive": ",".join(_dm_settings.get("defensive", ["AGG"])),
                "pen_dm_canary": ",".join(_dm_settings.get("canary", ["BIL"])),
                "pen_dm_w1": float(_dmw.get("m1", 12.0)),
                "pen_dm_w3": float(_dmw.get("m3", 4.0)),
                "pen_dm_w6": float(_dmw.get("m6", 2.0)),
                "pen_dm_w12": float(_dmw.get("m12", 1.0)),
                "pen_dm_kr_spy": str(_dm_kr_map.get("SPY", "360750")),
                "pen_dm_kr_efa": str(_dm_kr_map.get("EFA", "453850")),
                "pen_dm_kr_agg": str(_dm_kr_map.get("AGG", "453540")),
                "pen_dm_kr_bil": str(_dm_kr_map.get("BIL", "114470")),
                "pen_dm_agg_etf": str(_dm_kr_map.get("SPY", "360750")),
                "pen_dm_def_etf": str(_dm_kr_map.get("AGG", "453540")),
            })
        # VAA 설정
        _vaa_kr = _vaa_settings.get("kr_etf_map", {})
        pen_data.update({
            "pen_vaa_kr_spy": str(_vaa_kr.get("SPY", "379800")),
            "pen_vaa_kr_efa": str(_vaa_kr.get("EFA", "195930")),
            "pen_vaa_kr_eem": str(_vaa_kr.get("EEM", "295820")),
            "pen_vaa_kr_agg": str(_vaa_kr.get("AGG", "305080")),
            "pen_vaa_kr_lqd": str(_vaa_kr.get("LQD", "329750")),
            "pen_vaa_kr_ief": str(_vaa_kr.get("IEF", "305080")),
            "pen_vaa_kr_shy": str(_vaa_kr.get("SHY", "329750")),
        })
        # CDM 설정
        _cdm_kr = _cdm_settings.get("kr_etf_map", {})
        pen_data.update({
            "pen_cdm_kr_spy": str(_cdm_kr.get("SPY", "379800")),
            "pen_cdm_kr_veu": str(_cdm_kr.get("VEU", "195930")),
            "pen_cdm_kr_vnq": str(_cdm_kr.get("VNQ", "352560")),
            "pen_cdm_kr_rem": str(_cdm_kr.get("REM", "352560")),
            "pen_cdm_kr_lqd": str(_cdm_kr.get("LQD", "305080")),
            "pen_cdm_kr_hyg": str(_cdm_kr.get("HYG", "305080")),
            "pen_cdm_kr_tlt": str(_cdm_kr.get("TLT", "304660")),
            "pen_cdm_kr_gld": str(_cdm_kr.get("GLD", "132030")),
            "pen_cdm_kr_bil": str(_cdm_kr.get("BIL", "329750")),
        })
        # 모드별 설정 파일 저장 (config/pension.json)
        save_mode_config("pension", pen_data)
        # 전역 config에도 반영 (하위호환)
        new_cfg = config.copy()
        new_cfg.update(pen_data)
        save_config(new_cfg)
        st.sidebar.success("연금저축 설정을 저장했습니다.")

    if not (kis_ak and kis_sk and kis_acct):
        st.warning("KIS 연금저축 API 키와 계좌번호를 설정해 주세요.")
        return

    trader = KISTrader(is_mock=False)
    trader.app_key = kis_ak
    trader.app_secret = kis_sk
    trader.account_no = kis_acct
    trader.acnt_prdt_cd = kis_prdt

    # 토큰 캐싱 — ISA/연금저축 공용 토큰 재사용 (KIS 1분 1회 발급 제한 대응)
    _pen_token_key = f"pen_token_{kis_acct}"
    _kis_shared_token_key = f"kis_token_shared_{str(kis_ak or '')[-8:]}"
    _cached_pen = st.session_state.get(_pen_token_key)
    _cached_shared = st.session_state.get(_kis_shared_token_key)
    _tok = None
    if _cached_pen and (_cached_pen.get("expiry", 0) - time.time()) > 300:
        _tok = _cached_pen
    elif _cached_shared and (_cached_shared.get("expiry", 0) - time.time()) > 300:
        _tok = _cached_shared

    if _tok:
        trader.access_token = _tok.get("token")
        trader.token_expiry = float(_tok.get("expiry", 0))
        st.session_state[_pen_token_key] = {"token": trader.access_token, "expiry": trader.token_expiry}
    else:
        if not trader.auth():
            st.error("KIS 인증에 실패했습니다. API 키/계좌를 확인해 주세요.")
            return
        _new_tok = {"token": trader.access_token, "expiry": trader.token_expiry}
        st.session_state[_pen_token_key] = _new_tok
        st.session_state[_kis_shared_token_key] = _new_tok

    # ── 예약 주문 자동 실행 (예정 시간 지난 대기 건) ──
    _pen_exec_key = "_pen_orders_last_check"
    _pen_last_check = st.session_state.get(_pen_exec_key, "")
    _pen_now_min = datetime.now().strftime("%Y-%m-%d %H:%M")
    if _pen_last_check != _pen_now_min:
        st.session_state[_pen_exec_key] = _pen_now_min
        _exec_results = _execute_pending_pen_orders(trader)
        if _exec_results:
            for _er in _exec_results:
                st.toast(_er)

    pen_bal_key = f"pension_balance_cache_{kis_acct}_{kis_prdt}"
    pen_bal_ts_key = f"pension_balance_cache_ts_{kis_acct}_{kis_prdt}"
    pen_bal_err_key = f"pension_balance_cache_err_{kis_acct}_{kis_prdt}"
    try:
        _pen_bal_auto_refresh_sec = float(_pen_cfg.get("pen_balance_auto_refresh_sec", 12) or 12.0)
    except Exception:
        _pen_bal_auto_refresh_sec = 12.0
    _pen_bal_auto_refresh_sec = max(2.0, _pen_bal_auto_refresh_sec)

    def _clear_pen_signal_caches():
        st.session_state.pop("pen_signal_result", None)
        st.session_state.pop("pen_signal_params", None)
        st.session_state.pop("pen_dm_signal_result", None)
        st.session_state.pop("pen_dm_signal_params", None)
        st.session_state.pop("pen_vaa_signal_result", None)
        st.session_state.pop("pen_vaa_signal_params", None)
        st.session_state.pop("pen_cdm_signal_result", None)
        st.session_state.pop("pen_cdm_signal_params", None)

    def _fetch_pen_balance_with_retry(retries: int = 3, delay_sec: float = 0.45):
        _tries = max(1, int(retries))
        _last = None
        for _i in range(_tries):
            try:
                _last = trader.get_balance()
            except Exception as _e:
                _last = None
                logging.warning(f"연금저축 잔고 조회 예외 ({_i+1}/{_tries}): {_e}")
            if isinstance(_last, dict) and not bool(_last.get("error")):
                return _last
            if _i < _tries - 1:
                time.sleep(float(delay_sec))
        return _last

    def _refresh_pension_balance(force: bool = False, show_spinner: bool = False) -> bool:
        _now = float(time.time())
        _last_ts = float(st.session_state.get(pen_bal_ts_key, 0.0) or 0.0)
        _has_cache = pen_bal_key in st.session_state
        _need = bool(force) or (not _has_cache) or ((_now - _last_ts) >= _pen_bal_auto_refresh_sec)
        if not _need:
            return False

        def _do_fetch():
            return _fetch_pen_balance_with_retry(retries=3, delay_sec=0.45)

        _new_bal = None
        if show_spinner:
            with st.spinner("연금저축 잔고를 조회하는 중..."):
                _new_bal = _do_fetch()
        else:
            _new_bal = _do_fetch()

        _prev_bal = st.session_state.get(pen_bal_key)
        _ok = isinstance(_new_bal, dict) and not bool(_new_bal.get("error"))

        if _ok:
            st.session_state[pen_bal_key] = _new_bal
            st.session_state[pen_bal_ts_key] = float(time.time())
            st.session_state[pen_bal_err_key] = ""
            _clear_pen_signal_caches()
            return True

        # 신규 조회 실패 시, 기존 정상 캐시가 있으면 유지하여 화면 공백을 방지
        _msg = "잔고 최신 조회 실패"
        if isinstance(_new_bal, dict) and _new_bal.get("error"):
            _msg = f"{_msg} [{_new_bal.get('msg_cd', '')}] {str(_new_bal.get('msg1', '')).strip()}".strip()
        elif _new_bal is None:
            _msg = f"{_msg} (응답 None)"
        else:
            _msg = f"{_msg} ({type(_new_bal).__name__})"
        st.session_state[pen_bal_err_key] = _msg

        if _prev_bal is None:
            st.session_state[pen_bal_key] = _new_bal
            st.session_state[pen_bal_ts_key] = float(time.time())
        return False

    # 연금저축 데이터 로딩 정책: 로컬 파일 우선 + 부족 시 API 보강
    _pen_local_first_raw = str(_pen_cfg.get("pen_local_first", _pen_cfg.get("pen_local_data_only", "1"))).strip().lower()
    _pen_local_first = _pen_local_first_raw not in ("0", "false", "no", "off")
    _pen_api_fallback_raw = str(_pen_cfg.get("pen_api_fallback", "1")).strip().lower()
    _pen_api_fallback = _pen_api_fallback_raw not in ("0", "false", "no", "off")
    _pen_live_bt_raw = str(_pen_cfg.get("pen_live_auto_backtest", "0")).strip().lower()
    _pen_live_auto_backtest = _pen_live_bt_raw in ("1", "true", "yes", "on")

    # 연금저축 모드 내 KIS 호출 캐시 (재렌더링 시 중복 호출 최소화)
    _pen_cache_key = f"pension_api_cache_{kis_acct}_{kis_prdt}"
    _pen_cache = st.session_state.get(_pen_cache_key)
    if not isinstance(_pen_cache, dict):
        _pen_cache = {"daily": {}, "price": {}, "orderbook": {}}
        st.session_state[_pen_cache_key] = _pen_cache

    def _cache_put(_bucket: str, _key: str, _value):
        _store = _pen_cache.setdefault(_bucket, {})
        _store[_key] = {"ts": float(time.time()), "value": _value}
        if len(_store) > 120:
            _oldest = sorted(_store.items(), key=lambda kv: kv[1].get("ts", 0.0))[:20]
            for _k, _ in _oldest:
                _store.pop(_k, None)

    def _cache_get(_bucket: str, _key: str, _ttl_sec: float):
        _store = _pen_cache.get(_bucket, {})
        _hit = _store.get(_key)
        if not _hit:
            return None
        _age = float(time.time()) - float(_hit.get("ts", 0.0))
        if _age > float(_ttl_sec):
            return None
        return _hit.get("value")

    def _normalize_ohlcv_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return df
        out = df.copy().sort_index()
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        has_upper = any(k in out.columns for k in rename_map)
        if has_upper:
            out = out.rename(columns=rename_map)
        return out

    def _get_pen_daily_chart(_code: str, count: int = 420, end_date: str | None = None, use_disk_cache: bool = False):
        _c = _code_only(_code)
        if not _c:
            return None
        _cnt = int(max(1, count))
        _end = str(end_date or "")
        _cache_k = f"{_c}|{_cnt}|{_end}|{int(bool(use_disk_cache))}"
        _ttl = 1800.0 if _cnt >= 3000 else 300.0
        _cached = _cache_get("daily", _cache_k, _ttl)
        if isinstance(_cached, pd.DataFrame) and not _cached.empty:
            _cached_ok = True
            try:
                _cached_len = int(len(_cached))
            except Exception:
                _cached_len = 0

            # 수동주문 차트(count=120 등)는 짧은 세션 캐시(예: 3봉)면 재조회
            if not _end and _cnt >= 60 and _cached_len < min(_cnt, 60):
                _cached_ok = False

            # 최신 차트는 최근 데이터가 아니면 재조회
            if _cached_ok and not _end and _pen_api_fallback and not _is_recent_pen_daily_df(_cached, max_stale_days=7):
                _cached_ok = False

            if _cached_ok:
                return _cached.copy()

        _df = None
        if use_disk_cache or _pen_local_first:
            _df = data_cache.get_kis_domestic_local_first(
                trader if _pen_api_fallback else None,
                _c,
                count=_cnt,
                end_date=_end or None,
                allow_api_fallback=bool(_pen_api_fallback),
            )
        else:
            try:
                _df = trader.get_daily_chart(_c, count=_cnt, end_date=_end or None)
            except Exception:
                _df = None

        _df = _normalize_ohlcv_df(_df)
        if isinstance(_df, pd.DataFrame) and not _df.empty:
            try:
                _df.index = pd.to_datetime(_df.index)
            except Exception:
                pass
            if _end:
                try:
                    _end_ts = pd.to_datetime(_end)
                    _df = _df[_df.index <= _end_ts]
                except Exception:
                    pass
            if len(_df) > _cnt:
                _df = _df.tail(_cnt)

        if isinstance(_df, pd.DataFrame) and not _df.empty:
            _cache_put("daily", _cache_k, _df)
            return _df.copy()
        return _df

    def _is_recent_pen_daily_df(_df: pd.DataFrame | None, max_stale_days: int = 7) -> bool:
        if _df is None or _df.empty:
            return False
        try:
            _last_ts = pd.to_datetime(_df.index[-1])
            _last_date = _last_ts.date()
            _today = pd.Timestamp.now(tz="Asia/Seoul").date()
            _age = int((_today - _last_date).days)
            return _age <= int(max_stale_days)
        except Exception:
            return False

    def _resolve_pen_quote_code(_code: str, ttl_sec: float = 30.0) -> tuple[str, str]:
        _c = _code_only(_code)
        if not _c:
            return "", ""

        _cached = _cache_get("quote_code", _c, ttl_sec)
        if isinstance(_cached, dict):
            _cached_code = _code_only(_cached.get("code", _c))
            _cached_msg = str(_cached.get("msg", "") or "")
            if _cached_code:
                return _cached_code, _cached_msg

        _resolved = _c
        _msg = ""
        _legacy_map = {"453540": "305080"}

        if _pen_api_fallback:
            _base_live = 0.0
            try:
                _base_live = float(trader.get_current_price(_c) or 0.0)
            except Exception:
                _base_live = 0.0
            if _base_live > 0:
                _cache_put("price", _c, float(_base_live))

            if _base_live <= 0:
                _alt = _legacy_map.get(_c, "")
                if _alt:
                    _alt_live = 0.0
                    try:
                        _alt_live = float(trader.get_current_price(_alt) or 0.0)
                    except Exception:
                        _alt_live = 0.0
                    if _alt_live > 0:
                        _cache_put("price", _alt, float(_alt_live))
                        _resolved = _alt
                        _msg = (
                            f"{_fmt_etf_code_name(_c)} 실시간 시세 조회가 되지 않아 "
                            f"{_fmt_etf_code_name(_alt)} 시세/차트로 대체 표시합니다."
                        )

        _cache_put("quote_code", _c, {"code": _resolved, "msg": _msg})
        return _resolved, _msg

    def _get_pen_current_price(_code: str, ttl_sec: float = 8.0) -> float:
        _c = _code_only(_code)
        if not _c:
            return 0.0
        _cached = _cache_get("price", _c, ttl_sec)
        if _cached is not None:
            return float(_cached or 0.0)
        _p = 0.0

        # 실시간 API를 우선 사용하고, 실패 시 최근(기본 7일 이내) 일봉 종가만 보조값으로 사용
        if _pen_api_fallback:
            try:
                _p = float(trader.get_current_price(_c) or 0.0)
            except Exception:
                _p = 0.0

        if _p <= 0:
            _ref_df = _get_pen_daily_chart(_c, count=3, use_disk_cache=True if _pen_local_first else False)
            if _is_recent_pen_daily_df(_ref_df, max_stale_days=7):
                if _ref_df is not None and not _ref_df.empty:
                    if "close" in _ref_df.columns:
                        _p = float(_ref_df["close"].iloc[-1] or 0.0)
                    elif "Close" in _ref_df.columns:
                        _p = float(_ref_df["Close"].iloc[-1] or 0.0)

        _cache_put("price", _c, float(_p if _p > 0 else 0.0))
        return float(_p if _p > 0 else 0.0)

    def _get_pen_orderbook(_code: str, ttl_sec: float = 8.0):
        _c = _code_only(_code)
        if not _c:
            return None
        _cached = _cache_get("orderbook", _c, ttl_sec)
        if _cached is not None:
            return _cached
        _ob = None
        try:
            _ob = trader.get_orderbook(_c)
        except Exception:
            _ob = None
        _cache_put("orderbook", _c, _ob)
        return _ob

    tab_p1, tab_p2, tab_p3, tab_p4, tab_p5, tab_p6, tab_p7, tab_p8 = st.tabs([
        "🚀 실시간 모니터링",
        "🧪 백테스트",
        "🛒 수동 주문",
        "📜 거래내역",
        "📖 전략 가이드",
        "📋 주문방식",
        "💳 수수료/세금",
        "⏰ 트리거",
    ])

    # ══════════════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링
    # ══════════════════════════════════════════════════════════════
    with tab_p1:
        _strat_summary = " + ".join([f"{r['strategy']} {r['weight']}%" for _, r in _pen_port_edited.iterrows()]) if not _pen_port_edited.empty else "LAA 100%"
        st.header("포트폴리오 모니터링")
        st.caption(f"구성: {_strat_summary}")

        # 잔고 표시 — 화면 진입/재렌더링 시 짧은 TTL로 자동 최신화
        _refresh_pension_balance(force=False, show_spinner=(pen_bal_key not in st.session_state))

        rcol1, rcol2 = st.columns([1, 5])
        with rcol1:
            if st.button("잔고 새로고침", key="pen_refresh_balance"):
                _refresh_pension_balance(force=True, show_spinner=True)
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

        # LAA 시그널 자동 계산 (비중 > 0 전략만 자동 계산)
        pen_sig_params = {
            "acct": str(kis_acct),
            "prdt": str(kis_prdt),
            "kr_spy": str(kr_spy),
            "kr_iwd": str(kr_iwd),
            "kr_gld": str(kr_gld),
            "kr_ief": str(kr_ief),
            "kr_qqq": str(kr_qqq),
            "kr_shy": str(kr_shy),
            "bt_start": str(_pen_bt_start_raw),
            "bt_cap": float(_pen_bt_cap),
        }

        def _compute_pen_signal_result():
            tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
            price_data = {}
            source_map = {
                "SPY": _code_only(kr_spy or kr_iwd or "360750"),
                "IWD": _code_only(kr_iwd),
                "GLD": _code_only(kr_gld),
                "IEF": _code_only(kr_ief),
                "QQQ": _code_only(kr_qqq),
                "SHY": _code_only(kr_shy),
            }
            _today = pd.Timestamp.now().date()
            for ticker in tickers:
                _code = _code_only(source_map.get(ticker, ""))
                if not _code:
                    return {"error": f"{ticker} 국내 ETF 코드가 비어 있습니다."}
                df_t = _get_pen_daily_chart(_code, count=420)
                if df_t is None or df_t.empty:
                    return {"error": f"{ticker} ({_code}) 국내 데이터 조회에 실패했습니다."}
                df_t = df_t.copy().sort_index()
                if "close" not in df_t.columns and "Close" in df_t.columns:
                    df_t["close"] = df_t["Close"]
                if "close" not in df_t.columns:
                    return {"error": f"{ticker} ({_code}) 종가 컬럼이 없습니다."}
                if len(df_t) >= 2:
                    _last_dt = pd.to_datetime(df_t.index[-1]).date()
                    if _last_dt >= _today:
                        df_t = df_t.iloc[:-1]
                if df_t is None or df_t.empty:
                    return {"error": f"{ticker} ({_code}) 전일 종가 데이터가 없습니다."}
                price_data[ticker] = df_t

            strategy = LAAStrategy(settings={"kr_etf_map": _kr_etf_map})
            signal = strategy.analyze(price_data)
            if not signal:
                return {"error": "LAA 분석에 실패했습니다."}

            # 리스크 판단 보조 차트: TIGER 미국S&P500(기본 360750) + 200일선 + 이격도
            _risk_chart_code = _code_only((_kr_etf_map or {}).get("IWD", "")) or "360750"
            _risk_df = _get_pen_daily_chart(str(_risk_chart_code), count=800)
            if _risk_df is not None and not _risk_df.empty:
                _risk_df = _risk_df.copy().sort_index()
                if "close" not in _risk_df.columns and "Close" in _risk_df.columns:
                    _risk_df["close"] = _risk_df["Close"]
                if "close" in _risk_df.columns:
                    _risk_df["ma200"] = _risk_df["close"].rolling(200).mean()
                    _risk_df["divergence"] = np.where(
                        _risk_df["ma200"] > 0,
                        (_risk_df["close"] / _risk_df["ma200"] - 1.0) * 100.0,
                        np.nan,
                    )
                else:
                    _risk_df = None

            bal_local = st.session_state.get(pen_bal_key) or trader.get_balance() or {"cash": 0.0, "holdings": []}
            cash_local = float(bal_local.get("cash", 0.0))
            holdings_local = bal_local.get("holdings", []) or []

            current_vals = {}
            current_qtys = {}
            current_prices = {}
            for h in holdings_local:
                code = str(h.get("code", "")).strip()
                if not code:
                    continue
                current_vals[code] = current_vals.get(code, 0.0) + float(h.get("eval_amt", 0.0) or 0.0)
                _qty = float(h.get("qty", 0.0) or 0.0)
                current_qtys[code] = current_qtys.get(code, 0) + int(np.floor(max(_qty, 0.0)))
                _cur_p = float(h.get("cur_price", 0.0) or 0.0)
                if _cur_p > 0:
                    current_prices[code] = _cur_p

            price_cache = {}

            def _resolve_etf_price(_code: str, _cur_v: float, _cur_q: int) -> float:
                _key = str(_code).strip()
                if not _key:
                    return 0.0
                if _key in price_cache:
                    return float(price_cache[_key])

                _p = 0.0
                try:
                    _p = float(_get_pen_current_price(_key) or 0.0)
                except Exception:
                    _p = 0.0

                if _p <= 0:
                    _p = float(current_prices.get(_key, 0.0) or 0.0)

                if _p <= 0 and _cur_q > 0:
                    _p = float(_cur_v) / float(_cur_q)

                if _p <= 0:
                    try:
                        _ch = _get_pen_daily_chart(_key, count=5)
                        if _ch is not None and not _ch.empty:
                            _p = float(_ch["close"].iloc[-1])
                    except Exception:
                        _p = 0.0

                price_cache[_key] = float(_p if _p > 0 else 0.0)
                return float(price_cache[_key])

            total_eval = float(bal_local.get("total_eval", 0.0)) or (cash_local + sum(current_vals.values()))
            total_eval = max(total_eval, 1.0)

            rows = []
            max_gap = 0.0
            for code, target_w in signal["target_weights_kr"].items():
                cur_v = float(current_vals.get(str(code), 0.0))
                cur_w = cur_v / total_eval
                gap = float(target_w) - float(cur_w)
                max_gap = max(max_gap, abs(gap))
                cur_qty = int(current_qtys.get(str(code), 0))
                px = float(_resolve_etf_price(str(code), cur_v, cur_qty))
                target_v = total_eval * float(target_w)
                target_qty = int(np.floor(target_v / px)) if px > 0 else 0
                trade_qty = int(target_qty - cur_qty)
                trade_side = "매수" if trade_qty > 0 else ("매도" if trade_qty < 0 else "유지")
                trade_notional = abs(trade_qty) * px if px > 0 else 0.0
                rows.append({
                    "ETF": _fmt_etf_code_name(code),
                    "ETF 코드": str(code),
                    "목표 비중(%)": round(target_w * 100.0, 2),
                    "현재 비중(%)": round(cur_w * 100.0, 2),
                    "비중 차이(%p)": round(gap * 100.0, 2),
                    "현재가(KRW)": f"{px:,.0f}" if px > 0 else "-",
                    "현재수량(주)": int(cur_qty),
                    "목표수량(주)": int(target_qty),
                    "주문": trade_side,
                    "매매수량(주)": int(abs(trade_qty)),
                    "예상주문금액(KRW)": f"{trade_notional:,.0f}" if px > 0 and abs(trade_qty) > 0 else "0",
                    "현재 평가(KRW)": f"{cur_v:,.0f}",
                    "목표 평가(KRW)": f"{target_v:,.0f}",
                })

            action = "HOLD" if max_gap <= 0.03 else "REBALANCE"

            bt_result = None
            bt_benchmark_series = None
            bt_benchmark_label = "SPY Buy & Hold"
            if _pen_live_auto_backtest:
                # 실시간 탭 초기 진입 지연을 줄이기 위해 자동 백테스트는 기본 비활성화
                bt_price_data = {}
                for ticker in tickers:
                    _df_bt = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
                    if _df_bt is None or _df_bt.empty:
                        bt_price_data = {}
                        break
                    _df_bt = _df_bt.copy().sort_index()
                    if len(_df_bt) >= 2:
                        _last_dt = pd.to_datetime(_df_bt.index[-1]).date()
                        if _last_dt >= _today:
                            _df_bt = _df_bt.iloc[:-1]
                    _df_bt = _df_bt[_df_bt.index >= _pen_bt_start_ts]
                    if _df_bt.empty:
                        bt_price_data = {}
                        break
                    bt_price_data[ticker] = _df_bt

                if bt_price_data:
                    bt_result = strategy.run_backtest(
                        bt_price_data,
                        initial_balance=float(_pen_bt_cap),
                        fee=0.0002,
                    )
                    _bm_ticker = "SPY"
                    _bm_series = _normalize_numeric_series(
                        bt_price_data.get(_bm_ticker),
                        preferred_cols=("close", "Close"),
                    )
                    if not _bm_series.empty:
                        bt_benchmark_series = _bm_series

            return {
                "signal": signal,
                "action": action,
                "source_map": source_map,
                "alloc_df": pd.DataFrame(rows),
                "price_data": price_data,
                "risk_chart_df": _risk_df,
                "risk_chart_code": _risk_chart_code,
                "balance": bal_local,
                "bt_start_date": str(_pen_bt_start_raw),
                "bt_initial_cap": float(_pen_bt_cap),
                "bt_result": bt_result,
                "bt_benchmark_series": bt_benchmark_series,
                "bt_benchmark_label": bt_benchmark_label,
            }

        res = None
        if "LAA" in _auto_signal_strategies:
            if st.session_state.get("pen_signal_result") is None or st.session_state.get("pen_signal_params") != pen_sig_params:
                with st.spinner("LAA 시그널을 자동 계산하는 중입니다..."):
                    st.session_state["pen_signal_result"] = _compute_pen_signal_result()
                    st.session_state["pen_signal_params"] = pen_sig_params
                    _pen_res = st.session_state["pen_signal_result"]
                    if isinstance(_pen_res, dict) and _pen_res.get("balance"):
                        st.session_state[pen_bal_key] = _pen_res["balance"]
            res = st.session_state.get("pen_signal_result")
        else:
            st.session_state.pop("pen_signal_result", None)
            st.session_state.pop("pen_signal_params", None)

        # 듀얼모멘텀 시그널 자동 계산 (비중 > 0 전략만 자동 계산)
        _dm_res = None
        if "듀얼모멘텀" in _auto_signal_strategies and _dm_settings:
            _dm_weights = _dm_settings.get("momentum_weights", {})
            _dm_params = {
                "acct": str(kis_acct),
                "prdt": str(kis_prdt),
                "off": tuple(_dm_settings.get("offensive", [])),
                "def": tuple(_dm_settings.get("defensive", [])),
                "canary": tuple(_dm_settings.get("canary", [])),
                "lookback": int(_dm_settings.get("lookback", 12)),
                "td": int(_dm_settings.get("trading_days_per_month", 22)),
                "w1": float(_dm_weights.get("m1", 12.0)),
                "w3": float(_dm_weights.get("m3", 4.0)),
                "w6": float(_dm_weights.get("m6", 2.0)),
                "w12": float(_dm_weights.get("m12", 1.0)),
                "kr_spy": str((_dm_settings.get("kr_etf_map", {}) or {}).get("SPY", "")),
                "kr_efa": str((_dm_settings.get("kr_etf_map", {}) or {}).get("EFA", "")),
                "kr_agg": str((_dm_settings.get("kr_etf_map", {}) or {}).get("AGG", "")),
                "kr_bil": str((_dm_settings.get("kr_etf_map", {}) or {}).get("BIL", "")),
                "bt_start": str(_pen_bt_start_raw),
                "bt_cap": float(_pen_bt_cap),
            }

            def _compute_dm_signal_result():
                _dm_tickers = []
                for _tk in (_dm_settings.get("offensive", []) + _dm_settings.get("defensive", []) + _dm_settings.get("canary", [])):
                    _u = str(_tk).strip().upper()
                    if _u and _u not in _dm_tickers:
                        _dm_tickers.append(_u)

                if not _dm_tickers:
                    return {"error": "듀얼모멘텀 티커 설정이 비어 있습니다."}

                _dm_price_data = {}
                _dm_source_map = {}
                _dm_kr_map = _dm_settings.get("kr_etf_map", {}) or {}
                _today = pd.Timestamp.now().date()
                for _ticker in _dm_tickers:
                    _kr_code = str(_dm_kr_map.get(_ticker, "")).strip()
                    if not _kr_code:
                        return {"error": f"{_ticker} 국내 ETF 매핑이 없습니다."}
                    _df_t = _get_pen_daily_chart(_kr_code, count=420)
                    if _df_t is None or _df_t.empty:
                        return {"error": f"{_ticker} ({_kr_code}) 국내 데이터 조회 실패"}
                    _df_t = _df_t.copy().sort_index()
                    if "close" not in _df_t.columns and "Close" in _df_t.columns:
                        _df_t["close"] = _df_t["Close"]
                    if "close" not in _df_t.columns:
                        return {"error": f"{_ticker} ({_kr_code}) 종가 컬럼이 없습니다."}

                    if len(_df_t) >= 2:
                        _last_dt = pd.to_datetime(_df_t.index[-1]).date()
                        if _last_dt >= _today:
                            _df_t = _df_t.iloc[:-1]
                    if _df_t is None or _df_t.empty:
                        return {"error": f"{_ticker} ({_kr_code}) 전일 종가 데이터가 없습니다."}

                    _dm_price_data[_ticker] = _df_t
                    _dm_source_map[_ticker] = _kr_code

                _dm_strategy = DualMomentumStrategy(settings=_dm_settings)
                _dm_sig = _dm_strategy.analyze(_dm_price_data)
                if not _dm_sig:
                    return {"error": "듀얼모멘텀 분석에 실패했습니다."}

                _dm_td_local = int(_dm_settings.get("trading_days_per_month", 22))
                _dm_lb_local = int(_dm_settings.get("lookback", 12))
                _dm_w_local = _dm_settings.get("momentum_weights", {}) or {}
                _w1 = float(_dm_w_local.get("m1", 12.0))
                _w3 = float(_dm_w_local.get("m3", 4.0))
                _w6 = float(_dm_w_local.get("m6", 2.0))
                _w12 = float(_dm_w_local.get("m12", 1.0))

                _mom_rows = []
                _ref_dates = []
                _score_rows = []
                _lookback_col = f"룩백{_dm_lb_local}개월(%)"
                for _tk in _dm_tickers:
                    _dfm = _dm_price_data[_tk]
                    _prices = _dfm["close"].astype(float).values
                    _m1 = DualMomentumStrategy.calc_monthly_return(_prices, 1, _dm_td_local)
                    _m3 = DualMomentumStrategy.calc_monthly_return(_prices, 3, _dm_td_local)
                    _m6 = DualMomentumStrategy.calc_monthly_return(_prices, 6, _dm_td_local)
                    _m12 = DualMomentumStrategy.calc_monthly_return(_prices, 12, _dm_td_local)
                    _lb_ret = DualMomentumStrategy.calc_monthly_return(_prices, _dm_lb_local, _dm_td_local)
                    _score = (_m1 * _w1 + _m3 * _w3 + _m6 * _w6 + _m12 * _w12) / 4.0
                    _role = "공격" if _tk in _dm_settings.get("offensive", []) else ("방어" if _tk in _dm_settings.get("defensive", []) else "카나리아")
                    _last_close = float(_dfm["close"].iloc[-1])
                    _last_date = pd.to_datetime(_dfm.index[-1]).date()
                    _ref_dates.append(_last_date)
                    _score_rows.append({"티커": _tk, "모멘텀 점수": float(_score)})
                    _mom_rows.append({
                        "역할": _role,
                        "티커": _tk,
                        "국내 ETF": _fmt_etf_code_name(_dm_source_map.get(_tk, "")),
                        "기준 종가일": str(_last_date),
                        "기준 종가": f"{_last_close:,.0f}",
                        "1개월(%)": round(_m1 * 100.0, 2),
                        "3개월(%)": round(_m3 * 100.0, 2),
                        "6개월(%)": round(_m6 * 100.0, 2),
                        "12개월(%)": round(_m12 * 100.0, 2),
                        _lookback_col: round(_lb_ret * 100.0, 2),
                        "가중 모멘텀 점수": round(_score, 6),
                    })

                _ref_date = min(_ref_dates) if _ref_dates else None
                _lag_days = int((_today - _ref_date).days) if _ref_date else None

                _bal_local = st.session_state.get(pen_bal_key) or trader.get_balance() or {"cash": 0.0, "holdings": []}
                _holdings_local = _bal_local.get("holdings", []) or []
                _target_code = str(_dm_sig.get("target_kr_code", ""))
                _current_qty_map = {}
                for _h in _holdings_local:
                    _code = str(_h.get("code", "")).strip()
                    if not _code:
                        continue
                    _q = float(_h.get("qty", 0.0) or 0.0)
                    _current_qty_map[_code] = _current_qty_map.get(_code, 0) + int(np.floor(max(_q, 0.0)))

                _all_dm_codes = []
                for _k in ("SPY", "EFA", "AGG"):
                    _v = str((_dm_settings.get("kr_etf_map", {}) or {}).get(_k, "")).strip()
                    if _v and _v not in _all_dm_codes:
                        _all_dm_codes.append(_v)

                _target_holding = next((h for h in _holdings_local if str(h.get("code", "")) == _target_code), None)
                _other_holdings = [h for h in _holdings_local if str(h.get("code", "")) in _all_dm_codes and str(h.get("code", "")) != _target_code]

                if _target_holding and not _other_holdings:
                    _action = "HOLD"
                elif _other_holdings:
                    _action = "REBALANCE"
                else:
                    _action = "BUY"

                _total_eval_local = float(_bal_local.get("total_eval", 0.0)) or (
                    float(_bal_local.get("cash", 0.0))
                    + sum(float(_h.get("eval_amt", 0.0) or 0.0) for _h in _holdings_local)
                )
                _total_eval_local = max(_total_eval_local, 0.0)
                _dm_weight_pct = float(
                    pd.to_numeric(
                        _pen_port_edited.loc[_pen_port_edited["strategy"] == "듀얼모멘텀", "weight"],
                        errors="coerce",
                    ).fillna(0).sum()
                )
                _dm_weight_pct = max(0.0, min(100.0, _dm_weight_pct))
                _sleeve_eval = _total_eval_local * (_dm_weight_pct / 100.0)
                _target_ticker = str(_dm_sig.get("target_ticker", "")).strip().upper()

                _expected_rows = []
                _alloc_tickers = []
                for _k in (_dm_settings.get("offensive", []) + _dm_settings.get("defensive", [])):
                    _ku = str(_k).strip().upper()
                    if _ku and _ku not in _alloc_tickers:
                        _alloc_tickers.append(_ku)
                for _tk in _alloc_tickers:
                    _code = str(_dm_kr_map.get(_tk, "")).strip()
                    _px = 0.0
                    if _tk in _dm_price_data and len(_dm_price_data[_tk]) > 0:
                        _px = float(_dm_price_data[_tk]["close"].iloc[-1])
                    _target_w_sleeve = 100.0 if _tk == _target_ticker else 0.0
                    _target_w_total = _dm_weight_pct if _tk == _target_ticker else 0.0
                    _target_amt = _sleeve_eval if _tk == _target_ticker else 0.0
                    _target_qty = int(np.floor(_target_amt / _px)) if _px > 0 else 0
                    _cur_qty = int(_current_qty_map.get(_code, 0))
                    _delta_qty = int(_target_qty - _cur_qty)
                    _side = "매수" if _delta_qty > 0 else ("매도" if _delta_qty < 0 else "유지")
                    _expected_rows.append({
                        "티커": _tk,
                        "국내 ETF": _fmt_etf_code_name(_code),
                        "ETF 코드": str(_code),
                        "기준 종가일": str(_ref_date) if _ref_date else "-",
                        "기준 종가": f"{_px:,.0f}" if _px > 0 else "-",
                        "듀얼모멘텀 비중(%)": round(_target_w_sleeve, 2),
                        "전체 포트폴리오 환산 비중(%)": round(_target_w_total, 2),
                        "목표 평가금액(KRW)": f"{_target_amt:,.0f}",
                        "목표수량(주,버림)": int(_target_qty),
                        "현재수량(주)": int(_cur_qty),
                        "예상 주문": _side,
                        "예상 주문수량(주)": int(abs(_delta_qty)),
                    })

                _sleeve_alloc_amt = 0.0
                for _row in _expected_rows:
                    try:
                        _pxf = float(str(_row.get("기준 종가", "0")).replace(",", ""))
                    except Exception:
                        _pxf = 0.0
                    _sleeve_alloc_amt += float(_row.get("목표수량(주,버림)", 0)) * _pxf
                _sleeve_cash_est = max(_sleeve_eval - _sleeve_alloc_amt, 0.0)

                _dm_bt_result = None
                _dm_bm_series = None
                _dm_bm_label = "SPY Buy & Hold"
                if _pen_live_auto_backtest:
                    # 실시간 탭 초기 진입 지연을 줄이기 위해 자동 백테스트는 기본 비활성화
                    _dm_bt_price_data = {}
                    for _tk in _dm_tickers:
                        _code = str(_dm_kr_map.get(_tk, "")).strip()
                        if not _code:
                            continue
                        _df_bt = _get_pen_daily_chart(_code, count=3000, use_disk_cache=True)
                        if _df_bt is None or _df_bt.empty:
                            _dm_bt_price_data = {}
                            break
                        _df_bt = _df_bt.copy().sort_index()
                        if "close" not in _df_bt.columns and "Close" in _df_bt.columns:
                            _df_bt["close"] = _df_bt["Close"]
                        if "close" not in _df_bt.columns:
                            _dm_bt_price_data = {}
                            break
                        if len(_df_bt) >= 2:
                            _last_dt = pd.to_datetime(_df_bt.index[-1]).date()
                            if _last_dt >= _today:
                                _df_bt = _df_bt.iloc[:-1]
                        _df_bt = _df_bt[_df_bt.index >= _pen_bt_start_ts]
                        if _df_bt.empty:
                            _dm_bt_price_data = {}
                            break
                        _dm_bt_price_data[_tk] = _df_bt

                    if _dm_bt_price_data:
                        _dm_bt_result = _dm_strategy.run_backtest(
                            _dm_bt_price_data,
                            initial_balance=float(_pen_bt_cap),
                            fee=0.0002,
                        )
                        _dm_bm_ticker = ""
                        for _t in (_dm_settings.get("offensive", []) or []):
                            if _t in _dm_bt_price_data:
                                _dm_bm_ticker = str(_t)
                                break
                        if not _dm_bm_ticker:
                            for _t in _dm_tickers:
                                if _t in _dm_bt_price_data:
                                    _dm_bm_ticker = str(_t)
                                    break
                        _dm_bm_series_norm = _normalize_numeric_series(
                            _dm_bt_price_data.get(_dm_bm_ticker),
                            preferred_cols=("close", "Close"),
                        )
                        if not _dm_bm_series_norm.empty:
                            _dm_bm_series = _dm_bm_series_norm
                            _dm_bm_code = _code_only(str(_dm_kr_map.get(_dm_bm_ticker, "")))
                            if _dm_bm_code:
                                _dm_bm_label = f"{_fmt_etf_code_name(_dm_bm_code)} Buy & Hold"

                return {
                    "signal": _dm_sig,
                    "action": _action,
                    "source_map": _dm_source_map,
                    "score_df": pd.DataFrame(_score_rows),
                    "momentum_detail_df": pd.DataFrame(_mom_rows),
                    "expected_rebalance_df": pd.DataFrame(_expected_rows),
                    "expected_meta": {
                        "ref_date": str(_ref_date) if _ref_date else "",
                        "today": str(_today),
                        "lag_days": _lag_days,
                        "dm_weight_pct": _dm_weight_pct,
                        "sleeve_eval": _sleeve_eval,
                        "sleeve_cash_est": _sleeve_cash_est,
                    },
                    "balance": _bal_local,
                    "bt_start_date": str(_pen_bt_start_raw),
                    "bt_initial_cap": float(_pen_bt_cap),
                    "bt_result": _dm_bt_result,
                    "bt_benchmark_series": _dm_bm_series,
                    "bt_benchmark_label": _dm_bm_label,
                }

            if st.session_state.get("pen_dm_signal_result") is None or st.session_state.get("pen_dm_signal_params") != _dm_params:
                with st.spinner("듀얼모멘텀 시그널을 자동 계산하는 중입니다..."):
                    st.session_state["pen_dm_signal_result"] = _compute_dm_signal_result()
                    st.session_state["pen_dm_signal_params"] = _dm_params
                    _dm_res_cache = st.session_state["pen_dm_signal_result"]
                    if isinstance(_dm_res_cache, dict) and _dm_res_cache.get("balance"):
                        st.session_state[pen_bal_key] = _dm_res_cache["balance"]

            _dm_res = st.session_state.get("pen_dm_signal_result")
        else:
            st.session_state.pop("pen_dm_signal_result", None)
            st.session_state.pop("pen_dm_signal_params", None)

        # ──────────────────────────────────────────────────────────
        # VAA / CDM 시그널 자동 계산 (비중 > 0 전략만 자동 계산)
        # ──────────────────────────────────────────────────────────
        _vaa_res = None
        _cdm_res = None

        if "VAA" in _auto_signal_strategies:
            _vaa_sig_params = {"acct": str(kis_acct), "vaa_settings": str(_vaa_settings)}
            if (st.session_state.get("pen_vaa_signal_params") != _vaa_sig_params
                    or "pen_vaa_signal_result" not in st.session_state):
                with st.spinner("VAA 시그널 계산 중..."):
                    try:
                        import src.engine.data_cache as _vaa_dc
                        _vaa_strat = _VAAStrategy(settings=_vaa_settings)
                        _vaa_all_tickers = list(set(_vaa_settings['offensive'] + _vaa_settings['defensive']))
                        _vaa_price = {}
                        for _t in _vaa_all_tickers:
                            _df = _vaa_dc.fetch_and_cache_yf(_t, start="2020-01-01")
                            if _df is not None and not _df.empty:
                                _vaa_price[_t] = _df
                        _vaa_sig = _vaa_strat.analyze(_vaa_price)
                        if _vaa_sig:
                            _kr_map = _vaa_settings['kr_etf_map']
                            _tw_kr = _vaa_sig.get("target_weights_kr", {})
                            _bal_v = st.session_state.get(pen_bal_key) or {}
                            _hold_v = _bal_v.get("holdings", []) or []
                            _cash_v = float(_bal_v.get("cash", 0.0) or 0.0)
                            _tot_v = float(_bal_v.get("total_eval", 0.0) or 0.0)
                            if _tot_v <= 0:
                                _tot_v = _cash_v + sum(float(h.get("eval_amt", 0) or 0) for h in _hold_v)
                            _vaa_port_w = 0.0
                            for _, _r in _pen_port_edited.iterrows():
                                if _r["strategy"] == "VAA":
                                    _vaa_port_w += float(_r.get("weight", 0)) / 100.0
                            _vaa_sleeve = _tot_v * _vaa_port_w
                            _hold_qty_v = {}
                            for _h in _hold_v:
                                _c = str(_h.get("code", "")).strip()
                                _hold_qty_v[_c] = _hold_qty_v.get(_c, 0) + int(_h.get("qty", 0) or 0)
                            _vaa_rebal_rows = []
                            _max_gap = 0.0
                            for code, w in _tw_kr.items():
                                _tgt_eval = _vaa_sleeve * w
                                _px = float(_get_pen_current_price(code) or 0)
                                _tgt_qty = int(np.floor(_tgt_eval / _px)) if _px > 0 else 0
                                _cur_qty = _hold_qty_v.get(code, 0)
                                _cur_eval = _cur_qty * _px if _px > 0 else 0
                                _cur_w = (_cur_eval / max(_tot_v, 1)) * 100
                                _tgt_w = (_tgt_eval / max(_tot_v, 1)) * 100
                                _max_gap = max(_max_gap, abs(_tgt_w - _cur_w))
                                _vaa_rebal_rows.append({
                                    "ETF 코드": code, "ETF": _fmt_etf_code_name(code),
                                    "현재수량(주)": _cur_qty, "목표수량(주)": _tgt_qty,
                                    "목표 비중(%)": round(w * 100, 2),
                                    "현재 비중(%)": round(_cur_w, 2),
                                })
                            _vaa_action = "REBALANCE" if _max_gap > 3.0 else "HOLD"
                            st.session_state["pen_vaa_signal_result"] = {
                                "signal": _vaa_sig, "action": _vaa_action,
                                "alloc_df": pd.DataFrame(_vaa_rebal_rows) if _vaa_rebal_rows else pd.DataFrame(),
                            }
                        else:
                            st.session_state["pen_vaa_signal_result"] = {"error": "VAA 시그널 계산 실패"}
                        st.session_state["pen_vaa_signal_params"] = _vaa_sig_params
                    except Exception as _e:
                        st.session_state["pen_vaa_signal_result"] = {"error": str(_e)}
            _vaa_res = st.session_state.get("pen_vaa_signal_result")
        else:
            st.session_state.pop("pen_vaa_signal_result", None)
            st.session_state.pop("pen_vaa_signal_params", None)

        if "CDM" in _auto_signal_strategies:
            _cdm_sig_params = {"acct": str(kis_acct), "cdm_settings": str(_cdm_settings)}
            if (st.session_state.get("pen_cdm_signal_params") != _cdm_sig_params
                    or "pen_cdm_signal_result" not in st.session_state):
                with st.spinner("CDM 시그널 계산 중..."):
                    try:
                        import src.engine.data_cache as _cdm_dc
                        _cdm_strat = _CDMStrategy(settings=_cdm_settings)
                        _cdm_all_tickers = list(set(_cdm_settings['offensive'] + _cdm_settings['defensive']))
                        _cdm_price = {}
                        for _t in _cdm_all_tickers:
                            _df = _cdm_dc.fetch_and_cache_yf(_t, start="2020-01-01")
                            if _df is not None and not _df.empty:
                                _cdm_price[_t] = _df
                        _cdm_sig = _cdm_strat.analyze(_cdm_price)
                        if _cdm_sig:
                            _kr_map_c = _cdm_settings['kr_etf_map']
                            _tw_kr_c = _cdm_sig.get("target_weights_kr", {})
                            _bal_c = st.session_state.get(pen_bal_key) or {}
                            _hold_c = _bal_c.get("holdings", []) or []
                            _cash_c = float(_bal_c.get("cash", 0.0) or 0.0)
                            _tot_c = float(_bal_c.get("total_eval", 0.0) or 0.0)
                            if _tot_c <= 0:
                                _tot_c = _cash_c + sum(float(h.get("eval_amt", 0) or 0) for h in _hold_c)
                            _cdm_port_w = 0.0
                            for _, _r in _pen_port_edited.iterrows():
                                if _r["strategy"] == "CDM":
                                    _cdm_port_w += float(_r.get("weight", 0)) / 100.0
                            _cdm_sleeve = _tot_c * _cdm_port_w
                            _hold_qty_c = {}
                            for _h in _hold_c:
                                _c = str(_h.get("code", "")).strip()
                                _hold_qty_c[_c] = _hold_qty_c.get(_c, 0) + int(_h.get("qty", 0) or 0)
                            _cdm_rebal_rows = []
                            _max_gap_c = 0.0
                            for code, w in _tw_kr_c.items():
                                _tgt_eval = _cdm_sleeve * w
                                _px = float(_get_pen_current_price(code) or 0)
                                _tgt_qty = int(np.floor(_tgt_eval / _px)) if _px > 0 else 0
                                _cur_qty = _hold_qty_c.get(code, 0)
                                _cur_eval = _cur_qty * _px if _px > 0 else 0
                                _cur_w = (_cur_eval / max(_tot_c, 1)) * 100
                                _tgt_w = (_tgt_eval / max(_tot_c, 1)) * 100
                                _max_gap_c = max(_max_gap_c, abs(_tgt_w - _cur_w))
                                _cdm_rebal_rows.append({
                                    "ETF 코드": code, "ETF": _fmt_etf_code_name(code),
                                    "현재수량(주)": _cur_qty, "목표수량(주)": _tgt_qty,
                                    "목표 비중(%)": round(w * 100, 2),
                                    "현재 비중(%)": round(_cur_w, 2),
                                })
                            _cdm_action = "REBALANCE" if _max_gap_c > 3.0 else "HOLD"
                            st.session_state["pen_cdm_signal_result"] = {
                                "signal": _cdm_sig, "action": _cdm_action,
                                "alloc_df": pd.DataFrame(_cdm_rebal_rows) if _cdm_rebal_rows else pd.DataFrame(),
                            }
                        else:
                            st.session_state["pen_cdm_signal_result"] = {"error": "CDM 시그널 계산 실패"}
                        st.session_state["pen_cdm_signal_params"] = _cdm_sig_params
                    except Exception as _e:
                        st.session_state["pen_cdm_signal_result"] = {"error": str(_e)}
            _cdm_res = st.session_state.get("pen_cdm_signal_result")
        else:
            st.session_state.pop("pen_cdm_signal_result", None)
            st.session_state.pop("pen_cdm_signal_params", None)

        # ──────────────────────────────────────────────────────────
        # 섹션 2: 전체 포트폴리오 합산
        # ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("전체 포트폴리오 합산")

        # 동적 전략 컬럼 구성
        _strat_col_map = {}  # {"LAA": "LAA 목표(주)", "DM": "DM 목표(주)", ...}
        if "LAA" in _active_strategies:
            _strat_col_map["LAA"] = "LAA 목표(주)"
        if "듀얼모멘텀" in _active_strategies:
            _strat_col_map["DM"] = "DM 목표(주)"
        if "VAA" in _active_strategies:
            _strat_col_map["VAA"] = "VAA 목표(주)"
        if "CDM" in _active_strategies:
            _strat_col_map["CDM"] = "CDM 목표(주)"

        _combined_rows = {}

        def _ensure_combined_row(code, etf_name, cur_qty):
            if code not in _combined_rows:
                _combined_rows[code] = {"ETF": etf_name, "현재수량(주)": int(cur_qty)}
                for _col in _strat_col_map.values():
                    _combined_rows[code][_col] = 0
            _combined_rows[code]["현재수량(주)"] = max(
                _combined_rows[code]["현재수량(주)"], int(cur_qty))

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

        _dm_action_val = None
        if _dm_res and not _dm_res.get("error"):
            _dm_action_val = _dm_res.get("action")
            _dm_df = _dm_res.get("expected_rebalance_df")
            if isinstance(_dm_df, pd.DataFrame) and not _dm_df.empty:
                for _, row in _dm_df.iterrows():
                    code = str(row.get("ETF 코드", "")).strip()
                    if not code:
                        continue
                    _ensure_combined_row(code, row.get("국내 ETF", _fmt_etf_code_name(code)), row.get("현재수량(주)", 0))
                    if "DM 목표(주)" in _combined_rows[code]:
                        _combined_rows[code]["DM 목표(주)"] = int(row.get("목표수량(주,버림)", 0))

        _vaa_action_val = None
        if _vaa_res and not _vaa_res.get("error"):
            _vaa_action_val = _vaa_res.get("action")
            _vaa_df = _vaa_res.get("alloc_df")
            if isinstance(_vaa_df, pd.DataFrame) and not _vaa_df.empty:
                for _, row in _vaa_df.iterrows():
                    code = str(row.get("ETF 코드", "")).strip()
                    if not code:
                        continue
                    _ensure_combined_row(code, row.get("ETF", _fmt_etf_code_name(code)), row.get("현재수량(주)", 0))
                    if "VAA 목표(주)" in _combined_rows[code]:
                        _combined_rows[code]["VAA 목표(주)"] = int(row.get("목표수량(주)", 0))

        _cdm_action_val = None
        if _cdm_res and not _cdm_res.get("error"):
            _cdm_action_val = _cdm_res.get("action")
            _cdm_df = _cdm_res.get("alloc_df")
            if isinstance(_cdm_df, pd.DataFrame) and not _cdm_df.empty:
                for _, row in _cdm_df.iterrows():
                    code = str(row.get("ETF 코드", "")).strip()
                    if not code:
                        continue
                    _ensure_combined_row(code, row.get("ETF", _fmt_etf_code_name(code)), row.get("현재수량(주)", 0))
                    if "CDM 목표(주)" in _combined_rows[code]:
                        _combined_rows[code]["CDM 목표(주)"] = int(row.get("목표수량(주)", 0))

        if _combined_rows:
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
            if _cdm_action_val == "REBALANCE":
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
            for _h in _holdings_combo:
                _code = str(_h.get("code", "")).strip()
                if not _code:
                    continue
                _hold_eval_map[_code] = _hold_eval_map.get(_code, 0.0) + float(_h.get("eval_amt", 0.0) or 0.0)
                _cur_px = float(_h.get("cur_price", 0.0) or 0.0)
                if _cur_px > 0:
                    _hold_px_map[_code] = _cur_px

            _combo_price_cache = {}

            def _resolve_combo_price(_code: str, _cur_qty: int, _cur_eval: float) -> float:
                _k = str(_code).strip()
                if not _k:
                    return 0.0
                if _k in _combo_price_cache:
                    return float(_combo_price_cache[_k])

                _p = float(_hold_px_map.get(_k, 0.0) or 0.0)
                if _p <= 0:
                    try:
                        _p = float(_get_pen_current_price(_k) or 0.0)
                    except Exception:
                        _p = 0.0

                if _p <= 0 and _cur_qty > 0:
                    _p = float(_cur_eval) / float(_cur_qty)

                if _p <= 0:
                    try:
                        _ch = _get_pen_daily_chart(_k, count=5)
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
            from datetime import date as _d_date, timedelta as _d_td
            import calendar as _cal
            _today = _d_date.today()
            def _next_rebal_date(_ref):
                """매월 마지막 영업일(월~금) 계산. _ref 이후 가장 가까운 날짜 반환."""
                y, m = _ref.year, _ref.month
                last_day = _cal.monthrange(y, m)[1]
                dt = _d_date(y, m, last_day)
                while dt.weekday() >= 5:  # 토(5)/일(6) → 금요일로
                    dt -= _d_td(days=1)
                if dt >= _ref:
                    return dt
                # 이미 지남 → 다음 달
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
            st.dataframe(pd.DataFrame(_combo_list), use_container_width=True, hide_index=True)
            st.session_state["pen_combined_rebal_data"] = _combo_list
        else:
            st.info("시그널 계산 결과가 없습니다. 잔고를 새로고침해주세요.")
            st.session_state.pop("pen_combined_rebal_data", None)

        # ──────────────────────────────────────────────────────────
        # 섹션 3: 전략별 상세 하위탭
        # ──────────────────────────────────────────────────────────
        st.divider()
        import streamlit.components.v1 as components

        _detail_tab_names = []
        _detail_tab_keys = []
        if "LAA" in _active_strategies:
            _detail_tab_names.append("LAA 전략")
            _detail_tab_keys.append("LAA")
        if "듀얼모멘텀" in _active_strategies:
            _detail_tab_names.append("듀얼모멘텀 전략")
            _detail_tab_keys.append("DM")
        if "VAA" in _active_strategies:
            _detail_tab_names.append("VAA 전략")
            _detail_tab_keys.append("VAA")
        if "CDM" in _active_strategies:
            _detail_tab_names.append("CDM 전략")
            _detail_tab_keys.append("CDM")

        if _detail_tab_names:
            _detail_tabs = st.tabs(_detail_tab_names)
            _detail_tab_map = dict(zip(_detail_tab_keys, _detail_tabs))
        else:
            _detail_tab_map = {}

        # ── LAA 전략 상세 ──
        if "LAA" in _detail_tab_map:
          with _detail_tab_map["LAA"]:
            _laa_hdr_col1, _laa_hdr_col2 = st.columns([4, 1])
            with _laa_hdr_col1:
                st.subheader("LAA 전략 포트폴리오")
            with _laa_hdr_col2:
                if st.button("📖 전략 가이드", key="pen_laa_guide_btn"):
                    components.html("""
                    <script>
                    const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    for (let t of tabs) {
                        if (t.textContent.includes('전략 가이드')) { t.click(); break; }
                    }
                    </script>
                    """, height=0)

            if res and res.get("error"):
                st.error(res["error"])
            elif res:
                sig = res["signal"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("리스크 상태", "공격 (Risk-On)" if sig.get("risk_on") else "방어 (Risk-Off)")
                c2.metric("리스크 자산", sig["selected_risk_asset"])
                c3.metric("국내 ETF", _fmt_etf_code_name(sig["selected_risk_kr_code"]))
                c4.metric("권장 동작", res["action"])
                st.info(sig.get("reason", ""))

                if "source_map" in res:
                    st.caption("시그널 데이터(국내): " + ", ".join([f"{k}→{v}" for k, v in res["source_map"].items()]))

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

                        st.subheader("TIGER 미국S&P500 + 200일선")
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("현재가", f"{_last_close:,.0f} KRW")
                        cc2.metric("200일선", f"{_last_ma200:,.0f} KRW")
                        cc3.metric("이격도(200일)", f"{_last_div:+.2f}%")
                        st.caption(f"현재 모드: {_mode_label} | 기준 ETF: {_fmt_etf_code_name(_risk_chart_code)}")

                        fig_risk = go.Figure()
                        fig_risk.add_trace(go.Scatter(
                            x=_plot_df.index, y=_plot_df["close"],
                            name=f"{_fmt_etf_code_name(_risk_chart_code)} 종가",
                            line=dict(color="royalblue", width=2),
                        ))
                        fig_risk.add_trace(go.Scatter(
                            x=_plot_df.index, y=_plot_df["ma200"],
                            name="200일 이동평균",
                            line=dict(color="orange", width=2, dash="dash"),
                        ))
                        fig_risk.add_annotation(
                            xref="paper", yref="paper", x=0.01, y=0.98,
                            text=f"현재 모드: {_mode_label}",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor=_mode_color,
                            font=dict(color=_mode_color, size=13),
                        )
                        fig_risk.update_layout(
                            height=430,
                            xaxis_title="날짜",
                            yaxis_title="가격 (KRW)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.06),
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)

                        st.subheader("이격도 차트")
                        fig_div = go.Figure()
                        fig_div.add_trace(go.Scatter(
                            x=_plot_df.index, y=_plot_df["divergence"],
                            name="이격도(%)",
                            line=dict(color="#7c3aed", width=2),
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
                            text=f"현재 모드: {_mode_label} ({_last_div:+.2f}%)",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor=_mode_color,
                            font=dict(color=_mode_color, size=13),
                        )
                        fig_div.update_layout(
                            height=320,
                            xaxis_title="날짜",
                            yaxis_title="이격도 (%)",
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
                        _laa_start = str(res.get("bt_start_date", _pen_bt_start_raw))
                        _laa_bt_cap = float(res.get("bt_initial_cap", _pen_bt_cap) or _pen_bt_cap)
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
                        st.write(
                            f"수익률: **{_laa_total_ret:+.2f}%** | MDD: **{_laa_mdd:.2f}%** | CAGR: **{_laa_cagr:.2f}%**"
                        )
                        st.write(f"최종자산: {_laa_final_eq:,.0f}원 (초기자본 {_laa_bt_cap:,.0f}원 기준)")

                        st.divider()
                        ac1, ac2, ac3, ac4 = st.columns(4)
                        ac1.metric(
                            "백테스트 자산",
                            f"{_laa_final_eq:,.0f}원",
                            delta=f"{_laa_total_ret:+.2f}%",
                        )
                        ac1.caption(f"초기자본 {_laa_bt_cap:,.0f}원 기준")
                        ac2.metric(
                            "실제 총자산",
                            f"{_actual_total_v:,.0f}원" if _bal_valid else "조회 불가",
                        )
                        ac3.metric("최대 비중 차이", f"{_max_gap_pct:.2f}%p")
                        ac4.metric("포지션 동기화", "일치" if _sync_ok else "불일치")
                        ac4.caption("주문 상태가 모두 '유지'이면 일치로 판단합니다.")

                        _laa_bm_series = res.get("bt_benchmark_series")
                        _laa_bm_label = str(res.get("bt_benchmark_label", "SPY Buy & Hold"))
                        _laa_bm_ret = None
                        if isinstance(_laa_bm_series, pd.Series):
                            _laa_bm_series = _laa_bm_series.dropna()
                            _laa_bm_series = _laa_bm_series[_laa_bm_series.index >= pd.Timestamp(_laa_start)]
                            if len(_laa_bm_series) > 1 and float(_laa_bm_series.iloc[0]) > 0:
                                _laa_bm_ret = (_laa_bm_series / float(_laa_bm_series.iloc[0]) - 1.0) * 100.0

                        _eq_ret = (_laa_eq["equity"] / float(_laa_bt_cap) - 1.0) * 100.0
                        # Redundant manual charts removed - handled by _render_performance_analysis below

                        _render_performance_analysis(
                            equity_series=_laa_eq["equity"],
                            benchmark_series=_laa_bm_series if isinstance(_laa_bm_series, pd.Series) else None,
                            strategy_metrics=_laa_m,
                            strategy_label="LAA 전략",
                            benchmark_label=_laa_bm_label,
                            monte_carlo_sims=400,
                        )
            else:
                if "LAA" not in _auto_signal_strategies:
                    st.info("LAA 비중이 0%라 자동 시그널 계산을 생략했습니다.")
                else:
                    st.info("LAA 시그널이 계산되지 않았습니다.")

        # ── 듀얼모멘텀 전략 상세 ──
        if "DM" in _detail_tab_map:
          with _detail_tab_map["DM"]:
            _dm_hdr_col1, _dm_hdr_col2 = st.columns([4, 1])
            with _dm_hdr_col1:
                st.subheader("듀얼모멘텀 전략 포트폴리오")
            with _dm_hdr_col2:
                if st.button("📖 전략 가이드", key="pen_dm_guide_btn"):
                    components.html("""
                    <script>
                    const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    for (let t of tabs) {
                        if (t.textContent.includes('전략 가이드')) { t.click(); break; }
                    }
                    </script>
                    """, height=0)

            st.caption("공격 ETF 2종 상대모멘텀 + 카나리아 절대모멘텀 기반으로 공격/방어 1종을 선택합니다.")

            if _dm_res:
                if _dm_res.get("error"):
                    st.error(_dm_res["error"])
                else:
                    _dm_sig = _dm_res["signal"]
                    _d1, _d2, _d3, _d4 = st.columns(4)
                    _d1.metric("선택 자산", str(_dm_sig.get("target_ticker", "-")))
                    _d2.metric("국내 ETF", _fmt_etf_code_name(_dm_sig.get("target_kr_code", "")))
                    _d3.metric("카나리아 수익률", f"{float(_dm_sig.get('canary_return', 0.0)) * 100:+.2f}%")
                    _d4.metric("권장 동작", str(_dm_res.get("action", "HOLD")))
                    st.info(str(_dm_sig.get("reason", "")))

                    if "source_map" in _dm_res:
                        st.caption("시그널 데이터(국내): " + ", ".join([f"{k}→{v}" for k, v in _dm_res["source_map"].items()]))

                    _dm_w = _dm_settings.get("momentum_weights", {}) or {}
                    _w1v = float(_dm_w.get("m1", 12.0))
                    _w3v = float(_dm_w.get("m3", 4.0))
                    _w6v = float(_dm_w.get("m6", 2.0))
                    _w12v = float(_dm_w.get("m12", 1.0))
                    _lbv = int(_dm_settings.get("lookback", 12))
                    _tdv = int(_dm_settings.get("trading_days_per_month", 22))
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

                    _score_df = _dm_res.get("score_df")
                    if isinstance(_score_df, pd.DataFrame) and not _score_df.empty:
                        st.subheader("요약 점수")
                        st.dataframe(_score_df, use_container_width=True, hide_index=True)

                    _mom_df = _dm_res.get("momentum_detail_df")
                    if isinstance(_mom_df, pd.DataFrame) and not _mom_df.empty:
                        st.subheader("모멘텀 계산 과정")
                        st.dataframe(_mom_df, use_container_width=True, hide_index=True)

                    _exp_df = _dm_res.get("expected_rebalance_df")
                    _meta = _dm_res.get("expected_meta", {}) or {}
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

                    _dm_bt = _dm_res.get("bt_result")
                    if isinstance(_dm_bt, dict):
                        _dm_eq = _dm_bt.get("equity_df")
                        if isinstance(_dm_eq, pd.DataFrame) and not _dm_eq.empty and "equity" in _dm_eq.columns:
                            _dm_m = _dm_bt.get("metrics", {}) or {}
                            _dm_start = str(_dm_res.get("bt_start_date", _pen_bt_start_raw))
                            _dm_bt_cap = float(_dm_res.get("bt_initial_cap", _pen_bt_cap) or _pen_bt_cap)
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

                            _dm_kr_map_local = _dm_settings.get("kr_etf_map", {}) or {}
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
                            if _actual_dm_codes:
                                _actual_pos_label = ", ".join([_fmt_etf_code_name(c) for c in _actual_dm_codes])
                            else:
                                _actual_pos_label = "CASH"

                            st.write(f"**전략 성과 ({_dm_start} ~ 현재)**")
                            st.write(
                                f"수익률: **{_dm_total_ret:+.2f}%** | MDD: **{_dm_mdd:.2f}%** | CAGR: **{_dm_cagr:.2f}%**"
                            )
                            st.write(f"최종자산: {_dm_final_eq:,.0f}원 (초기자본 {_dm_bt_cap:,.0f}원 기준)")

                            st.divider()
                            dc1, dc2, dc3, dc4 = st.columns(4)
                            dc1.metric(
                                "백테스트 자산",
                                f"{_dm_final_eq:,.0f}원",
                                delta=f"{_dm_total_ret:+.2f}%",
                            )
                            dc1.caption(f"초기자본 {_dm_bt_cap:,.0f}원 기준")
                            dc2.metric(
                                "실제 총자산",
                                f"{_actual_total_v:,.0f}원" if _bal_valid else "조회 불가",
                            )
                            dc3.metric("백테/실제 포지션", f"{_bt_pos_label} / {_actual_pos_label}")
                            dc4.metric("포지션 동기화", "일치" if _sync_dm else "불일치")

                            _dm_bm_series = _dm_res.get("bt_benchmark_series")
                            _dm_bm_label = str(_dm_res.get("bt_benchmark_label", "SPY Buy & Hold"))
                            _dm_bm_ret = None
                            if isinstance(_dm_bm_series, pd.Series):
                                _dm_bm_series = _dm_bm_series.dropna()
                                _dm_bm_series = _dm_bm_series[_dm_bm_series.index >= pd.Timestamp(_dm_start)]
                                if len(_dm_bm_series) > 1 and float(_dm_bm_series.iloc[0]) > 0:
                                    _dm_bm_ret = (_dm_bm_series / float(_dm_bm_series.iloc[0]) - 1.0) * 100.0

                            _dm_eq_ret = (_dm_eq["equity"] / float(_dm_bt_cap) - 1.0) * 100.0
                            # Redundant manual charts removed - handled by _render_performance_analysis below

                            _render_performance_analysis(
                                equity_series=_dm_eq["equity"],
                                benchmark_series=_dm_bm_series if isinstance(_dm_bm_series, pd.Series) else None,
                                strategy_metrics=_dm_m,
                                strategy_label="듀얼모멘텀 전략",
                                benchmark_label=_dm_bm_label,
                                monte_carlo_sims=400,
                            )
            else:
                if "듀얼모멘텀" not in _auto_signal_strategies:
                    st.info("듀얼모멘텀 비중이 0%라 자동 시그널 계산을 생략했습니다.")
                else:
                    st.info("듀얼모멘텀 시그널이 계산되지 않았습니다.")

        # ── VAA 전략 상세 ──
        if "VAA" in _detail_tab_map:
          with _detail_tab_map["VAA"]:
            st.subheader("VAA 전략 포트폴리오")
            st.caption("공격자산 4종 13612W 모멘텀 → 양수 최고 1개 선택, 전부 음수 시 방어자산 최고 1개")
            if _vaa_res:
                if _vaa_res.get("error"):
                    st.error(str(_vaa_res["error"]))
                else:
                    _vs = _vaa_res.get("signal", {})
                    _v1, _v2, _v3, _v4 = st.columns(4)
                    _tgt_tickers = _vs.get("target_tickers", [])
                    _v1.metric("선택 자산", ", ".join(_tgt_tickers) if _tgt_tickers else "-")
                    _v2.metric("포지션", "공격" if _vs.get("is_offensive") else "방어")
                    _v3.metric("선택 ETF", ", ".join([_fmt_etf_code_name(c) for c in _vs.get("target_kr_codes", [])]))
                    _v4.metric("권장 동작", str(_vaa_res.get("action", "HOLD")))
                    st.info(str(_vs.get("reason", "")))

                    # 모멘텀 스코어 표시
                    _off_sc = _vs.get("offensive_scores", {})
                    _def_sc = _vs.get("defensive_scores", {})
                    if _off_sc or _def_sc:
                        _score_rows = []
                        for t, s in _off_sc.items():
                            _score_rows.append({"티커": t, "유형": "공격", "모멘텀": round(s * 100, 2)})
                        for t, s in _def_sc.items():
                            _score_rows.append({"티커": t, "유형": "방어", "모멘텀": round(s * 100, 2)})
                        st.dataframe(pd.DataFrame(_score_rows), use_container_width=True, hide_index=True)

                    # 배분 테이블
                    _vaa_alloc = _vaa_res.get("alloc_df")
                    if isinstance(_vaa_alloc, pd.DataFrame) and not _vaa_alloc.empty:
                        st.markdown("**목표 배분 vs 현재 보유**")
                        st.dataframe(_vaa_alloc, use_container_width=True, hide_index=True)
            else:
                if "VAA" not in _auto_signal_strategies:
                    st.info("VAA 비중이 0%라 자동 시그널 계산을 생략했습니다.")
                else:
                    st.info("VAA 시그널이 계산되지 않았습니다.")

        # ── CDM 전략 상세 ──
        if "CDM" in _detail_tab_map:
          with _detail_tab_map["CDM"]:
            st.subheader("CDM 전략 포트폴리오")
            st.caption("4모듈 듀얼모멘텀 — 각 모듈 상대+절대 모멘텀 (12개월 수익률)")
            if _cdm_res:
                if _cdm_res.get("error"):
                    st.error(str(_cdm_res["error"]))
                else:
                    _cs = _cdm_res.get("signal", {})
                    _c1, _c2, _c3, _c4 = st.columns(4)
                    _c1.metric("공격 모듈 수", f"{_cs.get('offensive_count', 0)}/{_cs.get('total_modules', 4)}")
                    _c2.metric("방어자산 12M수익률", f"{_cs.get('defensive_return', 0):+.2f}%")
                    _c3.metric("권장 동작", str(_cdm_res.get("action", "HOLD")))
                    _tw_kr = _cs.get("target_weights_kr", {})
                    _top_etf = max(_tw_kr, key=_tw_kr.get) if _tw_kr else "-"
                    _c4.metric("최대 비중 ETF", _fmt_etf_code_name(_top_etf) if _top_etf != "-" else "-")
                    st.info(str(_cs.get("reason", "")))

                    # 모듈별 결과 표시
                    _mod_results = _cs.get("module_results", [])
                    if _mod_results:
                        _mod_rows = []
                        for m in _mod_results:
                            _mod_rows.append({
                                "모듈": f"M{m['module']}",
                                "페어": " vs ".join(m['pair']),
                                "승자": m['winner'],
                                "12M수익률(%)": m['winner_return'],
                                "포지션": "공격" if m['is_offensive'] else "방어",
                            })
                        st.dataframe(pd.DataFrame(_mod_rows), use_container_width=True, hide_index=True)

                    # 배분 테이블
                    _cdm_alloc = _cdm_res.get("alloc_df")
                    if isinstance(_cdm_alloc, pd.DataFrame) and not _cdm_alloc.empty:
                        st.markdown("**목표 배분 vs 현재 보유**")
                        st.dataframe(_cdm_alloc, use_container_width=True, hide_index=True)
            else:
                if "CDM" not in _auto_signal_strategies:
                    st.info("CDM 비중이 0%라 자동 시그널 계산을 생략했습니다.")
                else:
                    st.info("CDM 시그널이 계산되지 않았습니다.")

    # ══════════════════════════════════════════════════════════════
    # Tab 2: 백테스트
    # ══════════════════════════════════════════════════════════════
    with tab_p2:
        if _pen_local_first:
            st.info("백테스트 가격 데이터: 로컬 파일(cache/data) 우선, 부족 시 API 보강 모드입니다.")
        _bt_candidates = [s for s in ["LAA", "듀얼모멘텀", "VAA", "CDM"] if s in _active_strategies]
        # 듀얼모멘텀/VAA/CDM 백테스트는 항상 선택 가능하게 제공
        for _s in ["듀얼모멘텀", "VAA", "CDM"]:
            if _s not in _bt_candidates:
                _bt_candidates.append(_s)
        if not _bt_candidates:
            st.warning("포트폴리오에 활성화된 전략이 없어 백테스트를 실행할 수 없습니다.")
        else:
            _bt_strategy = st.selectbox("백테스트 전략", _bt_candidates, key="pen_bt_strategy_select")

            pen_bt_start = st.date_input("시작일", value=_pen_bt_start_ts.date(), key="pen_bt_start_date")
            pen_bt_cap = st.number_input("초기 자본 (KRW)", value=int(_pen_bt_cap), step=1_000_000, key="pen_bt_cap")
            pen_bt_fee = st.number_input("수수료 (%)", value=0.02, format="%.2f", key="pen_bt_fee") / 100.0
            _pen_bt_start_filter_ts = pd.Timestamp(pen_bt_start).normalize()

            if _bt_strategy == "LAA":
                st.header("LAA 백테스트")
                st.caption("국내 ETF 매핑(SPY/IWD/GLD/IEF/QQQ/SHY) 기반 월간 리밸런싱 시뮬레이션")

                if st.button("LAA 백테스트 실행", key="pen_bt_run_laa", type="primary"):
                    with st.spinner("LAA 백테스트 실행 중... (국내 데이터 조회)"):
                        tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
                        source_map = {
                            "SPY": _code_only(kr_spy or kr_iwd or "360750"),
                            "IWD": _code_only(kr_iwd),
                            "GLD": _code_only(kr_gld),
                            "IEF": _code_only(kr_ief),
                            "QQQ": _code_only(kr_qqq),
                            "SHY": _code_only(kr_shy),
                        }
                        # 백테스트: yfinance 미국 원본 티커 사용 (국내 ETF 상장일 제약 해소)
                        price_data = {}
                        for ticker in tickers:
                            df_t = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
                            if df_t is None or df_t.empty:
                                st.error(f"{ticker} yfinance 데이터 조회 실패")
                                price_data = None
                                break
                            df_t = df_t.copy().sort_index()
                            df_t = df_t[df_t.index >= _pen_bt_start_filter_ts]
                            if df_t.empty:
                                st.error(f"{ticker} 시작일 이후 데이터가 없습니다. 시작일을 조정해 주세요.")
                                price_data = None
                                break
                            price_data[ticker] = df_t

                        if price_data:
                            strategy = LAAStrategy(settings={"kr_etf_map": _kr_etf_map})
                            bt_result = strategy.run_backtest(price_data, initial_balance=float(pen_bt_cap), fee=float(pen_bt_fee))
                            if bt_result:
                                st.session_state["pen_bt_laa_result"] = bt_result
                                st.session_state["pen_bt_result"] = bt_result
                                _laa_bm_ticker = "SPY"
                                _laa_bm_series = _normalize_numeric_series(
                                    price_data.get(_laa_bm_ticker),
                                    preferred_cols=("close", "Close"),
                                )
                                if not _laa_bm_series.empty:
                                    st.session_state["pen_bt_laa_benchmark_series"] = _laa_bm_series
                                    st.session_state["pen_bt_laa_benchmark_label"] = f"{_laa_bm_ticker} Buy & Hold"
                            else:
                                st.error("백테스트 실행 실패 (데이터 부족)")

                bt_res = st.session_state.get("pen_bt_laa_result") or st.session_state.get("pen_bt_result")
                if bt_res:
                    metrics = bt_res["metrics"]
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("총 수익률", f"{metrics['total_return']:.2f}%")
                    mc2.metric("CAGR", f"{metrics['cagr']:.2f}%")
                    mc3.metric("MDD", f"{metrics['mdd']:.2f}%")
                    mc4.metric("샤프", f"{metrics['sharpe']:.2f}")
                    mc5.metric("최종 자산", f"{metrics['final_equity']:,.0f}")

                    eq_df = bt_res["equity_df"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq_df.index, y=eq_df["equity"],
                        name="포트폴리오", line=dict(color="royalblue"),
                    ))
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

                    _laa_bm_series = st.session_state.get("pen_bt_laa_benchmark_series")
                    _laa_bm_label = str(st.session_state.get("pen_bt_laa_benchmark_label", "SPY Buy & Hold"))
                    _render_performance_analysis(
                        equity_series=eq_df.get("equity"),
                        benchmark_series=_laa_bm_series,
                        strategy_metrics=metrics,
                        strategy_label="LAA 전략",
                        benchmark_label=_laa_bm_label,
                        show_drawdown=True,
                        show_weight=True,
                        equity_df=eq_df
                    )

            elif _bt_strategy == "듀얼모멘텀":
                st.header("듀얼모멘텀 백테스트")
                st.caption("사이드바의 국내 ETF 설정(공격 2종/방어/카나리아) 기반 월간 리밸런싱 시뮬레이션")

                if not _dm_settings:
                    st.warning("사이드바에서 듀얼모멘텀 설정을 먼저 입력해 주세요.")
                else:
                    if st.button("듀얼모멘텀 백테스트 실행", key="pen_bt_run_dm", type="primary"):
                        with st.spinner("듀얼모멘텀 백테스트 실행 중... (국내 데이터 조회)"):
                            dm_tickers = []
                            for tk in (_dm_settings.get("offensive", []) + _dm_settings.get("defensive", []) + _dm_settings.get("canary", [])):
                                tku = str(tk).strip().upper()
                                if tku and tku not in dm_tickers:
                                    dm_tickers.append(tku)

                            dm_price_data = {}
                            dm_kr_map = _dm_settings.get("kr_etf_map", {}) or {}
                            for ticker in dm_tickers:
                                kr_code = str(dm_kr_map.get(ticker, "")).strip()
                                if not kr_code:
                                    st.error(f"{ticker} 국내 ETF 매핑이 없습니다.")
                                    dm_price_data = None
                                    break
                                df_t = _get_pen_daily_chart(kr_code, count=3000, use_disk_cache=True)
                                if df_t is None or df_t.empty:
                                    st.error(f"{ticker} ({kr_code}) 로컬 데이터가 없습니다. cache 또는 data 폴더를 확인하세요.")
                                    dm_price_data = None
                                    break
                                df_t = df_t.copy().sort_index()
                                if "close" not in df_t.columns and "Close" in df_t.columns:
                                    df_t["close"] = df_t["Close"]
                                if "close" not in df_t.columns:
                                    st.error(f"{ticker} ({kr_code}) 종가 컬럼이 없습니다.")
                                    dm_price_data = None
                                    break
                                df_t = df_t[df_t.index >= _pen_bt_start_filter_ts]
                                if df_t.empty:
                                    st.error(f"{ticker} ({kr_code}) 시작일 이후 데이터가 없습니다. 시작일을 조정해 주세요.")
                                    dm_price_data = None
                                    break
                                dm_price_data[ticker] = df_t

                            if dm_price_data:
                                dm_strategy = DualMomentumStrategy(settings=_dm_settings)
                                dm_bt_result = dm_strategy.run_backtest(
                                    dm_price_data,
                                    initial_balance=float(pen_bt_cap),
                                    fee=float(pen_bt_fee),
                                )
                                if dm_bt_result:
                                    st.session_state["pen_bt_dm_result"] = dm_bt_result
                                    _dm_bm_ticker = ""
                                    for _t in (_dm_settings.get("offensive", []) or []):
                                        if _t in dm_price_data:
                                            _dm_bm_ticker = str(_t)
                                            break
                                    if not _dm_bm_ticker:
                                        for _t in dm_tickers:
                                            if _t in dm_price_data:
                                                _dm_bm_ticker = str(_t)
                                                break

                                    _dm_bm_series = _normalize_numeric_series(
                                        dm_price_data.get(_dm_bm_ticker),
                                        preferred_cols=("close", "Close"),
                                    )
                                    if not _dm_bm_series.empty:
                                        st.session_state["pen_bt_dm_benchmark_series"] = _dm_bm_series
                                        _dm_bm_code = _code_only(str(dm_kr_map.get(_dm_bm_ticker, "")))
                                        if _dm_bm_code:
                                            st.session_state["pen_bt_dm_benchmark_label"] = f"{_fmt_etf_code_name(_dm_bm_code)} Buy & Hold"
                                        else:
                                            st.session_state["pen_bt_dm_benchmark_label"] = f"{_dm_bm_ticker} Buy & Hold"
                                else:
                                    st.error("백테스트 실행 실패 (데이터 부족)")

                    dm_res = st.session_state.get("pen_bt_dm_result")
                    if dm_res:
                        metrics = dm_res["metrics"]
                        _render_performance_analysis(
                            equity_series=eq_df.get("equity"),
                            benchmark_series=_dm_bm_series,
                            strategy_metrics=metrics,
                            strategy_label="듀얼모멘텀 전략",
                            benchmark_label=_dm_bm_label,
                            show_drawdown=True,
                            show_weight=True,
                            equity_df=eq_df
                        )

            elif _bt_strategy == "VAA":
                st.header("VAA 백테스트")
                st.caption("13612W 모멘텀 기반 공격/방어 전환, yfinance 미국 원본 데이터 사용")

                if st.button("VAA 백테스트 실행", key="pen_bt_run_vaa", type="primary"):
                    with st.spinner("VAA 백테스트 실행 중..."):
                        vaa_tickers = list(set(_vaa_settings.get("offensive", []) + _vaa_settings.get("defensive", [])))
                        vaa_price_data = {}
                        for ticker in vaa_tickers:
                            df_t = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
                            if df_t is None or df_t.empty:
                                st.error(f"{ticker} yfinance 데이터 조회 실패")
                                vaa_price_data = None
                                break
                            df_t = df_t.copy().sort_index()
                            df_t = df_t[df_t.index >= _pen_bt_start_filter_ts]
                            if df_t.empty:
                                st.error(f"{ticker} 시작일 이후 데이터가 없습니다.")
                                vaa_price_data = None
                                break
                            vaa_price_data[ticker] = df_t

                        if vaa_price_data:
                            from src.strategy.vaa import VAAStrategy
                            vaa_strategy = VAAStrategy(settings=_vaa_settings)
                            vaa_bt_result = vaa_strategy.run_backtest(
                                vaa_price_data,
                                initial_balance=float(pen_bt_cap),
                                fee=float(pen_bt_fee),
                            )
                            if vaa_bt_result:
                                st.session_state["pen_bt_vaa_result"] = vaa_bt_result
                                _vaa_bm_ticker = "SPY"
                                if _vaa_bm_ticker in vaa_price_data:
                                    st.session_state["pen_bt_vaa_benchmark_series"] = _normalize_numeric_series(
                                        vaa_price_data.get(_vaa_bm_ticker),
                                        preferred_cols=("close", "Close"),
                                    )
                                    st.session_state["pen_bt_vaa_benchmark_label"] = "SPY Buy & Hold"
                            else:
                                st.error("VAA 백테스트 실행 실패 (데이터 부족)")

                vaa_bt_res = st.session_state.get("pen_bt_vaa_result")
                if vaa_bt_res:
                    metrics = vaa_bt_res["metrics"]
                    _vaa_bm_series = st.session_state.get("pen_bt_vaa_benchmark_series")
                    _vaa_bm_label = str(st.session_state.get("pen_bt_vaa_benchmark_label", "SPY Buy & Hold"))
                    _render_performance_analysis(
                        equity_series=eq_df.get("equity"),
                        benchmark_series=_vaa_bm_series,
                        strategy_metrics=metrics,
                        strategy_label="VAA 전략",
                        benchmark_label=_vaa_bm_label,
                        show_drawdown=True,
                        show_weight=True,
                        equity_df=eq_df
                    )
                    
                    pos_df = vaa_bt_res.get("positions")
                    if isinstance(pos_df, pd.DataFrame) and not pos_df.empty:
                        st.subheader("월별 포지션 이력")
                        st.dataframe(pos_df.tail(36), use_container_width=True, hide_index=True)

            elif _bt_strategy == "CDM":
                st.header("CDM 백테스트")
                st.caption("4모듈 듀얼모멘텀, yfinance 미국 원본 데이터 사용")

                if st.button("CDM 백테스트 실행", key="pen_bt_run_cdm", type="primary"):
                    with st.spinner("CDM 백테스트 실행 중..."):
                        cdm_tickers = list(set(_cdm_settings.get("offensive", []) + _cdm_settings.get("defensive", [])))
                        cdm_price_data = {}
                        for ticker in cdm_tickers:
                            df_t = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
                            if df_t is None or df_t.empty:
                                st.error(f"{ticker} yfinance 데이터 조회 실패")
                                cdm_price_data = None
                                break
                            df_t = df_t.copy().sort_index()
                            df_t = df_t[df_t.index >= _pen_bt_start_filter_ts]
                            if df_t.empty:
                                st.error(f"{ticker} 시작일 이후 데이터가 없습니다.")
                                cdm_price_data = None
                                break
                            cdm_price_data[ticker] = df_t

                        if cdm_price_data:
                            from src.strategy.cdm import CDMStrategy
                            cdm_strategy = CDMStrategy(settings=_cdm_settings)
                            cdm_bt_result = cdm_strategy.run_backtest(
                                cdm_price_data,
                                initial_balance=float(pen_bt_cap),
                                fee=float(pen_bt_fee),
                            )
                            if cdm_bt_result:
                                st.session_state["pen_bt_cdm_result"] = cdm_bt_result
                                _cdm_bm_ticker = "SPY"
                                if _cdm_bm_ticker in cdm_price_data:
                                    st.session_state["pen_bt_cdm_benchmark_series"] = _normalize_numeric_series(
                                        cdm_price_data.get(_cdm_bm_ticker),
                                        preferred_cols=("close", "Close"),
                                    )
                                    st.session_state["pen_bt_cdm_benchmark_label"] = "SPY Buy & Hold"
                            else:
                                st.error("CDM 백테스트 실행 실패 (데이터 부족)")

                cdm_bt_res = st.session_state.get("pen_bt_cdm_result")
                if cdm_bt_res:
                    metrics = cdm_bt_res["metrics"]
                    _cdm_bm_series = st.session_state.get("pen_bt_cdm_benchmark_series")
                    _cdm_bm_label = str(st.session_state.get("pen_bt_cdm_benchmark_label", "SPY Buy & Hold"))
                    _render_performance_analysis(
                        equity_series=eq_df.get("equity"),
                        benchmark_series=_cdm_bm_series,
                        strategy_metrics=metrics,
                        strategy_label="CDM 전략",
                        benchmark_label=_cdm_bm_label,
                        show_drawdown=True,
                        show_weight=True,
                        equity_df=eq_df
                    )

                    alloc_df = cdm_bt_res.get("allocations")
                    if isinstance(alloc_df, pd.DataFrame) and not alloc_df.empty:
                        st.subheader("월별 배분 이력")
                        st.dataframe(alloc_df.tail(36), use_container_width=True, hide_index=True)

            else:
                st.info("선택된 전략의 백테스트를 준비 중입니다.")

    # ══════════════════════════════════════════════════════════════
    # Tab 3: 수동 주문
    # ══════════════════════════════════════════════════════════════
    with tab_p3:
        st.header("수동 주문")

        bal = st.session_state.get(pen_bal_key)
        if not bal:
            st.warning("잔고를 먼저 조회해 주세요.")
        else:
            _bal_sum = _compute_kis_balance_summary(bal)
            cash = _bal_sum["cash"]
            buyable_cash = _bal_sum["buyable_cash"]
            holdings = _bal_sum["holdings"]

            # 매매 대상 ETF 선택 (활성 전략 전체 반영)
            all_etf_codes = []
            all_etf_codes.extend([kr_iwd, kr_gld, kr_ief, kr_qqq, kr_shy])
            if _dm_settings:
                _dm_map_local = _dm_settings.get("kr_etf_map", {}) or {}
                all_etf_codes.extend([_dm_map_local.get("SPY", ""), _dm_map_local.get("EFA", ""), _dm_map_local.get("AGG", "")])
            if _vaa_settings:
                _vaa_map_local = _vaa_settings.get("kr_etf_map", {}) or {}
                all_etf_codes.extend([str(v) for v in _vaa_map_local.values() if str(v).strip()])
            if _cdm_settings:
                _cdm_map_local = _cdm_settings.get("kr_etf_map", {}) or {}
                all_etf_codes.extend([str(v) for v in _cdm_map_local.values() if str(v).strip()])
            all_etf_codes = [str(c).strip() for c in all_etf_codes if str(c).strip()]
            all_etf_codes = list(dict.fromkeys(all_etf_codes))
            if not all_etf_codes:
                all_etf_codes = ["360750"]

            etf_options = {_fmt_etf_code_name(c): c for c in all_etf_codes}
            selected_pen_label = st.selectbox("매매 ETF 선택", list(etf_options.keys()), key="pen_trade_etf_label")
            selected_pen_etf = etf_options[selected_pen_label]
            selected_pen_quote_etf, _pen_quote_note = _resolve_pen_quote_code(str(selected_pen_etf))
            _pen_quote_substituted = str(selected_pen_quote_etf) != str(selected_pen_etf)
            if _pen_quote_substituted and _pen_quote_note:
                st.warning(_pen_quote_note)
                st.caption(
                    f"주문 코드는 {_fmt_etf_code_name(selected_pen_etf)} 그대로 유지됩니다. "
                    f"시세/차트/호가만 {_fmt_etf_code_name(selected_pen_quote_etf)} 기준으로 표시합니다."
                )
            _pen_order_disabled = bool(IS_CLOUD or _pen_quote_substituted)

            pen_holding = next((h for h in holdings if str(h.get("code", "")) == str(selected_pen_etf)), None)
            pen_qty = int(pen_holding.get("qty", 0)) if pen_holding else 0

            # ── 상단 정보 바 (골드 패널과 동일 구조) ──
            cur_price = _get_pen_current_price(str(selected_pen_quote_etf))
            _pen_cur = float(cur_price) if cur_price and cur_price > 0 else 0
            _pen_eval = _pen_cur * pen_qty if _pen_cur > 0 else 0

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("현재가", f"{_pen_cur:,.0f}원" if _pen_cur > 0 else "-")
            pc2.metric(f"{_fmt_etf_code_name(selected_pen_etf)} 보유", f"{pen_qty}주")
            pc3.metric("평가금액", f"{_pen_eval:,.0f}원")
            pc4.metric("매수 가능금액", f"{buyable_cash:,.0f}원")

            # ═══ 일봉 차트 (상단 전체폭) ═══
            _pen_chart_df = _get_pen_daily_chart(str(selected_pen_quote_etf), count=260)
            if _pen_chart_df is not None and len(_pen_chart_df) > 0:
                _pen_chart_df = _pen_chart_df.copy().sort_index()
                if "close" not in _pen_chart_df.columns and "Close" in _pen_chart_df.columns:
                    _pen_chart_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
                st.markdown(f"**{_fmt_etf_code_name(selected_pen_quote_etf)} 일봉 차트**")
                try:
                    _chart_last_date = pd.to_datetime(_pen_chart_df.index[-1]).date()
                    _chart_age_days = int((pd.Timestamp.now(tz="Asia/Seoul").date() - _chart_last_date).days)
                    if _chart_age_days > 7:
                        st.warning(
                            f"일봉 최신일이 {_chart_last_date} (KST)로 오래되었습니다. "
                            "실시간 조회 가능 종목코드인지 확인해 주세요."
                        )
                except Exception:
                    pass
                try:
                    _chart_start = pd.to_datetime(_pen_chart_df.index[0]).date()
                    _chart_end = pd.to_datetime(_pen_chart_df.index[-1]).date()
                    st.caption(f"차트 구간: {_chart_start} ~ {_chart_end} | 캔들: {len(_pen_chart_df):,}개")
                except Exception:
                    pass
                _fig_pen = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
                _fig_pen.add_trace(go.Candlestick(
                    x=_pen_chart_df.index, open=_pen_chart_df['open'], high=_pen_chart_df['high'],
                    low=_pen_chart_df['low'], close=_pen_chart_df['close'], name='일봉',
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                ), row=1, col=1)
                _sma_periods = [5, 20, 60, 120, 200]
                _sma_colors = {
                    5: "#FF9800",
                    20: "#2196F3",
                    60: "#00897B",
                    120: "#8D6E63",
                    200: "#455A64",
                }
                for _p in _sma_periods:
                    _sma = _pen_chart_df["close"].rolling(_p).mean()
                    _fig_pen.add_trace(
                        go.Scatter(
                            x=_pen_chart_df.index,
                            y=_sma,
                            name=f"SMA{_p}",
                            line=dict(color=_sma_colors.get(_p, "#666666"), width=1.2 if _p < 200 else 1.5),
                        ),
                        row=1,
                        col=1,
                    )
                if 'volume' in _pen_chart_df.columns:
                    _pen_vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(_pen_chart_df['close'], _pen_chart_df['open'])]
                    _fig_pen.add_trace(go.Bar(x=_pen_chart_df.index, y=_pen_chart_df['volume'], marker_color=_pen_vol_colors, name='거래량', showlegend=False), row=2, col=1)
                _fig_pen.update_layout(
                    height=450, margin=dict(l=0, r=0, t=10, b=30),
                    xaxis_rangeslider_visible=False, showlegend=True,
                    legend=dict(orientation="h", y=1.06, x=0),
                    xaxis2=dict(showticklabels=True, tickformat='%m/%d', tickangle=-45),
                    yaxis=dict(title="", side="right"),
                    yaxis2=dict(title="", side="right"),
                )
                st.plotly_chart(_fig_pen, use_container_width=True, key=f"pen_manual_chart_{selected_pen_etf}")
            else:
                st.info("차트 데이터 로딩 중...")


            st.divider()

            ob_col, order_col = st.columns([2, 3])

            # ── 좌: 호가창 ──
            with ob_col:
                ob = _get_pen_orderbook(str(selected_pen_quote_etf))
                if ob and ob.get("asks") and ob.get("bids"):
                    asks = ob["asks"]
                    bids = ob["bids"]
                    all_qtys = [a["qty"] for a in asks] + [b["qty"] for b in bids]
                    max_qty = max(all_qtys) if all_qtys else 1
                    html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                    html.append(
                        '<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666">'
                        '<td>구분</td><td style="text-align:right">잔량</td>'
                        '<td style="text-align:right">가격(원)</td>'
                        '<td style="text-align:right">등락</td><td>비율</td></tr>'
                    )
                    for a in reversed(asks):
                        ap, aq = a["price"], a["qty"]
                        diff = ((ap / _pen_cur) - 1) * 100 if _pen_cur > 0 else 0
                        bar_w = int(aq / max_qty * 100) if max_qty > 0 else 0
                        html.append(
                            f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                            f'<td>매도</td>'
                            f'<td style="text-align:right">{aq:,}</td>'
                            f'<td style="text-align:right;font-weight:bold">{ap:,.0f}</td>'
                            f'<td style="text-align:right">{diff:+.2f}%</td>'
                            f'<td><div style="background:#1976D233;height:14px;width:{bar_w}%"></div></td></tr>'
                        )
                    html.append(
                        f'<tr style="background:#FFF3E0;border:2px solid #FF9800;height:32px;font-weight:bold">'
                        f'<td colspan="2" style="color:#E65100">현재가</td>'
                        f'<td style="text-align:right;color:#E65100;font-size:15px">{_pen_cur:,.0f}</td>'
                        f'<td colspan="2"></td></tr>'
                    )
                    for b in bids:
                        bp, bq = b["price"], b["qty"]
                        diff = ((bp / _pen_cur) - 1) * 100 if _pen_cur > 0 else 0
                        bar_w = int(bq / max_qty * 100) if max_qty > 0 else 0
                        html.append(
                            f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                            f'<td>매수</td>'
                            f'<td style="text-align:right">{bq:,}</td>'
                            f'<td style="text-align:right;font-weight:bold">{bp:,.0f}</td>'
                            f'<td style="text-align:right">{diff:+.2f}%</td>'
                            f'<td><div style="background:#D32F2F33;height:14px;width:{bar_w}%"></div></td></tr>'
                        )
                    html.append("</table>")
                    st.markdown("".join(html), unsafe_allow_html=True)
                    if asks and bids:
                        spread = asks[0]["price"] - bids[0]["price"]
                        spread_pct = (spread / _pen_cur * 100) if _pen_cur > 0 else 0
                        total_ask_q = sum(a["qty"] for a in asks)
                        total_bid_q = sum(b["qty"] for b in bids)
                        st.caption(
                            f"스프레드: {spread:,.0f}원 ({spread_pct:.3f}%) | "
                            f"매도잔량: {total_ask_q:,} | 매수잔량: {total_bid_q:,}"
                        )
                else:
                    st.info("호가 데이터를 불러오는 중...")

            # ── 우: 주문 패널 ──
            with order_col:
                if _pen_quote_substituted:
                    st.info(
                        f"{_fmt_etf_code_name(selected_pen_etf)} 주문은 비활성화했습니다. "
                        f"설정에서 종목코드를 {_fmt_etf_code_name(selected_pen_quote_etf)}로 변경한 뒤 주문해 주세요."
                    )
                buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                with buy_tab:
                    pen_buy_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"], key="pen_buy_method", horizontal=True)

                    pen_buy_price = 0
                    if pen_buy_method == "지정가":
                        pen_buy_price = st.number_input("매수 지정가 (원)", min_value=0, value=int(_pen_cur) if _pen_cur > 0 else 0, step=50, key="pen_buy_price")
                    else:
                        pen_buy_price = _pen_cur

                    pen_buy_qty = st.number_input("매수 수량 (주)", min_value=0, value=0, step=1, key="pen_buy_qty")

                    _pen_buy_unit = pen_buy_price if pen_buy_price > 0 else _pen_cur
                    _pen_total = int(pen_buy_qty * _pen_buy_unit) if pen_buy_qty > 0 and _pen_buy_unit > 0 else 0
                    st.markdown(
                        f"<div style='background:#fff3f3;border:1px solid #ffcdd2;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                        f"<b>단가:</b> {_pen_buy_unit:,.0f}원 &nbsp;|&nbsp; "
                        f"<b>수량:</b> {pen_buy_qty:,}주 &nbsp;|&nbsp; "
                        f"<b>총 금액:</b> <span style='color:#D32F2F;font-weight:bold'>{_pen_total:,}원</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    pen_buy_amt = _pen_total

                    if st.button("매수 실행", key="pen_exec_buy", type="primary", disabled=_pen_order_disabled):
                        if pen_buy_amt <= 0:
                            st.error("매수 금액을 입력해 주세요.")
                        elif _pen_total > buyable_cash:
                            st.error(f"매수 가능금액 부족 (필요: {_pen_total:,}원 / 가능: {buyable_cash:,.0f}원)")
                        else:
                            with st.spinner("매수 주문 실행 중..."):
                                if pen_buy_method == "동시호가 (장마감)":
                                    result = trader.execute_closing_auction_buy(str(selected_pen_etf), pen_buy_qty) if pen_buy_qty > 0 else None
                                elif pen_buy_method == "지정가" and pen_buy_price > 0:
                                    result = trader.send_order("BUY", str(selected_pen_etf), pen_buy_qty, price=pen_buy_price, ord_dvsn="00") if pen_buy_qty > 0 else None
                                elif pen_buy_method == "시간외 종가":
                                    result = trader.send_order("BUY", str(selected_pen_etf), pen_buy_qty, price=0, ord_dvsn="06") if pen_buy_qty > 0 else None
                                else:
                                    result = trader.send_order("BUY", str(selected_pen_etf), pen_buy_qty, ord_dvsn="01") if pen_buy_qty > 0 else None
                                if result and (isinstance(result, dict) and result.get("success")):
                                    st.success(f"매수 완료: {result}")
                                    st.session_state[pen_bal_key] = trader.get_balance()
                                else:
                                    st.error(f"매수 실패: {result}")

                with sell_tab:
                    pen_sell_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"], key="pen_sell_method", horizontal=True)

                    pen_sell_price = 0
                    if pen_sell_method == "지정가":
                        pen_sell_price = st.number_input("매도 지정가 (원)", min_value=0, value=int(_pen_cur) if _pen_cur > 0 else 0, step=50, key="pen_sell_price")
                    else:
                        pen_sell_price = _pen_cur

                    pen_sell_qty = st.number_input("매도 수량 (주)", min_value=0, max_value=max(pen_qty, 1), value=pen_qty, step=1, key="pen_sell_qty")
                    pen_sell_all = st.checkbox("전량 매도", value=True, key="pen_sell_all")

                    _pen_sell_unit = pen_sell_price if pen_sell_price > 0 else _pen_cur
                    _pen_sell_qty_final = pen_qty if pen_sell_all else pen_sell_qty
                    _pen_sell_total = int(_pen_sell_qty_final * _pen_sell_unit) if _pen_sell_qty_final > 0 and _pen_sell_unit > 0 else 0
                    st.markdown(
                        f"<div style='background:#f3f8ff;border:1px solid #bbdefb;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                        f"<b>단가:</b> {_pen_sell_unit:,.0f}원 &nbsp;|&nbsp; "
                        f"<b>수량:</b> {_pen_sell_qty_final:,}주 &nbsp;|&nbsp; "
                        f"<b>총 금액:</b> <span style='color:#1976D2;font-weight:bold'>{_pen_sell_total:,}원</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if st.button("매도 실행", key="pen_exec_sell", type="primary", disabled=_pen_order_disabled):
                        _sq = pen_qty if pen_sell_all else pen_sell_qty
                        if _sq <= 0:
                            st.error("매도할 수량이 없습니다.")
                        else:
                            with st.spinner("매도 주문 실행 중..."):
                                if pen_sell_method == "동시호가 (장마감)":
                                    result = trader.smart_sell_all_closing(str(selected_pen_etf)) if pen_sell_all else trader.smart_sell_qty_closing(str(selected_pen_etf), _sq)
                                elif pen_sell_method == "지정가" and pen_sell_price > 0:
                                    result = trader.send_order("SELL", str(selected_pen_etf), _sq, price=pen_sell_price, ord_dvsn="00")
                                elif pen_sell_method == "시간외 종가":
                                    result = trader.send_order("SELL", str(selected_pen_etf), _sq, price=0, ord_dvsn="06")
                                else:
                                    result = trader.smart_sell_all(str(selected_pen_etf)) if pen_sell_all else trader.smart_sell_qty(str(selected_pen_etf), _sq)
                                if result and (isinstance(result, dict) and result.get("success")):
                                    st.success(f"매도 완료: {result}")
                                    st.session_state[pen_bal_key] = trader.get_balance()
                                else:
                                    st.error(f"매도 실패: {result}")

            # ═══════════════════════════════════════════════════
            # 예약 주문 관리
            # ═══════════════════════════════════════════════════
            st.divider()
            st.subheader("예약 주문 관리")
            st.caption("주문을 예약하고 이력을 영구 보관합니다. 실행된 주문도 삭제되지 않습니다.")

            # ── 리밸런싱 주문 미리보기 ──
            _combo_rebal = st.session_state.get("pen_combined_rebal_data")
            if _combo_rebal:
                _rebal_buy = [r for r in _combo_rebal if r.get("주문 상태") == "매수" and int(r.get("매수 예정(주)", 0)) > 0]
                _rebal_sell = [r for r in _combo_rebal if r.get("주문 상태") == "매도" and int(r.get("매도 예정(주)", 0)) > 0]
                _has_rebal = bool(_rebal_buy or _rebal_sell)

                # 리밸런싱 예정일 계산
                from datetime import date as _rb_date
                import calendar as _rb_cal
                _rb_today = _rb_date.today()
                _rb_y, _rb_m = _rb_today.year, _rb_today.month
                _rb_last = _rb_cal.monthrange(_rb_y, _rb_m)[1]
                _rb_dt = _rb_date(_rb_y, _rb_m, _rb_last)
                while _rb_dt.weekday() >= 5:
                    _rb_dt -= timedelta(days=1)
                _is_rebal_day = (_rb_today == _rb_dt)
                _rb_days_left = (_rb_dt - _rb_today).days

                if _has_rebal:
                    _rb_title = "리밸런싱 주문 미리보기"
                    if _is_rebal_day:
                        _rb_title += " (오늘 리밸런싱일)"
                    with st.expander(_rb_title, expanded=_is_rebal_day):
                        if _is_rebal_day:
                            st.success("오늘은 리밸런싱 예정일입니다. 아래 주문을 확인 후 일괄 예약 등록하세요.")
                        else:
                            st.info(f"다음 리밸런싱: {_rb_dt.strftime('%Y-%m-%d')} (D-{_rb_days_left})")

                        # 매도 주문 먼저
                        if _rebal_sell:
                            st.markdown("**매도 예정**")
                            _sell_rows = []
                            for _rs in _rebal_sell:
                                _sell_rows.append({
                                    "ETF": _rs.get("ETF", ""),
                                    "현재수량(주)": _rs.get("현재수량(주)", 0),
                                    "합산 목표(주)": _rs.get("합산 목표(주)", 0),
                                    "매도수량(주)": _rs.get("매도 예정(주)", 0),
                                    "현재 비중(%)": _rs.get("현재 비중(%)", 0),
                                    "목표 비중(%)": _rs.get("목표 비중(%)", 0),
                                })
                            st.dataframe(pd.DataFrame(_sell_rows), use_container_width=True, hide_index=True)

                        # 매수 주문
                        if _rebal_buy:
                            st.markdown("**매수 예정**")
                            _buy_rows = []
                            for _rb in _rebal_buy:
                                _buy_rows.append({
                                    "ETF": _rb.get("ETF", ""),
                                    "현재수량(주)": _rb.get("현재수량(주)", 0),
                                    "합산 목표(주)": _rb.get("합산 목표(주)", 0),
                                    "매수수량(주)": _rb.get("매수 예정(주)", 0),
                                    "현재 비중(%)": _rb.get("현재 비중(%)", 0),
                                    "목표 비중(%)": _rb.get("목표 비중(%)", 0),
                                })
                            st.dataframe(pd.DataFrame(_buy_rows), use_container_width=True, hide_index=True)

                        # 일괄 예약 등록
                        st.markdown("---")
                        _rbc1, _rbc2 = st.columns(2)
                        with _rbc1:
                            _rb_method = st.selectbox(
                                "주문 방식",
                                ["동시호가 (장마감)", "시간외 종가", "시장가"],
                                key="pen_rebal_bulk_method",
                            )
                        with _rbc2:
                            _rb_exec_date = st.date_input("실행 예정일", value=_rb_dt, key="pen_rebal_bulk_date")

                        if _rb_method == "동시호가 (장마감)":
                            _rb_time = pd.Timestamp("15:20").time()
                            st.caption("실행 시각: 15:20 (동시호가 고정)")
                        elif _rb_method == "시간외 종가":
                            _rb_time = pd.Timestamp("15:40").time()
                            st.caption("실행 시각: 15:40 (시간외 종가 고정)")
                        else:
                            _rb_time = pd.Timestamp("15:10").time()

                        if st.button("일괄 예약 등록", key="pen_rebal_bulk_register", type="primary"):
                            _rb_sched = f"{_rb_exec_date.strftime('%Y-%m-%d')} {_rb_time.strftime('%H:%M')}"
                            _rb_count = 0
                            # 매도 먼저 등록
                            for _rs in _rebal_sell:
                                _etf_code = str(_rs.get("ETF코드", "")).strip()
                                _qty = int(_rs.get("매도 예정(주)", 0))
                                if _etf_code and _qty > 0:
                                    _add_pen_order(
                                        etf_code=_etf_code, side="매도", qty=_qty,
                                        method=_rb_method, price=0,
                                        scheduled_kst=_rb_sched, note="리밸런싱",
                                    )
                                    _rb_count += 1
                            # 매수
                            for _rb in _rebal_buy:
                                _etf_code = str(_rb.get("ETF코드", "")).strip()
                                _qty = int(_rb.get("매수 예정(주)", 0))
                                if _etf_code and _qty > 0:
                                    _add_pen_order(
                                        etf_code=_etf_code, side="매수", qty=_qty,
                                        method=_rb_method, price=0,
                                        scheduled_kst=_rb_sched, note="리밸런싱",
                                    )
                                    _rb_count += 1
                            if _rb_count > 0:
                                st.success(f"리밸런싱 주문 {_rb_count}건 예약 등록 완료")
                                st.rerun()
                            else:
                                st.warning("등록할 주문이 없습니다. (ETF 코드 매핑 확인 필요)")

            with st.expander("새 예약 주문 등록", expanded=False):
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    rsv_etf_label = st.selectbox("ETF 종목", list(etf_options.keys()), key="pen_rsv_etf")
                    rsv_etf_code = etf_options[rsv_etf_label]
                with rc2:
                    rsv_side = st.selectbox("매매 방향", ["매수", "매도"], key="pen_rsv_side")
                with rc3:
                    rsv_method = st.selectbox("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"], key="pen_rsv_method")

                rc4, rc5, rc6 = st.columns(3)
                with rc4:
                    rsv_qty = st.number_input("주문 수량 (주)", min_value=1, value=1, step=1, key="pen_rsv_qty")
                with rc5:
                    rsv_price = st.number_input(
                        "지정가 (원, 시장가=0)",
                        min_value=0,
                        value=int(_pen_cur) if rsv_method == "지정가" and _pen_cur > 0 else 0,
                        step=50,
                        key="pen_rsv_price",
                    )
                with rc6:
                    rsv_date = st.date_input("실행 예정일", key="pen_rsv_date")
                    # 주문 방식에 따라 시간 자동 결정
                    if rsv_method == "동시호가 (장마감)":
                        rsv_time = pd.Timestamp("15:20").time()
                        st.markdown("**실행 예정 시각**")
                        st.info("15:20 (동시호가 고정)")
                    elif rsv_method == "시간외 종가":
                        rsv_time = pd.Timestamp("15:40").time()
                        st.markdown("**실행 예정 시각**")
                        st.info("15:40 (시간외 종가 고정)")
                    else:
                        rsv_time = st.time_input("실행 예정 시각", value=pd.Timestamp("15:10").time(), key="pen_rsv_time")

                rsv_note = st.text_input("메모 (선택)", key="pen_rsv_note", placeholder="예: 채권 리밸런싱")

                _rsv_unit = rsv_price if rsv_price > 0 else (_pen_cur if _pen_cur > 0 else 0)
                _rsv_total = int(rsv_qty * _rsv_unit) if _rsv_unit > 0 else 0
                st.markdown(
                    f"<div style='background:#fffde7;border:1px solid #fff9c4;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                    f"<b>종목:</b> {_fmt_etf_code_name(rsv_etf_code)} &nbsp;|&nbsp; "
                    f"<b>방향:</b> {rsv_side} &nbsp;|&nbsp; "
                    f"<b>수량:</b> {rsv_qty}주 &nbsp;|&nbsp; "
                    f"<b>예상 금액:</b> <span style='font-weight:bold'>{_rsv_total:,}원</span> &nbsp;|&nbsp; "
                    f"<b>방식:</b> {rsv_method}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if st.button("예약 등록", key="pen_rsv_add", type="primary"):
                    sched_str = f"{rsv_date.strftime('%Y-%m-%d')} {rsv_time.strftime('%H:%M')}"
                    new_order = _add_pen_order(
                        etf_code=rsv_etf_code,
                        side=rsv_side,
                        qty=rsv_qty,
                        method=rsv_method,
                        price=rsv_price,
                        scheduled_kst=sched_str,
                        note=rsv_note,
                    )
                    st.success(f"예약 등록 완료: {new_order['id']} — {_fmt_etf_code_name(rsv_etf_code)} {rsv_side} {rsv_qty}주 @ {sched_str}")
                    st.rerun()

            # ── 예약 주문 내역 표시 ──
            pen_orders = _load_pen_orders()
            if pen_orders:
                # 대기 → 완료 → 실패 → 취소 순 정렬, 최신 먼저
                _status_priority = {"대기": 0, "완료": 1, "실패": 2, "취소": 3}
                pen_orders_sorted = sorted(
                    pen_orders,
                    key=lambda o: (_status_priority.get(o.get("status", ""), 9), -(o.get("created_at", "") or "").__hash__()),
                )

                pending_orders = [o for o in pen_orders_sorted if o.get("status") == "대기"]
                done_orders = [o for o in pen_orders_sorted if o.get("status") != "대기"]

                if pending_orders:
                    st.markdown(f"**대기 중 ({len(pending_orders)}건)**")
                    for o in pending_orders:
                        oid = o.get("id", "")
                        label = f"{o.get('side','')} {_fmt_etf_code_name(o.get('etf_code',''))} {o.get('qty',0)}주 | {o.get('method','')} | 예정: {o.get('scheduled_kst','')}"
                        if o.get("note"):
                            label += f" | {o['note']}"
                        _pc1, _pc2 = st.columns([6, 1])
                        _pc1.markdown(f"🟡 {label}")
                        if _pc2.button("취소", key=f"pen_rsv_cancel_{oid}"):
                            _update_pen_order_status(oid, "취소")
                            st.rerun()

                # ── 전체 이력 테이블 ──
                with st.expander(f"주문 이력 ({len(pen_orders)}건)", expanded=bool(done_orders)):
                    _rows = []
                    for o in reversed(pen_orders):
                        _status_icon = {"대기": "🟡", "완료": "🟢", "실패": "🔴", "취소": "⚪"}.get(o.get("status", ""), "❓")
                        _rows.append({
                            "상태": f"{_status_icon} {o.get('status', '')}",
                            "방향": o.get("side", ""),
                            "종목": _fmt_etf_code_name(o.get("etf_code", "")),
                            "수량": o.get("qty", 0),
                            "방식": o.get("method", ""),
                            "지정가": f"{o.get('price', 0):,}" if o.get("price", 0) > 0 else "-",
                            "예정일시": o.get("scheduled_kst", ""),
                            "등록일시": o.get("created_at", ""),
                            "실행일시": o.get("executed_at", "") or "-",
                            "결과": (o.get("result", "") or "-")[:60],
                            "메모": o.get("note", ""),
                        })
                    st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            else:
                st.info("등록된 예약 주문이 없습니다.")

    # ══════════════════════════════════════════════════════════════
    # Tab 4: 거래내역
    # ══════════════════════════════════════════════════════════════
    with tab_p4:
        st.header("거래내역 조회")
        st.caption("연금저축 계좌의 주문/체결 내역을 조회합니다.")

        _hist_cache_key = f"pen_trade_history_cache_{kis_acct}_{kis_prdt}"
        _dep_cache_key = f"pen_deposit_history_cache_{kis_acct}_{kis_prdt}"

        _default_end = pd.Timestamp.now().date()
        _default_start = (pd.Timestamp.now() - pd.Timedelta(days=30)).date()

        h1, h2, h3, h4 = st.columns([1.2, 1.2, 1.0, 1.0])
        with h1:
            hist_start = st.date_input("조회 시작일", value=_default_start, key="pen_hist_start")
        with h2:
            hist_end = st.date_input("조회 종료일", value=_default_end, key="pen_hist_end")
        with h3:
            hist_side_label = st.selectbox("매매구분", ["전체", "매수", "매도"], key="pen_hist_side")
        with h4:
            hist_ccld_label = st.selectbox("체결구분", ["전체", "체결", "미체결"], key="pen_hist_ccld")

        h5, h6, h7 = st.columns([1.1, 1.1, 1.2])
        with h5:
            hist_stock_code = _code_only(st.text_input("종목코드(선택)", value="", key="pen_hist_code"))
        with h6:
            hist_order_no = st.text_input("주문번호(선택)", value="", key="pen_hist_order_no").strip()
        with h7:
            hist_max_rows = int(
                st.selectbox("최대 조회 건수", [50, 100, 200, 500], index=2, key="pen_hist_max_rows")
            )

        st.divider()
        st.subheader("입출금/정산 내역")
        st.caption("KIS 퇴직연금 예수금조회 기준으로 입출금 관련 금액과 정산 금액을 표시합니다.")

        def _pick_dep_num(_data: dict, _keys: list[str]) -> float:
            if not isinstance(_data, dict):
                return 0.0
            for _k in _keys:
                if _k in _data:
                    return float(_safe_float(_data.get(_k), 0.0))
            return 0.0

        def _dep_core_sum(_data: dict) -> float:
            if not isinstance(_data, dict):
                return 0.0
            return float(
                _pick_dep_num(_data, ["dnca_tota", "dnca_tot_amt"])
                + _pick_dep_num(_data, ["nxdy_excc_amt"])
                + _pick_dep_num(_data, ["nxdy_sttl_amt"])
                + _pick_dep_num(_data, ["nx2_day_sttl_amt"])
            )

        def _fetch_best_pension_deposit() -> dict:
            _codes = ["00", "01", "02", "03", "04", "05", "99"]
            _best_res = None
            _best_sum = -1.0
            _trace = []
            for _cd in _codes:
                _res = trader.get_pension_deposit_info(acca_dvsn_cd=_cd)
                _ok = bool(_res.get("success")) if isinstance(_res, dict) else False
                _msg = str(_res.get("msg", "") or "") if isinstance(_res, dict) else ""
                _data = _res.get("data", {}) if isinstance(_res, dict) else {}
                _sum = _dep_core_sum(_data) if _ok else -1.0
                _trace.append(f"{_cd}:{'ok' if _ok else 'fail'}:{_sum:,.0f}")
                if _ok and _sum > _best_sum:
                    _best_sum = _sum
                    _best_res = {
                        "success": True,
                        "msg": _msg or "ok",
                        "data": _data if isinstance(_data, dict) else {},
                        "source": str(_res.get("source", "") or ""),
                        "acca_dvsn_cd": _cd,
                    }
            if _best_res is None:
                return {
                    "success": False,
                    "msg": "입출금 조회 실패",
                    "data": {},
                    "all_zero": True,
                    "trace": " / ".join(_trace),
                }
            _best_res["all_zero"] = bool(_best_sum <= 0.0)
            _best_res["trace"] = " / ".join(_trace)
            return _best_res

        if st.button("입출금 내역 조회", key="pen_dep_fetch_btn"):
            with st.spinner("입출금/정산 내역을 조회하는 중..."):
                _dep_result = _fetch_best_pension_deposit()
            st.session_state[_dep_cache_key] = _dep_result

        _dep_res = st.session_state.get(_dep_cache_key)
        if _dep_res:
            _dep_ok = bool(_dep_res.get("success")) if isinstance(_dep_res, dict) else False
            _dep_msg = str(_dep_res.get("msg", "") or "") if isinstance(_dep_res, dict) else ""
            _dep_data = _dep_res.get("data", {}) if isinstance(_dep_res, dict) else {}
            _dep_all_zero = bool(_dep_res.get("all_zero")) if isinstance(_dep_res, dict) else False
            _dep_trace = str(_dep_res.get("trace", "") or "") if isinstance(_dep_res, dict) else ""
            _dep_sel_cd = str(_dep_res.get("acca_dvsn_cd", "") or "") if isinstance(_dep_res, dict) else ""

            if not _dep_ok:
                st.error(f"입출금 내역 조회 실패: {_dep_msg or '응답 없음'}")
            else:
                _dep_rows = [
                    {"항목": "예수금총액", "금액(원)": _pick_dep_num(_dep_data, ["dnca_tota", "dnca_tot_amt"])},
                    {"항목": "익일정산액", "금액(원)": _pick_dep_num(_dep_data, ["nxdy_excc_amt"])},
                    {"항목": "익일결제금액", "금액(원)": _pick_dep_num(_dep_data, ["nxdy_sttl_amt"])},
                    {"항목": "2익일결제금액", "금액(원)": _pick_dep_num(_dep_data, ["nx2_day_sttl_amt"])},
                ]

                _in_amt = _pick_dep_num(_dep_data, ["in_amt", "dpst_amt", "depo_amt", "in_acmt_amt"])
                _out_amt = _pick_dep_num(_dep_data, ["out_amt", "wdrw_amt", "drwl_amt", "out_acmt_amt"])
                if _in_amt != 0:
                    _dep_rows.append({"항목": "입금금액", "금액(원)": _in_amt})
                if _out_amt != 0:
                    _dep_rows.append({"항목": "출금금액", "금액(원)": _out_amt})

                if _dep_all_zero:
                    _bal_for_dep = st.session_state.get(pen_bal_key)
                    if not isinstance(_bal_for_dep, dict) or bool(_bal_for_dep.get("error")):
                        _bal_for_dep = trader.get_balance()
                        if isinstance(_bal_for_dep, dict) and not bool(_bal_for_dep.get("error")):
                            st.session_state[pen_bal_key] = _bal_for_dep
                            st.session_state[pen_bal_ts_key] = float(time.time())
                    _bal_sum_for_dep = _compute_kis_balance_summary(_bal_for_dep if isinstance(_bal_for_dep, dict) else {})
                    _fallback_cash = float(
                        _bal_sum_for_dep.get("buyable_cash", 0.0) or _bal_sum_for_dep.get("cash", 0.0) or 0.0
                    )
                    if _fallback_cash > 0:
                        _dep_rows[0] = {"항목": "예수금총액(잔고기준)", "금액(원)": _fallback_cash}
                        st.info("입출금 API 응답이 0으로 내려와 잔고 조회 기준 예수금으로 보정 표시했습니다.")
                    else:
                        st.warning("입출금 API 응답이 0으로 내려왔습니다. 계좌 유형/거래시간에 따라 0으로 반환될 수 있습니다.")

                _dep_df = pd.DataFrame(_dep_rows)
                st.dataframe(_dep_df, use_container_width=True, hide_index=True)
                if _dep_sel_cd:
                    st.caption(f"입출금 조회 구분코드: {_dep_sel_cd}")
                if _dep_trace:
                    st.caption(f"조회 시도: {_dep_trace}")
                with st.expander("입출금 원본 응답 보기"):
                    st.dataframe(pd.DataFrame([_dep_data]), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("주문/체결 내역")

        _side_map = {"전체": "00", "매도": "01", "매수": "02"}
        _ccld_map = {"전체": "00", "체결": "01", "미체결": "02"}

        if st.button("거래내역 조회", key="pen_hist_fetch_btn", type="primary"):
            _start_ymd = pd.Timestamp(hist_start).strftime("%Y%m%d")
            _end_ymd = pd.Timestamp(hist_end).strftime("%Y%m%d")
            with st.spinner("연금저축 거래내역을 조회하는 중..."):
                _hist_result = trader.get_pension_trade_history(
                    start_date=_start_ymd,
                    end_date=_end_ymd,
                    side=_side_map.get(hist_side_label, "00"),
                    ccld=_ccld_map.get(hist_ccld_label, "00"),
                    stock_code=hist_stock_code,
                    order_no=hist_order_no,
                    max_rows=hist_max_rows,
                )
            st.session_state[_hist_cache_key] = {
                "start": _start_ymd,
                "end": _end_ymd,
                "result": _hist_result,
            }

        _hist_pack = st.session_state.get(_hist_cache_key)
        if not _hist_pack:
            st.info("조회 조건을 설정한 뒤 `거래내역 조회` 버튼을 눌러주세요.")
        else:
            _hist_res = _hist_pack.get("result", {}) if isinstance(_hist_pack, dict) else {}
            _hist_rows = _hist_res.get("rows", []) if isinstance(_hist_res, dict) else []
            _hist_ok = bool(_hist_res.get("success")) if isinstance(_hist_res, dict) else False
            _hist_msg = str(_hist_res.get("msg", "") or "") if isinstance(_hist_res, dict) else ""
            _hist_source = str(_hist_res.get("source", "") or "") if isinstance(_hist_res, dict) else ""
            _fallback_msg = str(_hist_res.get("fallback_msg", "") or "") if isinstance(_hist_res, dict) else ""

            if _hist_ok:
                st.caption(
                    f"조회기간: {_hist_pack.get('start', '')} ~ {_hist_pack.get('end', '')}"
                    + (f" | 소스: {_hist_source}" if _hist_source else "")
                )
                if _fallback_msg:
                    st.caption(f"보조 조회 사용: {_fallback_msg}")
            else:
                st.error(f"거래내역 조회 실패: {_hist_msg or '응답 없음'}")

            if not _hist_rows:
                st.info("조회된 거래내역이 없습니다.")
            else:
                def _pick_first(_row: dict, _keys: list[str], _default=""):
                    for _k in _keys:
                        if _k in _row:
                            _v = _row.get(_k)
                            if _v not in ("", None):
                                return _v
                    return _default

                def _fmt_date(_v) -> str:
                    _s = "".join(ch for ch in str(_v or "") if ch.isdigit())
                    if len(_s) == 8:
                        return f"{_s[0:4]}-{_s[4:6]}-{_s[6:8]}"
                    return str(_v or "")

                def _fmt_time(_v) -> str:
                    _s = "".join(ch for ch in str(_v or "") if ch.isdigit())
                    if len(_s) >= 6:
                        _s = _s[:6]
                        return f"{_s[0:2]}:{_s[2:4]}:{_s[4:6]}"
                    return str(_v or "")

                _view_rows = []
                for _r in _hist_rows:
                    if not isinstance(_r, dict):
                        continue

                    _code = str(_pick_first(_r, ["pdno", "mksc_shrn_iscd", "shtn_pdno"], "")).strip()
                    _name = str(_pick_first(_r, ["prdt_name", "pd_name", "item_name", "prdt_abrv_name"], "")).strip()
                    _side_cd = str(_pick_first(_r, ["sll_buy_dvsn_cd", "sll_buy_dvsn", "trde_dvsn_cd"], "")).strip()
                    _side_nm = str(_pick_first(_r, ["sll_buy_dvsn_name", "sll_buy_dvsn_cd_name"], "")).strip()
                    _status_cd = str(_pick_first(_r, ["ccld_dvsn", "ccld_nccs_dvsn"], "")).strip()

                    _order_qty = _safe_int(_pick_first(_r, ["ord_qty", "tot_ord_qty", "order_qty"], 0), 0)
                    _filled_qty = _safe_int(_pick_first(_r, ["tot_ccld_qty", "ccld_qty", "exec_qty"], 0), 0)
                    _remain_qty = _safe_int(_pick_first(_r, ["rmn_qty", "nccs_qty", "ord_rmn_qty"], 0), 0)
                    _order_px = _safe_float(_pick_first(_r, ["ord_unpr", "order_price"], 0.0), 0.0)
                    _filled_px = _safe_float(_pick_first(_r, ["avg_prvs", "avg_ccld_unpr", "ccld_unpr"], 0.0), 0.0)
                    _filled_amt = _safe_float(_pick_first(_r, ["tot_ccld_amt", "ccld_amt", "exec_amt"], 0.0), 0.0)

                    _trade_date = _fmt_date(_pick_first(_r, ["ord_dt", "ord_dd", "trde_dt", "ccld_dt"], ""))
                    _trade_time = _fmt_time(_pick_first(_r, ["ord_tmd", "ord_tm", "trde_tmd", "ccld_tmd"], ""))
                    _trade_dt = f"{_trade_date} {_trade_time}".strip()

                    if _side_cd in ("01", "1"):
                        _side = "매도"
                    elif _side_cd in ("02", "2"):
                        _side = "매수"
                    else:
                        _side = _side_nm or "-"

                    if _filled_qty > 0 and _remain_qty > 0:
                        _status = "부분체결"
                    elif _filled_qty > 0:
                        _status = "체결"
                    elif _status_cd in ("02",):
                        _status = "미체결"
                    elif _status_cd in ("01",):
                        _status = "체결"
                    else:
                        _status = "미체결" if _remain_qty > 0 else "-"

                    _view_rows.append(
                        {
                            "거래일시": _trade_dt,
                            "종목": _fmt_etf_code_name(_code) if _code else (_name or "-"),
                            "매매": _side,
                            "상태": _status,
                            "주문수량(주)": int(_order_qty),
                            "체결수량(주)": int(_filled_qty),
                            "미체결수량(주)": int(_remain_qty),
                            "주문가격(원)": float(_order_px),
                            "체결평균가(원)": float(_filled_px),
                            "체결금액(원)": float(_filled_amt),
                            "주문번호": str(_pick_first(_r, ["odno", "ord_no"], "")),
                        }
                    )

                _hist_df = pd.DataFrame(_view_rows)
                if _hist_df.empty:
                    st.info("표시할 거래내역이 없습니다.")
                else:
                    _filled_notional = float(_hist_df["체결금액(원)"].sum())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("조회건수", f"{len(_hist_df):,}건")
                    m2.metric("체결금액 합계", f"{_filled_notional:,.0f}원")
                    m3.metric("체결건수", f"{int((_hist_df['상태'] != '미체결').sum()):,}건")

                    st.dataframe(_hist_df, use_container_width=True, hide_index=True)

                with st.expander("원본 응답 보기"):
                    st.dataframe(pd.DataFrame(_hist_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # Tab 5: 전략 가이드
    # ══════════════════════════════════════════════════════════════
    with tab_p5:
        st.header("연금저축 전략 가이드")
        st.caption("전략별 하위탭에서 원하는 전략 설명을 바로 확인할 수 있습니다.")
        guide_tab_laa, guide_tab_dm, guide_tab_vaa, guide_tab_cdm = st.tabs(
            ["LAA", "듀얼모멘텀", "VAA", "CDM"]
        )

        with guide_tab_laa:
            st.subheader("LAA (Lethargic Asset Allocation) 전략")
            st.markdown("""
**LAA**는 Keller & Keuning이 제안한 게으른 자산배분 전략입니다.

- **코어 자산 (75%)**: IWD(미국 가치주), GLD(금), IEF(미국 중기채) 각 25%
- **리스크 자산 (25%)**: SPY가 200일 이동평균선 위 → QQQ, 아래 → SHY(단기채)
- **리밸런싱**: 월 1회 (월말 기준)
""")

            st.subheader("의사결정 흐름")
            st.markdown("""
```
매월 말 기준:
  1. SPY 종가 vs SPY 200일 이동평균
  2. SPY > 200일선 → 리스크 자산 = QQQ (공격)
     SPY < 200일선 → 리스크 자산 = SHY (방어)
  3. 코어 3종목 25%씩 + 리스크 자산 25% 배분
  4. 목표 비중 대비 괴리 > 3%p이면 리밸런싱 실행
```
""")

            st.subheader("국내 ETF 매핑")
            st.dataframe(pd.DataFrame([
                {"미국 티커": "IWD", "국내 ETF": "TIGER 미국S&P500 (360750)", "역할": "코어 - 미국 가치주"},
                {"미국 티커": "GLD", "국내 ETF": "KODEX Gold선물(H) (132030)", "역할": "코어 - 금"},
                {"미국 티커": "IEF", "국내 ETF": "TIGER 미국채10년선물 (453540)", "역할": "코어 - 중기채"},
                {"미국 티커": "QQQ", "국내 ETF": "TIGER 미국나스닥100 (133690)", "역할": "리스크 공격"},
                {"미국 티커": "SHY", "국내 ETF": "KODEX 국고채3년 (114470)", "역할": "리스크 방어"},
            ]), use_container_width=True, hide_index=True)
            st.caption("연금저축 계좌에서 해외 ETF 직접 매매 불가 → 국내 ETF로 대체 실행")

        with guide_tab_dm:
            _guide_dm_map = (_dm_settings.get("kr_etf_map", {}) if isinstance(_dm_settings, dict) else {}) or {}
            _guide_dm_spy = str(_guide_dm_map.get("SPY", _code_only(_pen_cfg.get("pen_dm_kr_spy", "360750"))))
            _guide_dm_efa = str(_guide_dm_map.get("EFA", _code_only(_pen_cfg.get("pen_dm_kr_efa", "453850"))))
            _guide_dm_agg = str(_guide_dm_map.get("AGG", _code_only(_pen_cfg.get("pen_dm_kr_agg", "453540"))))
            _guide_dm_bil = str(_guide_dm_map.get("BIL", _code_only(_pen_cfg.get("pen_dm_kr_bil", "114470"))))
            _guide_dm_w = (_dm_settings.get("momentum_weights", {}) if isinstance(_dm_settings, dict) else {}) or {}
            _guide_dm_w1 = float(_guide_dm_w.get("m1", _pen_cfg.get("pen_dm_w1", 12.0)))
            _guide_dm_w3 = float(_guide_dm_w.get("m3", _pen_cfg.get("pen_dm_w3", 4.0)))
            _guide_dm_w6 = float(_guide_dm_w.get("m6", _pen_cfg.get("pen_dm_w6", 2.0)))
            _guide_dm_w12 = float(_guide_dm_w.get("m12", _pen_cfg.get("pen_dm_w12", 1.0)))
            _guide_dm_lb = int((_dm_settings.get("lookback", _pen_cfg.get("pen_dm_lookback", 12)) if isinstance(_dm_settings, dict) else _pen_cfg.get("pen_dm_lookback", 12)))
            _guide_dm_td = int((_dm_settings.get("trading_days_per_month", _pen_cfg.get("pen_dm_trading_days", 22)) if isinstance(_dm_settings, dict) else _pen_cfg.get("pen_dm_trading_days", 22)))

            st.subheader("듀얼모멘텀 (GEM) 전략")
            st.markdown(f"""
**듀얼모멘텀(GEM)**은 상대모멘텀 + 절대모멘텀을 결합한 월간 리밸런싱 전략입니다.

- **공격 자산(2개)**: SPY, EFA 중 모멘텀 점수 상위 1개 선택
- **방어 자산(1개)**: AGG
- **카나리아(1개)**: BIL
- **리밸런싱**: 월 1회 (월말 기준)

모멘텀 점수식:
`((1개월수익률 × {_guide_dm_w1:g}) + (3개월수익률 × {_guide_dm_w3:g}) + (6개월수익률 × {_guide_dm_w6:g}) + (12개월수익률 × {_guide_dm_w12:g})) ÷ 4`

절대모멘텀 기준:
`카나리아 룩백 {_guide_dm_lb}개월 수익률` (월 환산 거래일 `{_guide_dm_td}`일 기준)
""")

            st.subheader("의사결정 흐름")
            st.markdown(f"""
```
매월 말 기준:
  1. 공격 자산(SPY, EFA)의 가중 모멘텀 점수 계산
  2. 카나리아(BIL) 룩백 {_guide_dm_lb}개월 수익률 계산
  3. 공격 1위 점수 > 카나리아 수익률  → 공격 1위 100%
     공격 1위 점수 <= 카나리아 수익률 → 방어(AGG) 100%
  4. 목표 비중 대비 괴리 발생 시 리밸런싱 실행
```
""")

            st.subheader("국내 ETF 매핑")
            st.dataframe(pd.DataFrame([
                {"전략 키": "SPY", "국내 ETF": _fmt_etf_code_name(_guide_dm_spy), "역할": "공격 자산 1"},
                {"전략 키": "EFA", "국내 ETF": _fmt_etf_code_name(_guide_dm_efa), "역할": "공격 자산 2"},
                {"전략 키": "AGG", "국내 ETF": _fmt_etf_code_name(_guide_dm_agg), "역할": "방어 자산"},
                {"전략 키": "BIL", "국내 ETF": _fmt_etf_code_name(_guide_dm_bil), "역할": "카나리아"},
            ]), use_container_width=True, hide_index=True)
            st.caption("듀얼모멘텀도 연금저축 계좌에서 국내 ETF로 시그널/실매매를 수행합니다.")

            st.subheader("운용 특성")
            st.markdown("""
- **시장 대응**: 상승장에서는 공격 자산, 하락장/둔화장에서는 방어 자산으로 자동 전환
- **리스크 관리**: 절대모멘텀(카나리아) 기준으로 하락 추세 회피
- **운용 빈도**: 월 1회 리밸런싱으로 과도한 매매 방지
- **과세이연**: 연금저축 계좌 내 매매차익 과세이연 효과로 복리 운용에 유리
""")

        with guide_tab_vaa:
            st.subheader("VAA (Vigilant Asset Allocation) 전략")
            st.markdown("""
**VAA**는 Wouter Keller가 제안한 경계적 자산배분 전략으로, **13612W 모멘텀 스코어**를 활용하여
공격/방어 자산을 동적으로 전환합니다.

- **공격 자산 (4개)**: SPY(미국), EFA(선진국), EEM(이머징), AGG(채권)
- **방어 자산 (3개)**: LQD(회사채), IEF(중기채), SHY(단기채)
- **선택 규칙**: 공격 자산 중 모멘텀 양수인 것 최고 1개 선택 → 전부 음수 시 방어 자산 최고 1개
- **리밸런싱**: 월 1회 (월말 기준)

**13612W 모멘텀 스코어 계산식:**
`(1개월수익률 × 12 + 3개월수익률 × 4 + 6개월수익률 × 2 + 12개월수익률 × 1) ÷ 19`

단기(1개월)에 높은 가중치를 부여하여 추세 반전에 빠르게 대응합니다.
""")

            st.subheader("의사결정 흐름")
            st.markdown("""
```
매월 말 기준:
  1. 공격 자산 4개(SPY, EFA, EEM, AGG)의 13612W 모멘텀 스코어 계산
  2. 방어 자산 3개(LQD, IEF, SHY)의 13612W 모멘텀 스코어 계산
  3. 공격 자산 중 모멘텀 > 0 인 것이 있으면:
     → 양수 모멘텀 중 최고 스코어 1개에 100% 투자 (공격 모드)
  4. 모든 공격 자산 모멘텀 ≤ 0 이면:
     → 방어 자산 중 최고 스코어 1개에 100% 투자 (방어 모드)
  5. 목표 대비 괴리 발생 시 리밸런싱 실행
```
""")

            st.subheader("국내 ETF 매핑")
            _guide_vaa_map = (_vaa_settings.get("kr_etf_map", {}) if isinstance(_vaa_settings, dict) else {}) or {}
            st.dataframe(pd.DataFrame([
                {"전략 키": "SPY", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("SPY", "379800"))), "유형": "공격", "역할": "미국 주식"},
                {"전략 키": "EFA", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("EFA", "195930"))), "유형": "공격", "역할": "선진국 주식"},
                {"전략 키": "EEM", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("EEM", "295820"))), "유형": "공격", "역할": "이머징 주식"},
                {"전략 키": "AGG", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("AGG", "305080"))), "유형": "공격", "역할": "미국 채권"},
                {"전략 키": "LQD", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("LQD", "329750"))), "유형": "방어", "역할": "회사채"},
                {"전략 키": "IEF", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("IEF", "305080"))), "유형": "방어", "역할": "중기채"},
                {"전략 키": "SHY", "국내 ETF": _fmt_etf_code_name(str(_guide_vaa_map.get("SHY", "329750"))), "유형": "방어", "역할": "단기채"},
            ]), use_container_width=True, hide_index=True)

        with guide_tab_cdm:
            st.subheader("CDM (Composite Dual Momentum) 전략")
            st.markdown("""
**CDM**은 4개 모듈로 구성된 복합 듀얼모멘텀 전략입니다.
공격자산 8개를 2개씩 4모듈로 나누고, 각 모듈에서 **상대모멘텀**(승자 선택)과
**절대모멘텀**(방어 전환 여부)을 동시에 적용합니다.

- **공격 자산 (8개, 4모듈 × 2자산)**:
  - Module 1: SPY(미국) vs VEU(해외)
  - Module 2: VNQ(부동산) vs REM(리츠)
  - Module 3: LQD(회사채) vs HYG(하이일드)
  - Module 4: TLT(장기채) vs GLD(금)
- **방어 자산**: BIL(초단기채)
- **각 모듈 비중**: 25% (총 100%)
- **리밸런싱**: 월 1회 (월말 기준)
""")

            st.subheader("의사결정 흐름")
            st.markdown("""
```
매월 말 기준 (각 모듈 독립 처리):
  Module 1: SPY vs VEU
    1. SPY, VEU 12개월 수익률 비교 → 승자(상대모멘텀) 선택
    2. 승자의 12개월 수익률 > BIL 12개월 수익률?
       → YES: 승자에 25% 투자 (공격)
       → NO:  BIL에 25% 투자 (방어)

  Module 2~4도 동일한 규칙으로 각각 25% 배분

최종 포트폴리오: 4모듈 합산 = 100%
(공격 모듈이 많을수록 공격적, 적을수록 방어적)
```
""")

            st.subheader("국내 ETF 매핑")
            _guide_cdm_map = (_cdm_settings.get("kr_etf_map", {}) if isinstance(_cdm_settings, dict) else {}) or {}
            st.dataframe(pd.DataFrame([
                {"전략 키": "SPY", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("SPY", "379800"))), "모듈": "M1", "역할": "미국 주식"},
                {"전략 키": "VEU", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("VEU", "195930"))), "모듈": "M1", "역할": "해외 주식"},
                {"전략 키": "VNQ", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("VNQ", "352560"))), "모듈": "M2", "역할": "부동산"},
                {"전략 키": "REM", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("REM", "352560"))), "모듈": "M2", "역할": "리츠"},
                {"전략 키": "LQD", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("LQD", "305080"))), "모듈": "M3", "역할": "회사채"},
                {"전략 키": "HYG", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("HYG", "305080"))), "모듈": "M3", "역할": "하이일드"},
                {"전략 키": "TLT", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("TLT", "304660"))), "모듈": "M4", "역할": "장기채"},
                {"전략 키": "GLD", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("GLD", "132030"))), "모듈": "M4", "역할": "금"},
                {"전략 키": "BIL", "국내 ETF": _fmt_etf_code_name(str(_guide_cdm_map.get("BIL", "329750"))), "모듈": "-", "역할": "방어자산"},
            ]), use_container_width=True, hide_index=True)
            st.caption("연금저축 계좌에서 해외 ETF 직접 매매 불가 → 국내 ETF로 대체. 일부 종목은 국내 대안이 제한적이어서 동일 ETF를 중복 매핑합니다.")


    # ══════════════════════════════════════════════════════════════
    # Tab 6: 주문방식
    # ══════════════════════════════════════════════════════════════
    with tab_p6:
        st.header("KIS 국내 ETF 주문방식 안내")
        st.dataframe(pd.DataFrame([
            {"구분": "시장가", "API": 'ord_dvsn="01"', "설명": "즉시 체결 (최우선 호가)"},
            {"구분": "지정가", "API": 'ord_dvsn="00"', "설명": "원하는 가격에 주문"},
            {"구분": "동시호가 매수", "API": '상한가(+30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
            {"구분": "동시호가 매도", "API": '하한가(-30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
            {"구분": "시간외 종가", "API": 'ord_dvsn="06"', "설명": "15:40~16:00 당일 종가로 체결"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("호가단위")
        st.dataframe(pd.DataFrame([
            {"가격대": "~5,000원", "호가단위": "5원"},
            {"가격대": "5,000~10,000원", "호가단위": "10원"},
            {"가격대": "10,000~50,000원", "호가단위": "50원"},
            {"가격대": "50,000원~", "호가단위": "100원"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("자동매매 흐름 (GitHub Actions)")
        st.markdown("""
0. 로컬 PC에서 직접 주문하지 않고, GitHub Actions에서만 주문 실행
1. 매월 25~31일 평일 KST 15:20 실행 (`TRADING_MODE=kis_pension`)
2. 국내 ETF(SPY/IWD/GLD/IEF/QQQ/SHY 매핑) 일봉 조회
3. SPY vs 200일선 → 리스크 자산 결정 (QQQ or SHY)
4. 목표 배분 vs 현재 보유 비교 → 리밸런싱 필요 여부 판단
5. 매도 → `smart_sell_all_closing()` (동시호가+시간외)
6. 매수 → `smart_buy_krw_closing()` (동시호가+시간외)
""")

    # ══════════════════════════════════════════════════════════════
    # Tab 7: 수수료/세금
    # ══════════════════════════════════════════════════════════════
    with tab_p7:
        st.header("연금저축 수수료 및 세금 안내")

        st.subheader("1. 매매 수수료")
        st.dataframe(pd.DataFrame([
            {"증권사": "한국투자증권", "매매 수수료": "0.0140396%", "비고": "나무 온라인 (현재 사용)"},
            {"증권사": "키움증권", "매매 수수료": "0.015%", "비고": "영웅문 온라인"},
            {"증권사": "미래에셋", "매매 수수료": "0.014%", "비고": "m.Stock 온라인"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("2. 매매 대상 ETF 보수")
        st.dataframe(pd.DataFrame([
            {"ETF": "TIGER 미국S&P500", "코드": "360750", "총보수": "0.07%", "추종": "SPY 대용"},
            {"ETF": "KODEX Gold선물(H)", "코드": "132030", "총보수": "0.09%", "추종": "GLD 대용"},
            {"ETF": "TIGER 미국채10년선물", "코드": "453540", "총보수": "0.10%", "추종": "IEF 대용"},
            {"ETF": "TIGER 미국나스닥100", "코드": "133690", "총보수": "0.07%", "추종": "QQQ 대용"},
            {"ETF": "KODEX 국고채3년", "코드": "114470", "총보수": "0.05%", "추종": "SHY 대용"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("3. 연금저축 세제혜택")
        st.markdown("""
| 항목 | 내용 |
|------|------|
| 세액공제 | 연 최대 600만원 (IRP 합산 900만원) |
| 공제율 | 총급여 5,500만원 이하: 16.5% / 초과: 13.2% |
| 과세이연 | 매매차익·배당 세금 인출 시까지 이연 |
| 연금 수령 시 | 3.3~5.5% 연금소득세 (일반 15.4% 대비 유리) |
| 중도 인출 시 | 16.5% 기타소득세 (불이익) |
        """)
        st.caption("LAA 월간 리밸런싱 매매차익이 모두 과세이연되어 복리 효과 극대화 (일반 계좌 대비 연 1~2% 추가 수익)")

    with tab_p8:
        render_strategy_trigger_tab("PENSION")


