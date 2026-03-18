import streamlit as st
import pandas as pd
import time
import logging
from datetime import datetime

import src.engine.data_cache as data_cache
from src.utils.formatting import (
    _fmt_etf_code_name,
    _code_only,
)
from src.ui.components.triggers import render_strategy_trigger_tab
from src.ui.components.pension_orders import (
    execute_pending_pen_orders,
    render_pension_orders_tab,
)
from src.ui.components.pension_guide import (
    render_pension_guide_tab,
    render_pension_order_info_tab,
    render_pension_fee_tab,
)
from src.ui.components.pension_history import render_pension_history_tab
from src.ui.components.pension_manual import render_pension_manual_tab
from src.ui.components.pension_backtest import render_pension_backtest_tab
from src.ui.components.pension_sidebar import render_pension_sidebar
from src.ui.components.pension_monitoring import render_monitoring_tab
from src.ui.components.pension_asset_mgmt import render_pension_asset_mgmt_tab


def render_kis_pension_mode(config, save_config):
    """KIS 연금저축 포트폴리오 모드 - 다중 전략 지원."""
    from src.engine.kis_trader import KISTrader

    # ── 사이드바 설정 (별도 모듈) ──
    _sb = render_pension_sidebar(config, save_config)
    if _sb is None:
        return
    _pen_cfg = _sb["pen_cfg"]
    _pen_bt_start_raw = _sb["pen_bt_start_raw"]
    _pen_bt_start_ts = _sb["pen_bt_start_ts"]
    _pen_bt_cap = _sb["pen_bt_cap"]
    kis_ak = _sb["kis_ak"]
    kis_sk = _sb["kis_sk"]
    kis_acct = _sb["kis_acct"]
    kis_prdt = _sb["kis_prdt"]
    _pen_port_edited = _sb["pen_port_edited"]
    _active_strategies = _sb["active_strategies"]
    _auto_signal_strategies = _sb["auto_signal_strategies"]
    kr_spy = _sb["kr_spy"]
    kr_iwd = _sb["kr_iwd"]
    kr_gld = _sb["kr_gld"]
    kr_ief = _sb["kr_ief"]
    kr_qqq = _sb["kr_qqq"]
    kr_shy = _sb["kr_shy"]
    _kr_etf_map = _sb["kr_etf_map"]
    _dm_settings = _sb["dm_settings"]
    _vaa_settings = _sb["vaa_settings"]

    # ── 트레이더 초기화 ──
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
        _exec_results = execute_pending_pen_orders(trader)
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
        _legacy_map = {"453540": "305080", "453850": "251350", "295820": "195980", "114470": "329750"}

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

    tab_p1, tab_p2, tab_p3, tab_p_orders, tab_p4, tab_p_am, tab_p5, tab_p6, tab_p7, tab_p8 = st.tabs([
        "🚀 실시간 모니터링",
        "🧪 백테스트",
        "🛒 수동 주문",
        "📋 예약 주문",
        "📜 거래내역",
        "💰 자산관리",
        "📖 전략 가이드",
        "📋 주문방식",
        "💳 수수료/세금",
        "⏰ 트리거",
    ])

    # ══════════════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링
    # ══════════════════════════════════════════════════════════════
    with tab_p1:
        render_monitoring_tab(
            pen_port_edited=_pen_port_edited,
            active_strategies=_active_strategies,
            auto_signal_strategies=_auto_signal_strategies,
            pen_bal_key=pen_bal_key,
            pen_bal_ts_key=pen_bal_ts_key,
            pen_bal_err_key=pen_bal_err_key,
            kis_acct=kis_acct,
            kis_prdt=kis_prdt,
            refresh_pension_balance=_refresh_pension_balance,
            trader=trader,
            kr_spy=kr_spy, kr_iwd=kr_iwd, kr_gld=kr_gld,
            kr_ief=kr_ief, kr_qqq=kr_qqq, kr_shy=kr_shy,
            kr_etf_map=_kr_etf_map,
            dm_settings=_dm_settings,
            vaa_settings=_vaa_settings,
            pen_bt_start_raw=_pen_bt_start_raw,
            pen_bt_cap=_pen_bt_cap,
            pen_bt_start_ts=_pen_bt_start_ts,
            pen_live_auto_backtest=_pen_live_auto_backtest,
            pen_api_fallback=_pen_api_fallback,
            get_daily_chart=_get_pen_daily_chart,
            get_current_price=_get_pen_current_price,
        )


    # ══════════════════════════════════════════════════════════════
    # Tab 2: 백테스트
    # ══════════════════════════════════════════════════════════════
    with tab_p2:
        render_pension_backtest_tab(
            active_strategies=_active_strategies,
            pen_bt_start_ts=_pen_bt_start_ts,
            pen_bt_cap=_pen_bt_cap,
            kr_etf_map=_kr_etf_map,
            kr_spy=kr_spy, kr_iwd=kr_iwd, kr_gld=kr_gld,
            kr_ief=kr_ief, kr_qqq=kr_qqq, kr_shy=kr_shy,
            dm_settings=_dm_settings,
            vaa_settings=_vaa_settings,
            get_pen_daily_chart=_get_pen_daily_chart,
            pen_local_first=_pen_local_first,
        )

    # ── 전략별 ETF 코드 통합 목록 (수동주문 + 예약주문 공용) ──
    _all_etf_codes = list(filter(None, [kr_iwd, kr_gld, kr_ief, kr_qqq, kr_shy]))
    if _dm_settings:
        _dm_map_t = _dm_settings.get("kr_etf_map", {}) or {}
        _all_etf_codes.extend([_dm_map_t.get("SPY", ""), _dm_map_t.get("EFA", ""), _dm_map_t.get("AGG", "")])
    if _vaa_settings:
        _vaa_map_t = _vaa_settings.get("kr_etf_map", {}) or {}
        _all_etf_codes.extend([str(v) for v in _vaa_map_t.values() if str(v).strip()])
    _all_etf_codes = list(dict.fromkeys(c.strip() for c in _all_etf_codes if str(c).strip()))
    if not _all_etf_codes:
        _all_etf_codes = ["360750"]

    # ══════════════════════════════════════════════════════════════
    # Tab 3: 수동 주문
    # ══════════════════════════════════════════════════════════════
    with tab_p3:
        render_pension_manual_tab(
            trader, pen_bal_key, _all_etf_codes,
            _resolve_pen_quote_code, _get_pen_current_price,
            _get_pen_daily_chart, _get_pen_orderbook,
        )

    # ══════════════════════════════════════════════════════════════
    # Tab: 예약 주문 (pension_orders.py)
    # ══════════════════════════════════════════════════════════════
    with tab_p_orders:
        _po_etf_options = {_fmt_etf_code_name(c): c for c in _all_etf_codes}
        render_pension_orders_tab(_po_etf_options, 0)

    # ══════════════════════════════════════════════════════════════
    # Tab 4: 거래내역
    # ══════════════════════════════════════════════════════════════
    with tab_p4:
        render_pension_history_tab(trader, kis_acct, kis_prdt, pen_bal_key, pen_bal_ts_key)

    # ══════════════════════════════════════════════════════════════
    # Tab: 자산관리
    # ══════════════════════════════════════════════════════════════
    with tab_p_am:
        render_pension_asset_mgmt_tab(
            pen_port_edited=_pen_port_edited,
            pen_bal_key=pen_bal_key,
        )

    # ══════════════════════════════════════════════════════════════
    # Tab 5-8: 가이드 / 주문방식 / 수수료 / 트리거
    # ══════════════════════════════════════════════════════════════
    with tab_p5:
        render_pension_guide_tab(_dm_settings, _vaa_settings, _pen_cfg)
    with tab_p6:
        render_pension_order_info_tab()
    with tab_p7:
        render_pension_fee_tab()
    with tab_p8:
        render_strategy_trigger_tab("PENSION")


