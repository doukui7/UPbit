import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.constants import *
from src.backtest.engine import BacktestEngine
from src.trading.upbit_trader import UpbitTrader
from src.strategy.sma import SMAStrategy
from src.strategy.donchian import DonchianStrategy
import src.engine.data_cache as data_cache
from src.ui.components.performance import render_performance_table
from src.ui.components.triggers import render_strategy_trigger_tab

_BALANCE_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "balance_cache.json")

def _load_balance_cache():
    """VM에서 저장한 잔고 캐시 파일 로드."""
    try:
        if os.path.exists(_BALANCE_CACHE_FILE):
            with open(_BALANCE_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def render_coin_mode(config, save_config):
# --- Coin Mode Logic ---
    st.title("🪙 업비트 자동매매 시스템")

    # Sticky Header (JS로 Streamlit DOM 직접 조작)
    import streamlit.components.v1 as components
    components.html("""
    <script>
        const doc = window.parent.document;
        if (!doc.getElementById('sticky-title-style')) {
            const style = doc.createElement('style');
            style.id = 'sticky-title-style';
            style.textContent = `
                section[data-testid="stMain"] > div.block-container {
                    overflow: visible !important;
                }
                #sticky-title-wrap {
                    position: sticky;
                    top: 0;
                    background: white;
                    z-index: 999;
                    padding-bottom: 6px;
                    border-bottom: 2px solid #e6e6e6;
                }
            `;
            doc.head.appendChild(style);
        }

        function applySticky() {
            if (doc.getElementById('sticky-title-wrap')) return;
            const titles = doc.querySelectorAll('h1');
            for (const h1 of titles) {
                if (h1.textContent.includes('Upbit SMA')) {
                    const wrapper = h1.closest('[data-testid="stVerticalBlockBorderWrapper"]')
                                  || h1.parentElement?.parentElement;
                    if (wrapper) {
                        wrapper.id = 'sticky-title-wrap';
                    }
                    break;
                }
            }
        }
        applySticky();
        setTimeout(applySticky, 500);
        setTimeout(applySticky, 1500);
    </script>
    """, height=0)

    # --- Sidebar: Configuration ---
    st.sidebar.header("설정")
    
    # API Keys (Streamlit Cloud secrets 또는 .env 지원)
    try:
        env_access = st.secrets["UPBIT_ACCESS_KEY"]
        env_secret = st.secrets["UPBIT_SECRET_KEY"]
    except Exception:
        env_access = os.getenv("UPBIT_ACCESS_KEY")
        env_secret = os.getenv("UPBIT_SECRET_KEY")
    
    if IS_CLOUD:
        # Cloud: secrets에서 자동 로드, 편집 불가
        current_ak = env_access
        current_sk = env_secret
        st.sidebar.info("📱 조회 전용 모드 (Cloud)")
    else:
        with st.sidebar.expander("API 키", expanded=False):
            ak_input = st.text_input("Access Key", value=env_access if env_access else "", type="password")
            sk_input = st.text_input("Secret Key", value=env_secret if env_secret else "", type="password")
            current_ak = ak_input if ak_input else env_access
            current_sk = sk_input if sk_input else env_secret

    # 포트폴리오 관리
    st.sidebar.subheader("포트폴리오")
    st.sidebar.caption("메인 행의 [+]를 체크하면 바로 아래에 보조 행이 추가됩니다.")

    # Interval mapping (UI label -> API key)
    INTERVAL_MAP = {
        "1D": "day",
        "4H": "minute240",
        "1H": "minute60",
        "30m": "minute30",
        "15m": "minute15",
        "5m": "minute5",
        "1m": "minute1",
    }
    INTERVAL_REV_MAP = {v: k for k, v in INTERVAL_MAP.items()}
    CANDLES_PER_DAY = {
        "day": 1, "minute240": 6, "minute60": 24,
        "minute30": 48, "minute15": 96, "minute5": 288, "minute1": 1440,
    }
    ROW_TYPE_MAIN = "메인"
    ROW_TYPE_AUX = "보조"
    STRATEGY_AUX = "보조"

    def _is_aux_row(row_type_val="", strategy_val="", force_aux=False):
        if force_aux:
            return True
        rt = str(row_type_val or "").strip().lower()
        stg = str(strategy_val or "").strip().lower()
        return (rt in {"aux", "보조"}) or (stg in {"aux", "보조"})

    def _normalize_aux_ma_count_label(val) -> str:
        _v = str(val).strip().lower()
        if _v in {"1", "1개", "single", "one"}:
            return "1개"
        return "2개"

    def _apply_strategy_no(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        out = df.copy()
        main_no_map = {}
        next_no = 1

        # 1차: 메인 행 번호 부여
        for _, rr in out.iterrows():
            is_aux = _is_aux_row(rr.get("row_type", ""), rr.get("strategy", ""))
            if is_aux:
                continue
            rid = str(rr.get("row_id", "") or "").strip()
            pid = str(rr.get("parent_id", "") or "").strip()

            no = None
            if rid and rid in main_no_map:
                no = main_no_map[rid]
            elif pid and pid in main_no_map:
                no = main_no_map[pid]
            else:
                no = next_no
                next_no += 1

            if rid:
                main_no_map[rid] = no
            if pid:
                main_no_map[pid] = no

        # 2차: 전체 행(보조 포함) 번호 계산
        no_vals = []
        for _, rr in out.iterrows():
            is_aux = _is_aux_row(rr.get("row_type", ""), rr.get("strategy", ""))
            rid = str(rr.get("row_id", "") or "").strip()
            pid = str(rr.get("parent_id", "") or "").strip()

            if is_aux:
                no = main_no_map.get(pid) or main_no_map.get(rid)
            else:
                no = main_no_map.get(rid) or main_no_map.get(pid)

            no_vals.append("" if no is None else int(no))

        out["strategy_no"] = no_vals
        return out

    # Load portfolio from user config, then portfolio.json
    PORTFOLIO_JSON_LOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")
    _pjson_config = {}
    if os.path.exists(PORTFOLIO_JSON_LOAD):
        try:
            with open(PORTFOLIO_JSON_LOAD, "r", encoding="utf-8") as f:
                _pjson_raw = json.load(f)
            if isinstance(_pjson_raw, dict):
                _pjson_config = _pjson_raw
            elif isinstance(_pjson_raw, list):
                _pjson_config = {"portfolio": _pjson_raw}
        except Exception:
            pass

    default_portfolio = config.get("portfolio", None) or _pjson_config.get("portfolio", None)
    default_aux_portfolio = config.get("aux_portfolio", None)
    if default_aux_portfolio is None:
        default_aux_portfolio = _pjson_config.get("aux_portfolio", [])

    if not default_portfolio:
        st.error("portfolio.json에 포트폴리오 데이터가 없습니다.")
        st.stop()

    merged_default = list(default_portfolio)
    for ax in (default_aux_portfolio or []):
        ax_row = dict(ax)
        ax_row["is_aux"] = True
        merged_default.append(ax_row)

    # Normalize rows for sidebar editor
    sanitized_portfolio = []
    main_count = 0
    for p in merged_default:
        is_aux = _is_aux_row(
            p.get("row_type", ""),
            p.get("strategy", ""),
            force_aux=bool(p.get("is_aux", False)),
        )
        if not is_aux:
            main_count += 1
    if main_count <= 0:
        main_count = 1

    strat_map = {"SMA ??": "SMA", "??? ??": "Donchian", "Donchian Trend": "Donchian"}

    for idx, p in enumerate(merged_default):
        coin_val = str(p.get("coin", "BTC")).upper()
        is_aux = _is_aux_row(
            p.get("row_type", ""),
            p.get("strategy", ""),
            force_aux=bool(p.get("is_aux", False)),
        )
        row_type = ROW_TYPE_AUX if is_aux else ROW_TYPE_MAIN

        api_interval = p.get("interval", "day")
        label_interval = INTERVAL_REV_MAP.get(api_interval, "1D")

        strat_val = p.get("strategy", "SMA")
        strat_val = strat_map.get(strat_val, strat_val)
        if is_aux:
            strat_val = STRATEGY_AUX

        row_id = str(p.get("row_id", "")) or f"{row_type}_{coin_val}_{idx}"
        parent_id = str(p.get("parent_id", ""))
        if (not parent_id) and (not is_aux):
            parent_id = row_id

        parameter = None if is_aux else int(p.get("parameter", p.get("sma", 20)) or 20)
        sell_param = None if is_aux else int(p.get("sell_parameter", 0) or 0)
        if is_aux:
            weight_val = None
        else:
            weight_val = float(p.get("weight", (100 // main_count)) or 0)

        sanitized_portfolio.append({
            "add_aux": False,
            "row_type": row_type,
            "coin": coin_val,
            "strategy": strat_val,
            "parameter": parameter,
            "sell_parameter": sell_param,
            "weight": weight_val,
            "interval": label_interval,
            "aux_ma_count": _normalize_aux_ma_count_label(p.get("aux_ma_count", 2)),
            "aux_ma_short": int(p.get("aux_ma_short", 5) or 5),
            "aux_ma_long": int(p.get("aux_ma_long", 20) or 20),
            "aux_threshold": float(p.get("aux_threshold", -5.0) or -5.0),
            "aux_tp1": float(p.get("aux_tp1", 3.0) or 3.0),
            "aux_tp2": float(p.get("aux_tp2", 10.0) or 10.0),
            "aux_split": int(p.get("aux_split", 3) or 3),
            "aux_seed_mode": {"equal": "균등", "pyramiding": "피라미딩"}.get(str(p.get("aux_seed_mode", "equal") or "equal"), "균등"),
            "aux_pyramid_ratio": float(p.get("aux_pyramid_ratio", 1.3) or 1.3),
            "row_id": row_id,
            "parent_id": parent_id,
        })

    df_portfolio = pd.DataFrame(sanitized_portfolio)
    editor_columns = [
        "add_aux", "strategy_no", "row_type", "coin", "strategy", "parameter", "sell_parameter", "weight", "interval",
        "aux_ma_count", "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio",
        "row_id", "parent_id",
    ]
    for c in editor_columns:
        if c not in df_portfolio.columns:
            df_portfolio[c] = "" if c in {"strategy_no", "row_type", "coin", "strategy", "interval", "aux_ma_count", "aux_seed_mode", "row_id", "parent_id"} else 0
    if "aux_ma_count" in df_portfolio.columns:
        df_portfolio["aux_ma_count"] = df_portfolio["aux_ma_count"].apply(_normalize_aux_ma_count_label)
    df_portfolio = df_portfolio[editor_columns]
    df_portfolio = _apply_strategy_no(df_portfolio)

    interval_options = list(INTERVAL_MAP.keys())
    strategy_options = ["SMA", "Donchian", STRATEGY_AUX]

    editor_state_key = "portfolio_editor_df"
    source_df = st.session_state.get(editor_state_key)
    if not isinstance(source_df, pd.DataFrame) or list(source_df.columns) != editor_columns:
        source_df = df_portfolio.copy()
    if "aux_ma_count" in source_df.columns:
        source_df["aux_ma_count"] = source_df["aux_ma_count"].apply(_normalize_aux_ma_count_label)
    source_df = _apply_strategy_no(source_df)

    _aux_cols = ["aux_ma_count", "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio"]
    _main_display_cols = [c for c in source_df.columns if c not in _aux_cols and c not in ("row_id", "parent_id")]

    if IS_CLOUD:
        st.sidebar.dataframe(source_df[_main_display_cols], use_container_width=True, hide_index=True)
        edited_portfolio = source_df.copy()
    else:
        _editor_df = source_df.drop(columns=["row_id", "parent_id"])
        edited_portfolio = st.sidebar.data_editor(
            _editor_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="portfolio_editor_widget",
            column_order=_main_display_cols,
            column_config={
                "add_aux": st.column_config.CheckboxColumn("+", help="메인 행 아래에 보조 행 추가", default=False),
                "strategy_no": st.column_config.TextColumn("번호", disabled=True),
                "row_type": st.column_config.TextColumn("유형", disabled=True),
                "coin": st.column_config.TextColumn("코인", required=True),
                "strategy": st.column_config.SelectboxColumn("전략", options=strategy_options, required=True, default="SMA"),
                "parameter": st.column_config.NumberColumn("매수", min_value=0, max_value=300, step=1, required=False),
                "sell_parameter": st.column_config.NumberColumn("매도", min_value=0, max_value=300, step=1, required=False),
                "weight": st.column_config.NumberColumn("비중", min_value=0, max_value=100, step=1, required=False, format="%d%%"),
                "interval": st.column_config.SelectboxColumn("주기", options=interval_options, required=True, default="1D"),
            },
        )

        # Re-attach internal columns from source_df by position (best effort)
        src_internal = source_df[["row_id", "parent_id"]].reset_index(drop=True)
        edited_portfolio = edited_portfolio.reset_index(drop=True)
        if len(src_internal) >= len(edited_portfolio):
            edited_portfolio["row_id"] = src_internal.loc[:len(edited_portfolio)-1, "row_id"].values
            edited_portfolio["parent_id"] = src_internal.loc[:len(edited_portfolio)-1, "parent_id"].values
        else:
            edited_portfolio["row_id"] = [""] * len(edited_portfolio)
            edited_portfolio["parent_id"] = [""] * len(edited_portfolio)

        # Add AUX row right below main row when [+] checked
        rows = edited_portfolio.to_dict("records")
        existing_aux_parent = set()
        for rr in rows:
            if _is_aux_row(rr.get("row_type", ""), rr.get("strategy", "")):
                existing_aux_parent.add(str(rr.get("parent_id", "")))

        new_rows = []
        added_any = False
        for ridx, r in enumerate(rows):
            rr = dict(r)
            rr["coin"] = str(rr.get("coin", "BTC")).upper()
            rr["row_type"] = ROW_TYPE_AUX if _is_aux_row(rr.get("row_type", ""), rr.get("strategy", "")) else ROW_TYPE_MAIN
            rr["strategy"] = STRATEGY_AUX if rr["row_type"] == ROW_TYPE_AUX else str(rr.get("strategy", "SMA"))
            rr["interval"] = str(rr.get("interval", "1D"))
            rr["row_id"] = str(rr.get("row_id", "")) or f"{rr['row_type']}_{rr['coin']}_{ridx}"
            if rr["row_type"] == ROW_TYPE_MAIN:
                rr["parent_id"] = rr["row_id"]
                if pd.isna(rr.get("parameter", None)):
                    rr["parameter"] = 20
                if pd.isna(rr.get("sell_parameter", None)):
                    rr["sell_parameter"] = 0
            else:
                rr["parent_id"] = str(rr.get("parent_id", ""))
                rr["weight"] = None
                rr["parameter"] = None
                rr["sell_parameter"] = None

            trigger_add = bool(rr.get("add_aux", False))
            rr["add_aux"] = False
            new_rows.append(rr)

            if rr["row_type"] == ROW_TYPE_MAIN and trigger_add:
                parent_id = rr["row_id"]
                has_aux_already = (parent_id in existing_aux_parent) or any(
                    _is_aux_row(x.get("row_type", ""), x.get("strategy", "")) and str(x.get("parent_id", "")) == parent_id
                    for x in new_rows
                )
                if has_aux_already:
                    st.sidebar.info(f"{rr['coin']} / {rr.get('strategy', '')} 아래에 이미 보조 행이 있습니다.")
                else:
                    aux_row = {
                        "add_aux": False,
                        "row_type": ROW_TYPE_AUX,
                        "coin": rr["coin"],
                        "strategy": STRATEGY_AUX,
                        "parameter": None,
                        "sell_parameter": None,
                        "weight": None,
                        "interval": "1H",
                        "aux_ma_count": "2개",
                        "aux_ma_short": int(rr.get("aux_ma_short", 5) or 5),
                        "aux_ma_long": max(int(rr.get("aux_ma_long", 20) or 20), int(rr.get("aux_ma_short", 5) or 5) + 1),
                        "aux_threshold": float(rr.get("aux_threshold", -5.0) or -5.0),
                        "aux_tp1": float(rr.get("aux_tp1", 3.0) or 3.0),
                        "aux_tp2": max(float(rr.get("aux_tp2", 10.0) or 10.0), float(rr.get("aux_tp1", 3.0) or 3.0)),
                        "aux_split": int(rr.get("aux_split", 3) or 3),
                        "aux_seed_mode": "균등",
                        "aux_pyramid_ratio": float(rr.get("aux_pyramid_ratio", 1.3) or 1.3),
                        "row_id": f"aux_{rr['coin']}_{ridx}",
                        "parent_id": parent_id,
                    }
                    new_rows.append(aux_row)
                    added_any = True

        edited_portfolio = pd.DataFrame(new_rows)
        for c in editor_columns:
            if c not in edited_portfolio.columns:
                edited_portfolio[c] = "" if c in {"strategy_no", "row_type", "coin", "strategy", "interval", "aux_ma_count", "aux_seed_mode", "row_id", "parent_id"} else 0
        if "aux_ma_count" in edited_portfolio.columns:
            edited_portfolio["aux_ma_count"] = edited_portfolio["aux_ma_count"].apply(_normalize_aux_ma_count_label)
        edited_portfolio = edited_portfolio[editor_columns]
        edited_portfolio = _apply_strategy_no(edited_portfolio)

        st.session_state[editor_state_key] = edited_portfolio.copy()
        if added_any:
            st.rerun()

    edited_portfolio = _apply_strategy_no(edited_portfolio)

    # ── 보조 전략 파라미터 편집 (별도 expander + data_editor) ──
    _aux_mask = edited_portfolio.apply(
        lambda _r: _is_aux_row(_r.get("row_type", ""), _r.get("strategy", "")),
        axis=1,
    )
    if _aux_mask.any() and not IS_CLOUD:
        with st.sidebar.expander("⚙️ 보조 전략 설정", expanded=True):
            st.caption("이평선 수가 1개일 때는 장기MA 값이 무시되고 단기MA 이격도만 사용됩니다.")
            _aux_display_cols = ["strategy_no", "coin", "aux_ma_count", "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio"]
            _aux_df = edited_portfolio.loc[_aux_mask, _aux_display_cols].copy()
            _aux_df = _aux_df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
            _aux_edit = st.data_editor(
                _aux_df.drop(columns=["_orig_idx"]),
                use_container_width=True,
                hide_index=True,
                key="aux_param_editor",
                column_config={
                    "strategy_no": st.column_config.TextColumn("번호", disabled=True),
                    "coin": st.column_config.TextColumn("코인", disabled=True),
                    "aux_ma_count": st.column_config.SelectboxColumn("이평선 수", options=["1개", "2개"], required=True),
                    "aux_ma_short": st.column_config.NumberColumn("단기MA", min_value=2, max_value=500, step=1),
                    "aux_ma_long": st.column_config.NumberColumn("장기MA", min_value=3, max_value=500, step=1),
                    "aux_threshold": st.column_config.NumberColumn("임계(%)", min_value=-50.0, max_value=0.0, step=0.5, format="%.1f"),
                    "aux_tp1": st.column_config.NumberColumn("TP1(%)", min_value=0.1, max_value=50.0, step=0.5, format="%.1f"),
                    "aux_tp2": st.column_config.NumberColumn("TP2(%)", min_value=0.1, max_value=100.0, step=0.5, format="%.1f"),
                    "aux_split": st.column_config.NumberColumn("매수분할", min_value=1, max_value=10, step=1),
                    "aux_seed_mode": st.column_config.SelectboxColumn("시드모드", options=["균등", "피라미딩"]),
                    "aux_pyramid_ratio": st.column_config.NumberColumn("피라미딩비율", min_value=1.0, max_value=5.0, step=0.1, format="%.1f"),
                },
            )
            # 변경사항을 edited_portfolio에 반영
            _aux_orig_indices = []
            for _ri in range(len(_aux_edit)):
                _orig = int(_aux_df.iloc[_ri]["_orig_idx"])
                _aux_orig_indices.append(_orig)
                for _col in _aux_display_cols:
                    if _col in {"strategy_no", "coin"}:
                        continue
                    edited_portfolio.at[_orig, _col] = _aux_edit.iloc[_ri][_col]
            for _orig in _aux_orig_indices:
                _ma_count_label = _normalize_aux_ma_count_label(edited_portfolio.at[_orig, "aux_ma_count"])
                edited_portfolio.at[_orig, "aux_ma_count"] = _ma_count_label

                _ms_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_ma_short"], errors="coerce")
                _ms = int(_ms_raw) if pd.notna(_ms_raw) else 5
                _ms = max(2, _ms)

                _ml_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_ma_long"], errors="coerce")
                _ml = int(_ml_raw) if pd.notna(_ml_raw) else 20
                _ml = max(3, _ml)
                if _ma_count_label == "2개" and _ml <= _ms:
                    _ml = _ms + 1
                if _ma_count_label == "1개":
                    _ml = _ms

                _tp1_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_tp1"], errors="coerce")
                _tp2_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_tp2"], errors="coerce")
                _tp1 = float(_tp1_raw) if pd.notna(_tp1_raw) else 3.0
                _tp2 = float(_tp2_raw) if pd.notna(_tp2_raw) else 10.0
                if _tp2 < _tp1:
                    _tp2 = _tp1

                edited_portfolio.at[_orig, "aux_ma_short"] = _ms
                edited_portfolio.at[_orig, "aux_ma_long"] = _ml
                edited_portfolio.at[_orig, "aux_tp1"] = _tp1
                edited_portfolio.at[_orig, "aux_tp2"] = _tp2
            edited_portfolio = _apply_strategy_no(edited_portfolio)
            st.session_state[editor_state_key] = edited_portfolio.copy()

    # Calculate total weight from main rows only
    _main_rows_df = edited_portfolio[~_aux_mask]
    total_weight = pd.to_numeric(_main_rows_df["weight"], errors="coerce").fillna(0).sum()
    if total_weight > 100:
        st.sidebar.error(f"총 비중이 {total_weight}%입니다 (100% 이하여야 합니다)")
    else:
        cash_weight = 100 - total_weight
        st.sidebar.info(f"투자: {total_weight}% | 현금: {cash_weight}%")

    # Convert back to dict lists
    portfolio_list = []
    aux_portfolio_list = []
    for r in edited_portfolio.to_dict('records'):
        label_key = r.get('interval', '1D')
        api_key = INTERVAL_MAP.get(label_key, 'day')
        row_is_aux = _is_aux_row(r.get('row_type', ''), r.get('strategy', ''))

        coin_val = str(r.get('coin', 'BTC')).upper().strip()
        if not coin_val:
            continue

        if row_is_aux:
            _ma_count_label = _normalize_aux_ma_count_label(r.get('aux_ma_count', 2))
            _ma_short_raw = pd.to_numeric(r.get('aux_ma_short', 5), errors='coerce')
            _ma_long_raw = pd.to_numeric(r.get('aux_ma_long', 20), errors='coerce')
            _ma_short = int(_ma_short_raw) if pd.notna(_ma_short_raw) else 5
            _ma_long = int(_ma_long_raw) if pd.notna(_ma_long_raw) else 20
            _ma_short = max(2, _ma_short)
            _ma_long = max(3, _ma_long)
            if _ma_count_label == "2개" and _ma_long <= _ma_short:
                _ma_long = _ma_short + 1
            if _ma_count_label == "1개":
                _ma_long = _ma_short
            aux_portfolio_list.append({
                'coin': coin_val,
                'interval': api_key,
                'parent_id': str(r.get('parent_id', '')),
                'aux_ma_count': 1 if _ma_count_label == "1개" else 2,
                'aux_ma_short': _ma_short,
                'aux_ma_long': _ma_long,
                'aux_threshold': float(r.get('aux_threshold', -5.0) or -5.0),
                'aux_tp1': float(r.get('aux_tp1', 3.0) or 3.0),
                'aux_tp2': float(r.get('aux_tp2', 10.0) or 10.0),
                'aux_split': int(r.get('aux_split', 3) or 3),
                'aux_seed_mode': {"균등": "equal", "피라미딩": "pyramiding"}.get(str(r.get('aux_seed_mode', '균등') or '균등'), 'equal'),
                'aux_pyramid_ratio': float(r.get('aux_pyramid_ratio', 1.3) or 1.3),
            })
            continue

        param_raw = r.get('parameter', 20)
        if pd.isna(param_raw) or str(param_raw).strip() == "":
            param_val = 20
        else:
            param_val = int(float(param_raw))

        sell_raw = r.get('sell_parameter', 0)
        if pd.isna(sell_raw) or str(sell_raw).strip() == "":
            sell_p = 0
        else:
            sell_p = int(float(sell_raw))

        weight_raw = r.get('weight', 0)
        if pd.isna(weight_raw) or str(weight_raw).strip() == "":
            weight_val = 0.0
        else:
            weight_val = float(weight_raw)
        portfolio_list.append({
            'market': 'KRW',
            'coin': coin_val,
            'strategy': str(r.get('strategy', 'SMA')),
            'parameter': param_val,
            'sell_parameter': sell_p,
            'weight': weight_val,
            'interval': api_key,
        })

    # Global Settings
    st.sidebar.subheader("공통 설정")
    # Interval Removed (Per-Coin Setting)
    
    default_start_str = config.get("start_date", None) or _pjson_config.get("start_date", None)
    if not default_start_str:
        st.error("start_date 설정이 없습니다. 로컬에서 portfolio.json에 start_date를 설정 후 push 해주세요.")
        st.stop()
    try:
        default_start = pd.to_datetime(default_start_str).date()
    except:
        st.error(f"start_date 형식 오류: {default_start_str}")
        st.stop()
    start_date = st.sidebar.date_input(
        "기준 시작일",
        value=default_start,
        help="수익률 계산 및 이론적 자산 비교를 위한 기준일입니다. 실제 매매 신호와는 무관합니다.",
        disabled=IS_CLOUD
    )

    # Capital Input Customization
    default_cap = config.get("initial_cap", None) or _pjson_config.get("initial_cap", None)
    if not default_cap:
        st.error("initial_cap 설정이 없습니다. 로컬에서 portfolio.json에 initial_cap을 설정 후 push 해주세요.")
        st.stop()
    initial_cap = st.sidebar.number_input(
        "초기 자본금 (KRW - 원 단위)",
        value=default_cap, step=100000, format="%d",
        help="시뮬레이션을 위한 초기 투자금 설정입니다. 실제 계좌 잔고와는 무관하며, 수익률 계산의 기준이 됩니다.",
        disabled=IS_CLOUD
    )
    st.sidebar.caption(f"설정: **{initial_cap:,.0f} KRW**")
    
    # Strategy Selection REMOVED (Moved to Per-Coin)

    PORTFOLIO_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")

    if not IS_CLOUD:
        if st.sidebar.button("Save"):
            new_config = {
                "portfolio": portfolio_list,
                "aux_portfolio": aux_portfolio_list,
                "start_date": str(start_date),
                "initial_cap": initial_cap,
            }
            save_config(new_config)
            portfolio_json_data = {
                "portfolio": portfolio_list,
                "aux_portfolio": aux_portfolio_list,
                "start_date": str(start_date),
                "initial_cap": initial_cap,
            }
            with open(PORTFOLIO_JSON, "w", encoding="utf-8") as f:
                json.dump(portfolio_json_data, f, indent=2, ensure_ascii=False)
            st.sidebar.success("Saved")

    # --- data_manager Import ---
    from src.engine.data_manager import MarketDataWorker

    # ... (Keep existing history cache if useful, or move to worker too. Let's keep separate for now)
    @st.cache_data(ttl=60)
    # Function to fetch history (Caching disabled for now due to obj hashing)
    def fetch_history_cached(_trader, kind, currency="KRW"):
         try:
            return _trader.get_history(kind, currency)
         except TypeError:
            # Fallback if get_history signature issues
            return _trader.get_history(kind)
    
    # Initialize Objects
    backtest_engine = BacktestEngine()

    @st.cache_data(ttl=300)
    def _cached_backtest(ticker, period, interval, count, start_date_str, initial_balance, strategy_mode, sell_period_ratio, _df_hash):
        """백테스트 결과를 5분간 캐싱 (동일 파라미터 재계산 방지)"""
        df_bt_local = data_cache.load_cached(ticker, interval)
        if df_bt_local is None or len(df_bt_local) < period:
            return None
        return backtest_engine.run_backtest(
            ticker, period=period, interval=interval, count=count,
            start_date=start_date_str, initial_balance=initial_balance,
            df=df_bt_local, strategy_mode=strategy_mode,
            sell_period_ratio=sell_period_ratio
        )
    
    trader = None
    if current_ak and current_sk:
        @st.cache_resource
        def get_trader(ak, sk):
            return UpbitTrader(ak, sk)
        trader = get_trader(current_ak, current_sk)

    # --- Background Worker Setup ---
    from src.engine.data_manager import CoinTradingWorker

    @st.cache_resource
    def get_worker():
        return MarketDataWorker()

    @st.cache_resource
    def get_coin_trading_worker():
        w = CoinTradingWorker()
        w.start()
        return w

    worker = get_worker()
    _ = get_coin_trading_worker()

    # 업비트 KRW 마켓 호가 단위 (Tick Size)
    def get_tick_size(price):
        """가격에 따른 업비트 호가 단위 반환"""
        if price >= 2_000_000: return 1000
        elif price >= 1_000_000: return 1000
        elif price >= 500_000: return 500
        elif price >= 100_000: return 100
        elif price >= 50_000: return 50
        elif price >= 10_000: return 10
        elif price >= 5_000: return 5
        elif price >= 1_000: return 1
        elif price >= 100: return 1
        elif price >= 10: return 0.1
        elif price >= 1: return 0.01
        else: return 0.001

    def align_price(price, tick_size):
        """가격을 호가 단위에 맞게 정렬"""
        if tick_size >= 1:
            return int(price // tick_size * tick_size)
        else:
            import math
            decimals = max(0, -int(math.floor(math.log10(tick_size))))
            return round(price // tick_size * tick_size, decimals)

    # ── TTL 캐시: API 호출 최소화 ──
    def _ttl_cache(key, fn, ttl=5):
        """세션 기반 TTL 캐시. ttl초 이내 재호출시 캐시 반환."""
        now = time.time()
        ck, tk = f"__c_{key}", f"__t_{key}"
        if ck in st.session_state and (now - st.session_state.get(tk, 0)) < ttl:
            return st.session_state[ck]
        val = fn()
        st.session_state[ck] = val
        st.session_state[tk] = now
        return val

    def _clear_cache(*keys):
        """거래 후 캐시 무효화"""
        for k in keys:
            st.session_state.pop(f"__c_{k}", None)
            st.session_state.pop(f"__t_{k}", None)

    # 시가총액 상위 20 티커 (글로벌 Market Cap 기준)
    TOP_20_TICKERS = [
        "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
        "KRW-ADA", "KRW-SHIB", "KRW-TRX", "KRW-AVAX", "KRW-LINK",
        "KRW-BCH", "KRW-DOT", "KRW-NEAR", "KRW-POL", "KRW-ETC",
        "KRW-XLM", "KRW-STX", "KRW-HBAR", "KRW-EOS", "KRW-SAND"
    ]

    # --- Tabs ---
    tab1, tab5, tab3, tab4, tab6 = st.tabs(["🚀 실시간 포트폴리오", "🛒 수동 주문", "📜 거래 내역", "📊 백테스트", "⏰ 트리거"])

    # --- Tab 1: Live Portfolio (Default) ---
    with tab1:
        st.header("실시간 포트폴리오 대시보드")
        st.caption("설정된 모든 자산을 모니터링합니다.")
        
        if not trader:
            st.warning("사이드바에서 API 키를 설정해주세요.")
        else:
            # Configure and Start Worker
            worker.update_config(portfolio_list)
            worker.start_worker()

            w_msg, w_time = worker.get_status()

            # Control Bar
            col_ctrl1, col_ctrl2 = st.columns([1,3])
            with col_ctrl1:
                if st.button("🔄 새로고침"):
                    _clear_cache("krw_bal_t1", "prices_t1", "balances_t1")
                    st.rerun()
            with col_ctrl2:
                st.info(f"워커 상태: **{w_msg}**")

            if not portfolio_list:
                st.warning("사이드바에서 포트폴리오에 코인을 추가해주세요.")
            else:
                count = len(portfolio_list)
                per_coin_cap = initial_cap / count

                # ── 일괄 API 호출 (TTL 캐시): 가격·잔고를 1회씩만 가져옴 ──
                unique_coins = list(dict.fromkeys(item['coin'].upper() for item in portfolio_list))
                unique_tickers = list(dict.fromkeys(f"{item['market']}-{item['coin'].upper()}" for item in portfolio_list))

                def _fetch_all_prices():
                    """모든 코인 가격을 한번에 가져옴 (Public API - IP 제한 없음)"""
                    return data_cache.get_current_prices_local_first(
                        unique_tickers,
                        ttl_sec=5.0,
                        allow_api_fallback=True,
                    )

                all_prices = _ttl_cache("prices_t1", _fetch_all_prices, ttl=5)

                # 잔고 조회: API 우선, 실패 시 캐시 파일 사용
                _balance_from_cache = False
                _balance_cache_time = ""

                def _fetch_all_balances():
                    """모든 코인 잔고를 1회 API 호출로 가져옴"""
                    if hasattr(trader, 'get_all_balances'):
                        raw = trader.get_all_balances()
                        if raw and isinstance(raw, dict) and len(raw) > 0:
                            return raw
                    return None

                _live_bal = _ttl_cache("balances_t1", _fetch_all_balances, ttl=10)

                if _live_bal and isinstance(_live_bal, dict) and len(_live_bal) > 0:
                    krw_bal = float(_live_bal.get('KRW', 0) or 0)
                    all_balances = {c: float(_live_bal.get(c, 0) or 0) for c in unique_coins}
                else:
                    # API 실패 → 캐시 파일에서 로드
                    _cached = _load_balance_cache()
                    if _cached and _cached.get("balances"):
                        _bal = _cached["balances"]
                        krw_bal = float(_bal.get('KRW', 0) or 0)
                        all_balances = {c: float(_bal.get(c, 0) or 0) for c in unique_coins}
                        _balance_from_cache = True
                        _balance_cache_time = _cached.get("updated_at", "")
                    else:
                        krw_bal = 0
                        all_balances = {c: 0 for c in unique_coins}

                # --- Total Summary Container ---
                st.subheader("🏁 포트폴리오 요약")
                if _balance_from_cache:
                    st.caption(f"초기자본: {initial_cap:,.0f} KRW | 자산수: {count} | 자산당: {per_coin_cap:,.0f} KRW")
                    st.info(f"잔고: VM 캐시 기준 ({_balance_cache_time})" if _balance_cache_time else "잔고: VM 캐시 기준")
                else:
                    st.caption(f"초기자본: {initial_cap:,.0f} KRW | 자산수: {count} | 자산당: {per_coin_cap:,.0f} KRW")

                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                total_real_val = krw_bal
                total_init_val = initial_cap

                # Cash Logic
                total_weight_alloc = sum([item.get('weight', 0) for item in portfolio_list])
                cash_ratio = max(0, 100 - total_weight_alloc) / 100.0
                reserved_cash = initial_cap * cash_ratio

                # Add reserved cash to Theo Value (as it stays as cash)
                total_theo_val = reserved_cash

                # --- 전체 자산 현황 테이블 (캐시된 데이터 사용) ---
                asset_summary_rows = [{"자산": "KRW (현금)", "보유량": f"{krw_bal:,.0f}", "현재가": "-", "평가금액(KRW)": f"{krw_bal:,.0f}", "상태": "-"}]
                seen_coins_summary = set()
                for s_item in portfolio_list:
                    s_coin = s_item['coin'].upper()
                    if s_coin in seen_coins_summary:
                        continue
                    seen_coins_summary.add(s_coin)
                    s_ticker = f"{s_item['market']}-{s_coin}"
                    s_bal = all_balances.get(s_coin, 0)
                    s_price = all_prices.get(s_ticker, 0) or 0
                    s_val = s_bal * s_price
                    is_holding = s_val >= 5000
                    asset_summary_rows.append({
                        "자산": s_coin,
                        "보유량": (f"{s_bal:.8f}" if s_bal < 1 else f"{s_bal:,.4f}") if s_bal > 0 else "0",
                        "현재가": f"{s_price:,.0f}",
                        "평가금액(KRW)": f"{s_val:,.0f}",
                        "상태": "보유중" if is_holding else "미보유",
                    })
                total_real_summary = krw_bal + sum(
                    all_balances.get(c, 0) * (all_prices.get(f"KRW-{c}", 0) or 0)
                    for c in seen_coins_summary
                )
                asset_summary_rows.append({
                    "자산": "합계",
                    "보유량": "",
                    "현재가": "",
                    "평가금액(KRW)": f"{total_real_summary:,.0f}",
                    "상태": "",
                })
                with st.expander(f"💰 전체 자산 현황 (Total: {total_real_summary:,.0f} KRW)", expanded=True):
                    st.dataframe(pd.DataFrame(asset_summary_rows), use_container_width=True, hide_index=True)

                    # ── 포트폴리오 리밸런싱 (자산현황 내 통합) ──
                    st.divider()
                    st.markdown("**⚖️ 포트폴리오 리밸런싱**")
                    krw_balance = krw_bal

                    asset_states = []
                    for rb_idx, rb_item in enumerate(portfolio_list):
                        rb_ticker = f"{rb_item['market']}-{rb_item['coin'].upper()}"
                        rb_coin = rb_item['coin'].upper()
                        rb_weight = rb_item.get('weight', 0)
                        rb_interval = rb_item.get('interval', 'day')
                        rb_strategy = rb_item.get('strategy', 'SMA Strategy')
                        rb_param = rb_item.get('parameter', 20)
                        rb_sell_param = rb_item.get('sell_parameter', 0)

                        rb_coin_bal = all_balances.get(rb_coin, 0)
                        rb_price = all_prices.get(rb_ticker, 0) or 0
                        rb_coin_val = rb_coin_bal * rb_price
                        rb_status = "HOLD" if rb_coin_val > 5000 else "CASH"

                        rb_signal = "N/A"
                        try:
                            rb_df = worker.get_data(rb_ticker, rb_interval)
                            if rb_df is not None and len(rb_df) >= rb_param:
                                if rb_strategy == "Donchian":
                                    rb_eng = DonchianStrategy()
                                    rb_sp = rb_sell_param or max(5, rb_param // 2)
                                    rb_df = rb_eng.create_features(rb_df, buy_period=rb_param, sell_period=rb_sp)
                                    rb_signal = rb_eng.get_signal(rb_df.iloc[-2], buy_period=rb_param, sell_period=rb_sp)
                                else:
                                    rb_eng = SMAStrategy()
                                    rb_df = rb_eng.create_features(rb_df, periods=[rb_param])
                                    rb_signal = rb_eng.get_signal(rb_df.iloc[-2], strategy_type='SMA_CROSS', ma_period=rb_param)
                        except Exception:
                            pass

                        rb_target_krw = total_real_summary * (rb_weight / 100.0)

                        asset_states.append({
                            "ticker": rb_ticker, "coin": rb_coin, "weight": rb_weight,
                            "interval": rb_interval, "strategy": rb_strategy,
                            "param": rb_param, "sell_param": rb_sell_param,
                            "status": rb_status, "signal": rb_signal,
                            "coin_bal": rb_coin_bal, "coin_val": rb_coin_val,
                            "price": rb_price, "target_krw": rb_target_krw,
                        })

                    # ── 같은 코인 그룹핑 → 통합 시그널 ──
                    from collections import OrderedDict
                    coin_groups = OrderedDict()
                    for a in asset_states:
                        key = a['ticker']
                        if key not in coin_groups:
                            coin_groups[key] = []
                        coin_groups[key].append(a)

                    merged_assets = []
                    for ticker, group in coin_groups.items():
                        signals = [a['signal'] for a in group]
                        # 통합 시그널: 모든 전략 BUY → BUY, 그 외 → SELL
                        unified_signal = 'BUY' if all(s == 'BUY' for s in signals) else 'SELL'
                        total_weight = sum(a['weight'] for a in group)
                        coin_val = group[0]['coin_val']  # 같은 코인이므로 동일
                        price = group[0]['price']
                        status = group[0]['status']
                        target_krw = sum(a['target_krw'] for a in group)
                        strategies = " / ".join(f"{a['strategy']}{a['param']}({a['interval']})" for a in group)
                        detail_signals = " / ".join(f"{a['strategy']}{a['param']}={a['signal']}" for a in group)
                        merged_assets.append({
                            "ticker": ticker, "weight": total_weight,
                            "strategies": strategies, "detail_signals": detail_signals,
                            "status": status, "signal": unified_signal,
                            "coin_val": coin_val, "price": price,
                            "target_krw": target_krw, "group": group,
                        })

                    cash_merged = [a for a in merged_assets if a['status'] == 'CASH']
                    buy_merged = [a for a in merged_assets if a['signal'] == 'BUY']

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("보유 현금 (KRW)", f"{krw_balance:,.0f}")
                    rc2.metric("CASH 자산", f"{len(cash_merged)} / {len(merged_assets)}")
                    rc3.metric("BUY 시그널", f"{len(buy_merged)} / {len(merged_assets)}")

                    rebal_data = []
                    for a in merged_assets:
                        # BUY: 미보유+BUY시그널 → 매수, SELL: 보유중+SELL시그널 → 매도
                        if a['signal'] == 'BUY' and a['status'] == 'CASH':
                            action = "매수"
                        elif a['signal'] == 'SELL' and a['status'] == 'HOLD':
                            action = "매도"
                        elif a['signal'] == 'BUY' and a['status'] == 'HOLD':
                            action = "보유 유지"
                        else:  # SELL + CASH
                            action = "대기"
                        rebal_data.append({
                            "종목": a['ticker'],
                            "전략": a['strategies'],
                            "비중": f"{a['weight']}%",
                            "시그널": a['signal'],
                            "상세": a['detail_signals'],
                            "현재가치(KRW)": f"{a['coin_val']:,.0f}",
                            "목표(KRW)": f"{a['target_krw']:,.0f}",
                            "액션": action,
                        })
                    st.dataframe(pd.DataFrame(rebal_data), use_container_width=True, hide_index=True)

                    buyable = [a for a in merged_assets if a['status'] == 'CASH' and a['signal'] == 'BUY']
                    if not buyable:
                        if len(cash_merged) == 0:
                            st.success("모든 자산이 이미 보유 중입니다.")
                        else:
                            st.info(f"현금 자산 {len(cash_merged)}개가 있지만 BUY 시그널이 없습니다. 시그널 발생 시 매수 가능합니다.")
                    else:
                        st.warning(f"**{len(buyable)}개 자산**에 BUY 시그널이 있습니다. 리밸런싱 매수를 실행할 수 있습니다.")
                        total_buy_weight = sum(a['weight'] for a in buyable)
                        available_krw = krw_balance * 0.999

                        buy_plan = []
                        for a in buyable:
                            alloc_krw = available_krw * (a['weight'] / total_buy_weight) if total_buy_weight > 0 else 0
                            alloc_krw = min(alloc_krw, available_krw)
                            buy_plan.append({
                                "종목": a['ticker'], "비중": f"{a['weight']}%",
                                "배분 금액(KRW)": f"{alloc_krw:,.0f}",
                                "전략": a['strategies'], "현재가": f"{a['price']:,.0f}",
                                "_ticker": a['ticker'], "_krw": alloc_krw, "_group": a['group'],
                            })
                        plan_df = pd.DataFrame(buy_plan)
                        st.dataframe(plan_df[["종목", "비중", "배분 금액(KRW)", "전략", "현재가"]], use_container_width=True, hide_index=True)
                        st.caption(f"총 배분 금액: {sum(p['_krw'] for p in buy_plan):,.0f} KRW / 보유 현금: {krw_balance:,.0f} KRW")

                        if st.button("🚀 리밸런싱 매수 실행", key="btn_rebalance_exec", type="primary"):
                            rebal_results = []
                            rebal_progress = st.progress(0)
                            rebal_log = st.empty()
                            for pi, plan in enumerate(buy_plan):
                                p_ticker = plan['_ticker']
                                p_krw = plan['_krw']
                                p_interval = plan['_group'][0]['interval']
                                if p_krw < 5000:
                                    rebal_results.append({"종목": p_ticker, "결과": "금액 부족 (5,000원 미만)"})
                                    continue
                                rebal_log.text(f"매수 중: {p_ticker} ({p_krw:,.0f} KRW)...")
                                try:
                                    exec_res = trader.smart_buy(p_ticker, p_krw, interval=p_interval)
                                    avg_p = exec_res.get('avg_price', 0)
                                    vol = exec_res.get('filled_volume', 0)
                                    rebal_results.append({
                                        "종목": p_ticker,
                                        "결과": f"체결 완료: {vol:.6f} @ {avg_p:,.0f}",
                                        "금액": f"{exec_res.get('total_krw', 0):,.0f} KRW"
                                    })
                                except Exception as e:
                                    rebal_results.append({"종목": p_ticker, "결과": f"오류: {e}"})
                                rebal_progress.progress((pi + 1) / len(buy_plan))
                                time.sleep(0.5)
                            rebal_progress.progress(1.0)
                            rebal_log.empty()
                            st.success("리밸런싱 완료!")
                            st.dataframe(pd.DataFrame(rebal_results), use_container_width=True, hide_index=True)

                # --- 단기 모니터링 차트 (60봉) ---
                with st.expander("📊 단기 시그널 모니터링 (60봉)", expanded=True):
                    signal_rows = []

                    # BTC / 비BTC 분리 (BTC: 일봉→4시간봉 순)
                    interval_order = {'day': 0, 'minute240': 1, 'minute60': 2, 'minute30': 3, 'minute15': 4, 'minute10': 5}
                    btc_items = sorted(
                        [x for x in portfolio_list if x.get('coin', '').upper() == 'BTC'],
                        key=lambda x: interval_order.get(x.get('interval', 'day'), 99)
                    )
                    other_items = sorted(
                        [x for x in portfolio_list if x.get('coin', '').upper() != 'BTC'],
                        key=lambda x: interval_order.get(x.get('interval', 'day'), 99)
                    )

                    # 차트 데이터 수집 + 렌더링 함수
                    def render_chart_row(items):
                        if not items:
                            return
                        cols = st.columns(len(items))
                        for ci, item in enumerate(items):
                            p_ticker = f"{item['market']}-{item['coin'].upper()}"
                            p_strategy = item.get('strategy', 'SMA')
                            p_param = item.get('parameter', 20)
                            p_sell_param = item.get('sell_parameter', 0) or max(5, p_param // 2)
                            p_interval = item.get('interval', 'day')
                            iv_label = INTERVAL_REV_MAP.get(p_interval, p_interval)

                            try:
                                # Worker 캐시 데이터 우선 사용 (API 호출 제거)
                                df_60 = worker.get_data(p_ticker, p_interval)
                                if df_60 is None or len(df_60) < p_param + 5:
                                    # Worker 데이터 없으면 TTL 캐시로 API 호출
                                    df_60 = _ttl_cache(
                                        f"ohlcv_{p_ticker}_{p_interval}",
                                        lambda t=p_ticker, iv=p_interval, pp=p_param: data_cache.get_ohlcv_local_first(
                                            t,
                                            interval=iv,
                                            count=max(60 + pp, 200),
                                            allow_api_fallback=True,
                                        ),
                                        ttl=30
                                    )
                                if df_60 is None or len(df_60) < p_param + 5:
                                    continue

                                close_now = df_60['close'].iloc[-1]

                                if p_strategy == "Donchian":
                                    upper_vals = df_60['high'].rolling(window=p_param).max().shift(1)
                                    lower_vals = df_60['low'].rolling(window=p_sell_param).min().shift(1)
                                    buy_target = upper_vals.iloc[-1]
                                    sell_target = lower_vals.iloc[-1]
                                    buy_dist = (close_now - buy_target) / buy_target * 100 if buy_target else 0
                                    sell_dist = (close_now - sell_target) / sell_target * 100 if sell_target else 0

                                    # 포지션 상태 시뮬레이션 (돈치안은 상태 기반)
                                    in_position = False
                                    for i in range(len(df_60)):
                                        u = upper_vals.iloc[i]
                                        l = lower_vals.iloc[i]
                                        c = df_60['close'].iloc[i]
                                        if not pd.isna(u) and c > u:
                                            in_position = True
                                        elif not pd.isna(l) and c < l:
                                            in_position = False

                                    if in_position:
                                        position_label = "보유"
                                        signal = "SELL" if close_now < sell_target else "HOLD"
                                    else:
                                        position_label = "현금"
                                        signal = "BUY" if close_now > buy_target else "WAIT"
                                else:
                                    sma_vals = df_60['close'].rolling(window=p_param).mean()
                                    buy_target = sma_vals.iloc[-1]
                                    sell_target = buy_target
                                    buy_dist = (close_now - buy_target) / buy_target * 100 if buy_target else 0
                                    sell_dist = buy_dist
                                    if close_now > buy_target:
                                        signal = "BUY"
                                        position_label = "보유"
                                    else:
                                        signal = "SELL"
                                        position_label = "현금"

                                signal_rows.append({
                                    "종목": p_ticker.replace("KRW-", ""),
                                    "전략": f"{p_strategy} {p_param}",
                                    "시간봉": iv_label,
                                    "포지션": position_label,
                                    "현재가": f"{close_now:,.0f}",
                                    "매수목표": f"{buy_target:,.0f}",
                                    "매도목표": f"{sell_target:,.0f}",
                                    "매수이격도": f"{buy_dist:+.2f}%",
                                    "매도이격도": f"{sell_dist:+.2f}%",
                                })

                                df_chart = df_60.iloc[-60:]
                                with cols[ci]:
                                    fig_m = go.Figure()
                                    fig_m.add_trace(go.Candlestick(
                                        x=df_chart.index, open=df_chart['open'],
                                        high=df_chart['high'], low=df_chart['low'],
                                        close=df_chart['close'], name='가격',
                                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                                    ))

                                    if p_strategy == "Donchian":
                                        upper_chart = upper_vals.loc[df_chart.index]
                                        lower_chart = lower_vals.loc[df_chart.index]
                                        fig_m.add_trace(go.Scatter(
                                            x=df_chart.index, y=upper_chart,
                                            name=f'상단({p_param})', line=dict(color='green', width=1, dash='dot')
                                        ))
                                        fig_m.add_trace(go.Scatter(
                                            x=df_chart.index, y=lower_chart,
                                            name=f'하단({p_sell_param})', line=dict(color='red', width=1, dash='dot')
                                        ))
                                    else:
                                        sma_chart = sma_vals.loc[df_chart.index]
                                        fig_m.add_trace(go.Scatter(
                                            x=df_chart.index, y=sma_chart,
                                            name=f'SMA({p_param})', line=dict(color='orange', width=2)
                                        ))

                                    sig_color = "green" if signal == "BUY" else ("red" if signal == "SELL" else ("blue" if signal == "WAIT" else "gray"))
                                    title_pos = f" [{position_label}]" if p_strategy == "Donchian" else ""
                                    fig_m.update_layout(
                                        title=f"{p_ticker.replace('KRW-','')} {p_strategy}{p_param} ({iv_label}){title_pos} [{buy_dist:+.1f}%]",
                                        title_font_color=sig_color,
                                        height=300, margin=dict(l=0, r=0, t=35, b=30),
                                        xaxis_rangeslider_visible=False,
                                        showlegend=False,
                                        xaxis=dict(showticklabels=True, tickformat='%m/%d %H:%M', tickangle=-45, nticks=6),
                                    )
                                    st.plotly_chart(fig_m, use_container_width=True)

                            except Exception as chart_err:
                                with cols[ci]:
                                    st.warning(f"{p_ticker} 데이터 로드 실패: {chart_err}")
                                continue

                    # 1행: BTC 전략 (일봉 → 4시간봉)
                    render_chart_row(btc_items)
                    # 2행: ETH, SOL 등
                    render_chart_row(other_items)

                    # 시그널 요약 테이블
                    if signal_rows:
                        df_sig = pd.DataFrame(signal_rows)
                        st.dataframe(df_sig, use_container_width=True, hide_index=True)

                # 리밸런싱 규칙 (항상 표시)
                with st.expander("⚖️ 리밸런싱 규칙", expanded=False):
                    st.markdown("""
**실행 시점**: GitHub Action 실행 시마다 (자동: 매일 09:05 KST / 수동 실행 가능)

**실행 순서**: 전체 시그널 분석 → 매도 먼저 실행 (현금 확보) → 현금 비례 배분 매수

**매매 판단** (전일 종가 기준)

| 현재 상태 | 시그널 | 실행 내용 |
|-----------|--------|-----------|
| 코인 미보유 | 매수 시그널 | **매수** — 현금에서 비중 비례 배분 |
| 코인 미보유 | 매도/중립 | **대기** — 현금 보존 (비중만큼 예비) |
| 코인 보유 중 | 매도 시그널 | **매도** — 전량 시장가 매도 |
| 코인 보유 중 | 매수/중립 | **유지** — 계속 보유 (추가 매수 없음) |

**매수 금액 계산**: 보유 중인 자산은 무시, 현금을 미보유 자산 비중끼리 비례 배분

> 예) BTC 40%(보유중), ETH 30%(미보유), SOL 30%(미보유)
> → 미보유 비중 합계 = 60%
> → ETH 매수액 = 현금 × 30/60, SOL 매수액 = 현금 × 30/60

**시그널 발생 조건**

| | 매수 시그널 | 매도 시그널 |
|---|---------|---------|
| **SMA** | 종가 > 이동평균선 | 종가 < 이동평균선 |
| **Donchian** | 종가 > N일 최고가 돌파 | 종가 < M일 최저가 이탈 |
""")

                # 합산 포트폴리오 자리 미리 확보 (데이터 수집 후 렌더링)
                combined_portfolio_container = st.container()

                st.write(f"### 📋 자산 상세 (현금 예비: {reserved_cash:,.0f} KRW)")

                # 포트폴리오 합산용 에쿼티 수집
                portfolio_equity_data = []  # [(label, equity_series, close_series, per_coin_cap, perf)]

                for asset_idx, item in enumerate(portfolio_list):
                    ticker = f"{item['market']}-{item['coin'].upper()}"
                    
                    # Per-Coin Strategy Settings
                    strategy_mode = item.get('strategy', 'SMA Strategy')
                    param_val = item.get('parameter', item.get('sma', 20)) # Backwards compat
                    
                    weight = item.get('weight', 0)
                    interval = item.get('interval', 'day')
                    
                    # Calculate Allocated Capital
                    per_coin_cap = initial_cap * (weight / 100.0)
                    
                    # Collapse by default to save rendering time
                    with st.expander(f"**{ticker}** ({strategy_mode} {param_val}, {weight}%, {interval})", expanded=False):
                        try:
                            # 1. Get Data from Worker
                            df_curr = worker.get_data(ticker, interval)
                            
                            if df_curr is None or len(df_curr) < param_val:
                                st.warning(f"데이터 대기 중... ({ticker}, {interval})")
                                total_theo_val += per_coin_cap 
                                continue
                                
                            # Dynamic Strategy Selection
                            if strategy_mode == "Donchian":
                                strategy_eng = DonchianStrategy()
                                buy_p = param_val
                                sell_p = item.get('sell_parameter', 0) or max(5, buy_p // 2)
                                
                                df_curr = strategy_eng.create_features(df_curr, buy_period=buy_p, sell_period=sell_p)
                                last_candle = df_curr.iloc[-2]
                                
                                # Visuals for Donchian
                                curr_upper = last_candle.get(f'Donchian_Upper_{buy_p}', 0)
                                curr_lower = last_candle.get(f'Donchian_Lower_{sell_p}', 0)
                                curr_sma = (curr_upper + curr_lower) / 2 # Mid for display
                                
                                
                            else: # SMA Strategy (Default)
                                strategy_eng = SMAStrategy()
                                calc_periods = [param_val]
                                    
                                df_curr = strategy_eng.create_features(df_curr, periods=calc_periods)
                                last_candle = df_curr.iloc[-2]
                                
                                curr_sma = last_candle[f'SMA_{param_val}']
                            # 캐시된 가격·잔고 사용 (일괄 조회 결과)
                            curr_price = all_prices.get(ticker, 0) or 0
                            coin_sym = item['coin'].upper()
                            coin_bal = all_balances.get(coin_sym, 0)

                            # 3. Theo Backtest (Sync Check) - 캐시된 백테스트 사용
                            sell_ratio = (item.get('sell_parameter', 0) or max(5, param_val // 2)) / param_val if param_val > 0 else 0.5
                            df_bt = data_cache.load_cached(ticker, interval)
                            if df_bt is not None and len(df_bt) >= param_val:
                                req_count = len(df_bt)
                                df_hash = f"{len(df_bt)}_{df_bt.index[-1]}"
                            else:
                                df_bt = df_curr
                                req_count = len(df_bt)
                                df_hash = f"{len(df_bt)}_{df_bt.index[-1]}"
                            bt_res = _cached_backtest(
                                ticker, param_val, interval, req_count,
                                str(start_date), per_coin_cap, strategy_mode,
                                sell_ratio, df_hash
                            )
                            if bt_res is None:
                                bt_res = backtest_engine.run_backtest(
                                    ticker, period=param_val, interval=interval,
                                    count=req_count, start_date=start_date,
                                    initial_balance=per_coin_cap, df=df_bt,
                                    strategy_mode=strategy_mode,
                                    sell_period_ratio=sell_ratio
                                )
                            
                            expected_eq = 0
                            theo_status = "UNKNOWN"
                            
                            if "error" not in bt_res:
                                perf = bt_res['performance']
                                theo_status = perf['final_status']
                                expected_eq = perf['final_equity']
                                total_theo_val += expected_eq
                                # 합산 포트폴리오용 에쿼티 수집
                                hist_df_tmp = bt_res['df']
                                label = f"{ticker} ({strategy_mode} {param_val}, {interval})"
                                portfolio_equity_data.append({
                                    "label": label,
                                    "equity": hist_df_tmp['equity'],
                                    "close": hist_df_tmp['close'],
                                    "cap": per_coin_cap,
                                    "perf": perf,
                                })
                            else:
                                total_theo_val += per_coin_cap # Fallback if error
                                
                            # 4. Real Status
                            coin_val = coin_bal * curr_price
                            total_real_val += coin_val # Add coin value to total
                            real_status = "HOLD" if coin_val > 5000 else "CASH"
                            
                            # --- Display Metrics ---
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("가격 / SMA", f"{curr_price:,.0f}", delta=f"{curr_price - curr_sma:,.0f}")
                            
                            
                            # Signal Metric Removed as requested
                            # c2.markdown(f"**Signal**: :{sig_color}[{curr_signal}]")
                            if strategy_mode == "Donchian":
                                c2.metric("채널", f"{buy_p}/{sell_p}")
                            else:
                                c2.metric("SMA 기간", f"{param_val}")
                            
                            # Asset Performance
                            roi_theo = (expected_eq - per_coin_cap) / per_coin_cap * 100
                            c3.metric(f"이론 자산", f"{expected_eq:,.0f}", delta=f"{roi_theo:.2f}%")
                            
                            match = (real_status == theo_status)
                            match_color = "green" if match else "red"
                            c4.markdown(f"**동기화**: :{match_color}[{'일치' if match else '불일치'}]")
                            c4.caption(f"실제: {coin_bal:,.4f} {coin_sym} ({real_status})")
                            
                            st.divider()
                            
                            # --- Tabs for Charts & Orders ---
                            p_tab1, p_tab2 = st.tabs(["📈 분석 & 벤치마크", "📋 체결 내역"])

                            with p_tab1:
                                if "error" not in bt_res:
                                    hist_df = bt_res['df']
                                    start_equity = hist_df['equity'].iloc[0]
                                    start_price = hist_df['close'].iloc[0]

                                    # Normalized Comparison
                                    hist_df['Norm_Strat'] = hist_df['equity'] / start_equity * 100
                                    hist_df['Norm_Bench'] = hist_df['close'] / start_price * 100

                                    fig_comp = go.Figure()
                                    fig_comp.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Norm_Strat'], name='전략', line=dict(color='blue')))
                                    fig_comp.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Norm_Bench'], name='벤치마크', line=dict(color='gray', dash='dot')))

                                    # 매수/매도 마커 추가
                                    if perf.get('trades'):
                                        buy_trades = [t for t in perf['trades'] if t['type'] == 'buy']
                                        sell_trades = [t for t in perf['trades'] if t['type'] == 'sell']
                                        if buy_trades:
                                            buy_dates = [t['date'] for t in buy_trades]
                                            buy_vals = [hist_df.loc[d, 'Norm_Strat'] if d in hist_df.index else None for d in buy_dates]
                                            fig_comp.add_trace(go.Scatter(
                                                x=buy_dates, y=buy_vals, mode='markers', name='매수',
                                                marker=dict(symbol='triangle-up', size=10, color='green')
                                            ))
                                        if sell_trades:
                                            sell_dates = [t['date'] for t in sell_trades]
                                            sell_vals = [hist_df.loc[d, 'Norm_Strat'] if d in hist_df.index else None for d in sell_dates]
                                            fig_comp.add_trace(go.Scatter(
                                                x=sell_dates, y=sell_vals, mode='markers', name='매도',
                                                marker=dict(symbol='triangle-down', size=10, color='red')
                                            ))

                                    fig_comp.update_layout(height=300, title="전략 vs 단순보유 (정규화)", margin=dict(l=0,r=0,t=80,b=0),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                                    st.plotly_chart(fig_comp, use_container_width=True)

                                    # 연도별 성과 테이블
                                    if 'yearly_stats' in perf:
                                        st.caption("📅 연도별 성과")
                                        ys = perf['yearly_stats'].copy()
                                        ys.index.name = "연도"
                                        st.dataframe(ys.style.format("{:.2f}"), use_container_width=True)

                                    _render_performance_analysis(
                                        equity_series=hist_df.get("equity"),
                                        benchmark_series=hist_df.get("close"),
                                        strategy_metrics=perf,
                                        strategy_label=f"{strategy_mode} 전략",
                                        benchmark_label=f"{ticker} 단순보유",
                                        monte_carlo_sims=220,
                                    )
                            
                            with p_tab2:
                                # ── 가상(백테스트) 체결 내역 ──
                                st.markdown("**가상 체결 (백테스트)**")
                                if "error" not in bt_res and perf.get('trades'):
                                    vt_rows = []
                                    for t in perf['trades']:
                                        t_date = t.get('date', '')
                                        if hasattr(t_date, 'strftime'):
                                            t_date = t_date.strftime('%Y-%m-%d')
                                        t_type = t.get('type', '')
                                        t_side = "매수" if t_type == 'buy' else ("매도" if t_type == 'sell' else t_type)
                                        t_price = t.get('price', 0)
                                        t_amount = t.get('amount', 0)
                                        t_equity = t.get('equity', 0)
                                        vt_rows.append({
                                            "일시": t_date,
                                            "구분": f"🔴 {t_side}" if t_type == 'buy' else f"🔵 {t_side}",
                                            "체결가": f"{t_price:,.0f}",
                                            "수량": f"{t_amount:.6f}" if t_amount < 1 else f"{t_amount:,.4f}",
                                            "자산(KRW)": f"{t_equity:,.0f}",
                                        })
                                    if vt_rows:
                                        st.dataframe(pd.DataFrame(vt_rows[-20:]), use_container_width=True, hide_index=True)
                                        st.caption(f"최근 {min(20, len(vt_rows))}건 / 총 {len(vt_rows)}건")
                                    else:
                                        st.info("백테스트 체결 기록 없음")
                                else:
                                    st.info("백테스트 데이터 없음")

                                # ── 실제 체결 내역 (거래소) ──
                                st.markdown("**실제 체결 (거래소)**")
                                try:
                                    done = _ttl_cache(
                                        f"done_{ticker}",
                                        lambda t=ticker: trader.get_done_orders(t),
                                        ttl=30
                                    )
                                    if done:
                                        rt_rows = []
                                        for r in done[:20]:
                                            side = r.get('side', '')
                                            side_kr = "매수" if side == 'bid' else ("매도" if side == 'ask' else side)
                                            price_r = float(r.get('price', 0) or 0)
                                            exec_vol = float(r.get('executed_volume', 0) or 0)
                                            if price_r > 0 and exec_vol > 0:
                                                total_k = price_r * exec_vol
                                            elif 'trades' in r and r['trades']:
                                                total_k = sum(float(tr.get('funds', 0)) for tr in r['trades'])
                                            else:
                                                total_k = price_r
                                            created = r.get('created_at', '')
                                            if pd.notna(created):
                                                try: created = pd.to_datetime(created).strftime('%Y-%m-%d %H:%M')
                                                except: pass
                                            rt_rows.append({
                                                "일시": created,
                                                "구분": f"🔴 {side_kr}" if side == 'bid' else f"🔵 {side_kr}",
                                                "체결가": f"{price_r:,.0f}" if price_r > 0 else "-",
                                                "수량": f"{exec_vol:.6f}" if exec_vol < 1 else f"{exec_vol:,.4f}",
                                                "금액(KRW)": f"{total_k:,.0f}",
                                            })
                                        if rt_rows:
                                            st.dataframe(pd.DataFrame(rt_rows), use_container_width=True, hide_index=True)
                                        else:
                                            st.info("체결 완료 주문 없음")
                                    else:
                                        st.info("체결 완료 주문 없음")
                                except Exception:
                                    st.info("체결 내역 조회 불가 (API 권한 확인)")

                        except Exception as e:
                            st.error(f"{ticker} 처리 오류: {e}")
                
                # --- Populate Total Summary ---
                total_roi = (total_theo_val - total_init_val) / total_init_val * 100 if total_init_val else 0
                real_roi = (total_real_val - total_init_val) / total_init_val * 100 if total_init_val else 0
                diff_val = total_real_val - total_theo_val

                sum_col1.metric("초기 자본", f"{total_init_val:,.0f} KRW")
                sum_col2.metric("이론 총자산", f"{total_theo_val:,.0f} KRW", delta=f"{total_roi:.2f}%")
                sum_col3.metric("실제 총자산", f"{total_real_val:,.0f} KRW", delta=f"{real_roi:.2f}%")
                sum_col4.metric("차이 (실제-이론)", f"{diff_val:,.0f} KRW", delta_color="off" if abs(diff_val)<1000 else "inverse")

                # --- 합산 포트폴리오 성과 (Combined Portfolio) → 위에 예약한 container에 렌더링 ---
                if portfolio_equity_data:
                    with combined_portfolio_container:
                        with st.expander("📊 합산 포트폴리오 성과", expanded=True):

                            # 각 자산의 에쿼티를 일자 기준으로 합산
                            equity_dfs = []
                            bench_dfs = []
                            for ed in portfolio_equity_data:
                                eq = ed['equity'].copy()
                                cl = ed['close'].copy()
                                cap = ed['cap']

                                if hasattr(eq.index, 'tz') and eq.index.tz is not None:
                                    eq.index = eq.index.tz_localize(None)
                                    cl.index = cl.index.tz_localize(None)
                                eq_daily = eq.resample('D').last().dropna()
                                cl_daily = cl.resample('D').last().dropna()

                                bench_daily = (cl_daily / cl_daily.iloc[0]) * cap

                                eq_daily.name = ed['label']
                                bench_daily.name = ed['label']
                                equity_dfs.append(eq_daily)
                                bench_dfs.append(bench_daily)

                            combined_eq = pd.concat(equity_dfs, axis=1).sort_index()
                            combined_bench = pd.concat(bench_dfs, axis=1).sort_index()

                            combined_eq = combined_eq.ffill().bfill()
                            combined_bench = combined_bench.ffill().bfill()

                            combined_eq['cash_reserve'] = reserved_cash
                            combined_bench['cash_reserve'] = reserved_cash

                            total_eq = combined_eq.sum(axis=1)
                            total_bench = combined_bench.sum(axis=1)

                            norm_eq = total_eq / total_eq.iloc[0] * 100
                            norm_bench = total_bench / total_bench.iloc[0] * 100

                            # 성과 지표 계산
                            port_final = total_eq.iloc[-1]
                            port_init = total_eq.iloc[0]
                            port_return = (port_final - port_init) / port_init * 100

                            port_days = (total_eq.index[-1] - total_eq.index[0]).days
                            port_cagr = 0
                            if port_days > 0 and port_final > 0:
                                port_cagr = ((port_final / port_init) ** (365 / port_days) - 1) * 100

                            port_peak = total_eq.cummax()
                            port_dd = (total_eq - port_peak) / port_peak * 100
                            port_mdd = port_dd.min()

                            port_returns = total_eq.pct_change().dropna()
                            port_sharpe = 0
                            if port_returns.std() > 0:
                                port_sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(365)

                            bench_final = total_bench.iloc[-1]
                            bench_init = total_bench.iloc[0]
                            bench_return = (bench_final - bench_init) / bench_init * 100

                            # 메트릭 표시
                            pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                            pm1.metric("총 수익률", f"{port_return:.2f}%")
                            pm2.metric("CAGR", f"{port_cagr:.2f}%")
                            pm3.metric("MDD", f"{port_mdd:.2f}%")
                            pm4.metric("Sharpe", f"{port_sharpe:.2f}")
                            pm5.metric("vs 단순보유", f"{port_return - bench_return:+.2f}%p")

                            st.caption(f"기간: {total_eq.index[0].strftime('%Y-%m-%d')} ~ {total_eq.index[-1].strftime('%Y-%m-%d')} ({port_days}일) | 초기자금: {port_init:,.0f} → 최종: {port_final:,.0f} KRW")

                            # 합산 차트
                            fig_port = go.Figure()
                            fig_port.add_trace(go.Scatter(
                                x=norm_eq.index, y=norm_eq.values,
                                name='포트폴리오 (전략)', line=dict(color='blue', width=2)
                            ))
                            fig_port.add_trace(go.Scatter(
                                x=norm_bench.index, y=norm_bench.values,
                                name='포트폴리오 (단순보유)', line=dict(color='gray', dash='dot')
                            ))

                            # 합산 차트에 매수/매도 마커 표시
                            all_buy_dates = []
                            all_sell_dates = []
                            for ed in portfolio_equity_data:
                                for t in ed['perf'].get('trades', []):
                                    if t['type'] == 'buy':
                                        all_buy_dates.append(t['date'])
                                    elif t['type'] == 'sell':
                                        all_sell_dates.append(t['date'])

                            if all_buy_dates:
                                # 날짜를 norm_eq 인덱스와 매칭 (일봉 리샘플링 됐으므로 가장 가까운 날짜 사용)
                                buy_vals = []
                                buy_dates_valid = []
                                for d in all_buy_dates:
                                    d_ts = pd.Timestamp(d)
                                    if hasattr(d_ts, 'tz') and d_ts.tz is not None:
                                        d_ts = d_ts.tz_localize(None)
                                    idx = norm_eq.index.get_indexer([d_ts], method='nearest')
                                    if idx[0] >= 0:
                                        buy_dates_valid.append(norm_eq.index[idx[0]])
                                        buy_vals.append(norm_eq.iloc[idx[0]])
                                if buy_dates_valid:
                                    fig_port.add_trace(go.Scatter(
                                        x=buy_dates_valid, y=buy_vals, mode='markers', name='매수',
                                        marker=dict(symbol='triangle-up', size=8, color='green', opacity=0.7)
                                    ))

                            if all_sell_dates:
                                sell_vals = []
                                sell_dates_valid = []
                                for d in all_sell_dates:
                                    d_ts = pd.Timestamp(d)
                                    if hasattr(d_ts, 'tz') and d_ts.tz is not None:
                                        d_ts = d_ts.tz_localize(None)
                                    idx = norm_eq.index.get_indexer([d_ts], method='nearest')
                                    if idx[0] >= 0:
                                        sell_dates_valid.append(norm_eq.index[idx[0]])
                                        sell_vals.append(norm_eq.iloc[idx[0]])
                                if sell_dates_valid:
                                    fig_port.add_trace(go.Scatter(
                                        x=sell_dates_valid, y=sell_vals, mode='markers', name='매도',
                                        marker=dict(symbol='triangle-down', size=8, color='red', opacity=0.7)
                                    ))

                            fig_port.update_layout(
                                height=350,
                                title="합산 포트폴리오: 전략 vs 단순보유 (정규화)",
                                yaxis_title="정규화 (%)",
                                margin=dict(l=0, r=0, t=80, b=0),
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            fig_port = _apply_return_hover_format(fig_port, apply_all=True)
                            st.plotly_chart(fig_port, use_container_width=True)

                            # 포트폴리오 MDD(Drawdown) 차트 추가
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=port_dd.index, y=port_dd.values,
                                name='낙폭', fill='tozeroy',
                                line=dict(color='red', width=1)
                            ))
                            fig_dd.update_layout(
                                height=200,
                                title="포트폴리오 낙폭 (%)",
                                yaxis_title="낙폭 (%)",
                                margin=dict(l=0, r=0, t=80, b=0),
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            fig_dd = _apply_dd_hover_format(fig_dd)
                            st.plotly_chart(fig_dd, use_container_width=True)

                            # 개별 자산 에쿼티 기여도 차트
                            fig_stack = go.Figure()
                            for ed in portfolio_equity_data:
                                eq = ed['equity'].copy()
                                if hasattr(eq.index, 'tz') and eq.index.tz is not None:
                                    eq.index = eq.index.tz_localize(None)
                                eq_d = eq.resample('D').last().dropna()
                                fig_stack.add_trace(go.Scatter(
                                    x=eq_d.index, y=eq_d.values,
                                    name=ed['label'], stackgroup='one'
                                ))
                            if reserved_cash > 0:
                                fig_stack.add_trace(go.Scatter(
                                    x=total_eq.index, y=[reserved_cash] * len(total_eq),
                                    name='현금 예비', stackgroup='one',
                                    line=dict(color='lightgray')
                                ))
                            fig_stack.update_layout(
                                height=350,
                                title="자산별 기여도 (적층)",
                                yaxis_title="KRW",
                                margin=dict(l=0, r=0, t=80, b=0),
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            st.plotly_chart(fig_stack, use_container_width=True)

                            # 개별 자산 성과 테이블
                            asset_perf_rows = []
                            for ed in portfolio_equity_data:
                                p = ed['perf']
                                asset_perf_rows.append({
                                    "자산": ed['label'],
                                    "배분자본": f"{ed['cap']:,.0f}",
                                    "최종자산": f"{p['final_equity']:,.0f}",
                                    "수익률(%)": f"{p['total_return']:.2f}",
                                    "CAGR(%)": f"{p['cagr']:.2f}",
                                    "MDD(%)": f"{p['mdd']:.2f}",
                                    "승률(%)": f"{p['win_rate']:.1f}",
                                    "거래수": p['trade_count'],
                                    "Sharpe": f"{p['sharpe']:.2f}",
                                    "상태": p['final_status'],
                                })
                            st.dataframe(pd.DataFrame(asset_perf_rows), use_container_width=True, hide_index=True)

                            # 📅 합산 포트폴리오 연도별 성과 테이블
                            st.caption("📅 합산 포트폴리오 연도별 성과")
                            port_daily_ret = total_eq.pct_change().fillna(0)
                            port_year = total_eq.index.year
                            port_dd_series = port_dd

                            yearly_rows = []
                            for yr in sorted(port_year.unique()):
                                yr_mask = port_year == yr
                                yr_ret = (1 + port_daily_ret[yr_mask]).prod() - 1
                                yr_mdd = port_dd_series[yr_mask].min()
                                yr_eq_start = total_eq[yr_mask].iloc[0]
                                yr_eq_end = total_eq[yr_mask].iloc[-1]

                                # 벤치마크 연도별
                                yr_bench_start = total_bench[yr_mask].iloc[0]
                                yr_bench_end = total_bench[yr_mask].iloc[-1]
                                yr_bench_ret = (yr_bench_end - yr_bench_start) / yr_bench_start * 100

                                yearly_rows.append({
                                    "연도": yr,
                                    "수익률(%)": f"{yr_ret * 100:.2f}",
                                    "MDD(%)": f"{yr_mdd:.2f}",
                                    "시작자산": f"{yr_eq_start:,.0f}",
                                    "최종자산": f"{yr_eq_end:,.0f}",
                                    "Buy&Hold(%)": f"{yr_bench_ret:.2f}",
                                    "초과수익(%p)": f"{yr_ret * 100 - yr_bench_ret:.2f}",
                                })
                            st.dataframe(pd.DataFrame(yearly_rows), use_container_width=True, hide_index=True)

    # --- Tab 5: Manual Trade (거래소 스타일) ---
    with tab5:
        st.header("수동 주문")

        if not trader:
            st.warning("사이드바에서 API 키를 설정해주세요.")
        else:
            # 코인 선택 (변경시만 full rerun)
            port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
            manual_options = list(dict.fromkeys(port_tickers + TOP_20_TICKERS))
            mt_col1, mt_col2 = st.columns(2)
            mt_selected = mt_col1.selectbox("코인 선택", manual_options + ["직접입력"], key="mt_ticker")
            if mt_selected == "직접입력":
                mt_custom = mt_col2.text_input("코인 심볼", "BTC", key="mt_custom")
                mt_ticker = f"KRW-{mt_custom.upper()}"
            else:
                mt_ticker = mt_selected
                mt_col2.empty()

            mt_coin = mt_ticker.split("-")[1] if "-" in mt_ticker else mt_ticker

            # ── 코인 트레이딩 워커 시작 (백그라운드 갱신) ──
            from src.engine.data_manager import CoinTradingWorker

            @st.cache_resource
            def _get_coin_worker(_trader):
                w = CoinTradingWorker()
                return w

            coin_worker = _get_coin_worker(trader)
            coin_worker.configure(trader, mt_ticker)
            coin_worker.start()

            # ═══ 트레이딩 패널 (fragment → 3초마다 자동갱신, 워커에서 읽기만) ═══
            @st.fragment
            def trading_panel():
                # ── 워커에서 즉시 읽기 (API 호출 없음 → 블로킹 없음) ──
                mt_price = coin_worker.get('price', 0)
                krw_avail = coin_worker.get('krw_bal', 0)
                mt_coin_bal = coin_worker.get('coin_bal', 0)
                mt_coin_val = mt_coin_bal * mt_price
                mt_tick = get_tick_size(mt_price) if mt_price > 0 else 1
                mt_min_qty = round(5000 / mt_price, 8) if mt_price > 0 else 0.00000001

                # 상단 정보 바
                ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                ic1.metric("현재가", f"{mt_price:,.0f}")
                ic2.metric(f"{mt_coin} 보유", f"{mt_coin_bal:.8f}" if mt_coin_bal < 1 else f"{mt_coin_bal:,.4f}")
                ic3.metric("평가금액", f"{mt_coin_val:,.0f} KRW")
                ic4.metric("보유 KRW", f"{krw_avail:,.0f}")
                ic5.metric("호가단위", f"{mt_tick:,g}원" if mt_tick >= 1 else f"{mt_tick}원")

                # ── 최근 거래 결과 알림 바 (세션 유지) ──
                last_trade = st.session_state.get('_last_trade')
                if last_trade:
                    t_type = last_trade.get('type', '')
                    t_time = last_trade.get('time', '')
                    t_ticker = last_trade.get('ticker', '')
                    t_amt = last_trade.get('amount', '')
                    t_price = last_trade.get('price', '')
                    t_qty = last_trade.get('qty', '')
                    is_buy = '매수' in t_type
                    color = '#D32F2F' if is_buy else '#1976D2'
                    detail = t_amt if t_amt else f"{t_price} x {t_qty}"
                    nc1, nc2 = st.columns([6, 1])
                    nc1.markdown(
                        f'<div style="padding:6px 12px;border-radius:6px;background:{color}22;border-left:4px solid {color};font-size:14px;">'
                        f'<b style="color:{color}">{t_type}</b> {t_ticker} | {detail} | {t_time}</div>',
                        unsafe_allow_html=True
                    )
                    if nc2.button("✕", key="_dismiss_trade"):
                        del st.session_state['_last_trade']
                        st.rerun()

                st.divider()

                # ═══ 레이아웃: 30분봉 차트(상단 전체폭) + 주문가격/주문 패널(하단) ═══
                st.markdown("**30분봉 차트**")
                df_30m = _ttl_cache(
                    f"m30_{mt_ticker}",
                    lambda: data_cache.get_ohlcv_local_first(
                        mt_ticker,
                        interval="minute30",
                        count=48,
                        allow_api_fallback=True,
                    ),
                    ttl=30,
                )
                if df_30m is not None and len(df_30m) > 0:
                    fig_30m = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
                    fig_30m.add_trace(go.Candlestick(
                        x=df_30m.index, open=df_30m['open'], high=df_30m['high'],
                        low=df_30m['low'], close=df_30m['close'], name='30분봉',
                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                    ), row=1, col=1)
                    ma5 = df_30m['close'].rolling(5).mean()
                    ma20 = df_30m['close'].rolling(20).mean()
                    fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma5, name='MA5', line=dict(color='#FF9800', width=1)), row=1, col=1)
                    fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma20, name='MA20', line=dict(color='#2196F3', width=1)), row=1, col=1)
                    colors_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_30m['close'], df_30m['open'])]
                    fig_30m.add_trace(go.Bar(
                        x=df_30m.index, y=df_30m['volume'], marker_color=colors_vol, name='거래량', showlegend=False
                    ), row=2, col=1)
                    fig_30m.update_layout(
                        height=520, margin=dict(l=0, r=0, t=10, b=30),
                        xaxis_rangeslider_visible=False, showlegend=True,
                        legend=dict(orientation="h", y=1.06, x=0),
                        xaxis2=dict(showticklabels=True, tickformat='%H:%M', tickangle=-45),
                        yaxis=dict(title="", side="right"),
                        yaxis2=dict(title="", side="right"),
                    )
                    st.plotly_chart(fig_30m, use_container_width=True, key=f"chart30m_{mt_ticker}")
                else:
                    st.info("차트 데이터 로딩 중...")

                st.divider()

                # ═══ 좌: 호가창 | 우: 주문 패널 (가로 배치 — 골드와 동일 구조) ═══
                ob_col, order_col = st.columns([2, 3])

                # ── 좌: 호가창 (HTML 렌더링) ──
                with ob_col:
                    st.markdown("**주문가격 표**")
                    raw_prices = []
                    try:
                        ob_data = coin_worker.get('orderbook')
                        if ob_data and len(ob_data) > 0:
                            ob = ob_data[0] if isinstance(ob_data, list) else ob_data
                            units = ob.get('orderbook_units', [])[:10]

                            if units:
                                max_size = max(
                                    max(u.get('ask_size', 0) for u in units),
                                    max(u.get('bid_size', 0) for u in units)
                                )

                                # ── HTML 호가창 테이블 생성 ──
                                html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                                html.append('<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666"><td>구분</td><td style="text-align:right">잔량</td><td style="text-align:right">가격</td><td style="text-align:right">등락</td><td>비율</td></tr>')

                                ask_prices = []
                                bid_prices = []

                                for u in reversed(units):
                                    ask_p = u.get('ask_price', 0)
                                    ask_s = u.get('ask_size', 0)
                                    diff = ((ask_p / mt_price) - 1) * 100 if mt_price > 0 else 0
                                    bar_w = int(ask_s / max_size * 100) if max_size > 0 else 0
                                    raw_prices.append(ask_p)
                                    ask_prices.append(ask_p)
                                    html.append(
                                        f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                                        f'<td>매도</td>'
                                        f'<td style="text-align:right">{ask_s:.4f}</td>'
                                        f'<td style="text-align:right;font-weight:bold">{ask_p:,.0f}</td>'
                                        f'<td style="text-align:right">{diff:+.2f}%</td>'
                                        f'<td><div style="background:#1976D2;height:12px;width:{bar_w}%;opacity:0.3"></div></td>'
                                        f'</tr>'
                                    )

                                # 중간 구분선
                                html.append('<tr style="border-top:3px solid #333;border-bottom:3px solid #333;height:4px"><td colspan="5"></td></tr>')

                                for u in units:
                                    bid_p = u.get('bid_price', 0)
                                    bid_s = u.get('bid_size', 0)
                                    diff = ((bid_p / mt_price) - 1) * 100 if mt_price > 0 else 0
                                    bar_w = int(bid_s / max_size * 100) if max_size > 0 else 0
                                    raw_prices.append(bid_p)
                                    bid_prices.append(bid_p)
                                    html.append(
                                        f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                                        f'<td>매수</td>'
                                        f'<td style="text-align:right">{bid_s:.4f}</td>'
                                        f'<td style="text-align:right;font-weight:bold">{bid_p:,.0f}</td>'
                                        f'<td style="text-align:right">{diff:+.2f}%</td>'
                                        f'<td><div style="background:#D32F2F;height:12px;width:{bar_w}%;opacity:0.3"></div></td>'
                                        f'</tr>'
                                    )

                                html.append('</table>')
                                st.markdown(''.join(html), unsafe_allow_html=True)

                                # ── 호가 선택 → 주문가 반영 ──
                                def _on_ob_select():
                                    """사용자가 직접 선택했을 때만 주문가에 반영"""
                                    sel_label = st.session_state.get('_ob_sel_label', '')
                                    try:
                                        price_str = sel_label.split(' ', 1)[1].replace(',', '')
                                        chosen = int(float(price_str))
                                        tick = get_tick_size(chosen)
                                        if tick >= 1:
                                            st.session_state['mt_buy_price'] = int(chosen)
                                            st.session_state['mt_sell_price'] = int(chosen)
                                        else:
                                            st.session_state['mt_buy_price'] = float(chosen)
                                            st.session_state['mt_sell_price'] = float(chosen)
                                    except (IndexError, ValueError):
                                        pass

                                price_labels = (
                                    [f"매도 {p:,.0f}" for p in ask_prices] +
                                    [f"매수 {p:,.0f}" for p in bid_prices]
                                )

                                st.selectbox(
                                    "호가 선택 → 주문가 반영",
                                    price_labels,
                                    index=len(ask_prices),  # 기본: 최우선 매수호가
                                    key="_ob_sel_label",
                                    on_change=_on_ob_select,
                                )

                                best_ask = units[0].get('ask_price', 0)
                                best_bid = units[0].get('bid_price', 0)
                                spread = best_ask - best_bid
                                spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
                                total_ask = ob.get('total_ask_size', 0)
                                total_bid = ob.get('total_bid_size', 0)
                                ob_ratio = total_bid / (total_ask + total_bid) * 100 if (total_ask + total_bid) > 0 else 50
                                st.caption(f"스프레드 **{spread:,.0f}** ({spread_pct:.3f}%) | 매도 {total_ask:.2f} | 매수 {total_bid:.2f} | 매수비율 {ob_ratio:.0f}%")
                            else:
                                st.info("호가 데이터가 없습니다.")
                        else:
                            st.info("호가 데이터를 불러올 수 없습니다.")
                    except Exception as e:
                        st.warning(f"호가 조회 실패: {e}")

                # ── 우: 주문 패널 ──
                with order_col:
                    st.markdown("**주문 실행**")
                    buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                    with buy_tab:
                        buy_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_buy_type")

                        if buy_type == "시장가":
                            buy_amount = st.number_input("매수 금액 (KRW)", min_value=5000, value=10000, step=1000, key="mt_buy_amt")
                            qb1, qb2, qb3, qb4 = st.columns(4)
                            if qb1.button("10%", key="mt_b10"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.1)
                                st.rerun()
                            if qb2.button("25%", key="mt_b25"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.25)
                                st.rerun()
                            if qb3.button("50%", key="mt_b50"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.5)
                                st.rerun()
                            if qb4.button("100%", key="mt_b100"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.999)
                                st.rerun()

                            if mt_price > 0:
                                st.caption(f"예상 수량: ~{buy_amount / mt_price:.8f} {mt_coin}")

                            if st.button("시장가 매수", type="primary", key="mt_buy_exec", use_container_width=True):
                                if buy_amount < 5000:
                                    st.toast("최소 주문금액: 5,000 KRW", icon="⚠️")
                                elif buy_amount > krw_avail:
                                    st.toast(f"잔고 부족 ({krw_avail:,.0f} KRW)", icon="⚠️")
                                else:
                                    with st.spinner("매수 주문 중..."):
                                        result = trader.buy_market(mt_ticker, buy_amount)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 시장가 매수 체결! {buy_amount:,.0f} KRW", icon="🟢")
                                        st.session_state['_last_trade'] = {"type": "시장가 매수", "ticker": mt_ticker, "amount": f"{buy_amount:,.0f} KRW", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"매수 실패: {result}", icon="🔴")

                        else:  # 지정가
                            buy_default = align_price(mt_price * 0.99, mt_tick) if mt_price > 0 else 1
                            bc1, bc2 = st.columns(2)
                            if mt_tick >= 1:
                                buy_price = bc1.number_input("매수 가격", min_value=1, value=int(buy_default), step=int(mt_tick), key="mt_buy_price")
                            else:
                                buy_price = bc1.number_input("매수 가격", min_value=0.0001, value=float(buy_default), step=float(mt_tick), format="%.4f", key="mt_buy_price")
                            buy_qty = bc2.number_input("매수 수량", min_value=mt_min_qty, value=max(mt_min_qty, 0.001), format="%.8f", key="mt_buy_qty")

                            buy_total = buy_price * buy_qty
                            st.caption(f"총액: **{buy_total:,.0f} KRW** | 호가: {mt_tick:,g}원 | 최소: {mt_min_qty:.8f}")

                            qbc1, qbc2, qbc3, qbc4 = st.columns(4)
                            if buy_price > 0:
                                if qbc1.button("10%", key="mt_lb10"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.1 / buy_price, 8)
                                    st.rerun()
                                if qbc2.button("25%", key="mt_lb25"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.25 / buy_price, 8)
                                    st.rerun()
                                if qbc3.button("50%", key="mt_lb50"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.5 / buy_price, 8)
                                    st.rerun()
                                if qbc4.button("100%", key="mt_lb100"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.999 / buy_price, 8)
                                    st.rerun()

                            if st.button("지정가 매수", type="primary", key="mt_lbuy_exec", use_container_width=True):
                                if buy_total < 5000:
                                    st.toast("최소 주문금액: 5,000 KRW", icon="⚠️")
                                elif buy_total > krw_avail:
                                    st.toast(f"잔고 부족 ({krw_avail:,.0f} KRW)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매수 주문 중..."):
                                        result = trader.buy_limit(mt_ticker, buy_price, buy_qty)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 지정가 매수 등록! {buy_price:,.0f} × {buy_qty:.8f}", icon="🟢")
                                        st.session_state['_last_trade'] = {"type": "지정가 매수", "ticker": mt_ticker, "price": f"{buy_price:,.0f}", "qty": f"{buy_qty:.8f}", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

                    with sell_tab:
                        sell_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_sell_type")

                        if sell_type == "시장가":
                            sell_qty = st.number_input(
                                f"매도 수량 ({mt_coin})", min_value=0.00000001,
                                value=mt_coin_bal if mt_coin_bal > 0 else 0.00000001,
                                format="%.8f", key="mt_sell_qty"
                            )
                            qs1, qs2, qs3, qs4 = st.columns(4)
                            if qs1.button("25%", key="mt_s25"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal * 0.25, 8)
                                st.rerun()
                            if qs2.button("50%", key="mt_s50"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal * 0.5, 8)
                                st.rerun()
                            if qs3.button("75%", key="mt_s75"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal * 0.75, 8)
                                st.rerun()
                            if qs4.button("100%", key="mt_s100"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal, 8)
                                st.rerun()

                            if mt_price > 0:
                                st.caption(f"예상 금액: ~{sell_qty * mt_price:,.0f} KRW")

                            if st.button("시장가 매도", type="primary", key="mt_sell_exec", use_container_width=True):
                                if sell_qty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif mt_price > 0 and sell_qty * mt_price < 5000:
                                    st.toast(f"최소 주문금액 미달 ({sell_qty * mt_price:,.0f} KRW < 5,000)", icon="⚠️")
                                elif sell_qty > mt_coin_bal:
                                    st.toast(f"보유량 초과 ({mt_coin_bal:.8f})", icon="⚠️")
                                else:
                                    with st.spinner("매도 주문 중..."):
                                        result = trader.sell_market(mt_ticker, sell_qty)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 시장가 매도 체결! {sell_qty:.8f} {mt_coin}", icon="🔴")
                                        st.session_state['_last_trade'] = {"type": "시장가 매도", "ticker": mt_ticker, "qty": f"{sell_qty:.8f}", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"매도 실패: {result}", icon="🔴")

                        else:  # 지정가
                            sell_default = align_price(mt_price * 1.01, mt_tick) if mt_price > 0 else 1
                            sc1, sc2 = st.columns(2)
                            if mt_tick >= 1:
                                sell_price = sc1.number_input("매도 가격", min_value=1, value=int(sell_default), step=int(mt_tick), key="mt_sell_price")
                            else:
                                sell_price = sc1.number_input("매도 가격", min_value=0.0001, value=float(sell_default), step=float(mt_tick), format="%.4f", key="mt_sell_price")
                            sell_default_qty = mt_coin_bal if mt_coin_bal > mt_min_qty else mt_min_qty
                            sell_limit_qty = sc2.number_input("매도 수량", min_value=mt_min_qty, value=sell_default_qty, format="%.8f", key="mt_sell_lqty")

                            sell_total = sell_price * sell_limit_qty
                            st.caption(f"총액: **{sell_total:,.0f} KRW** | 호가: {mt_tick:,g}원 | 최소: {mt_min_qty:.8f}")

                            qsc1, qsc2, qsc3, qsc4 = st.columns(4)
                            if mt_coin_bal > 0:
                                if qsc1.button("25%", key="mt_ls25"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal * 0.25, 8)
                                    st.rerun()
                                if qsc2.button("50%", key="mt_ls50"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal * 0.5, 8)
                                    st.rerun()
                                if qsc3.button("75%", key="mt_ls75"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal * 0.75, 8)
                                    st.rerun()
                                if qsc4.button("100%", key="mt_ls100"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal, 8)
                                    st.rerun()

                            if st.button("지정가 매도", type="primary", key="mt_lsell_exec", use_container_width=True):
                                if sell_limit_qty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif sell_limit_qty > mt_coin_bal:
                                    st.toast(f"보유량 초과 ({mt_coin_bal:.8f})", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매도 주문 중..."):
                                        result = trader.sell_limit(mt_ticker, sell_price, sell_limit_qty)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 지정가 매도 등록! {sell_price:,.0f} × {sell_limit_qty:.8f}", icon="🔴")
                                        st.session_state['_last_trade'] = {"type": "지정가 매도", "ticker": mt_ticker, "price": f"{sell_price:,.0f}", "qty": f"{sell_limit_qty:.8f}", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

                # ── 미체결 주문 ──
                st.divider()
                st.subheader("미체결 주문")
                if st.button("미체결 주문 조회", key="mt_pending"):
                    with st.spinner("조회 중..."):
                        pending = trader.get_orders(state="wait")
                    if pending and len(pending) > 0:
                        for i, order in enumerate(pending):
                            side_kr = "매수" if order.get('side') == 'bid' else "매도"
                            side_color = "red" if order.get('side') == 'bid' else "blue"
                            market = order.get('market', '')
                            price = float(order.get('price', 0) or 0)
                            remaining = float(order.get('remaining_volume', 0) or 0)
                            created = order.get('created_at', '')
                            if pd.notna(created):
                                try:
                                    created = pd.to_datetime(created).strftime('%m/%d %H:%M')
                                except:
                                    pass
                            oc1, oc2 = st.columns([4, 1])
                            oc1.markdown(f"**:{side_color}[{side_kr}]** {market} | {price:,.0f} × {remaining:.8f} | {created}")
                            if oc2.button("취소", key=f"mt_cancel_{i}"):
                                cancel_result = trader.cancel_order(order.get('uuid'))
                                if cancel_result and "error" not in cancel_result:
                                    st.toast("주문 취소 완료", icon="✅")
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    st.rerun()
                                else:
                                    st.toast(f"취소 실패: {cancel_result}", icon="🔴")
                    else:
                        st.info("미체결 주문이 없습니다.")

            trading_panel()

    # --- Tab 3: History ---
    with tab3:
        st.header("거래 내역")

        hist_tab1, hist_tab2, hist_tab3 = st.tabs(["💸 실제 거래 내역 (거래소)", "🧪 가상 로그 (페이퍼)", "📊 슬리피지 분석"])

        with hist_tab1:
            st.subheader("실제 거래 내역")

            if not trader:
                st.warning("사이드바에서 API 키를 설정해주세요.")
            else:
                c_h1, c_h2 = st.columns(2)
                h_type = c_h1.selectbox("조회 유형", ["전체", "입금", "출금", "체결 주문"])
                h_curr = c_h2.selectbox("화폐", ["전체", "KRW", "BTC", "ETH", "XRP", "SOL", "USDT", "DOGE", "ADA", "AVAX", "LINK"])

                d_h1, d_h2 = st.columns(2)
                h_date_start = d_h1.date_input("조회 시작일", value=datetime.now().date() - timedelta(days=90), key="hist_start")
                h_date_end = d_h2.date_input("조회 종료일", value=datetime.now().date(), key="hist_end")

                if st.button("조회"):
                    with st.spinner("Upbit API 조회 중..."):
                        api_curr = None if h_curr == "전체" else h_curr

                        # ── 조회 유형별 데이터 수집 ──
                        def _parse_deposit_withdraw(raw, type_label):
                            """입금/출금 데이터를 통합 포맷으로 변환"""
                            rows = []
                            for r in raw:
                                done = r.get('done_at', r.get('created_at', ''))
                                if pd.notna(done):
                                    try: done = pd.to_datetime(done).strftime('%Y-%m-%d %H:%M')
                                    except: pass
                                amount = float(r.get('amount', 0))
                                fee_val = float(r.get('fee', 0))
                                state = r.get('state', '')
                                state_kr = {"ACCEPTED": "완료", "REJECTED": "거부", "CANCELLED": "취소", "PROCESSING": "처리중", "WAITING": "대기중"}.get(state, state)
                                rows.append({
                                    "거래일시": done, "유형": type_label,
                                    "화폐/코인": r.get('currency', ''),
                                    "구분": type_label,
                                    "금액/수량": f"{amount:,.4f}" if amount < 100 else f"{amount:,.0f}",
                                    "체결금액(KRW)": "-",
                                    "수수료": f"{fee_val:,.4f}" if fee_val > 0 else "-",
                                    "상태": state_kr,
                                    "_sort_dt": done,
                                })
                            return rows

                        def _parse_orders(raw):
                            """체결 주문 데이터를 통합 포맷으로 변환"""
                            rows = []
                            for r in raw:
                                market = r.get('market', '')
                                coin = market.split('-')[1] if '-' in str(market) else market
                                side = r.get('side', '')
                                side_kr = "매수" if side == 'bid' else ("매도" if side == 'ask' else side)
                                state = r.get('state', '')
                                state_kr = {"done": "체결완료", "cancel": "취소", "wait": "대기"}.get(state, state)
                                price = float(r.get('price', 0) or 0)
                                executed_vol = float(r.get('executed_volume', 0) or 0)
                                paid_fee = float(r.get('paid_fee', 0) or 0)
                                if price > 0 and executed_vol > 0:
                                    total_krw = price * executed_vol
                                elif 'trades' in r and r['trades']:
                                    total_krw = sum(float(t.get('funds', 0)) for t in r['trades'])
                                else:
                                    total_krw = price
                                ord_type = r.get('ord_type', '')
                                type_kr = {"limit": "지정가", "price": "시장가(매수)", "market": "시장가(매도)"}.get(ord_type, ord_type)
                                created = r.get('created_at', '')
                                if pd.notna(created):
                                    try: created = pd.to_datetime(created).strftime('%Y-%m-%d %H:%M')
                                    except: pass
                                rows.append({
                                    "거래일시": created, "유형": f"체결({type_kr})",
                                    "화폐/코인": coin,
                                    "구분": side_kr,
                                    "금액/수량": f"{executed_vol:,.8f}" if executed_vol < 1 else f"{executed_vol:,.4f}",
                                    "체결금액(KRW)": f"{total_krw:,.0f}",
                                    "수수료": f"{paid_fee:,.2f}",
                                    "상태": state_kr,
                                    "_sort_dt": created,
                                })
                            return rows

                        api_curr = None if h_curr == "전체" else h_curr
                        all_rows = []
                        error_msgs = []

                        # 조회 대상 결정
                        query_types = []
                        if h_type == "전체":
                            query_types = [("deposit", "입금"), ("withdraw", "출금"), ("order", "체결")]
                        elif "입금" in h_type:
                            query_types = [("deposit", "입금")]
                        elif "출금" in h_type:
                            query_types = [("withdraw", "출금")]
                        elif "체결" in h_type:
                            query_types = [("order", "체결")]

                        for api_type, label in query_types:
                            try:
                                data, err = trader.get_history(api_type, api_curr)
                                if err:
                                    error_msgs.append(f"{label}: {err}")
                                if data:
                                    if api_type in ("deposit", "withdraw"):
                                        all_rows.extend(_parse_deposit_withdraw(data, label))
                                    else:
                                        all_rows.extend(_parse_orders(data))
                            except Exception as e:
                                error_msgs.append(f"{label}: {e}")

                        # 에러 표시
                        for em in error_msgs:
                            if "out_of_scope" in em or "권한" in em:
                                st.error(f"API 권한 부족 ({em.split(':')[0]})")
                            else:
                                st.error(f"API 오류: {em}")
                        if error_msgs and not all_rows:
                            st.info("[업비트 > 마이페이지 > Open API 관리]에서 **자산조회**, **입출금 조회** 권한을 활성화해주세요.")

                        # 날짜 필터 + 표시
                        if all_rows:
                            result_df = pd.DataFrame(all_rows)
                            # 날짜 필터링
                            try:
                                result_df['_dt'] = pd.to_datetime(result_df['_sort_dt'], errors='coerce')
                                mask = (result_df['_dt'].dt.date >= h_date_start) & (result_df['_dt'].dt.date <= h_date_end)
                                result_df = result_df[mask].sort_values('_dt', ascending=False)
                            except Exception:
                                pass
                            result_df = result_df.drop(columns=['_sort_dt', '_dt'], errors='ignore')

                            if len(result_df) > 0:
                                st.success(f"{len(result_df)}건 조회됨")
                                def _color_side(val):
                                    if val == "매수": return "color: #e74c3c"
                                    elif val == "매도": return "color: #2980b9"
                                    elif val == "입금": return "color: #27ae60"
                                    elif val == "출금": return "color: #8e44ad"
                                    return ""
                                st.dataframe(
                                    result_df.style.map(_color_side, subset=["구분"]),
                                    use_container_width=True, hide_index=True
                                )
                            else:
                                st.warning("해당 기간에 내역이 없습니다.")
                        elif not error_msgs:
                            st.warning(f"조회 결과 없음. (유형: {h_type}, 화폐: {h_curr})")
                            st.caption("Upbit API는 최근 내역만 반환합니다. 조회 유형을 변경해보세요.")

            st.caption("Upbit API 제한: 최근 100건까지 조회 가능")

        with hist_tab2:
            st.subheader("가상 계좌 관리")

            if 'virtual_adjustment' not in st.session_state:
                st.session_state.virtual_adjustment = 0

            c1, c2 = st.columns(2)
            amount = c1.number_input("금액 (KRW)", step=100000)
            if c2.button("입출금 (가상)"):
                st.session_state.virtual_adjustment += amount
                st.success(f"가상 잔고 조정: {amount:,.0f} KRW")

            st.info(f"누적 가상 조정액: {st.session_state.virtual_adjustment:,.0f} KRW")

        with hist_tab3:
            st.subheader("슬리피지 분석 (실제 체결 vs 백테스트)")

            if not trader:
                st.warning("API Key가 필요합니다.")
            else:
                sa_col1, sa_col2 = st.columns(2)
                sa_ticker_list = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
                sa_ticker = sa_col1.selectbox("코인 선택", sa_ticker_list, key="sa_ticker")
                sa_interval = sa_col2.selectbox("시간봉", list(INTERVAL_MAP.keys()), key="sa_interval")

                if st.button("슬리피지 분석", key="sa_run"):
                    with st.spinner("체결 데이터 조회 중..."):
                        # 1. 실제 체결 주문 조회
                        done_orders = trader.get_done_orders(sa_ticker)

                        if not done_orders:
                            st.info("체결 완료된 주문이 없습니다.")
                        else:
                            df_orders = pd.DataFrame(done_orders)

                            # 필요한 컬럼 처리
                            if 'created_at' in df_orders.columns:
                                df_orders['date'] = pd.to_datetime(df_orders['created_at'])
                            if 'price' in df_orders.columns:
                                df_orders['exec_price'] = pd.to_numeric(df_orders['price'], errors='coerce')
                            if 'executed_volume' in df_orders.columns:
                                df_orders['exec_volume'] = pd.to_numeric(df_orders['executed_volume'], errors='coerce')

                            # 2. 해당 기간 OHLCV 조회 → Open 가격과 비교
                            api_interval = INTERVAL_MAP.get(sa_interval, "day")
                            df_ohlcv = data_cache.get_ohlcv_local_first(
                                sa_ticker,
                                interval=api_interval,
                                count=200,
                                allow_api_fallback=True,
                            )

                            if df_ohlcv is not None and 'date' in df_orders.columns and 'exec_price' in df_orders.columns:
                                # 날짜별 Open 가격 매핑
                                df_ohlcv['open_price'] = df_ohlcv['open']

                                slip_data = []
                                for _, order in df_orders.iterrows():
                                    order_date = order.get('date')
                                    exec_price = order.get('exec_price', 0)
                                    side = order.get('side', '')

                                    if pd.isna(order_date) or exec_price == 0:
                                        continue

                                    # 가장 가까운 캔들의 Open 가격 찾기
                                    if df_ohlcv.index.tz is not None and order_date.tzinfo is None:
                                        order_date = order_date.tz_localize(df_ohlcv.index.tz)

                                    idx = df_ohlcv.index.searchsorted(order_date)
                                    if idx < len(df_ohlcv):
                                        candle_open = df_ohlcv.iloc[idx]['open']
                                        slippage_pct = (exec_price - candle_open) / candle_open * 100
                                        if side == 'ask':  # 매도
                                            slippage_pct = -slippage_pct

                                        slip_data.append({
                                            'date': order_date,
                                            'side': 'BUY' if side == 'bid' else 'SELL',
                                            'exec_price': exec_price,
                                            'candle_open': candle_open,
                                            'slippage_pct': slippage_pct,
                                            'volume': order.get('exec_volume', 0)
                                        })

                                if slip_data:
                                    df_slip = pd.DataFrame(slip_data)

                                    # 요약 통계
                                    avg_slip = df_slip['slippage_pct'].mean()
                                    max_slip = df_slip['slippage_pct'].max()
                                    min_slip = df_slip['slippage_pct'].min()

                                    sc1, sc2, sc3, sc4 = st.columns(4)
                                    sc1.metric("평균 슬리피지", f"{avg_slip:.3f}%")
                                    sc2.metric("최대 (불리)", f"{max_slip:.3f}%")
                                    sc3.metric("최소 (유리)", f"{min_slip:.3f}%")
                                    sc4.metric("거래 수", f"{len(df_slip)}건")

                                    # 매수/매도 분리 통계
                                    buy_slip = df_slip[df_slip['side'] == 'BUY']
                                    sell_slip = df_slip[df_slip['side'] == 'SELL']

                                    if not buy_slip.empty:
                                        st.caption(f"매수 평균 슬리피지: {buy_slip['slippage_pct'].mean():.3f}% ({len(buy_slip)}건)")
                                    if not sell_slip.empty:
                                        st.caption(f"매도 평균 슬리피지: {sell_slip['slippage_pct'].mean():.3f}% ({len(sell_slip)}건)")

                                    # 차트
                                    fig_slip = go.Figure()
                                    fig_slip.add_trace(go.Bar(
                                        x=df_slip['date'], y=df_slip['slippage_pct'],
                                        marker_color=['red' if s > 0 else 'green' for s in df_slip['slippage_pct']],
                                        name='슬리피지 %'
                                    ))
                                    fig_slip.add_hline(y=avg_slip, line_dash="dash", line_color="blue",
                                                       annotation_text=f"Avg: {avg_slip:.3f}%")
                                    fig_slip.update_layout(title="거래 슬리피지 (+ = 불리)", height=350, margin=dict(t=80),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                                    st.plotly_chart(fig_slip, use_container_width=True)

                                    # 상세 테이블
                                    st.dataframe(
                                        df_slip.style.format({
                                            'exec_price': '{:,.0f}',
                                            'candle_open': '{:,.0f}',
                                            'slippage_pct': '{:.3f}%',
                                            'volume': '{:.6f}'
                                        }).background_gradient(cmap='RdYlGn_r', subset=['slippage_pct']),
                                        use_container_width=True
                                    )

                                    st.info(
                                        f"권장 백테스트 슬리피지: **{abs(avg_slip):.2f}%** "
                                        f"(실제 평균 기반, 백테스트 탭에서 설정)"
                                    )
                                else:
                                    st.info("매칭 가능한 체결-캔들 데이터가 없습니다.")
                            else:
                                st.dataframe(df_orders)
                                st.caption("OHLCV 매칭 불가 - 원본 주문 데이터 표시")

    # --- Tab 6: 트리거 ---
    with tab6:
        render_strategy_trigger_tab("COIN", coin_portfolio=portfolio_list)

    # --- Tab 4: 백테스트 ---
    with tab4:
        bt_sub1, bt_sub2, bt_sub4, bt_sub3 = st.tabs(
            ["📈 개별 백테스트", "🛠️ 파라미터 최적화", "🧩 보조 전략(역추세)", "📡 전체 종목 스캔"]
        )

        # === 서브탭1: 개별 백테스트 ===
        with bt_sub1:
            st.header("개별 자산 백테스트")

            port_tickers_bt = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
            base_options_bt = list(dict.fromkeys(port_tickers_bt + TOP_20_TICKERS))

            selected_ticker_bt = st.selectbox("백테스트 대상", base_options_bt + ["직접입력"], key="bt_target")

            bt_ticker = ""
            bt_sma = 0
            bt_buy_period = 20
            bt_sell_period = 10

            if selected_ticker_bt == "직접입력":
                bt_custom = st.text_input("코인", "BTC", key="bt_c")
                bt_ticker = f"KRW-{bt_custom.upper()}"
            else:
                bt_ticker = selected_ticker_bt

            port_match = next((item for item in portfolio_list if f"{item['market']}-{item['coin'].upper()}" == bt_ticker), None)

            default_strat_idx = 0
            if port_match and port_match.get('strategy') == 'Donchian':
                default_strat_idx = 1

            bt_strategy = st.selectbox("전략 선택", ["SMA 전략", "돈키안 전략"], index=default_strat_idx, key="bt_strategy_sel")

            # 기본값 초기화 (전략에 따라 덮어씀)
            bt_sma = 60
            bt_buy_period = 20
            bt_sell_period = 10
            bt_sell_mode = "하단선 (Lower)"  # SMA일 때 사용 안 되지만 undefined 방지

            if bt_strategy == "SMA 전략":
                default_sma = port_match.get('parameter', 60) if port_match else 60
                bt_sma = st.number_input("단기 SMA (추세)", value=default_sma, key="bt_sma_select", min_value=5, step=1)
            else:
                default_buy = int(port_match.get('parameter', 20)) if port_match and port_match.get('strategy') == 'Donchian' else 20
                default_sell = int(port_match.get('sell_parameter', 10)) if port_match and port_match.get('strategy') == 'Donchian' else 10
                if default_sell == 0:
                    default_sell = max(5, default_buy // 2)
                dc_col1, dc_col2 = st.columns(2)
                with dc_col1:
                    bt_buy_period = st.number_input("매수 채널 기간", value=default_buy, min_value=5, max_value=300, step=1, key="bt_dc_buy")
                with dc_col2:
                    bt_sell_period = st.number_input("매도 채널 기간", value=default_sell, min_value=5, max_value=300, step=1, key="bt_dc_sell")

                st.divider()
                st.caption("📌 매도 기준 선택")
                bt_sell_mode = st.radio(
                    "매도 라인",
                    ["하단선 (Lower)", "중심선 (Midline)", "두 방법 비교"],
                    horizontal=True,
                    key="bt_sell_mode",
                    help="하단선: 저가 채널 이탈 시 매도 / 중심선: (상단+하단)/2 이탈 시 매도"
                )

            default_interval_idx = 0
            if port_match:
                port_iv_label = INTERVAL_REV_MAP.get(port_match.get('interval', 'day'), '일봉')
                interval_keys = list(INTERVAL_MAP.keys())
                if port_iv_label in interval_keys:
                    default_interval_idx = interval_keys.index(port_iv_label)

            bt_interval_label = st.selectbox("시간봉 선택", options=list(INTERVAL_MAP.keys()), index=default_interval_idx, key="bt_interval_sel")
            bt_interval = INTERVAL_MAP[bt_interval_label]

            DEFAULT_SLIPPAGE = {
                "major": {"day": 0.03, "minute240": 0.05, "minute60": 0.08, "minute30": 0.08, "minute15": 0.10, "minute5": 0.15, "minute1": 0.20},
                "mid":   {"day": 0.05, "minute240": 0.08, "minute60": 0.10, "minute30": 0.10, "minute15": 0.15, "minute5": 0.20, "minute1": 0.30},
                "alt":   {"day": 0.10, "minute240": 0.15, "minute60": 0.20, "minute30": 0.20, "minute15": 0.25, "minute5": 0.35, "minute1": 0.50},
            }
            MAJOR_COINS = {"BTC", "ETH"}
            MID_COINS = {"XRP", "SOL", "DOGE", "ADA", "TRX", "AVAX", "LINK", "BCH", "DOT", "ETC"}

            def get_default_slippage(ticker, interval):
                coin = ticker.split("-")[-1].upper() if "-" in ticker else ticker.upper()
                if coin in MAJOR_COINS:
                    tier = "major"
                elif coin in MID_COINS:
                    tier = "mid"
                else:
                    tier = "alt"
                return DEFAULT_SLIPPAGE[tier].get(interval, 0.10)

            default_slip = get_default_slippage(bt_ticker, bt_interval)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.caption("백테스트 기간")
                d_col1, d_col2 = st.columns(2)
                try:
                    default_start_bt = datetime(2020, 1, 1).date()
                except:
                    default_start_bt = datetime.now().date() - timedelta(days=365)
                default_end_bt = datetime.now().date()

                bt_start = d_col1.date_input("시작일", value=default_start_bt, max_value=datetime.now().date(), format="YYYY.MM.DD", key="bt_start")
                bt_end = d_col2.date_input("종료일", value=default_end_bt, max_value=datetime.now().date(), format="YYYY.MM.DD", key="bt_end")

                if bt_start > bt_end:
                    st.error("시작일은 종료일보다 빨라야 합니다.")
                    bt_end = bt_start

                days_diff = (bt_end - bt_start).days
                st.caption(f"기간: {days_diff}일")

                fee = st.number_input("매매 수수료 (%)", value=0.05, format="%.2f", key="bt_fee") / 100
                bt_slippage = st.number_input("슬리피지 (%)", value=default_slip, min_value=0.0, max_value=2.0, step=0.01, format="%.2f", key="bt_slip")

                fee_pct = fee * 100
                cost_per_trade = fee_pct + bt_slippage
                cost_round_trip = (fee_pct * 2) + (bt_slippage * 2)
                st.caption(f"편도: {cost_per_trade:.2f}% | 왕복: {cost_round_trip:.2f}%")

                run_btn = st.button("백테스트 실행", type="primary", key="bt_run")

            if run_btn:
                if bt_strategy == "돈키안 전략":
                    req_period = max(bt_buy_period, bt_sell_period)
                    bt_strategy_mode = "Donchian"
                    bt_sell_ratio = bt_sell_period / bt_buy_period if bt_buy_period > 0 else 0.5
                    # 매도방식 파싱
                    _smode_raw = bt_sell_mode if bt_strategy == "돈키안 전략" else "하단선 (Lower)"
                    _compare_mode = _smode_raw == "두 방법 비교"
                    _sell_mode_api = "midline" if _smode_raw == "중심선 (Midline)" else "lower"
                else:
                    req_period = bt_sma
                    bt_strategy_mode = "SMA 전략"
                    bt_sell_ratio = 0.5
                    _compare_mode = False
                    _sell_mode_api = "lower"

                to_date = bt_end + timedelta(days=1)
                to_str = to_date.strftime("%Y-%m-%d 09:00:00")
                cpd = CANDLES_PER_DAY.get(bt_interval, 1)
                req_count = days_diff * cpd + req_period + 300
                fetch_count = max(req_count, req_period + 300)

                with st.spinner(f"백테스트 실행 중 ({bt_start} ~ {bt_end}, {bt_interval_label}, {bt_strategy})..."):
                    df_bt = data_cache.get_ohlcv_local_first(
                        bt_ticker,
                        interval=bt_interval,
                        to=to_str,
                        count=fetch_count,
                        allow_api_fallback=True,
                    )
                    if df_bt is None or df_bt.empty:
                        st.error("데이터를 가져올 수 없습니다.")
                        st.stop()

                    st.caption(f"조회된 캔들: {len(df_bt)}개 ({df_bt.index[0].strftime('%Y-%m-%d')} ~ {df_bt.index[-1].strftime('%Y-%m-%d')})")

                    result = backtest_engine.run_backtest(
                        bt_ticker, period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                        interval=bt_interval, count=fetch_count, fee=fee,
                        start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                        strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=bt_slippage,
                        sell_mode="lower" if _compare_mode else _sell_mode_api
                    )
                    # 비교 모드: 중심선 결과도 실행
                    if _compare_mode and bt_strategy_mode == "Donchian":
                        result_mid = backtest_engine.run_backtest(
                            bt_ticker, period=bt_buy_period,
                            interval=bt_interval, count=fetch_count, fee=fee,
                            start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                            strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=bt_slippage,
                            sell_mode="midline"
                        )
                    else:
                        result_mid = None

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        df = result["df"]
                        res = result["performance"]

                        # ── 비교 요약 테이블 (비교모드) ──
                        if _compare_mode and result_mid and "error" not in result_mid:
                            res_mid = result_mid["performance"]
                            st.subheader("📊 하단선 vs 중심선 비교")
                            cmp_data = {
                                "항목": ["총 수익률", "CAGR", "MDD", "샤프비율", "승률", "거래 횟수", "최종 자산"],
                                f"하단선 Lower({bt_sell_period})": [
                                    f"{res['total_return']:,.2f}%",
                                    f"{res.get('cagr', 0):,.2f}%",
                                    f"{res['mdd']:,.2f}%",
                                    f"{res.get('sharpe', 0):.2f}",
                                    f"{res['win_rate']:,.2f}%",
                                    f"{res['trade_count']}회",
                                    f"{res['final_equity']:,.0f} KRW",
                                ],
                                f"중심선 Midline": [
                                    f"{res_mid['total_return']:,.2f}%",
                                    f"{res_mid.get('cagr', 0):,.2f}%",
                                    f"{res_mid['mdd']:,.2f}%",
                                    f"{res_mid.get('sharpe', 0):.2f}",
                                    f"{res_mid['win_rate']:,.2f}%",
                                    f"{res_mid['trade_count']}회",
                                    f"{res_mid['final_equity']:,.0f} KRW",
                                ],
                            }
                            st.dataframe(pd.DataFrame(cmp_data).set_index("항목"), use_container_width=True)

                            # 승자 표시
                            if res['total_return'] > res_mid['total_return']:
                                st.success(f"✅ 하단선(Lower) 방식이 수익률 {res['total_return']:.2f}% vs {res_mid['total_return']:.2f}% 로 우수")
                            elif res_mid['total_return'] > res['total_return']:
                                st.success(f"✅ 중심선(Midline) 방식이 수익률 {res_mid['total_return']:.2f}% vs {res['total_return']:.2f}% 로 우수")
                            else:
                                st.info("두 방식의 수익률이 동일합니다.")
                            st.divider()

                        sell_mode_label = "중심선(Midline)" if _sell_mode_api == "midline" and not _compare_mode else ("하단선(Lower)" if not _compare_mode else "하단선(Lower) [기준]")
                        if not _compare_mode:
                            st.caption(f"매도방식: **{sell_mode_label}**")

                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("총 수익률", f"{res['total_return']:,.2f}%")
                        m2.metric("연평균(CAGR)", f"{res.get('cagr', 0):,.2f}%")
                        m3.metric("승률", f"{res['win_rate']:,.2f}%")
                        m4.metric("최대낙폭(MDD)", f"{res['mdd']:,.2f}%")
                        m5.metric("샤프비율", f"{res['sharpe']:.2f}")

                        trade_count = res['trade_count']
                        total_cost_pct = cost_round_trip * trade_count
                        st.success(
                            f"최종 잔고: **{res['final_equity']:,.0f} KRW** (초기 {initial_cap:,.0f} KRW) | "
                            f"거래 {trade_count}회 | 왕복비용 {cost_round_trip:.2f}% | 누적 약 {total_cost_pct:.1f}%"
                        )

                        if bt_slippage > 0:
                            result_no_slip = backtest_engine.run_backtest(
                                bt_ticker, period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                                interval=bt_interval, count=fetch_count, fee=fee,
                                start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                                strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=0.0
                            )
                            if "error" not in result_no_slip:
                                res_ns = result_no_slip['performance']
                                slip_ret_diff = res_ns['total_return'] - res['total_return']
                                slip_cost = res_ns['final_equity'] - res['final_equity']
                                st.info(f"슬리피지 영향: 수익률 차이 **{slip_ret_diff:,.2f}%p**, 금액 차이 **{slip_cost:,.0f} KRW**")

                        st.subheader("가격 & 전략 성과")
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

                        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name='가격'), row=1, col=1, secondary_y=False)

                        if bt_strategy_mode == "Donchian":
                            upper_col = f'Donchian_Upper_{bt_buy_period}'
                            lower_col = f'Donchian_Lower_{bt_sell_period}'
                            if upper_col in df.columns:
                                fig.add_trace(go.Scatter(x=df.index, y=df[upper_col], line=dict(color='green', width=1.5, dash='dash'), name=f'상단 ({bt_buy_period})'), row=1, col=1, secondary_y=False)
                            if lower_col in df.columns:
                                fig.add_trace(go.Scatter(x=df.index, y=df[lower_col], line=dict(color='red', width=1.5, dash='dash'), name=f'하단 ({bt_sell_period})'), row=1, col=1, secondary_y=False)
                        else:
                            fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{bt_sma}'], line=dict(color='orange', width=2), name=f'SMA {bt_sma}'), row=1, col=1, secondary_y=False)

                        fig.add_trace(go.Scatter(x=df.index, y=df['equity'], line=dict(color='blue', width=2), name='전략 자산'), row=1, col=1, secondary_y=True)

                        buy_dates = [t['date'] for t in res['trades'] if t['type'] == 'buy']
                        buy_prices = [t['price'] for t in res['trades'] if t['type'] == 'buy']
                        sell_dates = [t['date'] for t in res['trades'] if t['type'] == 'sell']
                        sell_prices = [t['price'] for t in res['trades'] if t['type'] == 'sell']
                        if buy_dates:
                            fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='매수'), row=1, col=1, secondary_y=False)
                        if sell_dates:
                            fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='매도'), row=1, col=1, secondary_y=False)

                        fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], name='낙폭 (%)', fill='tozeroy', line=dict(color='red', width=1)), row=2, col=1)
                        fig.update_layout(height=800, title_text="백테스트 결과", xaxis_rangeslider_visible=False, margin=dict(t=80),
                            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                        fig.update_yaxes(title_text="가격 (KRW)", row=1, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="자산 (KRW)", row=1, col=1, secondary_y=True)
                        fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
                        fig = _apply_dd_hover_format(fig)
                        st.plotly_chart(fig, use_container_width=True)

                        if 'yearly_stats' in res:
                            st.subheader("연도별 성과")
                            st.dataframe(res['yearly_stats'].style.format("{:.2f}%"))

                        _render_performance_analysis(
                            equity_series=df.get("equity"),
                            benchmark_series=df.get("close"),
                            strategy_metrics=res,
                            strategy_label="백테스트 전략",
                            benchmark_label=f"{bt_ticker} 단순보유",
                            monte_carlo_sims=400,
                        )

                        st.info(f"전략 상태: **{res['final_status']}** | 다음 행동: **{res['next_action'] if res['next_action'] else '없음'}**")

                        with st.expander("거래 내역"):
                            if res['trades']:
                                trades_df = pd.DataFrame(res['trades'])
                                st.dataframe(trades_df.style.format({"price": "{:,.2f}", "amount": "{:,.6f}", "balance": "{:,.2f}", "profit": "{:,.2f}%"}))
                            else:
                                st.info("실행된 거래가 없습니다.")

                        csv_data = df.to_csv(index=True).encode('utf-8')
                        st.download_button(label="일별 로그 다운로드", data=csv_data, file_name=f"{bt_ticker}_{bt_start}_daily_log.csv", mime="text/csv")

        # === 서브탭2: 파라미터 최적화 ===
        with bt_sub2:
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

            # opt_dc_sell_mode가 form 외부에서도 참조되지 않도록 기본값 세팅
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
                                    buy_range  = range(opt_buy_start,  opt_buy_end  + 1, opt_buy_step)
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
                                        # 단일 모드일 때는 opt_results에도 담기
                                        if len(_modes_to_run) == 1:
                                            opt_results = _mode_rows
                                    # 비교 모드 결과를 세션 상태에 저장
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

                            # Optuna/SMA Grid 결과도 세션에 저장 (아직 저장 안 된 경우)
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

            # ── 결과 표시 (모든 지점에서 세션 상태로 유지됨) ───────────────────────

            def _add_robustness(df_in, neighbor_steps=2):
                """각 (Buy Period, Sell Period) 조합에 대해 인접 ±neighbor_steps 단계 Calmar 평균 = Robustness
                절대값이 아닌 정렬된 고유값 인덱스 기준으로 이웃을 찾아 step 크기에 무관하게 동작."""
                if "Robustness" in df_in.columns: return df_in
                df_out = df_in.copy()
                
                # 1. Donchian (2D)
                if "Buy Period" in df_out.columns and "Sell Period" in df_out.columns:
                    # 고유 Buy/Sell 값을 정렬해 인덱스 매핑
                    buy_vals  = sorted(df_out["Buy Period"].unique())
                    sell_vals = sorted(df_out["Sell Period"].unique())
                    buy_idx   = {v: i for i, v in enumerate(buy_vals)}
                    sell_idx  = {v: i for i, v in enumerate(sell_vals)}
                    max_bi, max_si = len(buy_vals) - 1, len(sell_vals) - 1

                    calmar_lookup = {}
                    for _, row in df_out.iterrows():
                        calmar_lookup[(row["Buy Period"], row["Sell Period"])] = row["Calmar"]

                    def _rob(bp, sp):
                        bi, si = buy_idx.get(bp, -1), sell_idx.get(sp, -1)
                        if bi == -1 or si == -1: return 0.0
                        vals = []
                        for b_i in range(max(0, bi - neighbor_steps), min(max_bi, bi + neighbor_steps) + 1):
                            for s_i in range(max(0, si - neighbor_steps), min(max_si, si + neighbor_steps) + 1):
                                k = (buy_vals[b_i], sell_vals[s_i])
                                if k in calmar_lookup:
                                    vals.append(calmar_lookup[k])
                        return round(sum(vals) / len(vals), 2) if vals else 0.0
                    df_out["Robustness"] = df_out.apply(lambda r: _rob(r["Buy Period"], r["Sell Period"]), axis=1)
                
                # 2. SMA (1D)
                elif "SMA Period" in df_out.columns:
                    vals = sorted(df_out["SMA Period"].unique())
                    v_idx = {v: i for i, v in enumerate(vals)}
                    max_i = len(vals) - 1
                    lookup = {}
                    for _, row in df_out.iterrows():
                        lookup[row["SMA Period"]] = row["Calmar"]
                    
                    def _rob_sma(val):
                        idx = v_idx.get(val, -1)
                        if idx == -1: return 0.0
                        n_vals = []
                        for i in range(max(0, idx - neighbor_steps), min(max_i, idx + neighbor_steps) + 1):
                            nv = vals[i]
                            if nv in lookup:
                                n_vals.append(lookup[nv])
                        return round(sum(n_vals) / len(n_vals), 2) if n_vals else 0.0
                    df_out["Robustness"] = df_out["SMA Period"].apply(_rob_sma)

                return df_out

            # ── 히트맵 헬퍼 (★ 표시) ──
            def _render_go_heatmap(df_in, x_col, y_col, z_col, title=""):
                import plotly.graph_objects as _go
                pivot = df_in.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
                pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
                if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
                    return None
                colorscale = "RdYlGn_r" if "MDD" in z_col else "RdYlGn"
                vals = pivot.values.copy()
                text_strs = np.where(np.isnan(vals), "", np.vectorize(lambda x: f"{x:.1f}")(vals))
                mask_valid = ~np.isnan(vals)
                if mask_valid.any():
                    best_flat = int(np.nanargmax(vals))
                    best_r, best_c = np.unravel_index(best_flat, vals.shape)
                    text_strs[best_r][best_c] = f"\u2605{text_strs[best_r][best_c]}"
                fig = _go.Figure(data=_go.Heatmap(
                    z=vals, x=[str(int(c)) for c in pivot.columns], y=[str(int(r)) for r in pivot.index],
                    colorscale=colorscale, text=text_strs, texttemplate="%{text}", textfont=dict(size=11),
                    hovertemplate=f"{y_col}: %{{y}}<br>{x_col}: %{{x}}<br>{z_col}: %{{z:.2f}}<extra></extra>",
                    colorbar=dict(title=z_col),
                ))
                fig.update_layout(title=title or f"{y_col} vs {x_col} ({z_col})",
                    xaxis_title=x_col, yaxis_title=y_col, xaxis_type="category", yaxis_type="category", height=400)
                return fig

            def _evaluate_cv(cv):
                if cv <= 10: return "매우 안정", "\U0001f7e2"
                elif cv <= 20: return "안정", "\U0001f535"
                elif cv <= 35: return "보통", "\U0001f7e1"
                elif cv <= 50: return "불안정", "\U0001f7e0"
                else: return "매우 불안정", "\U0001f534"

            def _compute_stability(df_in, metrics=None):
                if metrics is None:
                    metrics = ["Calmar", "Total Return (%)", "MDD (%)"]
                stability = {}
                for col in metrics:
                    if col in df_in.columns and len(df_in) > 1:
                        mean_v = df_in[col].mean()
                        std_v = df_in[col].std()
                        cv = abs(std_v / mean_v) * 100 if mean_v != 0 else 0
                        stability[col] = {"평균": round(mean_v, 2), "표준편차": round(std_v, 2),
                            "CV(%)": round(cv, 1), "최소": round(df_in[col].min(), 2), "최대": round(df_in[col].max(), 2)}
                return stability

            def _monte_carlo_sim(daily_rets, n_sims=1000, n_days=756, init_cap=10000):
                """부트스트랩 몬테카를로 시뮬레이션."""
                paths = np.zeros((n_sims, n_days + 1))
                paths[:, 0] = init_cap
                for i in range(n_sims):
                    sampled = np.random.choice(daily_rets, size=n_days, replace=True)
                    paths[i, 1:] = init_cap * np.cumprod(1 + sampled)
                final_values = paths[:, -1]
                years = n_days / 252
                cagr_dist = ((final_values / init_cap) ** (1 / max(years, 0.01)) - 1) * 100
                mdd_dist = np.zeros(n_sims)
                for i in range(n_sims):
                    peak = np.maximum.accumulate(paths[i])
                    dd = (paths[i] - peak) / peak * 100
                    mdd_dist[i] = dd.min()
                pcts = {p: float(np.percentile(final_values, p)) for p in [5, 25, 50, 75, 95]}
                return {"paths": paths, "final_values": final_values, "cagr_dist": cagr_dist,
                        "mdd_dist": mdd_dist, "percentiles": pcts}

            def _kelly_criterion(daily_rets):
                """켈리 기준 투자비중 계산."""
                wins = daily_rets[daily_rets > 0]
                losses = daily_rets[daily_rets < 0]
                if len(wins) == 0 or len(losses) == 0:
                    return {"kelly_full": 0, "kelly_half": 0, "kelly_quarter": 0, "win_rate": 0,
                            "avg_win": 0, "avg_loss": 0, "payoff_ratio": 0,
                            "grade": "분석 불가", "recommendation": "승/패 데이터가 부족합니다."}
                win_rate = len(wins) / len(daily_rets)
                avg_win = wins.mean() * 100
                avg_loss = abs(losses.mean()) * 100
                payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                kelly_full = max(0, min(((payoff_ratio * win_rate - (1 - win_rate)) / payoff_ratio * 100) if payoff_ratio > 0 else 0, 100))
                kelly_half = kelly_full / 2
                kelly_quarter = kelly_full / 4
                if kelly_full <= 0:
                    grade, rec = "투자 불가", "Kelly 기준으로 이 전략에 투자하지 않는 것이 좋습니다."
                elif kelly_full <= 15:
                    grade, rec = "보수적", "소규모 투자만 권장합니다. 하프 켈리 이하로 운용하세요."
                elif kelly_full <= 30:
                    grade, rec = "적정", "적정 수준의 투자가 가능합니다. 하프 켈리를 기준으로 운용하세요."
                elif kelly_full <= 50:
                    grade, rec = "공격적", "높은 비중이 가능하지만, 하프 켈리로 보수적 접근을 권장합니다."
                else:
                    grade, rec = "매우 공격적", "풀 켈리는 과도한 위험을 수반합니다. 쿼터~하프 켈리를 권장합니다."
                return {"kelly_full": kelly_full, "kelly_half": kelly_half, "kelly_quarter": kelly_quarter,
                        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "payoff_ratio": payoff_ratio,
                        "grade": grade, "recommendation": rec}

            def _show_robustness_evaluation(stab):
                """CV 기반 종합 평가 한글 텍스트 표시."""
                if not stab:
                    return
                st.markdown("#### 종합 평가")
                lines = []
                for metric, info in stab.items():
                    cv = info.get("CV(%)", 0)
                    grade, icon = _evaluate_cv(cv)
                    mn, mx = info.get("최소", 0), info.get("최대", 0)
                    avg = info.get("평균", 0)
                    if metric == "MDD (%)":
                        lines.append(f"- {icon} **{metric}**: 평균 {avg:.2f}%, 범위 {mn:.2f}% ~ {mx:.2f}%, CV {cv:.1f}% → **{grade}**")
                    else:
                        lines.append(f"- {icon} **{metric}**: 평균 {avg:.2f}, 범위 {mn:.2f} ~ {mx:.2f}, CV {cv:.1f}% → **{grade}**")
                st.markdown("\n".join(lines))
                # 종합 등급
                key_metrics = ["Calmar", "Total Return (%)", "MDD (%)"]
                cvs = [stab[m]["CV(%)"] for m in key_metrics if m in stab]
                avg_cv = np.mean(cvs) if cvs else 0
                overall_grade, overall_icon = _evaluate_cv(avg_cv)
                st.markdown(f"### {overall_icon} 종합 안정성: **{overall_grade}** (평균 CV {avg_cv:.1f}%)")
                # 실전 권고
                calmar_cv = stab.get("Calmar", {}).get("CV(%)", 0)
                if calmar_cv <= 15:
                    st.markdown("> **Calmar Ratio**가 파라미터 변화에 둔감하여 안정적입니다.")
                elif calmar_cv <= 30:
                    st.markdown("> **Calmar Ratio**가 어느 정도 변동하지만 수용 가능한 수준입니다.")
                else:
                    st.markdown("> **Calmar Ratio**가 파라미터 변화에 민감합니다. 과적합 가능성을 주의하세요.")
                if avg_cv <= 15:
                    st.success("파라미터 안정성이 높아 **실전 적용에 적합**합니다.")
                elif avg_cv <= 30:
                    st.info(f"파라미터 안정성이 보통 수준입니다. 주변 파라미터 평균 성과를 기대값으로 잡는 것이 현실적입니다.")
                else:
                    st.warning("파라미터 민감도가 높아 **과적합 위험**이 있습니다. 다른 기간으로 교차 검증을 권장합니다.")

            _saved_compare = st.session_state.get("opt_compare_results", {})
            _saved_single  = st.session_state.get("opt_single_results", {})

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
                            # 비교 데이터프레임에도 Robustness 추가
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
                opt_results  = _saved_single["rows"]
                _s_strategy  = _saved_single["strategy"]
                _s_optuna    = _saved_single["use_optuna"]
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

                    # 정렬
                    _sort_map = {"Calmar (CAGR/MDD)": ("Calmar", False), "수익률 (높은순)": ("Total Return (%)", False),
                                 "CAGR (높은순)": ("CAGR (%)", False), "MDD (낮은순)": ("MDD (%)", True),
                                 "Sharpe (높은순)": ("Sharpe", False), "Robustness (높은순)": ("Robustness", False)}
                    _scol, _sasc = _sort_map.get(_opt_sort, ("Calmar", False))
                    if _scol in opt_df.columns:
                        opt_df = opt_df.sort_values(_scol, ascending=_sasc).reset_index(drop=True)
                    best_row = opt_df.iloc[0]

                    # MDD 필터 (표시용)
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

                    # 결과 테이블 (Top-N, 필터 적용)
                    _display_src = _filtered_df if _n_filtered > 0 else opt_df
                    _display_df = _display_src.head(_opt_top_n).copy()
                    _display_df.index = _display_df.index + 1
                    _display_df.index.name = "순위"
                    _grad_cols = [c for c in ['Total Return (%)', 'Calmar', 'Sharpe', 'Robustness'] if c in _display_df.columns]
                    st.dataframe(
                        _display_df.style.background_gradient(cmap='RdYlGn', subset=_grad_cols)
                            .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)']).format("{:,.2f}"),
                        use_container_width=True, height=500)

                    # 🔍 상세 분석 Expander (Robustness Check)
                    with st.expander("🔍 최적 파라미터 주변 상세 분석 (Robustness)", expanded=False):
                        try:
                            if _s_strategy == "돈키안 전략" and "Buy Period" in opt_df.columns:
                                st.caption("최적 (Buy, Sell) 파라미터 기준 ±2단계 이웃들의 성과를 분석합니다.")
                                b_val, s_val = int(best_row["Buy Period"]), int(best_row["Sell Period"])
                                b_uniq = sorted(opt_df["Buy Period"].unique())
                                s_uniq = sorted(opt_df["Sell Period"].unique())
                                
                                if b_val in b_uniq and s_val in s_uniq:
                                    b_idx, s_idx = b_uniq.index(b_val), s_uniq.index(s_val)
                                    nb_vals = b_uniq[max(0, b_idx-2) : min(len(b_uniq), b_idx+3)]
                                    ns_vals = s_uniq[max(0, s_idx-2) : min(len(s_uniq), s_idx+3)]
                                    
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
                                    np_vals = p_uniq[max(0, p_idx-2) : min(len(p_uniq), p_idx+3)]
                                    
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

        # === 서브탭3: 전체 종목 스캔 ===
        with bt_sub4:
            st.header("보조 전략 백테스트")
            st.caption("메인 전략이 CASH일 때만 보조 분할매수 전략을 실행합니다.")

            # 위젯 생성 전 pending 적용 (Streamlit widget key 직접 수정 오류 방지)
            _aux_pending_apply = st.session_state.pop("aux_opt_apply_pending", None)
            if isinstance(_aux_pending_apply, dict) and _aux_pending_apply:
                for _k, _v in _aux_pending_apply.items():
                    st.session_state[_k] = _v

            aux_col1, aux_col2 = st.columns(2)

            with aux_col1:
                st.subheader("메인 전략 설정")
                _aux_port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
                _aux_base = list(dict.fromkeys(_aux_port_tickers + TOP_20_TICKERS))
                aux_ticker = st.selectbox("대상 티커", _aux_base + ["직접 입력"], key="aux_bt_ticker")
                if aux_ticker == "직접 입력":
                    aux_ticker = st.text_input("티커 입력 (예: KRW-BTC)", "KRW-BTC", key="aux_bt_ticker_custom")

                aux_main_strat = st.selectbox("메인 전략", ["Donchian", "SMA"], key="aux_bt_main_strat")
                amc1, amc2 = st.columns(2)
                aux_main_buy = amc1.number_input("메인 매수 기간", 5, 300, 115, key="aux_bt_main_buy")
                aux_main_sell = amc2.number_input(
                    "메인 매도 기간", 0, 300, 55, key="aux_bt_main_sell", help="SMA 선택 시 0이면 자동으로 buy_period/2를 사용합니다."
                )
                if aux_main_sell == 0:
                    aux_main_sell = max(5, int(aux_main_buy) // 2)

            with aux_col2:
                st.subheader("보조 전략 설정")
                aux_ma_count_label = st.radio("이평선 수", ["2개", "1개"], horizontal=True, key="aux_bt_ma_count")
                apc1, apc2 = st.columns(2)
                aux_ma_short = apc1.number_input("단기 MA", 3, 500, 5, key="aux_bt_ma_short")
                if aux_ma_count_label == "2개":
                    aux_ma_long = apc2.number_input("장기 MA", 5, 300, 20, key="aux_bt_ma_long")
                    if aux_ma_long <= aux_ma_short:
                        aux_ma_long = aux_ma_short + 1
                else:
                    aux_ma_long = int(aux_ma_short)
                    apc2.caption("1개 모드: 장기 MA는 사용하지 않습니다.")

                aux_threshold = st.slider("과매도 임계값(이격도 %)", -30.0, -0.5, -5.0, 0.5, key="aux_bt_threshold")
                aux_use_rsi = st.checkbox("RSI 필터 사용", value=False, key="aux_bt_use_rsi")
                if aux_use_rsi:
                    arc1, arc2 = st.columns(2)
                    aux_rsi_period = int(arc1.number_input("RSI 기간", min_value=2, max_value=50, value=2, step=1, key="aux_bt_rsi_period"))
                    aux_rsi_threshold = float(arc2.number_input("RSI 과매도 기준", min_value=1.0, max_value=30.0, value=8.0, step=0.5, key="aux_bt_rsi_threshold"))
                else:
                    aux_rsi_period = int(st.session_state.get("aux_bt_rsi_period", 2))
                    aux_rsi_threshold = float(st.session_state.get("aux_bt_rsi_threshold", 8.0))

                atc1, atc2 = st.columns(2)
                aux_tp1 = atc1.number_input("TP1 - 1차 매도 (%)", 1.0, 30.0, 3.0, 0.5, key="aux_bt_tp1")
                aux_tp2 = atc2.number_input("TP2 - 2차 매도 (%)", 1.0, 50.0, 10.0, 0.5, key="aux_bt_tp2")
                if aux_tp2 < aux_tp1:
                    aux_tp2 = aux_tp1
                st.caption("매도: TP1 도달 시 50% 매도 → TP2 도달 시 나머지 50% 매도")

                aux_split = st.number_input("분할 매수 횟수", 1, 20, 3, key="aux_bt_split")
                aux_seed_label = st.radio("매수 시드 방식", ["동일", "피라미딩"], horizontal=True, key="aux_bt_seed_mode")
                aux_seed_mode = "pyramiding" if aux_seed_label == "피라미딩" else "equal"

                aux_pyramid_ratio = 1.0
                if aux_seed_mode == "pyramiding":
                    aux_pyramid_ratio = st.number_input("피라미딩 배율", 1.05, 3.00, 1.30, 0.05, key="aux_bt_pyramid_ratio")

                _weights = np.ones(int(aux_split), dtype=float)
                if aux_seed_mode == "pyramiding":
                    _weights = np.array([aux_pyramid_ratio ** i for i in range(int(aux_split))], dtype=float)
                _weights = _weights / _weights.sum()
                st.caption("매수 시드 비중: " + " / ".join([f"{w * 100:.1f}%" for w in _weights]))

            iv_col1, iv_col2, iv_col3, iv_col4, iv_col5 = st.columns(5)
            aux_interval_label = iv_col1.selectbox(
                "보조 실행 주기",
                list(INTERVAL_MAP.keys()),
                index=2 if len(INTERVAL_MAP) > 2 else 0,
                key="aux_bt_interval",
            )
            aux_main_interval_label = iv_col2.selectbox(
                "메인 신호 주기",
                list(INTERVAL_MAP.keys()),
                index=1 if len(INTERVAL_MAP) > 1 else 0,
                key="aux_bt_main_interval",
            )
            _aux_start_default = datetime(2020, 1, 1).date()
            try:
                _aux_start_default = start_date
            except Exception:
                pass
            aux_start = iv_col3.date_input("시작일", value=_aux_start_default, key="aux_bt_start")
            aux_fee = iv_col4.number_input("수수료(%)", 0.0, 1.0, 0.05, 0.01, key="aux_bt_fee") / 100.0
            aux_slippage = iv_col5.number_input("슬리피지(%)", 0.0, 2.0, 0.10, 0.05, key="aux_bt_slip")

            def _prepare_aux_frames(_warmup_max: int):
                api_iv = INTERVAL_MAP.get(aux_interval_label, "day")
                api_main_iv = INTERVAL_MAP.get(aux_main_interval_label, api_iv)

                days = max((datetime.now().date() - aux_start).days, 30)
                cpd = CANDLES_PER_DAY.get(api_iv, 1)
                main_cpd = CANDLES_PER_DAY.get(api_main_iv, 1)

                base_warmup = max(int(_warmup_max), int(aux_main_buy), int(aux_main_sell), 30)
                aux_count = min(max(days * cpd + base_warmup + 300, 500), 12000)
                main_count = min(max(days * main_cpd + base_warmup + 300, 500), 12000)

                df_aux_local = data_cache.get_ohlcv_cached(aux_ticker, interval=api_iv, count=aux_count)
                if df_aux_local is None or len(df_aux_local) < max(50, int(_warmup_max) + 5):
                    return None, None, api_iv, api_main_iv, "보조 실행 캔들 데이터가 부족합니다."

                df_main_local = None
                if api_main_iv != api_iv:
                    df_main_local = data_cache.get_ohlcv_cached(aux_ticker, interval=api_main_iv, count=main_count)
                    if df_main_local is None or len(df_main_local) < max(50, int(aux_main_buy) + 5):
                        return None, None, api_iv, api_main_iv, "메인 신호 캔들 데이터가 부족합니다."

                return df_aux_local, df_main_local, api_iv, api_main_iv, None

            run_aux = st.button("보조 전략 백테스트 실행", type="primary", key="run_aux_bt")

            if run_aux:
                with st.spinner("보조 전략 백테스트 실행 중..."):
                    _warmup_bt = int(aux_ma_long) if aux_ma_count_label == "2개" else int(aux_ma_short)
                    if aux_use_rsi:
                        _warmup_bt = max(_warmup_bt, int(aux_rsi_period))
                    df_aux, df_main_aux, api_iv, api_main_iv, _prep_err = _prepare_aux_frames(_warmup_bt)
                    if _prep_err:
                        st.error(_prep_err)
                    elif api_main_iv == api_iv or df_main_aux is not None:
                        res_aux = backtest_engine.run_aux_backtest(
                            df_aux,
                            main_strategy=aux_main_strat,
                            main_buy_p=int(aux_main_buy),
                            main_sell_p=int(aux_main_sell),
                            ma_count=(1 if aux_ma_count_label == "1개" else 2),
                            ma_short=int(aux_ma_short),
                            ma_long=int(aux_ma_long),
                            oversold_threshold=float(aux_threshold),
                            tp1_pct=float(aux_tp1),
                            tp2_pct=float(aux_tp2),
                            fee=float(aux_fee),
                            slippage=float(aux_slippage),
                            start_date=str(aux_start),
                            initial_balance=initial_cap,
                            split_count=int(aux_split),
                            buy_seed_mode=aux_seed_mode,
                            pyramid_ratio=float(aux_pyramid_ratio),
                            use_rsi_filter=bool(aux_use_rsi),
                            rsi_period=int(aux_rsi_period),
                            rsi_threshold=float(aux_rsi_threshold),
                            main_df=(None if api_main_iv == api_iv else df_main_aux),
                        )
                        st.session_state["aux_bt_result"] = res_aux

            if "aux_bt_result" in st.session_state:
                abr = st.session_state["aux_bt_result"]
                if isinstance(abr, dict) and "error" in abr:
                    st.error(f"백테스트 오류: {abr['error']}")
                elif isinstance(abr, dict):
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("총수익률", f"{abr.get('total_return', 0):.2f}%")
                    m2.metric("CAGR", f"{abr.get('cagr', 0):.2f}%")
                    m3.metric("MDD", f"{abr.get('mdd', 0):.2f}%")
                    _calmar = abs(abr.get('cagr', 0) / abr.get('mdd', 1e-9)) if abr.get('mdd', 0) != 0 else 0
                    m4.metric("Calmar", f"{_calmar:.2f}")
                    m5.metric("승률", f"{abr.get('win_rate', 0):.1f}%")
                    m6.metric("거래 수", f"{abr.get('trade_count', 0)}")

                    _seed_mode_out = abr.get("buy_seed_mode", aux_seed_mode)
                    _seed_note = (
                        f" x{abr.get('pyramid_ratio', aux_pyramid_ratio):.2f}"
                        if _seed_mode_out == "pyramiding"
                        else ""
                    )
                    _use_rsi_out = bool(abr.get("use_rsi_filter", aux_use_rsi))
                    _rsi_note = ""
                    if _use_rsi_out:
                        _rsi_note = (
                            f" | RSI({int(abr.get('rsi_period', aux_rsi_period))})"
                            f"<={float(abr.get('rsi_threshold', aux_rsi_threshold)):.1f}"
                        )
                    st.caption(
                        f"시드={_seed_mode_out}{_seed_note}"
                        + f" | MA={aux_ma_count_label}"
                        + f" | split={int(aux_split)}"
                        + _rsi_note
                        + f" | interval={aux_interval_label}/{aux_main_interval_label}"
                    )
                    st.info(
                        f"상태: {abr.get('final_status', 'N/A')} | "
                        f"다음 액션: {abr.get('next_action') if abr.get('next_action') else '-'}"
                    )

                    _dates = abr.get("dates")
                    _strat_ret = abr.get("strategy_return_curve")
                    _bench_ret = abr.get("benchmark_return_curve")
                    _strat_dd = abr.get("drawdown_curve")
                    _bench_dd = abr.get("benchmark_dd_curve")

                    if _dates is not None and _strat_ret is not None and len(_strat_ret) > 1:
                        _plot_df = pd.DataFrame({"date": pd.to_datetime(_dates)})
                        _plot_df["strategy_ret"] = np.asarray(_strat_ret, dtype=float)
                        if _bench_ret is not None and len(_bench_ret) == len(_plot_df):
                            _plot_df["benchmark_ret"] = np.asarray(_bench_ret, dtype=float)
                        if _strat_dd is not None and len(_strat_dd) == len(_plot_df):
                            _plot_df["strategy_dd"] = np.asarray(_strat_dd, dtype=float)
                        if _bench_dd is not None and len(_bench_dd) == len(_plot_df):
                            _plot_df["benchmark_dd"] = np.asarray(_bench_dd, dtype=float)

                        fig_aux = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                        fig_aux.add_trace(
                            go.Scatter(x=_plot_df["date"], y=_plot_df["strategy_ret"], mode="lines", name="Aux Return (%)"),
                            row=1,
                            col=1,
                        )
                        if "benchmark_ret" in _plot_df.columns:
                            fig_aux.add_trace(
                                go.Scatter(x=_plot_df["date"], y=_plot_df["benchmark_ret"], mode="lines", name="B&H Return (%)"),
                                row=1,
                                col=1,
                            )
                        if "strategy_dd" in _plot_df.columns:
                            fig_aux.add_trace(
                                go.Scatter(x=_plot_df["date"], y=_plot_df["strategy_dd"], mode="lines", name="Aux DD (%)"),
                                row=2,
                                col=1,
                            )
                        if "benchmark_dd" in _plot_df.columns:
                            fig_aux.add_trace(
                                go.Scatter(x=_plot_df["date"], y=_plot_df["benchmark_dd"], mode="lines", name="B&H DD (%)"),
                                row=2,
                                col=1,
                            )
                        fig_aux.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=20))
                        fig_aux.update_yaxes(title_text="Return (%)", row=1, col=1)
                        fig_aux.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
                        fig_aux = _apply_return_hover_format(fig_aux)
                        fig_aux = _apply_dd_hover_format(fig_aux)
                        st.plotly_chart(fig_aux, use_container_width=True)

                        _rsi_curve = abr.get("rsi_curve")
                        _rsi_enabled = bool(abr.get("use_rsi_filter", False))
                        if _rsi_enabled and _rsi_curve is not None and len(_rsi_curve) == len(_plot_df):
                            _rsi_thr = float(abr.get("rsi_threshold", aux_rsi_threshold))
                            _rsi_p = int(abr.get("rsi_period", aux_rsi_period))
                            _fig_rsi = go.Figure()
                            _fig_rsi.add_trace(
                                go.Scatter(
                                    x=_plot_df["date"],
                                    y=np.asarray(_rsi_curve, dtype=float),
                                    mode="lines",
                                    name=f"RSI({_rsi_p})",
                                )
                            )
                            _fig_rsi.add_hline(y=_rsi_thr, line_dash="dash", line_color="#ef4444", annotation_text=f"과매도 기준 {_rsi_thr:.1f}")
                            _fig_rsi.add_hline(y=30.0, line_dash="dot", line_color="#9ca3af")
                            _fig_rsi.add_hline(y=70.0, line_dash="dot", line_color="#9ca3af")
                            _fig_rsi.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="RSI")
                            st.plotly_chart(_fig_rsi, use_container_width=True)

            st.divider()
            st.subheader("보조 전략 최적화")
            st.caption("보조 전략 백테스트 아래에서 바로 최적화를 실행합니다.")

            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
            aux_opt_method_label = opt_col1.selectbox("최적화 방식", ["Optuna", "그리드"], key="aux_opt_method")
            if aux_opt_method_label == "Optuna":
                aux_opt_trials = int(opt_col2.number_input("시도 횟수", min_value=20, max_value=500, value=80, step=10, key="aux_opt_trials"))
                aux_opt_max_grid_evals = 30000
            else:
                aux_opt_max_grid_evals = int(
                    opt_col2.number_input(
                        "그리드 최대 평가 수",
                        min_value=500,
                        max_value=200000,
                        value=30000,
                        step=500,
                        key="aux_opt_max_grid_evals",
                    )
                )
                aux_opt_trials = int(st.session_state.get("aux_opt_trials", 80))

            aux_opt_obj_label = opt_col3.selectbox("목표 지표", ["Calmar", "Sharpe", "수익률", "낮은 MDD"], key="aux_opt_objective")
            aux_opt_min_trades = int(opt_col4.number_input("최소 거래 수", min_value=0, max_value=200, value=5, step=1, key="aux_opt_min_trades"))
            aux_opt_use_rsi = st.checkbox("최적화에 RSI 필터 포함", value=bool(aux_use_rsi), key="aux_opt_use_rsi")

            st.caption("최적화 범위")
            ms_col1, ms_col2 = st.columns(2)
            aux_opt_ms_min = int(ms_col1.number_input("단기 MA 최소", min_value=2, max_value=500, value=3, step=1, key="aux_opt_ms_min"))
            aux_opt_ms_max = int(ms_col2.number_input("단기 MA 최대", min_value=2, max_value=500, value=30, step=1, key="aux_opt_ms_max"))

            if aux_ma_count_label == "2개":
                ml_col1, ml_col2 = st.columns(2)
                aux_opt_ml_min = int(ml_col1.number_input("장기 MA 최소", min_value=3, max_value=200, value=10, step=1, key="aux_opt_ml_min"))
                aux_opt_ml_max = int(ml_col2.number_input("장기 MA 최대", min_value=3, max_value=240, value=120, step=1, key="aux_opt_ml_max"))
            else:
                aux_opt_ml_min = int(aux_opt_ms_min)
                aux_opt_ml_max = int(aux_opt_ms_max)
                st.caption("1개 모드에서는 장기 MA를 최적화하지 않습니다.")

            thr_col1, thr_col2 = st.columns(2)
            aux_opt_thr_min = float(thr_col1.number_input("임계값 최소(%)", min_value=-30.0, max_value=-0.5, value=-15.0, step=0.5, key="aux_opt_thr_min"))
            aux_opt_thr_max = float(thr_col2.number_input("임계값 최대(%)", min_value=-30.0, max_value=-0.5, value=-1.0, step=0.5, key="aux_opt_thr_max"))

            tp_row1, tp_row2, split_row1, split_row2 = st.columns(4)
            aux_opt_tp1_min = float(tp_row1.number_input("TP1 최소(%)", min_value=0.5, max_value=30.0, value=2.0, step=0.5, key="aux_opt_tp1_min"))
            aux_opt_tp1_max = float(tp_row2.number_input("TP1 최대(%)", min_value=0.5, max_value=30.0, value=10.0, step=0.5, key="aux_opt_tp1_max"))
            aux_opt_split_min = int(split_row1.number_input("분할수 최소", min_value=1, max_value=20, value=1, step=1, key="aux_opt_split_min"))
            aux_opt_split_max = int(split_row2.number_input("분할수 최대", min_value=1, max_value=20, value=5, step=1, key="aux_opt_split_max"))

            tp2_row1, tp2_row2 = st.columns(2)
            aux_opt_tp2_min = float(tp2_row1.number_input("TP2 최소(%)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, key="aux_opt_tp2_min"))
            aux_opt_tp2_max = float(tp2_row2.number_input("TP2 최대(%)", min_value=0.5, max_value=50.0, value=20.0, step=0.5, key="aux_opt_tp2_max"))

            if aux_opt_use_rsi:
                rsi_row1, rsi_row2 = st.columns(2)
                aux_opt_rsi_p_min = int(rsi_row1.number_input("RSI 기간 최소", min_value=2, max_value=50, value=2, step=1, key="aux_opt_rsi_p_min"))
                aux_opt_rsi_p_max = int(rsi_row2.number_input("RSI 기간 최대", min_value=2, max_value=50, value=10, step=1, key="aux_opt_rsi_p_max"))
                rsi_t_row1, rsi_t_row2 = st.columns(2)
                aux_opt_rsi_t_min = float(rsi_t_row1.number_input("RSI 기준 최소", min_value=1.0, max_value=30.0, value=5.0, step=0.5, key="aux_opt_rsi_t_min"))
                aux_opt_rsi_t_max = float(rsi_t_row2.number_input("RSI 기준 최대", min_value=1.0, max_value=30.0, value=10.0, step=0.5, key="aux_opt_rsi_t_max"))
            else:
                aux_opt_rsi_p_min = int(aux_rsi_period)
                aux_opt_rsi_p_max = int(aux_rsi_period)
                aux_opt_rsi_t_min = float(aux_rsi_threshold)
                aux_opt_rsi_t_max = float(aux_rsi_threshold)

            aux_opt_ms_max = max(aux_opt_ms_min, aux_opt_ms_max)
            if aux_ma_count_label == "2개":
                aux_opt_ml_min = max(aux_opt_ms_min + 1, aux_opt_ml_min)
                aux_opt_ml_max = max(aux_opt_ml_min, aux_opt_ml_max)
            else:
                aux_opt_ml_min = aux_opt_ms_min
                aux_opt_ml_max = aux_opt_ms_max
            if aux_opt_thr_min > aux_opt_thr_max:
                aux_opt_thr_min, aux_opt_thr_max = aux_opt_thr_max, aux_opt_thr_min
            if aux_opt_tp1_min > aux_opt_tp1_max:
                aux_opt_tp1_min, aux_opt_tp1_max = aux_opt_tp1_max, aux_opt_tp1_min
            if aux_opt_tp2_min > aux_opt_tp2_max:
                aux_opt_tp2_min, aux_opt_tp2_max = aux_opt_tp2_max, aux_opt_tp2_min
            if aux_opt_tp2_min < aux_opt_tp1_min:
                aux_opt_tp2_min = aux_opt_tp1_min
            if aux_opt_tp2_max < aux_opt_tp2_min:
                aux_opt_tp2_max = aux_opt_tp2_min
            if aux_opt_split_min > aux_opt_split_max:
                aux_opt_split_min, aux_opt_split_max = aux_opt_split_max, aux_opt_split_min
            if aux_opt_rsi_p_min > aux_opt_rsi_p_max:
                aux_opt_rsi_p_min, aux_opt_rsi_p_max = aux_opt_rsi_p_max, aux_opt_rsi_p_min
            if aux_opt_rsi_t_min > aux_opt_rsi_t_max:
                aux_opt_rsi_t_min, aux_opt_rsi_t_max = aux_opt_rsi_t_max, aux_opt_rsi_t_min

            if aux_opt_method_label == "그리드":
                st.caption("그리드 간격")
                gs1, gs2, gs3, gs4 = st.columns(4)
                aux_opt_ms_step = int(gs1.number_input("단기 MA 간격", min_value=1, max_value=100, value=5, step=1, key="aux_opt_ms_step"))
                if aux_ma_count_label == "2개":
                    aux_opt_ml_step = int(gs2.number_input("장기 MA 간격", min_value=1, max_value=100, value=5, step=1, key="aux_opt_ml_step"))
                else:
                    aux_opt_ml_step = int(max(1, aux_opt_ms_step))
                    gs2.caption("1개 모드: 장기 MA 간격 미사용")
                aux_opt_thr_step = float(gs3.number_input("임계값 간격(%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="aux_opt_thr_step"))
                aux_opt_tp_step = float(gs4.number_input("TP 간격(%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="aux_opt_tp_step"))
                aux_opt_split_step = int(st.number_input("분할수 간격", min_value=1, max_value=5, value=1, step=1, key="aux_opt_split_step"))
                if aux_opt_use_rsi:
                    gr1, gr2 = st.columns(2)
                    aux_opt_rsi_p_step = int(gr1.number_input("RSI 기간 간격", min_value=1, max_value=10, value=1, step=1, key="aux_opt_rsi_p_step"))
                    aux_opt_rsi_t_step = float(gr2.number_input("RSI 기준 간격", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="aux_opt_rsi_t_step"))
                else:
                    aux_opt_rsi_p_step = 1
                    aux_opt_rsi_t_step = 0.5

                def _grid_count(_vmin, _vmax, _step):
                    _step = max(float(_step), 1e-9)
                    _span = max(float(_vmax) - float(_vmin), 0.0)
                    return int(np.floor(_span / _step + 1e-9)) + 1

                _ms_n = _grid_count(aux_opt_ms_min, aux_opt_ms_max, aux_opt_ms_step)
                _ml_n = 1 if aux_ma_count_label == "1개" else _grid_count(aux_opt_ml_min, aux_opt_ml_max, aux_opt_ml_step)
                _thr_n = _grid_count(aux_opt_thr_min, aux_opt_thr_max, aux_opt_thr_step)
                _tp1_n = _grid_count(aux_opt_tp1_min, aux_opt_tp1_max, aux_opt_tp_step)
                _tp2_n = _grid_count(aux_opt_tp2_min, aux_opt_tp2_max, aux_opt_tp_step)
                _sp_n = _grid_count(aux_opt_split_min, aux_opt_split_max, aux_opt_split_step)
                _rsi_p_n = _grid_count(aux_opt_rsi_p_min, aux_opt_rsi_p_max, aux_opt_rsi_p_step) if aux_opt_use_rsi else 1
                _rsi_t_n = _grid_count(aux_opt_rsi_t_min, aux_opt_rsi_t_max, aux_opt_rsi_t_step) if aux_opt_use_rsi else 1
                _est_grid = int(_ms_n * _ml_n * _thr_n * _tp1_n * _tp2_n * _sp_n * _rsi_p_n * _rsi_t_n)
                st.caption(f"예상 조합 수(필터 전): 약 {_est_grid:,}개 | 최대 평가 수: {int(aux_opt_max_grid_evals):,}개")
            else:
                aux_opt_ms_step = 1
                aux_opt_ml_step = 1
                aux_opt_thr_step = 0.5
                aux_opt_tp_step = 0.5
                aux_opt_split_step = 1
                aux_opt_rsi_p_step = 1
                aux_opt_rsi_t_step = 0.5

            run_aux_opt = st.button("보조 전략 최적화 실행", type="secondary", key="run_aux_opt")
            if run_aux_opt:
                with st.spinner("보조 전략 최적화 실행 중..."):
                    _objective_map = {"Calmar": "calmar", "Sharpe": "sharpe", "수익률": "return", "낮은 MDD": "mdd"}
                    _warmup_opt = int(aux_opt_ml_max) if aux_ma_count_label == "2개" else int(aux_opt_ms_max)
                    if aux_opt_use_rsi:
                        _warmup_opt = max(_warmup_opt, int(aux_opt_rsi_p_max))
                    df_aux_opt, df_main_aux_opt, api_iv_opt, api_main_iv_opt, _prep_err = _prepare_aux_frames(_warmup_opt)
                    if _prep_err:
                        st.error(_prep_err)
                    else:
                        _pbar = st.progress(0)
                        _pmsg = st.empty()

                        def _aux_opt_progress(cur, total, msg):
                            _pct = int((float(cur) / max(float(total), 1.0)) * 100.0)
                            _pbar.progress(max(0, min(100, _pct)))
                            _pmsg.caption(f"{cur}/{total} | {msg}")

                        try:
                            _opt_result = backtest_engine.optimize_aux(
                                df_aux_opt,
                                main_strategy=aux_main_strat,
                                main_buy_p=int(aux_main_buy),
                                main_sell_p=int(aux_main_sell),
                                ma_count=(1 if aux_ma_count_label == "1개" else 2),
                                ma_short_range=(int(aux_opt_ms_min), int(aux_opt_ms_max)),
                                ma_long_range=(int(aux_opt_ml_min), int(aux_opt_ml_max)),
                                threshold_range=(float(aux_opt_thr_min), float(aux_opt_thr_max)),
                                tp1_range=(float(aux_opt_tp1_min), float(aux_opt_tp1_max)),
                                tp2_range=(float(aux_opt_tp2_min), float(aux_opt_tp2_max)),
                                split_count_range=(int(aux_opt_split_min), int(aux_opt_split_max)),
                                fee=float(aux_fee),
                                slippage=float(aux_slippage),
                                start_date=str(aux_start),
                                initial_balance=initial_cap,
                                n_trials=int(aux_opt_trials),
                                objective_metric=_objective_map.get(aux_opt_obj_label, "calmar"),
                                progress_callback=_aux_opt_progress,
                                buy_seed_mode=aux_seed_mode,
                                pyramid_ratio=float(aux_pyramid_ratio),
                                main_df=(None if api_main_iv_opt == api_iv_opt else df_main_aux_opt),
                                min_trade_count=int(aux_opt_min_trades),
                                optimization_method=("grid" if aux_opt_method_label == "그리드" else "optuna"),
                                ma_short_step=int(aux_opt_ms_step),
                                ma_long_step=int(aux_opt_ml_step),
                                threshold_step=float(aux_opt_thr_step),
                                tp_step=float(aux_opt_tp_step),
                                split_step=int(aux_opt_split_step),
                                max_grid_evals=int(aux_opt_max_grid_evals),
                                use_rsi_filter=bool(aux_opt_use_rsi),
                                rsi_period_range=(int(aux_opt_rsi_p_min), int(aux_opt_rsi_p_max)),
                                rsi_threshold_range=(float(aux_opt_rsi_t_min), float(aux_opt_rsi_t_max)),
                                rsi_period_step=int(aux_opt_rsi_p_step),
                                rsi_threshold_step=float(aux_opt_rsi_t_step),
                            )
                            _pbar.progress(100)
                            _pmsg.caption("최적화 완료")
                            st.session_state["aux_opt_result"] = {
                                "raw": _opt_result,
                                "method_label": aux_opt_method_label,
                                "objective_label": aux_opt_obj_label,
                                "ma_label": aux_ma_count_label,
                                "ticker": aux_ticker,
                                "interval": f"{aux_interval_label}/{aux_main_interval_label}",
                            }
                        except Exception as e:
                            st.session_state["aux_opt_result"] = {"error": str(e)}
                            st.error(f"보조 전략 최적화 오류: {e}")

            if "aux_opt_result" in st.session_state:
                _aor = st.session_state["aux_opt_result"]
                st.markdown("#### 최적화 결과")
                if isinstance(_aor, dict) and _aor.get("error"):
                    st.error(_aor.get("error", "알 수 없는 오류"))
                else:
                    _opt_rows = ((_aor or {}).get("raw", {}) or {}).get("trials", [])
                    if not _opt_rows:
                        st.info("최적화 결과가 없습니다.")
                    else:
                        _opt_df = pd.DataFrame(_opt_rows)
                        if "score" in _opt_df.columns:
                            _opt_df = _opt_df.sort_values("score", ascending=False).reset_index(drop=True)
                        _best = _opt_df.iloc[0]

                        b1, b2, b3, b4, b5, b6 = st.columns(6)
                        b1.metric("최적 점수", f"{float(_best.get('score', 0.0)):.2f}")
                        b2.metric("총수익률", f"{float(_best.get('total_return', 0.0)):.2f}%")
                        b3.metric("CAGR", f"{float(_best.get('cagr', 0.0)):.2f}%")
                        b4.metric("MDD", f"{float(_best.get('mdd', 0.0)):.2f}%")
                        b5.metric("Calmar", f"{float(_best.get('calmar', 0.0)):.2f}")
                        b6.metric("거래 수", f"{int(_best.get('trade_count', 0))}")

                        _raw_opt = ((_aor or {}).get("raw", {}) or {})
                        _eval_cnt = int(_raw_opt.get("evaluated_count", len(_opt_rows)))
                        _est_cnt = _raw_opt.get("total_estimated", None)
                        _method_lbl = str(_aor.get("method_label", "Optuna"))

                        st.caption(
                            f"방식={_method_lbl} | "
                            f"평가={_eval_cnt:,}건"
                            + (f" / 예상={int(_est_cnt):,}건" if _est_cnt is not None else "")
                            + " | "
                            f"목표={_aor.get('objective_label', 'Calmar')} | "
                            f"MA={_aor.get('ma_label', aux_ma_count_label)} | "
                            f"티커={_aor.get('ticker', aux_ticker)} | "
                            f"주기={_aor.get('interval', f'{aux_interval_label}/{aux_main_interval_label}')}"
                        )

                        _top_n = int(st.number_input("결과 표시 개수", min_value=5, max_value=200, value=30, step=5, key="aux_opt_top_n"))
                        _view_cols = [
                            "MA Count", "MA Short", "MA Long", "Threshold", "TP1 %", "TP2 %", "Split",
                            "Use RSI", "RSI Period", "RSI Threshold",
                            "total_return", "cagr", "mdd", "calmar", "sharpe", "win_rate", "trade_count", "score",
                        ]
                        _view_cols = [c for c in _view_cols if c in _opt_df.columns]
                        _show_df = _opt_df[_view_cols].head(_top_n).copy()
                        _show_df = _show_df.rename(columns={
                            "MA Count": "MA수",
                            "MA Short": "단기MA",
                            "MA Long": "장기MA",
                            "Threshold": "임계(%)",
                            "TP1 %": "TP1(%)",
                            "TP2 %": "TP2(%)",
                            "Split": "분할수",
                            "Use RSI": "RSI사용",
                            "RSI Period": "RSI기간",
                            "RSI Threshold": "RSI기준",
                            "total_return": "총수익률(%)",
                            "cagr": "CAGR(%)",
                            "mdd": "MDD(%)",
                            "calmar": "Calmar",
                            "sharpe": "Sharpe",
                            "win_rate": "승률(%)",
                            "trade_count": "거래수",
                            "score": "점수",
                        })
                        if "RSI사용" in _show_df.columns:
                            _show_df["RSI사용"] = _show_df["RSI사용"].map(lambda v: "예" if bool(v) else "아니오")
                        _grad_cols = [c for c in ["총수익률(%)", "CAGR(%)", "Calmar", "Sharpe", "승률(%)", "점수"] if c in _show_df.columns]
                        _num_cols = [c for c in _show_df.columns if pd.api.types.is_numeric_dtype(_show_df[c])]
                        st.dataframe(
                            _show_df.style.background_gradient(cmap="RdYlGn", subset=_grad_cols)
                            .background_gradient(cmap="RdYlGn_r", subset=[c for c in ["MDD(%)"] if c in _show_df.columns])
                            .format("{:,.2f}", subset=_num_cols),
                            use_container_width=True,
                            hide_index=True,
                        )

                        _best_ma_count = int(_best.get("MA Count", 2))
                        _best_ma_short = int(_best.get("MA Short", aux_ma_short))
                        _best_ma_long = int(_best.get("MA Long", aux_ma_long))
                        _best_thr = float(_best.get("Threshold", aux_threshold))
                        _best_tp1 = float(_best.get("TP1 %", aux_tp1))
                        _best_tp2 = float(_best.get("TP2 %", aux_tp2))
                        _best_split = int(_best.get("Split", aux_split))
                        _best_use_rsi = bool(_best.get("Use RSI", aux_opt_use_rsi))
                        _best_rsi_p = int(_best.get("RSI Period", aux_rsi_period))
                        _best_rsi_t = float(_best.get("RSI Threshold", aux_rsi_threshold))

                        # 최적 파라미터 기준 보조 전략 자체 성과/DD 차트
                        _curve_sig = (
                            str(aux_ticker),
                            str(aux_interval_label),
                            str(aux_main_interval_label),
                            str(aux_start),
                            float(aux_fee),
                            float(aux_slippage),
                            int(aux_main_buy),
                            int(aux_main_sell),
                            int(_best_ma_count),
                            int(_best_ma_short),
                            int(_best_ma_long),
                            float(_best_thr),
                            float(_best_tp1),
                            float(_best_tp2),
                            int(_best_split),
                            bool(_best_use_rsi),
                            int(_best_rsi_p),
                            float(_best_rsi_t),
                            str(aux_seed_mode),
                            float(aux_pyramid_ratio),
                            int(initial_cap),
                        )
                        if st.session_state.get("aux_opt_best_curve_sig") != _curve_sig:
                            _warmup_curve = int(_best_ma_long) if _best_ma_count == 2 else int(_best_ma_short)
                            if _best_use_rsi:
                                _warmup_curve = max(_warmup_curve, int(_best_rsi_p))
                            _df_aux_curve, _df_main_curve, _api_iv_curve, _api_main_iv_curve, _curve_prep_err = _prepare_aux_frames(_warmup_curve)
                            if _curve_prep_err:
                                st.session_state["aux_opt_best_curve"] = {"error": _curve_prep_err}
                            else:
                                with st.spinner("최적 파라미터 성과/DD 차트 계산 중..."):
                                    _curve_res = backtest_engine.run_aux_backtest(
                                        _df_aux_curve,
                                        main_strategy=aux_main_strat,
                                        main_buy_p=int(aux_main_buy),
                                        main_sell_p=int(aux_main_sell),
                                        ma_count=int(_best_ma_count),
                                        ma_short=int(_best_ma_short),
                                        ma_long=int(_best_ma_long),
                                        oversold_threshold=float(_best_thr),
                                        tp1_pct=float(_best_tp1),
                                        tp2_pct=float(_best_tp2),
                                        fee=float(aux_fee),
                                        slippage=float(aux_slippage),
                                        start_date=str(aux_start),
                                        initial_balance=initial_cap,
                                        split_count=int(_best_split),
                                        buy_seed_mode=aux_seed_mode,
                                        pyramid_ratio=float(aux_pyramid_ratio),
                                        use_rsi_filter=bool(_best_use_rsi),
                                        rsi_period=int(_best_rsi_p),
                                        rsi_threshold=float(_best_rsi_t),
                                        main_df=(None if _api_main_iv_curve == _api_iv_curve else _df_main_curve),
                                    )
                                st.session_state["aux_opt_best_curve"] = _curve_res
                            st.session_state["aux_opt_best_curve_sig"] = _curve_sig

                        _curve_show = st.session_state.get("aux_opt_best_curve", {})
                        if isinstance(_curve_show, dict) and _curve_show.get("error"):
                            st.warning(f"최적 파라미터 차트 생성 실패: {_curve_show.get('error')}")
                        elif isinstance(_curve_show, dict):
                            _dates2 = _curve_show.get("dates")
                            _ret2 = _curve_show.get("strategy_return_curve")
                            _dd2 = _curve_show.get("drawdown_curve")
                            _bench_ret2 = _curve_show.get("benchmark_return_curve")
                            _bench_dd2 = _curve_show.get("benchmark_dd_curve")
                            _disp_s2 = _curve_show.get("disparity_short_curve")
                            _disp_l2 = _curve_show.get("disparity_long_curve")
                            _thr2 = float(_curve_show.get("oversold_threshold", _best_thr))
                            _ma_count2 = int(_curve_show.get("ma_count", _best_ma_count))
                            _ma_s2 = int(_curve_show.get("ma_short", _best_ma_short))
                            _ma_l2 = int(_curve_show.get("ma_long", _best_ma_long))
                            _use_rsi2 = bool(_curve_show.get("use_rsi_filter", _best_use_rsi))
                            _rsi_p2 = int(_curve_show.get("rsi_period", _best_rsi_p))
                            _rsi_thr2 = float(_curve_show.get("rsi_threshold", _best_rsi_t))
                            _rsi2 = _curve_show.get("rsi_curve")
                            if _dates2 is not None and _ret2 is not None and len(_ret2) > 1:
                                _cdf = pd.DataFrame({"date": pd.to_datetime(_dates2)})
                                _cdf["strategy_ret"] = np.asarray(_ret2, dtype=float)
                                if _bench_ret2 is not None and len(_bench_ret2) == len(_cdf):
                                    _cdf["benchmark_ret"] = np.asarray(_bench_ret2, dtype=float)
                                if _dd2 is not None and len(_dd2) == len(_cdf):
                                    _cdf["strategy_dd"] = np.asarray(_dd2, dtype=float)
                                if _bench_dd2 is not None and len(_bench_dd2) == len(_cdf):
                                    _cdf["benchmark_dd"] = np.asarray(_bench_dd2, dtype=float)

                                st.markdown("##### 보조 전략 자체 성과 차트")
                                _fig_perf = go.Figure()
                                _fig_perf.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["strategy_ret"], mode="lines", name="보조 전략 수익률(%)"))
                                if "benchmark_ret" in _cdf.columns:
                                    _fig_perf.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["benchmark_ret"], mode="lines", name="단순보유 수익률(%)", line=dict(dash="dot")))
                                _fig_perf.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="수익률(%)")
                                _fig_perf = _apply_return_hover_format(_fig_perf, apply_all=True)
                                st.plotly_chart(_fig_perf, use_container_width=True)

                                st.markdown("##### 보조 전략 DD 차트")
                                _fig_dd = go.Figure()
                                if "strategy_dd" in _cdf.columns:
                                    _fig_dd.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["strategy_dd"], mode="lines", name="보조 전략 DD(%)"))
                                if "benchmark_dd" in _cdf.columns:
                                    _fig_dd.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["benchmark_dd"], mode="lines", name="단순보유 DD(%)", line=dict(dash="dot")))
                                _fig_dd.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="DD(%)")
                                _fig_dd = _apply_dd_hover_format(_fig_dd)
                                st.plotly_chart(_fig_dd, use_container_width=True)

                                if _disp_s2 is not None and len(_disp_s2) == len(_cdf):
                                    st.markdown("##### 기준 이평선 이격도 차트")
                                    _fig_disp = go.Figure()
                                    _fig_disp.add_trace(
                                        go.Scatter(
                                            x=_cdf["date"],
                                            y=np.asarray(_disp_s2, dtype=float),
                                            mode="lines",
                                            name=f"단기MA({_ma_s2}) 이격도(%)",
                                        )
                                    )
                                    if _ma_count2 == 2 and _disp_l2 is not None and len(_disp_l2) == len(_cdf):
                                        _fig_disp.add_trace(
                                            go.Scatter(
                                                x=_cdf["date"],
                                                y=np.asarray(_disp_l2, dtype=float),
                                                mode="lines",
                                                name=f"장기MA({_ma_l2}) 이격도(%)",
                                                line=dict(dash="dot"),
                                            )
                                        )
                                    _fig_disp.add_hline(y=0.0, line_dash="dot", line_color="#9ca3af")
                                    _fig_disp.add_hline(
                                        y=float(_thr2),
                                        line_dash="dash",
                                        line_color="#ef4444",
                                        annotation_text=f"과매도 임계값 {_thr2:.2f}%",
                                        annotation_position="top right",
                                    )
                                    _fig_disp.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="이격도(%)")
                                    st.plotly_chart(_fig_disp, use_container_width=True)

                                if _use_rsi2 and _rsi2 is not None and len(_rsi2) == len(_cdf):
                                    st.markdown("##### RSI 차트")
                                    _fig_rsi2 = go.Figure()
                                    _fig_rsi2.add_trace(
                                        go.Scatter(
                                            x=_cdf["date"],
                                            y=np.asarray(_rsi2, dtype=float),
                                            mode="lines",
                                            name=f"RSI({_rsi_p2})",
                                        )
                                    )
                                    _fig_rsi2.add_hline(y=float(_rsi_thr2), line_dash="dash", line_color="#ef4444", annotation_text=f"RSI 기준 {_rsi_thr2:.1f}")
                                    _fig_rsi2.add_hline(y=30.0, line_dash="dot", line_color="#9ca3af")
                                    _fig_rsi2.add_hline(y=70.0, line_dash="dot", line_color="#9ca3af")
                                    _fig_rsi2.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="RSI")
                                    st.plotly_chart(_fig_rsi2, use_container_width=True)

                        if st.button("최적 파라미터를 현재 입력값에 반영", key="aux_opt_apply_best"):
                            st.session_state["aux_opt_apply_pending"] = {
                                "aux_bt_ma_count": "1개" if _best_ma_count == 1 else "2개",
                                "aux_bt_ma_short": int(_best_ma_short),
                                "aux_bt_ma_long": int(max(5, _best_ma_long)),
                                "aux_bt_threshold": float(_best_thr),
                                "aux_bt_tp1": float(_best_tp1),
                                "aux_bt_tp2": float(_best_tp2),
                                "aux_bt_split": int(_best_split),
                                "aux_bt_use_rsi": bool(_best_use_rsi),
                                "aux_bt_rsi_period": int(_best_rsi_p),
                                "aux_bt_rsi_threshold": float(_best_rsi_t),
                            }
                            st.rerun()

        with bt_sub3:
            st.header("전체 종목 스캔")
            st.caption("상위 종목을 전 시간대/전략으로 백테스트하여 Calmar 순으로 정렬합니다.")

            # 스캔 설정
            scan_col1, scan_col2, scan_col3 = st.columns(3)
            scan_strategy = scan_col1.selectbox("전략", ["SMA", "Donchian"], key="scan_strat")
            scan_period = scan_col2.number_input("기간 (Period)", 5, 300, 20, key="scan_period")
            scan_count = scan_col3.number_input("백테스트 캔들 수", 200, 10000, 2000, step=200, key="scan_count")

            scan_col4, scan_col5 = st.columns(2)
            _scan_interval_alias = {
                "일봉": "1D",
                "4시간": "4H",
                "1시간": "1H",
                "30분": "30m",
                "15분": "15m",
                "5분": "5m",
                "1분": "1m",
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

            st.caption(f"대상: 시가총액 상위 {len(TOP_20_TICKERS)}개 — {', '.join(t.replace('KRW-','') for t in TOP_20_TICKERS)}")

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
                                # 데이터 조회 (캐시 우선)
                                df = data_cache.get_ohlcv_cached(ticker, interval=interval_api, count=scan_count)
                                if df is None or len(df) < scan_period + 10:
                                    continue

                                df = df.copy()

                                # 시그널 생성
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

                                # 고속 시뮬레이션
                                res = engine._fast_simulate(open_arr, close_arr, signal_arr, fee=0.0005, slippage=0.0, initial_balance=1000000)

                                # Buy & Hold 수익률
                                bnh_return = (close_arr[-1] / close_arr[0] - 1) * 100

                                # Calmar = CAGR / |MDD| (MDD가 0이면 inf 방지)
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
                        df_results.index = df_results.index + 1  # 1부터 시작
                        df_results.index.name = "순위"

                        # 요약
                        st.success(f"스캔 완료: {len(results)}건 중 수익 {len(df_results[df_results['수익률 (%)'] > 0])}건, 손실 {len(df_results[df_results['수익률 (%)'] <= 0])}건")

                        # Calmar 상위 결과 테이블
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

                        # 전략별/시간봉별 요약
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


