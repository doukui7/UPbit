import streamlit as st
import pandas as pd
import time
import os
import json
from datetime import datetime
from src.constants import *
from src.backtest.engine import BacktestEngine
from src.trading.upbit_trader import UpbitTrader
import src.engine.data_cache as data_cache

from src.ui.components.triggers import render_strategy_trigger_tab
from src.ui.components.ops_log import render_ops_log_tab
from src.ui.components.scheduled_orders import render_scheduled_orders_tab
from src.ui.components.backtest_tab import render_backtest_tab
from src.ui.components.backtest_opt_tab import render_optimization_tab, render_scan_tab
from src.ui.components.live_portfolio import render_live_portfolio_tab
from src.ui.components.manual_trade import render_manual_trade_tab
from src.ui.components.trade_history import render_trade_history_tab
from src.ui.components.asset_mgmt import render_asset_mgmt_tab
from src.ui.coin_utils import (
    load_balance_cache as _load_balance_cache,
    sync_account_cache_from_github as _sync_account_cache_from_github,
    trigger_and_wait_gh as _trigger_and_wait_gh,
)

def render_coin_mode(config, save_config):
    from src.utils.helpers import load_mode_config, save_mode_config
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
    PORTFOLIO_JSON_LOAD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "portfolio.json")
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

    _coin_cfg = load_mode_config("coin")
    default_portfolio = _coin_cfg.get("portfolio", None) or config.get("portfolio", None) or _pjson_config.get("portfolio", None)
    default_aux_portfolio = _coin_cfg.get("aux_portfolio", None) or config.get("aux_portfolio", None)
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
    
    from src.ui.components.asset_mgmt import get_earliest_start_date, get_total_invested
    _am_earliest = get_earliest_start_date("coin")
    default_start_str = _am_earliest or _coin_cfg.get("start_date", None) or config.get("start_date", None) or _pjson_config.get("start_date", None)
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
    _am_invested = get_total_invested("coin")
    if _am_invested > 0:
        default_cap = int(_am_invested)
    else:
        default_cap = _coin_cfg.get("initial_cap", None) or config.get("initial_cap", None) or _pjson_config.get("initial_cap", None)
    if not default_cap:
        st.error("initial_cap 설정이 없습니다. 로컬에서 portfolio.json에 initial_cap을 설정 후 push 해주세요.")
        st.stop()
    initial_cap = st.sidebar.number_input(
        "초기 자본금 (KRW - 원 단위)",
        value=default_cap, step=100000, format="%d",
        help="자산관리 탭의 입출금 내역에서 자동 반영됩니다. 직접 수정도 가능합니다.",
        disabled=IS_CLOUD
    )
    _cap_src = "자산관리" if _am_invested > 0 else "설정"
    st.sidebar.caption(f"{_cap_src}: **{initial_cap:,.0f} KRW**")
    
    # Strategy Selection REMOVED (Moved to Per-Coin)

    PORTFOLIO_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "portfolio.json")

    if not IS_CLOUD:
        if st.sidebar.button("Save"):
            coin_data = {
                "portfolio": portfolio_list,
                "aux_portfolio": aux_portfolio_list,
                "start_date": str(start_date),
                "initial_cap": initial_cap,
            }
            # 모드별 설정 파일 저장 (config/coin.json)
            save_mode_config("coin", coin_data)
            # 전역 config에도 반영 (하위호환)
            new_config = config.copy()
            new_config.update(coin_data)
            save_config(new_config)
            # portfolio.json도 동기화 (GitHub Actions 호환)
            with open(PORTFOLIO_JSON, "w", encoding="utf-8") as f:
                json.dump(coin_data, f, indent=2, ensure_ascii=False)
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
    _worker_cache_version = "worker-cache-v2-2026-03-02"

    @st.cache_resource
    def get_worker(_cache_ver: str = _worker_cache_version):
        # Reset singleton when cache version changes so patched code is applied.
        try:
            MarketDataWorker._instance = None
        except Exception:
            pass
        return MarketDataWorker()

    worker = get_worker()

    # 공유 유틸리티는 coin_utils에서 import

    # ═══════════════════════════════════════════════════════════════
    # 실시간 가격 & 자산 — 대시보드 최상단 (10초 자동 갱신)
    # ═══════════════════════════════════════════════════════════════
    # fragment 내부에서 참조할 변수를 closure로 캡처
    _frag_portfolio = portfolio_list
    # 자산관리 탭의 총 투입원금이 있으면 우선 사용
    _frag_initial_cap = float(st.session_state.get("_asset_mgmt_total_invested_coin", 0) or 0)
    if _frag_initial_cap <= 0:
        _frag_initial_cap = initial_cap

    @st.fragment(run_every=10)
    def _render_live_ticker():
        _prices = {}
        _balances = {}
        _krw = 0.0
        _from_cache = False
        _cache_time = ""

        # 잔고 동기화 버튼 (fragment 내부)
        _sync_col1, _sync_col2 = st.columns([1, 5])
        with _sync_col1:
            if st.button("잔고 동기화", key="top_bal_sync", help="VM 경유로 최신 잔고를 가져옵니다"):
                _sync_status = _sync_col2.empty()
                _sync_status.info("VM에서 최신 잔고 조회 중...")
                _ok, _msg = _trigger_and_wait_gh("account_sync", _sync_status)
                if _ok:
                    _sync_status.success(f"잔고 동기화 완료 ({_msg})")
                else:
                    # GH Actions 실패 시 기존 캐시라도 pull
                    _sync_account_cache_from_github()
                    _sync_status.warning(f"동기화: {_msg} — 기존 캐시 로드")
                st.rerun(scope="fragment")

        # 1) 잔고: GitHub에서 pull한 캐시 사용 (로컬 API는 IP 차단으로 불가)
        # 60초마다 자동으로 git에서 최신 캐시 pull
        _bal_pull_key = "__t_bal_git_pull"
        _now_ts = time.time()
        if (_now_ts - st.session_state.get(_bal_pull_key, 0)) > 60:
            _sync_account_cache_from_github()
            st.session_state[_bal_pull_key] = _now_ts

        _cached = _load_balance_cache()
        if _cached and _cached.get("balances"):
            _bal = _cached["balances"]
            _krw = float(_bal.get('KRW', 0) or 0)
            for _c, _v in _bal.items():
                _c = str(_c).upper()
                if _c != 'KRW':
                    try:
                        _fv = float(_v or 0)
                        if _fv > 0:
                            _balances[_c] = _fv
                    except (ValueError, TypeError):
                        pass
            _from_cache = True
            _cache_time = _cached.get("updated_at", "")

        # 2) 가격: Public API (IP 차단 무관, 항상 최신)
        _coins = set(_balances.keys())
        if _frag_portfolio:
            _coins.update(item['coin'].upper() for item in _frag_portfolio)
        _coins.add("BTC")
        _tickers = [f"KRW-{c}" for c in sorted(_coins)]
        try:
            _prices = data_cache.get_current_prices_local_first(
                _tickers, ttl_sec=3.0, allow_api_fallback=True,
            ) or {}
        except Exception:
            pass

        # 3) 계산
        _coin_val = sum(
            bal * (_prices.get(f"KRW-{c}", 0) or 0)
            for c, bal in _balances.items()
        )
        _total = _krw + _coin_val
        _pnl = _total - _frag_initial_cap
        _pnl_pct = (_pnl / _frag_initial_cap * 100) if _frag_initial_cap > 0 else 0
        _btc = _prices.get("KRW-BTC", 0) or 0
        _now = datetime.now().strftime("%H:%M:%S")

        # 4) 표시
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("BTC", f"{_btc:,.0f}원")
        _c2.metric("총 자산", f"{_total:,.0f}원")
        _c3.metric("손익 (P&L)", f"{_pnl:+,.0f}원 ({_pnl_pct:+.2f}%)")
        _c4.metric("현금 (KRW)", f"{_krw:,.0f}원")
        _note = f"가격: {_now}"
        _cache_stale = False
        if _from_cache:
            _cache_lag_str = ""
            try:
                # 캐시 시간 파싱 → 현재 시각과 차이 계산
                _ct = _cache_time.replace(" KST", "").strip()
                _cache_dt = datetime.strptime(_ct, "%Y-%m-%d %H:%M:%S")
                _lag_sec = (datetime.now() - _cache_dt).total_seconds()
                if _lag_sec < 0:
                    _lag_sec = 0
                _lag_min = int(_lag_sec // 60)
                _lag_s = int(_lag_sec % 60)
                _cache_lag_str = f" ({_lag_min}분 {_lag_s}초 전)"
                # 10분 이상 지연이면 stale 표시
                if _lag_sec > 600:
                    _cache_stale = True
            except Exception:
                pass
            _note += f" | 잔고: VM캐시 {_cache_time}{_cache_lag_str}"
        # 워커 상태도 같은 줄에 표시
        try:
            _w_msg, _w_time = worker.get_status()
            _note += f" | 워커: {_w_msg} ({_w_time})"
        except Exception:
            pass
        st.caption(_note)

        # ── 캐시 지연 자동 복구: 10분 초과 시 강제 재동기화 ──
        if _cache_stale:
            _recovery_key = "__cache_recovery_ts"
            _last_recovery = st.session_state.get(_recovery_key, 0)
            if time.time() - _last_recovery > 120:  # 복구 시도 간격: 2분
                st.session_state[_recovery_key] = time.time()
                # 강제 git fetch + checkout + ff-only
                _sync_account_cache_from_github()
                # 동기화 타이머도 리셋
                st.session_state["__last_auto_sync"] = time.time()
                st.toast("⚠️ 잔고 캐시 10분+ 지연 → 강제 재동기화 실행", icon="🔄")
                st.rerun()

    _render_live_ticker()

    # ── 60초마다 전체 페이지 자동 갱신 (모든 탭에서 동작) ──
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, limit=None, key="coin_page_refresh")
    except ImportError:
        pass

    # --- Tabs ---
    tab1, tab_orders, tab5, tab3, tab4, tab_am, tab6, tab7 = st.tabs(["🚀 실시간 포트폴리오", "📋 예약 주문", "🛒 수동 주문", "📜 거래 내역", "📊 백테스트", "💰 자산관리", "⏰ 트리거", "📝 운영 로그"])

    # --- Tab 1: Live Portfolio ---
    with tab1:
        render_live_portfolio_tab(
            trader, worker, portfolio_list, initial_cap, config, save_config,
            start_date, backtest_engine, INTERVAL_REV_MAP,
        )

    # --- Tab Orders: 예약 주문 ---
    with tab_orders:
        render_scheduled_orders_tab(portfolio_list, initial_cap, config, save_config)

    # --- Tab 5: 수동 주문 ---
    with tab5:
        render_manual_trade_tab(portfolio_list, trader=trader)

    # --- Tab 3: 거래 내역 ---
    with tab3:
        render_trade_history_tab(trader, portfolio_list)

    # --- Tab: 자산관리 ---
    with tab_am:
        render_asset_mgmt_tab(portfolio_list, start_date=start_date, initial_cap=initial_cap)

    # --- Tab 6: 트리거 ---
    with tab6:
        render_strategy_trigger_tab("COIN", coin_portfolio=portfolio_list)

    # --- Tab 4: 백테스트 ---
    with tab4:
        bt_sub1, bt_sub2, bt_sub3 = st.tabs(
            ["📈 개별 백테스트", "🛠️ 파라미터 최적화", "📡 전체 종목 스캔"]
        )
        with bt_sub1:
            render_backtest_tab(portfolio_list, initial_cap, backtest_engine)
        with bt_sub2:
            render_optimization_tab(portfolio_list, initial_cap, backtest_engine)
        with bt_sub3:
            render_scan_tab()

    # --- Tab 7: 운영 로그 ---
    with tab7:
        render_ops_log_tab()

