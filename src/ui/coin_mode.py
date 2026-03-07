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

_COIN_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ui.components.performance import render_performance_table, _render_performance_analysis, _apply_return_hover_format, _apply_dd_hover_format
from src.ui.components.triggers import render_strategy_trigger_tab
from src.ui.components.ops_log import render_ops_log_tab
from src.ui.components.scheduled_orders import render_scheduled_orders_tab
from src.ui.components.backtest_tab import render_backtest_tab
from src.ui.components.backtest_opt_tab import render_optimization_tab, render_scan_tab

def _load_balance_cache():
    """최근 잔고 캐시 파일(balance_cache.json) 로드."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_file = os.path.join(project_root, "balance_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                file_cached = json.load(f)
            if isinstance(file_cached, dict) and file_cached.get("balances"):
                return file_cached
    except Exception:
        pass
    return {}


def _load_account_cache():
    """계좌 캐시 파일(account_cache.json) 로드. VM 경유 조회 결과."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_file = os.path.join(project_root, "account_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _trigger_gh_workflow(job_name: str, extra_inputs: dict | None = None) -> tuple[bool, str]:
    """GitHub Actions workflow를 트리거하고 결과 반환."""
    import subprocess
    try:
        # job_name에 따라 적절한 workflow 파일 선택
        _wf_map = {
            "trade": "coin_trade.yml", "manual_order": "coin_trade.yml",
            "account_sync": "coin_trade.yml",
            "kiwoom_gold": "gold_trade.yml",
            "kis_isa": "isa_trade.yml", "kis_pension": "pension_trade.yml",
            "health_check": "monitoring.yml", "daily_status": "monitoring.yml",
            "vm_once_add": "monitoring.yml", "vm_once_show": "monitoring.yml",
        }
        _wf = _wf_map.get(job_name, "coin_trade.yml")
        cmd = ["gh", "workflow", "run", _wf, "-f", f"run_job={job_name}"]
        for k, v in (extra_inputs or {}).items():
            cmd.extend(["-f", f"{k}={v}"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, f"{job_name} 실행 요청 완료"
        return False, f"실행 실패: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "gh CLI가 설치되어 있지 않습니다. GitHub CLI를 설치해주세요."
    except Exception as e:
        return False, f"오류: {e}"


def _sync_account_cache_from_github():
    """GitHub에서 account_cache.json을 pull (git fetch + checkout)."""
    import subprocess
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        subprocess.run(
            ["git", "fetch", "origin", "--quiet"],
            cwd=project_root, capture_output=True, timeout=15,
        )
        subprocess.run(
            ["git", "checkout", "origin/master", "--",
             "account_cache.json", "balance_cache.json", "signal_state.json", "trade_log.json"],
            cwd=project_root, capture_output=True, timeout=15,
        )
        return True
    except Exception:
        return False


def _trigger_and_wait_gh(job_name: str, status_placeholder=None, extra_inputs: dict | None = None) -> tuple[bool, str]:
    """workflow 트리거 → 완료 대기 → 캐시 pull. 한 번에 처리."""
    import subprocess

    # workflow 파일명 결정
    _wf_map = {
        "trade": "coin_trade.yml", "manual_order": "coin_trade.yml",
        "account_sync": "coin_trade.yml",
        "kiwoom_gold": "gold_trade.yml",
        "kis_isa": "isa_trade.yml", "kis_pension": "pension_trade.yml",
        "health_check": "monitoring.yml", "daily_status": "monitoring.yml",
        "vm_once_add": "monitoring.yml", "vm_once_show": "monitoring.yml",
    }
    wf_file = _wf_map.get(job_name, "coin_trade.yml")

    # 0) 트리거 전 최신 run ID 기록 (새 run 구분용)
    prev_run_id = None
    try:
        r = subprocess.run(
            ["gh", "run", "list", f"--workflow={wf_file}", "--limit", "1",
             "--json", "databaseId"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            _runs = json.loads(r.stdout)
            if _runs:
                prev_run_id = _runs[0].get("databaseId")
    except Exception:
        pass

    # 1) 트리거
    if status_placeholder:
        status_placeholder.text("워크플로우 트리거 중...")
    ok, msg = _trigger_gh_workflow(job_name, extra_inputs=extra_inputs)
    if not ok:
        return False, msg

    # 2) 새 run ID 찾기 (이전 run과 다른 ID, 최대 20초 대기)
    run_id = None
    for _ in range(10):
        time.sleep(2)
        try:
            r = subprocess.run(
                ["gh", "run", "list", f"--workflow={wf_file}", "--limit", "5",
                 "--json", "databaseId,status"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                for run in json.loads(r.stdout):
                    rid = run.get("databaseId")
                    if rid and rid != prev_run_id:
                        run_id = rid
                        break
                if run_id:
                    break
        except Exception:
            pass
    if not run_id:
        _sync_account_cache_from_github()
        return False, "실행 ID를 찾을 수 없습니다."

    # 3) 완료 대기 (최대 5분 = 300초)
    if status_placeholder:
        status_placeholder.text(f"실행 중... (run #{run_id})")
    for i in range(60):
        time.sleep(5)
        if status_placeholder:
            status_placeholder.text(f"실행 중... ({(i+1)*5}초 경과)")
        try:
            r = subprocess.run(
                ["gh", "run", "view", str(run_id), "--json", "status,conclusion"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                info = json.loads(r.stdout)
                if info.get("status") == "completed":
                    success = info.get("conclusion") == "success"
                    # 4) 캐시 pull
                    if status_placeholder:
                        status_placeholder.text("결과 가져오는 중...")
                    _sync_account_cache_from_github()
                    return success, f"{'성공' if success else '실패'} ({(i+1)*5}초)"
        except Exception:
            pass

    # 타임아웃 - 그래도 pull 시도
    _sync_account_cache_from_github()
    return False, "타임아웃 (300초) - 캐시는 업데이트 시도했습니다."


def _load_signal_state():
    """최근 전략 포지션 상태(signal_state.json) 로드."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        state_file = os.path.join(project_root, "signal_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                return state
    except Exception:
        pass
    return {}


def _normalize_coin_interval(interval):
    """github_action_trade.py와 동일한 interval 정규화."""
    iv = str(interval or "day").strip().lower()
    if iv in {"1d", "d", "day", "daily"}:
        return "day"
    if iv in {"4h", "240", "240m", "minute240"}:
        return "minute240"
    if iv in {"1h", "60", "60m", "minute60"}:
        return "minute60"
    return iv


def _make_signal_key(item):
    """github_action_trade.py와 동일한 시그널 키 생성."""
    ticker = f"{item.get('market', 'KRW')}-{str(item.get('coin', '')).upper()}"
    strategy = item.get("strategy", "SMA")
    try:
        param = int(float(item.get("parameter", 20) or 20))
    except Exception:
        param = 20
    interval = _normalize_coin_interval(item.get("interval", "day"))
    return f"{ticker}_{strategy}_{param}_{interval}"


def _determine_signal(position_state, prev_state):
    """
    github_action_trade.py와 동일한 전환 감지 로직.
    """
    pos = str(position_state or "").upper()
    prev = str(prev_state or "").upper() or None
    if pos == "HOLD":
        return "HOLD"
    if prev is None:
        return pos
    if pos == prev:
        return "HOLD"
    return pos


def _resolve_effective_state(position_state, prev_state, signal):
    """
    UI 표시용 최종 상태.
    - 포지션은 '마지막 확정 상태(이전 상태)'를 따른다.
    - 즉, 신호(BUY/SELL/HOLD)는 주문 필요 여부이며 포지션과 분리한다.
    - 이전 상태가 없으면 미확인(UNKNOWN)으로 표기
    """
    prev = str(prev_state or "").upper()
    if prev in {"BUY", "SELL"}:
        return prev
    if prev in {"HOLD"}:
        return "HOLD"
    return "UNKNOWN"


def _state_to_position_label(state):
    s = str(state or "").upper()
    if s == "BUY":
        return "보유"
    if s == "HOLD":
        return "보유"  # 돈치안 HOLD = 매수 후 채널 내부 유지
    if s == "SELL":
        return "현금"
    if s == "UNKNOWN":
        return "미확인"
    return "현금"

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
    
    default_start_str = _coin_cfg.get("start_date", None) or config.get("start_date", None) or _pjson_config.get("start_date", None)
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
    default_cap = _coin_cfg.get("initial_cap", None) or config.get("initial_cap", None) or _pjson_config.get("initial_cap", None)
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

    # ═══════════════════════════════════════════════════════════════
    # 실시간 가격 & 자산 — 대시보드 최상단 (10초 자동 갱신)
    # ═══════════════════════════════════════════════════════════════
    # fragment 내부에서 참조할 변수를 closure로 캡처
    _frag_portfolio = portfolio_list
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
        if _from_cache:
            _note += f" | 잔고: VM캐시 {_cache_time}"
        # 워커 상태도 같은 줄에 표시
        try:
            _w_msg, _w_time = worker.get_status()
            _note += f" | 워커: {_w_msg} ({_w_time})"
        except Exception:
            pass
        st.caption(_note)

    _render_live_ticker()

    # --- Tabs ---
    tab1, tab_orders, tab5, tab3, tab4, tab6, tab7 = st.tabs(["🚀 실시간 포트폴리오", "📋 예약 주문", "🛒 수동 주문", "📜 거래 내역", "📊 백테스트", "⏰ 트리거", "📝 운영 로그"])

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

            # Control Bar
            if st.button("🔄 새로고침"):
                _clear_cache("krw_bal_t1", "prices_t1", "balances_t1")
                st.session_state["coin_force_price_refresh_once"] = True
                st.rerun()

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
                    _force_refresh = bool(st.session_state.pop("coin_force_price_refresh_once", False))
                    _ttl_sec = 0.0 if _force_refresh else 5.0
                    return data_cache.get_current_prices_local_first(
                        unique_tickers,
                        ttl_sec=_ttl_sec,
                        allow_api_fallback=True,
                    )

                all_prices = _ttl_cache("prices_t1", _fetch_all_prices, ttl=5)

                # 잔고 조회: API 우선, 실패 시 캐시 파일 사용

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
                        pass  # 캐시에서 로드 성공
                    else:
                        krw_bal = 0
                        all_balances = {c: 0 for c in unique_coins}

                # --- Total Summary Container ---
                st.subheader("🏁 포트폴리오 요약")

                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                total_init_val = initial_cap

                # Cash Logic
                total_weight_alloc = sum([item.get('weight', 0) for item in portfolio_list])
                cash_ratio = max(0, 100 - total_weight_alloc) / 100.0
                reserved_cash = initial_cap * cash_ratio

                # Add reserved cash to Theo Value (as it stays as cash)
                total_theo_val = reserved_cash

                # --- 전체 자산 현황 테이블 (실제 계좌 전체 반영) ---
                asset_summary_rows = [{"자산": "KRW (현금)", "보유량": f"{krw_bal:,.0f}", "현재가": "-", "평가금액(KRW)": f"{krw_bal:,.0f}", "상태": "-"}]
                seen_coins_summary = set()
                # 포트폴리오 코인
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
                # 포트폴리오에 없지만 계좌에 보유 중인 코인 추가
                if _live_bal and isinstance(_live_bal, dict):
                    for _extra_c, _extra_v in _live_bal.items():
                        _extra_c = str(_extra_c).upper()
                        if _extra_c == 'KRW' or _extra_c in seen_coins_summary:
                            continue
                        _extra_bal = float(_extra_v or 0)
                        if _extra_bal <= 0:
                            continue
                        _extra_ticker = f"KRW-{_extra_c}"
                        _extra_price = all_prices.get(_extra_ticker, 0) or 0
                        # 가격이 없으면 API로 한 번 시도
                        if _extra_price == 0:
                            try:
                                _extra_prices = data_cache.get_current_prices_local_first(
                                    [_extra_ticker], ttl_sec=5.0, allow_api_fallback=True,
                                )
                                _extra_price = (_extra_prices or {}).get(_extra_ticker, 0) or 0
                            except Exception:
                                pass
                        _extra_val = _extra_bal * _extra_price
                        if _extra_val < 100:
                            continue
                        seen_coins_summary.add(_extra_c)
                        asset_summary_rows.append({
                            "자산": f"{_extra_c} *",
                            "보유량": (f"{_extra_bal:.8f}" if _extra_bal < 1 else f"{_extra_bal:,.4f}"),
                            "현재가": f"{_extra_price:,.0f}" if _extra_price > 0 else "-",
                            "평가금액(KRW)": f"{_extra_val:,.0f}",
                            "상태": "미등록 보유",
                        })
                total_real_summary = krw_bal + sum(
                    float(all_balances.get(c, 0) or 0) * (all_prices.get(f"KRW-{c}", 0) or 0)
                    for c in seen_coins_summary
                )
                # 미등록 코인도 합산 (all_balances에 없을 수 있으므로 _live_bal 기준 재계산)
                if _live_bal and isinstance(_live_bal, dict):
                    for _sc in seen_coins_summary:
                        if _sc not in all_balances:
                            _sb = float(_live_bal.get(_sc, 0) or 0)
                            _sp = all_prices.get(f"KRW-{_sc}", 0) or 0
                            total_real_summary += _sb * _sp
                asset_summary_rows.append({
                    "자산": "합계",
                    "보유량": "",
                    "현재가": "",
                    "평가금액(KRW)": f"{total_real_summary:,.0f}",
                    "상태": "",
                })
                with st.expander(f"💰 전체 자산 현황 (Total: {total_real_summary:,.0f} KRW)", expanded=True):
                    st.dataframe(pd.DataFrame(asset_summary_rows), use_container_width=True, hide_index=True)
                    st.caption("* 표시 = 포트폴리오 미등록 코인 (계좌에 보유 중)")

                # --- 단기 모니터링 차트 (60봉) ---
                with st.expander("📊 단기 시그널 모니터링 (60봉)", expanded=True):
                    signal_rows = []
                    signal_state = _load_signal_state()
                    if not signal_state:
                        st.caption("signal_state.json이 없어 이전 상태는 '-' 및 포지션 '미확인'으로 표시됩니다.")

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

                    def _to_float(val, default=0.0):
                        try:
                            num = float(val)
                            return default if pd.isna(num) else num
                        except Exception:
                            return default

                    # 차트 데이터 수집 + 렌더링 함수
                    def render_chart_row(items):
                        if not items:
                            return
                        cols = st.columns(len(items))
                        for ci, item in enumerate(items):
                            p_ticker = f"{item['market']}-{item['coin'].upper()}"
                            p_strategy = item.get('strategy', 'SMA')
                            try:
                                p_param = int(float(item.get('parameter', 20) or 20))
                            except Exception:
                                p_param = 20
                            try:
                                _sp = item.get('sell_parameter', 0)
                                p_sell_param = int(float(_sp)) if _sp not in (None, "", 0, "0") else max(5, p_param // 2)
                            except Exception:
                                p_sell_param = max(5, p_param // 2)
                            p_interval = _normalize_coin_interval(item.get('interval', 'day'))
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

                                latest_close = _to_float(df_60['close'].iloc[-1], default=0.0)
                                close_now = _to_float(all_prices.get(p_ticker, latest_close), default=latest_close)

                                # github_action_trade.py와 동일하게 "완성봉(df.iloc[-2])" 기준 판단
                                strategy_name = str(p_strategy).strip().lower()
                                if strategy_name.startswith("donchian"):
                                    strat = DonchianStrategy()
                                    df_feat = strat.create_features(df_60, buy_period=p_param, sell_period=p_sell_param)
                                    if df_feat is None or len(df_feat) < 2:
                                        continue
                                    eval_candle = df_feat.iloc[-2]
                                    close_eval = _to_float(eval_candle.get("close"), default=latest_close)
                                    upper_key = f"Donchian_Upper_{p_param}"
                                    lower_key = f"Donchian_Lower_{p_sell_param}"
                                    buy_target = _to_float(eval_candle.get(upper_key), default=0.0)
                                    sell_target = _to_float(eval_candle.get(lower_key), default=0.0)
                                    position_state = str(
                                        strat.get_signal(eval_candle, buy_period=p_param, sell_period=p_sell_param)
                                    ).upper()
                                    upper_vals = df_feat[upper_key]
                                    lower_vals = df_feat[lower_key]
                                else:
                                    strat = SMAStrategy()
                                    df_feat = strat.create_features(df_60, periods=[p_param])
                                    if df_feat is None or len(df_feat) < 2:
                                        continue
                                    eval_candle = df_feat.iloc[-2]
                                    close_eval = _to_float(eval_candle.get("close"), default=latest_close)
                                    sma_key = f"SMA_{p_param}"
                                    sma_val = _to_float(eval_candle.get(sma_key), default=0.0)
                                    buy_target = sma_val
                                    sell_target = sma_val
                                    position_state = str(
                                        strat.get_signal(eval_candle, strategy_type='SMA_CROSS', ma_period=p_param)
                                    ).upper()
                                    sma_vals = df_feat[sma_key]

                                buy_dist = ((close_now - buy_target) / buy_target * 100) if buy_target > 0 else 0
                                sell_dist = ((close_now - sell_target) / sell_target * 100) if sell_target > 0 else 0

                                signal_key = _make_signal_key({
                                    "market": item.get("market", "KRW"),
                                    "coin": item.get("coin", ""),
                                    "strategy": p_strategy,
                                    "parameter": p_param,
                                    "interval": p_interval,
                                })
                                prev_state_raw = signal_state.get(signal_key)
                                prev_state = str(prev_state_raw).upper() if prev_state_raw is not None else None
                                if prev_state not in {"BUY", "SELL", "HOLD"}:
                                    prev_state = None
                                exec_signal = _determine_signal(position_state, prev_state)
                                effective_state = _resolve_effective_state(position_state, prev_state, exec_signal)
                                position_label = _state_to_position_label(effective_state)

                                coin_sym = item['coin'].upper()
                                coin_bal = _to_float(all_balances.get(coin_sym, 0), default=0.0)
                                coin_val = coin_bal * close_now
                                is_holding = coin_val >= 5000

                                signal_rows.append({
                                    "종목": p_ticker.replace("KRW-", ""),
                                    "전략": f"{p_strategy} {p_param}",
                                    "시간봉": iv_label,
                                    "포지션": position_label,
                                    "실보유": "보유" if is_holding else "미보유",
                                    "실행": exec_signal,
                                    "판단": (prev_state if prev_state in ("BUY", "SELL") else "BUY") if position_state == "HOLD" else position_state,
                                    "이전": prev_state if prev_state else "-",
                                    "현재가": f"{close_now:,.0f}",
                                    "판단종가": f"{close_eval:,.0f}",
                                    "매수목표": f"{buy_target:,.0f}",
                                    "매도목표": f"{sell_target:,.0f}",
                                    "매수이격도": f"{buy_dist:+.2f}%",
                                    "매도이격도": f"{sell_dist:+.2f}%",
                                })

                                df_chart = df_feat.iloc[-60:]
                                with cols[ci]:
                                    fig_m = go.Figure()
                                    fig_m.add_trace(go.Candlestick(
                                        x=df_chart.index, open=df_chart['open'],
                                        high=df_chart['high'], low=df_chart['low'],
                                        close=df_chart['close'], name='가격',
                                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                                    ))

                                    if strategy_name.startswith("donchian"):
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

                                    if exec_signal == "BUY":
                                        sig_color = "green"
                                    elif exec_signal == "SELL":
                                        sig_color = "red"
                                    elif effective_state == "BUY":
                                        sig_color = "green"
                                    elif effective_state == "SELL":
                                        sig_color = "red"
                                    else:
                                        sig_color = "gray"
                                    # 돈치안: 보유→매도이격도, 미보유→매수이격도 표시
                                    if strategy_name.startswith("donchian"):
                                        if effective_state in ("BUY", "HOLD"):
                                            _chart_dist = f"매도이격:{sell_dist:+.1f}%"
                                        else:
                                            _chart_dist = f"매수이격:{buy_dist:+.1f}%"
                                    else:
                                        _chart_dist = f"{buy_dist:+.1f}%"
                                    fig_m.update_layout(
                                        title=f"{p_ticker.replace('KRW-','')} {p_strategy}{p_param} ({iv_label}) [{position_label}] [실행:{exec_signal}] [{_chart_dist}]",
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

                # 합산 포트폴리오 자리 미리 확보 (데이터 수집 후 렌더링)
                combined_portfolio_container = st.container()

                st.write(f"### 📋 자산 상세 (현금 예비: {reserved_cash:,.0f} KRW)")

                # 포트폴리오 합산용 에쿼티 수집
                portfolio_equity_data = []  # [(label, equity_series, close_series, per_coin_cap, perf)]

                for asset_idx, item in enumerate(portfolio_list):
                    ticker = f"{item['market']}-{item['coin'].upper()}"
                    
                    # Per-Coin Strategy Settings
                    strategy_mode = item.get('strategy', 'SMA Strategy')
                    # Defensive coercion: portfolio rows may contain string values.
                    try:
                        param_val = int(float(item.get('parameter', item.get('sma', 20)) or 20))
                    except Exception:
                        param_val = 20
                    if param_val <= 0:
                        param_val = 20
                    try:
                        sell_param_val = int(float(item.get('sell_parameter', 0) or 0))
                    except Exception:
                        sell_param_val = 0
                    if sell_param_val < 0:
                        sell_param_val = 0
                    try:
                        weight = float(item.get('weight', 0) or 0)
                    except Exception:
                        weight = 0.0
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
                                sell_p = sell_param_val or max(5, buy_p // 2)
                                
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
                            _cp = pd.to_numeric(all_prices.get(ticker, 0), errors="coerce")
                            curr_price = float(_cp) if pd.notna(_cp) else 0.0
                            coin_sym = item['coin'].upper()
                            _cb = pd.to_numeric(all_balances.get(coin_sym, 0), errors="coerce")
                            coin_bal = float(_cb) if pd.notna(_cb) else 0.0
                            coin_value_krw = coin_bal * curr_price

                            # 3. Theo Backtest (Sync Check) - 캐시된 백테스트 사용
                            sell_ratio = (sell_param_val or max(5, param_val // 2)) / param_val if param_val > 0 else 0.5
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
                                    "ticker": ticker,
                                    "label": label,
                                    "equity": hist_df_tmp['equity'],
                                    "close": hist_df_tmp['close'],
                                    "cap": per_coin_cap,
                                    "perf": perf,
                                })
                            else:
                                total_theo_val += per_coin_cap # Fallback if error
                                
                            # 4. Real Status
                            real_status = "HOLD" if coin_value_krw >= 5000 else "CASH"
                            
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

                                    # Redundant manual chart removed - handled by _render_performance_analysis below

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
                real_roi = (total_real_summary - total_init_val) / total_init_val * 100 if total_init_val else 0
                diff_val = total_real_summary - total_theo_val

                sum_col1.metric("초기 자본", f"{total_init_val:,.0f} KRW")
                sum_col2.metric("이론 총자산", f"{total_theo_val:,.0f} KRW", delta=f"{total_roi:.2f}%")
                sum_col3.metric("실제 총자산", f"{total_real_summary:,.0f} KRW", delta=f"{real_roi:.2f}%")
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

                            # 합산 차트 마커는 "실제 체결(거래소)" 기준으로 표시
                            _mk_start = norm_eq.index.min()
                            _mk_end = norm_eq.index.max() + pd.Timedelta(days=1)
                            all_buy_dates = []
                            all_sell_dates = []
                            for ed in portfolio_equity_data:
                                _mk_ticker = str(ed.get("ticker", "") or "")
                                if not _mk_ticker:
                                    continue
                                try:
                                    _done_orders = _ttl_cache(
                                        f"done_combined_{_mk_ticker}",
                                        lambda t=_mk_ticker: trader.get_done_orders(t),
                                        ttl=20,
                                    )
                                except Exception:
                                    _done_orders = []
                                if not isinstance(_done_orders, list):
                                    continue

                                for _od in _done_orders:
                                    _side = str(_od.get("side", "") or "").lower()
                                    _raw_ts = (
                                        _od.get("created_at")
                                        or _od.get("done_at")
                                        or _od.get("created")
                                        or _od.get("timestamp")
                                    )
                                    if not _raw_ts:
                                        continue
                                    _d_ts = pd.to_datetime(_raw_ts, errors="coerce", utc=True)
                                    if pd.isna(_d_ts):
                                        _d_ts = pd.to_datetime(_raw_ts, errors="coerce")
                                    if pd.isna(_d_ts):
                                        continue
                                    try:
                                        if getattr(_d_ts, "tzinfo", None) is not None:
                                            _d_ts = _d_ts.tz_convert("Asia/Seoul").tz_localize(None)
                                    except Exception:
                                        try:
                                            _d_ts = _d_ts.tz_localize(None)
                                        except Exception:
                                            pass

                                    if _d_ts < _mk_start or _d_ts > _mk_end:
                                        continue
                                    if _side == "bid":
                                        all_buy_dates.append(_d_ts)
                                    elif _side == "ask":
                                        all_sell_dates.append(_d_ts)

                            def _add_real_trade_markers(_date_list, _name, _symbol, _color):
                                if not _date_list:
                                    return 0
                                _vals = []
                                _valid_dates = []
                                _seen_days = set()
                                for _d in _date_list:
                                    _d_ts = pd.Timestamp(_d)
                                    if getattr(_d_ts, "tzinfo", None) is not None:
                                        _d_ts = _d_ts.tz_localize(None)
                                    _idx = norm_eq.index.get_indexer([_d_ts], method='nearest')
                                    if _idx[0] < 0:
                                        continue
                                    _x = norm_eq.index[_idx[0]]
                                    _day_key = _x.strftime("%Y-%m-%d") + _name
                                    if _day_key in _seen_days:
                                        continue
                                    _seen_days.add(_day_key)
                                    _valid_dates.append(_x)
                                    _vals.append(norm_eq.iloc[_idx[0]])
                                if _valid_dates:
                                    fig_port.add_trace(go.Scatter(
                                        x=_valid_dates, y=_vals, mode='markers', name=_name,
                                        marker=dict(symbol=_symbol, size=8, color=_color, opacity=0.8)
                                    ))
                                return len(_valid_dates)

                            _buy_mk_cnt = _add_real_trade_markers(all_buy_dates, "매수", "triangle-up", "green")
                            _sell_mk_cnt = _add_real_trade_markers(all_sell_dates, "매도", "triangle-down", "red")
                            if _buy_mk_cnt or _sell_mk_cnt:
                                st.caption(f"마커 기준: 실제 체결(거래소) | 매수 {_buy_mk_cnt}건 / 매도 {_sell_mk_cnt}건")
                            else:
                                st.caption("마커 기준: 실제 체결(거래소) | 해당 기간 체결 없음")

                            fig_port.update_layout(
                                height=350,
                                title="합산 포트폴리오: 전략 vs 단순보유 (정규화, 마커=실제 체결)",
                                yaxis_title="정규화 (%)",
                                yaxis_type="log",
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

    # --- Tab Orders: 예약 주문 ---
    with tab_orders:
        render_scheduled_orders_tab(portfolio_list, initial_cap, config, save_config)

    # --- Tab 5: Manual Trade (거래소 스타일) ---
    with tab5:
        st.header("수동 주문")

        # ── 잔고 표시 ──
        _acct = _load_account_cache()
        if _acct.get("updated_at"):
            bals = _acct.get("balances", {})
            bal_parts = []
            krw = bals.get("KRW", 0)
            if krw > 0:
                bal_parts.append(f"KRW: {krw:,.0f}")
            for k, v in bals.items():
                if k != "KRW" and float(v) > 0:
                    bal_parts.append(f"{k}: {float(v):.8f}")
            if bal_parts:
                st.caption(f"잔고 ({_acct['updated_at']}): " + " | ".join(bal_parts))

        # ── 코인 선택 + 30분봉 차트 ──
        port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
        manual_options = list(dict.fromkeys(port_tickers + TOP_20_TICKERS))
        mt_ticker = st.selectbox("코인 선택", manual_options, key="mt_ticker_chart")
        import pyupbit as _pyupbit
        df_30m = _ttl_cache(
            f"m30_{mt_ticker}",
            lambda: _pyupbit.get_ohlcv(mt_ticker, interval="minute30", count=48),
            ttl=10,
        )
        if df_30m is not None and len(df_30m) > 0:
            last_dt = df_30m.index[-1]
            if hasattr(last_dt, 'strftime'):
                refresh_text = last_dt.strftime('%Y-%m-%d %H:%M')
            else:
                refresh_text = str(last_dt)
            _c_title, _c_time = st.columns([3, 1])
            _c_title.markdown("**30분봉 차트**")
            _c_time.caption(f"최종 봉: {refresh_text}")
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

        # ═══ 호가창 + 주문 패널 (VM 경유 실행) ═══
        mt_coin = mt_ticker.split("-")[1] if "-" in mt_ticker else mt_ticker

        # 호가 데이터 + 현재가 조회 (pyupbit 공개 API)
        ob_data = _ttl_cache(
            f"ob_{mt_ticker}",
            lambda: _pyupbit.get_orderbook(mt_ticker),
            ttl=5,
        )
        cur_price = _ttl_cache(
            f"price_{mt_ticker}",
            lambda: _pyupbit.get_current_price(mt_ticker) or 0,
            ttl=5,
        )

        ob_col, order_col = st.columns([2, 3])

        # ── 좌: 호가창 (HTML 렌더링) ──
        with ob_col:
            st.markdown("**호가창**")
            try:
                if ob_data and len(ob_data) > 0:
                    ob = ob_data[0] if isinstance(ob_data, list) else ob_data
                    units = ob.get('orderbook_units', [])[:10]

                    if units:
                        max_size = max(
                            max(u.get('ask_size', 0) for u in units),
                            max(u.get('bid_size', 0) for u in units),
                        )

                        html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                        html.append('<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666"><td>구분</td><td style="text-align:right">잔량</td><td style="text-align:right">가격</td><td style="text-align:right">등락</td><td>비율</td></tr>')

                        ask_prices = []
                        bid_prices = []

                        for u in reversed(units):
                            ask_p = u.get('ask_price', 0)
                            ask_s = u.get('ask_size', 0)
                            diff = ((ask_p / cur_price) - 1) * 100 if cur_price > 0 else 0
                            bar_w = int(ask_s / max_size * 100) if max_size > 0 else 0
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

                        html.append('<tr style="border-top:3px solid #333;border-bottom:3px solid #333;height:4px"><td colspan="5"></td></tr>')

                        for u in units:
                            bid_p = u.get('bid_price', 0)
                            bid_s = u.get('bid_size', 0)
                            diff = ((bid_p / cur_price) - 1) * 100 if cur_price > 0 else 0
                            bar_w = int(bid_s / max_size * 100) if max_size > 0 else 0
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

        # ── 우: 주문 패널 (VM 경유 실행) ──
        with order_col:
            st.markdown("**주문 실행 (VM 경유)**")
            buy_tab, sell_tab = st.tabs(["매수", "매도"])

            # 주문 성공 후 자산 동기화 헬퍼
            def _order_and_sync(order_json, status_el, label):
                ok, msg = _trigger_and_wait_gh(
                    "manual_order", status_el,
                    extra_inputs={"manual_order_params": order_json},
                )
                if ok:
                    status_el.success(f"{label} 완료 ({msg})")
                    # 자산 자동 동기화
                    status_el.info("잔고 동기화 중...")
                    _trigger_and_wait_gh("account_sync", status_el)
                    _clear_cache("krw_bal_t1", "balances_t1", "prices_t1")
                    status_el.success(f"{label} 완료 · 잔고 갱신됨")
                else:
                    status_el.error(f"{label} 실패: {msg}")

            with buy_tab:
                buy_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_buy_type")

                if buy_type == "시장가":
                    buy_pct = st.slider("매수 비율 (%)", 10, 100, 50, 10, key="mt_buy_pct")
                    st.caption(f"가용 KRW의 {buy_pct}%를 시장가 매수합니다.")

                    if st.button(
                        f"{mt_coin} {buy_pct}% 시장가 매수",
                        key="mt_buy_vm",
                        type="primary",
                        use_container_width=True,
                    ):
                        _buy_json = json.dumps({"coin": mt_coin, "side": "buy", "pct": buy_pct})
                        _order_and_sync(_buy_json, st.empty(), "시장가 매수")

                else:  # 지정가
                    _bc1, _bc2 = st.columns(2)
                    buy_price = _bc1.number_input(
                        "매수 가격 (KRW)", min_value=1, value=int(cur_price * 0.99) if cur_price > 0 else 1,
                        step=1000, key="mt_buy_price",
                    )
                    buy_vol = _bc2.number_input(
                        f"매수 수량 ({mt_coin})", min_value=0.00000001, value=0.001,
                        format="%.8f", key="mt_buy_vol",
                    )
                    buy_total = buy_price * buy_vol
                    st.caption(f"총액: **{buy_total:,.0f} KRW**")

                    if st.button(
                        f"{mt_coin} 지정가 매수 ({buy_price:,.0f} × {buy_vol:.8g})",
                        key="mt_lbuy_vm",
                        type="primary",
                        use_container_width=True,
                    ):
                        if buy_total < 5000:
                            st.error("최소 주문금액: 5,000 KRW")
                        else:
                            _buy_json = json.dumps({
                                "coin": mt_coin, "side": "buy",
                                "order_type": "limit", "price": buy_price, "volume": buy_vol,
                            })
                            _order_and_sync(_buy_json, st.empty(), "지정가 매수")

            with sell_tab:
                sell_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_sell_type")

                if sell_type == "시장가":
                    sell_pct = st.slider("매도 비율 (%)", 10, 100, 100, 10, key="mt_sell_pct")
                    st.caption(f"보유 {mt_coin}의 {sell_pct}%를 시장가 매도합니다.")

                    if st.button(
                        f"{mt_coin} {sell_pct}% 시장가 매도",
                        key="mt_sell_vm",
                        type="primary",
                        use_container_width=True,
                    ):
                        _sell_json = json.dumps({"coin": mt_coin, "side": "sell", "pct": sell_pct})
                        _order_and_sync(_sell_json, st.empty(), "시장가 매도")

                else:  # 지정가
                    _sc1, _sc2 = st.columns(2)
                    sell_price = _sc1.number_input(
                        "매도 가격 (KRW)", min_value=1, value=int(cur_price * 1.01) if cur_price > 0 else 1,
                        step=1000, key="mt_sell_price",
                    )
                    sell_vol = _sc2.number_input(
                        f"매도 수량 ({mt_coin})", min_value=0.00000001, value=0.001,
                        format="%.8f", key="mt_sell_vol",
                    )
                    sell_total = sell_price * sell_vol
                    st.caption(f"총액: **{sell_total:,.0f} KRW**")

                    if st.button(
                        f"{mt_coin} 지정가 매도 ({sell_price:,.0f} × {sell_vol:.8g})",
                        key="mt_lsell_vm",
                        type="primary",
                        use_container_width=True,
                    ):
                        if sell_total < 5000:
                            st.error("최소 주문금액: 5,000 KRW")
                        else:
                            _sell_json = json.dumps({
                                "coin": mt_coin, "side": "sell",
                                "order_type": "limit", "price": sell_price, "volume": sell_vol,
                            })
                            _order_and_sync(_sell_json, st.empty(), "지정가 매도")

        # ── 미체결 주문 ──
        st.divider()
        st.markdown("**미체결 주문**")
        if st.button("미체결 주문 조회 (VM 경유)", key="mt_pending_btn"):
            _pend_status = st.empty()
            _pend_status.info("미체결 주문 조회 중...")
            ok, msg = _trigger_and_wait_gh("account_sync", _pend_status)
            if ok:
                _pend_status.success("잔고 동기화 완료")
            else:
                _pend_status.warning(f"동기화 실패: {msg}")

        # account_cache.json 에서 미체결 정보 표시
        _acct2 = _load_account_cache()
        pending_orders = _acct2.get("pending_orders", [])
        if pending_orders:
            for i, order in enumerate(pending_orders):
                side_kr = "매수" if order.get('side') == 'bid' else "매도"
                side_color = "red" if order.get('side') == 'bid' else "blue"
                market = order.get('market', '')
                price = float(order.get('price', 0) or 0)
                remaining = float(order.get('remaining_volume', 0) or 0)
                created = order.get('created_at', '')
                if created:
                    try:
                        created = pd.to_datetime(created).strftime('%m/%d %H:%M')
                    except Exception:
                        pass
                st.markdown(f"**:{side_color}[{side_kr}]** {market} | {price:,.0f} × {remaining:.8f} | {created}")
        else:
            st.caption("미체결 주문이 없습니다.")

        # 보충 매수/매도 설정 + 실행 스케줄은 "📋 예약 주문" 탭으로 이동

        # 실행 스케줄은 "📋 예약 주문" 탭으로 이동

    # --- Tab 3: History ---
    with tab3:
        st.header("거래 내역")

        hist_tab1, hist_tab2, hist_tab3 = st.tabs(["💸 실제 거래 내역 (거래소)", "📋 VM 매매 로그", "📊 슬리피지 분석"])

        with hist_tab1:
            st.subheader("실제 거래 내역")

            c_h1, c_h2 = st.columns(2)
            h_type = c_h1.selectbox("조회 유형", ["전체", "입금", "출금", "체결 주문"])
            h_curr = c_h2.selectbox("화폐", ["전체", "KRW", "BTC", "ETH", "XRP", "SOL", "USDT", "DOGE", "ADA", "AVAX", "LINK"])

            d_h1, d_h2 = st.columns(2)
            h_date_start = d_h1.date_input("조회 시작일", value=datetime.now().date() - timedelta(days=90), key="hist_start")
            h_date_end = d_h2.date_input("조회 종료일", value=datetime.now().date(), key="hist_end")

            def _parse_deposit_withdraw(raw, type_label):
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
                rows = []
                for r in raw:
                    market = r.get('market', '')
                    coin = market.split('-')[1] if '-' in str(market) else market
                    side = r.get('side', '')
                    side_kr = "매수" if side == 'bid' else ("매도" if side == 'ask' else side)
                    state = r.get('state', '')
                    state_kr = {"done": "체결완료", "cancel": "취소", "wait": "대기"}.get(state, state)
                    price = float(r.get('price', 0) or 0)
                    avg_price = float(r.get('avg_price', 0) or 0)
                    executed_vol = float(r.get('executed_volume', 0) or 0)
                    paid_fee = float(r.get('paid_fee', 0) or 0)
                    # avg_price = 실제 체결 단가 (시장가 주문 시 price는 주문총액이므로 avg_price 우선)
                    unit_price = avg_price if avg_price > 0 else price
                    if unit_price > 0 and executed_vol > 0:
                        total_krw = unit_price * executed_vol
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

            def _display_history_df(all_rows):
                if all_rows:
                    result_df = pd.DataFrame(all_rows)
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
                else:
                    st.warning(f"조회 결과 없음. (유형: {h_type}, 화폐: {h_curr})")

            def _get_rows_from_cache(acct, h_type, h_curr):
                """캐시 데이터에서 행 추출"""
                api_curr = None if h_curr == "전체" else h_curr
                rows = []
                if h_type in ("전체", "입금"):
                    rows.extend(_parse_deposit_withdraw(acct.get("deposits", []), "입금"))
                if h_type in ("전체", "출금"):
                    rows.extend(_parse_deposit_withdraw(acct.get("withdraws", []), "출금"))
                if h_type in ("전체", "체결 주문"):
                    rows.extend(_parse_orders(acct.get("orders", [])))
                if api_curr and rows:
                    rows = [r for r in rows if api_curr.upper() in r.get("화폐/코인", "").upper()]
                return rows

            # ── 조회 ──
            _hist_btn_col1, _hist_btn_col2 = st.columns([1, 3])
            with _hist_btn_col1:
                _do_query = st.button("조회", key="hist_query", type="primary")
            with _hist_btn_col2:
                _acct_cache = _load_account_cache()
                _cache_time = _acct_cache.get("updated_at", "")
                if _cache_time:
                    st.caption(f"마지막 동기화: {_cache_time}")

            # 조회 버튼 클릭 시 API 직접 호출
            if _do_query and trader:
                api_curr = None if h_curr == "전체" else h_curr
                query_types = []
                if h_type == "전체":
                    query_types = [("deposit", "입금"), ("withdraw", "출금"), ("order", "체결")]
                elif "입금" in h_type:
                    query_types = [("deposit", "입금")]
                elif "출금" in h_type:
                    query_types = [("withdraw", "출금")]
                elif "체결" in h_type:
                    query_types = [("order", "체결")]

                all_rows = []
                ip_blocked = False
                with st.spinner("API 조회 중..."):
                    for api_type, label in query_types:
                        try:
                            data, err = trader.get_history(api_type, api_curr)
                            if err and ("authorization_ip" in err or "verified IP" in err or "401" in str(err)):
                                ip_blocked = True
                                break
                            if data:
                                if api_type in ("deposit", "withdraw"):
                                    all_rows.extend(_parse_deposit_withdraw(data, label))
                                else:
                                    all_rows.extend(_parse_orders(data))
                        except Exception as _he:
                            st.warning(f"API 오류: {_he}")
                            ip_blocked = True
                            break

                if all_rows and not ip_blocked:
                    st.session_state["_hist_rows"] = all_rows
                    st.session_state["_hist_source"] = "API 실시간"
                elif ip_blocked:
                    if _acct_cache.get("orders") or _acct_cache.get("deposits"):
                        st.session_state["_hist_rows"] = _get_rows_from_cache(_acct_cache, h_type, h_curr)
                        st.session_state["_hist_source"] = f"캐시 ({_cache_time})"
                else:
                    st.info("조회 결과가 없습니다.")

            # 세션에 저장된 결과 즉시 표시
            if st.session_state.get("_hist_rows"):
                _src = st.session_state.get("_hist_source", "")
                if _src:
                    st.caption(f"데이터 출처: {_src}")
                _display_history_df(st.session_state["_hist_rows"])
            elif _acct_cache.get("orders") or _acct_cache.get("deposits"):
                # 버튼 안 눌러도 캐시가 있으면 바로 표시
                st.caption(f"캐시 데이터 ({_cache_time})")
                _display_history_df(_get_rows_from_cache(_acct_cache, h_type, h_curr))

            st.caption("Upbit API 제한: 최근 100건까지 조회 가능")

        with hist_tab2:
            st.subheader("VM 매매 로그")

            # trade_log.json sync from GitHub
            _tl_path = os.path.join(_COIN_PROJECT_ROOT, "trade_log.json")
            if st.button("동기화", key="tl_sync"):
                try:
                    subprocess.run(
                        ["git", "fetch", "origin", "--quiet"],
                        cwd=_COIN_PROJECT_ROOT, capture_output=True, timeout=10,
                    )
                    subprocess.run(
                        ["git", "checkout", "origin/master", "--", "trade_log.json"],
                        cwd=_COIN_PROJECT_ROOT, capture_output=True, timeout=5,
                    )
                    st.toast("동기화 완료")
                except Exception:
                    pass

            # 로드
            _tl_entries = []
            if os.path.exists(_tl_path):
                try:
                    with open(_tl_path, "r", encoding="utf-8") as _f:
                        _tl_entries = json.load(_f)
                    if not isinstance(_tl_entries, list):
                        _tl_entries = []
                except Exception:
                    _tl_entries = []

            if not _tl_entries:
                st.info("VM 매매 로그가 없습니다. (trade_log.json 미동기화)")
            else:
                # 필터
                _tl_modes = ["전체", "auto", "manual", "signal"]
                _tl_mode = st.selectbox("모드 필터", _tl_modes, key="tl_mode_filter")

                _tl_rows = []
                for e in _tl_entries:
                    mode = e.get("mode", "")
                    if _tl_mode != "전체" and mode != _tl_mode:
                        continue
                    side = e.get("side", "")
                    side_kr = {"BUY": "매수", "SELL": "매도", "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도"}.get(side, side)
                    mode_kr = {"auto": "자동", "manual": "수동", "signal": "시그널"}.get(mode, mode)
                    amount_str = e.get("amount", e.get("qty", ""))
                    _tl_rows.append({
                        "시간": e.get("time", ""),
                        "모드": mode_kr,
                        "코인": e.get("ticker", ""),
                        "구분": side_kr,
                        "전략": e.get("strategy", ""),
                        "금액/수량": str(amount_str),
                        "결과": e.get("result", ""),
                        "상세": str(e.get("detail", ""))[:80],
                    })

                if _tl_rows:
                    _tl_df = pd.DataFrame(_tl_rows)
                    st.success(f"{len(_tl_df)}건")

                    def _color_trade_side(val):
                        if val in ("매수", "보충매수"):
                            return "color: #e74c3c"
                        elif val in ("매도", "보충매도"):
                            return "color: #2980b9"
                        elif val == "시그널":
                            return "color: #f39c12"
                        return ""

                    def _color_result(val):
                        if val == "success":
                            return "color: #27ae60"
                        elif val == "error":
                            return "color: #e74c3c"
                        return ""

                    st.dataframe(
                        _tl_df.style
                            .map(_color_trade_side, subset=["구분"])
                            .map(_color_result, subset=["결과"]),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("필터 조건에 해당하는 로그 없음")

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

