import streamlit as st
import pyupbit
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import requests
from dotenv import load_dotenv
import json
import data_cache

# Import modules
from backtest.engine import BacktestEngine
from trading.upbit_trader import UpbitTrader
from strategy.sma import SMAStrategy
from strategy.donchian import DonchianStrategy

# Load environment variables
load_dotenv(override=True)

# --- Configuration Persistence ---
CONFIG_FILE = "user_config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
        
config = load_config()

# 국내 ETF 코드 -> 한글 종목명 매핑 (ISA/연금저축 표기용)
ETF_NAME_KR = {
    "418660": "TIGER 미국나스닥100레버리지(합성)",
    "409820": "KODEX 미국나스닥100레버리지(합성 H)",
    "423920": "TIGER 미국필라델피아반도체레버리지(합성)",
    "465610": "ACE 미국빅테크TOP7 Plus레버리지(합성)",
    "461910": "PLUS 미국테크TOP10레버리지(합성)",
    "133690": "TIGER 미국나스닥100",
    "360750": "TIGER 미국S&P500",
    "132030": "KODEX Gold선물(H)",
    "453540": "TIGER 미국채10년선물",
    "114470": "KODEX 국고채3년",
    "453850": "TIGER 선진국MSCI World",
    "251350": "KODEX 선진국MSCI World",
    "308620": "KODEX 미국채10년선물",
    "471460": "ACE 미국30년국채액티브",
}


def _etf_name_kr(code: str) -> str:
    return ETF_NAME_KR.get(str(code).strip(), "종목명 미확인")


def _fmt_etf_code_name(code: str) -> str:
    c = str(code).strip()
    if not c:
        return "-"
    return f"{c} {_etf_name_kr(c)}"


def _code_only(v: str) -> str:
    return str(v or "").strip().split()[0] if str(v or "").strip() else ""


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


def _get_runtime_value(keys, default=""):
    """Read value from env first, then Streamlit secrets."""
    if isinstance(keys, str):
        keys = (keys,)

    for key in keys:
        v = os.getenv(key, "")
        if str(v).strip():
            return v

    try:
        for key in keys:
            v = st.secrets.get(key, "")
            if str(v).strip():
                return v
    except Exception:
        pass

    return default


def render_telegram_sidebar(prefix: str = "coin"):
    """Sidebar Telegram settings (load from .env/secrets, save to user config)."""
    tg_token_default = str(config.get("telegram_bot_token", "") or _get_runtime_value("TELEGRAM_BOT_TOKEN", ""))
    tg_chat_default = str(config.get("telegram_chat_id", "") or _get_runtime_value("TELEGRAM_CHAT_ID", ""))

    with st.sidebar.expander("📨 텔레그램 알림", expanded=False):
        tg_token = st.text_input(
            "봇 토큰",
            value=tg_token_default,
            type="password",
            key=f"{prefix}_telegram_bot_token",
            disabled=IS_CLOUD,
        )
        tg_chat = st.text_input(
            "채팅 ID",
            value=tg_chat_default,
            key=f"{prefix}_telegram_chat_id",
            disabled=IS_CLOUD,
        )

        if IS_CLOUD:
            st.caption("Cloud 환경에서는 편집/저장이 비활성화됩니다.")
            return

        c1, c2 = st.columns(2)
        with c1:
            if st.button("저장", key=f"{prefix}_telegram_save"):
                new_cfg = config.copy()
                new_cfg["telegram_bot_token"] = str(tg_token).strip()
                new_cfg["telegram_chat_id"] = str(tg_chat).strip()
                save_config(new_cfg)
                st.success("텔레그램 설정을 저장했습니다.")
                st.rerun()
        with c2:
            if st.button("테스트 전송", key=f"{prefix}_telegram_test"):
                if not tg_token or not tg_chat:
                    st.warning("봇 토큰과 채팅 ID를 입력해 주세요.")
                else:
                    try:
                        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
                        text = f"✅ 테스트 메시지\n시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        resp = requests.post(
                            url,
                            json={"chat_id": tg_chat, "text": text},
                            timeout=10,
                        )
                        ok = resp.ok and resp.json().get("ok", False)
                        if ok:
                            st.success("테스트 메시지 전송 성공")
                        else:
                            st.error(f"전송 실패: HTTP {resp.status_code}")
                    except Exception as e:
                        st.error(f"전송 오류: {e}")

# Cloud 환경 감지 (Streamlit Cloud에서는 HOSTNAME이 *.streamlit.app 또는 /mount/src 경로)
IS_CLOUD = os.path.exists("/mount/src") or "streamlit.app" in os.getenv("HOSTNAME", "")

st.set_page_config(page_title="업비트 자동매매", layout="wide")

# --- Custom CSS for Better Readability ---
st.markdown("""
    <style>
    /* Global Font Adjustments */
    html, body, [class*="css"] {
        font-size: 18px;
    }
    .stMarkdown p {
        font-size: 18px !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
        color: #666;
    }
    [data-testid="stMetricDelta"] {
        font-size: 16px !important;
    }

    /* Expander Headers */
    .streamlit-expanderHeader {
        font-size: 22px !important;
        font-weight: 600 !important;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    /* Sidebar Input Labels */
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Width Override (PC) */
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 520px !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    /* === 글자 겹침 방지 === */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        overflow: hidden !important;
    }
    [data-testid="column"] {
        overflow: hidden !important;
    }
    /* 탭 버튼 겹침 방지 */
    [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    /* 셀렉트박스/인풋 라벨 겹침 방지 */
    .stSelectbox label, .stNumberInput label, .stDateInput label, .stTextInput label {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        max-width: 100% !important;
    }
    /* 캡션/텍스트 겹침 방지 */
    .stCaption, .stMarkdown {
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }

    /* ===== Mobile Responsive ===== */
    @media (max-width: 768px) {
        html, body, [class*="css"] {
            font-size: 14px;
        }
        .stMarkdown p {
            font-size: 14px !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 12px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        .streamlit-expanderHeader {
            font-size: 16px !important;
        }
        button[data-baseweb="tab"] {
            font-size: 11px !important;
            padding: 4px 8px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-baseweb="tab-list"] {
            gap: 2px !important;
        }
        /* 모바일 메트릭 겹침 방지 */
        [data-testid="stMetricValue"] {
            font-size: 18px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 11px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 320px !important;
        }
        /* 모바일에서 컬럼 세로 스택 */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        /* 차트 높이 조정 */
        .js-plotly-plot {
            max-height: 250px !important;
        }
        /* 데이터프레임 가로 스크롤 */
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
        }
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    }

    </style>
""", unsafe_allow_html=True)

def render_gold_mode():
    """금(Gold) 현물 거래 모드 - 키움증권 KRX 금시장 (코인 탭과 동일한 구조)"""
    from kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG
    from backtest.engine import BacktestEngine

    st.title("🥇 Gold Trading System (키움증권 KRX)")

    # ── 사이드바: 설정 ─────────────────────────────────────
    st.sidebar.header("Gold 설정")

    # API Keys
    try:
        kiwoom_ak      = st.secrets.get("Kiwoom_App_Key", "")
        kiwoom_sk      = st.secrets.get("Kiwoom_Secret_Key", "")
        kiwoom_account = st.secrets.get("KIWOOM_ACCOUNT", "")
    except Exception:
        kiwoom_ak      = os.getenv("Kiwoom_App_Key", "")
        kiwoom_sk      = os.getenv("Kiwoom_Secret_Key", "")
        kiwoom_account = os.getenv("KIWOOM_ACCOUNT", "")

    if IS_CLOUD:
        st.sidebar.info("📱 조회 전용 모드 (Cloud)")
    else:
        with st.sidebar.expander("키움 API Keys", expanded=False):
            kiwoom_ak      = st.text_input("App Key",    value=kiwoom_ak,      type="password", key="kiwoom_ak")
            kiwoom_sk      = st.text_input("Secret Key", value=kiwoom_sk,      type="password", key="kiwoom_sk")
            kiwoom_account = st.text_input("계좌번호",    value=kiwoom_account, key="kiwoom_acc")

    # 전략 설정 (코인 포트폴리오와 동일하게 다중 전략 지원)
    st.sidebar.subheader("전략 설정")
    st.sidebar.caption("여러 전략을 추가하여 포트폴리오를 구성할 수 있습니다.")

    _gold_cfg_default = [{"strategy": "Donchian", "buy": 90, "sell": 55, "weight": 100}]
    _gold_cfg = config.get("gold_strategy", _gold_cfg_default)

    df_gold_strat = pd.DataFrame(_gold_cfg)
    if IS_CLOUD:
        st.sidebar.dataframe(df_gold_strat, use_container_width=True, hide_index=True)
        edited_gold_strat = df_gold_strat
    else:
        edited_gold_strat = st.sidebar.data_editor(
            df_gold_strat, num_rows="dynamic", use_container_width=True, hide_index=True,
            key="gold_strat_editor",
            column_config={
                "strategy": st.column_config.SelectboxColumn("전략", options=["Donchian", "SMA"], required=True),
                "buy":      st.column_config.NumberColumn("매수", min_value=5, max_value=300, step=1, required=True),
                "sell":     st.column_config.NumberColumn("매도", min_value=0, max_value=300, step=1, required=True, help="Donchian 매도 채널 (SMA는 무시됨, 0=매수의 절반)"),
                "weight":   st.column_config.NumberColumn("비중 %", min_value=1, max_value=100, step=1, required=True),
            },
        )

    # 비중 검증
    gold_total_weight = int(edited_gold_strat["weight"].sum())
    if gold_total_weight > 100:
        st.sidebar.error(f"총 비중이 {gold_total_weight}% 입니다. (100% 이하로 설정해주세요)")
    else:
        gold_cash_weight = 100 - gold_total_weight
        st.sidebar.info(f"투자 비중: {gold_total_weight}% | 현금: {gold_cash_weight}%")

    # 골드 포트폴리오 리스트 생성
    gold_portfolio_list = []
    for _, row in edited_gold_strat.iterrows():
        bp = int(row.get("buy", 90))
        sp = int(row.get("sell", 0) or 0)
        if sp == 0:
            sp = max(5, bp // 2)
        gold_portfolio_list.append({
            "strategy": str(row.get("strategy", "Donchian")),
            "buy_period": bp,
            "sell_period": sp,
            "weight": int(row.get("weight", 100)),
        })

    # 첫 번째 전략 (기본값)
    if gold_portfolio_list:
        _g_first = gold_portfolio_list[0]
        buy_period = _g_first["buy_period"]
        sell_period = _g_first["sell_period"]
    else:
        buy_period = 90
        sell_period = 55

    # 공통 설정
    st.sidebar.subheader("공통 설정")
    _gold_start_default = config.get("gold_start_date", "2022-06-01")
    gold_start_date = st.sidebar.date_input(
        "기준 시작일", value=pd.to_datetime(_gold_start_default).date(),
        help="백테스트 평가 시작일", disabled=IS_CLOUD, key="gold_start_date"
    )
    _gold_cap_default = config.get("gold_initial_cap", 10_000_000)
    gold_initial_cap = st.sidebar.number_input(
        "초기 자본금 (KRW)", value=_gold_cap_default, step=100_000, format="%d",
        disabled=IS_CLOUD, key="gold_initial_cap"
    )
    st.sidebar.caption(f"설정: **{gold_initial_cap:,.0f} KRW**")

    if not IS_CLOUD:
        if st.sidebar.button("💾 Gold 설정 저장", key="gold_save_btn"):
            new_gold_cfg = config.copy()
            new_gold_cfg["gold_strategy"]    = edited_gold_strat.to_dict("records")
            new_gold_cfg["gold_start_date"]  = str(gold_start_date)
            new_gold_cfg["gold_initial_cap"] = gold_initial_cap
            save_config(new_gold_cfg)
            st.sidebar.success("저장 완료!")

    # ── 트레이더 + 백그라운드 워커 초기화 ────────────────────
    from data_manager import GoldDataWorker

    @st.cache_resource
    def _get_gold_trader(ak, sk, acct):
        t = KiwoomGoldTrader(is_mock=False)
        t.app_key = ak
        t.app_secret = sk
        t.account_no = acct
        return t

    @st.cache_resource
    def _get_gold_worker(_trader):
        """백그라운드 워커: 잔고/시세/호가를 3초마다 병렬 갱신"""
        w = GoldDataWorker()
        w.configure(_trader)
        w.start()
        return w

    gold_trader = None
    gold_worker = None
    if kiwoom_ak and kiwoom_sk:
        gold_trader = _get_gold_trader(kiwoom_ak, kiwoom_sk, kiwoom_account)
        gold_worker = _get_gold_worker(gold_trader)

    # ── 데이터 로드 헬퍼 (parquet 캐시 → 워커 → CSV 폴백) ──
    def load_gold_data(buy_p: int) -> pd.DataFrame | None:
        """일봉 데이터: parquet 캐시 → 워커 차트 → CSV 폴백."""
        import data_cache
        # 1순위: parquet 캐시 (사전 다운로드된 대량 데이터)
        cached = data_cache.load_cached_gold()
        if cached is not None and len(cached) >= buy_p + 5:
            return cached
        # 2순위: 백그라운드 워커 차트 데이터
        if gold_worker:
            df_w = gold_worker.get_chart()
            if df_w is not None and len(df_w) >= buy_p + 5:
                return df_w
        # 3순위: CSV 파일 (오프라인 폴백)
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "krx_gold_daily.csv")
        if os.path.exists(csv_path):
            df_csv = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df_csv.columns = [c.lower() for c in df_csv.columns]
            if "open" not in df_csv.columns: df_csv["open"] = df_csv["close"]
            if "high" not in df_csv.columns: df_csv["high"] = df_csv["close"]
            if "low"  not in df_csv.columns: df_csv["low"]  = df_csv["close"]
            return df_csv
        return None

    # ── 탭 구성 ───────────────────────────────────────────
    tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs(
        ["🚀 실시간 모니터링", "🛒 수동 주문", "📊 백테스트", "💳 수수료/세금"]
    )

    # ══════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링 (코인 탭1과 동일 구조)
    # ══════════════════════════════════════════════════════
    with tab_g1:
        st.header("실시간 금 모니터링")
        _strat_labels = [f"{g['strategy']}({g['buy_period']}/{g['sell_period']}) {g['weight']}%" for g in gold_portfolio_list]
        st.caption(f"전략: {', '.join(_strat_labels)} | 초기자본: {gold_initial_cap:,.0f}원")

        # 새로고침
        col_r1, col_r2 = st.columns([1, 5])
        with col_r1:
            if st.button("🔄 새로고침", key="gold_refresh"):
                for k in list(st.session_state.keys()):
                    if k.startswith("__gc_") or k.startswith("__gt_"):
                        del st.session_state[k]
                st.cache_data.clear()
                st.rerun()

        # 계좌 잔고 (워커에서 읽기 — 블로킹 없음)
        with st.expander("💰 계좌 현황", expanded=True):
            if not gold_worker:
                st.warning("사이드바에서 키움 API Key를 입력해주세요.")
            else:
                bal = gold_worker.get('balance')
                if bal:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("예수금", f"{bal['cash_krw']:,.0f}원")
                    b2.metric("금 보유량", f"{bal['gold_qty']:.4f}g")
                    b3.metric("금 평가금액", f"{bal['gold_eval']:,.0f}원")
                    total_asset = bal['cash_krw'] + bal['gold_eval']
                    pnl = total_asset - gold_initial_cap
                    b4.metric("총 평가", f"{total_asset:,.0f}원", delta=f"{pnl:+,.0f}원")
                elif not gold_worker.is_ready():
                    st.info("데이터 로딩 중... (백그라운드 갱신)")
                    bal = None
                else:
                    st.warning("잔고 조회 실패 (API 인증 확인)")
                    bal = None

        # 시그널 차트 (다중 전략 지원)
        with st.expander("📊 시그널 모니터링", expanded=True):
            # 가장 큰 buy_period로 데이터 로드
            max_buy_p = max((g['buy_period'] for g in gold_portfolio_list), default=90)
            df_gold = load_gold_data(max_buy_p)

            if df_gold is None or len(df_gold) < max_buy_p + 5:
                st.warning("일봉 데이터 부족. API 연결 또는 krx_gold_daily.csv를 확인하세요.")
            else:
                close_now = float(df_gold['close'].iloc[-1])
                gold_signal_rows = []

                # 전략별 차트 렌더링
                n_strats = len(gold_portfolio_list)
                if n_strats > 0:
                    chart_cols = st.columns(n_strats)

                for gi, gp in enumerate(gold_portfolio_list):
                    g_strat = gp['strategy']
                    g_bp = gp['buy_period']
                    g_sp = gp['sell_period']
                    g_wt = gp['weight']

                    if g_strat == "Donchian":
                        g_upper = df_gold['high'].rolling(window=g_bp).max().shift(1)
                        g_lower = df_gold['low'].rolling(window=g_sp).min().shift(1)
                        g_buy_target = float(g_upper.iloc[-1])
                        g_sell_target = float(g_lower.iloc[-1])
                        g_buy_dist = (close_now - g_buy_target) / g_buy_target * 100 if g_buy_target else 0
                        g_sell_dist = (close_now - g_sell_target) / g_sell_target * 100 if g_sell_target else 0
                        in_pos = False
                        for i in range(len(df_gold)):
                            u = g_upper.iloc[i]; l = g_lower.iloc[i]; c = float(df_gold['close'].iloc[i])
                            if not pd.isna(u) and c > u: in_pos = True
                            elif not pd.isna(l) and c < l: in_pos = False
                        g_signal = ("SELL" if close_now < g_sell_target else "HOLD") if in_pos else \
                                   ("BUY" if close_now > g_buy_target else "WAIT")
                        g_pos_label = "보유" if in_pos else "현금"
                    else:
                        g_sma = df_gold['close'].rolling(window=g_bp).mean()
                        g_buy_target = float(g_sma.iloc[-1])
                        g_sell_target = g_buy_target
                        g_buy_dist = (close_now - g_buy_target) / g_buy_target * 100 if g_buy_target else 0
                        g_sell_dist = g_buy_dist
                        g_signal = "BUY" if close_now > g_buy_target else "SELL"
                        g_pos_label = "보유" if close_now > g_buy_target else "현금"

                    gold_signal_rows.append({
                        "전략": f"{g_strat} {g_bp}/{g_sp}",
                        "비중": f"{g_wt}%",
                        "현재가": f"{close_now:,.0f}",
                        "매수목표": f"{g_buy_target:,.0f}",
                        "매도목표": f"{g_sell_target:,.0f}",
                        "매수이격도": f"{g_buy_dist:+.2f}%",
                        "매도이격도": f"{g_sell_dist:+.2f}%",
                        "포지션": g_pos_label,
                        "시그널": g_signal,
                    })

                    # 차트 렌더링
                    g_sig_color = "green" if g_signal == "BUY" else ("red" if g_signal == "SELL" else ("blue" if g_signal == "WAIT" else "gray"))
                    df_chart = df_gold.iloc[-120:]
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Candlestick(
                        x=df_chart.index, open=df_chart['open'],
                        high=df_chart['high'], low=df_chart['low'],
                        close=df_chart['close'], name='금 일봉',
                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                    ))
                    if g_strat == "Donchian":
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_upper.loc[df_chart.index],
                            name=f'상단({g_bp})', line=dict(color='green', width=1.5, dash='dot')))
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_lower.loc[df_chart.index],
                            name=f'하단({g_sp})', line=dict(color='red', width=1.5, dash='dot')))
                    else:
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_sma.loc[df_chart.index],
                            name=f'SMA({g_bp})', line=dict(color='orange', width=2)))
                    fig_g.update_layout(
                        title=f"KRX 금현물 {g_strat}({g_bp}/{g_sp}) [{g_pos_label}] [{g_buy_dist:+.1f}%]",
                        title_font_color=g_sig_color,
                        height=400, margin=dict(l=0, r=0, t=40, b=30),
                        xaxis_rangeslider_visible=False, showlegend=True,
                        xaxis=dict(showticklabels=True, tickformat='%Y/%m/%d', tickangle=-45, nticks=10),
                        yaxis_title="가격 (원/g)",
                    )
                    with chart_cols[gi]:
                        st.plotly_chart(fig_g, use_container_width=True)

                # 시그널 요약 테이블
                if gold_signal_rows:
                    st.dataframe(pd.DataFrame(gold_signal_rows), use_container_width=True, hide_index=True)

        # 자동매매 규칙
        with st.expander("⚖️ 자동매매 규칙", expanded=False):
            rules_lines = ["**실행 시점**: GitHub Actions - 매 평일 KST 09:05\n"]
            for gp in gold_portfolio_list:
                if gp['strategy'] == "Donchian":
                    rules_lines.append(f"**{gp['strategy']}({gp['buy_period']}/{gp['sell_period']})** 비중 {gp['weight']}%")
                    rules_lines.append(f"- 매수: 종가 > {gp['buy_period']}일 최고가 → 시장가 매수")
                    rules_lines.append(f"- 매도: 종가 < {gp['sell_period']}일 최저가 → 시장가 매도\n")
                else:
                    rules_lines.append(f"**{gp['strategy']}({gp['buy_period']})** 비중 {gp['weight']}%")
                    rules_lines.append(f"- 매수: 종가 > SMA({gp['buy_period']}) → 시장가 매수")
                    rules_lines.append(f"- 매도: 종가 < SMA({gp['buy_period']}) → 시장가 매도\n")
            rules_lines.append("**수수료**: 키움증권 0.165% (왕복 ~0.34%)")
            st.markdown("\n".join(rules_lines))

    # ══════════════════════════════════════════════════════
    # Tab 2: 수동 주문 — HTS 스타일 (코인 트레이딩 패널과 동일 구조)
    # ══════════════════════════════════════════════════════
    with tab_g2:
        st.header("수동 주문")

        if not gold_trader:
            st.warning("API Key를 사이드바에서 입력해주세요.")
        else:
            GOLD_TICK = 10  # KRX 금현물 호가단위: 10원
            GOLD_MIN_QTY = 1.0  # 1KG 종목 최소 수량: 1g

            def _gold_align_price(price, tick=GOLD_TICK):
                return round(price / tick) * tick

            # ── 호가 선택 콜백 ──
            def _on_gold_ob_select():
                sel = st.session_state.get('_gold_ob_sel', '')
                try:
                    price_str = sel.split(' ', 1)[1].replace(',', '')
                    chosen = int(float(price_str))
                    st.session_state['gold_buy_price'] = chosen
                    st.session_state['gold_sell_price'] = chosen
                except (IndexError, ValueError):
                    pass

            # ═══ 트레이딩 패널 (3초 자동갱신, 워커에서 읽기만 → 블로킹 없음) ═══
            @st.fragment
            def gold_trading_panel():
                # ── 워커에서 즉시 읽기 (API 호출 없음) ──
                g_bal = gold_worker.get('balance') if gold_worker else None
                g_price = gold_worker.get('price', 0) if gold_worker else 0

                g_cash = g_bal['cash_krw'] if g_bal else 0.0
                g_qty  = g_bal['gold_qty'] if g_bal else 0.0
                g_eval = g_bal['gold_eval'] if g_bal else 0.0
                g_hold_val = g_qty * g_price if g_price > 0 else g_eval

                # ── 상단 정보 바 ──
                gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                gc1.metric("현재가 (원/g)", f"{g_price:,.0f}")
                gc2.metric("금 보유", f"{g_qty:.2f}g")
                gc3.metric("평가금액", f"{g_hold_val:,.0f}원")
                gc4.metric("예수금", f"{g_cash:,.0f}원")
                gc5.metric("호가단위", f"{GOLD_TICK}원")

                # ── 최근 거래 알림 바 ──
                g_last_trade = st.session_state.get('_gold_last_trade')
                if g_last_trade:
                    gt_type = g_last_trade.get('type', '')
                    gt_time = g_last_trade.get('time', '')
                    gt_detail = g_last_trade.get('detail', '')
                    is_buy = '매수' in gt_type
                    g_color = '#D32F2F' if is_buy else '#1976D2'
                    gnc1, gnc2 = st.columns([6, 1])
                    gnc1.markdown(
                        f'<div style="padding:6px 12px;border-radius:6px;background:{g_color}22;'
                        f'border-left:4px solid {g_color};font-size:14px;">'
                        f'<b style="color:{g_color}">{gt_type}</b> | {gt_detail} | {gt_time}</div>',
                        unsafe_allow_html=True
                    )
                    if gnc2.button("✕", key="_gold_dismiss"):
                        del st.session_state['_gold_last_trade']
                        st.rerun()

                st.divider()

                # ═══ 메인 레이아웃: 호가창(좌) + 주문(우) ═══
                ob_col, order_col = st.columns([2, 3])

                # ── 좌: 호가창 (HTML 테이블) ──
                with ob_col:
                    price_labels = []

                    ob = gold_worker.get('orderbook') if gold_worker else None

                    if ob and ob.get('asks') and ob.get('bids'):
                        asks = ob['asks']  # 매도호가 (낮→높)
                        bids = ob['bids']  # 매수호가 (높→낮)

                        all_qtys = [a['qty'] for a in asks] + [b['qty'] for b in bids]
                        max_qty = max(all_qtys) if all_qtys else 1

                        html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                        html.append(
                            '<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666">'
                            '<td>구분</td><td style="text-align:right">잔량(g)</td>'
                            '<td style="text-align:right">가격(원)</td>'
                            '<td style="text-align:right">등락</td><td>비율</td></tr>'
                        )

                        ask_prices = []
                        bid_prices = []

                        # 매도호가 (높→낮 순서로 표시)
                        for a in reversed(asks):
                            ap = a['price']
                            aq = a['qty']
                            diff = ((ap / g_price) - 1) * 100 if g_price > 0 else 0
                            bar_w = int(aq / max_qty * 100) if max_qty > 0 else 0
                            ask_prices.append(ap)
                            html.append(
                                f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                                f'<td>매도</td>'
                                f'<td style="text-align:right">{aq:.2f}</td>'
                                f'<td style="text-align:right;font-weight:bold">{ap:,.0f}</td>'
                                f'<td style="text-align:right">{diff:+.2f}%</td>'
                                f'<td><div style="background:#1976D233;height:14px;width:{bar_w}%"></div></td></tr>'
                            )

                        # 현재가 구분선
                        html.append(
                            f'<tr style="background:#FFF3E0;border:2px solid #FF9800;height:32px;font-weight:bold">'
                            f'<td colspan="2" style="color:#E65100">현재가</td>'
                            f'<td style="text-align:right;color:#E65100;font-size:15px">{g_price:,.0f}</td>'
                            f'<td colspan="2"></td></tr>'
                        )

                        # 매수호가 (높→낮 순서)
                        for b in bids:
                            bp = b['price']
                            bq = b['qty']
                            diff = ((bp / g_price) - 1) * 100 if g_price > 0 else 0
                            bar_w = int(bq / max_qty * 100) if max_qty > 0 else 0
                            bid_prices.append(bp)
                            html.append(
                                f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                                f'<td>매수</td>'
                                f'<td style="text-align:right">{bq:.2f}</td>'
                                f'<td style="text-align:right;font-weight:bold">{bp:,.0f}</td>'
                                f'<td style="text-align:right">{diff:+.2f}%</td>'
                                f'<td><div style="background:#D32F2F33;height:14px;width:{bar_w}%"></div></td></tr>'
                            )

                        html.append('</table>')
                        st.markdown(''.join(html), unsafe_allow_html=True)

                        # 호가 선택 selectbox
                        ask_prices.reverse()  # 낮→높 → 높→낮으로 복원
                        for ap in ask_prices:
                            price_labels.append(f"매도 {ap:,.0f}")
                        price_labels.append(f"── {g_price:,.0f} ──")
                        for bp in bid_prices:
                            price_labels.append(f"매수 {bp:,.0f}")

                        st.selectbox(
                            "호가 선택 → 주문가 반영", price_labels,
                            index=len(ask_prices),
                            key="_gold_ob_sel", on_change=_on_gold_ob_select
                        )

                        # 스프레드 정보
                        if asks and bids:
                            spread = asks[0]['price'] - bids[0]['price']
                            spread_pct = (spread / g_price * 100) if g_price > 0 else 0
                            total_ask_qty = sum(a['qty'] for a in asks)
                            total_bid_qty = sum(b['qty'] for b in bids)
                            st.caption(
                                f"스프레드: {spread:,.0f}원 ({spread_pct:.3f}%) | "
                                f"매도잔량: {total_ask_qty:.2f}g | 매수잔량: {total_bid_qty:.2f}g"
                            )
                    else:
                        st.info("호가 데이터를 불러오는 중...")

                # ── 우: 주문 패널 ──
                with order_col:
                    buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                    with buy_tab:
                        gb_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="g_buy_type")

                        if gb_type == "시장가":
                            gb_amount = st.number_input(
                                "매수 금액 (원)", min_value=10_000,
                                value=min(int(g_cash * 0.99), 1_000_000) if g_cash > 10_000 else 10_000,
                                step=10_000, key="g_buy_amt"
                            )
                            # % 버튼
                            gqb1, gqb2, gqb3, gqb4 = st.columns(4)
                            if gqb1.button("10%", key="gb10"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.1)
                                st.rerun()
                            if gqb2.button("25%", key="gb25"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.25)
                                st.rerun()
                            if gqb3.button("50%", key="gb50"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.5)
                                st.rerun()
                            if gqb4.button("100%", key="gb100"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.999)
                                st.rerun()

                            if g_price > 0:
                                st.caption(f"예상 수량: ~{gb_amount / g_price:.2f}g")

                            if st.button("시장가 매수", type="primary", key="g_buy_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gb_amount < 10_000:
                                    st.toast("최소 주문금액: 10,000원", icon="⚠️")
                                elif gb_amount > g_cash:
                                    st.toast(f"예수금 부족 ({g_cash:,.0f}원)", icon="⚠️")
                                else:
                                    with st.spinner("매수 주문 중..."):
                                        if gold_trader.auth():
                                            cur_p = gold_trader.get_current_price(GOLD_CODE_1KG) or g_price or 1
                                            buy_qty = round(gb_amount / cur_p, 2)
                                            result = gold_trader.send_order("BUY", GOLD_CODE_1KG, qty=buy_qty, ord_tp="3")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 시장가 매수! {gb_amount:,.0f}원 ≈ {buy_qty:.2f}g", icon="🟢")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "시장가 매수", "detail": f"{gb_amount:,.0f}원 ≈ {buy_qty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"매수 실패: {result}", icon="🔴")

                        else:  # 지정가
                            gbc1, gbc2 = st.columns(2)
                            gb_price = gbc1.number_input(
                                "매수 가격 (원/g)", min_value=1,
                                value=_gold_align_price(g_price * 0.99) if g_price > 0 else 100_000,
                                step=GOLD_TICK, key="gold_buy_price"
                            )
                            gb_qty = gbc2.number_input(
                                "매수 수량 (g)", min_value=0.01, value=max(GOLD_MIN_QTY, 1.0),
                                step=0.01, format="%.2f", key="g_buy_qty"
                            )
                            gb_total = gb_price * gb_qty
                            st.caption(f"총액: **{gb_total:,.0f}원** | 호가: {GOLD_TICK}원 | 최소: {GOLD_MIN_QTY}g")

                            gqbc1, gqbc2, gqbc3, gqbc4 = st.columns(4)
                            if gb_price > 0:
                                if gqbc1.button("10%", key="glb10"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.1 / gb_price, 2)
                                    st.rerun()
                                if gqbc2.button("25%", key="glb25"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.25 / gb_price, 2)
                                    st.rerun()
                                if gqbc3.button("50%", key="glb50"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.5 / gb_price, 2)
                                    st.rerun()
                                if gqbc4.button("100%", key="glb100"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.999 / gb_price, 2)
                                    st.rerun()

                            if st.button("지정가 매수", type="primary", key="g_lbuy_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gb_total < 10_000:
                                    st.toast("최소 주문금액: 10,000원", icon="⚠️")
                                elif gb_total > g_cash:
                                    st.toast(f"예수금 부족 ({g_cash:,.0f}원)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매수 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("BUY", GOLD_CODE_1KG, qty=gb_qty, price=gb_price, ord_tp="1")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 지정가 매수 등록! {gb_price:,.0f}원 × {gb_qty:.2f}g", icon="🟢")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "지정가 매수", "detail": f"{gb_price:,.0f}원 × {gb_qty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

                    with sell_tab:
                        gs_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="g_sell_type")

                        if gs_type == "시장가":
                            gs_qty = st.number_input(
                                f"매도 수량 (g)", min_value=0.01,
                                value=g_qty if g_qty > 0 else 1.0,
                                step=0.01, format="%.2f", key="g_sell_qty"
                            )
                            gqs1, gqs2, gqs3, gqs4 = st.columns(4)
                            if gqs1.button("25%", key="gs25"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.25, 2)
                                st.rerun()
                            if gqs2.button("50%", key="gs50"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.5, 2)
                                st.rerun()
                            if gqs3.button("75%", key="gs75"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.75, 2)
                                st.rerun()
                            if gqs4.button("100%", key="gs100"):
                                st.session_state['g_sell_qty'] = round(g_qty, 2)
                                st.rerun()

                            if g_price > 0:
                                st.caption(f"예상 금액: ~{gs_qty * g_price:,.0f}원")

                            if st.button("시장가 매도", type="primary", key="g_sell_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gs_qty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif gs_qty > g_qty:
                                    st.toast(f"보유량 초과 ({g_qty:.2f}g)", icon="⚠️")
                                else:
                                    with st.spinner("매도 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("SELL", GOLD_CODE_1KG, qty=gs_qty, ord_tp="3")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 시장가 매도! {gs_qty:.2f}g", icon="🔴")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "시장가 매도", "detail": f"{gs_qty:.2f}g ≈ {gs_qty * g_price:,.0f}원",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"매도 실패: {result}", icon="🔴")

                        else:  # 지정가
                            gsc1, gsc2 = st.columns(2)
                            gs_price = gsc1.number_input(
                                "매도 가격 (원/g)", min_value=1,
                                value=_gold_align_price(g_price * 1.01) if g_price > 0 else 100_000,
                                step=GOLD_TICK, key="gold_sell_price"
                            )
                            gs_lqty = gsc2.number_input(
                                "매도 수량 (g)", min_value=0.01,
                                value=g_qty if g_qty > 0.01 else 1.0,
                                step=0.01, format="%.2f", key="g_sell_lqty"
                            )
                            gs_total = gs_price * gs_lqty
                            st.caption(f"총액: **{gs_total:,.0f}원** | 호가: {GOLD_TICK}원")

                            gqsc1, gqsc2, gqsc3, gqsc4 = st.columns(4)
                            if g_qty > 0:
                                if gqsc1.button("25%", key="gls25"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.25, 2)
                                    st.rerun()
                                if gqsc2.button("50%", key="gls50"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.5, 2)
                                    st.rerun()
                                if gqsc3.button("75%", key="gls75"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.75, 2)
                                    st.rerun()
                                if gqsc4.button("100%", key="gls100"):
                                    st.session_state['g_sell_lqty'] = round(g_qty, 2)
                                    st.rerun()

                            if st.button("지정가 매도", type="primary", key="g_lsell_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gs_lqty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif gs_lqty > g_qty:
                                    st.toast(f"보유량 초과 ({g_qty:.2f}g)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매도 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("SELL", GOLD_CODE_1KG, qty=gs_lqty, price=gs_price, ord_tp="1")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 지정가 매도 등록! {gs_price:,.0f}원 × {gs_lqty:.2f}g", icon="🔴")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "지정가 매도", "detail": f"{gs_price:,.0f}원 × {gs_lqty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

            gold_trading_panel()

    # ══════════════════════════════════════════════════════
    # Tab 3: 백테스트 (코인 탭4와 동일 구조 - 3개 서브탭)
    # ══════════════════════════════════════════════════════
    with tab_g3:
        gbt1, gbt2 = st.tabs(["📈 단일 백테스트", "🛠️ 파라미터 최적화"])

        # ── 서브탭1: 단일 백테스트 ──────────────────────────
        with gbt1:
            st.header("금현물 단일 백테스트")

            # ── 데이터 가용 범위 + 사전 다운로드 ──
            import data_cache as _dc_gold
            _gold_info = _dc_gold.gold_cache_info()
            if _gold_info["exists"]:
                _gi_start = _gold_info["start"]
                _gi_end = _gold_info["end"]
                _gi_start_str = _gi_start.strftime('%Y-%m-%d') if hasattr(_gi_start, 'strftime') else str(_gi_start)[:10]
                _gi_end_str = _gi_end.strftime('%Y-%m-%d') if hasattr(_gi_end, 'strftime') else str(_gi_end)[:10]
                st.info(f"사용 가능 데이터: **{_gold_info['rows']:,}**개 캔들 ({_gi_start_str} ~ {_gi_end_str})")
            else:
                st.warning("캐시된 Gold 데이터가 없습니다. 아래 버튼으로 사전 다운로드하세요.")

            if gold_trader and st.button("Gold 일봉 전체 다운로드 (2014~ 전체)", key="gold_predownload"):
                with st.status("Gold 일봉 다운로드 중...", expanded=True) as dl_status:
                    prog_dl = st.progress(0)
                    log_dl = st.empty()
                    def _dl_progress(fetched, total, msg):
                        pct = min(fetched / total, 1.0) if total > 0 else 1.0
                        prog_dl.progress(pct)
                        log_dl.text(msg)
                    df_dl = _dc_gold.fetch_and_cache_gold(gold_trader, count=5000, progress_callback=_dl_progress)
                    if df_dl is not None and len(df_dl) > 0:
                        dl_status.update(label=f"완료! {len(df_dl):,}개 캔들 다운로드됨", state="complete")
                        st.rerun()
                    else:
                        dl_status.update(label="다운로드 실패", state="error")

            st.divider()

            bt_col1, bt_col2, bt_col3 = st.columns(3)
            with bt_col1:
                bt_strategy = st.selectbox("전략", ["Donchian", "SMA"], key="gold_bt_strat")
            with bt_col2:
                bt_buy_p  = st.number_input("매수 기간", min_value=5, max_value=300, value=buy_period,  step=1, key="gold_bt_buy")
            with bt_col3:
                bt_sell_p = st.number_input("매도 기간", min_value=5, max_value=300, value=sell_period, step=1, key="gold_bt_sell",
                                            help="Donchian 매도 채널 (SMA는 무시됨)")

            bt_start = st.date_input("백테스트 시작일", value=gold_start_date, key="gold_bt_start")
            bt_cap   = st.number_input("초기 자본 (원)", value=gold_initial_cap, step=100_000, format="%d", key="gold_bt_cap")

            if st.button("🚀 백테스트 실행", key="gold_bt_run", type="primary"):
                df_bt = load_gold_data(bt_buy_p)
                if df_bt is None or len(df_bt) < bt_buy_p + 5:
                    st.error("데이터 부족. 사전 다운로드를 실행하세요.")
                else:
                    st.caption(f"조회된 캔들: {len(df_bt):,}개 ({df_bt.index[0].strftime('%Y-%m-%d')} ~ {df_bt.index[-1].strftime('%Y-%m-%d')})")
                    with st.spinner("백테스트 실행 중..."):
                        engine = BacktestEngine()
                        result = engine.run_backtest(
                            ticker=None, df=df_bt,
                            period=bt_buy_p,
                            interval="day",
                            fee=0.003,
                            start_date=str(bt_start),
                            initial_balance=bt_cap,
                            strategy_mode=bt_strategy,
                            sell_period_ratio=(bt_sell_p / bt_buy_p) if bt_strategy == "Donchian" else 1.0,
                            slippage=0.0,
                        )

                    if "error" in result:
                        st.error(f"백테스트 오류: {result['error']}")
                    else:
                        p = result["performance"]
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("총 수익률",  f"{p['total_return']:+.2f}%")
                        r2.metric("CAGR",      f"{p['cagr']:+.2f}%")
                        r3.metric("MDD",       f"{p['mdd']:.2f}%")
                        r4.metric("샤프 비율",  f"{p['sharpe']:.2f}")
                        r5, r6, r7, r8 = st.columns(4)
                        r5.metric("매매 횟수",  f"{p['trade_count']}회")
                        r6.metric("승률",      f"{p['win_rate']:.1f}%")
                        r7.metric("최종 자산",  f"{p['final_equity']:,.0f}원")
                        calmar = abs(p['cagr'] / p['mdd']) if p['mdd'] != 0 else 0
                        r8.metric("Calmar",    f"{calmar:.2f}")

                        # ── 연도별 성과 (상세 컬럼) ──────────────────
                        df_result_full = result.get("df")
                        equity_curve   = result.get("equity_curve")

                        if df_result_full is not None and equity_curve is not None:
                            eq_series = pd.Series(equity_curve, index=df_result_full.index[-len(equity_curve):])
                            yearly_rows_g = []
                            years = eq_series.index.year.unique()
                            for yr in sorted(years):
                                yr_eq = eq_series[eq_series.index.year == yr]
                                if yr_eq.empty:
                                    continue
                                yr_start_eq = yr_eq.iloc[0]
                                yr_end_eq   = yr_eq.iloc[-1]
                                yr_ret      = (yr_end_eq / yr_start_eq - 1) * 100
                                peak_yr     = yr_eq.cummax()
                                yr_mdd      = ((yr_eq - peak_yr) / peak_yr * 100).min()
                                # 상태: 연도 마지막 일의 포지션
                                yr_trades = [t for t in p.get("trades", []) if pd.Timestamp(t["date"]).year <= yr]
                                last_type = yr_trades[-1]["type"] if yr_trades else "—"
                                yr_state = "보유" if last_type == "buy" else "현금"
                                yearly_rows_g.append({
                                    "연도": yr,
                                    "수익률(%)": f"{yr_ret:+.2f}",
                                    "MDD(%)": f"{yr_mdd:.2f}",
                                    "시작자산(원)": f"{yr_start_eq:,.0f}",
                                    "최종자산(원)": f"{yr_end_eq:,.0f}",
                                    "상태": yr_state,
                                })
                            if yearly_rows_g:
                                st.subheader("연도별 성과")
                                st.dataframe(pd.DataFrame(yearly_rows_g), use_container_width=True, hide_index=True)

                        # ── 에쿼티 커브 차트 ──────────────────────────
                        if equity_curve is not None and len(equity_curve) > 0:
                            df_eq = pd.DataFrame({"equity": equity_curve})
                            df_eq.index = df_result_full.index[-len(equity_curve):]
                            df_eq["return_pct"] = (df_eq["equity"] / bt_cap - 1) * 100
                            # Buy & Hold 비교
                            df_bt_chart = df_bt.loc[df_bt.index >= pd.Timestamp(str(bt_start))]
                            if not df_bt_chart.empty:
                                bh_base = df_bt_chart["close"].iloc[0]
                                bh_pct  = (df_bt_chart["close"] / bh_base - 1) * 100

                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                x=df_eq.index, y=df_eq["return_pct"], mode="lines",
                                name="전략", line=dict(color="gold", width=2)
                            ))
                            if not df_bt_chart.empty:
                                fig_eq.add_trace(go.Scatter(
                                    x=bh_pct.index, y=bh_pct.values, mode="lines",
                                    name="Buy & Hold", line=dict(color="gray", width=1, dash="dot")
                                ))
                            fig_eq.update_layout(
                                title=f"누적 수익률 ({bt_strategy} {bt_buy_p}/{bt_sell_p})",
                                yaxis_title="수익률 (%)", height=350,
                                margin=dict(l=0, r=0, t=40, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                            )
                            st.plotly_chart(fig_eq, use_container_width=True)

                        # ── 거래 내역 ─────────────────────────────────
                        with st.expander("거래 내역"):
                            trades_list = p.get("trades", [])
                            if trades_list:
                                st.dataframe(pd.DataFrame(trades_list), use_container_width=True, hide_index=True)
                            else:
                                st.info("실행된 거래가 없습니다.")

        # ── 서브탭2: 파라미터 최적화 ────────────────────────
        with gbt2:
            st.header("파라미터 최적화")

            # 데이터 가용 범위 표시
            _gold_info_opt = _dc_gold.gold_cache_info()
            if _gold_info_opt["exists"]:
                _gio_s = _gold_info_opt["start"]
                _gio_e = _gold_info_opt["end"]
                _gio_s_str = _gio_s.strftime('%Y-%m-%d') if hasattr(_gio_s, 'strftime') else str(_gio_s)[:10]
                _gio_e_str = _gio_e.strftime('%Y-%m-%d') if hasattr(_gio_e, 'strftime') else str(_gio_e)[:10]
                st.info(f"사용 가능 데이터: **{_gold_info_opt['rows']:,}**개 캔들 ({_gio_s_str} ~ {_gio_e_str})")
            else:
                st.warning("캐시된 Gold 데이터가 없습니다. 백테스트 탭에서 사전 다운로드를 실행하세요.")

            opt_strat_g = st.selectbox("전략", ["Donchian", "SMA"], key="gold_opt_strat")

            with st.form("gold_optimization_form"):
                gopt_method = st.radio("최적화 방법", ["Grid Search (전수 탐색)", "Optuna (베이지안 최적화)"], horizontal=True, key="gold_opt_method")
                use_optuna_g = "Optuna" in gopt_method

                if opt_strat_g == "Donchian":
                    st.markdown("##### 매수 채널 기간")
                    goc1, goc2, goc3 = st.columns(3)
                    g_buy_start = goc1.number_input("시작", 5, 300, 30,  key="gold_opt_buy_start")
                    g_buy_end   = goc2.number_input("끝",   5, 300, 150, key="gold_opt_buy_end")
                    g_buy_step  = goc3.number_input("간격", 1, 50,  5,   key="gold_opt_buy_step")
                    st.markdown("##### 매도 채널 기간")
                    goc4, goc5, goc6 = st.columns(3)
                    g_sell_start = goc4.number_input("시작", 5, 300, 10, key="gold_opt_sell_start")
                    g_sell_end   = goc5.number_input("끝",   5, 300, 80, key="gold_opt_sell_end")
                    g_sell_step  = goc6.number_input("간격", 1, 50,  5,  key="gold_opt_sell_step")
                else:
                    st.markdown("##### SMA 기간")
                    goc1, goc2, goc3 = st.columns(3)
                    g_sma_start = goc1.number_input("시작", 5, 300, 10, key="gold_opt_sma_start")
                    g_sma_end   = goc2.number_input("끝",   5, 300, 100, key="gold_opt_sma_end")
                    g_sma_step  = goc3.number_input("간격", 1, 50,  5,   key="gold_opt_sma_step")

                if use_optuna_g:
                    st.divider()
                    st.markdown("##### Optuna 설정")
                    uoc1, uoc2 = st.columns(2)
                    g_n_trials  = uoc1.number_input("탐색 횟수", 50, 2000, 200, step=50, key="gold_optuna_trials")
                    g_obj_label = uoc2.selectbox("목적함수", ["Calmar (CAGR/|MDD|)", "Sharpe", "수익률 (Return)", "MDD 최소"], key="gold_optuna_obj")

                st.divider()
                g_opt_start = st.date_input("시작일", value=gold_start_date, key="gold_opt_start")
                g_opt_fee   = st.number_input("수수료 (%)", value=0.3, format="%.2f", key="gold_opt_fee") / 100
                g_opt_cap   = st.number_input("초기 자본 (원)", value=gold_initial_cap, step=100_000, format="%d", key="gold_opt_cap")

                gopt_submitted = st.form_submit_button("최적화 시작", type="primary")

            if gopt_submitted:
                import plotly.express as px
                df_opt_src = load_gold_data(max(
                    g_buy_end if opt_strat_g == "Donchian" else g_sma_end, 300
                ))
                if df_opt_src is None or df_opt_src.empty:
                    st.error("데이터 로드 실패. 백테스트 탭에서 사전 다운로드를 실행하세요.")
                else:
                    st.caption(f"조회된 캔들: {len(df_opt_src):,}개 ({df_opt_src.index[0].strftime('%Y-%m-%d')} ~ {df_opt_src.index[-1].strftime('%Y-%m-%d')})")
                    with st.status("최적화 진행 중...", expanded=True) as gopt_status:
                        prog_bar_g  = st.progress(0)
                        log_area_g  = st.empty()

                        def g_opt_progress(idx, total, msg):
                            pct = min(idx / total, 1.0) if total > 0 else 0
                            prog_bar_g.progress(pct)
                            log_area_g.text(f"{msg} ({idx}/{total})")

                        engine_g = BacktestEngine()
                        try:
                            if use_optuna_g:
                                obj_map_g = {"Calmar (CAGR/|MDD|)": "calmar", "Sharpe": "sharpe", "수익률 (Return)": "return", "MDD 최소": "mdd"}
                                obj_key_g = obj_map_g.get(g_obj_label, "calmar")
                                if opt_strat_g == "Donchian":
                                    opt_res_g = engine_g.optuna_optimize(
                                        df_opt_src, strategy_mode="Donchian",
                                        buy_range=(g_buy_start, g_buy_end),
                                        sell_range=(g_sell_start, g_sell_end),
                                        fee=g_opt_fee, slippage=0.0,
                                        start_date=str(g_opt_start),
                                        initial_balance=g_opt_cap,
                                        n_trials=g_n_trials,
                                        objective_metric=obj_key_g,
                                        progress_callback=g_opt_progress,
                                    )
                                else:
                                    opt_res_g = engine_g.optuna_optimize(
                                        df_opt_src, strategy_mode="SMA",
                                        buy_range=(g_sma_start, g_sma_end),
                                        fee=g_opt_fee, slippage=0.0,
                                        start_date=str(g_opt_start),
                                        initial_balance=g_opt_cap,
                                        n_trials=g_n_trials,
                                        objective_metric=obj_key_g,
                                        progress_callback=g_opt_progress,
                                    )
                                gopt_results = opt_res_g.get("trials", [])
                                best_params  = opt_res_g.get("best_params", {})
                                gopt_status.update(label="Optuna 최적화 완료!", state="complete")
                                st.success(f"최적 파라미터: {best_params} | 목적함수 값: {opt_res_g['best_value']:.2f}")
                            else:
                                if opt_strat_g == "Donchian":
                                    buy_r  = range(g_buy_start,  g_buy_end  + 1, g_buy_step)
                                    sell_r = range(g_sell_start, g_sell_end + 1, g_sell_step)
                                    gopt_results = engine_g.optimize_donchian(
                                        df_opt_src, buy_range=buy_r, sell_range=sell_r,
                                        fee=g_opt_fee, slippage=0.0,
                                        start_date=str(g_opt_start),
                                        initial_balance=g_opt_cap,
                                        progress_callback=g_opt_progress,
                                    )
                                else:
                                    sma_r = range(g_sma_start, g_sma_end + 1, g_sma_step)
                                    gopt_results = engine_g.optimize_sma(
                                        df_opt_src, sma_range=sma_r,
                                        fee=g_opt_fee, slippage=0.0,
                                        start_date=str(g_opt_start),
                                        initial_balance=g_opt_cap,
                                        progress_callback=g_opt_progress,
                                    )
                                gopt_status.update(label="Grid Search 완료!", state="complete")

                        except Exception as e:
                            gopt_status.update(label=f"오류: {e}", state="error")
                            gopt_results = []

                    if gopt_results:
                        df_opt_res = pd.DataFrame(gopt_results)
                        df_opt_res["calmar"] = df_opt_res.apply(
                            lambda r: abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0, axis=1
                        )

                        # 정렬 기준: Calmar
                        df_opt_res_sorted = df_opt_res.sort_values("calmar", ascending=False)

                        st.subheader("상위 20개 파라미터")
                        disp_cols = ["Buy Period", "Sell Period", "total_return", "cagr", "mdd", "sharpe", "calmar", "win_rate", "trade_count"] \
                            if opt_strat_g == "Donchian" else \
                            ["sma_period", "total_return", "cagr", "mdd", "sharpe", "calmar", "win_rate", "trade_count"]
                        disp_cols = [c for c in disp_cols if c in df_opt_res_sorted.columns]
                        st.dataframe(
                            df_opt_res_sorted[disp_cols].head(20).style.format({
                                "total_return": "{:.2f}%", "cagr": "{:.2f}%",
                                "mdd": "{:.2f}%", "sharpe": "{:.2f}",
                                "calmar": "{:.2f}", "win_rate": "{:.1f}%",
                            }),
                            use_container_width=True, hide_index=True
                        )

                        # 히트맵 (Donchian Grid Search일 때)
                        if opt_strat_g == "Donchian" and not use_optuna_g and \
                                "Buy Period" in df_opt_res.columns and "Sell Period" in df_opt_res.columns:
                            st.subheader("Calmar 히트맵 (Buy × Sell)")
                            try:
                                df_heat = df_opt_res.pivot_table(index="Buy Period", columns="Sell Period", values="calmar")
                                fig_heat = px.imshow(
                                    df_heat, color_continuous_scale="RdYlGn",
                                    labels=dict(x="Sell Period", y="Buy Period", color="Calmar"),
                                    aspect="auto"
                                )
                                fig_heat.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=30))
                                st.plotly_chart(fig_heat, use_container_width=True)
                            except Exception:
                                pass

                        # 전체 결과 다운로드
                        csv_data_g = df_opt_res_sorted.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 전체 결과 다운로드 (CSV)", data=csv_data_g,
                                           file_name="gold_optimization_results.csv", mime="text/csv")

    # ══════════════════════════════════════════════════════
    # Tab 4: 수수료/세금 (기존 내용 유지)
    # ══════════════════════════════════════════════════════
    with tab_g4:
        st.header("KRX 금현물 수수료 및 세금 안내")
        st.caption("키움증권 기준 | 수수료는 변경될 수 있으니 공식 홈페이지에서 최신 정보를 확인하세요.")

        st.subheader("1. 거래 수수료")
        fee_data = pd.DataFrame([
            {"구분": "매매 수수료", "요율/금액": "약 0.165% (온라인)", "비고": "매수/매도 각각 부과, 부가세 포함"},
            {"구분": "유관기관 수수료", "요율/금액": "0.0046396%", "비고": "거래소/예탁원 등 (매매금액 기준)"},
            {"구분": "합계 (편도)", "요율/금액": "약 0.17%", "비고": "매수 또는 매도 1회당"},
            {"구분": "합계 (왕복)", "요율/금액": "약 0.34%", "비고": "매수 + 매도 1세트"},
        ])
        st.dataframe(fee_data, use_container_width=True, hide_index=True)

        st.subheader("2. 세금")
        tax_data = pd.DataFrame([
            {"구분": "양도소득세", "세율": "비과세", "비고": "KRX 금시장 매매차익은 양도세 면제"},
            {"구분": "배당소득세", "세율": "15.4%", "비고": "보관료 환급금(이자) 발생 시"},
            {"구분": "부가가치세 (매매)", "세율": "면세", "비고": "KRX 장내 거래 시 부가세 없음"},
            {"구분": "부가가치세 (실물 인출)", "세율": "10%", "비고": "평균 매수단가 x 인출 수량 기준"},
        ])
        st.dataframe(tax_data, use_container_width=True, hide_index=True)
        st.info("KRX 금시장의 최대 장점: **매매차익 비과세 + 부가세 면세** (실물 인출 시에만 부가세 10% 부과)")

        st.subheader("3. 보관료")
        st.markdown(
            "| 항목 | 내용 |\n"
            "|------|------|\n"
            "| **보관료율** | 매일 잔량의 시가 환산 금액 x **0.02% (연율)** |\n"
            "| **일할 계산** | 시가 x 보유수량 x 0.0002% / 365일 |\n"
            "| **부가세** | 별도 (보관료의 10%) |\n"
            "| **부과 주기** | 매월 말 정산 |\n"
            "\n"
            "> 예시: 금 100g 보유, 시가 13만원/g → 연간 보관료 약 **2,600원** (부가세 별도)\n"
        )

        st.subheader("4. 금 투자 방법별 비교")
        compare_data = pd.DataFrame([
            {"투자 방법": "KRX 금현물", "매매차익 세금": "비과세", "부가세": "면세 (인출시 10%)", "거래 수수료": "~0.17%", "실물 인출": "가능 (100g/1kg)"},
            {"투자 방법": "금 ETF", "매매차익 세금": "15.4%", "부가세": "해당없음", "거래 수수료": "~0.015%+보수", "실물 인출": "불가"},
            {"투자 방법": "골드뱅킹 (은행)", "매매차익 세금": "15.4%", "부가세": "매입시 면세, 인출시 10%", "거래 수수료": "~1%", "실물 인출": "가능"},
            {"투자 방법": "금 실물 (귀금속점)", "매매차익 세금": "비과세", "부가세": "10% (매입 시)", "거래 수수료": "5~15% (스프레드)", "실물 인출": "즉시"},
        ])
        st.dataframe(compare_data, use_container_width=True, hide_index=True)
        st.caption("출처: 키움증권 금현물 수수료 안내, KRX 금시장 안내서 | 수수료는 변경될 수 있습니다.")





def render_kis_isa_mode():
    """KIS ISA 위대리(WDR) 전략 모드 - 4탭 구성."""
    from kis_trader import KISTrader
    from strategy.widaeri import WDRStrategy
    import data_cache as _dc

    st.title("ISA 위대리(WDR) 전략")

    # ── 사이드바 설정 ──
    st.sidebar.header("ISA 설정")
    kis_ak = _get_runtime_value(("KIS_ISA_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_ISA_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(config.get("kis_isa_account_no", "") or _get_runtime_value(("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(config.get("kis_isa_prdt_cd", "") or _get_runtime_value(("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))

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

    _isa_trade_options = _build_etf_options(["418660", "409820", "423920", "465610", "461910", "133690"])
    _isa_trend_options = _build_etf_options(["133690", "360750", "453850", "251350", "418660", "409820", "423920", "465610", "461910"])

    _saved_trade_etf = str(config.get("kis_isa_etf_code", _get_runtime_value("KIS_ISA_ETF_CODE", "418660")))
    _saved_trend_etf = str(config.get("kis_isa_trend_etf_code", _get_runtime_value("KIS_ISA_TREND_ETF_CODE", "133690")))

    if _saved_trade_etf and _saved_trade_etf not in _isa_trade_options.values():
        _isa_trade_options[f"{_saved_trade_etf} {_etf_name_kr(_saved_trade_etf)}"] = _saved_trade_etf
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
    isa_etf_code = _isa_trade_options[selected_etf_label]
    isa_trend_etf_code = _isa_trend_options[selected_trend_etf_label]

    wdr_ov = st.sidebar.number_input(
        "고평가 임계값 (%)", min_value=0.0, max_value=30.0,
        value=float(config.get("kis_isa_wdr_ov", 5.0)), step=0.5,
        key="isa_wdr_ov", disabled=IS_CLOUD,
    )
    wdr_un = st.sidebar.number_input(
        "저평가 임계값 (%)", min_value=-30.0, max_value=0.0,
        value=float(config.get("kis_isa_wdr_un", -6.0)), step=0.5,
        key="isa_wdr_un", disabled=IS_CLOUD,
    )
    _isa_start_default = config.get("kis_isa_start_date", "2020-01-01")
    isa_start_date = st.sidebar.date_input(
        "시작일",
        value=pd.to_datetime(_isa_start_default).date(),
        key="isa_start_date",
        disabled=IS_CLOUD,
    )

    if not IS_CLOUD and st.sidebar.button("ISA 설정 저장", key="isa_save_cfg"):
        new_cfg = config.copy()
        new_cfg["kis_isa_account_no"] = str(kis_acct).strip()
        new_cfg["kis_isa_prdt_cd"] = str(kis_prdt).strip() or "01"
        new_cfg["kis_isa_etf_code"] = isa_etf_code
        new_cfg["kis_isa_trend_etf_code"] = isa_trend_etf_code
        new_cfg["kis_isa_wdr_ov"] = float(wdr_ov)
        new_cfg["kis_isa_wdr_un"] = float(wdr_un)
        new_cfg["kis_isa_start_date"] = str(isa_start_date)
        save_config(new_cfg)
        st.sidebar.success("ISA 설정을 저장했습니다.")

    if not (kis_ak and kis_sk and kis_acct):
        st.warning("KIS ISA API 키와 계좌번호를 설정해 주세요.")
        return

    trader = KISTrader(is_mock=False)
    trader.app_key = kis_ak
    trader.app_secret = kis_sk
    trader.account_no = kis_acct
    trader.acnt_prdt_cd = kis_prdt

    # 토큰 캐싱 — F5 때만 새로 인증, 이후 rerun에서는 캐시된 토큰 재사용
    _isa_token_key = f"isa_token_{kis_acct}"
    _cached = st.session_state.get(_isa_token_key)
    if _cached and (_cached["expiry"] - time.time()) > 300:
        trader.access_token = _cached["token"]
        trader.token_expiry = _cached["expiry"]
    else:
        if not trader.auth():
            st.error("KIS 인증에 실패했습니다. API 키/계좌를 확인해 주세요.")
            return
        st.session_state[_isa_token_key] = {"token": trader.access_token, "expiry": trader.token_expiry}

    isa_bal_key = f"isa_balance_cache_{kis_acct}_{kis_prdt}"

    tab_i1, tab_i2, tab_i3, tab_i4, tab_i5 = st.tabs([
        "🚀 실시간 모니터링", "🛒 수동 주문", "📋 주문방식", "💳 수수료/세금", "📊 QQQ→TQQQ 백테스트"
    ])

    # ══════════════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링 (잔고 + WDR 시그널)
    # ══════════════════════════════════════════════════════════════
    with tab_i1:
        st.header("WDR 시그널 모니터링")
        st.caption(f"매매 ETF: {_fmt_etf_code_name(isa_etf_code)} | TREND ETF: {_fmt_etf_code_name(isa_trend_etf_code)}")

        # 잔고 표시 — F5 새로고침 시 자동 조회, 이후 캐시 사용
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
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt}")
        elif bal.get("error"):
            st.error(f"잔고 조회 API 오류: [{bal.get('msg_cd', '')}] {bal.get('msg1', '알 수 없는 오류')}")
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt} / rt_cd: {bal.get('rt_cd', '')}")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []
            total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
            stock_eval = total_eval - cash

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("예수금", f"{cash:,.0f} KRW")
            m2.metric("주식 평가", f"{stock_eval:,.0f} KRW")
            m3.metric("총 평가", f"{total_eval:,.0f} KRW")
            m4.metric("보유 종목 수", f"{len(holdings)}")

            if holdings:
                df_h = pd.DataFrame(holdings)
                cols = [c for c in ["code", "name", "qty", "avg_price", "cur_price", "eval_amt", "pnl_rate"] if c in df_h.columns]
                st.dataframe(df_h[cols], use_container_width=True, hide_index=True)

        st.divider()

        # WDR 시그널 자동 계산 (버튼 없이 항상 표시)
        isa_sig_params = {
            "trade_etf": str(isa_etf_code),
            "trend_etf": str(isa_trend_etf_code),
            "ov": float(wdr_ov),
            "un": float(wdr_un),
            "acct": str(kis_acct),
            "prdt": str(kis_prdt),
        }

        def _compute_isa_signal_result():
            sig_df = trader.get_daily_chart(str(isa_trend_etf_code), count=1500)
            if sig_df is None or len(sig_df) < 260:
                _fallback_df = _dc.load_bundled_csv(str(isa_trend_etf_code))
                if _fallback_df is not None and len(_fallback_df) >= 260:
                    sig_df = _fallback_df

            if sig_df is None or len(sig_df) < 260:
                return {"error": f"시그널 데이터가 부족합니다. ({isa_trend_etf_code})"}

            strategy = WDRStrategy(settings={
                "overvalue_threshold": float(wdr_ov),
                "undervalue_threshold": float(wdr_un),
            })
            signal = strategy.analyze(sig_df)
            if not signal:
                return {"error": "WDR 분석에 실패했습니다."}

            bal_local = trader.get_balance() or {"cash": 0.0, "holdings": []}
            cash_local = float(bal_local.get("cash", 0.0))
            holdings_local = bal_local.get("holdings", []) or []
            etf_holding = next((h for h in holdings_local if str(h.get("code", "")) == str(isa_etf_code)), None)
            qty = int(etf_holding.get("qty", 0)) if etf_holding else 0
            cur = trader.get_current_price(str(isa_etf_code)) or 0.0

            weekly_pnl = 0.0
            ch = trader.get_daily_chart(str(isa_etf_code), count=10)
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

            weekly = strategy.daily_to_weekly(sig_df)
            trend = strategy.calc_growth_trend(weekly)
            return {
                "signal": signal,
                "action": action,
                "weekly_df": weekly,
                "trend": trend,
                "balance": bal_local,
            }

        if st.session_state.get("isa_signal_result") is None or st.session_state.get("isa_signal_params") != isa_sig_params:
            with st.spinner("WDR 시그널을 자동 계산하는 중입니다..."):
                st.session_state["isa_signal_result"] = _compute_isa_signal_result()
                st.session_state["isa_signal_params"] = isa_sig_params
                if isinstance(st.session_state["isa_signal_result"], dict) and st.session_state["isa_signal_result"].get("balance"):
                    st.session_state[isa_bal_key] = st.session_state["isa_signal_result"]["balance"]

        res = st.session_state.get("isa_signal_result")
        if res:
            if res.get("error"):
                st.error(res["error"])
            else:
                sig = res["signal"]
                act = res["action"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("이격도", f"{float(sig['divergence']):+.2f}%")
                c2.metric("시장 상태", str(sig["state"]))
                c3.metric("매도 비율", f"{float(sig['sell_ratio']):.1f}%")
                c4.metric("매수 비율", f"{float(sig['buy_ratio']):.1f}%")

                # 권장 동작
                act_str = act["action"] or "HOLD"
                st.info(f"**권장 동작**: {act_str} {act['quantity']}주 | 주간 손익: {act['weekly_pnl']:+,.0f} KRW")

                # 추세선 차트
                weekly_df = res.get("weekly_df")
                trend = res.get("trend")
                if weekly_df is not None and trend is not None:
                    import plotly.graph_objects as go
                    start_ts = pd.Timestamp(isa_start_date)
                    mask = weekly_df.index >= start_ts
                    weekly_plot = weekly_df.loc[mask]
                    trend_plot = np.asarray(trend)[mask]
                    if len(weekly_plot) == 0:
                        st.warning(f"시작일({start_ts.date()}) 이후 표시할 데이터가 없습니다.")
                    else:
                        fig = go.Figure()
                        _trend_label = _fmt_etf_code_name(isa_trend_etf_code)
                        fig.add_trace(go.Scatter(
                            x=weekly_plot.index, y=weekly_plot["close"],
                            name=f"{_trend_label} 주간 종가", line=dict(color="royalblue"),
                        ))
                        fig.add_trace(go.Scatter(
                            x=weekly_plot.index, y=trend_plot,
                            name="성장 추세선", line=dict(color="orange", dash="dash"),
                        ))
                        # 이격도 영역 표시
                        ov_th = float(wdr_ov) / 100.0
                        un_th = float(wdr_un) / 100.0
                        valid = ~np.isnan(trend_plot)
                        if valid.any():
                            ov_line = trend_plot.copy()
                            ov_line[valid] = trend_plot[valid] * (1 + ov_th)
                            un_line = trend_plot.copy()
                            un_line[valid] = trend_plot[valid] * (1 + un_th)
                            fig.add_trace(go.Scatter(
                                x=weekly_plot.index, y=ov_line,
                                name=f"고평가 (+{wdr_ov}%)", line=dict(color="red", dash="dot", width=1),
                            ))
                            fig.add_trace(go.Scatter(
                                x=weekly_plot.index, y=un_line,
                                name=f"저평가 ({wdr_un}%)", line=dict(color="green", dash="dot", width=1),
                            ))
                        fig.update_layout(
                            title=f"{_trend_label} 주간 종가 vs 성장 추세선",
                            xaxis_title="날짜", yaxis_title="가격",
                            height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # Tab 2: 수동 주문
    # ══════════════════════════════════════════════════════════════
    with tab_i2:
        st.header("수동 주문")
        st.caption(f"매매 대상: {selected_etf_label}")

        bal = st.session_state.get(isa_bal_key)
        if not bal:
            st.warning("잔고를 먼저 조회해 주세요. (모니터링 탭에서 잔고 새로고침)")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []
            etf_holding = next((h for h in holdings if str(h.get("code", "")) == str(isa_etf_code)), None)
            holding_qty = int(etf_holding.get("qty", 0)) if etf_holding else 0

            st.info(f"예수금: {cash:,.0f} KRW | {_fmt_etf_code_name(isa_etf_code)} 보유: {holding_qty}주")

            buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

            with buy_tab:
                buy_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)"], key="isa_buy_method", horizontal=True)
                buy_amount = st.number_input("매수 금액 (KRW)", min_value=0, value=0, step=10000, key="isa_buy_amt")

                cur_price = trader.get_current_price(str(isa_etf_code))
                if cur_price and cur_price > 0:
                    est_qty = int(buy_amount / cur_price) if buy_amount > 0 else 0
                    st.caption(f"현재가: {cur_price:,.0f}원 | 예상 수량: {est_qty}주")

                buy_price = 0
                if buy_method == "지정가":
                    buy_price = st.number_input("매수 지정가", min_value=0, value=int(cur_price or 0), step=50, key="isa_buy_price")

                if st.button("매수 실행", key="isa_exec_buy", type="primary", disabled=IS_CLOUD):
                    if buy_amount <= 0:
                        st.error("매수 금액을 입력해 주세요.")
                    elif buy_amount > cash:
                        st.error(f"예수금 부족 (예수금: {cash:,.0f}원)")
                    else:
                        with st.spinner("매수 주문 실행 중..."):
                            if buy_method == "동시호가 (장마감)":
                                result = trader.smart_buy_krw_closing(str(isa_etf_code), float(buy_amount))
                            elif buy_method == "지정가" and buy_price > 0:
                                qty = int(buy_amount / buy_price)
                                result = trader.send_order("BUY", str(isa_etf_code), qty, price=buy_price, ord_dvsn="00") if qty > 0 else None
                            else:
                                result = trader.smart_buy_krw(str(isa_etf_code), float(buy_amount))
                            if result and (isinstance(result, dict) and result.get("success")):
                                st.success(f"매수 주문 완료: {result}")
                                st.session_state[isa_bal_key] = trader.get_balance()
                            else:
                                st.error(f"매수 실패: {result}")

            with sell_tab:
                sell_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)"], key="isa_sell_method", horizontal=True)
                sell_qty = st.number_input("매도 수량", min_value=0, max_value=max(holding_qty, 1), value=holding_qty, step=1, key="isa_sell_qty")
                sell_all = st.checkbox("전량 매도", value=True, key="isa_sell_all")

                sell_price = 0
                if sell_method == "지정가":
                    sell_price = st.number_input("매도 지정가", min_value=0, value=int(cur_price or 0), step=50, key="isa_sell_price")

                if st.button("매도 실행", key="isa_exec_sell", type="primary", disabled=IS_CLOUD):
                    _sq = holding_qty if sell_all else sell_qty
                    if _sq <= 0:
                        st.error("매도할 수량이 없습니다.")
                    else:
                        with st.spinner("매도 주문 실행 중..."):
                            if sell_method == "동시호가 (장마감)":
                                if sell_all:
                                    result = trader.smart_sell_all_closing(str(isa_etf_code))
                                else:
                                    result = trader.smart_sell_qty_closing(str(isa_etf_code), _sq)
                            elif sell_method == "지정가" and sell_price > 0:
                                result = trader.send_order("SELL", str(isa_etf_code), _sq, price=sell_price, ord_dvsn="00")
                            else:
                                if sell_all:
                                    result = trader.smart_sell_all(str(isa_etf_code))
                                else:
                                    result = trader.smart_sell_qty(str(isa_etf_code), _sq)
                            if result and (isinstance(result, dict) and result.get("success")):
                                st.success(f"매도 주문 완료: {result}")
                                st.session_state[isa_bal_key] = trader.get_balance()
                            else:
                                st.error(f"매도 실패: {result}")

    # ══════════════════════════════════════════════════════════════
    # Tab 3: 주문방식
    # ══════════════════════════════════════════════════════════════
    with tab_i3:
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
        st.markdown(f"""
1. 매주 금요일 KST 15:20 실행 (`TRADING_MODE=kis_isa`)
2. TREND ETF({_fmt_etf_code_name(isa_trend_etf_code)}) 일봉 → 주봉 변환 → 성장 추세선 계산
3. 이격도 기반 시장 상태 판단 (고평가/중립/저평가)
4. 주간 손익 × 매도/매수 비율 → 주문 수량 산출
5. `execute_closing_auction_buy/sell` → 동시호가 주문 + 미체결 시 시간외 재주문
""")

    # ══════════════════════════════════════════════════════════════
    # Tab 4: 수수료/세금
    # ══════════════════════════════════════════════════════════════
    with tab_i4:
        st.header("ISA 계좌 수수료 및 세금 안내")
        st.subheader("1. 매매 수수료")
        st.dataframe(pd.DataFrame([
            {"증권사": "한국투자증권", "매매 수수료": "0.0140396%", "비고": "나무 온라인 (현재 사용)"},
            {"증권사": "키움증권", "매매 수수료": "0.015%", "비고": "영웅문 온라인"},
            {"증권사": "미래에셋", "매매 수수료": "0.014%", "비고": "m.Stock 온라인"},
            {"증권사": "토스증권", "매매 수수료": "무료~0.015%", "비고": "이벤트 시 무료"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("2. 매매 대상 ETF 보수")
        _isa_fee_rows = [
            {"ETF": "TIGER 미국나스닥100레버리지(합성)", "코드": "418660", "총보수": "0.25%", "구분": "나스닥 2x"},
            {"ETF": "KODEX 미국나스닥100레버리지(합성 H)", "코드": "409820", "총보수": "0.30%", "구분": "나스닥 2x(H)"},
            {"ETF": "TIGER 미국필라델피아반도체레버리지(합성)", "코드": "423920", "총보수": "0.58%", "구분": "반도체 2x"},
            {"ETF": "ACE 미국빅테크TOP7 Plus레버리지(합성)", "코드": "465610", "총보수": "0.60%", "구분": "빅테크 2x"},
            {"ETF": _etf_name_kr(isa_trend_etf_code), "코드": str(isa_trend_etf_code), "총보수": "-", "구분": "TREND ETF(시그널)"},
        ]
        st.dataframe(pd.DataFrame(_isa_fee_rows), use_container_width=True, hide_index=True)

        st.subheader("3. ISA 세제혜택")
        st.markdown("""
| 항목 | 내용 |
|------|------|
| 비과세 한도 | 일반형 200만원 / 서민·청년형 400만원 |
| 한도 초과 | 9.9% 분리과세 (일반 15.4% 대비 유리) |
| 의무가입기간 | 3년 |
| 납입한도 | 연 2,000만원 (총 1억원) |
""")
        st.caption("위대리 전략의 주간 리밸런싱 매매차익이 비과세 한도 내에서 면세 처리되어 절세 효과 극대화")

    # ══════════════════════════════════════════════════════════════
    # Tab 5: QQQ→TQQQ 백테스트
    # ══════════════════════════════════════════════════════════════
    with tab_i5:
        st.header("QQQ 추세 → TQQQ 트레이딩 백테스트")
        st.caption("QQQ가 200일 이동평균선 위일 때 TQQQ 보유, 아래일 때 현금(또는 SHY) 보유")

        import plotly.graph_objects as go

        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            bt_sma_period = st.number_input("SMA 기간 (일)", min_value=20, max_value=400, value=200, step=10, key="tqqq_sma")
        with bcol2:
            bt_start_year = st.number_input("시작 연도", min_value=2010, max_value=2025, value=2010, step=1, key="tqqq_start")
        with bcol3:
            bt_fee = st.number_input("매매 수수료 (%)", min_value=0.0, max_value=1.0, value=0.015, step=0.005, format="%.3f", key="tqqq_fee")

        bt_risk_off = st.radio("리스크 오프 자산", ["현금 (0% 수익)", "SHY (미국 단기채)"], key="tqqq_risk_off", horizontal=True)

        if st.button("백테스트 실행", key="tqqq_run_bt", type="primary"):
            with st.spinner("yfinance에서 QQQ/TQQQ 데이터를 다운로드하는 중..."):
                import yfinance as yf

                tickers_to_fetch = ["QQQ", "TQQQ"]
                if "SHY" in bt_risk_off:
                    tickers_to_fetch.append("SHY")

                start_str = f"{bt_start_year}-01-01"
                raw = yf.download(tickers_to_fetch, start=start_str, auto_adjust=True, progress=False)

                if raw is None or raw.empty:
                    st.error("데이터 다운로드에 실패했습니다. 네트워크를 확인해 주세요.")
                else:
                    # Close 가격 추출
                    if isinstance(raw.columns, pd.MultiIndex):
                        closes = raw["Close"].copy()
                    else:
                        closes = raw[["Close"]].copy()
                        closes.columns = tickers_to_fetch[:1]

                    closes = closes.dropna()
                    if len(closes) < bt_sma_period + 30:
                        st.error(f"데이터가 부족합니다. (최소 {bt_sma_period + 30}일 필요, 현재 {len(closes)}일)")
                    else:
                        qqq = closes["QQQ"]
                        tqqq = closes["TQQQ"]
                        shy = closes["SHY"] if "SHY" in closes.columns else None

                        # SMA 계산
                        qqq_sma = qqq.rolling(window=bt_sma_period).mean()

                        # 시그널: QQQ > SMA → 1 (TQQQ 보유), 아니면 0
                        signal = (qqq > qqq_sma).astype(int)
                        signal = signal.shift(1).fillna(0).astype(int)  # 다음날 반영

                        # 일별 수익률
                        tqqq_ret = tqqq.pct_change().fillna(0)
                        qqq_ret = qqq.pct_change().fillna(0)
                        shy_ret = shy.pct_change().fillna(0) if shy is not None else pd.Series(0.0, index=tqqq_ret.index)

                        # 리스크 오프 수익률
                        if "SHY" in bt_risk_off and shy is not None:
                            off_ret = shy_ret
                        else:
                            off_ret = pd.Series(0.0, index=tqqq_ret.index)

                        # 전략 수익률 = 시그널 * TQQQ + (1-시그널) * off
                        strat_ret = signal * tqqq_ret + (1 - signal) * off_ret

                        # 매매 수수료 반영 (포지션 변경 시)
                        fee_rate = bt_fee / 100.0
                        pos_change = signal.diff().abs().fillna(0)
                        strat_ret = strat_ret - pos_change * fee_rate

                        # SMA 이후 유효 구간만
                        valid_start = qqq_sma.dropna().index[0]
                        strat_ret = strat_ret.loc[valid_start:]
                        tqqq_ret_valid = tqqq_ret.loc[valid_start:]
                        qqq_ret_valid = qqq_ret.loc[valid_start:]
                        signal_valid = signal.loc[valid_start:]

                        # 누적 자산 곡선
                        equity_strat = (1 + strat_ret).cumprod() * 100
                        equity_tqqq = (1 + tqqq_ret_valid).cumprod() * 100
                        equity_qqq = (1 + qqq_ret_valid).cumprod() * 100

                        # 지표 계산 함수
                        def _calc_metrics(eq_series, label):
                            total_ret = (eq_series.iloc[-1] / eq_series.iloc[0] - 1) * 100
                            days = (eq_series.index[-1] - eq_series.index[0]).days
                            cagr = ((eq_series.iloc[-1] / eq_series.iloc[0]) ** (365.0 / max(days, 1)) - 1) * 100
                            peak = eq_series.cummax()
                            dd = (eq_series - peak) / peak * 100
                            mdd = dd.min()
                            daily_ret = eq_series.pct_change().dropna()
                            sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
                            return {
                                "전략": label,
                                "총 수익률(%)": round(total_ret, 1),
                                "CAGR(%)": round(cagr, 1),
                                "MDD(%)": round(mdd, 1),
                                "샤프비율": round(sharpe, 2),
                                "최종 자산": f"{eq_series.iloc[-1]:,.0f}",
                            }

                        # 거래 횟수
                        trade_count = int(pos_change.sum())
                        # 보유 비율
                        hold_ratio = signal_valid.mean() * 100

                        st.session_state["tqqq_bt_result"] = {
                            "equity_strat": equity_strat,
                            "equity_tqqq": equity_tqqq,
                            "equity_qqq": equity_qqq,
                            "signal": signal_valid,
                            "qqq": qqq.loc[valid_start:],
                            "qqq_sma": qqq_sma.loc[valid_start:],
                            "metrics": [
                                _calc_metrics(equity_strat, f"QQQ SMA{bt_sma_period}→TQQQ"),
                                _calc_metrics(equity_tqqq, "TQQQ 바이앤홀드"),
                                _calc_metrics(equity_qqq, "QQQ 바이앤홀드"),
                            ],
                            "trade_count": trade_count,
                            "hold_ratio": hold_ratio,
                            "sma_period": bt_sma_period,
                        }

        # 결과 표시
        bt_res = st.session_state.get("tqqq_bt_result")
        if bt_res:
            st.subheader("성과 비교")
            st.dataframe(pd.DataFrame(bt_res["metrics"]), use_container_width=True, hide_index=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("총 거래 횟수", f"{bt_res['trade_count']}회")
            mc2.metric("TQQQ 보유 비율", f"{bt_res['hold_ratio']:.1f}%")

            # 자산 곡선 차트
            st.subheader("자산 곡선 (시작 100)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bt_res["equity_strat"].index, y=bt_res["equity_strat"],
                name=f"QQQ SMA{bt_res['sma_period']}→TQQQ", line=dict(color="royalblue", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=bt_res["equity_tqqq"].index, y=bt_res["equity_tqqq"],
                name="TQQQ 바이앤홀드", line=dict(color="red", width=1, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=bt_res["equity_qqq"].index, y=bt_res["equity_qqq"],
                name="QQQ 바이앤홀드", line=dict(color="gray", width=1, dash="dash"),
            ))
            fig.update_layout(
                yaxis_type="log", yaxis_title="자산 (로그 스케일)",
                xaxis_title="날짜", height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # QQQ + SMA + 시그널 차트
            st.subheader("QQQ 가격 & SMA 시그널")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=bt_res["qqq"].index, y=bt_res["qqq"],
                name="QQQ", line=dict(color="black", width=1),
            ))
            fig2.add_trace(go.Scatter(
                x=bt_res["qqq_sma"].index, y=bt_res["qqq_sma"],
                name=f"SMA {bt_res['sma_period']}", line=dict(color="orange", width=1, dash="dash"),
            ))
            # 보유 구간 배경 표시
            sig = bt_res["signal"]
            in_pos = False
            start_date = None
            shapes = []
            for dt, val in sig.items():
                if val == 1 and not in_pos:
                    start_date = dt
                    in_pos = True
                elif val == 0 and in_pos:
                    shapes.append(dict(
                        type="rect", xref="x", yref="paper",
                        x0=start_date, x1=dt, y0=0, y1=1,
                        fillcolor="rgba(0,100,255,0.08)", line_width=0,
                    ))
                    in_pos = False
            if in_pos and start_date is not None:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=start_date, x1=sig.index[-1], y0=0, y1=1,
                    fillcolor="rgba(0,100,255,0.08)", line_width=0,
                ))
            fig2.update_layout(
                shapes=shapes,
                xaxis_title="날짜", yaxis_title="QQQ 가격",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("파란 배경 = TQQQ 보유 구간")

            # 연도별 수익률
            st.subheader("연도별 수익률 (%)")
            eq_s = bt_res["equity_strat"]
            eq_t = bt_res["equity_tqqq"]
            eq_q = bt_res["equity_qqq"]
            yearly = {}
            for year in sorted(set(eq_s.index.year)):
                mask = eq_s.index.year == year
                if mask.sum() < 2:
                    continue
                s_year = eq_s[mask]
                t_year = eq_t[mask]
                q_year = eq_q[mask]
                yearly[year] = {
                    "연도": year,
                    f"SMA{bt_res['sma_period']}→TQQQ": round((s_year.iloc[-1] / s_year.iloc[0] - 1) * 100, 1),
                    "TQQQ B&H": round((t_year.iloc[-1] / t_year.iloc[0] - 1) * 100, 1),
                    "QQQ B&H": round((q_year.iloc[-1] / q_year.iloc[0] - 1) * 100, 1),
                }
            if yearly:
                st.dataframe(pd.DataFrame(yearly.values()), use_container_width=True, hide_index=True)

            # Drawdown 차트
            st.subheader("Drawdown")
            peak_s = eq_s.cummax()
            dd_s = (eq_s - peak_s) / peak_s * 100
            peak_t = eq_t.cummax()
            dd_t = (eq_t - peak_t) / peak_t * 100
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=dd_s.index, y=dd_s,
                name=f"SMA{bt_res['sma_period']}→TQQQ", fill="tozeroy",
                line=dict(color="royalblue", width=1),
            ))
            fig3.add_trace(go.Scatter(
                x=dd_t.index, y=dd_t,
                name="TQQQ B&H",
                line=dict(color="red", width=1, dash="dot"),
            ))
            fig3.update_layout(
                yaxis_title="Drawdown (%)", xaxis_title="날짜", height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig3, use_container_width=True)


def render_kis_pension_mode():
    """KIS 연금저축 포트폴리오 모드 - 다중 전략 지원."""
    from kis_trader import KISTrader
    from strategy.laa import LAAStrategy

    # ── 사용 가능한 전략 목록 ──
    PEN_STRATEGIES = ["LAA", "듀얼모멘텀", "정적배분"]

    st.title("연금저축 포트폴리오")
    st.sidebar.header("연금저축 설정")

    kis_ak = _get_runtime_value(("KIS_PENSION_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(config.get("kis_pension_account_no", "") or _get_runtime_value(("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(config.get("kis_pension_prdt_cd", "") or _get_runtime_value(("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))

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

    # ── 포트폴리오 편집기 ──
    st.sidebar.subheader("포트폴리오")
    _saved_portfolio = config.get("pension_portfolio", [{"strategy": "LAA", "weight": 100}])
    if not _saved_portfolio:
        _saved_portfolio = [{"strategy": "LAA", "weight": 100}]
    _pen_port_df = pd.DataFrame(_saved_portfolio)
    # 컬럼 보정
    if "strategy" not in _pen_port_df.columns:
        _pen_port_df["strategy"] = "LAA"
    if "weight" not in _pen_port_df.columns:
        _pen_port_df["weight"] = 100

    _pen_port_edited = st.sidebar.data_editor(
        _pen_port_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="pen_portfolio_editor",
        column_config={
            "strategy": st.column_config.SelectboxColumn("전략", options=PEN_STRATEGIES, required=True),
            "weight": st.column_config.NumberColumn("비중(%)", min_value=0, max_value=100, step=5, required=True),
        },
    )

    _total_w = int(_pen_port_edited["weight"].sum()) if not _pen_port_edited.empty else 0
    if _total_w != 100:
        st.sidebar.warning(f"비중 합계: {_total_w}% (100%가 되어야 합니다)")
    else:
        st.sidebar.caption(f"비중 합계: {_total_w}%")

    # 현재 포트폴리오에 포함된 전략 목록
    _active_strategies = list(_pen_port_edited["strategy"].unique()) if not _pen_port_edited.empty else []

    # ── LAA 전략 설정 (포트폴리오에 LAA가 있을 때만) ──
    _kr_etf_map = {}
    kr_iwd = kr_gld = kr_ief = kr_qqq = kr_shy = ""
    if "LAA" in _active_strategies:
        with st.sidebar.expander("LAA 전략 설정", expanded=False):
            _kr_iwd_default = _code_only(config.get("kr_etf_laa_iwd", _get_runtime_value("KR_ETF_LAA_IWD", _get_runtime_value("KR_ETF_SPY", "360750"))))
            _kr_gld_default = _code_only(config.get("kr_etf_laa_gld", _get_runtime_value("KR_ETF_LAA_GLD", "132030")))
            _kr_ief_default = _code_only(config.get("kr_etf_laa_ief", _get_runtime_value("KR_ETF_LAA_IEF", _get_runtime_value("KR_ETF_AGG", "453540"))))
            _kr_qqq_default = _code_only(config.get("kr_etf_laa_qqq", _get_runtime_value("KR_ETF_LAA_QQQ", "133690")))
            _kr_shy_default = _code_only(config.get("kr_etf_laa_shy", _get_runtime_value("KR_ETF_LAA_SHY", "114470")))

            kr_iwd = st.text_input("IWD 대체 ETF", value=_kr_iwd_default, key="pen_laa_iwd")
            kr_gld = st.text_input("GLD 대체 ETF", value=_kr_gld_default, key="pen_laa_gld")
            kr_ief = st.text_input("IEF 대체 ETF", value=_kr_ief_default, key="pen_laa_ief")
            kr_qqq = st.text_input("QQQ 대체 ETF", value=_kr_qqq_default, key="pen_laa_qqq")
            kr_shy = st.text_input("SHY 대체 ETF", value=_kr_shy_default, key="pen_laa_shy")

        _kr_etf_map = {"IWD": str(kr_iwd), "GLD": str(kr_gld), "IEF": str(kr_ief), "QQQ": str(kr_qqq), "SHY": str(kr_shy)}

    # ── 듀얼모멘텀 전략 설정 ──
    _dm_settings = {}
    if "듀얼모멘텀" in _active_strategies:
        with st.sidebar.expander("듀얼모멘텀 설정", expanded=False):
            _dm_lookback = st.number_input("모멘텀 기간 (개월)", min_value=1, max_value=24,
                                           value=int(config.get("pen_dm_lookback", 12)), step=1, key="pen_dm_lookback")
            _dm_aggressive = st.text_input("공격 ETF", value=config.get("pen_dm_agg_etf", "133690"), key="pen_dm_agg_etf")
            _dm_defensive = st.text_input("방어 ETF", value=config.get("pen_dm_def_etf", "114470"), key="pen_dm_def_etf")
            _dm_settings = {"lookback": _dm_lookback, "aggressive_etf": _dm_aggressive, "defensive_etf": _dm_defensive}

    # ── 정적배분 전략 설정 ──
    _static_settings = {}
    if "정적배분" in _active_strategies:
        with st.sidebar.expander("정적배분 설정", expanded=False):
            st.caption("ETF별 고정 비중으로 월간 리밸런싱")
            _sa_etf1 = st.text_input("ETF 1 코드", value=config.get("pen_sa_etf1", "360750"), key="pen_sa_etf1")
            _sa_w1 = st.number_input("ETF 1 비중(%)", value=int(config.get("pen_sa_w1", 60)), min_value=0, max_value=100, step=5, key="pen_sa_w1")
            _sa_etf2 = st.text_input("ETF 2 코드", value=config.get("pen_sa_etf2", "453540"), key="pen_sa_etf2")
            _sa_w2 = st.number_input("ETF 2 비중(%)", value=int(config.get("pen_sa_w2", 40)), min_value=0, max_value=100, step=5, key="pen_sa_w2")
            _static_settings = {"etfs": [{"code": _sa_etf1, "weight": _sa_w1}, {"code": _sa_etf2, "weight": _sa_w2}]}

    # ── 설정 저장 ──
    if not IS_CLOUD and st.sidebar.button("연금저축 설정 저장", key="pen_save_cfg"):
        new_cfg = config.copy()
        new_cfg["kis_pension_account_no"] = str(kis_acct).strip()
        new_cfg["kis_pension_prdt_cd"] = str(kis_prdt).strip() or "01"
        # 포트폴리오 저장
        new_cfg["pension_portfolio"] = _pen_port_edited.to_dict("records")
        # LAA 설정
        if kr_iwd: new_cfg["kr_etf_laa_iwd"] = kr_iwd
        if kr_gld: new_cfg["kr_etf_laa_gld"] = kr_gld
        if kr_ief: new_cfg["kr_etf_laa_ief"] = kr_ief
        if kr_qqq: new_cfg["kr_etf_laa_qqq"] = kr_qqq
        if kr_shy: new_cfg["kr_etf_laa_shy"] = kr_shy
        # 듀얼모멘텀 설정
        if _dm_settings:
            new_cfg["pen_dm_lookback"] = _dm_settings.get("lookback", 12)
            new_cfg["pen_dm_agg_etf"] = _dm_settings.get("aggressive_etf", "133690")
            new_cfg["pen_dm_def_etf"] = _dm_settings.get("defensive_etf", "114470")
        # 정적배분 설정
        if _static_settings:
            new_cfg["pen_sa_etf1"] = _static_settings["etfs"][0]["code"] if _static_settings.get("etfs") else ""
            new_cfg["pen_sa_w1"] = _static_settings["etfs"][0]["weight"] if _static_settings.get("etfs") else 60
            new_cfg["pen_sa_etf2"] = _static_settings["etfs"][1]["code"] if len(_static_settings.get("etfs", [])) > 1 else ""
            new_cfg["pen_sa_w2"] = _static_settings["etfs"][1]["weight"] if len(_static_settings.get("etfs", [])) > 1 else 40
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

    # 토큰 캐싱 — F5 때만 새로 인증, 이후 rerun에서는 캐시된 토큰 재사용
    _pen_token_key = f"pen_token_{kis_acct}"
    _cached = st.session_state.get(_pen_token_key)
    if _cached and (_cached["expiry"] - time.time()) > 300:
        trader.access_token = _cached["token"]
        trader.token_expiry = _cached["expiry"]
    else:
        if not trader.auth():
            st.error("KIS 인증에 실패했습니다. API 키/계좌를 확인해 주세요.")
            return
        st.session_state[_pen_token_key] = {"token": trader.access_token, "expiry": trader.token_expiry}

    pen_bal_key = f"pension_balance_cache_{kis_acct}_{kis_prdt}"

    tab_p1, tab_p2, tab_p3, tab_p4, tab_p5, tab_p6 = st.tabs([
        "🚀 실시간 모니터링", "🧪 백테스트", "🛒 수동 주문",
        "📖 전략 가이드", "📋 주문방식", "💳 수수료/세금"
    ])

    def _fetch_overseas_chart(t, ticker, count=420):
        for ex in ("NAS", "NYS", "AMS"):
            df = t.get_overseas_daily_chart(ticker, exchange=ex, count=count)
            if df is not None and not df.empty:
                return df, ex
        return None, None

    # ══════════════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링
    # ══════════════════════════════════════════════════════════════
    with tab_p1:
        _strat_summary = " + ".join([f"{r['strategy']} {r['weight']}%" for _, r in _pen_port_edited.iterrows()]) if not _pen_port_edited.empty else "LAA 100%"
        st.header("포트폴리오 모니터링")
        st.caption(f"구성: {_strat_summary}")

        # 잔고 표시 — F5 새로고침 시 자동 조회, 이후 캐시 사용
        if pen_bal_key not in st.session_state:
            with st.spinner("연금저축 잔고를 조회하는 중..."):
                st.session_state[pen_bal_key] = trader.get_balance()

        rcol1, rcol2 = st.columns([1, 5])
        with rcol1:
            if st.button("잔고 새로고침", key="pen_refresh_balance"):
                with st.spinner("연금저축 잔고를 다시 조회하는 중..."):
                    st.session_state[pen_bal_key] = trader.get_balance()
                st.session_state.pop("pen_signal_result", None)
                st.session_state.pop("pen_signal_params", None)
                st.rerun()

        bal = st.session_state.get(pen_bal_key)
        if not bal:
            st.warning("잔고 조회에 실패했습니다. (응답 None — 네트워크 또는 인증 오류)")
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt}")
        elif bal.get("error"):
            st.error(f"잔고 조회 API 오류: [{bal.get('msg_cd', '')}] {bal.get('msg1', '알 수 없는 오류')}")
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt} / rt_cd: {bal.get('rt_cd', '')}")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []
            total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
            stock_eval = total_eval - cash

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("예수금", f"{cash:,.0f} KRW")
            m2.metric("주식 평가", f"{stock_eval:,.0f} KRW")
            m3.metric("총 평가", f"{total_eval:,.0f} KRW")
            m4.metric("보유 종목 수", f"{len(holdings)}")

            if holdings:
                df_h = pd.DataFrame(holdings)
                cols = [c for c in ["code", "name", "qty", "avg_price", "cur_price", "eval_amt", "pnl_rate"] if c in df_h.columns]
                st.dataframe(df_h[cols], use_container_width=True, hide_index=True)

        st.divider()

        # LAA 시그널 자동 계산 (버튼 없이 항상 표시)
        pen_sig_params = {
            "acct": str(kis_acct),
            "prdt": str(kis_prdt),
            "kr_iwd": str(kr_iwd),
            "kr_gld": str(kr_gld),
            "kr_ief": str(kr_ief),
            "kr_qqq": str(kr_qqq),
            "kr_shy": str(kr_shy),
        }

        def _compute_pen_signal_result():
            tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
            price_data = {}
            exchange_map = {}
            for ticker in tickers:
                df_t, ex = _fetch_overseas_chart(trader, ticker, 420)
                if df_t is None:
                    return {"error": "해외 지수 데이터 조회에 실패했습니다."}
                price_data[ticker] = df_t
                exchange_map[ticker] = ex

            strategy = LAAStrategy(settings={"kr_etf_map": _kr_etf_map})
            signal = strategy.analyze(price_data)
            if not signal:
                return {"error": "LAA 분석에 실패했습니다."}

            bal_local = st.session_state.get(pen_bal_key) or trader.get_balance() or {"cash": 0.0, "holdings": []}
            cash_local = float(bal_local.get("cash", 0.0))
            holdings_local = bal_local.get("holdings", []) or []

            current_vals = {}
            for h in holdings_local:
                code = str(h.get("code", ""))
                current_vals[code] = current_vals.get(code, 0.0) + float(h.get("eval_amt", 0.0))

            total_eval = float(bal_local.get("total_eval", 0.0)) or (cash_local + sum(current_vals.values()))
            total_eval = max(total_eval, 1.0)

            rows = []
            max_gap = 0.0
            for code, target_w in signal["target_weights_kr"].items():
                cur_v = float(current_vals.get(str(code), 0.0))
                cur_w = cur_v / total_eval
                gap = float(target_w) - float(cur_w)
                max_gap = max(max_gap, abs(gap))
                rows.append({
                    "ETF": _fmt_etf_code_name(code),
                    "ETF 코드": str(code),
                    "목표 비중(%)": round(target_w * 100.0, 2),
                    "현재 비중(%)": round(cur_w * 100.0, 2),
                    "비중 차이(%p)": round(gap * 100.0, 2),
                    "현재 평가(KRW)": f"{cur_v:,.0f}",
                    "목표 평가(KRW)": f"{total_eval * float(target_w):,.0f}",
                })

            action = "HOLD" if max_gap <= 0.03 else "REBALANCE"
            return {
                "signal": signal,
                "action": action,
                "exchange_map": exchange_map,
                "alloc_df": pd.DataFrame(rows),
                "price_data": price_data,
                "balance": bal_local,
            }

        if st.session_state.get("pen_signal_result") is None or st.session_state.get("pen_signal_params") != pen_sig_params:
            with st.spinner("LAA 시그널을 자동 계산하는 중입니다..."):
                st.session_state["pen_signal_result"] = _compute_pen_signal_result()
                st.session_state["pen_signal_params"] = pen_sig_params
                _pen_res = st.session_state["pen_signal_result"]
                if isinstance(_pen_res, dict) and _pen_res.get("balance"):
                    st.session_state[pen_bal_key] = _pen_res["balance"]

        res = st.session_state.get("pen_signal_result")
        if res:
            if res.get("error"):
                st.error(res["error"])
            else:
                sig = res["signal"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("리스크 상태", "공격 (Risk-On)" if sig.get("risk_on") else "방어 (Risk-Off)")
                c2.metric("리스크 자산", sig["selected_risk_asset"])
                c3.metric("국내 ETF", _fmt_etf_code_name(sig["selected_risk_kr_code"]))
                c4.metric("권장 동작", res["action"])
                st.info(sig.get("reason", ""))

                if "exchange_map" in res:
                    st.caption("조회 거래소: " + ", ".join([f"{k}={v}" for k, v in res["exchange_map"].items()]))

                if "alloc_df" in res and isinstance(res["alloc_df"], pd.DataFrame) and not res["alloc_df"].empty:
                    st.subheader("목표 배분 vs 현재 보유")
                    st.dataframe(res["alloc_df"], use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # Tab 2: 백테스트
    # ══════════════════════════════════════════════════════════════
    with tab_p2:
        st.header("LAA 백테스트")
        st.caption("해외 지수(SPY/IWD/GLD/IEF/QQQ/SHY) 기반 월간 리밸런싱 시뮬레이션")

        pen_bt_cap = st.number_input("초기 자본 (KRW)", value=10_000_000, step=1_000_000, key="pen_bt_cap")
        pen_bt_fee = st.number_input("수수료 (%)", value=0.02, format="%.2f", key="pen_bt_fee") / 100.0

        if st.button("백테스트 실행", key="pen_bt_run", type="primary"):
            with st.spinner("LAA 백테스트 실행 중... (해외 데이터 조회)"):
                tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
                price_data = {}
                for ticker in tickers:
                    df_t, _ex = _fetch_overseas_chart(trader, ticker, 3000)
                    if df_t is None:
                        st.error(f"{ticker} 데이터 조회 실패")
                        price_data = None
                        break
                    price_data[ticker] = df_t

                if price_data:
                    strategy = LAAStrategy(settings={"kr_etf_map": _kr_etf_map})
                    bt_result = strategy.run_backtest(price_data, initial_balance=float(pen_bt_cap), fee=float(pen_bt_fee))
                    if bt_result:
                        st.session_state["pen_bt_result"] = bt_result
                    else:
                        st.error("백테스트 실행 실패 (데이터 부족)")

        bt_res = st.session_state.get("pen_bt_result")
        if bt_res:
            metrics = bt_res["metrics"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("총 수익률", f"{metrics['total_return']:.2f}%")
            mc2.metric("CAGR", f"{metrics['cagr']:.2f}%")
            mc3.metric("MDD", f"{metrics['mdd']:.2f}%")
            mc4.metric("Sharpe", f"{metrics['sharpe']:.2f}")
            mc5.metric("최종 자산", f"{metrics['final_equity']:,.0f}")

            # Equity 차트
            eq_df = bt_res["equity_df"]
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df.index, y=eq_df["equity"],
                name="포트폴리오", line=dict(color="royalblue"),
            ))
            fig.update_layout(title="LAA 백테스트 Equity Curve", xaxis_title="날짜", yaxis_title="자산 (KRW)", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # 연도별 수익률
            if "equity" in eq_df.columns:
                yearly = eq_df["equity"].resample("YE").last()
                if len(yearly) > 1:
                    yr_ret = yearly.pct_change().dropna() * 100
                    yr_data = [{"연도": str(d.year), "수익률(%)": f"{r:.2f}"} for d, r in yr_ret.items()]
                    st.subheader("연도별 수익률")
                    st.dataframe(pd.DataFrame(yr_data), use_container_width=True, hide_index=True)

            # 배분 이력
            if "allocations" in bt_res:
                alloc_df = bt_res["allocations"]
                if not alloc_df.empty:
                    st.subheader("월별 자산 배분 이력")
                    st.dataframe(alloc_df.tail(24), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # Tab 3: 수동 주문
    # ══════════════════════════════════════════════════════════════
    with tab_p3:
        st.header("수동 주문")

        bal = st.session_state.get(pen_bal_key)
        if not bal:
            st.warning("잔고를 먼저 조회해 주세요.")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []
            st.info(f"예수금: {cash:,.0f} KRW | 보유 종목: {len(holdings)}개")

            # 매매 대상 ETF 선택
            all_etf_codes = list(set([kr_iwd, kr_gld, kr_ief, kr_qqq, kr_shy]))
            etf_options = { _fmt_etf_code_name(c): c for c in all_etf_codes }
            selected_pen_label = st.selectbox("매매 ETF 선택", list(etf_options.keys()), key="pen_trade_etf_label")
            selected_pen_etf = etf_options[selected_pen_label]

            pen_holding = next((h for h in holdings if str(h.get("code", "")) == str(selected_pen_etf)), None)
            pen_qty = int(pen_holding.get("qty", 0)) if pen_holding else 0
            st.caption(f"{_fmt_etf_code_name(selected_pen_etf)} 보유: {pen_qty}주")

            buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

            with buy_tab:
                pen_buy_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)"], key="pen_buy_method", horizontal=True)
                pen_buy_amt = st.number_input("매수 금액 (KRW)", min_value=0, value=0, step=10000, key="pen_buy_amt")

                cur_price = trader.get_current_price(str(selected_pen_etf))
                if cur_price and cur_price > 0:
                    st.caption(f"현재가: {cur_price:,.0f}원 | 예상 수량: {int(pen_buy_amt / cur_price) if pen_buy_amt > 0 else 0}주")

                pen_buy_price = 0
                if pen_buy_method == "지정가":
                    pen_buy_price = st.number_input("매수 지정가", min_value=0, value=int(cur_price or 0), step=50, key="pen_buy_price")

                if st.button("매수 실행", key="pen_exec_buy", type="primary", disabled=IS_CLOUD):
                    if pen_buy_amt <= 0:
                        st.error("매수 금액을 입력해 주세요.")
                    else:
                        with st.spinner("매수 주문 실행 중..."):
                            if pen_buy_method == "동시호가 (장마감)":
                                result = trader.smart_buy_krw_closing(str(selected_pen_etf), float(pen_buy_amt))
                            elif pen_buy_method == "지정가" and pen_buy_price > 0:
                                qty = int(pen_buy_amt / pen_buy_price)
                                result = trader.send_order("BUY", str(selected_pen_etf), qty, price=pen_buy_price, ord_dvsn="00") if qty > 0 else None
                            else:
                                result = trader.smart_buy_krw(str(selected_pen_etf), float(pen_buy_amt))
                            if result and (isinstance(result, dict) and result.get("success")):
                                st.success(f"매수 완료: {result}")
                                st.session_state[pen_bal_key] = trader.get_balance()
                            else:
                                st.error(f"매수 실패: {result}")

            with sell_tab:
                pen_sell_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)"], key="pen_sell_method", horizontal=True)
                pen_sell_qty = st.number_input("매도 수량", min_value=0, max_value=max(pen_qty, 1), value=pen_qty, step=1, key="pen_sell_qty")
                pen_sell_all = st.checkbox("전량 매도", value=True, key="pen_sell_all")

                pen_sell_price = 0
                if pen_sell_method == "지정가":
                    pen_sell_price = st.number_input("매도 지정가", min_value=0, value=int(cur_price or 0), step=50, key="pen_sell_price")

                if st.button("매도 실행", key="pen_exec_sell", type="primary", disabled=IS_CLOUD):
                    _sq = pen_qty if pen_sell_all else pen_sell_qty
                    if _sq <= 0:
                        st.error("매도할 수량이 없습니다.")
                    else:
                        with st.spinner("매도 주문 실행 중..."):
                            if pen_sell_method == "동시호가 (장마감)":
                                result = trader.smart_sell_all_closing(str(selected_pen_etf)) if pen_sell_all else trader.smart_sell_qty_closing(str(selected_pen_etf), _sq)
                            elif pen_sell_method == "지정가" and pen_sell_price > 0:
                                result = trader.send_order("SELL", str(selected_pen_etf), _sq, price=pen_sell_price, ord_dvsn="00")
                            else:
                                result = trader.smart_sell_all(str(selected_pen_etf)) if pen_sell_all else trader.smart_sell_qty(str(selected_pen_etf), _sq)
                            if result and (isinstance(result, dict) and result.get("success")):
                                st.success(f"매도 완료: {result}")
                                st.session_state[pen_bal_key] = trader.get_balance()
                            else:
                                st.error(f"매도 실패: {result}")

    # ══════════════════════════════════════════════════════════════
    # Tab 4: 전략 가이드
    # ══════════════════════════════════════════════════════════════
    with tab_p4:
        st.header("LAA (Lethargic Asset Allocation) 전략 가이드")

        st.subheader("1. 전략 개요")
        st.markdown("""
**LAA**는 Keller & Keuning이 제안한 게으른 자산배분 전략입니다.

- **코어 자산 (75%)**: IWD(미국 가치주), GLD(금), IEF(미국 중기채) 각 25%
- **리스크 자산 (25%)**: SPY가 200일 이동평균선 위 → QQQ, 아래 → SHY(단기채)
- **리밸런싱**: 월 1회 (월말 기준)
""")

        st.subheader("2. 의사결정 흐름")
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

        st.subheader("3. 국내 ETF 매핑")
        st.dataframe(pd.DataFrame([
            {"미국 티커": "IWD", "국내 ETF": "TIGER 미국S&P500 (360750)", "역할": "코어 - 미국 가치주"},
            {"미국 티커": "GLD", "국내 ETF": "KODEX Gold선물(H) (132030)", "역할": "코어 - 금"},
            {"미국 티커": "IEF", "국내 ETF": "TIGER 미국채10년선물 (453540)", "역할": "코어 - 중기채"},
            {"미국 티커": "QQQ", "국내 ETF": "TIGER 미국나스닥100 (133690)", "역할": "리스크 공격"},
            {"미국 티커": "SHY", "국내 ETF": "KODEX 국고채3년 (114470)", "역할": "리스크 방어"},
        ]), use_container_width=True, hide_index=True)
        st.caption("연금저축 계좌에서 해외 ETF 직접 매매 불가 → 국내 ETF로 대체 실행")

        st.subheader("4. 기대 성과")
        st.markdown("""
- **CAGR**: 약 7~10% (백테스트 기준)
- **MDD**: 약 -10~-15% (60/40 대비 개선)
- **리밸런싱 빈도**: 월 1회 → 매매 수수료 부담 최소
- **과세이연**: 연금저축 내 매매차익 비과세 → 복리 효과 극대화
""")

    # ══════════════════════════════════════════════════════════════
    # Tab 5: 주문방식
    # ══════════════════════════════════════════════════════════════
    with tab_p5:
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
1. 매월 25~31일 평일 KST 15:20 실행 (`TRADING_MODE=kis_pension`)
2. 해외 지수(SPY/IWD/GLD/IEF/QQQ/SHY) 일봉 조회
3. SPY vs 200일선 → 리스크 자산 결정 (QQQ or SHY)
4. 목표 배분 vs 현재 보유 비교 → 리밸런싱 필요 여부 판단
5. 매도 → `smart_sell_all_closing()` (동시호가+시간외)
6. 매수 → `smart_buy_krw_closing()` (동시호가+시간외)
""")

    # ══════════════════════════════════════════════════════════════
    # Tab 6: 수수료/세금
    # ══════════════════════════════════════════════════════════════
    with tab_p6:
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



def main():
    # --- Mode Select (Sidebar Top) ---
    _mode_map = {
        "코인": "COIN",
        "골드": "GOLD",
        "ISA": "ISA",
        "연금저축": "PENSION",
    }
    _mode_keys = list(_mode_map.keys())
    _mode_reverse = {v: k for k, v in _mode_map.items()}

    # query_params에서 저장된 모드 복원
    _qp = st.query_params
    _saved_mode = _qp.get("mode", "")
    _default_idx = 0
    if _saved_mode in _mode_reverse:
        _restored_label = _mode_reverse[_saved_mode]
        if _restored_label in _mode_keys:
            _default_idx = _mode_keys.index(_restored_label)

    _mode_label = st.sidebar.selectbox(
        "거래 모드",
        _mode_keys,
        index=_default_idx,
        key="trading_mode_label",
        label_visibility="collapsed",
    )
    trading_mode = _mode_map[_mode_label]

    # 사용자가 모드를 변경했을 때만 query_params 갱신 (최초 로드 시 rerun 방지)
    _prev_mode = st.session_state.get("_last_trading_mode", "")
    if _prev_mode and _prev_mode != trading_mode:
        st.query_params["mode"] = trading_mode
    elif not _saved_mode:
        # 최초 진입 시 URL에 모드 기록 (rerun 없이)
        st.query_params["mode"] = trading_mode
    st.session_state["_last_trading_mode"] = trading_mode

    if trading_mode == "GOLD":
        render_gold_mode()
        return
    if trading_mode == "ISA":
        render_kis_isa_mode()
        return
    if trading_mode == "PENSION":
        render_kis_pension_mode()
        return
    # === 코인 모드 (기존 코드) ===
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

    # 텔레그램 알림 설정
    render_telegram_sidebar(prefix="coin")

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
        "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio",
        "row_id", "parent_id",
    ]
    for c in editor_columns:
        if c not in df_portfolio.columns:
            df_portfolio[c] = "" if c in {"strategy_no", "row_type", "coin", "strategy", "interval", "aux_seed_mode", "row_id", "parent_id"} else 0
    df_portfolio = df_portfolio[editor_columns]
    df_portfolio = _apply_strategy_no(df_portfolio)

    interval_options = list(INTERVAL_MAP.keys())
    strategy_options = ["SMA", "Donchian", STRATEGY_AUX]

    editor_state_key = "portfolio_editor_df"
    source_df = st.session_state.get(editor_state_key)
    if not isinstance(source_df, pd.DataFrame) or list(source_df.columns) != editor_columns:
        source_df = df_portfolio.copy()
    source_df = _apply_strategy_no(source_df)

    _aux_cols = ["aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio"]
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
                edited_portfolio[c] = "" if c in {"strategy_no", "row_type", "coin", "strategy", "interval", "aux_seed_mode", "row_id", "parent_id"} else 0
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
            _aux_display_cols = ["strategy_no", "coin", "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio"]
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
                    "aux_ma_short": st.column_config.NumberColumn("단기MA", min_value=2, max_value=100, step=1),
                    "aux_ma_long": st.column_config.NumberColumn("장기MA", min_value=3, max_value=500, step=1),
                    "aux_threshold": st.column_config.NumberColumn("임계(%)", min_value=-50.0, max_value=0.0, step=0.5, format="%.1f"),
                    "aux_tp1": st.column_config.NumberColumn("TP1(%)", min_value=0.1, max_value=50.0, step=0.5, format="%.1f"),
                    "aux_tp2": st.column_config.NumberColumn("TP2(%)", min_value=0.1, max_value=100.0, step=0.5, format="%.1f"),
                    "aux_split": st.column_config.NumberColumn("분할수", min_value=1, max_value=10, step=1),
                    "aux_seed_mode": st.column_config.SelectboxColumn("시드모드", options=["균등", "피라미딩"]),
                    "aux_pyramid_ratio": st.column_config.NumberColumn("피라미딩비율", min_value=1.0, max_value=5.0, step=0.1, format="%.1f"),
                },
            )
            # 변경사항을 edited_portfolio에 반영
            for _ri in range(len(_aux_edit)):
                _orig = int(_aux_df.iloc[_ri]["_orig_idx"])
                for _col in _aux_display_cols:
                    if _col in {"strategy_no", "coin"}:
                        continue
                    edited_portfolio.at[_orig, _col] = _aux_edit.iloc[_ri][_col]
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
            aux_portfolio_list.append({
                'coin': coin_val,
                'interval': api_key,
                'parent_id': str(r.get('parent_id', '')),
                'aux_ma_short': int(r.get('aux_ma_short', 5) or 5),
                'aux_ma_long': int(r.get('aux_ma_long', 20) or 20),
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
        save_col1, save_col2 = st.sidebar.columns(2)

        if save_col1.button("Save"):
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

        if save_col2.button("Load"):
            if os.path.exists(PORTFOLIO_JSON):
                try:
                    with open(PORTFOLIO_JSON, "r", encoding="utf-8") as f:
                        imported = json.load(f)

                    if isinstance(imported, dict):
                        imported_port = imported.get("portfolio", [])
                        imported_aux = imported.get("aux_portfolio", [])
                        new_config = {
                            "portfolio": imported_port,
                            "aux_portfolio": imported_aux,
                            "start_date": str(imported.get("start_date", start_date)),
                            "initial_cap": int(imported.get("initial_cap", initial_cap)),
                        }
                        save_config(new_config)
                        st.sidebar.success(f"Loaded: main={len(imported_port)}, aux={len(imported_aux)}")
                        st.rerun()
                    elif isinstance(imported, list):
                        new_config = {
                            "portfolio": imported,
                            "aux_portfolio": [],
                            "start_date": str(start_date),
                            "initial_cap": initial_cap,
                        }
                        save_config(new_config)
                        st.sidebar.success(f"Loaded: main={len(imported)}")
                        st.rerun()
                    else:
                        st.sidebar.error("Invalid portfolio.json format")
                except json.JSONDecodeError:
                    st.sidebar.error("JSON parse error")
            else:
                st.sidebar.warning("portfolio.json not found")

    # --- data_manager Import ---
    from data_manager import MarketDataWorker

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
    from data_manager import CoinTradingWorker

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
    tab1, tab5, tab3, tab4 = st.tabs(["🚀 실시간 포트폴리오", "🛒 수동 주문", "📜 거래 내역", "📊 백테스트"])

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

                krw_bal = _ttl_cache("krw_bal_t1", lambda: trader.get_balance("KRW") or 0, ttl=10)

                def _fetch_all_prices():
                    """모든 코인 가격을 한번에 가져옴"""
                    try:
                        result = pyupbit.get_current_price(unique_tickers)
                        if isinstance(result, dict):
                            return result
                        elif isinstance(result, (int, float)) and len(unique_tickers) == 1:
                            return {unique_tickers[0]: result}
                    except:
                        pass
                    return {t: (pyupbit.get_current_price(t) or 0) for t in unique_tickers}

                all_prices = _ttl_cache("prices_t1", _fetch_all_prices, ttl=5)

                def _fetch_all_balances():
                    """모든 코인 잔고를 1회 API 호출로 가져옴"""
                    if hasattr(trader, 'get_all_balances'):
                        raw = trader.get_all_balances()
                        return {c: raw.get(c, 0) for c in unique_coins}
                    # 폴백: 개별 호출
                    return {c: (trader.get_balance(c) or 0) for c in unique_coins}

                all_balances = _ttl_cache("balances_t1", _fetch_all_balances, ttl=10)

                # --- Total Summary Container ---
                st.subheader("🏁 포트폴리오 요약")
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

                    cash_assets = [a for a in asset_states if a['status'] == 'CASH']
                    buy_signal_assets = [a for a in asset_states if a['signal'] == 'BUY']

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("보유 현금 (KRW)", f"{krw_balance:,.0f}")
                    rc2.metric("CASH 자산", f"{len(cash_assets)} / {len(asset_states)}")
                    rc3.metric("BUY 시그널", f"{len(buy_signal_assets)} / {len(asset_states)}")

                    rebal_data = []
                    for a in asset_states:
                        action = ""
                        if a['status'] == 'CASH' and a['signal'] == 'BUY':
                            action = "BUY"
                        elif a['status'] == 'CASH' and a['signal'] != 'BUY':
                            action = "대기 (시그널 없음)"
                        elif a['status'] == 'HOLD':
                            action = "보유 중"
                        rebal_data.append({
                            "종목": a['ticker'],
                            "전략": f"{a['strategy']} {a['param']}",
                            "비중": f"{a['weight']}%",
                            "시간봉": a['interval'],
                            "상태": a['status'],
                            "시그널": a['signal'],
                            "현재가치(KRW)": f"{a['coin_val']:,.0f}",
                            "목표(KRW)": f"{a['target_krw']:,.0f}",
                            "액션": action,
                        })
                    st.dataframe(pd.DataFrame(rebal_data), use_container_width=True, hide_index=True)

                    buyable = [a for a in asset_states if a['status'] == 'CASH' and a['signal'] == 'BUY']
                    if not buyable:
                        if len(cash_assets) == 0:
                            st.success("모든 자산이 이미 보유 중입니다.")
                        else:
                            st.info(f"현금 자산 {len(cash_assets)}개가 있지만 BUY 시그널이 없습니다. 시그널 발생 시 매수 가능합니다.")
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
                                "시간봉": a['interval'], "현재가": f"{a['price']:,.0f}",
                                "_ticker": a['ticker'], "_krw": alloc_krw, "_interval": a['interval'],
                            })
                        plan_df = pd.DataFrame(buy_plan)
                        st.dataframe(plan_df[["종목", "비중", "배분 금액(KRW)", "시간봉", "현재가"]], use_container_width=True, hide_index=True)
                        st.caption(f"총 배분 금액: {sum(p['_krw'] for p in buy_plan):,.0f} KRW / 보유 현금: {krw_balance:,.0f} KRW")

                        if st.button("🚀 리밸런싱 매수 실행", key="btn_rebalance_exec", type="primary"):
                            rebal_results = []
                            rebal_progress = st.progress(0)
                            rebal_log = st.empty()
                            for pi, plan in enumerate(buy_plan):
                                p_ticker = plan['_ticker']
                                p_krw = plan['_krw']
                                p_interval = plan['_interval']
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
                                        lambda t=p_ticker, iv=p_interval, pp=p_param: pyupbit.get_ohlcv(t, interval=iv, count=max(60 + pp, 200)),
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
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                                    st.plotly_chart(fig_comp, use_container_width=True)

                                    # 연도별 성과 테이블
                                    if 'yearly_stats' in perf:
                                        st.caption("📅 연도별 성과")
                                        ys = perf['yearly_stats'].copy()
                                        ys.index.name = "연도"
                                        st.dataframe(ys.style.format("{:.2f}"), use_container_width=True)
                            
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
                            import numpy as np

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
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                            )
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
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                            )
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
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
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
            from data_manager import CoinTradingWorker

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

                # ═══ 메인 레이아웃: 호가창(좌) + 주문(중) + 30분봉(우) ═══
                ob_col, order_col, chart_col = st.columns([2, 3, 4])

                # ── 좌: 호가창 (HTML 렌더링 — 더블클릭 편집 없음) ──
                with ob_col:
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

                                # ── 호가 선택 → 주문가 반영 (on_change: 사용자 선택시만 반영) ──
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

                # ── 우: 30분봉 차트 ──
                with chart_col:
                    st.markdown("**30분봉 차트**")
                    df_30m = _ttl_cache(f"m30_{mt_ticker}", lambda: pyupbit.get_ohlcv(mt_ticker, interval="minute30", count=48), ttl=30)
                    if df_30m is not None and len(df_30m) > 0:
                        fig_30m = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                row_heights=[0.8, 0.2], vertical_spacing=0.02)
                        fig_30m.add_trace(go.Candlestick(
                            x=df_30m.index, open=df_30m['open'], high=df_30m['high'],
                            low=df_30m['low'], close=df_30m['close'], name='30분봉',
                            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                        ), row=1, col=1)
                        # 이동평균선
                        ma5 = df_30m['close'].rolling(5).mean()
                        ma20 = df_30m['close'].rolling(20).mean()
                        fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma5, name='MA5',
                                                     line=dict(color='#FF9800', width=1)), row=1, col=1)
                        fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma20, name='MA20',
                                                     line=dict(color='#2196F3', width=1)), row=1, col=1)
                        # 거래량
                        colors_vol = ['#26a69a' if c >= o else '#ef5350'
                                      for c, o in zip(df_30m['close'], df_30m['open'])]
                        fig_30m.add_trace(go.Bar(x=df_30m.index, y=df_30m['volume'],
                                                 marker_color=colors_vol, name='거래량', showlegend=False),
                                          row=2, col=1)
                        fig_30m.update_layout(
                            height=520, margin=dict(l=0, r=0, t=10, b=30),
                            xaxis_rangeslider_visible=False, showlegend=True,
                            legend=dict(orientation="h", y=1.02, x=0),
                            xaxis2=dict(showticklabels=True, tickformat='%H:%M', tickangle=-45),
                            yaxis=dict(title="", side="right"),
                            yaxis2=dict(title="", side="right"),
                        )
                        st.plotly_chart(fig_30m, use_container_width=True, key=f"chart30m_{mt_ticker}")
                    else:
                        st.info("차트 데이터 로딩 중...")

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
                            df_ohlcv = pyupbit.get_ohlcv(sa_ticker, interval=api_interval, count=200)

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
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
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
                    df_bt = pyupbit.get_ohlcv(bt_ticker, interval=bt_interval, to=to_str, count=fetch_count)
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
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                        fig.update_yaxes(title_text="가격 (KRW)", row=1, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="자산 (KRW)", row=1, col=1, secondary_y=True)
                        fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
                        st.plotly_chart(fig, use_container_width=True)

                        if 'yearly_stats' in res:
                            st.subheader("연도별 성과")
                            st.dataframe(res['yearly_stats'].style.format("{:.2f}%"))

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
                            st.success(f"【{_lbl}】 최적: 매수 **{int(_best['Buy Period'])}**, 매도 **{int(_best['Sell Period'])}** → 수익률 {_best['Total Return (%)']:.1f}%, Calmar {_best['Calmar']:.2f}, Robustness {_best['Robustness']:.2f}")
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
                    sort_col = "Calmar" if _s_optuna else "Total Return (%)"
                    opt_df = opt_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
                    opt_df = _add_robustness(opt_df)
                    opt_df.index = opt_df.index + 1
                    opt_df.index.name = "순위"
                    best_row = opt_df.iloc[0]
                    if _s_strategy == "돈키안 전략":
                        st.success(f"최적: 매수 **{int(best_row['Buy Period'])}**, 매도 **{int(best_row['Sell Period'])}** → 수익률 {best_row['Total Return (%)']:.1f}%, Calmar {best_row['Calmar']:.2f}, Robustness {best_row.get('Robustness', 0):.2f}")
                    else:
                        st.success(f"최적: SMA **{int(best_row['SMA Period'])}** → 수익률 {best_row['Total Return (%)']:.1f}%, Calmar {best_row['Calmar']:.2f}, Robustness {best_row.get('Robustness', 0):.2f}")
                    _grad_cols = [c for c in ['Total Return (%)', 'Calmar', 'Sharpe', 'Robustness'] if c in opt_df.columns]
                    st.dataframe(
                        opt_df.style.background_gradient(cmap='RdYlGn', subset=_grad_cols)
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
                                    c1.metric("이웃 평균 수익률", f"{sub_df['Total Return (%)'].mean():.1f}%")
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
                                    c1.metric("이웃 평균 수익률", f"{sub_df['Total Return (%)'].mean():.1f}%")
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
                apc1, apc2 = st.columns(2)
                aux_ma_short = apc1.number_input("단기 MA", 3, 100, 5, key="aux_bt_ma_short")
                aux_ma_long = apc2.number_input("중기 MA", 5, 300, 20, key="aux_bt_ma_long")
                if aux_ma_long <= aux_ma_short:
                    aux_ma_long = aux_ma_short + 1

                aux_threshold = st.slider("과매도 임계값(이격도 %)", -30.0, -0.5, -5.0, 0.5, key="aux_bt_threshold")

                atc1, atc2 = st.columns(2)
                aux_tp1 = atc1.number_input("TP1 (%)", 1.0, 30.0, 3.0, 0.5, key="aux_bt_tp1")
                aux_tp2 = atc2.number_input("TP2 (%)", 1.0, 50.0, 10.0, 0.5, key="aux_bt_tp2")
                if aux_tp2 < aux_tp1:
                    aux_tp2 = aux_tp1

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

            run_aux = st.button("보조 전략 백테스트 실행", type="primary", key="run_aux_bt")

            if run_aux:
                with st.spinner("보조 전략 백테스트 실행 중..."):
                    api_iv = INTERVAL_MAP.get(aux_interval_label, "day")
                    api_main_iv = INTERVAL_MAP.get(aux_main_interval_label, api_iv)

                    days = max((datetime.now().date() - aux_start).days, 30)
                    cpd = CANDLES_PER_DAY.get(api_iv, 1)
                    main_cpd = CANDLES_PER_DAY.get(api_main_iv, 1)

                    base_warmup = max(int(aux_ma_long), int(aux_main_buy), int(aux_main_sell), 30)
                    aux_count = min(max(days * cpd + base_warmup + 300, 500), 12000)
                    main_count = min(max(days * main_cpd + base_warmup + 300, 500), 12000)

                    df_aux = data_cache.get_ohlcv_cached(aux_ticker, interval=api_iv, count=aux_count)
                    if df_aux is None or len(df_aux) < max(50, int(aux_ma_long) + 5):
                        st.error("보조 실행 캔들 데이터가 부족합니다.")
                    else:
                        df_main_aux = None
                        if api_main_iv != api_iv:
                            df_main_aux = data_cache.get_ohlcv_cached(aux_ticker, interval=api_main_iv, count=main_count)
                            if df_main_aux is None or len(df_main_aux) < max(50, int(aux_main_buy) + 5):
                                st.error("메인 신호 캔들 데이터가 부족합니다.")
                                df_main_aux = None

                        if api_main_iv == api_iv or df_main_aux is not None:
                            res_aux = backtest_engine.run_aux_backtest(
                                df_aux,
                                main_strategy=aux_main_strat,
                                main_buy_p=int(aux_main_buy),
                                main_sell_p=int(aux_main_sell),
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
                    st.caption(
                        f"시드={_seed_mode_out}{_seed_note}"
                        + f" | split={int(aux_split)}"
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
                        st.plotly_chart(fig_aux, use_container_width=True)

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


if __name__ == "__main__":
    main()
