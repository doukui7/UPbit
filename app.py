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
load_dotenv()

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
    """금(Gold) 현물 거래 모드 - 키움증권 KRX 금시장"""
    from kiwoom_gold import KiwoomGoldTrader

    st.title("🥇 Gold Trading System (키움증권)")

    # --- Sidebar: Gold 설정 ---
    st.sidebar.header("Gold 설정")

    # Kiwoom API Keys
    try:
        kiwoom_ak = st.secrets.get("KIWOOM_APP_KEY", "")
        kiwoom_sk = st.secrets.get("KIWOOM_SECRET_KEY", "")
        kiwoom_account = st.secrets.get("KIWOOM_ACCOUNT", "")
    except Exception:
        kiwoom_ak = os.getenv("KIWOOM_APP_KEY", "")
        kiwoom_sk = os.getenv("KIWOOM_SECRET_KEY", "")
        kiwoom_account = os.getenv("KIWOOM_ACCOUNT", "")

    if IS_CLOUD:
        st.sidebar.info("📱 조회 전용 모드 (Cloud)")
    else:
        with st.sidebar.expander("키움 API Keys", expanded=False):
            kiwoom_ak = st.text_input("App Key", value=kiwoom_ak, type="password", key="kiwoom_ak")
            kiwoom_sk = st.text_input("Secret Key", value=kiwoom_sk, type="password", key="kiwoom_sk")
            kiwoom_account = st.text_input("계좌번호", value=kiwoom_account, key="kiwoom_acc")

    # Gold 종목 설정
    GOLD_PRODUCTS = {
        "금 1g (KRX)": "401000",
        "금 미니 100g": "401001",
    }
    st.sidebar.subheader("금 종목")
    selected_gold = st.sidebar.selectbox("종목 선택", list(GOLD_PRODUCTS.keys()), key="gold_product")
    gold_ticker = GOLD_PRODUCTS[selected_gold]

    # 투자 설정
    st.sidebar.subheader("투자 설정")
    gold_initial_cap = st.sidebar.number_input(
        "투자금 (KRW)", value=1000000, step=100000, format="%d", key="gold_cap"
    )

    # Gold Trader 초기화
    gold_trader = None
    if kiwoom_ak and kiwoom_sk:
        gold_trader = KiwoomGoldTrader(is_mock=True)
        gold_trader.app_key = kiwoom_ak
        gold_trader.app_secret = kiwoom_sk
        gold_trader.account_no = kiwoom_account

    # --- Main Content ---
    tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs(["📊 금 시세", "💰 계좌/거래", "📈 차트 분석", "💳 수수료/세금"])

    # --- Tab 1: 금 시세 ---
    with tab_g1:
        st.header("금 현물 시세")

        if not gold_trader:
            st.warning("키움증권 API Key를 사이드바에서 입력해주세요.")
            st.info("`.env` 파일에 `KIWOOM_APP_KEY`, `KIWOOM_SECRET_KEY`, `KIWOOM_ACCOUNT`를 설정하면 자동으로 로드됩니다.")
        else:
            # 시세 조회 (현재 Mock)
            price_data = gold_trader.get_market_price(gold_ticker)

            if price_data and "output" in price_data:
                output = price_data["output"]
                current_price = int(output.get("price", 0))
                change = int(output.get("change", 0))

                st.caption(f"종목: {selected_gold} ({gold_ticker}) | 데이터: Mock (실제 API 연동 필요)")

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("현재가 (1g)", f"{current_price:,}원", delta=f"{change:,}원")
                p2.metric("투자금", f"{gold_initial_cap:,}원")
                p3.metric("매수 가능 수량", f"{gold_initial_cap // current_price if current_price > 0 else 0}g")
                p4.metric("API 상태", "Mock 모드" if gold_trader.is_mock else "실거래")

                st.divider()

                # 국제 금 시세 참고 정보
                st.subheader("참고: 국제 금 시세")
                st.markdown("""
| 구분 | 단위 | 비고 |
|------|------|------|
| 국제 금 (XAU/USD) | Troy oz (31.1g) | COMEX 기준 |
| KRX 금현물 | 1g | 원화 기준 |
| 순도 | 99.99% | KRX 금시장 표준 |
                """)
            else:
                st.error("시세 조회 실패. API 연결을 확인해주세요.")

        # API 연동 상태
        with st.expander("🔧 API 연동 상태", expanded=False):
            st.markdown(f"""
**현재 상태**: {'Mock 데이터 사용 중' if not gold_trader or gold_trader.is_mock else '실거래 연동'}

**구현 완료**:
- OAuth2 인증 메서드
- 금현물 현재가 조회 (Mock)

**추가 구현 필요**:
- 실제 API 엔드포인트 연동
- 일봉/분봉 차트 데이터 조회
- 매수/매도 주문 실행
- 잔고 조회

**필요 환경변수**:
- `KIWOOM_APP_KEY`: 키움 Open API App Key
- `KIWOOM_SECRET_KEY`: 키움 Open API Secret Key
- `KIWOOM_ACCOUNT`: 키움증권 계좌번호
            """)

    # --- Tab 2: 계좌/거래 ---
    with tab_g2:
        st.header("계좌 및 거래")

        if not gold_trader:
            st.warning("API Key를 먼저 설정해주세요.")
        else:
            # 계좌 정보 (Mock)
            st.subheader("계좌 정보")
            acc_c1, acc_c2, acc_c3 = st.columns(3)
            acc_c1.metric("계좌번호", kiwoom_account if kiwoom_account else "미설정")
            acc_c2.metric("예수금", "- 원 (조회 필요)")
            acc_c3.metric("금 보유량", "- g (조회 필요)")

            st.divider()

            # 매매 (Mock)
            st.subheader("수동 매매")
            trade_col1, trade_col2 = st.columns(2)

            with trade_col1:
                st.markdown("**매수**")
                buy_qty = st.number_input("매수 수량 (g)", min_value=1, value=1, step=1, key="gold_buy_qty")
                buy_price = st.number_input("매수 단가 (원)", min_value=0, value=100000, step=1000, key="gold_buy_price")
                buy_total = buy_qty * buy_price
                st.caption(f"매수 총액: {buy_total:,}원")
                if st.button("매수 주문", key="gold_buy_btn", type="primary"):
                    st.warning("실제 API 연동 후 사용 가능합니다. (현재 Mock 모드)")

            with trade_col2:
                st.markdown("**매도**")
                sell_qty = st.number_input("매도 수량 (g)", min_value=1, value=1, step=1, key="gold_sell_qty")
                sell_price = st.number_input("매도 단가 (원)", min_value=0, value=100000, step=1000, key="gold_sell_price")
                sell_total = sell_qty * sell_price
                st.caption(f"매도 총액: {sell_total:,}원")
                if st.button("매도 주문", key="gold_sell_btn", type="primary"):
                    st.warning("실제 API 연동 후 사용 가능합니다. (현재 Mock 모드)")

    # --- Tab 3: 차트 분석 ---
    with tab_g3:
        st.header("금 차트 분석")

        if not gold_trader:
            st.warning("API Key를 먼저 설정해주세요.")
        else:
            chart_interval = st.selectbox("차트 주기", ["일봉", "주봉", "월봉", "30분봉"], key="gold_chart_interval")

            st.info("차트 데이터는 실제 API 연동 후 표시됩니다. 현재 샘플 차트를 표시합니다.")

            # 샘플 차트 생성 (Mock 데이터)
            np.random.seed(42)
            dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
            base_price = 100000
            returns = np.random.normal(0.0003, 0.008, len(dates))
            prices = base_price * np.cumprod(1 + returns)
            high = prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
            low = prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
            open_p = prices * (1 + np.random.normal(0, 0.003, len(dates)))

            df_gold_mock = pd.DataFrame({
                'open': open_p, 'high': high, 'low': low, 'close': prices
            }, index=dates)

            # SMA 오버레이
            gold_sma_period = st.slider("SMA 기간", 5, 60, 20, key="gold_sma")
            df_gold_mock[f'SMA_{gold_sma_period}'] = df_gold_mock['close'].rolling(window=gold_sma_period).mean()

            fig_gold = go.Figure()
            fig_gold.add_trace(go.Candlestick(
                x=df_gold_mock.index, open=df_gold_mock['open'],
                high=df_gold_mock['high'], low=df_gold_mock['low'],
                close=df_gold_mock['close'], name='금 가격',
                increasing_line_color='#FF6B35', decreasing_line_color='#4169E1',
            ))
            fig_gold.add_trace(go.Scatter(
                x=df_gold_mock.index, y=df_gold_mock[f'SMA_{gold_sma_period}'],
                name=f'SMA({gold_sma_period})', line=dict(color='orange', width=2)
            ))
            fig_gold.update_layout(
                title=f"{selected_gold} - {chart_interval} (Mock 데이터)",
                height=500,
                xaxis_rangeslider_visible=False,
                yaxis_title="가격 (원/g)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_gold, use_container_width=True)

            # 기본 지표
            last_close = df_gold_mock['close'].iloc[-1]
            last_sma = df_gold_mock[f'SMA_{gold_sma_period}'].iloc[-1]
            disparity = (last_close - last_sma) / last_sma * 100

            ind_c1, ind_c2, ind_c3 = st.columns(3)
            ind_c1.metric("현재가", f"{last_close:,.0f}원")
            ind_c2.metric(f"SMA({gold_sma_period})", f"{last_sma:,.0f}원")
            ind_c3.metric("이격도", f"{disparity:+.2f}%")

            signal = "매수 (SMA 위)" if last_close > last_sma else "매도 (SMA 아래)"
            sig_color = "green" if last_close > last_sma else "red"
            st.markdown(f"**SMA 시그널**: :{sig_color}[{signal}]")


def main():
    # --- 모드 선택 (코인 / Gold) ---
    mode_col, title_col = st.columns([1, 5])
    with mode_col:
        trading_mode = st.selectbox(
            "거래 모드",
            ["🪙 코인", "🥇 Gold"],
            key="trading_mode",
            label_visibility="collapsed"
        )

    if trading_mode == "🥇 Gold":
        render_gold_mode()
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

    # Portfolio Management
    st.sidebar.subheader("포트폴리오 관리")
    st.sidebar.caption("각 코인의 개별 SMA(이동평균) 기간을 설정할 수 있습니다.")
    
    # Load portfolio from config or default
    # Load portfolio from config or default
    # Interval Mapping for User Friendliness (Simplified)
    INTERVAL_MAP = {
        "일봉": "day",
        "4시간": "minute240",
        "1시간": "minute60",
        "30분": "minute30",
        "15분": "minute15",
        "5분": "minute5",
        "1분": "minute1"
    }
    INTERVAL_REV_MAP = {v: k for k, v in INTERVAL_MAP.items()}
    CANDLES_PER_DAY = {
        "day": 1, "minute240": 6, "minute60": 24,
        "minute30": 48, "minute15": 96, "minute5": 288, "minute1": 1440
    }
    
    # Load portfolio: user_config.json → portfolio.json (기본값 없음, 없으면 오류)
    PORTFOLIO_JSON_LOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")
    # portfolio.json에서 설정값도 로드 (object 형태 지원)
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

    default_portfolio = config.get("portfolio", None)
    if not default_portfolio:
        default_portfolio = _pjson_config.get("portfolio", None)
    if not default_portfolio:
        st.error("portfolio.json 파일이 없거나 포트폴리오 데이터가 비어있습니다. 로컬에서 저장 후 push 해주세요.")
        st.stop()
    
    # Convert to DataFrame for Editor (Use Labels)
    sanitized_portfolio = []
    def_len = len(default_portfolio)
    for p in default_portfolio:
        api_interval = p.get("interval", "day")
        label_interval = INTERVAL_REV_MAP.get(api_interval, "일봉")
        
        # Migrate old 'sma' key to 'parameter' if needed
        param_val = p.get("parameter", p.get("sma", 20))
        
        # Migration: Map old long names to short names
        strat_map = {"SMA 전략": "SMA", "돈키안 전략": "Donchian", "Donchian Trend": "Donchian"}
        strat_val = p.get("strategy", "SMA")
        strat_val = strat_map.get(strat_val, strat_val)

        sell_param_val = p.get("sell_parameter", 0)

        sanitized_portfolio.append({
            "coin": str(p.get("coin", "BTC")).upper(),
            "strategy": strat_val,
            "parameter": param_val,
            "sell_parameter": sell_param_val,
            "weight": p.get("weight", 100 // def_len if def_len > 0 else 100),
            "interval": label_interval
        })
        
    df_portfolio = pd.DataFrame(sanitized_portfolio)
    
    interval_options = list(INTERVAL_MAP.keys())
    strategy_options = ["SMA", "Donchian"]

    if IS_CLOUD:
        # Cloud: 읽기 전용 테이블
        st.sidebar.dataframe(df_portfolio, use_container_width=True, hide_index=True)
        edited_portfolio = df_portfolio
    else:
        edited_portfolio = st.sidebar.data_editor(df_portfolio, num_rows="dynamic", use_container_width=True, hide_index=True,
                                                  column_config={
                                                      "coin": st.column_config.TextColumn("코인", required=True),
                                                      "strategy": st.column_config.SelectboxColumn("전략", options=strategy_options, required=True, default="SMA"),
                                                      "parameter": st.column_config.NumberColumn("매수", min_value=5, max_value=300, step=1, required=True),
                                                      "sell_parameter": st.column_config.NumberColumn("매도", min_value=0, max_value=300, step=1, required=False, default=0, help="돈치안 매도 채널 (0=매수의 절반)"),
                                                      "weight": st.column_config.NumberColumn("비중", min_value=0, max_value=100, step=1, required=True, format="%d%%"),
                                                      "interval": st.column_config.SelectboxColumn("시간봉", options=interval_options, required=True, default="일봉")
                                                  })
    
    # Calculate Total Weight & Cash
    total_weight = edited_portfolio["weight"].sum()
    if total_weight > 100:
        st.sidebar.error(f"총 비중이 {total_weight}% 입니다. (100% 이하로 설정해주세요)")
    else:
        cash_weight = 100 - total_weight
        st.sidebar.info(f"투자 비중: {total_weight}% | 현금: {cash_weight}%")
    
    # Convert back to list of dicts (Map Labels back to API Keys)
    portfolio_list = []
    for r in edited_portfolio.to_dict('records'):
        label_key = r['interval']
        api_key = INTERVAL_MAP.get(label_key, "day") # Default to day if not found
        
        sell_p = int(r.get('sell_parameter', 0) or 0)
        portfolio_list.append({
            "market": "KRW",
            "coin": r['coin'].upper(),
            "strategy": r['strategy'],
            "parameter": r['parameter'],
            "sell_parameter": sell_p,
            "weight": r['weight'],
            "interval": api_key
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

        if save_col1.button("💾 저장"):
            new_config = {
                "portfolio": portfolio_list,
                "start_date": str(start_date),
                "initial_cap": initial_cap
            }
            save_config(new_config)
            portfolio_json_data = {
                "portfolio": portfolio_list,
                "start_date": str(start_date),
                "initial_cap": initial_cap
            }
            with open(PORTFOLIO_JSON, "w", encoding="utf-8") as f:
                json.dump(portfolio_json_data, f, indent=2, ensure_ascii=False)
            st.sidebar.success("저장 완료!")

        if save_col2.button("📂 불러오기"):
            if os.path.exists(PORTFOLIO_JSON):
                try:
                    with open(PORTFOLIO_JSON, "r", encoding="utf-8") as f:
                        imported = json.load(f)
                    if isinstance(imported, list) and len(imported) > 0:
                        new_config = {
                            "portfolio": imported,
                            "start_date": str(start_date),
                            "initial_cap": initial_cap
                        }
                        save_config(new_config)
                        st.sidebar.success(f"{len(imported)}개 자산 불러오기 완료!")
                        st.rerun()
                    else:
                        st.sidebar.error("올바른 포트폴리오 JSON 형식이 아닙니다.")
                except json.JSONDecodeError:
                    st.sidebar.error("JSON 파싱 오류. 파일을 확인해주세요.")
            else:
                st.sidebar.warning("portfolio.json 파일이 없습니다.")

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
    
    trader = None
    if current_ak and current_sk:
        @st.cache_resource
        def get_trader(ak, sk):
            return UpbitTrader(ak, sk)
        trader = get_trader(current_ak, current_sk)

    # --- Background Worker Setup ---
    @st.cache_resource
    def get_worker():
        return MarketDataWorker()
    
    worker = get_worker()

    # 시가총액 상위 20 티커 (글로벌 Market Cap 기준)
    TOP_20_TICKERS = [
        "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
        "KRW-ADA", "KRW-SHIB", "KRW-TRX", "KRW-AVAX", "KRW-LINK",
        "KRW-BCH", "KRW-DOT", "KRW-NEAR", "KRW-POL", "KRW-ETC",
        "KRW-XLM", "KRW-STX", "KRW-HBAR", "KRW-EOS", "KRW-SAND"
    ]

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 실시간 포트폴리오", "📊 백테스트", "📜 자산 입출금", "📡 전체 종목 스캔"])

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
                    st.rerun()
            with col_ctrl2:
                st.info(f"워커 상태: **{w_msg}**")
                
            if not portfolio_list:
                st.warning("사이드바에서 포트폴리오에 코인을 추가해주세요.")
            else:
                count = len(portfolio_list)
                per_coin_cap = initial_cap / count
                
                # --- Total Summary Container ---
                st.subheader("🏁 포트폴리오 요약")
                st.caption(f"초기자본: {initial_cap:,.0f} KRW | 자산수: {count} | 자산당: {per_coin_cap:,.0f} KRW")
                
                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                
                total_real_val = trader.get_balance("KRW") or 0
                total_init_val = initial_cap
                
                # Cash Logic
                total_weight_alloc = sum([item.get('weight', 0) for item in portfolio_list])
                cash_ratio = max(0, 100 - total_weight_alloc) / 100.0
                reserved_cash = initial_cap * cash_ratio
                
                # Add reserved cash to Theo Value (as it stays as cash)
                total_theo_val = reserved_cash
                
                # --- 전체 자산 현황 테이블 ---
                krw_bal_summary = trader.get_balance("KRW") or 0
                asset_summary_rows = [{"자산": "KRW (현금)", "보유량": f"{krw_bal_summary:,.0f}", "현재가": "-", "평가금액(KRW)": f"{krw_bal_summary:,.0f}", "상태": "-"}]
                seen_coins_summary = set()
                for s_item in portfolio_list:
                    s_coin = s_item['coin'].upper()
                    if s_coin in seen_coins_summary:
                        continue
                    seen_coins_summary.add(s_coin)
                    s_ticker = f"{s_item['market']}-{s_coin}"
                    s_bal = trader.get_balance(s_coin) or 0
                    s_price = pyupbit.get_current_price(s_ticker) or 0
                    s_val = s_bal * s_price
                    is_holding = s_val >= 5000
                    asset_summary_rows.append({
                        "자산": s_coin,
                        "보유량": (f"{s_bal:.8f}" if s_bal < 1 else f"{s_bal:,.4f}") if s_bal > 0 else "0",
                        "현재가": f"{s_price:,.0f}",
                        "평가금액(KRW)": f"{s_val:,.0f}",
                        "상태": "보유중" if is_holding else "미보유",
                    })
                total_real_summary = krw_bal_summary + sum(
                    (trader.get_balance(c) or 0) * (pyupbit.get_current_price(f"KRW-{c}") or 0)
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
                                df_60 = pyupbit.get_ohlcv(p_ticker, interval=p_interval, count=max(60 + p_param, 200))
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

                            except Exception:
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
                                
                                curr_signal = strategy_eng.get_signal(last_candle, buy_period=buy_p, sell_period=sell_p)
                                
                            else: # SMA Strategy (Default)
                                strategy_eng = SMAStrategy()
                                calc_periods = [param_val]
                                    
                                df_curr = strategy_eng.create_features(df_curr, periods=calc_periods)
                                last_candle = df_curr.iloc[-2]
                                
                                curr_sma = last_candle[f'SMA_{param_val}']
                                curr_signal = strategy_eng.get_signal(last_candle, strategy_type='SMA_CROSS', ma_period=param_val)
                            # Current Price: Worker data might be 10s old, but good enough. 
                            # Or fetch current price live single?
                            curr_price = pyupbit.get_current_price(ticker) 
                            
                            # 2. Fetch Balance 
                            coin_sym = item['coin'].upper()
                            coin_bal = trader.get_balance(coin_sym) or 0

                            # 3. Theo Backtest (Sync Check) - 캐시 우선 (다운로드 없음)
                            sell_ratio = (item.get('sell_parameter', 0) or max(5, param_val // 2)) / param_val if param_val > 0 else 0.5
                            # 캐시 로드 (API 호출 없이 로컬 파일만)
                            df_bt = data_cache.load_cached(ticker, interval)
                            if df_bt is not None and len(df_bt) >= param_val:
                                req_count = len(df_bt)
                            else:
                                df_bt = df_curr  # 캐시 없으면 Worker 데이터 사용
                                req_count = len(df_bt)
                            bt_res = backtest_engine.run_backtest(ticker, period=param_val, interval=interval, count=req_count, start_date=start_date, initial_balance=per_coin_cap, df=df_bt, strategy_mode=strategy_mode, sell_period_ratio=sell_ratio)
                            
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
                            p_tab1, p_tab2 = st.tabs(["📈 분석 & 벤치마크", "🛒 주문 & 체결"])
                            
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
                                o_col1, o_col2 = st.columns([1, 1])
                                with o_col1:
                                    st.write("**호가창**")
                                    try:
                                        ob = pyupbit.get_orderbook(ticker)
                                        if isinstance(ob, list): ob = ob[0]
                                        if ob:
                                            asks = ob['orderbook_units'][:5]
                                            for a in reversed(asks):
                                                st.markdown(f"<div style='color:red; text-align:right'>{a['ask_price']:,.0f} | {a['ask_size']:.3f}</div>", unsafe_allow_html=True)
                                            st.divider()
                                            for b in asks: # Use same count
                                                 st.markdown(f"<div style='color:green; text-align:right'>{b['bid_price']:,.0f} | {b['bid_size']:.3f}</div>", unsafe_allow_html=True)
                                    except:
                                        st.write("호가 없음")
                                
                                with o_col2:
                                    st.write("**수동 실행**")
                                    if st.button(f"매매 로직 확인 ({item['coin']})", key=f"btn_{ticker}_{asset_idx}"):
                                        res = trader.check_and_trade(ticker, interval=interval, sma_period=param_val)
                                        st.info(res)

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

                # --- Portfolio Rebalancing Section ---
                st.divider()
                with st.expander("⚖️ 포트폴리오 리밸런싱 (Rebalancing)", expanded=False):
                    krw_balance = trader.get_balance("KRW") or 0

                    # 각 자산의 실제 보유 상태 확인
                    asset_states = []
                    for rb_idx, rb_item in enumerate(portfolio_list):
                        rb_ticker = f"{rb_item['market']}-{rb_item['coin'].upper()}"
                        rb_coin = rb_item['coin'].upper()
                        rb_weight = rb_item.get('weight', 0)
                        rb_interval = rb_item.get('interval', 'day')
                        rb_strategy = rb_item.get('strategy', 'SMA Strategy')
                        rb_param = rb_item.get('parameter', 20)
                        rb_sell_param = rb_item.get('sell_parameter', 0)

                        rb_coin_bal = trader.get_balance(rb_coin) or 0
                        rb_price = pyupbit.get_current_price(rb_ticker) or 0
                        rb_coin_val = rb_coin_bal * rb_price
                        rb_status = "HOLD" if rb_coin_val > 5000 else "CASH"

                        # 전략 시그널 확인
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

                        # 목표 배분 금액 (현재 총 실제자산 기준)
                        rb_target_krw = total_real_val * (rb_weight / 100.0)

                        asset_states.append({
                            "ticker": rb_ticker,
                            "coin": rb_coin,
                            "weight": rb_weight,
                            "interval": rb_interval,
                            "strategy": rb_strategy,
                            "param": rb_param,
                            "sell_param": rb_sell_param,
                            "status": rb_status,
                            "signal": rb_signal,
                            "coin_bal": rb_coin_bal,
                            "coin_val": rb_coin_val,
                            "price": rb_price,
                            "target_krw": rb_target_krw,
                        })

                    # 상태 요약
                    cash_assets = [a for a in asset_states if a['status'] == 'CASH']
                    hold_assets = [a for a in asset_states if a['status'] == 'HOLD']
                    buy_signal_assets = [a for a in asset_states if a['signal'] == 'BUY']

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("보유 현금 (KRW)", f"{krw_balance:,.0f}")
                    rc2.metric("CASH 자산", f"{len(cash_assets)} / {len(asset_states)}")
                    rc3.metric("BUY 시그널", f"{len(buy_signal_assets)} / {len(asset_states)}")

                    # 리밸런싱 테이블
                    rebal_data = []
                    for a in asset_states:
                        diff_krw = a['target_krw'] - a['coin_val']
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

                    # BUY 시그널이 있는 CASH 자산만 매수 대상
                    buyable = [a for a in asset_states if a['status'] == 'CASH' and a['signal'] == 'BUY']

                    if not buyable:
                        if len(cash_assets) == 0:
                            st.success("모든 자산이 이미 보유 중입니다.")
                        else:
                            st.info(f"현금 자산 {len(cash_assets)}개가 있지만 BUY 시그널이 없습니다. 시그널 발생 시 매수 가능합니다.")
                    else:
                        # 매수 가능 자산 표시
                        st.warning(f"**{len(buyable)}개 자산**에 BUY 시그널이 있습니다. 리밸런싱 매수를 실행할 수 있습니다.")

                        # 배분 금액 계산
                        total_buy_weight = sum(a['weight'] for a in buyable)
                        available_krw = krw_balance * 0.999  # 수수료 여유분

                        buy_plan = []
                        for a in buyable:
                            # 비중 비례 배분
                            alloc_krw = available_krw * (a['weight'] / total_buy_weight) if total_buy_weight > 0 else 0
                            alloc_krw = min(alloc_krw, available_krw)
                            buy_plan.append({
                                "종목": a['ticker'],
                                "비중": f"{a['weight']}%",
                                "배분 금액(KRW)": f"{alloc_krw:,.0f}",
                                "시간봉": a['interval'],
                                "현재가": f"{a['price']:,.0f}",
                                "_ticker": a['ticker'],
                                "_krw": alloc_krw,
                                "_interval": a['interval'],
                            })

                        plan_df = pd.DataFrame(buy_plan)
                        st.dataframe(plan_df[["종목", "비중", "배분 금액(KRW)", "시간봉", "현재가"]], use_container_width=True, hide_index=True)

                        st.caption(f"총 배분 금액: {sum(p['_krw'] for p in buy_plan):,.0f} KRW / 보유 현금: {krw_balance:,.0f} KRW")

                        # 실행 버튼
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

    # --- Tab 2: Backtest (Single) ---
    with tab2:
        st.header("개별 자산 백테스트")
        
        # Select ticker from portfolio for convenience, or custom
        port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
        
        # Merge and Remove Duplicates
        base_options = list(dict.fromkeys(port_tickers + TOP_20_TICKERS))
        
        # --- Strategy Selection (Top) ---
        bt_strategy = st.selectbox(
            "전략 선택",
            ["SMA 전략", "돈키안 전략"],
            index=0,
            key="bt_strategy_sel"
        )

        selected_ticker_bt = st.selectbox("백테스트 대상", base_options + ["직접입력"])

        bt_ticker = ""
        bt_sma = 0
        bt_buy_period = 20
        bt_sell_period = 10

        if selected_ticker_bt == "직접입력":
            c1, c2 = st.columns(2)
            c = c2.text_input("코인", "BTC", key="bt_c")
            bt_ticker = f"KRW-{c.upper()}"
        else:
            bt_ticker = selected_ticker_bt

        # --- Strategy-specific Parameters ---
        if bt_strategy == "SMA 전략":
            item = next((item for item in portfolio_list if f"{item['market']}-{item['coin'].upper()}" == bt_ticker), None)
            default_sma = item.get('parameter', 60) if item else 60
            bt_sma = st.number_input("단기 SMA (추세)", value=default_sma, key="bt_sma_select", min_value=5, step=1)
        else:  # Donchian Strategy
            dc_col1, dc_col2 = st.columns(2)
            with dc_col1:
                bt_buy_period = st.number_input("매수 채널 기간", value=20, min_value=5, max_value=300, step=1, key="bt_dc_buy", help="N일 고가 돌파 시 매수")
            with dc_col2:
                bt_sell_period = st.number_input("매도 채널 기간", value=10, min_value=5, max_value=300, step=1, key="bt_dc_sell", help="N일 저가 이탈 시 매도")

        # Backtest Interval Selection
        bt_interval_label = st.selectbox("시간봉 선택", options=list(INTERVAL_MAP.keys()), index=0, key="bt_interval_sel")
        bt_interval = INTERVAL_MAP[bt_interval_label]

        # 코인/시간봉별 기본 슬리피지 테이블 (%)
        DEFAULT_SLIPPAGE = {
            # (coin_type, interval) -> slippage %
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
            # Date Range Selector (Split)
            st.caption("백테스트 기간")
            d_col1, d_col2 = st.columns(2)

            # Default Backtest Start: 2020-01-01
            try:
                default_start_bt = datetime(2020, 1, 1).date()
            except:
                default_start_bt = datetime.now().date() - timedelta(days=365)
            default_end_bt = datetime.now().date()

            bt_start = d_col1.date_input(
                "시작일",
                value=default_start_bt,
                max_value=datetime.now().date(),
                format="YYYY.MM.DD"
            )

            bt_end = d_col2.date_input(
                "종료일",
                value=default_end_bt,
                max_value=datetime.now().date(),
                format="YYYY.MM.DD"
            )

            if bt_start > bt_end:
                st.error("시작일은 종료일보다 빨라야 합니다.")
                bt_end = bt_start # Fallback to prevent crash

            days_diff = (bt_end - bt_start).days

            st.caption(f"기간: {days_diff}일")

            fee = st.number_input("매매 수수료 (%)", value=0.05, format="%.2f") / 100
            bt_slippage = st.number_input("슬리피지 (%)", value=default_slip, min_value=0.0, max_value=2.0, step=0.01, format="%.2f",
                                           help="매수시 +%, 매도시 -% 적용. 코인/시간봉에 따라 자동 설정됩니다.")

            # 거래당 총 비용 표시
            fee_pct = fee * 100  # 수수료 %
            cost_per_trade = fee_pct + bt_slippage  # 편도 비용
            cost_round_trip = (fee_pct * 2) + (bt_slippage * 2)  # 왕복 비용 (매수+매도)
            st.caption(f"편도 비용: {cost_per_trade:.2f}% (수수료 {fee_pct:.2f}% + 슬리피지 {bt_slippage:.2f}%)")
            st.caption(f"왕복 비용: {cost_round_trip:.2f}% (매수+매도)")

            run_btn = st.button("백테스트 실행", type="primary")

        if run_btn:
            # Determine period for data fetch buffer
            if bt_strategy == "돈키안 전략":
                req_period = max(bt_buy_period, bt_sell_period)
                bt_strategy_mode = "Donchian"
                bt_sell_ratio = bt_sell_period / bt_buy_period if bt_buy_period > 0 else 0.5
            else:
                req_period = bt_sma
                bt_strategy_mode = "SMA 전략"
                bt_sell_ratio = 0.5

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

                # Data range validation
                data_start = df_bt.index[0]
                data_end = df_bt.index[-1]
                st.caption(f"조회된 캔들: {len(df_bt)}개 ({data_start.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')})")

                result = backtest_engine.run_backtest(
                    bt_ticker,
                    period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                    interval=bt_interval,
                    count=fetch_count,
                    fee=fee,
                    start_date=bt_start,
                    initial_balance=initial_cap,
                    df=df_bt,
                    strategy_mode=bt_strategy_mode,
                    sell_period_ratio=bt_sell_ratio,
                    slippage=bt_slippage
                )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    df = result["df"]
                    res = result["performance"]
                    
                    # Metrics Row
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("총 수익률", f"{res['total_return']:,.2f}%")
                    m2.metric("연평균(CAGR)", f"{res.get('cagr', 0):,.2f}%")
                    m3.metric("승률", f"{res['win_rate']:,.2f}%")
                    m4.metric("최대낙폭(MDD)", f"{res['mdd']:,.2f}%")
                    m5.metric("샤프비율", f"{res['sharpe']:.2f}")

                    # 비용 & 결과 요약
                    trade_count = res['trade_count']
                    total_cost_pct = cost_round_trip * trade_count  # 총 왕복 비용 x 거래횟수
                    st.success(
                        f"최종 잔고: **{res['final_equity']:,.0f} KRW** (초기 {initial_cap:,.0f} KRW) | "
                        f"거래 {trade_count}회 | "
                        f"거래비용: 편도 {cost_per_trade:.2f}% · 왕복 {cost_round_trip:.2f}% "
                        f"(수수료 {fee_pct:.2f}% + 슬리피지 {bt_slippage:.2f}%) | "
                        f"누적 비용 약 {total_cost_pct:.1f}%"
                    )

                    # 슬리피지 비교 (0% vs 설정값)
                    if bt_slippage > 0:
                        result_no_slip = backtest_engine.run_backtest(
                            bt_ticker,
                            period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                            interval=bt_interval, count=fetch_count, fee=fee,
                            start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                            strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio,
                            slippage=0.0
                        )
                        if "error" not in result_no_slip:
                            res_ns = result_no_slip['performance']
                            slip_cost = res_ns['final_equity'] - res['final_equity']
                            slip_ret_diff = res_ns['total_return'] - res['total_return']
                            st.info(
                                f"슬리피지 영향: {bt_slippage}% 적용 시 "
                                f"수익률 차이 **{slip_ret_diff:,.2f}%p**, "
                                f"금액 차이 **{slip_cost:,.0f} KRW** "
                                f"(슬리피지 없는 경우 {res_ns['final_equity']:,.0f} KRW)"
                            )
                    
                    # --- Combined Chart ---
                    st.subheader("가격 & 전략 성과")

                    
                    # Create Dual Axis Chart + Drawdown
                    fig = make_subplots(
                        rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3],
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                    )
                    
                    # 1. Candlestick (Price) - Row 1, Primary Y
                    fig.add_trace(go.Candlestick(
                        x=df.index, open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'],
                        name='가격'
                    ), row=1, col=1, secondary_y=False)
                    
                    # 2. Strategy Indicator Lines - Row 1, Primary Y
                    if bt_strategy_mode == "Donchian":
                        upper_col = f'Donchian_Upper_{bt_buy_period}'
                        lower_col = f'Donchian_Lower_{bt_sell_period}'
                        if upper_col in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[upper_col],
                                line=dict(color='green', width=1.5, dash='dash'),
                                name=f'상단 ({bt_buy_period})'
                            ), row=1, col=1, secondary_y=False)
                        if lower_col in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[lower_col],
                                line=dict(color='red', width=1.5, dash='dash'),
                                name=f'하단 ({bt_sell_period})'
                            ), row=1, col=1, secondary_y=False)
                    else:
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df[f'SMA_{bt_sma}'],
                            line=dict(color='orange', width=2),
                            name=f'SMA {bt_sma}'
                        ), row=1, col=1, secondary_y=False)

                    
                    # 3. Strategy Equity - Row 1, Secondary Y
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['equity'],
                        line=dict(color='blue', width=2),
                        name='전략 자산'
                    ), row=1, col=1, secondary_y=True)
                    
                    # 4. Buy/Sell Signals - Row 1, Primary Y
                    # Use 'trades' list for accurate signal placement (Only actual trades)
                    buy_dates = [t['date'] for t in res['trades'] if t['type'] == 'buy']
                    buy_prices = [t['price'] for t in res['trades'] if t['type'] == 'buy']
                    sell_dates = [t['date'] for t in res['trades'] if t['type'] == 'sell']
                    sell_prices = [t['price'] for t in res['trades'] if t['type'] == 'sell']
                    
                    if buy_dates:
                        fig.add_trace(go.Scatter(
                            x=buy_dates, y=buy_prices,
                            mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
                            name='매수 신호'
                        ), row=1, col=1, secondary_y=False)

                    if sell_dates:
                        fig.add_trace(go.Scatter(
                            x=sell_dates, y=sell_prices,
                            mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                            name='매도 신호'
                        ), row=1, col=1, secondary_y=False)
                        
                    # 5. Drawdown - Row 2
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['drawdown'],
                        name='낙폭 (%)',
                        fill='tozeroy',
                        line=dict(color='red', width=1)
                    ), row=2, col=1)

                    fig.update_layout(height=800, title_text="백테스트 결과", xaxis_rangeslider_visible=False, margin=dict(t=80),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                    fig.update_yaxes(title_text="가격 (KRW)", row=1, col=1, secondary_y=False)
                    fig.update_yaxes(title_text="자산 (KRW)", row=1, col=1, secondary_y=True)
                    fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Yearly Performance Table
                    if 'yearly_stats' in res:
                        st.subheader("📊 연도별 성과")
                        st.dataframe(res['yearly_stats'].style.format("{:.2f}%"))
                        
                    st.info(f"전략 상태: **{res['final_status']}** | 다음 행동: **{res['next_action'] if res['next_action'] else '없음'}**")
                    
                    # Trade List
                    with st.expander("거래 내역"):
                        if res['trades']:
                            trades_df = pd.DataFrame(res['trades'])
                            st.dataframe(trades_df.style.format({"price": "{:,.2f}", "amount": "{:,.6f}", "balance": "{:,.2f}", "profit": "{:,.2f}%"}))
                        else:
                            st.info("실행된 거래가 없습니다.")
                            
                    # Export Full Daily Log
                    csv_data = df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 일별 로그 다운로드 (전체 데이터)",
                        data=csv_data,
                        file_name=f"{bt_ticker}_{bt_start}_daily_log.csv",
                        mime="text/csv",
                        help="일별 OHLCV + 지표 + 신호 데이터를 다운로드하여 로직을 검증합니다."
                    )

        # --- Optimization Section (Fragment: prevents full page dimming) ---
        @st.fragment
        def optimization_section():
            st.divider()
            st.subheader("🛠️ 파라미터 최적화")

            # 캐시 관리
            with st.expander("📦 데이터 캐시 관리", expanded=False):
                cache_list = data_cache.list_cache()
                if cache_list:
                    st.dataframe(pd.DataFrame(cache_list), use_container_width=True, hide_index=True)
                else:
                    st.info("캐시된 데이터가 없습니다. 최적화 실행 시 자동으로 캐시됩니다.")

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    if st.button("🔄 캐시 전체 삭제", key="opt_clear_cache"):
                        data_cache.clear_cache()
                        st.success("캐시가 삭제되었습니다.")
                        st.rerun()
                with cc2:
                    pre_ticker = bt_ticker
                    pre_interval = bt_interval
                    if st.button(f"📥 {pre_ticker} 사전 다운로드", key="opt_preload"):
                        with st.spinner(f"{pre_ticker} ({INTERVAL_REV_MAP.get(pre_interval, pre_interval)}) 데이터 다운로드 중..."):
                            pre_df = data_cache.fetch_and_cache(pre_ticker, interval=pre_interval, count=10000)
                            if pre_df is not None:
                                st.success(f"다운로드 완료: {len(pre_df)} candles ({pre_df.index[0].strftime('%Y-%m-%d')} ~ {pre_df.index[-1].strftime('%Y-%m-%d')})")
                            else:
                                st.error("다운로드 실패")
                with cc3:
                    ci = data_cache.get_cache_info(pre_ticker, pre_interval)
                    if ci.get("exists"):
                        st.caption(f"캐시: {ci['rows']}행, {ci['size_kb']:.1f}KB")
                        st.caption(f"{str(ci['start'])[:10]} ~ {str(ci['end'])[:10]}")
                    else:
                        st.caption("캐시 없음")

                # 전체 종목 일괄 다운로드
                st.divider()
                dl_intervals = st.multiselect(
                    "다운로드 시간봉",
                    options=list(INTERVAL_MAP.keys()),
                    default=list(INTERVAL_MAP.keys()),
                    key="batch_dl_intervals"
                )
                if st.button("📥 전체 종목 일괄 다운로드", key="batch_download"):
                    dl_interval_apis = [INTERVAL_MAP[k] for k in dl_intervals]
                    all_tickers = list(dict.fromkeys(
                        [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list] + TOP_20_TICKERS
                    ))
                    total_jobs = len(all_tickers) * len(dl_interval_apis)
                    st.write(f"종목 {len(all_tickers)}개 x 시간봉 {len(dl_interval_apis)}개 = 총 {total_jobs}건")

                    batch_progress = st.progress(0)
                    batch_log = st.empty()

                    def batch_cb(cur, total, ticker, interval, rows):
                        batch_progress.progress(cur / total)
                        iv_label = INTERVAL_REV_MAP.get(interval, interval)
                        batch_log.text(f"[{cur}/{total}] {ticker} ({iv_label}) → {rows:,}행")

                    batch_results = data_cache.batch_download(
                        all_tickers, intervals=dl_interval_apis,
                        count=10000, progress_callback=batch_cb
                    )

                    batch_progress.progress(1.0)
                    total_rows = sum(r['rows'] for r in batch_results)
                    st.success(f"일괄 다운로드 완료! {len(batch_results)}건, 총 {total_rows:,}행")
                    st.rerun()

            with st.form("optimization_form"):
                # 최적화 방법 선택
                opt_method = st.radio(
                    "최적화 방법", ["Grid Search (전수 탐색)", "Optuna (베이지안 최적화)"],
                    horizontal=True, key="opt_method"
                )
                use_optuna = "Optuna" in opt_method

                # 공통: 시간봉 선택
                opt_interval_label = st.selectbox(
                    "시간봉", options=list(INTERVAL_MAP.keys()),
                    index=0, key="opt_interval_sel"
                )
                opt_interval = INTERVAL_MAP[opt_interval_label]

                if bt_strategy == "돈키안 전략":
                    st.caption("돈치안 채널의 매수 기간과 매도 기간을 최적화합니다.")

                    st.markdown("##### 1. 매수 채널 기간")
                    c1, c2, c3 = st.columns(3)
                    opt_buy_start = c1.number_input("시작", 5, 200, 10, key="opt_dc_buy_start")
                    opt_buy_end = c2.number_input("끝", 5, 200, 60, key="opt_dc_buy_end")
                    opt_buy_step = c3.number_input("간격", 1, 50, 5, key="opt_dc_buy_step")

                    st.markdown("##### 2. 매도 채널 기간")
                    c1, c2, c3 = st.columns(3)
                    opt_sell_start = c1.number_input("시작", 5, 200, 5, key="opt_dc_sell_start")
                    opt_sell_end = c2.number_input("끝", 5, 200, 30, key="opt_dc_sell_end")
                    opt_sell_step = c3.number_input("간격", 1, 50, 5, key="opt_dc_sell_step")

                else:  # SMA Strategy
                    st.caption("SMA 이동평균 기간을 최적화합니다.")

                    st.markdown("##### SMA 기간")
                    c1, c2, c3 = st.columns(3)
                    opt_s_start = c1.number_input("시작", 5, 200, 20, key="opt_s_start")
                    opt_s_end = c2.number_input("끝", 5, 200, 60, key="opt_s_end")
                    opt_s_step = c3.number_input("간격", 1, 50, 5, key="opt_s_step")

                # Optuna 전용 설정
                if use_optuna:
                    st.divider()
                    st.markdown("##### Optuna 설정")
                    oc1, oc2 = st.columns(2)
                    optuna_n_trials = oc1.number_input("탐색 횟수 (Trials)", 50, 2000, 200, step=50, key="optuna_trials")
                    optuna_objective = oc2.selectbox("목적함수", ["Calmar (CAGR/|MDD|)", "Sharpe", "수익률 (Return)", "MDD 최소"], key="optuna_obj")

                opt_submitted = st.form_submit_button("Start Optimization", type="primary")

            if not opt_submitted:
                return

            import plotly.express as px
            results = []

            with st.status("🔄 고속 최적화 진행 중...", expanded=True) as status:
                progress_bar = st.progress(0)
                log_area = st.empty()

                try:
                    import time as _time
                    opt_cpd = CANDLES_PER_DAY.get(opt_interval, 1)
                    to_date_opt = bt_end + timedelta(days=1)
                    to_str_opt = to_date_opt.strftime("%Y-%m-%d 09:00:00")

                    # --- Phase 1: 데이터 다운로드 ---
                    if bt_strategy == "돈키안 전략":
                        buy_range = range(opt_buy_start, opt_buy_end + 1, opt_buy_step)
                        sell_range = range(opt_sell_start, opt_sell_end + 1, opt_sell_step)
                        total_iter = len(buy_range) * len(sell_range)
                        max_req_p = max(opt_buy_end, opt_sell_end)
                        fetch_count_opt = days_diff * opt_cpd + max_req_p + 300
                    else:
                        sma_range = range(opt_s_start, opt_s_end + 1, opt_s_step)
                        total_iter = len(sma_range)
                        fetch_count_opt = days_diff * opt_cpd + opt_s_end + 300

                    # 예상 시간 표시
                    est_api_calls = fetch_count_opt // 200
                    est_seconds = est_api_calls * 0.15
                    st.write(f"📊 데이터: {bt_ticker} ({fetch_count_opt:,} candles, {opt_interval_label})")
                    st.write(f"📅 기간: {bt_start} ~ {bt_end} ({days_diff}일)")

                    # 캐시 확인
                    cache_info = data_cache.get_cache_info(bt_ticker, opt_interval)
                    if cache_info.get("exists") and cache_info.get("rows", 0) >= fetch_count_opt:
                        st.write(f"⚡ 캐시 사용 ({cache_info['rows']:,}행)")
                    elif est_api_calls > 10:
                        st.write(f"⏳ 데이터 다운로드 중... (약 {est_api_calls}회 API 호출, 예상 {est_seconds:.0f}초)")

                    def dl_progress(fetched, total):
                        pct = min(fetched / total, 1.0) if total > 0 else 0
                        progress_bar.progress(pct * 0.3)  # 다운로드는 전체 진행률의 30%
                        log_area.text(f"다운로드: {fetched:,}/{total:,} candles ({pct*100:.0f}%)")

                    t0 = _time.time()

                    full_df = data_cache.get_ohlcv_cached(
                        bt_ticker, interval=opt_interval, to=to_str_opt,
                        count=fetch_count_opt, progress_callback=dl_progress
                    )

                    dl_elapsed = _time.time() - t0

                    if full_df is None or full_df.empty:
                        status.update(label="❌ 데이터 로드 실패", state="error")
                        return

                    st.write(f"✅ 데이터 준비 완료: {full_df.index[0].strftime('%Y-%m-%d')} ~ {full_df.index[-1].strftime('%Y-%m-%d')} ({len(full_df):,} candles, {dl_elapsed:.1f}초)")

                    # --- Phase 2: 최적화 실행 ---
                    def opt_progress(idx, total, msg):
                        pct = 0.3 + (idx / total) * 0.7
                        progress_bar.progress(min(pct, 1.0))
                        log_area.text(f"{msg} ({idx}/{total} · {idx/total*100:.0f}%)")

                    t1 = _time.time()
                    optuna_result = None

                    if use_optuna:
                        # --- Optuna 베이지안 최적화 ---
                        obj_map = {"Calmar (CAGR/|MDD|)": "calmar", "Sharpe": "sharpe",
                                   "수익률 (Return)": "return", "MDD 최소": "mdd"}
                        obj_key = obj_map.get(optuna_objective, "calmar")

                        if bt_strategy == "돈키안 전략":
                            st.write(f"🧠 Optuna {optuna_n_trials}회 탐색 (Buy {opt_buy_start}~{opt_buy_end}, Sell {opt_sell_start}~{opt_sell_end}, 목적: {optuna_objective})")
                            optuna_result = backtest_engine.optuna_optimize(
                                full_df, strategy_mode="Donchian",
                                buy_range=(opt_buy_start, opt_buy_end),
                                sell_range=(opt_sell_start, opt_sell_end),
                                fee=fee, slippage=bt_slippage,
                                start_date=bt_start, initial_balance=initial_cap,
                                n_trials=optuna_n_trials, objective_metric=obj_key,
                                progress_callback=opt_progress
                            )
                        else:
                            st.write(f"🧠 Optuna {optuna_n_trials}회 탐색 (SMA {opt_s_start}~{opt_s_end}, 목적: {optuna_objective})")
                            optuna_result = backtest_engine.optuna_optimize(
                                full_df, strategy_mode="SMA 전략",
                                buy_range=(opt_s_start, opt_s_end),
                                fee=fee, slippage=bt_slippage,
                                start_date=bt_start, initial_balance=initial_cap,
                                n_trials=optuna_n_trials, objective_metric=obj_key,
                                progress_callback=opt_progress
                            )

                        # Optuna 결과 → results 리스트 변환
                        for r in optuna_result['trials']:
                            row = {
                                "Total Return (%)": r["total_return"],
                                "CAGR (%)": r["cagr"],
                                "MDD (%)": r["mdd"],
                                "Calmar": r["calmar"],
                                "Win Rate (%)": r["win_rate"],
                                "Sharpe": r["sharpe"],
                                "Trades": r["trade_count"]
                            }
                            if bt_strategy == "돈키안 전략":
                                row["Buy Period"] = r["buy_period"]
                                row["Sell Period"] = r["sell_period"]
                            else:
                                row["SMA Period"] = r["sma_period"]
                            results.append(row)

                        total_iter = optuna_n_trials
                    else:
                        # --- Grid Search (기존) ---
                        st.write(f"🚀 총 {total_iter}개 조합 Grid Search 시작...")

                        if bt_strategy == "돈키안 전략":
                            raw_results = backtest_engine.optimize_donchian(
                                full_df, buy_range, sell_range,
                                fee=fee, slippage=bt_slippage,
                                start_date=bt_start, initial_balance=initial_cap,
                                progress_callback=opt_progress
                            )
                            for r in raw_results:
                                results.append({
                                    "Buy Period": r["Buy Period"],
                                    "Sell Period": r["Sell Period"],
                                    "Total Return (%)": r["total_return"],
                                    "CAGR (%)": r["cagr"],
                                    "MDD (%)": r["mdd"],
                                    "Calmar": abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0,
                                    "Win Rate (%)": r["win_rate"],
                                    "Sharpe": r["sharpe"],
                                    "Trades": r["trade_count"]
                                })
                        else:
                            raw_results = backtest_engine.optimize_sma(
                                full_df, sma_range,
                                fee=fee, slippage=bt_slippage,
                                start_date=bt_start, initial_balance=initial_cap,
                                progress_callback=opt_progress
                            )
                            for r in raw_results:
                                results.append({
                                    "SMA Period": r["SMA Period"],
                                    "Total Return (%)": r["total_return"],
                                    "CAGR (%)": r["cagr"],
                                    "MDD (%)": r["mdd"],
                                    "Calmar": abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0,
                                    "Win Rate (%)": r["win_rate"],
                                    "Sharpe": r["sharpe"],
                                    "Trades": r["trade_count"]
                                })

                    opt_elapsed = _time.time() - t1
                    total_elapsed = _time.time() - t0
                    method_label = "Optuna" if use_optuna else "Grid Search"
                    status.update(label=f"✅ {method_label} 완료! ({total_iter}건, {dl_elapsed:.1f}초 + {opt_elapsed:.1f}초 = 총 {total_elapsed:.1f}초)", state="complete")

                except Exception as e:
                    status.update(label=f"❌ 오류: {e}", state="error")
                    import traceback
                    st.code(traceback.format_exc())
                    return

            # --- Results Display ---
            if not results:
                st.warning("결과가 없습니다.")
                return

            opt_df = pd.DataFrame(results)

            # 정렬 기준: Optuna면 목적함수 기준, Grid면 수익률 기준
            sort_col = "Calmar" if use_optuna else "Total Return (%)"
            opt_df = opt_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
            opt_df.index = opt_df.index + 1
            opt_df.index.name = "순위"
            best_row = opt_df.iloc[0]

            # Best 결과 표시
            if bt_strategy == "돈키안 전략":
                st.subheader("🏆 최적 결과")
                st.success(f"매수: **{int(best_row['Buy Period'])}**, 매도: **{int(best_row['Sell Period'])}** → "
                           f"수익률: {best_row['Total Return (%)']:.1f}%, CAGR: {best_row['CAGR (%)']:.1f}%, "
                           f"MDD: {best_row['MDD (%)']:.1f}%, Calmar: {best_row['Calmar']:.2f}")
            else:
                st.subheader("🏆 최적 결과")
                st.success(f"SMA: **{int(best_row['SMA Period'])}** → "
                           f"수익률: {best_row['Total Return (%)']:.1f}%, CAGR: {best_row['CAGR (%)']:.1f}%, "
                           f"MDD: {best_row['MDD (%)']:.1f}%, Calmar: {best_row['Calmar']:.2f}")

            # 결과 테이블
            gradient_cols = ['Total Return (%)', 'Calmar', 'Sharpe']
            st.dataframe(
                opt_df.style
                    .background_gradient(cmap='RdYlGn', subset=[c for c in gradient_cols if c in opt_df.columns])
                    .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)'])
                    .format("{:,.2f}"),
                use_container_width=True, height=500
            )

            # 차트
            if bt_strategy == "돈키안 전략" and not use_optuna:
                fig_opt = px.density_heatmap(
                    opt_df.reset_index(), x="Buy Period", y="Sell Period", z="Total Return (%)",
                    histfunc="avg", title="돈키안 최적화 히트맵",
                    text_auto=".0f", color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_opt, use_container_width=True)
            elif bt_strategy != "돈키안 전략" and not use_optuna:
                st.line_chart(opt_df.reset_index().set_index("SMA Period")[['Total Return (%)', 'MDD (%)']])

            # Optuna 전용: 탐색 이력 차트
            if use_optuna and optuna_result:
                st.divider()
                st.subheader("📈 Optuna 탐색 이력")

                trial_df = opt_df.reset_index()
                trial_df['Trial'] = range(1, len(trial_df) + 1)

                # Best value 누적 추이
                import optuna.visualization as optuna_vis
                try:
                    fig_history = go.Figure()
                    study = optuna_result['study']
                    best_vals = []
                    running_best = float('-inf')
                    for t in study.trials:
                        if t.value is not None and t.value > running_best:
                            running_best = t.value
                        best_vals.append(running_best)
                    fig_history.add_trace(go.Scatter(
                        y=best_vals, mode='lines', name=f'최고 {optuna_objective}',
                        line=dict(color='blue', width=2)
                    ))
                    fig_history.update_layout(
                        title=f"최고 {optuna_objective} 추이",
                        xaxis_title="시행 횟수", yaxis_title=optuna_objective,
                        height=350,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
                except Exception:
                    pass

                # 파라미터 중요도
                if bt_strategy == "돈키안 전략":
                    st.caption("파라미터별 목적함수 분포")
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        fig_buy = px.scatter(trial_df, x="Buy Period", y="Calmar",
                                             color="MDD (%)", color_continuous_scale="RdYlGn_r",
                                             title="매수 기간 vs Calmar")
                        st.plotly_chart(fig_buy, use_container_width=True)
                    with pc2:
                        fig_sell = px.scatter(trial_df, x="Sell Period", y="Calmar",
                                             color="MDD (%)", color_continuous_scale="RdYlGn_r",
                                             title="매도 기간 vs Calmar")
                        st.plotly_chart(fig_sell, use_container_width=True)

        optimization_section()

    # --- Tab 3: History ---
    with tab3:
        st.header("거래 내역 & 자금 관리")

        hist_tab1, hist_tab2, hist_tab3 = st.tabs(["🧪 가상 로그 (백테스트/페이퍼)", "💸 실제 거래 내역 (거래소)", "📊 슬리피지 분석"])
        
        with hist_tab1:
            st.subheader("가상 계좌 관리")

            # Simulated Deposit/Withdraw
            if 'virtual_adjustment' not in st.session_state:
                st.session_state.virtual_adjustment = 0

            c1, c2 = st.columns(2)
            amount = c1.number_input("금액 (KRW)", step=100000)
            if c2.button("입출금 (가상)"):
                st.session_state.virtual_adjustment += amount
                st.success(f"가상 잔고 조정: {amount:,.0f} KRW")

            st.info(f"누적 가상 조정액: {st.session_state.virtual_adjustment:,.0f} KRW")
            st.write("전략 로그를 보려면 백테스트 탭에서 실행하거나, 개별 자산 탭에서 확인하세요.")

        with hist_tab2:
            st.subheader("실제 거래 내역")

            if not trader:
                st.warning("사이드바에서 API 키를 설정해주세요.")
            else:
                c_h1, c_h2 = st.columns(2)
                h_type = c_h1.selectbox("조회 유형", ["입금", "출금", "체결 주문"])
                h_curr = c_h2.selectbox("화폐", ["전체", "KRW", "BTC", "ETH", "XRP", "SOL", "USDT", "DOGE", "ADA", "AVAX", "LINK"])

                # 날짜 범위 필터
                d_h1, d_h2 = st.columns(2)
                h_date_start = d_h1.date_input("조회 시작일", value=datetime.now().date() - timedelta(days=90), key="hist_start")
                h_date_end = d_h2.date_input("조회 종료일", value=datetime.now().date(), key="hist_end")

                if st.button("조회"):
                    with st.spinner("Upbit API 조회 중..."):
                        # 화폐: "전체"면 None 전달
                        api_curr = None if h_curr == "전체" else h_curr

                        data = []
                        error_msg = None
                        try:
                            if "입금" in h_type:
                                data, error_msg = trader.get_history('deposit', api_curr)
                            elif "출금" in h_type:
                                data, error_msg = trader.get_history('withdraw', api_curr)
                            elif "체결" in h_type:
                                data, error_msg = trader.get_history('order', api_curr)
                        except Exception as e:
                            error_msg = str(e)

                        if error_msg:
                            if "out_of_scope" in error_msg or "권한" in error_msg:
                                st.error("⚠️ API 키에 해당 조회 권한이 없습니다.")
                                st.info("💡 [업비트 > 마이페이지 > Open API 관리]에서 **자산조회**, **입출금 조회** 권한을 활성화해주세요.")
                            else:
                                st.error(f"API 오류: {error_msg}")
                        if data and len(data) > 0:
                            df_hist = pd.DataFrame(data)
                            # 날짜 필터 적용
                            date_col = None
                            for col in ['created_at', 'done_at', 'datetime', 'date']:
                                if col in df_hist.columns:
                                    date_col = col
                                    break
                            if date_col:
                                df_hist[date_col] = pd.to_datetime(df_hist[date_col])
                                mask = (df_hist[date_col].dt.date >= h_date_start) & (df_hist[date_col].dt.date <= h_date_end)
                                df_hist = df_hist[mask]
                                df_hist = df_hist.sort_values(date_col, ascending=False)

                            st.success(f"{len(df_hist)}건 조회됨")
                            st.dataframe(df_hist, use_container_width=True)
                        elif not error_msg:
                            st.warning(f"조회 결과 없음. (유형: {h_type}, 화폐: {h_curr})")
                            st.caption("Upbit API는 최근 내역만 반환합니다. 조회 유형을 변경해보세요.")

            st.caption("Upbit API 제한: 최근 100건까지 조회 가능")

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

    # --- Tab 4: 전체 종목 스캔 ---
    with tab4:
        st.header("전체 종목 스캔")
        st.caption("상위 종목을 전 시간대/전략으로 백테스트하여 Calmar 순으로 정렬합니다. (조회 데이터는 로컬 캐시에 자동 저장)")

        # 스캔 설정
        scan_col1, scan_col2, scan_col3 = st.columns(3)
        scan_strategy = scan_col1.selectbox("전략", ["SMA", "Donchian"], key="scan_strat")
        scan_period = scan_col2.number_input("기간 (Period)", 5, 300, 20, key="scan_period")
        scan_count = scan_col3.number_input("백테스트 캔들 수", 200, 10000, 2000, step=200, key="scan_count")

        scan_col4, scan_col5 = st.columns(2)
        scan_intervals = scan_col4.multiselect(
            "시간봉", list(INTERVAL_MAP.keys()),
            default=["일봉", "4시간", "1시간"],
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
