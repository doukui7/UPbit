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

st.set_page_config(page_title="Upbit SMA Trader", layout="wide")

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
    
    /* Sidebar Width Override */
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 520px !important;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🪙 Upbit SMA Auto-Trading System")
    
    # --- Sidebar: Configuration ---
    st.sidebar.header("설정 (Configuration)")
    
    # API Keys
    env_access = os.getenv("UPBIT_ACCESS_KEY")
    env_secret = os.getenv("UPBIT_SECRET_KEY")
    
    with st.sidebar.expander("API Keys", expanded=False):
        ak_input = st.text_input("Access Key", value=env_access if env_access else "", type="password")
        sk_input = st.text_input("Secret Key", value=env_secret if env_secret else "", type="password")
        
        # dynamic update if user inputs
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
    
    # Load portfolio from config or default
    default_portfolio = config.get("portfolio", [
        {"coin": "BTC", "strategy": "SMA", "parameter": 120, "weight": 50, "interval": "day"},
        {"coin": "ETH", "strategy": "SMA", "parameter": 60, "weight": 50, "interval": "day"}
    ])
    
    # Convert to DataFrame for Editor (Use Labels)
    sanitized_portfolio = []
    def_len = len(default_portfolio)
    for p in default_portfolio:
        api_interval = p.get("interval", "day")
        label_interval = INTERVAL_REV_MAP.get(api_interval, "일봉")
        
        # Migrate old 'sma' key to 'parameter' if needed
        param_val = p.get("parameter", p.get("sma", 20))
        
        # Migration: Map old long names to short names
        strat_map = {"SMA Strategy": "SMA", "Donchian Trend": "Donchian"}
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
        st.sidebar.info(f"투자 비중: {total_weight}% | 현금(Cash): {cash_weight}%")
    
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
    
    default_start_str = config.get("start_date", "2025-01-01")
    try:
        default_start = pd.to_datetime(default_start_str).date()
    except:
        default_start = pd.to_datetime("2025-01-01").date()
    start_date = st.sidebar.date_input(
        "기준 시작일 (Start Date)", 
        value=default_start,
        help="수익률 계산 및 이론적 자산 비교를 위한 기준일입니다. 실제 매매 신호와는 무관합니다."
    )

    # Capital Input Customization
    default_cap = config.get("initial_cap", 1000000)
    initial_cap = st.sidebar.number_input(
        "초기 자본금 (KRW - 원 단위)", 
        value=default_cap, step=100000, format="%d",
        help="시뮬레이션을 위한 초기 투자금 설정입니다. 실제 계좌 잔고와는 무관하며, 수익률 계산의 기준이 됩니다."
    )
    st.sidebar.caption(f"Set: **{initial_cap:,.0f} KRW**") # 1. Formatting
    
    # Strategy Selection REMOVED (Moved to Per-Coin)

    PORTFOLIO_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")
    save_col1, save_col2 = st.sidebar.columns(2)

    if save_col1.button("💾 저장"):
        new_config = {
            "portfolio": portfolio_list,
            "start_date": str(start_date),
            "initial_cap": initial_cap
        }
        save_config(new_config)
        with open(PORTFOLIO_JSON, "w", encoding="utf-8") as f:
            json.dump(portfolio_list, f, indent=2, ensure_ascii=False)
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

    # --- Tabs ---
    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Live Portfolio", "📊 Backtest (Single)", "📜 자산 입출금 (History)", "📡 전체 종목 스캔"])

    # --- Tab 1: Live Portfolio (Default) ---
    with tab1:
        st.header("Real-Time Portfolio Dashboard")
        st.caption("Monitoring all configured assets.")
        
        if not trader:
            st.warning("Please enter valid API Keys in the sidebar to enable trading.")
        else:
            # Configure and Start Worker
            worker.update_config(portfolio_list)
            worker.start_worker()
            
            w_msg, w_time = worker.get_status()
            
            # Control Bar
            col_ctrl1, col_ctrl2 = st.columns([1,3])
            with col_ctrl1:
                if st.button("🔄 Refresh View"):
                    st.rerun()
            with col_ctrl2:
                st.info(f"Worker Status: **{w_msg}**")
                
            if not portfolio_list:
                st.warning("Please add coins to your portfolio in the Sidebar.")
            else:
                count = len(portfolio_list)
                per_coin_cap = initial_cap / count
                
                # --- Total Summary Container ---
                st.subheader("🏁 Portfolio Summary")
                st.caption(f"Init Capital: {initial_cap:,.0f} KRW | Assets: {count} | Per Asset: {per_coin_cap:,.0f} KRW")
                
                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                
                total_real_val = trader.get_balance("KRW") 
                total_init_val = initial_cap
                
                # Cash Logic
                total_weight_alloc = sum([item.get('weight', 0) for item in portfolio_list])
                cash_ratio = max(0, 100 - total_weight_alloc) / 100.0
                reserved_cash = initial_cap * cash_ratio
                
                # Add reserved cash to Theo Value (as it stays as cash)
                total_theo_val = reserved_cash
                
                # --- 전체 자산 현황 테이블 ---
                krw_bal_summary = trader.get_balance("KRW")
                asset_summary_rows = [{"자산": "KRW (현금)", "보유량": f"{krw_bal_summary:,.0f}", "현재가": "-", "평가금액(KRW)": f"{krw_bal_summary:,.0f}"}]
                seen_coins_summary = set()
                for s_item in portfolio_list:
                    s_coin = s_item['coin'].upper()
                    if s_coin in seen_coins_summary:
                        continue
                    seen_coins_summary.add(s_coin)
                    s_ticker = f"{s_item['market']}-{s_coin}"
                    s_bal = trader.get_balance(s_coin)
                    s_price = pyupbit.get_current_price(s_ticker) or 0
                    s_val = s_bal * s_price
                    if s_bal > 0 or s_val > 100:
                        asset_summary_rows.append({
                            "자산": s_coin,
                            "보유량": f"{s_bal:.8f}" if s_bal < 1 else f"{s_bal:,.4f}",
                            "현재가": f"{s_price:,.0f}",
                            "평가금액(KRW)": f"{s_val:,.0f}"
                        })
                total_real_summary = krw_bal_summary + sum(
                    trader.get_balance(c) * (pyupbit.get_current_price(f"KRW-{c}") or 0)
                    for c in seen_coins_summary
                )
                asset_summary_rows.append({
                    "자산": "합계",
                    "보유량": "",
                    "현재가": "",
                    "평가금액(KRW)": f"{total_real_summary:,.0f}"
                })
                with st.expander(f"💰 전체 자산 현황 (Total: {total_real_summary:,.0f} KRW)", expanded=True):
                    st.dataframe(pd.DataFrame(asset_summary_rows), use_container_width=True, hide_index=True)

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

                st.write(f"### 📋 Asset Details (Cash Reserve: {reserved_cash:,.0f} KRW)")

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
                                st.warning(f"Waiting for data... ({ticker}, {interval})")
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
                            coin_bal = trader.get_balance(coin_sym)
                            
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
                            c1.metric("Price / SMA", f"{curr_price:,.0f}", delta=f"{curr_price - curr_sma:,.0f}")
                            
                            sig_color = "green" if curr_signal=="BUY" else "red" if curr_signal=="SELL" else "gray"
                            c2.markdown(f"**Signal**: :{sig_color}[{curr_signal}]")
                            if strategy_mode == "Donchian":
                                c2.caption(f"Donch({buy_p}/{sell_p})")
                            else:
                                c2.caption(f"SMA({param_val})")
                            
                            # Asset Performance
                            roi_theo = (expected_eq - per_coin_cap) / per_coin_cap * 100
                            c3.metric(f"Theo Asset", f"{expected_eq:,.0f}", delta=f"{roi_theo:.2f}%")
                            
                            match = (real_status == theo_status)
                            match_color = "green" if match else "red"
                            c4.markdown(f"**Sync**: :{match_color}[{'MATCH' if match else 'DIFF'}]")
                            c4.caption(f"Real: {coin_bal:,.4f} {coin_sym} ({real_status})")
                            
                            st.divider()
                            
                            # --- Tabs for Charts & Orders ---
                            p_tab1, p_tab2 = st.tabs(["📈 Analysis & Benchmark", "🛒 Orders & Execution"])
                            
                            with p_tab1:
                                if "error" not in bt_res:
                                    hist_df = bt_res['df']
                                    start_equity = hist_df['equity'].iloc[0]
                                    start_price = hist_df['close'].iloc[0]

                                    # Normalized Comparison
                                    hist_df['Norm_Strat'] = hist_df['equity'] / start_equity * 100
                                    hist_df['Norm_Bench'] = hist_df['close'] / start_price * 100

                                    fig_comp = go.Figure()
                                    fig_comp.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Norm_Strat'], name='Strategy', line=dict(color='blue')))
                                    fig_comp.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Norm_Bench'], name='Benchmark', line=dict(color='gray', dash='dot')))

                                    # 매수/매도 마커 추가
                                    if perf.get('trades'):
                                        buy_trades = [t for t in perf['trades'] if t['type'] == 'buy']
                                        sell_trades = [t for t in perf['trades'] if t['type'] == 'sell']
                                        if buy_trades:
                                            buy_dates = [t['date'] for t in buy_trades]
                                            buy_vals = [hist_df.loc[d, 'Norm_Strat'] if d in hist_df.index else None for d in buy_dates]
                                            fig_comp.add_trace(go.Scatter(
                                                x=buy_dates, y=buy_vals, mode='markers', name='BUY',
                                                marker=dict(symbol='triangle-up', size=10, color='green')
                                            ))
                                        if sell_trades:
                                            sell_dates = [t['date'] for t in sell_trades]
                                            sell_vals = [hist_df.loc[d, 'Norm_Strat'] if d in hist_df.index else None for d in sell_dates]
                                            fig_comp.add_trace(go.Scatter(
                                                x=sell_dates, y=sell_vals, mode='markers', name='SELL',
                                                marker=dict(symbol='triangle-down', size=10, color='red')
                                            ))

                                    fig_comp.update_layout(height=300, title="Strategy vs Buy/Hold (Normalized)", margin=dict(l=0,r=0,t=30,b=0))
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
                                    st.write("**Orderbook**")
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
                                        st.write("N/A")
                                
                                with o_col2:
                                    st.write("**Manual Execution**")
                                    if st.button(f"Check Trade Logic ({item['coin']})", key=f"btn_{ticker}_{asset_idx}"):
                                        res = trader.check_and_trade(ticker, interval=interval, sma_period=param_val)
                                        st.info(res)

                        except Exception as e:
                            st.error(f"Error processing {ticker}: {e}")
                
                # --- Populate Total Summary ---
                total_roi = (total_theo_val - total_init_val) / total_init_val * 100 if total_init_val else 0
                real_roi = (total_real_val - total_init_val) / total_init_val * 100 if total_init_val else 0
                diff_val = total_real_val - total_theo_val

                sum_col1.metric("Initial Capital", f"{total_init_val:,.0f} KRW")
                sum_col2.metric("Total Theo Equity", f"{total_theo_val:,.0f} KRW", delta=f"{total_roi:.2f}%")
                sum_col3.metric("Total Real Assets", f"{total_real_val:,.0f} KRW", delta=f"{real_roi:.2f}%")
                sum_col4.metric("Difference (Real-Theo)", f"{diff_val:,.0f} KRW", delta_color="off" if abs(diff_val)<1000 else "inverse")

                # --- 합산 포트폴리오 성과 (Combined Portfolio) → 위에 예약한 container에 렌더링 ---
                if portfolio_equity_data:
                    with combined_portfolio_container:
                        with st.expander("📊 합산 포트폴리오 성과 (Combined Portfolio)", expanded=True):
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
                            pm1.metric("Total Return", f"{port_return:.2f}%")
                            pm2.metric("CAGR", f"{port_cagr:.2f}%")
                            pm3.metric("MDD", f"{port_mdd:.2f}%")
                            pm4.metric("Sharpe", f"{port_sharpe:.2f}")
                            pm5.metric("vs Buy&Hold", f"{port_return - bench_return:+.2f}%p")

                            st.caption(f"기간: {total_eq.index[0].strftime('%Y-%m-%d')} ~ {total_eq.index[-1].strftime('%Y-%m-%d')} ({port_days}일) | 초기자금: {port_init:,.0f} → 최종: {port_final:,.0f} KRW")

                            # 합산 차트
                            fig_port = go.Figure()
                            fig_port.add_trace(go.Scatter(
                                x=norm_eq.index, y=norm_eq.values,
                                name='Portfolio (Strategy)', line=dict(color='blue', width=2)
                            ))
                            fig_port.add_trace(go.Scatter(
                                x=norm_bench.index, y=norm_bench.values,
                                name='Portfolio (Buy & Hold)', line=dict(color='gray', dash='dot')
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
                                        x=buy_dates_valid, y=buy_vals, mode='markers', name='BUY',
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
                                        x=sell_dates_valid, y=sell_vals, mode='markers', name='SELL',
                                        marker=dict(symbol='triangle-down', size=8, color='red', opacity=0.7)
                                    ))

                            fig_port.update_layout(
                                height=350,
                                title="Combined Portfolio: Strategy vs Buy & Hold (Normalized)",
                                yaxis_title="Normalized (%)",
                                margin=dict(l=0, r=0, t=30, b=0),
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig_port, use_container_width=True)

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
                                    name='Cash Reserve', stackgroup='one',
                                    line=dict(color='lightgray')
                                ))
                            fig_stack.update_layout(
                                height=350,
                                title="Asset Contribution (Stacked)",
                                yaxis_title="KRW",
                                margin=dict(l=0, r=0, t=30, b=0),
                                hovermode='x unified'
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
                    krw_balance = trader.get_balance("KRW")

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

                        rb_coin_bal = trader.get_balance(rb_coin)
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
        st.header("Single Asset Backtest")
        
        # Select ticker from portfolio for convenience, or custom
        # Top 20 Market Cap (Approx Static List)
        TOP_20_TICKERS = [
            "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE", 
            "KRW-ADA", "KRW-SHIB", "KRW-TRX", "KRW-AVAX", "KRW-LINK", 
            "KRW-BCH", "KRW-DOT", "KRW-NEAR", "KRW-MATIC", "KRW-ETC", 
            "KRW-XLM", "KRW-STX", "KRW-WAVES", "KRW-EOS", "KRW-SAND"
        ]
        
        port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
        
        # Merge and Remove Duplicates
        base_options = list(dict.fromkeys(port_tickers + TOP_20_TICKERS))
        
        # --- Strategy Selection (Top) ---
        bt_strategy = st.selectbox(
            "전략 선택 (Strategy)",
            ["SMA Strategy", "Donchian Strategy"],
            index=0,
            key="bt_strategy_sel"
        )

        selected_ticker_bt = st.selectbox("백테스트 대상 (Target)", base_options + ["Custom"])

        bt_ticker = ""
        bt_sma = 0
        bt_buy_period = 20
        bt_sell_period = 10

        if selected_ticker_bt == "Custom":
            c1, c2 = st.columns(2)
            c = c2.text_input("Coin", "BTC", key="bt_c")
            bt_ticker = f"KRW-{c.upper()}"
        else:
            bt_ticker = selected_ticker_bt

        # --- Strategy-specific Parameters ---
        if bt_strategy == "SMA Strategy":
            item = next((item for item in portfolio_list if f"{item['market']}-{item['coin'].upper()}" == bt_ticker), None)
            default_sma = item.get('parameter', 60) if item else 60
            bt_sma = st.number_input("단기 SMA (추세)", value=default_sma, key="bt_sma_select", min_value=5, step=1)
        else:  # Donchian Strategy
            dc_col1, dc_col2 = st.columns(2)
            with dc_col1:
                bt_buy_period = st.number_input("매수 채널 기간 (Buy Period)", value=20, min_value=5, max_value=300, step=1, key="bt_dc_buy", help="N일 고가 돌파 시 매수")
            with dc_col2:
                bt_sell_period = st.number_input("매도 채널 기간 (Sell Period)", value=10, min_value=5, max_value=300, step=1, key="bt_dc_sell", help="N일 저가 이탈 시 매도")

        # Backtest Interval Selection
        bt_interval_label = st.selectbox("시간봉 선택 (Interval)", options=list(INTERVAL_MAP.keys()), index=0, key="bt_interval_sel")
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
            st.caption("백테스트 기간 (Period)")
            d_col1, d_col2 = st.columns(2)

            # Default Backtest Start: 2020-01-01
            try:
                default_start_bt = datetime(2020, 1, 1).date()
            except:
                default_start_bt = datetime.now().date() - timedelta(days=365)
            default_end_bt = datetime.now().date()

            bt_start = d_col1.date_input(
                "시작일 (Start)",
                value=default_start_bt,
                max_value=datetime.now().date(),
                format="YYYY.MM.DD"
            )

            bt_end = d_col2.date_input(
                "종료일 (End)",
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

            run_btn = st.button("Run Backtest", type="primary")

        if run_btn:
            # Determine period for data fetch buffer
            if bt_strategy == "Donchian Strategy":
                req_period = max(bt_buy_period, bt_sell_period)
                bt_strategy_mode = "Donchian"
                bt_sell_ratio = bt_sell_period / bt_buy_period if bt_buy_period > 0 else 0.5
            else:
                req_period = bt_sma
                bt_strategy_mode = "SMA Strategy"
                bt_sell_ratio = 0.5

            to_date = bt_end + timedelta(days=1)
            to_str = to_date.strftime("%Y-%m-%d 09:00:00")

            cpd = CANDLES_PER_DAY.get(bt_interval, 1)
            req_count = days_diff * cpd + req_period + 300
            fetch_count = max(req_count, req_period + 300)

            with st.spinner(f"Running Backtest ({bt_start} ~ {bt_end}, {bt_interval_label}, {bt_strategy})..."):
                df_bt = pyupbit.get_ohlcv(bt_ticker, interval=bt_interval, to=to_str, count=fetch_count)

                if df_bt is None or df_bt.empty:
                    st.error("No data fetched.")
                    st.stop()

                # Data range validation
                data_start = df_bt.index[0]
                data_end = df_bt.index[-1]
                st.caption(f"Fetched {len(df_bt)} candles: {data_start.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")

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
                    m1.metric("Total Return", f"{res['total_return']:,.2f}%")
                    m2.metric("CAGR", f"{res.get('cagr', 0):,.2f}%")
                    m3.metric("Win Rate", f"{res['win_rate']:,.2f}%")
                    m4.metric("MDD", f"{res['mdd']:,.2f}%")
                    m5.metric("Sharpe", f"{res['sharpe']:.2f}")

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
                                f"Slippage Impact: 슬리피지 {bt_slippage}% 적용 시 "
                                f"수익률 차이 **{slip_ret_diff:,.2f}%p**, "
                                f"금액 차이 **{slip_cost:,.0f} KRW** "
                                f"(슬리피지 없는 경우 {res_ns['final_equity']:,.0f} KRW)"
                            )
                    
                    # --- Combined Chart ---
                    st.subheader("Price & Strategy Performance")

                    
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
                        name='Price'
                    ), row=1, col=1, secondary_y=False)
                    
                    # 2. Strategy Indicator Lines - Row 1, Primary Y
                    if bt_strategy_mode == "Donchian":
                        upper_col = f'Donchian_Upper_{bt_buy_period}'
                        lower_col = f'Donchian_Lower_{bt_sell_period}'
                        if upper_col in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[upper_col],
                                line=dict(color='green', width=1.5, dash='dash'),
                                name=f'Upper ({bt_buy_period})'
                            ), row=1, col=1, secondary_y=False)
                        if lower_col in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[lower_col],
                                line=dict(color='red', width=1.5, dash='dash'),
                                name=f'Lower ({bt_sell_period})'
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
                        name='Strategy Equity'
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
                            name='Buy Signal'
                        ), row=1, col=1, secondary_y=False)

                    if sell_dates:
                        fig.add_trace(go.Scatter(
                            x=sell_dates, y=sell_prices,
                            mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
                            name='Sell Signal'
                        ), row=1, col=1, secondary_y=False)
                        
                    # 5. Drawdown - Row 2
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df['drawdown'],
                        name='Drawdown (%)',
                        fill='tozeroy',
                        line=dict(color='red', width=1)
                    ), row=2, col=1)

                    fig.update_layout(height=800, title_text="Backtest Results", xaxis_rangeslider_visible=False)
                    fig.update_yaxes(title_text="Price (KRW)", row=1, col=1, secondary_y=False)
                    fig.update_yaxes(title_text="Equity (KRW)", row=1, col=1, secondary_y=True)
                    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Yearly Performance Table
                    if 'yearly_stats' in res:
                        st.subheader("📊 Yearly Performance")
                        st.dataframe(res['yearly_stats'].style.format("{:.2f}%"))
                        
                    st.info(f"Strategy Status: **{res['final_status']}** | Next Action: **{res['next_action'] if res['next_action'] else 'None'}**")
                    
                    # Trade List
                    with st.expander("Trade Log"):
                        if res['trades']:
                            trades_df = pd.DataFrame(res['trades'])
                            st.dataframe(trades_df.style.format({"price": "{:,.2f}", "amount": "{:,.6f}", "balance": "{:,.2f}", "profit": "{:,.2f}%"}))
                        else:
                            st.info("No trades executed.")
                            
                    # Export Full Daily Log
                    csv_data = df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 Download Daily Log (Full Data)",
                        data=csv_data,
                        file_name=f"{bt_ticker}_{bt_start}_daily_log.csv",
                        mime="text/csv",
                        help="Download daily OHLCV + Indicators + Signals to verify logic."
                    )

        # --- Optimization Section (Fragment: prevents full page dimming) ---
        @st.fragment
        def optimization_section():
            st.divider()
            st.subheader("🛠️ 파라미터 최적화 (Parameter Optimization)")

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
                # 공통: 시간봉 선택
                opt_interval_label = st.selectbox(
                    "시간봉 (Interval)", options=list(INTERVAL_MAP.keys()),
                    index=0, key="opt_interval_sel"
                )
                opt_interval = INTERVAL_MAP[opt_interval_label]

                if bt_strategy == "Donchian Strategy":
                    st.caption("돈치안 채널의 매수 기간(Buy Period)과 매도 기간(Sell Period)을 최적화합니다.")

                    st.markdown("##### 1. 매수 채널 기간 (Buy Period)")
                    c1, c2, c3 = st.columns(3)
                    opt_buy_start = c1.number_input("Start", 5, 200, 10, key="opt_dc_buy_start")
                    opt_buy_end = c2.number_input("End", 5, 200, 60, key="opt_dc_buy_end")
                    opt_buy_step = c3.number_input("Step", 1, 50, 5, key="opt_dc_buy_step")

                    st.markdown("##### 2. 매도 채널 기간 (Sell Period)")
                    c1, c2, c3 = st.columns(3)
                    opt_sell_start = c1.number_input("Start", 5, 200, 5, key="opt_dc_sell_start")
                    opt_sell_end = c2.number_input("End", 5, 200, 30, key="opt_dc_sell_end")
                    opt_sell_step = c3.number_input("Step", 1, 50, 5, key="opt_dc_sell_step")

                else:  # SMA Strategy
                    st.caption("SMA 이동평균 기간을 최적화합니다.")

                    st.markdown("##### SMA 기간 (Period)")
                    c1, c2, c3 = st.columns(3)
                    opt_s_start = c1.number_input("Start", 5, 200, 20, key="opt_s_start")
                    opt_s_end = c2.number_input("End", 5, 200, 60, key="opt_s_end")
                    opt_s_step = c3.number_input("Step", 1, 50, 5, key="opt_s_step")

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
                    if bt_strategy == "Donchian Strategy":
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

                    # --- Phase 2: 고속 최적화 ---
                    def opt_progress(idx, total, msg):
                        pct = 0.3 + (idx / total) * 0.7  # 30%~100%
                        progress_bar.progress(min(pct, 1.0))
                        log_area.text(f"{msg} ({idx}/{total} · {idx/total*100:.0f}%)")

                    st.write(f"🚀 총 {total_iter}개 조합 고속 최적화 시작...")
                    t1 = _time.time()

                    if bt_strategy == "Donchian Strategy":
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
                                "Win Rate (%)": r["win_rate"],
                                "Sharpe": r["sharpe"],
                                "Trades": r["trade_count"]
                            })

                    opt_elapsed = _time.time() - t1
                    total_elapsed = _time.time() - t0

                    status.update(label=f"✅ 완료! ({total_iter}개, 다운로드 {dl_elapsed:.1f}초 + 최적화 {opt_elapsed:.1f}초 = 총 {total_elapsed:.1f}초)", state="complete")

                except Exception as e:
                    status.update(label=f"❌ 오류: {e}", state="error")
                    import traceback
                    st.code(traceback.format_exc())
                    return

            # --- Results Display (outside st.status) ---
            if not results:
                st.warning("No results found.")
                return

            opt_df = pd.DataFrame(results)
            best_idx = opt_df['Total Return (%)'].idxmax()
            best_row = opt_df.loc[best_idx]

            if bt_strategy == "Donchian Strategy":
                st.subheader("🏆 Best Result")
                st.success(f"Best Return: **{best_row['Total Return (%)']:.2f}%** (Buy: {int(best_row['Buy Period'])}, Sell: {int(best_row['Sell Period'])})")

                st.dataframe(opt_df.sort_values(by="Total Return (%)", ascending=False).style.background_gradient(cmap='RdYlGn', subset=['Total Return (%)', 'Sharpe']).format("{:,.2f}"), use_container_width=True)

                fig_opt = px.density_heatmap(
                    opt_df, x="Buy Period", y="Sell Period", z="Total Return (%)",
                    histfunc="avg", title="Donchian Optimization Heatmap (Return %)",
                    text_auto=".0f", color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_opt, use_container_width=True)
            else:
                st.subheader("🏆 Best Result")
                st.success(f"Best Return: **{best_row['Total Return (%)']:.2f}%** (SMA: {int(best_row['SMA Period'])})")

                st.dataframe(opt_df.sort_values(by="Total Return (%)", ascending=False).style.background_gradient(cmap='RdYlGn', subset=['Total Return (%)', 'Sharpe']).format("{:,.2f}"), use_container_width=True)

                st.line_chart(opt_df.set_index("SMA Period")[['Total Return (%)', 'MDD (%)']])

        optimization_section()

    # --- Tab 3: History ---
    with tab3:
        st.header("Trade Logs & Money Management")
        
        hist_tab1, hist_tab2, hist_tab3 = st.tabs(["🧪 Virtual Logs (Backtest/Paper)", "💸 Real Logs (Exchange)", "📊 Slippage Analysis"])
        
        with hist_tab1:
            st.subheader("Virtual Account Management")
            
            # Simulated Deposit/Withdraw
            if 'virtual_adjustment' not in st.session_state:
                st.session_state.virtual_adjustment = 0
            
            c1, c2 = st.columns(2)
            amount = c1.number_input("Amount (KRW)", step=100000)
            if c2.button("Deposit/Withdraw (Virtual)"):
                st.session_state.virtual_adjustment += amount
                st.success(f"Adjusted Virtual Balance by {amount:,.0f} KRW")
            
            st.info(f"Cumulative Virtual Adjustment: {st.session_state.virtual_adjustment:,.0f} KRW")
            st.write("To view strategy logs, run the Backtest in Tab 1 or check individual assets in Tab 2.")

        with hist_tab2:
            st.subheader("Real Operation Logs")
            
            if not trader:
                st.warning("Please configure API Keys first.")
            else:
                c_h1, c_h2 = st.columns(2)
                h_type = c_h1.selectbox("조회 유형 (Type)", ["Executed Orders", "Deposits", "Withdrawals"])
                h_curr = c_h2.selectbox("화폐 (Currency)", ["KRW", "USDT", "BTC", "ETH", "XRP"])
                
                if st.button("Fetch Real History"):
                    with st.spinner("Fetching data from Upbit..."):
                        if h_type == "Executed Orders":
                            # Use new method (Order history usually ignores currency or uses distinct method)
                            # UpbitTrader.get_history('order') does not use currency currently.
                            data = fetch_history_cached(trader, 'order')
                            if data:
                                df = pd.DataFrame(data)
                                st.dataframe(df)
                            else:
                                st.info("No recent orders found.")
                        elif h_type == "Deposits":
                            data = fetch_history_cached(trader, 'deposit', h_curr)
                            if data:
                                st.dataframe(pd.DataFrame(data))
                            else:
                                st.info(f"No recent deposits found for {h_curr}.")
                        elif h_type == "Withdrawals":
                            data = fetch_history_cached(trader, 'withdraw', h_curr)
                            if data:
                                st.dataframe(pd.DataFrame(data))
                            else:
                                st.info(f"No recent withdrawals found for {h_curr}.")
            
            st.caption("Data fetches are cached for 60 seconds.")

        with hist_tab3:
            st.subheader("Slippage Analysis (실제 체결 vs 백테스트)")

            if not trader:
                st.warning("API Key가 필요합니다.")
            else:
                sa_col1, sa_col2 = st.columns(2)
                sa_ticker_list = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
                sa_ticker = sa_col1.selectbox("코인 선택", sa_ticker_list, key="sa_ticker")
                sa_interval = sa_col2.selectbox("시간봉", list(INTERVAL_MAP.keys()), key="sa_interval")

                if st.button("Analyze Slippage", key="sa_run"):
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
                                        name='Slippage %'
                                    ))
                                    fig_slip.add_hline(y=avg_slip, line_dash="dash", line_color="blue",
                                                       annotation_text=f"Avg: {avg_slip:.3f}%")
                                    fig_slip.update_layout(title="Trade Slippage (+ = Unfavorable)", height=350)
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
        scan_top_n = scan_col5.number_input("상위 종목 수", 5, 50, 20, key="scan_top_n")

        sell_ratio = 0.5
        if scan_strategy == "Donchian":
            sell_ratio = st.slider("매도 채널 비율", 0.1, 1.0, 0.5, 0.1, key="scan_sell_ratio")

        if st.button("🔍 스캔 시작", key="scan_run", type="primary"):
            engine = BacktestEngine()

            with st.spinner("상위 종목 조회 중..."):
                # Upbit API로 거래대금 상위 종목 조회
                try:
                    all_krw_tickers = pyupbit.get_tickers(fiat="KRW")
                    url = "https://api.upbit.com/v1/ticker"
                    resp = requests.get(url, params={"markets": ",".join(all_krw_tickers)}, timeout=10)
                    ticker_data = resp.json()
                    # 24h 거래대금 기준 정렬
                    ticker_data.sort(key=lambda x: float(x.get('acc_trade_price_24h', 0)), reverse=True)
                    top_tickers = [t['market'] for t in ticker_data[:scan_top_n]]
                except Exception as e:
                    st.error(f"종목 조회 실패: {e}")
                    top_tickers = []

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
