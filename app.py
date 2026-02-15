import streamlit as st
import pyupbit
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import json # Added import

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
        min-width: 500px !important;
        max-width: 800px !important;
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

        sanitized_portfolio.append({
            "coin": str(p.get("coin", "BTC")).upper(),
            "strategy": strat_val,
            "parameter": param_val,
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
                                                  "parameter": st.column_config.NumberColumn("기간", min_value=5, max_value=300, step=1, required=True),
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
        
        portfolio_list.append({
            "market": "KRW",
            "coin": r['coin'].upper(),
            "strategy": r['strategy'],
            "parameter": r['parameter'],
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

    if st.sidebar.button("💾 설정 저장"):
        new_config = {
            "portfolio": portfolio_list,
            "start_date": str(start_date),
            "initial_cap": initial_cap
        }
        save_config(new_config)
        st.sidebar.success("저장되었습니다!")

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
    tab1, tab2, tab3 = st.tabs(["🚀 Live Portfolio", "📊 Backtest (Single)", "📜 자산 입출금 (History)"])

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
                
                st.write(f"### 📋 Asset Details (Cash Reserve: {reserved_cash:,.0f} KRW)")
                
                for item in portfolio_list:
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
                                sell_p = max(5, buy_p // 2)
                                
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
                            
                            # 3. Theo Backtest (Sync Check) with WORKER DF
                            req_p = param_val
                            req_count = max(365, req_p * 2)
                            bt_res = backtest_engine.run_backtest(ticker, period=param_val, interval=interval, count=req_count, start_date=start_date, initial_balance=per_coin_cap, df=df_curr, strategy_mode=strategy_mode, sell_period_ratio=0.5)
                            
                            expected_eq = 0
                            theo_status = "UNKNOWN"
                            
                            if "error" not in bt_res:
                                perf = bt_res['performance']
                                theo_status = perf['final_status']
                                expected_eq = perf['final_equity']
                                total_theo_val += expected_eq
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
                                    fig_comp.update_layout(height=300, title="Strategy vs Buy/Hold (Normalized)", margin=dict(l=0,r=0,t=30,b=0))
                                    st.plotly_chart(fig_comp, use_container_width=True)
                            
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
                                    if st.button(f"Check Trade Logic ({item['coin']})", key=f"btn_{ticker}"):
                                        res = trader.check_and_trade(ticker, interval=interval, sma_period=sma_p)
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
            
            from datetime import timedelta # Ensure import
            days_diff = (bt_end - bt_start).days
            
            st.caption(f"기간: {days_diff}일")
            
            fee = st.number_input("매매 수수료 (%)", value=0.05, format="%.2f") / 100
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

            req_count = days_diff + req_period + 300
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
                    sell_period_ratio=bt_sell_ratio
                )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    df = result["df"]
                    res = result["performance"]
                    
                    # Metrics Row
                    # Metrics Row
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Return", f"{res['total_return']:,.2f}%")
                    m2.metric("CAGR", f"{res.get('cagr', 0):,.2f}%")
                    m3.metric("Win Rate", f"{res['win_rate']:,.2f}%")
                    m4.metric("MDD", f"{res['mdd']:,.2f}%")
                    m5.metric("Sharpe", f"{res['sharpe']:.2f}")
                    
                    st.success(f"Expected Final Balance: {res['final_equity']:,.0f} KRW (from {initial_cap:,.0f} KRW)")
                    
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

        st.divider()
        st.subheader("🛠️ 파라미터 최적화 (Parameter Optimization)")

        with st.form("optimization_form"):
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
                st.caption("SMA 기간을 최적화합니다. Start = End 이면 고정값으로 처리됩니다.")

                st.markdown("##### 1. 단기 SMA (Short SMA)")
                c1, c2, c3 = st.columns(3)
                opt_s_start = c1.number_input("Start", 5, 200, 20, key="opt_s_start")
                opt_s_end = c2.number_input("End", 5, 200, 60, key="opt_s_end")
                opt_s_step = c3.number_input("Step", 1, 50, 5, key="opt_s_step")

                st.markdown("##### 2. 장기 SMA (Long SMA)")
                st.caption("Start = End = 0 이면 사용 안함")
                c1, c2, c3 = st.columns(3)
                opt_l_start = c1.number_input("Start", 0, 300, 0, key="opt_l_start")
                opt_l_end = c2.number_input("End", 0, 500, 0, key="opt_l_end")
                opt_l_step = c3.number_input("Step", 1, 50, 10, key="opt_l_step")

            opt_submitted = st.form_submit_button("Start Optimization", type="primary")

        if opt_submitted:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            try:
                if bt_strategy == "Donchian Strategy":
                    # --- Donchian Optimization ---
                    buy_range = range(opt_buy_start, opt_buy_end + 1, opt_buy_step)
                    sell_range = range(opt_sell_start, opt_sell_end + 1, opt_sell_step)
                    total_iter = len(buy_range) * len(sell_range)

                    max_req_p = max(opt_buy_end, opt_sell_end)
                    fetch_count_opt = days_diff + max_req_p + 300

                    to_date_opt = bt_end + timedelta(days=1)
                    to_str_opt = to_date_opt.strftime("%Y-%m-%d 09:00:00")
                    full_df = pyupbit.get_ohlcv(bt_ticker, interval=bt_interval, to=to_str_opt, count=fetch_count_opt)

                    if full_df is None:
                        st.error("Failed to fetch data for optimization")
                    else:
                        idx = 0
                        for bp in buy_range:
                            for sp in sell_range:
                                idx += 1
                                status_text.text(f"Running... Buy: {bp}, Sell: {sp} ({idx}/{total_iter})")

                                sell_ratio = sp / bp if bp > 0 else 0.5
                                res_opt = backtest_engine.run_backtest(
                                    bt_ticker,
                                    period=bp,
                                    interval=bt_interval,
                                    count=fetch_count_opt,
                                    fee=fee,
                                    start_date=bt_start,
                                    initial_balance=initial_cap,
                                    df=full_df,
                                    strategy_mode="Donchian",
                                    sell_period_ratio=sell_ratio
                                )

                                if "error" not in res_opt:
                                    perf = res_opt['performance']
                                    results.append({
                                        "Buy Period": bp,
                                        "Sell Period": sp,
                                        "Total Return (%)": perf['total_return'],
                                        "CAGR (%)": perf.get('cagr', 0),
                                        "MDD (%)": perf['mdd'],
                                        "Win Rate (%)": perf['win_rate'],
                                        "Sharpe": perf['sharpe'],
                                        "Trades": perf['trade_count']
                                    })

                                progress_bar.progress(idx / total_iter)

                        status_text.text("Optimization Completed!")

                        if results:
                            opt_df = pd.DataFrame(results)
                            best_idx = opt_df['Total Return (%)'].idxmax()
                            best_row = opt_df.loc[best_idx]

                            st.subheader("🏆 Best Result")
                            st.success(f"Best Return: **{best_row['Total Return (%)']:.2f}%** (Buy: {int(best_row['Buy Period'])}, Sell: {int(best_row['Sell Period'])})")

                            st.dataframe(opt_df.sort_values(by="Total Return (%)", ascending=False).style.background_gradient(cmap='RdYlGn', subset=['Total Return (%)', 'Sharpe']).format("{:,.2f}"), use_container_width=True)

                            import plotly.express as px
                            fig_opt = px.density_heatmap(
                                opt_df,
                                x="Buy Period",
                                y="Sell Period",
                                z="Total Return (%)",
                                histfunc="avg",
                                title="Donchian Optimization Heatmap (Return %)",
                                text_auto=".0f",
                                color_continuous_scale="RdYlGn"
                            )
                            st.plotly_chart(fig_opt, use_container_width=True)
                        else:
                            st.warning("No results found.")

                else:
                    # --- SMA Optimization ---
                    short_range = range(opt_s_start, opt_s_end + 1, opt_s_step)
                    long_range = range(opt_l_start, opt_l_end + 1, opt_l_step) if opt_l_end > opt_l_start else [opt_l_start]
                    total_iter = len(short_range) * len(long_range)

                    max_req_p = max(max(short_range), max(long_range)) if max(long_range) > 0 else max(short_range)
                    fetch_count_opt = days_diff + max_req_p + 300

                    to_date_opt = bt_end + timedelta(days=1)
                    to_str_opt = to_date_opt.strftime("%Y-%m-%d 09:00:00")
                    full_df = pyupbit.get_ohlcv(bt_ticker, interval=bt_interval, to=to_str_opt, count=fetch_count_opt)

                    if full_df is None:
                        st.error("Failed to fetch data for optimization")
                    else:
                        idx = 0
                        for s_p in short_range:
                            for l_p in long_range:
                                idx += 1
                                status_text.text(f"Running... Short: {s_p}, Long: {l_p} ({idx}/{total_iter})")

                                res_opt = backtest_engine.run_backtest(
                                    bt_ticker,
                                    period=s_p,
                                    interval=bt_interval,
                                    count=fetch_count_opt,
                                    fee=fee,
                                    start_date=bt_start,
                                    initial_balance=initial_cap,
                                    df=full_df,
                                    strategy_mode="SMA Strategy",
                                    sell_period_ratio=0.5
                                )

                                if "error" not in res_opt:
                                    perf = res_opt['performance']
                                    results.append({
                                        "Short SMA": s_p,
                                        "Long SMA": l_p,
                                        "Total Return (%)": perf['total_return'],
                                        "CAGR (%)": perf.get('cagr', 0),
                                        "MDD (%)": perf['mdd'],
                                        "Win Rate (%)": perf['win_rate'],
                                        "Sharpe": perf['sharpe'],
                                        "Trades": perf['trade_count']
                                    })

                                progress_bar.progress(idx / total_iter)

                        status_text.text("Optimization Completed!")

                        if results:
                            opt_df = pd.DataFrame(results)
                            best_idx = opt_df['Total Return (%)'].idxmax()
                            best_row = opt_df.loc[best_idx]

                            st.subheader("🏆 Best Result")
                            st.success(f"Best Return: **{best_row['Total Return (%)']:.2f}%** (Short: {int(best_row['Short SMA'])}, Long: {int(best_row['Long SMA'])})")

                            st.dataframe(opt_df.sort_values(by="Total Return (%)", ascending=False).style.background_gradient(cmap='RdYlGn', subset=['Total Return (%)', 'Sharpe']).format("{:,.2f}"), use_container_width=True)

                            short_optimized = len(short_range) > 1
                            long_optimized = len(long_range) > 1
                            if short_optimized and long_optimized:
                                import plotly.express as px
                                fig_opt = px.density_heatmap(
                                    opt_df,
                                    x="Short SMA",
                                    y="Long SMA",
                                    z="Total Return (%)",
                                    histfunc="avg",
                                    title="SMA Optimization Heatmap (Return %)",
                                    text_auto=".0f",
                                    color_continuous_scale="RdYlGn"
                                )
                                st.plotly_chart(fig_opt, use_container_width=True)
                            elif short_optimized:
                                st.line_chart(opt_df.set_index("Short SMA")[['Total Return (%)', 'MDD (%)']])
                            elif long_optimized:
                                st.line_chart(opt_df.set_index("Long SMA")[['Total Return (%)', 'MDD (%)']])
                        else:
                            st.warning("No results found.")

            except Exception as e:
                st.error(f"Optimization Error: {e}")

    # --- Tab 3: History ---
    with tab3:
        st.header("Trade Logs & Money Management")
        
        hist_tab1, hist_tab2 = st.tabs(["🧪 Virtual Logs (Backtest/Paper)", "💸 Real Logs (Exchange)"])
        
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

if __name__ == "__main__":
    main()
