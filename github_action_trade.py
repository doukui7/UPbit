import os
import sys
import json
import logging
import pyupbit
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.sma import SMAStrategy
from strategy.donchian import DonchianStrategy
from trading.upbit_trader import UpbitTrader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

MIN_ORDER_KRW = 5000


def get_portfolio():
    """Load portfolio from PORTFOLIO env var (JSON) or fallback to defaults."""
    raw = os.getenv("PORTFOLIO")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("PORTFOLIO env var is not valid JSON. Using default.")

    # Fallback: single coin from legacy env vars
    return [{
        "market": "KRW",
        "coin": os.getenv("TARGET_TICKER", "BTC"),
        "strategy": os.getenv("STRATEGY", "SMA"),
        "parameter": int(os.getenv("SMA_PERIOD", "20")),
        "weight": 100,
        "interval": os.getenv("INTERVAL", "day")
    }]


def trade_single_asset(trader, item, total_capital):
    """Process a single asset: fetch data, generate signal, execute trade."""
    ticker = f"{item['market']}-{item['coin'].upper()}"
    strategy_name = item.get("strategy", "SMA")
    param = item.get("parameter", 20)
    interval = item.get("interval", "day")
    weight = item.get("weight", 100)
    allocated_cap = total_capital * (weight / 100.0)

    logger.info(f"--- [{ticker}] {strategy_name}({param}), {weight}%, {interval} ---")

    # Fetch data
    count = max(200, param * 3)
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    if df is None or len(df) < param + 5:
        logger.error(f"[{ticker}] Insufficient data (got {len(df) if df is not None else 0}, need {param + 5})")
        return

    # Generate signal
    last_candle = df.iloc[-2]  # Last completed candle

    if strategy_name == "Donchian":
        strat = DonchianStrategy()
        buy_p = param
        sell_p = max(5, buy_p // 2)
        df = strat.create_features(df, buy_period=buy_p, sell_period=sell_p)
        last_candle = df.iloc[-2]
        signal = strat.get_signal(last_candle, buy_period=buy_p, sell_period=sell_p)
        indicator_info = f"Upper={last_candle.get(f'Donchian_Upper_{buy_p}', 'N/A')}, Lower={last_candle.get(f'Donchian_Lower_{sell_p}', 'N/A')}"
    else:
        strat = SMAStrategy()
        df = strat.create_features(df, periods=[param])
        last_candle = df.iloc[-2]
        signal = strat.get_signal(last_candle, strategy_type='SMA_CROSS', ma_period=param)
        indicator_info = f"SMA_{param}={last_candle.get(f'SMA_{param}', 'N/A')}"

    current_price = pyupbit.get_current_price(ticker)
    logger.info(f"[{ticker}] Close={last_candle['close']}, {indicator_info}, Signal={signal}, Price={current_price}")

    # Check balances
    coin_sym = item['coin'].upper()
    krw_balance = trader.get_balance("KRW")
    coin_balance = trader.get_balance(coin_sym)
    coin_value = coin_balance * current_price if current_price else 0

    logger.info(f"[{ticker}] KRW={krw_balance:,.0f}, {coin_sym}={coin_balance:.6f} (≈{coin_value:,.0f} KRW)")

    # Execute
    if signal == 'BUY':
        buy_budget = min(krw_balance, allocated_cap) * 0.999
        if buy_budget > MIN_ORDER_KRW:
            logger.info(f"[{ticker}] BUY {buy_budget:,.0f} KRW")
            result = trader.buy_market(ticker, buy_budget)
            logger.info(f"[{ticker}] Buy Result: {result}")
        else:
            logger.info(f"[{ticker}] BUY signal but insufficient KRW ({buy_budget:,.0f})")

    elif signal == 'SELL':
        if coin_value > MIN_ORDER_KRW:
            logger.info(f"[{ticker}] SELL {coin_balance:.6f} {coin_sym}")
            result = trader.sell_market(ticker, coin_balance)
            logger.info(f"[{ticker}] Sell Result: {result}")
        else:
            logger.info(f"[{ticker}] SELL signal but coin value too low ({coin_value:,.0f})")

    else:
        logger.info(f"[{ticker}] HOLD - no action")


def run_auto_trade():
    load_dotenv()

    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("API Keys not found. Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY.")
        return

    trader = UpbitTrader(ACCESS_KEY, SECRET_KEY)
    portfolio = get_portfolio()

    # Total capital = current KRW + all coin values
    krw_bal = trader.get_balance("KRW")
    total_capital = krw_bal
    for item in portfolio:
        coin_sym = item['coin'].upper()
        ticker = f"{item['market']}-{coin_sym}"
        cb = trader.get_balance(coin_sym)
        cp = pyupbit.get_current_price(ticker) or 0
        total_capital += cb * cp

    logger.info(f"=== Portfolio Auto Trade ===")
    logger.info(f"Total Capital: {total_capital:,.0f} KRW | Assets: {len(portfolio)}")

    for item in portfolio:
        try:
            trade_single_asset(trader, item, total_capital)
        except Exception as e:
            logger.error(f"Error processing {item.get('coin', '?')}: {e}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    run_auto_trade()
