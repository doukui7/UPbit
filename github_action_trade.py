import os
import sys
import json
import time
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


def analyze_asset(trader, item):
    """1단계: 데이터 조회 → 시그널 계산 → 보유 현황 확인"""
    ticker = f"{item['market']}-{item['coin'].upper()}"
    strategy_name = item.get("strategy", "SMA")
    param = item.get("parameter", 20)
    interval = item.get("interval", "day")
    weight = item.get("weight", 100)

    logger.info(f"--- [{ticker}] {strategy_name}({param}), {weight}%, {interval} ---")

    # Fetch data
    count = max(200, param * 3)
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    if df is None or len(df) < param + 5:
        logger.error(f"[{ticker}] Insufficient data (got {len(df) if df is not None else 0}, need {param + 5})")
        return None

    # Generate signal
    last_candle = df.iloc[-2]  # Last completed candle

    if strategy_name == "Donchian":
        strat = DonchianStrategy()
        buy_p = param
        sell_p = item.get("sell_parameter", 0) or max(5, buy_p // 2)
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
    coin_sym = item['coin'].upper()
    coin_balance = trader.get_balance(coin_sym)
    coin_value = coin_balance * current_price if current_price else 0
    is_holding = coin_value >= MIN_ORDER_KRW

    logger.info(f"[{ticker}] Close={last_candle['close']}, {indicator_info}, Signal={signal}, Price={current_price}")
    logger.info(f"[{ticker}] {coin_sym}={coin_balance:.6f} (≈{coin_value:,.0f} KRW), Holding={is_holding}")

    return {
        'ticker': ticker,
        'coin_sym': coin_sym,
        'signal': signal,
        'weight': weight,
        'coin_balance': coin_balance,
        'coin_value': coin_value,
        'current_price': current_price,
        'is_holding': is_holding,
        'interval': interval,  # Pass interval for smart/adaptive execution
    }


def run_auto_trade():
    load_dotenv()

    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")

    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("API Keys not found. Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY.")
        return

    trader = UpbitTrader(ACCESS_KEY, SECRET_KEY)
    portfolio = get_portfolio()

    logger.info(f"=== Portfolio Auto Trade ({len(portfolio)} assets) ===")

    # ── 1단계: 전체 시그널 분석 ──
    analyses = []
    for item in portfolio:
        try:
            result = analyze_asset(trader, item)
            if result:
                analyses.append(result)
        except Exception as e:
            logger.error(f"Error analyzing {item.get('coin', '?')}: {e}")

    if not analyses:
        logger.error("No assets analyzed. Exiting.")
        return

    # ── 2단계: 매도 먼저 실행 (현금 확보) ──
    for a in analyses:
        if a['signal'] == 'SELL' and a['is_holding']:
            logger.info(f"[{a['ticker']}] SELL {a['coin_balance']:.6f} {a['coin_sym']} (Smart Sell)")
            try:
                # Use Smart Sell (Limit Order Split)
                result = trader.smart_sell(a['ticker'], a['coin_balance'], interval=a.get('interval', 'day'))
                logger.info(f"[{a['ticker']}] Sell Result: {result}")
            except Exception as e:
                logger.error(f"[{a['ticker']}] Sell Error: {e}")
        elif a['signal'] == 'SELL' and not a['is_holding']:
            logger.info(f"[{a['ticker']}] SELL signal but not holding - skip")

    # 매도 체결 대기
    time.sleep(1)

    # ── 3단계: 현금 기준 비중 배분 후 매수 ──
    krw_balance = trader.get_balance("KRW")

    # 보유 중인 자산의 비중 합산 (매도된 건 제외)
    held_weight = sum(
        a['weight'] for a in analyses
        if a['is_holding'] and a['signal'] != 'SELL'  # 아직 보유 중 (방금 매도한 건 제외)
    )
    available_weight = 100 - held_weight  # 현금으로 운용할 비중

    logger.info(f"KRW={krw_balance:,.0f} | Held weight={held_weight}% | Available weight={available_weight}%")

    for a in analyses:
        if a['signal'] == 'BUY':
            if a['is_holding']:
                logger.info(f"[{a['ticker']}] Already holding (≈{a['coin_value']:,.0f} KRW) - HOLD")
            else:
                # 현금에서 비중 비례 배분
                if available_weight > 0:
                    buy_budget = krw_balance * (a['weight'] / available_weight) * 0.999
                else:
                    buy_budget = 0

                if buy_budget > MIN_ORDER_KRW:
                    logger.info(f"[{a['ticker']}] BUY {buy_budget:,.0f} KRW (Adaptive) | Weight {a['weight']}%")
                    try:
                        # Use Adaptive Buy (Monitor price spike)
                        result = trader.adaptive_buy(a['ticker'], buy_budget, interval=a.get('interval', 'day'))
                        logger.info(f"[{a['ticker']}] Buy Result: {result}")
                    except Exception as e:
                        logger.error(f"[{a['ticker']}] Buy Error: {e}")
                else:
                    logger.info(f"[{a['ticker']}] BUY signal but budget too low ({buy_budget:,.0f} KRW)")

        elif a['signal'] == 'HOLD':
            logger.info(f"[{a['ticker']}] HOLD - no action")

    logger.info("=== Done ===")


if __name__ == "__main__":
    run_auto_trade()
