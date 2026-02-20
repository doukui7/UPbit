import os
import sys
import json
import time
import logging
import pyupbit
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.sma import SMAStrategy
from strategy.donchian import DonchianStrategy
from trading.upbit_trader import UpbitTrader
from kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG

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

    # ── 총 포트폴리오 가치 계산 (목표 배분액 산정 기준) ──
    krw_balance = trader.get_balance("KRW")

    # 코인별 보유 가치 (중복 제거)
    seen_coins = set()
    total_coin_value = 0
    for a in analyses:
        if a['coin_sym'] not in seen_coins:
            total_coin_value += a['coin_value']
            seen_coins.add(a['coin_sym'])

    total_portfolio_value = krw_balance + total_coin_value

    # 코인별 비중 합산 (같은 코인에 여러 전략일 때)
    total_weight = sum(a['weight'] for a in analyses)
    coin_weight_sum = {}
    for a in analyses:
        coin_weight_sum[a['coin_sym']] = coin_weight_sum.get(a['coin_sym'], 0) + a['weight']

    logger.info(f"Portfolio Value={total_portfolio_value:,.0f} KRW (현금={krw_balance:,.0f}, 코인={total_coin_value:,.0f})")
    logger.info(f"Total weight={total_weight}% | Coin weights: {coin_weight_sum}")

    # ── 2단계: 매도 먼저 실행 (목표 배분액 기준) ──
    for a in analyses:
        if a['signal'] == 'SELL' and a['is_holding']:
            # 이 전략의 목표 배분액 = 총 자산 × (weight / 100)
            target_value = total_portfolio_value * (a['weight'] / 100)
            # 이 전략이 관리하는 코인 비중 비율
            coin_total_w = coin_weight_sum.get(a['coin_sym'], a['weight'])
            sell_ratio = a['weight'] / coin_total_w
            # 실제 보유량 중 이 전략 몫
            proportional_qty = a['coin_balance'] * sell_ratio
            # 목표 배분액 기준 매도 수량 (보유가 더 많으면 목표량만)
            if a['current_price'] and a['current_price'] > 0:
                target_qty = target_value / a['current_price']
                sell_qty = min(proportional_qty, target_qty)
            else:
                sell_qty = proportional_qty

            sell_value = sell_qty * a['current_price'] if a['current_price'] else 0
            if sell_value < MIN_ORDER_KRW:
                logger.info(f"[{a['ticker']}] SELL signal but sell value too low ({sell_value:,.0f} KRW < {MIN_ORDER_KRW}) - skip")
                continue

            logger.info(f"[{a['ticker']}] SELL {sell_qty:.6f}/{a['coin_balance']:.6f} {a['coin_sym']} "
                        f"(목표={target_value:,.0f}KRW, 비중={a['weight']}%/{coin_total_w}%)")
            try:
                result = trader.smart_sell(a['ticker'], sell_qty, interval=a.get('interval', 'day'))
                logger.info(f"[{a['ticker']}] Sell Result: {result}")
            except Exception as e:
                logger.error(f"[{a['ticker']}] Sell Error: {e}")

        elif a['signal'] == 'SELL' and not a['is_holding']:
            logger.info(f"[{a['ticker']}] SELL signal but not holding - skip")

    # 매도 체결 대기
    time.sleep(1)

    # ── 3단계: 목표 배분액 기준 매수 ──
    krw_balance = trader.get_balance("KRW")  # 매도 후 갱신

    buy_signals = [a for a in analyses if a['signal'] == 'BUY' and not a['is_holding']]

    logger.info(f"KRW(매도후)={krw_balance:,.0f} | Buy signals={len(buy_signals)}")

    for a in buy_signals:
        # 이 전략의 목표 배분액
        target_value = total_portfolio_value * (a['weight'] / 100)
        # 이미 보유 중인 가치 (같은 코인 다른 전략이 보유 중일 수 있음)
        current_holding_value = a['coin_value']
        # 추가 매수 필요액 = 목표 - 현재 보유(이 전략 몫)
        coin_total_w = coin_weight_sum.get(a['coin_sym'], a['weight'])
        my_holding = current_holding_value * (a['weight'] / coin_total_w)
        need_value = target_value - my_holding

        # 목표 배분액과 가용 현금 중 작은 값
        buy_budget = min(need_value, krw_balance) * 0.999

        if buy_budget > MIN_ORDER_KRW:
            logger.info(f"[{a['ticker']}] BUY {buy_budget:,.0f} KRW (Adaptive) | "
                        f"목표={target_value:,.0f}, 보유={my_holding:,.0f}, 추가필요={need_value:,.0f}")
            try:
                result = trader.adaptive_buy(a['ticker'], buy_budget, interval=a.get('interval', 'day'))
                logger.info(f"[{a['ticker']}] Buy Result: {result}")
                krw_balance -= buy_budget  # 사용한 현금 차감
            except Exception as e:
                logger.error(f"[{a['ticker']}] Buy Error: {e}")
        else:
            logger.info(f"[{a['ticker']}] BUY signal but budget insufficient "
                        f"(필요={need_value:,.0f}, 가용={krw_balance:,.0f})")

    # HOLD 및 이미 보유 중인 BUY 시그널 로깅
    for a in analyses:
        if a['signal'] == 'BUY' and a['is_holding']:
            target_value = total_portfolio_value * (a['weight'] / 100)
            logger.info(f"[{a['ticker']}] Already holding (≈{a['coin_value']:,.0f} KRW, 목표={target_value:,.0f}) - HOLD")
        elif a['signal'] == 'HOLD':
            logger.info(f"[{a['ticker']}] HOLD - no action")

    logger.info("=== Done ===")


# ─────────────────────────────────────────────────────────
# 키움 금현물 자동매매
# ─────────────────────────────────────────────────────────
def _is_market_hours() -> bool:
    """KST 09:00~15:30 장 시간인지 확인."""
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    # 주말 제외
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close


def run_kiwoom_gold_trade():
    """
    키움 금현물 자동매매.
    전략: Donchian(90/55) 일봉 (백테스트 최우수 성과)
    수수료: 0.3%
    실행 조건: KST 09:00~15:30 (장 시간)
    """
    load_dotenv()
    logger.info("=== 키움 금현물 자동매매 시작 ===")

    # 장 시간 확인
    if not _is_market_hours():
        logger.info("장 시간 외 (09:00~15:30 KST). 매매 생략.")
        return

    # 트레이더 초기화 & 인증
    trader = KiwoomGoldTrader(is_mock=False)
    if not trader.auth():
        logger.error("키움 인증 실패. 종료.")
        return

    # ── 1. 일봉 데이터 로드 (API 또는 CSV 폴백) ───────────
    code = GOLD_CODE_1KG
    BUY_PERIOD  = int(os.getenv("GOLD_BUY_PERIOD",  "90"))
    SELL_PERIOD = int(os.getenv("GOLD_SELL_PERIOD", "55"))

    logger.info(f"전략: Donchian({BUY_PERIOD}/{SELL_PERIOD}) | 종목: {code}")

    df = trader.get_daily_chart(code=code, count=max(BUY_PERIOD + 10, 200))

    # API 일봉이 없으면 CSV 폴백
    if df is None or len(df) < BUY_PERIOD + 5:
        logger.warning("API 일봉 데이터 부족 → krx_gold_daily.csv 폴백 사용")
        csv_path = os.path.join(os.path.dirname(__file__), "krx_gold_daily.csv")
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            if "open" not in df.columns: df["open"] = df["close"]
            if "high" not in df.columns: df["high"] = df["close"]
            if "low"  not in df.columns: df["low"]  = df["close"]
        except Exception as e:
            logger.error(f"CSV 로드 실패: {e}")
            return

    if df is None or len(df) < BUY_PERIOD + 5:
        logger.error("데이터 부족으로 매매 불가.")
        return

    # ── 2. 시그널 계산 (Donchian) ─────────────────────────
    strat = DonchianStrategy()
    df = strat.create_features(df, buy_period=BUY_PERIOD, sell_period=SELL_PERIOD)
    last = df.iloc[-2]  # 마지막 완성 봉
    signal = strat.get_signal(last, buy_period=BUY_PERIOD, sell_period=SELL_PERIOD)

    upper = last.get(f"Donchian_Upper_{BUY_PERIOD}", "N/A")
    lower = last.get(f"Donchian_Lower_{SELL_PERIOD}", "N/A")
    logger.info(f"시그널: {signal} | Upper={upper}, Lower={lower} | 종가={last['close']}")

    # ── 3. 잔고 확인 ──────────────────────────────────────
    bal = trader.get_balance()
    if bal is None:
        logger.error("잔고 조회 실패. 종료.")
        return

    cash_krw  = bal["cash_krw"]
    gold_qty  = bal["gold_qty"]
    is_holding = gold_qty > 0

    logger.info(f"예수금: {cash_krw:,.0f}원 | 금 보유: {gold_qty}g, 보유여부: {is_holding}")

    MIN_ORDER = 10_000  # 최소 주문금액 (10,000원)

    # ── 4. 매매 실행 ──────────────────────────────────────
    if signal == "BUY" and not is_holding:
        if cash_krw >= MIN_ORDER:
            logger.info(f"[BUY] {cash_krw:,.0f}원 어치 금 시장가 매수")
            result = trader.smart_buy_krw(code=code, krw_amount=cash_krw * 0.999)
            logger.info(f"매수 결과: {result}")
        else:
            logger.info(f"[BUY] 예수금 부족 ({cash_krw:,.0f}원 < {MIN_ORDER:,}원)")

    elif signal == "SELL" and is_holding:
        logger.info(f"[SELL] {gold_qty}g 전량 시장가 매도")
        result = trader.smart_sell_all(code=code)
        logger.info(f"매도 결과: {result}")

    elif signal == "BUY" and is_holding:
        logger.info("[HOLD] 이미 매수 보유 중")
    elif signal == "SELL" and not is_holding:
        logger.info("[HOLD] 매도 시그널이나 보유 없음")
    else:
        logger.info("[HOLD] 시그널 없음")

    logger.info("=== 키움 금현물 자동매매 완료 ===")

if __name__ == "__main__":
    load_dotenv()
    mode = os.getenv("TRADING_MODE", "upbit").lower()
    if mode == "kiwoom_gold":
        run_kiwoom_gold_trade()
    else:
        run_auto_trade()
