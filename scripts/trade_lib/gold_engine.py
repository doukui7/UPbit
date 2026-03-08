"""키움 금현물 자동매매 엔진."""
import os
import json
import logging
import pandas as pd
from datetime import datetime

from .constants import KST, PROJECT_ROOT, GOLD_KRX_ETF_CODE
from .utils import safe_float
from .notifier import send_telegram
from .data import get_gold_daily_local_first, get_gold_price_local_first

from src.strategy.donchian import DonchianStrategy
from src.engine.kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG

logger = logging.getLogger(__name__)


def is_market_hours() -> bool:
    """KST 09:00~16:00 장+시간외 주문 가능 시간 확인."""
    now = datetime.now(KST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_gold_portfolio():
    """GOLD_PORTFOLIO env var (JSON) 또는 레거시 env var에서 골드 포트폴리오 로드."""
    raw = os.getenv("GOLD_PORTFOLIO")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("GOLD_PORTFOLIO env var is not valid JSON. Using default.")

    return [{
        "strategy": "Donchian",
        "buy_period": int(os.getenv("GOLD_BUY_PERIOD", "90")),
        "sell_period": int(os.getenv("GOLD_SELL_PERIOD", "55")),
        "weight": 100,
    }]


def run_kiwoom_gold_trade():
    """키움 금현물 자동매매 (다중 전략 지원)."""
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("=== 키움 금현물 자동매매 시작 ===")

    if not is_market_hours():
        logger.info("장 시간 외 (09:00~15:30, 16:00~18:00 KST). 매매 생략.")
        return

    trader = KiwoomGoldTrader(is_mock=False)
    if not trader.auth():
        logger.error("키움 인증 실패. 종료.")
        return

    code = GOLD_CODE_1KG
    gold_portfolio = get_gold_portfolio()
    total_weight = sum(g['weight'] for g in gold_portfolio)

    logger.info(f"전략 수: {len(gold_portfolio)} | 총 비중: {total_weight}%")

    # ── 1. 일봉 데이터 로드 ──
    max_period = max(g['buy_period'] for g in gold_portfolio)
    df = get_gold_daily_local_first(trader, code=code, count=max(max_period + 10, 200))

    if df is None or len(df) < max_period + 5:
        logger.warning("API 일봉 데이터 부족 → krx_gold_daily.csv 폴백 사용")
        csv_path = os.path.join(PROJECT_ROOT, "krx_gold_daily.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "krx_gold_daily.csv")
        try:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df.columns = [c.lower() for c in df.columns]
            if "open" not in df.columns: df["open"] = df["close"]
            if "high" not in df.columns: df["high"] = df["close"]
            if "low" not in df.columns: df["low"] = df["close"]
        except Exception as e:
            logger.error(f"CSV 로드 실패: {e}")
            return

    if df is None or len(df) < max_period + 5:
        logger.error("데이터 부족으로 매매 불가.")
        return

    # ── 2. 전략별 시그널 계산 → 비중 가중 합산 ──
    buy_weight = 0
    for gp in gold_portfolio:
        strat_name = gp['strategy']
        bp = gp['buy_period']
        sp = gp.get('sell_period', 0) or max(5, bp // 2)
        w = gp['weight']

        if strat_name == "Donchian":
            strat = DonchianStrategy()
            df_s = strat.create_features(df.copy(), buy_period=bp, sell_period=sp)
            last = df_s.iloc[-2]
            signal = strat.get_signal(last, buy_period=bp, sell_period=sp)
            upper = last.get(f"Donchian_Upper_{bp}", "N/A")
            lower = last.get(f"Donchian_Lower_{sp}", "N/A")
            logger.info(f"  {strat_name}({bp}/{sp}) w={w}% → {signal} | Upper={upper}, Lower={lower}, Close={last['close']}")
        else:
            sma = df['close'].rolling(window=bp).mean()
            last_close = float(df['close'].iloc[-2])
            last_sma = float(sma.iloc[-2])
            signal = "BUY" if last_close > last_sma else "SELL"
            logger.info(f"  {strat_name}({bp}) w={w}% → {signal} | SMA={last_sma:.0f}, Close={last_close:.0f}")

        if signal == "BUY":
            buy_weight += w

    target_ratio = buy_weight / total_weight if total_weight > 0 else 0
    logger.info(f"시그널 합산: BUY비중={buy_weight}% / 총{total_weight}% → 목표 포지션 {target_ratio:.0%}")

    # ── 3. 잔고 확인 ──
    bal = trader.get_balance()
    if bal is None:
        logger.error("잔고 조회 실패. 종료.")
        return

    cash_krw = bal["cash_krw"]
    gold_qty = bal["gold_qty"]
    gold_price = get_gold_price_local_first(trader, code) or 0
    gold_value = gold_qty * gold_price if gold_price > 0 else 0
    total_value = cash_krw + gold_value

    current_ratio = gold_value / total_value if total_value > 0 else 0
    logger.info(f"예수금: {cash_krw:,.0f}원 | 금: {gold_qty}g (≈{gold_value:,.0f}원) | 총자산: {total_value:,.0f}원")
    logger.info(f"현재 금 비중: {current_ratio:.1%} → 목표: {target_ratio:.1%}")

    MIN_ORDER = 10_000

    # ── 4. 매매 실행 ──
    target_gold_value = total_value * target_ratio
    diff = target_gold_value - gold_value

    if diff > MIN_ORDER:
        buy_amount = min(diff, cash_krw) * 0.999
        if buy_amount >= MIN_ORDER:
            logger.info(f"[BUY] {buy_amount:,.0f}원 동시호가 매수")
            result = trader.smart_buy_krw_closing(code=code, krw_amount=buy_amount)
            logger.info(f"매수 결과: {result}")
        else:
            logger.info(f"[BUY] 예수금 부족 ({cash_krw:,.0f}원)")
    elif diff < -MIN_ORDER and gold_price > 0:
        sell_value = abs(diff)
        sell_qty = min(sell_value / gold_price, gold_qty)
        sell_qty = round(sell_qty, 2)
        if sell_qty >= gold_qty * 0.99:
            logger.info(f"[SELL] {gold_qty}g 전량 동시호가 매도")
            result = trader.smart_sell_all_closing(code=code)
            logger.info(f"매도 결과: {result}")
        elif sell_qty > 0 and (sell_qty * gold_price) >= MIN_ORDER:
            logger.info(f"[SELL] {sell_qty}g 부분 동시호가 매도")
            result = trader.execute_closing_auction_sell(code, qty=sell_qty)
            logger.info(f"매도 결과: {result}")
        else:
            logger.info(f"[SELL] 매도 금액 미달 ({sell_qty * gold_price:,.0f}원 < {MIN_ORDER:,}원)")
    else:
        logger.info(f"[HOLD] 현재 비중({current_ratio:.1%})이 목표({target_ratio:.1%})와 근사 - 유지")

    logger.info("=== 키움 금현물 자동매매 완료 ===")

    tg = [f"<b>키움 금현물</b>"]
    tg.append(f"총자산: {total_value:,.0f}원 (현금 {cash_krw:,.0f} / 금 {gold_qty}g)")
    tg.append(f"목표비중: {target_ratio:.0%} / 현재: {current_ratio:.1%}")
    if diff > MIN_ORDER:
        tg.append(f"[BUY] {abs(diff):,.0f}원")
    elif diff < -MIN_ORDER:
        tg.append(f"[SELL] {abs(diff):,.0f}원")
    else:
        tg.append("[HOLD]")
    send_telegram("\n".join(tg))
