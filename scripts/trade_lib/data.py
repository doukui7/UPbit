"""OHLCV 데이터 및 현재가 조회 (캐시 우선)."""
import logging

import src.engine.data_cache as data_cache

logger = logging.getLogger(__name__)


def fetch_overseas_chart_any_exchange(trader, symbol: str, count: int = 420):
    """Try NAS/NYS/AMS in order and return (df, exchange)."""
    for ex in ("NAS", "NYS", "AMS"):
        try:
            df = trader.get_overseas_daily_chart(symbol, exchange=ex, count=count)
        except Exception:
            df = None
        if df is not None and len(df) > 0:
            return df, ex
    return None, None


def get_upbit_ohlcv_local_first(ticker: str, interval: str, count: int):
    return data_cache.get_ohlcv_local_first(
        ticker,
        interval=interval,
        count=int(max(1, count)),
        allow_api_fallback=True,
    )


def get_upbit_price_local_first(ticker: str) -> float:
    return float(data_cache.get_current_price_local_first(ticker, ttl_sec=5.0, allow_api_fallback=True) or 0.0)


def get_kis_daily_local_first(trader, code: str, count: int, end_date: str | None = None):
    return data_cache.get_kis_domestic_local_first(
        trader,
        str(code).strip(),
        count=int(max(1, count)),
        end_date=end_date,
        allow_api_fallback=True,
    )


def get_kis_price_local_first(trader, code: str) -> float:
    """현재가 조회 - API 전용 (실시간). 캐시 사용 안 함."""
    try:
        return float(trader.get_current_price(str(code).strip()) or 0.0)
    except Exception:
        return 0.0


def get_gold_daily_local_first(trader, code: str, count: int):
    return data_cache.get_gold_daily_local_first(
        trader=trader,
        code=code,
        count=int(max(1, count)),
        allow_api_fallback=True,
    )


def get_gold_price_local_first(trader, code: str) -> float:
    p = float(
        data_cache.get_gold_current_price_local_first(
            trader=trader,
            code=code,
            allow_api_fallback=True,
            ttl_sec=8.0,
        )
        or 0.0
    )
    if p > 0:
        return p

    # 장외시간/API 히컵 대비: 최신 일봉 종가 사용
    try:
        df = trader.get_daily_chart(code=code, count=3)
        if df is not None and len(df) > 0:
            if "close" in df.columns:
                p = float(df["close"].iloc[-1] or 0.0)
            elif "Close" in df.columns:
                p = float(df["Close"].iloc[-1] or 0.0)
    except Exception:
        p = 0.0

    return float(p if p > 0 else 0.0)
