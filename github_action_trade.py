import os
import sys
import json
import time
import html
import re
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import data_cache

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.sma import SMAStrategy
from strategy.donchian import DonchianStrategy
from strategy.widaeri import WDRStrategy
from strategy.dual_momentum import DualMomentumStrategy
from strategy.laa import LAAStrategy
from trading.upbit_trader import UpbitTrader
from kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG
from kis_trader import KISTrader

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

MIN_ORDER_KRW = 5000


def _load_user_config():
    """Load user_config.json from project root."""
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_env_any(*keys, default=""):
    """Return first non-empty env value among keys."""
    for key in keys:
        val = os.getenv(key, "")
        if str(val).strip():
            return val
    return default


def _normalize_kis_account_fields(account_no: str, prdt_cd: str = "01") -> tuple[str, str]:
    """
    KIS 계좌번호 정규화.
    - 10자리 계좌번호 입력 시: 앞 8자리(CANO), 뒤 2자리(상품코드)로 자동 분리
    - 8자리 계좌번호 입력 시: 전달된 상품코드(없으면 01) 사용
    """
    raw_acct = "".join(ch for ch in str(account_no or "") if ch.isdigit())
    raw_prdt = "".join(ch for ch in str(prdt_cd or "") if ch.isdigit())

    if len(raw_acct) >= 10:
        cano = raw_acct[:8]
        acnt_prdt_cd = raw_acct[8:10]
    else:
        cano = raw_acct[:8] if len(raw_acct) > 8 else raw_acct
        acnt_prdt_cd = raw_prdt

    if not acnt_prdt_cd:
        acnt_prdt_cd = "01"

    return cano, acnt_prdt_cd.zfill(2)[:2]


def _fetch_overseas_chart_any_exchange(trader: KISTrader, symbol: str, count: int = 420):
    """Try NAS/NYS/AMS in order and return (df, exchange)."""
    for ex in ("NAS", "NYS", "AMS"):
        try:
            df = trader.get_overseas_daily_chart(symbol, exchange=ex, count=count)
        except Exception:
            df = None
        if df is not None and len(df) > 0:
            return df, ex
    return None, None


def _get_upbit_ohlcv_local_first(ticker: str, interval: str, count: int):
    return data_cache.get_ohlcv_local_first(
        ticker,
        interval=interval,
        count=int(max(1, count)),
        allow_api_fallback=True,
    )


def _get_upbit_price_local_first(ticker: str) -> float:
    return float(data_cache.get_current_price_local_first(ticker, ttl_sec=5.0, allow_api_fallback=True) or 0.0)


def _get_kis_daily_local_first(trader: KISTrader, code: str, count: int, end_date: str | None = None):
    return data_cache.get_kis_domestic_local_first(
        trader,
        str(code).strip(),
        count=int(max(1, count)),
        end_date=end_date,
        allow_api_fallback=True,
    )


def _get_kis_price_local_first(trader: KISTrader, code: str) -> float:
    _df = _get_kis_daily_local_first(trader, code, count=3)
    if _df is not None and not _df.empty:
        if "close" in _df.columns:
            return float(_df["close"].iloc[-1] or 0.0)
        if "Close" in _df.columns:
            return float(_df["Close"].iloc[-1] or 0.0)
    try:
        return float(trader.get_current_price(str(code).strip()) or 0.0)
    except Exception:
        return 0.0


def _get_gold_daily_local_first(trader: KiwoomGoldTrader, code: str, count: int):
    return data_cache.get_gold_daily_local_first(
        trader=trader,
        code=code,
        count=int(max(1, count)),
        allow_api_fallback=True,
    )


def _get_gold_price_local_first(trader: KiwoomGoldTrader, code: str) -> float:
    return float(
        data_cache.get_gold_current_price_local_first(
            trader=trader,
            code=code,
            allow_api_fallback=True,
            ttl_sec=8.0,
        )
        or 0.0
    )


def _send_telegram(message: str):
    """텔레그램 봇으로 메시지 전송. 토큰/챗ID 없으면 무시."""
    token = _get_env_any("TELEGRAM_BOT_TOKEN", "telegram_bot_token")
    chat_id = _get_env_any("TELEGRAM_CHAT_ID", "telegram_chat_id")
    if not token or not chat_id:
        cfg = _load_user_config()
        token = token or str(cfg.get("telegram_bot_token", "")).strip()
        chat_id = chat_id or str(cfg.get("telegram_chat_id", "")).strip()
    if not token or not chat_id:
        logger.warning("텔레그램 설정이 없어 알림 전송을 생략합니다. (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)")
        return

    def _normalize_bot_token(raw: str) -> str:
        t = str(raw or "").strip().strip('"').strip("'")
        t = re.sub(r"\s+", "", t)
        # 전체 API URL을 넣어도 토큰만 추출되도록 처리
        if "api.telegram.org" in t and "/bot" in t:
            m = re.search(r"/bot([^/\\s]+)", t)
            if m:
                t = m.group(1).strip()
        if t.lower().startswith("bot"):
            t = t[3:]
        if ":" in t:
            left, right = t.split(":", 1)
            left = "".join(ch for ch in left if ch.isdigit())
            right = "".join(ch for ch in right if re.match(r"[A-Za-z0-9_-]", ch))
            t = f"{left}:{right}" if left and right else t
        # 유효 패턴(숫자:영숫자/언더스코어/하이픈)만 추출
        m2 = re.search(r"([0-9]{6,}:[A-Za-z0-9_-]{20,})", t)
        if m2:
            t = m2.group(1)
        return t.strip().strip('"').strip("'")

    token = _normalize_bot_token(token)
    chat_id = str(chat_id).strip().strip('"').strip("'")
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def _sanitize_html_chunk(text: str) -> str:
        # parse_mode=HTML에서 '&', '<', '>'로 인한 파싱 오류 방지
        escaped = html.escape(str(text), quote=False)
        # 내부에서 사용하는 최소 태그만 허용
        escaped = escaped.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
        escaped = escaped.replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>")
        escaped = escaped.replace("&lt;pre&gt;", "<pre>").replace("&lt;/pre&gt;", "</pre>")
        return escaped

    try:
        import requests
        # 텔레그램 메시지 4096자 제한
        for i in range(0, len(message), 4000):
            chunk = message[i:i+4000]
            safe_chunk = _sanitize_html_chunk(chunk)
            resp = requests.post(url, json={
                "chat_id": chat_id,
                "text": safe_chunk,
                "parse_mode": "HTML",
            }, timeout=10)
            ok = False
            desc = ""
            try:
                body = resp.json()
                ok = bool(resp.ok and body.get("ok", False))
                desc = str(body.get("description", ""))
            except Exception:
                ok = bool(resp.ok)
                desc = resp.text[:200] if getattr(resp, "text", "") else ""

            if ok:
                logger.info("텔레그램 전송 성공")
            else:
                logger.warning(f"텔레그램 전송 실패: HTTP {resp.status_code} {desc}")
    except Exception as e:
        logger.warning(f"텔레그램 전송 실패: {e}")


def _mask_secret(value: str, left: int = 4, right: int = 4) -> str:
    """민감정보 마스킹."""
    s = str(value or "").strip()
    if not s:
        return "(empty)"
    if len(s) <= left + right:
        return "*" * len(s)
    return f"{s[:left]}...{s[-right:]}"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _append_step(result: dict, step: str, status: str, detail: str):
    """
    헬스체크 단계 로그 누적.
    status: PASS | FAIL | SKIP | INFO
    """
    st = str(status or "INFO").upper()
    detail_text = str(detail or "")
    if len(detail_text) > 280:
        detail_text = detail_text[:280] + "...(truncated)"
    line = f"{step} [{st}] {detail_text}"
    result.setdefault("steps", []).append(line)
    logger.info(f"[{result.get('name', '헬스체크')}] {line}")


def get_portfolio():
    """Load portfolio from PORTFOLIO env var (JSON). Raises if not set."""
    raw = os.getenv("PORTFOLIO")
    if not raw:
        raise RuntimeError("PORTFOLIO 환경변수가 설정되지 않았습니다. GitHub Secrets에 PORTFOLIO를 등록하세요.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"PORTFOLIO 환경변수가 올바른 JSON이 아닙니다: {raw[:100]}")


def _normalize_coin_interval(interval: str) -> str:
    """코인 전략 interval 값을 내부 표준 키로 정규화."""
    iv = str(interval or "day").strip().lower()
    if iv in {"1d", "d", "day", "daily"}:
        return "day"
    if iv in {"4h", "240", "240m", "minute240"}:
        return "minute240"
    if iv in {"1h", "60", "60m", "minute60"}:
        return "minute60"
    return iv


def _is_coin_interval_due(interval: str, now_kst: datetime) -> bool:
    """
    현재 시각 기준 전략 주기 실행 여부.
    - 4H(minute240): KST 01/05/09/13/17/21시
    - 1D(day): KST 09시
    - 그 외: 호출 시점마다 실행
    """
    iv = _normalize_coin_interval(interval)
    hour = int(now_kst.hour)

    if iv == "day":
        return hour == 9
    if iv == "minute240":
        return (hour % 4) == 1
    return True


def analyze_asset(trader, item):
    """1단계: 데이터 조회 → 시그널 계산 → 보유 현황 확인"""
    ticker = f"{item['market']}-{item['coin'].upper()}"
    strategy_name = item.get("strategy", "SMA")
    param = item.get("parameter", 20)
    interval_raw = item.get("interval", "day")
    interval = _normalize_coin_interval(interval_raw)
    weight = item.get("weight", 100)

    logger.info(f"--- [{ticker}] {strategy_name}({param}), {weight}%, {interval_raw}->{interval} ---")

    # Fetch data
    count = max(200, param * 3)
    df = _get_upbit_ohlcv_local_first(ticker, interval=interval, count=count)
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

    current_price = _get_upbit_price_local_first(ticker)
    coin_sym = item['coin'].upper()
    _raw_balance = trader.get_balance(coin_sym)
    try:
        coin_balance = float(_raw_balance or 0.0)
    except Exception:
        coin_balance = 0.0
    coin_value = coin_balance * float(current_price or 0.0)
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
    kst = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst)

    # 전략 주기(4H/1D)별 실행 필터
    due_portfolio = []
    for item in portfolio:
        iv_raw = item.get("interval", "day")
        if _is_coin_interval_due(iv_raw, now_kst):
            due_portfolio.append(item)
        else:
            ticker = f"{item.get('market', 'KRW')}-{str(item.get('coin', '')).upper()}"
            logger.info(f"[{ticker}] 주기 미도래로 스킵 (interval={iv_raw}, now={now_kst.strftime('%H:%M')} KST)")

    if not due_portfolio:
        logger.info(f"=== Portfolio Auto Trade: 실행 대상 없음 (now={now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST) ===")
        return

    logger.info(f"=== Portfolio Auto Trade ({len(due_portfolio)}/{len(portfolio)} assets due) ===")

    # ── 1단계: 전체 시그널 분석 ──
    analyses = []
    for item in due_portfolio:
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

    # 텔레그램 요약 전송
    tg_lines = [f"<b>코인 자동매매</b> ({len(due_portfolio)}종목 실행 / 전체 {len(portfolio)}종목)"]
    tg_lines.append(f"실행시각: {now_kst.strftime('%Y-%m-%d %H:%M')} KST")
    tg_lines.append(f"총자산: {total_portfolio_value:,.0f}원 (현금 {krw_balance:,.0f})")
    for a in analyses:
        action = a['signal']
        iv = _normalize_coin_interval(a.get('interval', 'day'))
        iv_label = "4H" if iv == "minute240" else ("1D" if iv == "day" else str(a.get('interval', 'day')))
        tg_lines.append(f"  {a['ticker']} [{iv_label}]: {action} (보유≈{a['coin_value']:,.0f}원)")
    _send_telegram("\n".join(tg_lines))


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


def get_gold_portfolio():
    """GOLD_PORTFOLIO env var (JSON) 또는 레거시 env var 에서 골드 포트폴리오 로드."""
    raw = os.getenv("GOLD_PORTFOLIO")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("GOLD_PORTFOLIO env var is not valid JSON. Using default.")

    # 레거시: 단일 전략 env var 폴백
    return [{
        "strategy": "Donchian",
        "buy_period": int(os.getenv("GOLD_BUY_PERIOD", "90")),
        "sell_period": int(os.getenv("GOLD_SELL_PERIOD", "55")),
        "weight": 100,
    }]


def run_kiwoom_gold_trade():
    """
    키움 금현물 자동매매 (다중 전략 지원).
    각 전략의 시그널을 비중 가중으로 합산하여 목표 포지션 비율 결정.
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

    code = GOLD_CODE_1KG
    gold_portfolio = get_gold_portfolio()
    total_weight = sum(g['weight'] for g in gold_portfolio)

    logger.info(f"전략 수: {len(gold_portfolio)} | 총 비중: {total_weight}%")
    for gp in gold_portfolio:
        logger.info(f"  - {gp['strategy']}({gp['buy_period']}/{gp.get('sell_period', '-')}) 비중={gp['weight']}%")

    # ── 1. 일봉 데이터 로드 (API 또는 CSV 폴백) ───────────
    max_period = max(g['buy_period'] for g in gold_portfolio)
    df = _get_gold_daily_local_first(trader, code=code, count=max(max_period + 10, 200))

    # API 일봉이 없으면 CSV 폴백
    if df is None or len(df) < max_period + 5:
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

    if df is None or len(df) < max_period + 5:
        logger.error("데이터 부족으로 매매 불가.")
        return

    # ── 2. 전략별 시그널 계산 → 비중 가중 합산 ─────────────
    buy_weight = 0

    for gp in gold_portfolio:
        strat_name = gp['strategy']
        bp = gp['buy_period']
        sp = gp.get('sell_period', 0) or max(5, bp // 2)
        w = gp['weight']

        if strat_name == "Donchian":
            strat = DonchianStrategy()
            df_s = strat.create_features(df.copy(), buy_period=bp, sell_period=sp)
            last = df_s.iloc[-2]  # 마지막 완성 봉
            signal = strat.get_signal(last, buy_period=bp, sell_period=sp)
            upper = last.get(f"Donchian_Upper_{bp}", "N/A")
            lower = last.get(f"Donchian_Lower_{sp}", "N/A")
            logger.info(f"  {strat_name}({bp}/{sp}) w={w}% → {signal} | Upper={upper}, Lower={lower}, Close={last['close']}")
        else:  # SMA
            sma = df['close'].rolling(window=bp).mean()
            last_close = float(df['close'].iloc[-2])
            last_sma = float(sma.iloc[-2])
            signal = "BUY" if last_close > last_sma else "SELL"
            logger.info(f"  {strat_name}({bp}) w={w}% → {signal} | SMA={last_sma:.0f}, Close={last_close:.0f}")

        if signal == "BUY":
            buy_weight += w

    target_ratio = buy_weight / total_weight if total_weight > 0 else 0
    logger.info(f"시그널 합산: BUY비중={buy_weight}% / 총{total_weight}% → 목표 포지션 {target_ratio:.0%}")

    # ── 3. 잔고 확인 ──────────────────────────────────────
    bal = trader.get_balance()
    if bal is None:
        logger.error("잔고 조회 실패. 종료.")
        return

    cash_krw = bal["cash_krw"]
    gold_qty = bal["gold_qty"]
    gold_price = _get_gold_price_local_first(trader, code) or 0
    gold_value = gold_qty * gold_price if gold_price > 0 else 0
    total_value = cash_krw + gold_value

    current_ratio = gold_value / total_value if total_value > 0 else 0
    logger.info(f"예수금: {cash_krw:,.0f}원 | 금: {gold_qty}g (≈{gold_value:,.0f}원) | 총자산: {total_value:,.0f}원")
    logger.info(f"현재 금 비중: {current_ratio:.1%} → 목표: {target_ratio:.1%}")

    MIN_ORDER = 10_000  # 최소 주문금액

    # ── 4. 매매 실행 (목표 비중으로 조정) ──────────────────
    target_gold_value = total_value * target_ratio
    diff = target_gold_value - gold_value  # 양수=매수 필요, 음수=매도 필요

    if diff > MIN_ORDER:
        # 매수: 부족분만큼 매수
        buy_amount = min(diff, cash_krw) * 0.999
        if buy_amount >= MIN_ORDER:
            logger.info(f"[BUY] {buy_amount:,.0f}원 동시호가 매수 (현재 {gold_value:,.0f} → 목표 {target_gold_value:,.0f}원)")
            result = trader.smart_buy_krw_closing(code=code, krw_amount=buy_amount)
            logger.info(f"매수 결과: {result}")
        else:
            logger.info(f"[BUY] 예수금 부족 ({cash_krw:,.0f}원)")

    elif diff < -MIN_ORDER and gold_price > 0:
        # 매도: 초과분만큼 매도
        sell_value = abs(diff)
        sell_qty = min(sell_value / gold_price, gold_qty)
        sell_qty = round(sell_qty, 2)

        if sell_qty >= gold_qty * 0.99:
            # 거의 전량이면 smart_sell_all 사용 (잔량 방지)
            logger.info(f"[SELL] {gold_qty}g 전량 동시호가 매도 (현재 {gold_value:,.0f} → 목표 {target_gold_value:,.0f}원)")
            result = trader.smart_sell_all_closing(code=code)
            logger.info(f"매도 결과: {result}")
        elif sell_qty > 0 and (sell_qty * gold_price) >= MIN_ORDER:
            logger.info(f"[SELL] {sell_qty}g 부분 동시호가 매도 (현재 {gold_value:,.0f} → 목표 {target_gold_value:,.0f}원)")
            result = trader.execute_closing_auction_sell(code, qty=sell_qty)
            logger.info(f"매도 결과: {result}")
        else:
            logger.info(f"[SELL] 매도 금액 미달 ({sell_qty * gold_price:,.0f}원 < {MIN_ORDER:,}원)")

    else:
        logger.info(f"[HOLD] 현재 비중({current_ratio:.1%})이 목표({target_ratio:.1%})와 근사 - 유지")

    logger.info("=== 키움 금현물 자동매매 완료 ===")

    # 텔레그램 요약
    tg = [f"<b>키움 금현물</b>"]
    tg.append(f"총자산: {total_value:,.0f}원 (현금 {cash_krw:,.0f} / 금 {gold_qty}g)")
    tg.append(f"목표비중: {target_ratio:.0%} / 현재: {current_ratio:.1%}")
    if diff > MIN_ORDER:
        tg.append(f"[BUY] {abs(diff):,.0f}원")
    elif diff < -MIN_ORDER:
        tg.append(f"[SELL] {abs(diff):,.0f}원")
    else:
        tg.append("[HOLD]")
    _send_telegram("\n".join(tg))

# ─────────────────────────────────────────────────────────
# KIS ISA 위대리(WDR) 자동매매
# ─────────────────────────────────────────────────────────
def _is_kr_market_hours() -> bool:
    """KST 09:00~16:00 국내 장 시간 확인 (동시호가 + 시간외 포함)."""
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def _is_kr_order_window() -> bool:
    """KST 09:00~15:20 주문 가능 시간 확인."""
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    if now.weekday() >= 5:
        return False
    order_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    order_close = now.replace(hour=15, minute=20, second=0, microsecond=0)
    return order_open <= now <= order_close


def run_kis_isa_trade():
    """
    한국투자증권 ISA 계좌 - 위대리(WDR) 3단계 전략 자동매매.

    전략 요약:
      - TREND ETF(기본 133690) 성장 추세선 대비 이격도로 3단계 시장 상태 판단
      - 주간 손익에 따라 매도/매수 비율 조정
      - 매매 대상: 국내 레버리지/액티브 ETF (기본: TIGER 미국나스닥100레버리지 = 418660)

    필요 환경변수:
      KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_ACNT_PRDT_CD
      KIS_ISA_ETF_CODE (매매 ETF 코드, 기본 418660)
      KIS_ISA_TREND_ETF_CODE (TREND ETF 코드, 기본 133690)
    """
    load_dotenv()
    logger.info("=== KIS ISA 위대리(WDR) 자동매매 시작 ===")

    if not _is_kr_order_window():
        logger.info("국내 주문 가능 시간 외 (09:00~15:20 KST). 매매 생략.")
        return

    isa_key = _get_env_any("KIS_ISA_APP_KEY", "KIS_APP_KEY")
    isa_secret = _get_env_any("KIS_ISA_APP_SECRET", "KIS_APP_SECRET")
    isa_account_raw = _get_env_any("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    isa_prdt_raw = _get_env_any("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    isa_account, isa_prdt = _normalize_kis_account_fields(isa_account_raw, isa_prdt_raw)

    # 트레이더 초기화
    trader = KISTrader(is_mock=False)
    if isa_key:
        trader.app_key = isa_key
    if isa_secret:
        trader.app_secret = isa_secret
    if isa_account:
        trader.account_no = isa_account
        trader.acnt_prdt_cd = isa_prdt

    if not trader.auth():
        logger.error("KIS 인증 실패. 종료.")
        return

    etf_code = os.getenv("KIS_ISA_ETF_CODE", "418660")  # 매매 ETF
    signal_etf_code = os.getenv("KIS_ISA_TREND_ETF_CODE", "133690")  # TREND ETF (시그널 소스)

    # WDR 전략 설정 (환경변수에서 오버라이드 가능)
    wdr_settings = {}
    if os.getenv("WDR_OVERVALUE_THRESHOLD"):
        wdr_settings['overvalue_threshold'] = float(os.getenv("WDR_OVERVALUE_THRESHOLD"))
    if os.getenv("WDR_UNDERVALUE_THRESHOLD"):
        wdr_settings['undervalue_threshold'] = float(os.getenv("WDR_UNDERVALUE_THRESHOLD"))
    wdr_eval_mode = int(os.getenv("WDR_EVAL_MODE", "3"))

    strategy = WDRStrategy(wdr_settings if wdr_settings else None, evaluation_mode=wdr_eval_mode)

    # ── 1. 시그널 소스(TREND ETF) 일봉 데이터 조회 ──
    logger.info(f"시그널 소스 ETF({signal_etf_code}) 일봉 데이터 조회 중...")
    signal_df = _get_kis_daily_local_first(trader, signal_etf_code, count=1500)
    if signal_df is None or len(signal_df) < 260 * 5:
        # API 실패 시 번들 CSV fallback
        logger.warning("API 데이터 부족 → 번들 CSV fallback 시도...")
        from data_cache import load_bundled_csv
        signal_df = load_bundled_csv(signal_etf_code)
    if signal_df is None or len(signal_df) < 260 * 5:
        logger.error(f"{signal_etf_code} 데이터 부족 (got {len(signal_df) if signal_df is not None else 0}, "
                     f"need {260 * 5}). 종료.")
        return

    # ── 2. WDR 시그널 분석 ──
    signal = strategy.analyze(signal_df)
    if signal is None:
        logger.error("WDR 시그널 분석 실패. 종료.")
        return

    logger.info(f"WDR 시그널: 이격도={signal['divergence']}%, "
                f"상태={signal['state']}, {signal_etf_code}={signal['qqq_price']}, "
                f"추세선={signal['trend']}")
    logger.info(f"  매도비율={signal['sell_ratio']}%, 매수비율={signal['buy_ratio']}%")

    # ── 3. 잔고 확인 ──
    bal = trader.get_balance()
    if bal is None:
        logger.error("잔고 조회 실패. 종료.")
        return

    cash = bal['cash']
    holdings = bal['holdings']

    # 현재 ETF 보유 정보
    etf_holding = None
    for h in holdings:
        if h['code'] == etf_code:
            etf_holding = h
            break

    current_shares = etf_holding['qty'] if etf_holding else 0
    current_price = _get_kis_price_local_first(trader, etf_code) or 0

    if current_price <= 0:
        logger.error(f"ETF({etf_code}) 현재가 조회 실패. 종료.")
        return

    etf_value = current_shares * current_price
    total_value = cash + etf_value

    logger.info(f"잔고: 예수금={cash:,.0f}원, ETF={current_shares}주 "
                f"(≈{etf_value:,.0f}원), 총자산={total_value:,.0f}원")

    # ── 4. 백테스트 기반 목표 비율 계산 (비중 기반 매매) ──
    isa_start_date = os.getenv("KIS_ISA_START_DATE", "2022-03-08")
    trade_df = _get_kis_daily_local_first(trader, etf_code, count=1500)

    bt_stock_ratio = None
    order_action = None
    order_qty = 0

    if trade_df is not None and len(trade_df) >= 60:
        # effective_start = max(시작일, 매매 ETF 상장일)
        _trade_first_date = str(trade_df.index[0].date())
        _effective_start = isa_start_date
        if _effective_start < _trade_first_date:
            _effective_start = _trade_first_date

        # 참조 백테스트 (비표준 ETF → 133690→418660 기준 initial_stock_ratio 추출)
        _ref_stock_ratio = None
        _REF_TRADE = "418660"
        if str(etf_code) != _REF_TRADE and _effective_start > "2022-03-08":
            _ref_trade_df = _get_kis_daily_local_first(trader, _REF_TRADE, count=1500)
            if signal_df is not None and _ref_trade_df is not None:
                _ref_bt = strategy.run_backtest(
                    signal_daily_df=signal_df,
                    trade_daily_df=_ref_trade_df,
                    initial_balance=10_000_000,
                    start_date="2022-03-08",
                )
                if _ref_bt and _ref_bt.get("equity_df") is not None:
                    _ref_eq = _ref_bt["equity_df"]
                    _eff_ts = pd.Timestamp(_effective_start)
                    _ref_mask = _ref_eq.index <= _eff_ts
                    if _ref_mask.any():
                        _ref_row = _ref_eq.loc[_ref_mask].iloc[-1]
                        _ref_equity = float(_ref_row["equity"])
                        _ref_sv = float(_ref_row["shares"]) * float(_ref_row["price"])
                        if _ref_equity > 0:
                            _ref_stock_ratio = _ref_sv / _ref_equity
                            logger.info(f"참조 백테스트 비율: {_ref_stock_ratio*100:.1f}% "
                                        f"(133690→418660, {_effective_start} 시점)")

        bt = strategy.run_backtest(
            signal_daily_df=signal_df,
            trade_daily_df=trade_df,
            initial_balance=10_000_000,
            start_date=_effective_start,
            initial_stock_ratio=_ref_stock_ratio,
        )

        if bt and bt.get("equity_df") is not None:
            eq_df = bt["equity_df"]
            bt_last = eq_df.iloc[-1]
            bt_shares = int(bt_last["shares"])
            bt_price = float(bt_last["price"])
            bt_equity = float(bt_last["equity"])
            if bt_equity > 0 and bt_price > 0:
                bt_stock_ratio = (bt_shares * bt_price) / bt_equity
                logger.info(f"백테스트 목표비율: 주식 {bt_stock_ratio*100:.1f}% "
                            f"(bt_shares={bt_shares}, bt_equity={bt_equity:,.0f})")

    # ── 5. 비중 기반 매매 판단 ──
    MIN_ORDER_KRW_KIS = 10_000

    if bt_stock_ratio is not None:
        target_stock_val = total_value * bt_stock_ratio
        target_shares = int(target_stock_val / current_price) if current_price > 0 else 0
        diff = target_shares - current_shares

        logger.info(f"목표비율 매매: 목표주식비율={bt_stock_ratio*100:.1f}%, "
                    f"목표가치={target_stock_val:,.0f}원, 목표주수={target_shares}, "
                    f"현재주수={current_shares}, 차이={diff:+d}주")

        if diff > 0:
            affordable = int(cash * 0.999 / current_price) if current_price > 0 else 0
            order_qty = min(diff, affordable)
            if order_qty > 0 and (order_qty * current_price) >= MIN_ORDER_KRW_KIS:
                order_action = "BUY"
            else:
                logger.info(f"[BUY] 매수 불가 (필요={diff}주, 가용={affordable}주)")
        elif diff < 0:
            order_qty = abs(diff)
            if (order_qty * current_price) >= MIN_ORDER_KRW_KIS:
                order_action = "SELL"
            else:
                logger.info(f"[SELL] 매도 금액 미달 ({order_qty * current_price:,.0f}원 < {MIN_ORDER_KRW_KIS:,}원)")
        else:
            logger.info("[HOLD] 목표 비율과 일치 - 유지")
    else:
        # 폴백: 백테스트 실패 시 기존 P&L 방식 사용
        logger.warning("백테스트 실패 → P&L 기반 폴백 사용")
        etf_chart = _get_kis_daily_local_first(trader, etf_code, count=10)
        weekly_pnl = 0.0
        if etf_chart is not None and len(etf_chart) >= 5 and current_shares > 0:
            price_5d_ago = float(etf_chart['close'].iloc[-5])
            weekly_pnl = (current_price - price_5d_ago) * current_shares
        fb_action = strategy.get_rebalance_action(
            weekly_pnl=weekly_pnl,
            divergence=signal['divergence'],
            current_shares=current_shares,
            current_price=current_price,
            cash=cash,
        )
        order_action = fb_action['action'] if fb_action['action'] else None
        order_qty = int(fb_action['quantity'])
        logger.info(f"폴백 판단: action={order_action}, qty={order_qty}")

    # ── 6. 매매 실행 ──
    if order_action == 'SELL' and order_qty > 0:
        sell_value = order_qty * current_price
        logger.info(f"[SELL] {etf_code} {order_qty}주 "
                    f"(≈{sell_value:,.0f}원) 동시호가 매도")
        result = trader.smart_sell_qty_closing(etf_code, order_qty)
        logger.info(f"매도 결과: {result}")

    elif order_action == 'BUY' and order_qty > 0:
        buy_value = order_qty * current_price
        logger.info(f"[BUY] {etf_code} {order_qty}주 "
                    f"(≈{buy_value:,.0f}원) 동시호가 매수")
        result = trader.execute_closing_auction_buy(etf_code, order_qty)
        logger.info(f"매수 결과: {result}")

    else:
        logger.info("[HOLD] 리밸런싱 불필요 또는 수량 0 - 유지")

    logger.info("=== KIS ISA 위대리(WDR) 자동매매 완료 ===")

    # 텔레그램 요약
    tg = [f"<b>KIS ISA 위대리</b>"]
    tg.append(f"총자산: {total_value:,.0f}원 (현금 {cash:,.0f} / ETF {current_shares}주)")
    tg.append(f"이격도: {signal['divergence']}% | 상태: {signal['state']}")
    if bt_stock_ratio is not None:
        tg.append(f"목표비율: {bt_stock_ratio*100:.0f}% | 판단: {order_action or 'HOLD'} {order_qty}주")
    else:
        tg.append(f"판단(폴백): {order_action or 'HOLD'} {order_qty}주")
    _send_telegram("\n".join(tg))


# ─────────────────────────────────────────────────────────
# KIS 연금저축 듀얼모멘텀 자동매매
# ─────────────────────────────────────────────────────────
def run_kis_pension_trade():
    """
    한국투자증권 연금저축 계좌 - 듀얼모멘텀(GEM) 자동매매.

    전략 요약:
      - SPY/EFA 상대 모멘텀 → 승자 선택
      - BIL 절대 모멘텀 → 위험 자산 vs 안전 자산 결정
      - 모멘텀 약세 시 AGG(채권)로 전환
      - 월 1회 리밸런싱
      - 시그널/집행 모두 국내 ETF 일봉 기반

    필요 환경변수:
      KIS_PENSION_APP_KEY, KIS_PENSION_APP_SECRET (또는 KIS_APP_KEY 공유)
      KIS_PENSION_ACCOUNT_NO, KIS_PENSION_ACNT_PRDT_CD
    """
    load_dotenv()
    logger.info("=== KIS 연금저축 듀얼모멘텀(GEM) 자동매매 시작 ===")

    if not _is_kr_order_window():
        logger.info("국내 주문 가능 시간 외 (09:00~15:20 KST). 매매 생략.")
        return

    pension_key = _get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
    pension_secret = _get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
    pension_acct_raw = _get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    pension_prdt_raw = _get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    pension_acct, pension_prdt = _normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)

    # 연금저축 전용 트레이더 (ISA와 계좌가 다를 수 있음)
    trader = KISTrader(is_mock=False)
    if pension_key:
        trader.app_key = pension_key
    if pension_secret:
        trader.app_secret = pension_secret
    if pension_acct:
        trader.account_no = pension_acct
        trader.acnt_prdt_cd = pension_prdt

    if not trader.auth():
        logger.error("KIS 인증 실패. 종료.")
        return

    # ETF 코드 매핑 (환경변수로 오버라이드 가능)
    kr_etf_map = {
        'SPY': os.getenv("KR_ETF_SPY", "360750"),    # TIGER 미국S&P500
        'EFA': os.getenv("KR_ETF_EFA", "453850"),    # TIGER 선진국MSCI World
        'AGG': os.getenv("KR_ETF_AGG", "453540"),    # TIGER 미국채10년선물
        'BIL': os.getenv("KR_ETF_BIL", os.getenv("KR_ETF_SHY", "114470")),  # 카나리아 대체
    }

    strategy = DualMomentumStrategy(settings={'kr_etf_map': kr_etf_map})

    # ── 1. 국내 ETF 일봉 데이터 조회 (시그널용) ──
    tickers_to_fetch = ['SPY', 'EFA', 'BIL', 'AGG']
    price_data = {}

    for ticker in tickers_to_fetch:
        code = kr_etf_map.get(ticker, "")
        logger.info(f"{ticker}({code}) 일봉 데이터 조회 중...")
        df = _get_kis_daily_local_first(trader, code, count=300)
        if df is not None and len(df) > 0:
            price_data[ticker] = df
            logger.info(f"  {ticker}({code}): {len(df)}일 로드 완료")
        else:
            logger.error(f"  {ticker}({code}): 데이터 조회 실패")

        time.sleep(0.3)  # API 호출 간격

    # ── 2. 듀얼모멘텀 시그널 분석 ──
    signal = strategy.analyze(price_data)
    if signal is None:
        logger.error("듀얼모멘텀 시그널 분석 실패 (데이터 부족). 종료.")
        return

    logger.info(f"듀얼모멘텀 시그널:")
    logger.info(f"  대상: {signal['target_ticker']} → KR ETF: {signal['target_kr_code']}")
    logger.info(f"  공격자산: {signal['is_offensive']}")
    logger.info(f"  모멘텀 점수: {signal['scores']}")
    logger.info(f"  카나리아(BIL) 수익률: {signal['canary_return']:.4f}")
    logger.info(f"  판단: {signal['reason']}")

    target_kr_code = signal['target_kr_code']
    if not target_kr_code:
        logger.error("대상 한국 ETF 코드 없음. 종료.")
        return

    # ── 3. 잔고 확인 ──
    bal = trader.get_balance()
    if bal is None:
        logger.error("잔고 조회 실패. 종료.")
        return

    cash = bal['cash']
    holdings = bal['holdings']
    total_eval = bal['total_eval'] or (cash + sum(h['eval_amt'] for h in holdings))

    logger.info(f"잔고: 예수금={cash:,.0f}원, 총평가={total_eval:,.0f}원")
    for h in holdings:
        logger.info(f"  {h['name']}({h['code']}): {h['qty']}주, "
                    f"평가={h['eval_amt']:,.0f}원, 수익률={h['pnl_rate']:.1f}%")

    # ── 4. 리밸런싱: 현재 보유와 목표 비교 ──
    # 현재 대상 ETF 보유 여부
    all_kr_codes = set(kr_etf_map.values())
    target_holding = None
    other_holdings = []

    for h in holdings:
        if h['code'] == target_kr_code:
            target_holding = h
        elif h['code'] in all_kr_codes:
            other_holdings.append(h)

    MIN_ORDER_KRW_KIS = 10_000

    # 이미 목표 종목만 보유 중이며 잔여 현금도 거의 없으면 리밸런싱 불필요
    if target_holding and not other_holdings and cash < MIN_ORDER_KRW_KIS:
        logger.info(f"[HOLD] 이미 목표 종목({target_kr_code}) 보유 중. 리밸런싱 불필요.")
        logger.info("=== KIS 연금저축 듀얼모멘텀(GEM) 자동매매 완료 ===")
        _send_telegram(f"<b>KIS 연금저축 듀얼모멘텀</b>\n대상: {signal['target_ticker']}→{target_kr_code}\n[HOLD] 이미 보유 중. 리밸런싱 불필요.")
        return
    elif target_holding and not other_holdings:
        logger.info(f"[TOP-UP] 목표 종목({target_kr_code}) 보유 중이나 잔여 현금 {cash:,.0f}원 존재 → 추가 매수 진행")

    # ── 5. 매매 실행: 기존 비대상 종목 매도 → 대상 종목 매수 ──
    # Step 1: 비대상 종목 전량 매도
    for h in other_holdings:
        logger.info(f"[SELL] {h['name']}({h['code']}) {h['qty']}주 전량 동시호가 매도")
        result = trader.smart_sell_all_closing(h['code'])
        logger.info(f"  매도 결과: {result}")
        time.sleep(1)

    # 매도 체결 대기
    if other_holdings:
        logger.info("매도 체결 대기 (3초)...")
        time.sleep(3)

    # Step 2: 잔고 재조회 후 대상 종목 매수
    bal = trader.get_balance()
    if bal is None:
        logger.error("매도 후 잔고 조회 실패.")
        return

    cash = bal['cash']

    # 이미 보유 중인 대상 종목 가치 확인
    # 추가 매수 필요 금액 = 가용 현금 전체 (이미 보유분 제외 안함 - 100% 배분)
    buy_amount = cash * 0.999  # 수수료 여유분

    if buy_amount >= MIN_ORDER_KRW_KIS:
        logger.info(f"[BUY] {target_kr_code} {buy_amount:,.0f}원 동시호가 매수")
        result = trader.smart_buy_krw_closing(target_kr_code, buy_amount)
        logger.info(f"  매수 결과: {result}")
    else:
        logger.info(f"[BUY] 매수 가능 금액 부족 ({cash:,.0f}원)")

    logger.info("=== KIS 연금저축 듀얼모멘텀(GEM) 자동매매 완료 ===")

    # 텔레그램 요약
    tg = [f"<b>KIS 연금저축 듀얼모멘텀</b>"]
    tg.append(f"대상: {signal['target_ticker']}→{target_kr_code}")
    tg.append(f"사유: {signal['reason']}")
    if other_holdings:
        tg.append(f"[SELL] {len(other_holdings)}종목 매도 → [BUY] {target_kr_code}")
    elif buy_amount >= MIN_ORDER_KRW_KIS:
        tg.append(f"[BUY] {target_kr_code} {buy_amount:,.0f}원")
    else:
        tg.append(f"[BUY] 금액 부족 ({cash:,.0f}원)")
    _send_telegram("\n".join(tg))


# ─────────────────────────────────────────────────────────
# 일일 헬스체크 (가상주문 점검)
# ─────────────────────────────────────────────────────────
def _check_kiwoom_gold() -> dict:
    """키움 금현물 시스템 헬스체크."""
    result = {
        'name': '키움 금현물',
        'auth': False, 'auth_msg': '',
        'balance': False, 'balance_msg': '',
        'price': False, 'price_msg': '',
        'signal': False, 'signal_msg': '',
        'order_test': False, 'order_msg': '',
    }

    try:
        trader = KiwoomGoldTrader(is_mock=False)

        # 1. 인증
        if not trader.auth():
            result['auth_msg'] = 'FAIL - 인증 실패'
            return result
        result['auth'] = True
        result['auth_msg'] = 'PASS'

        code = GOLD_CODE_1KG

        # 2. 잔고 조회
        bal = trader.get_balance()
        if bal is None:
            result['balance_msg'] = 'FAIL - 잔고 조회 실패'
            return result
        result['balance'] = True
        result['balance_msg'] = f"현금 {bal['cash_krw']:,.0f}원 / 금 {bal['gold_qty']}g"

        # 3. 시세 조회
        price = _get_gold_price_local_first(trader, code)
        if not price or price <= 0:
            result['price_msg'] = 'FAIL - 시세 조회 실패'
            return result
        result['price'] = True
        result['price_msg'] = f"미니금 현재가 {price:,.0f}원"

        # 4. 시그널 체크
        gold_portfolio = get_gold_portfolio()
        total_weight = sum(g['weight'] for g in gold_portfolio)
        max_period = max(g['buy_period'] for g in gold_portfolio)
        df = _get_gold_daily_local_first(trader, code=code, count=max(max_period + 10, 200))

        if df is None or len(df) < max_period + 5:
            csv_path = os.path.join(os.path.dirname(__file__), "krx_gold_daily.csv")
            try:
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                if "open" not in df.columns: df["open"] = df["close"]
                if "high" not in df.columns: df["high"] = df["close"]
                if "low"  not in df.columns: df["low"]  = df["close"]
            except Exception:
                df = None

        if df is not None and len(df) >= max_period + 5:
            buy_weight = 0
            for gp in gold_portfolio:
                bp = gp['buy_period']
                sp = gp.get('sell_period', 0) or max(5, bp // 2)
                w = gp['weight']
                if gp['strategy'] == "Donchian":
                    strat = DonchianStrategy()
                    df_s = strat.create_features(df.copy(), buy_period=bp, sell_period=sp)
                    signal = strat.get_signal(df_s.iloc[-2], buy_period=bp, sell_period=sp)
                else:
                    sma = df['close'].rolling(window=bp).mean()
                    signal = "BUY" if float(df['close'].iloc[-2]) > float(sma.iloc[-2]) else "SELL"
                if signal == "BUY":
                    buy_weight += w

            target_ratio = buy_weight / total_weight if total_weight > 0 else 0
            gold_value = bal['gold_qty'] * price
            total_value = bal['cash_krw'] + gold_value
            current_ratio = gold_value / total_value if total_value > 0 else 0
            diff = abs(target_ratio - current_ratio)

            if diff < 0.05:
                action_str = "HOLD"
            elif target_ratio > current_ratio:
                action_str = "BUY"
            else:
                action_str = "SELL"

            result['signal'] = True
            result['signal_msg'] = (f"{action_str} (목표 {target_ratio:.0%}, 현재 {current_ratio:.0%})")
        else:
            result['signal_msg'] = 'SKIP - 데이터 부족'

        # 5. 가상주문 왕복 테스트 (하한가 매수 1g → 조회 → 취소)
        if not _is_kr_order_window():
            result['order_msg'] = 'SKIP - 주문 가능 시간이 아닙니다 (09:00~15:20 KST)'
            return result

        limit_price = trader._get_limit_price(code, "SELL")  # 하한가 (체결 불가 가격)
        if limit_price <= 0:
            result['order_msg'] = 'FAIL - 하한가 계산 실패'
            return result

        order_result = trader.send_order("BUY", code, qty=1, price=limit_price, ord_tp="1")
        if not order_result or not order_result.get('success'):
            result['order_msg'] = f"FAIL - 주문 실패: {order_result}"
            return result

        ord_no = order_result.get('ord_no', '')
        time.sleep(2)

        # 미체결 조회
        pending = trader.get_pending_orders(code) or []
        found = any(p.get('ord_no') == ord_no for p in pending) if pending else bool(ord_no)

        # 취소
        cancel_result = trader.cancel_order(ord_no, code, qty=1)
        cancel_ok = bool(cancel_result and (cancel_result.get('success') if isinstance(cancel_result, dict) else cancel_result))
        time.sleep(1)

        # 취소 확인
        pending_after = trader.get_pending_orders(code) or []
        still_there = any(p.get('ord_no') == ord_no for p in pending_after) if pending_after else False

        if found and cancel_ok and not still_there:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 정상'
        elif found and cancel_ok:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 완료 (잔여 확인 필요)'
        else:
            result['order_msg'] = f"주문={bool(ord_no)}, 조회={found}, 취소={cancel_ok}"

    except Exception as e:
        result['order_msg'] = result.get('order_msg') or f'ERROR: {e}'
        logger.error(f"키움 금현물 헬스체크 오류: {e}")

    return result


def _check_kis_isa() -> dict:
    """KIS ISA 위대리 시스템 헬스체크."""
    result = {
        'name': 'KIS ISA 위대리',
        'auth': False, 'auth_msg': '',
        'balance': False, 'balance_msg': '',
        'price': False, 'price_msg': '',
        'signal': False, 'signal_msg': '',
        'order_test': False, 'order_msg': '',
    }

    try:
        trader = KISTrader(is_mock=False)
        isa_key = _get_env_any("KIS_ISA_APP_KEY", "KIS_APP_KEY")
        isa_secret = _get_env_any("KIS_ISA_APP_SECRET", "KIS_APP_SECRET")
        isa_acct_raw = _get_env_any("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO")
        isa_prdt_raw = _get_env_any("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
        isa_account, isa_prdt = _normalize_kis_account_fields(isa_acct_raw, isa_prdt_raw)

        if isa_key:
            trader.app_key = isa_key
        if isa_secret:
            trader.app_secret = isa_secret
        if isa_account:
            trader.account_no = isa_account
            trader.acnt_prdt_cd = isa_prdt

        # 1. 인증
        if not trader.auth():
            result['auth_msg'] = 'FAIL - 인증 실패'
            return result
        result['auth'] = True
        result['auth_msg'] = 'PASS'

        etf_code = os.getenv("KIS_ISA_ETF_CODE", "418660")
        signal_etf_code = os.getenv("KIS_ISA_TREND_ETF_CODE", "133690")

        # 2. 잔고 조회
        bal = trader.get_balance()
        if bal is None:
            result['balance_msg'] = 'FAIL - 잔고 조회 실패'
            return result
        result['balance'] = True

        cash = bal['cash']
        etf_holding = None
        for h in bal['holdings']:
            if h['code'] == etf_code:
                etf_holding = h
                break
        shares = etf_holding['qty'] if etf_holding else 0
        holding_str = f"{etf_holding['name']} {shares}주" if etf_holding else "미보유"
        result['balance_msg'] = f"현금 {cash:,.0f}원 / {holding_str}"

        # 3. 시세 조회
        price = _get_kis_price_local_first(trader, etf_code)
        if not price or price <= 0:
            result['price_msg'] = 'FAIL - 시세 조회 실패'
            return result
        result['price'] = True
        result['price_msg'] = f"{etf_code} 현재가 {price:,.0f}원"

        # 4. WDR 시그널 분석
        wdr_settings = {}
        if os.getenv("WDR_OVERVALUE_THRESHOLD"):
            wdr_settings['overvalue_threshold'] = float(os.getenv("WDR_OVERVALUE_THRESHOLD"))
        if os.getenv("WDR_UNDERVALUE_THRESHOLD"):
            wdr_settings['undervalue_threshold'] = float(os.getenv("WDR_UNDERVALUE_THRESHOLD"))
        wdr_eval_mode = int(os.getenv("WDR_EVAL_MODE", "3"))
        strategy = WDRStrategy(wdr_settings if wdr_settings else None, evaluation_mode=wdr_eval_mode)

        signal_df = _get_kis_daily_local_first(trader, signal_etf_code, count=1500)
        if signal_df is None or len(signal_df) < 260 * 5:
            from data_cache import load_bundled_csv
            signal_df = load_bundled_csv(signal_etf_code)

        if signal_df is not None and len(signal_df) >= 260 * 5:
            sig = strategy.analyze(signal_df)
            if sig:
                # 비중 기반 목표비율 매매 판단
                bt_msg = ""
                isa_start = os.getenv("KIS_ISA_START_DATE", "2022-03-08")
                trade_df = _get_kis_daily_local_first(trader, etf_code, count=1500)
                if trade_df is not None and len(trade_df) >= 60:
                    _eff_start = isa_start
                    _tfd = str(trade_df.index[0].date())
                    if isa_start < _tfd:
                        _eff_start = _tfd

                    _ref_sr = None
                    if str(etf_code) != "418660" and _eff_start > "2022-03-08":
                        _ref_trade = _get_kis_daily_local_first(trader, "418660", count=1500)
                        if _ref_trade is not None:
                            _ref_bt = strategy.run_backtest(
                                signal_daily_df=signal_df, trade_daily_df=_ref_trade,
                                initial_balance=10_000_000, start_date="2022-03-08",
                            )
                            if _ref_bt and _ref_bt.get("equity_df") is not None:
                                _re = _ref_bt["equity_df"]
                                _ets = pd.Timestamp(_eff_start)
                                _rm = _re.index <= _ets
                                if _rm.any():
                                    _rr = _re.loc[_rm].iloc[-1]
                                    _req = float(_rr["equity"])
                                    if _req > 0:
                                        _ref_sr = (float(_rr["shares"]) * float(_rr["price"])) / _req

                    bt = strategy.run_backtest(
                        signal_daily_df=signal_df, trade_daily_df=trade_df,
                        initial_balance=10_000_000, start_date=_eff_start,
                        initial_stock_ratio=_ref_sr,
                    )
                    if bt and bt.get("equity_df") is not None:
                        _last = bt["equity_df"].iloc[-1]
                        _bt_eq = float(_last["equity"])
                        if _bt_eq > 0 and price > 0:
                            _ratio = (float(_last["shares"]) * float(_last["price"])) / _bt_eq
                            total = cash + shares * price
                            tgt = int(total * _ratio / price)
                            diff = tgt - shares
                            if diff > 0:
                                affordable = min(diff, int(cash * 0.999 / price)) if price > 0 else 0
                                bt_msg = f"BUY {affordable}주 (목표비율 {_ratio*100:.0f}%)"
                            elif diff < 0:
                                bt_msg = f"SELL {abs(diff)}주 (목표비율 {_ratio*100:.0f}%)"
                            else:
                                bt_msg = f"HOLD (목표비율 {_ratio*100:.0f}% 일치)"

                kst = timezone(timedelta(hours=9))
                is_friday = datetime.now(kst).weekday() == 4
                day_note = "오늘 금요일=리밸런싱일" if is_friday else "오늘 리밸런싱일 아님"

                result['signal'] = True
                result['signal_msg'] = (
                    f"{bt_msg or 'HOLD 0주'} "
                    f"(이격도={sig['divergence']}%, 상태={sig['state']}) [{day_note}]"
                )
            else:
                result['signal_msg'] = 'FAIL - WDR 분석 실패'
        else:
            result['signal_msg'] = 'SKIP - 시그널 데이터 부족'

        # 5. 가상주문 왕복 테스트 (하한가 매수 1주 → 조회 → 취소)
        if not _is_kr_order_window():
            result['order_msg'] = 'SKIP - 주문 가능 시간이 아닙니다 (09:00~15:20 KST)'
            return result

        limit_price = trader._get_limit_price(etf_code, "SELL")  # 하한가
        if limit_price <= 0:
            result['order_msg'] = 'FAIL - 하한가 계산 실패'
            return result

        order_result = trader.send_order("BUY", etf_code, qty=1, price=limit_price, ord_dvsn="00")
        if not order_result or not order_result.get('success'):
            result['order_msg'] = f"FAIL - 주문 실패: {order_result}"
            return result

        ord_no = order_result.get('ord_no', '')
        time.sleep(2)

        pending = trader.get_pending_orders(etf_code) or []
        found = any(p.get('ord_no') == ord_no for p in pending) if pending else bool(ord_no)

        cancel_result = trader.cancel_order(ord_no, etf_code)
        cancel_ok = bool(cancel_result and (cancel_result.get('success') if isinstance(cancel_result, dict) else cancel_result))
        time.sleep(1)

        pending_after = trader.get_pending_orders(etf_code) or []
        still_there = any(p.get('ord_no') == ord_no for p in pending_after) if pending_after else False

        if found and cancel_ok and not still_there:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 정상'
        elif found and cancel_ok:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 완료 (잔여 확인 필요)'
        else:
            result['order_msg'] = f"주문={bool(ord_no)}, 조회={found}, 취소={cancel_ok}"

    except Exception as e:
        result['order_msg'] = result.get('order_msg') or f'ERROR: {e}'
        logger.error(f"KIS ISA 헬스체크 오류: {e}")

    return result


def _check_kis_pension() -> dict:
    """KIS 연금저축 듀얼모멘텀 시스템 헬스체크."""
    result = {
        'name': 'KIS 연금저축 듀얼모멘텀',
        'auth': False, 'auth_msg': '',
        'balance': False, 'balance_msg': '',
        'price': False, 'price_msg': '',
        'signal': False, 'signal_msg': '',
        'order_test': False, 'order_msg': '',
    }

    try:
        trader = KISTrader(is_mock=False)
        # 연금저축 전용 키 오버라이드
        pension_key = _get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
        pension_secret = _get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
        if pension_key and pension_secret:
            trader.app_key = pension_key
            trader.app_secret = pension_secret
        pension_acct_raw = _get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
        pension_prdt_raw = _get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
        pension_acct, pension_prdt = _normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)
        if pension_acct:
            trader.account_no = pension_acct
            trader.acnt_prdt_cd = pension_prdt

        # 1. 인증
        if not trader.auth():
            result['auth_msg'] = 'FAIL - 인증 실패'
            return result
        result['auth'] = True
        result['auth_msg'] = 'PASS'

        kr_etf_map = {
            'SPY': os.getenv("KR_ETF_SPY", "360750"),
            'EFA': os.getenv("KR_ETF_EFA", "453850"),
            'AGG': os.getenv("KR_ETF_AGG", "453540"),
            'BIL': os.getenv("KR_ETF_BIL", os.getenv("KR_ETF_SHY", "114470")),
        }

        # 2. 잔고 조회
        bal = trader.get_balance()
        if bal is None:
            result['balance_msg'] = 'FAIL - 잔고 조회 실패'
            return result
        result['balance'] = True

        cash = bal['cash']
        holdings = bal['holdings']
        total_eval = bal['total_eval'] or (cash + sum(h['eval_amt'] for h in holdings))
        if holdings:
            h_str = ", ".join(f"{h['name']} {h['qty']}주" for h in holdings)
        else:
            h_str = "미보유"
        result['balance_msg'] = f"현금 {cash:,.0f}원 / {h_str} / 총 {total_eval:,.0f}원"

        # 3. 국내 시세 조회 (듀얼모멘텀 시그널용)
        tickers = ['SPY', 'EFA', 'BIL', 'AGG']
        prices = {}
        price_data = {}

        for ticker in tickers:
            code = kr_etf_map.get(ticker, "")
            df = _get_kis_daily_local_first(trader, code, count=300)
            if df is not None and len(df) > 0:
                prices[ticker] = float(df['close'].iloc[-1])
                price_data[ticker] = df
            else:
                prices[ticker] = None
            time.sleep(0.3)

        if all(v is not None for v in prices.values()):
            result['price'] = True
            result['price_msg'] = ", ".join(f"{t}={p:.1f}" for t, p in prices.items())
        else:
            failed = [t for t, p in prices.items() if p is None]
            result['price_msg'] = f"FAIL - {', '.join(failed)} 조회 실패"
            return result

        # 4. 듀얼모멘텀 시그널 분석
        strategy = DualMomentumStrategy(settings={'kr_etf_map': kr_etf_map})
        sig = strategy.analyze(price_data)
        if sig:
            # 리밸런싱 필요 여부 판단
            all_kr_codes = set(kr_etf_map.values())
            target_kr_code = sig['target_kr_code']
            target_holding = None
            other_holdings = []
            for h in holdings:
                if h['code'] == target_kr_code:
                    target_holding = h
                elif h['code'] in all_kr_codes:
                    other_holdings.append(h)

            if target_holding and not other_holdings:
                action_str = f"HOLD ({target_kr_code} 보유 중)"
            elif other_holdings:
                action_str = f"REBALANCE ({target_kr_code}로 전환 필요)"
            else:
                action_str = f"BUY {target_kr_code}"

            kst = timezone(timedelta(hours=9))
            now = datetime.now(kst)
            is_rebal_window = 25 <= now.day <= 31
            day_note = "리밸런싱 기간" if is_rebal_window else "리밸런싱 기간 아님"

            result['signal'] = True
            result['signal_msg'] = (
                f"{action_str} | 대상={sig['target_ticker']}→{target_kr_code} "
                f"({sig['reason']}) [{day_note}]"
            )
        else:
            result['signal_msg'] = 'FAIL - 듀얼모멘텀 분석 실패'

        # 5. 가상주문 왕복 테스트
        if not _is_kr_order_window():
            result['order_msg'] = 'SKIP - 주문 가능 시간이 아닙니다 (09:00~15:20 KST)'
            return result

        # 연금저축 계좌의 대표 종목으로 테스트
        test_code = kr_etf_map['SPY']  # TIGER 미국S&P500
        test_price = _get_kis_price_local_first(trader, test_code)
        if not test_price or test_price <= 0:
            result['order_msg'] = 'FAIL - 테스트 종목 시세 조회 실패'
            return result

        limit_price = trader._get_limit_price(test_code, "SELL")  # 하한가
        if limit_price <= 0:
            result['order_msg'] = 'FAIL - 하한가 계산 실패'
            return result

        order_result = trader.send_order("BUY", test_code, qty=1, price=limit_price, ord_dvsn="00")
        if not order_result or not order_result.get('success'):
            result['order_msg'] = f"FAIL - 주문 실패: {order_result}"
            return result

        ord_no = order_result.get('ord_no', '')
        time.sleep(2)

        pending = trader.get_pending_orders(test_code) or []
        found = any(p.get('ord_no') == ord_no for p in pending) if pending else bool(ord_no)

        cancel_result = trader.cancel_order(ord_no, test_code)
        cancel_ok = bool(cancel_result and (cancel_result.get('success') if isinstance(cancel_result, dict) else cancel_result))
        time.sleep(1)

        pending_after = trader.get_pending_orders(test_code) or []
        still_there = any(p.get('ord_no') == ord_no for p in pending_after) if pending_after else False

        if found and cancel_ok and not still_there:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 정상'
        elif found and cancel_ok:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 완료 (잔여 확인 필요)'
        else:
            result['order_msg'] = f"주문={bool(ord_no)}, 조회={found}, 취소={cancel_ok}"

    except Exception as e:
        result['order_msg'] = result.get('order_msg') or f'ERROR: {e}'
        logger.error(f"KIS 연금저축 헬스체크 오류: {e}")

    return result


def _check_upbit() -> dict:
    """업비트 코인 시스템 헬스체크."""
    result = {
        'name': '업비트 코인',
        'auth': False, 'auth_msg': '',
        'balance': False, 'balance_msg': '',
        'price': False, 'price_msg': '',
        'signal': False, 'signal_msg': '',
        'order_test': False, 'order_msg': '',
        'steps': [],
    }

    try:
        ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
        SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
        _append_step(
            result,
            "0. 입력값 확인",
            "INFO",
            f"UPBIT_ACCESS_KEY={_mask_secret(ACCESS_KEY, 5, 3)}, "
            f"UPBIT_SECRET_KEY={_mask_secret(SECRET_KEY, 5, 3)}"
        )
        if not ACCESS_KEY or not SECRET_KEY:
            result['auth_msg'] = 'FAIL - API 키 미설정'
            _append_step(result, "1. 인증", "FAIL", result['auth_msg'])
            return result

        trader = UpbitTrader(ACCESS_KEY, SECRET_KEY)
        _append_step(result, "1. 클라이언트 생성", "PASS", "UpbitTrader 인스턴스 생성 완료")

        # 1. 인증 + 2. 잔고 조회 (한 번의 API 호출로 통합)
        all_bal = trader.get_all_balances()
        if all_bal is None:
            result['auth_msg'] = 'FAIL - 인증/잔고 조회 실패 (응답=None)'
            _append_step(result, "2. 인증/잔고 API", "FAIL", result['auth_msg'])
            return result
        if not isinstance(all_bal, dict):
            result['auth_msg'] = f"FAIL - 인증/잔고 조회 실패 (응답타입={type(all_bal).__name__})"
            _append_step(result, "2. 인증/잔고 API", "FAIL", result['auth_msg'])
            return result
        if len(all_bal) == 0:
            result['auth_msg'] = 'FAIL - 잔고 응답 비어있음 (권한/IP/API 오류 가능)'
            _append_step(result, "2. 인증/잔고 API", "FAIL", result['auth_msg'])
            return result

        krw = _safe_float(all_bal.get('KRW', 0))
        result['auth'] = True
        result['auth_msg'] = 'PASS'
        _currencies = sorted([str(k) for k in all_bal.keys()])
        _append_step(
            result,
            "2. 인증/잔고 API",
            "PASS",
            f"통화 {_currencies[:8]}{'...' if len(_currencies) > 8 else ''} (총 {len(_currencies)}개)"
        )

        result['balance'] = True
        coins = {}
        for k, v in all_bal.items():
            if str(k) == 'KRW':
                continue
            fv = _safe_float(v, 0.0)
            if fv > 0:
                coins[str(k)] = fv

        if coins:
            coin_str = ", ".join(f"{k}={v:.8f}" for k, v in sorted(coins.items()))
            result['balance_msg'] = f"KRW {krw:,.0f}원 / {coin_str}"
            _append_step(result, "3. 잔고 파싱", "PASS", f"KRW={krw:,.0f}원, 코인 {len(coins)}종")
        else:
            result['balance_msg'] = f"KRW {krw:,.0f}원 / 코인 미보유"
            _append_step(result, "3. 잔고 파싱", "PASS", f"KRW={krw:,.0f}원, 코인 미보유")

        # 3. 시세 조회
        portfolio = get_portfolio()
        tickers = list(set(f"{item['market']}-{item['coin'].upper()}" for item in portfolio if item.get("coin")))
        if not tickers:
            result['price_msg'] = 'SKIP - 포트폴리오 비어있음'
            result['signal_msg'] = 'SKIP - 포트폴리오 비어있음'
            result['order_msg'] = 'SKIP - 테스트 대상 없음'
            _append_step(result, "4. 시세 조회", "SKIP", result['price_msg'])
            _append_step(result, "5. 시그널 분석", "SKIP", result['signal_msg'])
            _append_step(result, "6. 가상주문", "SKIP", result['order_msg'])
            return result

        prices_raw = data_cache.get_current_prices_local_first(tickers, ttl_sec=5.0, allow_api_fallback=True)
        prices_raw = prices_raw if isinstance(prices_raw, dict) else {}
        prices = {k: _safe_float(v, 0.0) for k, v in prices_raw.items() if _safe_float(v, 0.0) > 0}

        if prices:
            result['price'] = True
            result['price_msg'] = ", ".join(f"{t}={p:,.0f}" for t, p in prices.items())
            _append_step(result, "4. 시세 조회", "PASS", f"요청 {len(tickers)}개 / 성공 {len(prices)}개")
        else:
            result['price_msg'] = 'FAIL - 시세 조회 실패'
            _append_step(result, "4. 시세 조회", "FAIL", result['price_msg'])
            return result

        # 4. 시그널 분석
        signals = []
        signal_errors = []
        signal_details = []
        for item in portfolio:
            _ticker = f"{item.get('market', 'KRW')}-{str(item.get('coin', '')).upper()}"
            try:
                a = analyze_asset(trader, item)
                if not a:
                    signal_errors.append(f"{_ticker}=ERROR(분석결과 없음)")
                    continue

                ticker = str(a.get('ticker') or _ticker)
                signal_value = str(a.get('signal', '')).upper()
                iv = _normalize_coin_interval(item.get("interval", "day"))
                if signal_value in {"BUY", "SELL", "HOLD"}:
                    signals.append(f"{ticker}={signal_value}")
                    signal_details.append(f"{ticker}:{signal_value}({iv})")
                else:
                    signal_errors.append(f"{ticker}=ERROR({a.get('signal')})")
            except Exception as e:
                signal_errors.append(f"{_ticker}=ERROR({e})")

        if signals and not signal_errors:
            result['signal'] = True
            result['signal_msg'] = ", ".join(signals)
            _append_step(result, "5. 시그널 분석", "PASS", f"{len(signals)}개 성공: {' | '.join(signal_details[:8])}")
        elif signals and signal_errors:
            result['signal'] = False
            result['signal_msg'] = f"FAIL - 일부 시그널 분석 실패 ({'; '.join(signal_errors)}) | 성공: {', '.join(signals)}"
            _append_step(result, "5. 시그널 분석", "FAIL", result['signal_msg'])
        elif signal_errors:
            result['signal_msg'] = f"FAIL - 시그널 분석 실패 ({'; '.join(signal_errors)})"
            _append_step(result, "5. 시그널 분석", "FAIL", result['signal_msg'])
        else:
            result['signal_msg'] = 'SKIP - 포트폴리오 비어있음'
            _append_step(result, "5. 시그널 분석", "SKIP", result['signal_msg'])

        # 5. 가상주문 왕복 테스트 (현재가 50% 아래 지정가 매수 → 조회 → 취소)
        if krw < MIN_ORDER_KRW:
            result['order_msg'] = f"SKIP - 가상주문 최소금액 부족 (KRW {krw:,.0f}원)"
            _append_step(result, "6. 가상주문", "SKIP", result['order_msg'])
            return result

        test_ticker = tickers[0]
        test_price = _safe_float(prices.get(test_ticker), 0.0)
        if test_price <= 0:
            result['order_msg'] = 'SKIP - 테스트 종목 시세 없음'
            _append_step(result, "6. 가상주문", "SKIP", f"{result['order_msg']} (ticker={test_ticker})")
            return result

        # 현재가의 50% 가격 (업비트 호가 단위 정리)
        dummy_price = test_price * 0.5
        if dummy_price >= 2000000:
            dummy_price = int(dummy_price // 1000) * 1000
        elif dummy_price >= 1000000:
            dummy_price = int(dummy_price // 500) * 500
        elif dummy_price >= 500000:
            dummy_price = int(dummy_price // 100) * 100
        elif dummy_price >= 100000:
            dummy_price = int(dummy_price // 50) * 50
        elif dummy_price >= 10000:
            dummy_price = int(dummy_price // 10) * 10
        elif dummy_price >= 1000:
            dummy_price = int(dummy_price // 5) * 5
        elif dummy_price >= 100:
            dummy_price = int(dummy_price // 1) * 1
        elif dummy_price >= 10:
            dummy_price = round(dummy_price, 1)
        elif dummy_price >= 1:
            dummy_price = round(dummy_price, 2)
        else:
            dummy_price = round(dummy_price, 4)

        # 최소 주문금액(5000원) 이상 되는 수량
        min_volume = max(5000 / dummy_price, 0.0001) if dummy_price > 0 else 0.001
        min_volume = round(min_volume, 8)
        _append_step(
            result,
            "6.1 가상주문 입력값",
            "INFO",
            f"ticker={test_ticker}, 현재가={test_price:,.0f}, 테스트가격={dummy_price:,.0f}, 수량={min_volume}"
        )

        order = trader.buy_limit(test_ticker, dummy_price, min_volume)
        if not order:
            result['order_msg'] = "FAIL - 주문 실패: 응답 없음"
            _append_step(result, "6.2 가상주문 접수", "FAIL", result['order_msg'])
            return result
        if 'error' in str(order).lower():
            result['order_msg'] = f"FAIL - 주문 실패: {order}"
            _append_step(result, "6.2 가상주문 접수", "FAIL", result['order_msg'])
            return result

        order_uuid = order.get('uuid', '')
        if not order_uuid:
            result['order_msg'] = f"FAIL - 주문 UUID 없음: {order}"
            _append_step(result, "6.2 가상주문 접수", "FAIL", result['order_msg'])
            return result
        _append_step(result, "6.2 가상주문 접수", "PASS", f"uuid={order_uuid}")
        time.sleep(2)

        # 미체결 조회
        pending = trader.get_orders(test_ticker, state='wait')
        found = False
        if pending:
            found = any(o.get('uuid') == order_uuid for o in pending)
        _append_step(result, "6.3 미체결 조회", "PASS" if found else "FAIL", f"found={found}, pending_count={len(pending) if pending else 0}")

        # 취소
        cancel = trader.cancel_order(order_uuid)
        cancel_ok = bool(cancel) and ('error' not in str(cancel).lower())
        _append_step(result, "6.4 주문 취소", "PASS" if cancel_ok else "FAIL", f"cancel={cancel}")
        time.sleep(1)

        # 취소 확인
        pending_after = trader.get_orders(test_ticker, state='wait')
        still_there = False
        if pending_after:
            still_there = any(o.get('uuid') == order_uuid for o in pending_after)

        if found and cancel_ok and not still_there:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 정상'
            _append_step(result, "6.5 최종 판정", "PASS", result['order_msg'])
        elif found and cancel_ok:
            result['order_test'] = True
            result['order_msg'] = '주문→조회→취소 완료 (잔여 확인 필요)'
            _append_step(result, "6.5 최종 판정", "PASS", f"{result['order_msg']} still_there={still_there}")
        else:
            result['order_msg'] = f"주문={bool(order_uuid)}, 조회={found}, 취소={cancel_ok}"
            _append_step(result, "6.5 최종 판정", "FAIL", result['order_msg'])

    except Exception as e:
        result['order_msg'] = result.get('order_msg') or f'ERROR: {e}'
        _append_step(result, "예외", "FAIL", str(e))
        logger.exception(f"업비트 헬스체크 오류: {e}")

    return result


def _print_health_report(results: dict):
    """헬스체크 종합 리포트 출력 + 텔레그램 전송."""
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)

    lines = []
    lines.append(f"<b>헬스체크 리포트</b> ({now.strftime('%Y-%m-%d %H:%M')} KST)")
    lines.append("")

    pass_count = 0
    total_count = len(results)

    msg_key_map = {
        'auth': 'auth_msg',
        'balance': 'balance_msg',
        'price': 'price_msg',
        'signal': 'signal_msg',
        'order_test': 'order_msg',
    }

    def _status_label(ok: bool, msg: str, allow_skip: bool = True) -> str:
        txt = str(msg or "").strip().upper()
        if ok:
            return "PASS"
        if allow_skip and txt.startswith("SKIP"):
            return "SKIP"
        return "FAIL"

    def _is_ok_or_skip(ok: bool, msg: str) -> bool:
        txt = str(msg or "").strip().upper()
        return bool(ok) or txt.startswith("SKIP")

    for key, r in results.items():
        name = r['name']
        checks = ['auth', 'balance', 'price', 'signal', 'order_test']
        all_pass = all(_is_ok_or_skip(r.get(c, False), r.get(msg_key_map[c], '')) for c in checks)
        if all_pass:
            pass_count += 1

        status_icon = "PASS" if all_pass else "WARN"
        lines.append(f"<b>[{name}]</b> {status_icon}")
        lines.append(f"  인증: {_status_label(r.get('auth', False), r.get('auth_msg', ''), allow_skip=False)} {r.get('auth_msg', '')}")
        lines.append(f"  잔고: {_status_label(r.get('balance', False), r.get('balance_msg', ''), allow_skip=False)} {r.get('balance_msg', '')}")
        lines.append(f"  시세: {_status_label(r.get('price', False), r.get('price_msg', ''), allow_skip=False)} {r.get('price_msg', '')}")
        lines.append(f"  시그널: {_status_label(r.get('signal', False), r.get('signal_msg', ''))} {r.get('signal_msg', '')}")
        lines.append(f"  가상주문: {_status_label(r.get('order_test', False), r.get('order_msg', ''))} {r.get('order_msg', '')}")
        step_lines = r.get('steps') if isinstance(r, dict) else None
        if step_lines:
            lines.append("  상세 단계:")
            max_steps = 12
            for s in step_lines[:max_steps]:
                lines.append(f"    {s}")
            if len(step_lines) > max_steps:
                lines.append(f"    ... 생략 {len(step_lines) - max_steps}개")
        lines.append("")

    summary_label = "시스템 정상" if pass_count == total_count else "시스템 점검 필요"
    lines.append(f"<b>종합: {pass_count}/{total_count} {summary_label}</b>")

    report = "\n".join(lines)

    # 콘솔 로그 출력
    logger.info("")
    logger.info("=" * 60)
    for line in lines:
        # HTML 태그 제거하여 로그 출력
        clean = line.replace("<b>", "").replace("</b>", "")
        logger.info(f"  {clean}")
    logger.info("=" * 60)

    # 텔레그램 전송
    _send_telegram(report)


def run_health_check():
    """매일 장마감 전 헬스체크 실행 (국내 시스템 + 업비트)."""
    load_dotenv()
    logger.info("=== 일일 헬스체크 시작 ===")

    results = {}

    # 업비트는 24시간 → 항상 점검
    if os.getenv("UPBIT_ACCESS_KEY"):
        results['upbit'] = _check_upbit()
        logger.info("--- 업비트 점검 완료 ---")

    # 국내 시스템은 장 시간 내만 점검
    if _is_kr_market_hours():
        results['kiwoom_gold'] = _check_kiwoom_gold()
        logger.info("--- 키움 금현물 점검 완료 ---")

        results['kis_isa'] = _check_kis_isa()
        logger.info("--- KIS ISA 점검 완료 ---")

        results['kis_pension'] = _check_kis_pension()
        logger.info("--- KIS 연금저축 점검 완료 ---")
    else:
        logger.info("장 시간 외 → 국내 시스템(키움/KIS) 점검 생략")

    if results:
        _print_health_report(results)
    else:
        logger.info("점검 대상 없음.")

    logger.info("=== 일일 헬스체크 완료 ===")


def run_upbit_health_check():
    """업비트 전용 헬스체크 (4시간마다 실행)."""
    load_dotenv()
    logger.info("=== 업비트 헬스체크 시작 ===")

    results = {}
    results['upbit'] = _check_upbit()
    logger.info("--- 업비트 점검 완료 ---")

    _print_health_report(results)
    logger.info("=== 업비트 헬스체크 완료 ===")


def _format_holdings_brief(holdings, max_items=3):
    if not holdings:
        return "??"
    parts = []
    for h in holdings[:max_items]:
        name = h.get("name") or h.get("code") or "UNKNOWN"
        qty = h.get("qty", 0)
        parts.append(f"{name} {qty}")
    if len(holdings) > max_items:
        parts.append(f"... +{len(holdings) - max_items}?")
    return ", ".join(parts)


def run_daily_status_report():
    """
    매일 아침 전체계좌 자산 잔고/보유현황 텔레그램으로 전송.
    실제 주문 하지않고 시그널만 확인하여 보고함.
    """
    load_dotenv()
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    lines = [f"<b>일일 자산 현황</b> ({now.strftime('%Y-%m-%d %H:%M')} KST)", ""]

    # Upbit
    if os.getenv("UPBIT_ACCESS_KEY") and os.getenv("UPBIT_SECRET_KEY"):
        try:
            trader = UpbitTrader(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
            all_bal = trader.get_all_balances()
            if all_bal is None:
                raise RuntimeError("잔고 API 실패 (응답=None)")
            if not isinstance(all_bal, dict):
                raise RuntimeError(f"잔고 API 응답 타입 오류 ({type(all_bal).__name__})")
            if len(all_bal) == 0:
                raise RuntimeError("잔고 API 응답 비어있음")
            krw = float(all_bal.get("KRW", 0.0))
            coins = {k: float(v) for k, v in all_bal.items() if k != "KRW" and float(v) > 0}
            est_coin_value = 0.0
            for sym, qty in coins.items():
                p = _get_upbit_price_local_first(f"KRW-{sym}")
                if p:
                    est_coin_value += qty * float(p)
            total = krw + est_coin_value
            coin_text = ", ".join([f"{k} {v:.6f}" for k, v in list(coins.items())[:5]]) if coins else "없음"
            lines.append("<b>[업비트]</b>")
            lines.append(f"현금: {krw:,.0f} KRW")
            lines.append(f"코인평가(추정): {est_coin_value:,.0f} KRW")
            lines.append(f"총자산(추정): {total:,.0f} KRW")
            lines.append(f"보유코인: {coin_text}")
            lines.append("")
        except Exception as e:
            lines.append("<b>[업비트]</b>")
            lines.append(f"조회 실패: {e}")
            lines.append("")

    # Kiwoom Gold
    if os.getenv("Kiwoom_App_Key") and os.getenv("Kiwoom_Secret_Key"):
        try:
            trader = KiwoomGoldTrader(is_mock=False)
            if trader.auth():
                bal = trader.get_balance() or {}
                cash = float(bal.get("cash_krw", 0.0))
                qty = float(bal.get("gold_qty", 0.0))
                eval_amt = float(bal.get("gold_eval", 0.0))
                if eval_amt <= 0:
                    p = _get_gold_price_local_first(trader, GOLD_CODE_1KG) or 0
                    eval_amt = qty * float(p) if p else 0.0
                total = cash + eval_amt
                lines.append("<b>[키움 금현물]</b>")
                lines.append(f"예수금: {cash:,.0f} KRW")
                lines.append(f"금보유: {qty:.4f} g")
                lines.append(f"금평가: {eval_amt:,.0f} KRW")
                lines.append(f"총자산: {total:,.0f} KRW")
                lines.append("")
            else:
                lines.append("<b>[키움 금현물]</b>")
                lines.append("인증 실패")
                lines.append("")
        except Exception as e:
            lines.append("<b>[키움 금현물]</b>")
            lines.append(f"조회 실패: {e}")
            lines.append("")

    # KIS ISA
    isa_key = _get_env_any("KIS_ISA_APP_KEY", "KIS_APP_KEY")
    isa_secret = _get_env_any("KIS_ISA_APP_SECRET", "KIS_APP_SECRET")
    isa_account_raw = _get_env_any("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    isa_prdt_raw = _get_env_any("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    isa_account, isa_prdt = _normalize_kis_account_fields(isa_account_raw, isa_prdt_raw)
    if isa_key and isa_secret and isa_account:
        try:
            trader = KISTrader(is_mock=False)
            trader.app_key = isa_key
            trader.app_secret = isa_secret
            trader.account_no = isa_account
            trader.acnt_prdt_cd = isa_prdt
            if trader.auth():
                bal = trader.get_balance() or {}
                cash = float(bal.get("cash", 0.0))
                holdings = bal.get("holdings", []) or []
                total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
                lines.append("<b>[KIS ISA]</b>")
                lines.append(f"예수금: {cash:,.0f} KRW")
                lines.append(f"총평가: {total_eval:,.0f} KRW")
                lines.append(f"보유: {_format_holdings_brief(holdings)}")
                lines.append("")
            else:
                lines.append("<b>[KIS ISA]</b>")
                lines.append("인증 실패")
                lines.append("")
        except Exception as e:
            lines.append("<b>[KIS ISA]</b>")
            lines.append(f"조회 실패: {e}")
            lines.append("")

    # KIS Pension
    pension_acct_raw = _get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    pension_prdt_raw = _get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    pension_acct, pension_prdt = _normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)
    if pension_acct:
        try:
            trader = KISTrader(is_mock=False)
            pension_key = _get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
            pension_secret = _get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
            if pension_key:
                trader.app_key = pension_key
            if pension_secret:
                trader.app_secret = pension_secret
            if pension_acct:
                trader.account_no = pension_acct
                trader.acnt_prdt_cd = pension_prdt

            if trader.auth():
                bal = trader.get_balance() or {}
                cash = float(bal.get("cash", 0.0))
                holdings = bal.get("holdings", []) or []
                total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
                lines.append("<b>[KIS 연금저축]</b>")
                lines.append(f"예수금: {cash:,.0f} KRW")
                lines.append(f"총평가: {total_eval:,.0f} KRW")
                lines.append(f"보유: {_format_holdings_brief(holdings)}")

                # LAA 전략 시그널 체크 (주문 없이 분석만)
                try:
                    kr_etf_map = {
                        "SPY": _get_env_any("KR_ETF_LAA_SPY", "KR_ETF_SPY", default="360750"),
                        "IWD": _get_env_any("KR_ETF_LAA_IWD", "KR_ETF_SPY", default="360750"),
                        "GLD": _get_env_any("KR_ETF_LAA_GLD", "KR_ETF_GOLD", default="132030"),
                        "IEF": _get_env_any("KR_ETF_LAA_IEF", "KR_ETF_AGG", default="453540"),
                        "QQQ": _get_env_any("KR_ETF_LAA_QQQ", default="133690"),
                        "SHY": _get_env_any("KR_ETF_LAA_SHY", default="114470"),
                    }
                    tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
                    price_data = {}
                    price_ok = True
                    for t in tickers:
                        code = str(kr_etf_map.get(t, "")).strip()
                        df = _get_kis_daily_local_first(trader, code, count=320) if code else None
                        if df is None or len(df) == 0:
                            price_ok = False
                            break
                        price_data[t] = df
                        time.sleep(0.15)

                    if price_ok:
                        strategy = LAAStrategy(settings={"kr_etf_map": kr_etf_map})
                        signal = strategy.analyze(price_data)
                        if signal:
                            risk_label = "공격(Risk-On)" if signal.get("risk_on") else "방어(Risk-Off)"
                            risk_asset = signal.get("selected_risk_asset", "?")
                            risk_kr = signal.get("selected_risk_kr_code", "?")
                            lines.append("")
                            lines.append(f"<b>LAA 시그널:</b> {risk_label}")
                            lines.append(f"리스크 자산: {risk_asset} → {risk_kr}")

                            # 목표 vs 현재 비교 → 예상 매매 내역
                            target_weights_kr = signal.get("target_weights_kr", {})
                            total_eval_safe = max(total_eval, 1.0)
                            tracked = set(str(c) for c in target_weights_kr.keys())
                            cur_vals = {c: 0.0 for c in tracked}
                            cur_qtys = {c: 0 for c in tracked}
                            for h in holdings:
                                code = str(h.get("code", ""))
                                if code in tracked:
                                    cur_vals[code] += float(h.get("eval_amt", 0.0))
                                    cur_qtys[code] += int(float(h.get("qty", 0)))

                            max_gap = 0.0
                            planned_orders = []
                            for code, tw in target_weights_kr.items():
                                code = str(code)
                                cur_v = cur_vals.get(code, 0.0)
                                cur_w = cur_v / total_eval_safe
                                gap = abs(float(tw) - cur_w)
                                max_gap = max(max_gap, gap)
                                tgt_v = total_eval_safe * float(tw)
                                price = _get_kis_price_local_first(trader, code) or 0.0
                                if price <= 0:
                                    continue
                                tgt_qty = int(tgt_v / price)
                                cur_q = cur_qtys.get(code, 0)
                                delta = tgt_qty - cur_q
                                if delta > 0:
                                    planned_orders.append(f"매수 {code} {delta}주")
                                elif delta < 0:
                                    planned_orders.append(f"매도 {code} {abs(delta)}주")

                            action = "HOLD" if max_gap <= 0.03 else "REBALANCE"
                            lines.append(f"판정: <b>{action}</b> (최대괴리 {max_gap*100:.1f}%p)")
                            if planned_orders:
                                lines.append("예상 매매:")
                                for po in planned_orders:
                                    lines.append(f"  - {po}")
                            else:
                                lines.append("예상 매매: 없음")
                        else:
                            lines.append("LAA 분석 실패")
                    else:
                        lines.append("국내 ETF 조회 실패 (시그널 체크 생략)")
                except Exception as sig_e:
                    lines.append(f"시그널 체크 오류: {sig_e}")

                lines.append("")
            else:
                lines.append("<b>[KIS 연금저축]</b>")
                lines.append("인증 실패")
                lines.append("")
        except Exception as e:
            lines.append("<b>[KIS 연금저축]</b>")
            lines.append(f"조회 실패: {e}")
            lines.append("")

    if len(lines) <= 2:
        lines.append("조회 가능한 계좌가 없습니다. API 키/계좌 설정을 확인하세요.")

    report = "\n".join(lines)
    logger.info(report.replace("<b>", "").replace("</b>", ""))
    _send_telegram(report)


def run_telegram_test_ping():
    """
    텔레그램 알림 경로 점검용: 매 정각 실행 가능한 경량 핑 메시지 전송.
    주문/시세 조회 없이 단순 알림만 보낸다.
    """
    load_dotenv()
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    msg = (
        f"<b>텔레그램 알림 테스트</b>\n"
        f"시각: {now.strftime('%Y-%m-%d %H:%M:%S')} KST\n"
        f"모드: telegram_test_ping\n"
        f"상태: 정상"
    )
    logger.info("텔레그램 테스트 핑 전송")
    _send_telegram(msg)


# ----------------------------------------------------------------------
# Override: KIS Pension uses LAA strategy
# ----------------------------------------------------------------------
def run_kis_pension_trade():
    """
    한국투자증권 연금저축 계좌 - LAA 전략 자동매매.

    - 코어 75%: IWD/GLD/IEF
    - 리스크 25%: SPY 200일선 위 QQQ, 아니면 SHY
    - 월간 리밸런싱
    """
    load_dotenv()
    logger.info("=== KIS 연금저축 LAA 자동매매 시작 ===")

    if not _is_kr_order_window():
        logger.info("국내 주문 가능 시간이 아닙니다(09:00~15:20 KST). 매매를 생략합니다.")
        return

    pension_key = _get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
    pension_secret = _get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
    pension_acct_raw = _get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    pension_prdt_raw = _get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    pension_acct, pension_prdt = _normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)

    trader = KISTrader(is_mock=False)
    if pension_key:
        trader.app_key = pension_key
    if pension_secret:
        trader.app_secret = pension_secret
    if pension_acct:
        trader.account_no = pension_acct
        trader.acnt_prdt_cd = pension_prdt

    if not trader.auth():
        logger.error("KIS 인증 실패. 종료.")
        return

    kr_etf_map = {
        "SPY": _get_env_any("KR_ETF_LAA_SPY", "KR_ETF_SPY", default="360750"),
        "IWD": _get_env_any("KR_ETF_LAA_IWD", "KR_ETF_SPY", default="360750"),
        "GLD": _get_env_any("KR_ETF_LAA_GLD", "KR_ETF_GOLD", default="132030"),
        "IEF": _get_env_any("KR_ETF_LAA_IEF", "KR_ETF_AGG", default="453540"),
        "QQQ": _get_env_any("KR_ETF_LAA_QQQ", default="133690"),
        "SHY": _get_env_any("KR_ETF_LAA_SHY", default="114470"),
    }

    tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
    price_data = {}
    for ticker in tickers:
        code = str(kr_etf_map.get(ticker, "")).strip()
        df = _get_kis_daily_local_first(trader, code, count=420) if code else None
        if df is None or len(df) == 0:
            logger.error(f"{ticker}({code}) 국내 데이터 조회 실패. 종료.")
            return
        price_data[ticker] = df
        logger.info(f"{ticker}({code}): {len(df)}건 로드")
        time.sleep(0.2)

    strategy = LAAStrategy(settings={"kr_etf_map": kr_etf_map})
    signal = strategy.analyze(price_data)
    if not signal:
        logger.error("LAA 시그널 분석 실패. 종료.")
        return

    bal = trader.get_balance()
    if not bal:
        logger.error("잔고 조회 실패. 종료.")
        return

    cash = float(bal.get("cash", 0.0))
    holdings = bal.get("holdings", []) or []
    total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
    total_eval = max(total_eval, 1.0)

    target_weights_kr = signal.get("target_weights_kr", {})
    tracked_codes = set(str(c) for c in target_weights_kr.keys())
    current_vals = {c: 0.0 for c in tracked_codes}
    current_qty = {c: 0 for c in tracked_codes}
    for h in holdings:
        code = str(h.get("code", ""))
        if code in tracked_codes:
            current_vals[code] += float(h.get("eval_amt", 0.0))
            current_qty[code] += int(float(h.get("qty", 0)))

    target_vals = {c: float(total_eval) * float(w) for c, w in target_weights_kr.items()}
    min_order = 10_000
    tolerance = 0.01  # 1%
    actions = []

    # 1) 매도부터 실행
    for code in tracked_codes:
        cur_v = float(current_vals.get(code, 0.0))
        tgt_v = float(target_vals.get(code, 0.0))
        if cur_v <= tgt_v * (1.0 + tolerance):
            continue
        excess = cur_v - tgt_v
        if excess < min_order:
            continue
        price = _get_kis_price_local_first(trader, code) or 0.0
        if price <= 0:
            continue
        qty = int(excess / price)
        qty = min(qty, int(current_qty.get(code, 0)))
        if qty <= 0:
            continue
        result = trader.smart_sell_qty_closing(code, qty)
        actions.append(f"SELL {code} {qty}주 ({'OK' if result else 'FAIL'})")
        logger.info(actions[-1])
        time.sleep(0.8)

    # 2) 매수 전 잔고 재조회
    bal2 = trader.get_balance() or {}
    cash = float(bal2.get("cash", cash))
    holdings2 = bal2.get("holdings", []) or holdings
    current_vals = {c: 0.0 for c in tracked_codes}
    for h in holdings2:
        code = str(h.get("code", ""))
        if code in tracked_codes:
            current_vals[code] += float(h.get("eval_amt", 0.0))

    # 3) 매수 실행
    buy_candidates = []
    for code in tracked_codes:
        deficit = float(target_vals.get(code, 0.0)) - float(current_vals.get(code, 0.0))
        if deficit > 0:
            buy_candidates.append((code, deficit))
    buy_candidates.sort(key=lambda x: x[1], reverse=True)

    for code, deficit in buy_candidates:
        if cash < min_order:
            break
        buy_amount = min(deficit, cash * 0.995)
        if buy_amount < min_order:
            continue
        result = trader.smart_buy_krw_closing(code, buy_amount)
        actions.append(f"BUY {code} {buy_amount:,.0f}원 ({'OK' if result else 'FAIL'})")
        logger.info(actions[-1])
        if result:
            cash -= buy_amount
        time.sleep(0.8)

    if not actions:
        actions.append("HOLD (리밸런싱 불필요)")

    logger.info("=== KIS 연금저축 LAA 자동매매 완료 ===")
    tg = [f"<b>KIS 연금저축 LAA</b>"]
    tg.append(f"리스크 상태: {'공격' if signal.get('risk_on') else '방어'}")
    tg.append(f"리스크 자산: {signal.get('selected_risk_asset')} -> {signal.get('selected_risk_kr_code')}")
    tg.append(signal.get("reason", ""))
    tg.append("실행:")
    tg.extend([f"- {a}" for a in actions[:8]])
    _send_telegram("\n".join(tg))


def _check_kis_pension() -> dict:
    """KIS 연금저축 LAA 시스템 헬스체크."""
    result = {
        "name": "KIS 연금저축 LAA",
        "auth": False,
        "auth_msg": "",
        "balance": False,
        "balance_msg": "",
        "price": False,
        "price_msg": "",
        "signal": False,
        "signal_msg": "",
        "order_test": False,
        "order_msg": "",
    }
    try:
        trader = KISTrader(is_mock=False)
        pension_key = _get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
        pension_secret = _get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
        pension_acct_raw = _get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
        pension_prdt_raw = _get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
        pension_acct, pension_prdt = _normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)
        if pension_key:
            trader.app_key = pension_key
        if pension_secret:
            trader.app_secret = pension_secret
        if pension_acct:
            trader.account_no = pension_acct
            trader.acnt_prdt_cd = pension_prdt

        if not trader.auth():
            result["auth_msg"] = "FAIL - 인증 실패"
            return result
        result["auth"] = True
        result["auth_msg"] = "PASS"

        bal = trader.get_balance()
        if bal is None:
            result["balance_msg"] = "FAIL - 잔고 조회 실패"
            return result
        result["balance"] = True
        cash = float(bal.get("cash", 0.0))
        holdings = bal.get("holdings", []) or []
        total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
        result["balance_msg"] = f"예수금 {cash:,.0f} / 총평가 {total_eval:,.0f}"

        tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
        kr_etf_map = {
            "SPY": _get_env_any("KR_ETF_LAA_SPY", "KR_ETF_SPY", default="360750"),
            "IWD": _get_env_any("KR_ETF_LAA_IWD", "KR_ETF_SPY", default="360750"),
            "GLD": _get_env_any("KR_ETF_LAA_GLD", "KR_ETF_GOLD", default="132030"),
            "IEF": _get_env_any("KR_ETF_LAA_IEF", "KR_ETF_AGG", default="453540"),
            "QQQ": _get_env_any("KR_ETF_LAA_QQQ", default="133690"),
            "SHY": _get_env_any("KR_ETF_LAA_SHY", default="114470"),
        }
        price_data = {}
        for t in tickers:
            code = str(kr_etf_map.get(t, "")).strip()
            if not code:
                result["price_msg"] = f"FAIL - {t} 국내 ETF 코드 미설정"
                return result
            df = _get_kis_daily_local_first(trader, code, count=320)
            if df is None or len(df) == 0:
                result["price_msg"] = f"FAIL - {t}({code}) 국내 데이터 조회 실패"
                return result
            price_data[t] = df
            time.sleep(0.1)
        result["price"] = True
        result["price_msg"] = "PASS - 국내 ETF SPY/IWD/GLD/IEF/QQQ/SHY"

        strategy = LAAStrategy(settings={"kr_etf_map": kr_etf_map})
        sig = strategy.analyze(price_data)
        if not sig:
            result["signal_msg"] = "FAIL - LAA 분석 실패"
            return result
        result["signal"] = True
        result["signal_msg"] = (
            f"{'공격' if sig.get('risk_on') else '방어'} | "
            f"{sig.get('selected_risk_asset')}->{sig.get('selected_risk_kr_code')}"
        )

        # 주문 테스트: 목표 종목 중 1개로 1주 지정가 주문 후 즉시 취소
        if not _is_kr_order_window():
            result["order_msg"] = "SKIP - 주문 가능 시간이 아닙니다 (09:00~15:20 KST)"
            return result

        test_code = next(iter(sig.get("target_weights_kr", {}).keys()), "")
        if not test_code:
            result["order_msg"] = "FAIL - 테스트 종목 없음"
            return result

        limit_price = trader._get_limit_price(test_code, "SELL")
        if limit_price <= 0:
            result["order_msg"] = "FAIL - 가격호가 계산 실패"
            return result

        order_result = trader.send_order("BUY", test_code, qty=1, price=limit_price, ord_dvsn="00")
        if not order_result or not order_result.get("success"):
            result["order_msg"] = f"FAIL - 주문 실패: {order_result}"
            return result

        ord_no = order_result.get("ord_no", "")
        time.sleep(1.5)
        pending = trader.get_pending_orders(test_code) or []
        found = any(p.get("ord_no") == ord_no for p in pending) if pending else bool(ord_no)
        cancel_result = trader.cancel_order(ord_no, test_code)
        cancel_ok = bool(cancel_result and (cancel_result.get('success') if isinstance(cancel_result, dict) else cancel_result))
        time.sleep(0.8)
        pending_after = trader.get_pending_orders(test_code) or []
        still_there = any(p.get("ord_no") == ord_no for p in pending_after) if pending_after else False

        if found and cancel_ok and not still_there:
            result["order_test"] = True
            result["order_msg"] = "PASS - 주문/취소 정상"
        elif found and cancel_ok:
            result["order_test"] = True
            result["order_msg"] = "PASS - 취소 전파 지연 가능"
        else:
            result["order_msg"] = f"FAIL - 주문={bool(ord_no)}, 조회={found}, 취소={cancel_ok}"

    except Exception as e:
        result["order_msg"] = result.get("order_msg") or f"ERROR: {e}"
        logger.error(f"KIS 연금저축 LAA 헬스체크 오류: {e}")
    return result


if __name__ == "__main__":
    load_dotenv()
    mode = os.getenv("TRADING_MODE", "upbit").lower()
    if mode == "kiwoom_gold":
        run_kiwoom_gold_trade()
    elif mode == "kis_isa":
        run_kis_isa_trade()
    elif mode == "kis_pension":
        run_kis_pension_trade()
    elif mode == "health_check":
        run_health_check()
    elif mode == "health_check_upbit":
        run_upbit_health_check()
    elif mode == "daily_status":
        run_daily_status_report()
    elif mode == "telegram_test_ping":
        run_telegram_test_ping()
    else:
        run_auto_trade()
