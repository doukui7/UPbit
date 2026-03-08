"""업비트 코인 자동매매 엔진."""
import os
import json
import time
import logging
import pandas as pd
from datetime import datetime
from collections import OrderedDict

from .constants import (
    KST, PROJECT_ROOT, MIN_ORDER_KRW,
    UPBIT_TEST_MIN_KRW, UPBIT_TEST_BUY_MULTIPLIER, UPBIT_TEST_SELL_MULTIPLIER,
)
from .utils import (
    safe_float, fmt_krw_price, calc_gap_pct, fmt_gap_pct,
    valid_price, abs_gap, round_upbit_price, ceil_volume_8,
    extract_order_error_text, load_user_config, normalize_coin_interval,
    is_coin_interval_due, wait_for_candle_boundary, fmt_interval_label,
    cond_text,
)
from .state import (
    append_trade_log, load_signal_state, save_signal_state,
    get_prev_state, build_signal_entry,
    load_signal_test_orders, save_signal_test_orders,
    cancel_old_signal_test_orders, save_balance_cache,
)
from .notifier import send_telegram
from .data import get_upbit_ohlcv_local_first, get_upbit_price_local_first

from src.strategy.sma import SMAStrategy
from src.strategy.donchian import DonchianStrategy
from src.trading.upbit_trader import UpbitTrader

logger = logging.getLogger(__name__)


# ── 시그널 키 / 전환 감지 ────────────────────────────────

def make_signal_key(item: dict) -> str:
    """포트폴리오 아이템의 고유 시그널 키 생성."""
    ticker = f"{item.get('market', 'KRW')}-{str(item.get('coin', '')).upper()}"
    strategy = item.get("strategy", "SMA")
    param = item.get("parameter", 20)
    interval = normalize_coin_interval(item.get("interval", "day"))
    return f"{ticker}_{strategy}_{param}_{interval}"


def determine_signal(position_state: str, prev_state: str | None) -> str:
    """포지션 상태 전환 감지 → 실행 시그널 결정."""
    if position_state == 'HOLD':
        return 'HOLD'
    if prev_state is None:
        return position_state
    if position_state == prev_state:
        return 'HOLD'
    return position_state


def is_filled_exec_result(result: dict, side: str) -> bool:
    """체결 결과(dict)가 실제 체결을 포함하는지 판별."""
    if not isinstance(result, dict):
        return False
    side_l = str(side or "").strip().lower()
    typ = str(result.get("type", "")).strip().lower()
    if side_l == "buy" and "buy" not in typ:
        return False
    if side_l == "sell" and "sell" not in typ:
        return False
    filled_vol = safe_float(result.get("filled_volume", 0.0), default=0.0)
    filled_krw = safe_float(result.get("total_krw", 0.0), default=0.0)
    return (filled_vol > 0) or (filled_krw > 0)


def has_strategy_filled_exec(exec_results: dict, ticker: str, strategy_label: str, side: str) -> bool:
    """특정 전략/방향의 체결 성공 여부."""
    for ex in exec_results.get(ticker, []) or []:
        if strategy_label and str(ex.get("_strategy_label", "")) != str(strategy_label):
            continue
        if is_filled_exec_result(ex, side=side):
            return True
    return False


# ── 테스트 주문 ──────────────────────────────────────────

def select_upbit_test_ticker(portfolio: list[dict], analyses: list[dict] | None = None) -> str:
    if analyses:
        for a in analyses:
            t = str(a.get("ticker", "")).strip().upper()
            if t.startswith("KRW-"):
                return t
    for item in portfolio or []:
        market = str(item.get("market", "KRW")).strip().upper() or "KRW"
        coin = str(item.get("coin", "")).strip().upper()
        if market == "KRW" and coin:
            return f"{market}-{coin}"
    return ""


def submit_and_cancel_upbit_test_order(
    trader: UpbitTrader, *, ticker: str, side: str, price: float, volume: float,
) -> tuple[bool, str]:
    side_u = str(side).upper()
    if side_u == "BUY":
        order = trader.buy_limit(ticker, price, volume)
    else:
        order = trader.sell_limit(ticker, price, volume)

    err = extract_order_error_text(order)
    if err:
        return False, f"{side_u} FAIL - 주문실패: {err}"
    if not isinstance(order, dict):
        return False, f"{side_u} FAIL - 주문응답 형식오류({type(order).__name__})"

    order_uuid = str(order.get("uuid", "")).strip()
    if not order_uuid:
        return False, f"{side_u} FAIL - 주문 UUID 없음"

    time.sleep(1.0)
    cancel = trader.cancel_order(order_uuid)
    cancel_err = extract_order_error_text(cancel)
    cancel_ok = not bool(cancel_err)
    time.sleep(0.8)

    still_wait = False
    try:
        pending_after = trader.get_orders(ticker, state="wait") or []
        still_wait = any(
            isinstance(o, dict) and str(o.get("uuid", "")).strip() == order_uuid
            for o in pending_after
        )
    except Exception:
        still_wait = False

    if cancel_ok and not still_wait:
        return True, f"{side_u} PASS - 주문/취소 완료 (가격 {price:,.0f}, 수량 {volume:.8f})"
    if cancel_ok and still_wait:
        return False, f"{side_u} FAIL - 취소 후 대기주문 잔존(uuid={order_uuid})"
    return False, f"{side_u} FAIL - 취소실패(uuid={order_uuid}, err={cancel_err or 'unknown'})"


def run_upbit_cycle_test_orders(
    trader: UpbitTrader, *, portfolio: list[dict], analyses: list[dict] | None = None,
) -> list[str]:
    notes: list[str] = []
    ticker = select_upbit_test_ticker(portfolio, analyses)
    if not ticker:
        return ["테스트주문 SKIP - 대상 코인 없음"]

    current_price = safe_float(trader.get_current_price(ticker), default=0.0)
    if current_price <= 0:
        return [f"테스트주문 SKIP - 현재가 조회 실패 ({ticker})"]

    buy_price = round_upbit_price(current_price * UPBIT_TEST_BUY_MULTIPLIER, mode="floor")
    if buy_price <= 0:
        notes.append(f"테스트주문 BUY SKIP - 테스트가격 계산 실패 ({ticker})")
    else:
        krw = safe_float(trader.get_balance("KRW"), default=0.0)
        if krw < UPBIT_TEST_MIN_KRW:
            notes.append(f"테스트주문 BUY SKIP - KRW 부족 ({krw:,.0f} < {UPBIT_TEST_MIN_KRW:,.0f})")
        else:
            buy_volume = ceil_volume_8(UPBIT_TEST_MIN_KRW / buy_price)
            _, msg = submit_and_cancel_upbit_test_order(
                trader, ticker=ticker, side="BUY", price=buy_price, volume=buy_volume,
            )
            notes.append(f"[{ticker}] 테스트주문 {msg}")

    sell_price = round_upbit_price(current_price * UPBIT_TEST_SELL_MULTIPLIER, mode="ceil")
    if sell_price <= 0:
        notes.append(f"테스트주문 SELL SKIP - 테스트가격 계산 실패 ({ticker})")
    else:
        coin_sym = ticker.split("-")[-1]
        coin_balance = safe_float(trader.get_balance(coin_sym), default=0.0)
        min_sell_volume = ceil_volume_8(UPBIT_TEST_MIN_KRW / sell_price)
        if coin_balance < min_sell_volume:
            notes.append(
                f"테스트주문 SELL SKIP - 보유수량 부족 "
                f"({coin_sym} {coin_balance:.8f} < 필요 {min_sell_volume:.8f})"
            )
        else:
            _, msg = submit_and_cancel_upbit_test_order(
                trader, ticker=ticker, side="SELL", price=sell_price, volume=min_sell_volume,
            )
            notes.append(f"[{ticker}] 테스트주문 {msg}")

    return notes


# ── 조건 정보 빌더 ───────────────────────────────────────

def pick_focus_target(
    *, buy_label, buy_level, buy_gap, buy_cond,
    sell_label, sell_level, sell_gap, sell_cond,
) -> dict:
    """매수/매도 타점 중 현재가와 더 가까운 1개만 선택."""
    cands = []
    if valid_price(buy_level):
        cands.append({"side": "BUY", "label": buy_label, "level": buy_level, "gap_pct": buy_gap, "cond": buy_cond})
    if valid_price(sell_level):
        cands.append({"side": "SELL", "label": sell_label, "level": sell_level, "gap_pct": sell_gap, "cond": sell_cond})
    if not cands:
        return {"side": "", "label": "타점", "level": 0.0, "gap_pct": None, "cond": None}
    cands.sort(key=lambda x: (abs_gap(x.get("gap_pct")), 0 if x.get("cond") is True else 1, 0 if x.get("side") == "BUY" else 1))
    return cands[0]


def build_upbit_condition_info(strategy_name, param, sell_param, interval, last_candle, current_price) -> dict:
    close_price = safe_float(last_candle.get("close"), default=0.0)
    interval_label = fmt_interval_label(interval)

    if strategy_name == "Donchian":
        buy_key = f"Donchian_Upper_{param}"
        sell_key = f"Donchian_Lower_{sell_param}"
        buy_level = safe_float(last_candle.get(buy_key), default=0.0)
        sell_level = safe_float(last_candle.get(sell_key), default=0.0)
        buy_cond = (current_price > buy_level) if buy_level > 0 else None
        sell_cond = (current_price < sell_level) if sell_level > 0 else None
        buy_gap = calc_gap_pct(current_price, buy_level)
        sell_gap = calc_gap_pct(current_price, sell_level)
        buy_label = f"매수타점(상단 {param})"
        sell_label = f"매도타점(하단 {sell_param})"
        label = f"Donchian({param}/{sell_param}, {interval_label})"
    else:
        sma_key = f"SMA_{param}"
        sma_level = safe_float(last_candle.get(sma_key), default=0.0)
        buy_level = sma_level
        sell_level = sma_level
        buy_cond = (current_price > sma_level) if sma_level > 0 else None
        sell_cond = (current_price < sma_level) if sma_level > 0 else None
        buy_gap = calc_gap_pct(current_price, sma_level)
        sell_gap = buy_gap
        buy_label = f"매수타점(SMA_{param})"
        sell_label = f"매도타점(SMA_{param})"
        label = f"SMA({param}, {interval_label})"

    focus = pick_focus_target(
        buy_label=buy_label, buy_level=buy_level, buy_gap=buy_gap, buy_cond=buy_cond,
        sell_label=sell_label, sell_level=sell_level, sell_gap=sell_gap, sell_cond=sell_cond,
    )
    focus_line = (
        f"{focus.get('label', '타점')} {fmt_krw_price(focus.get('level'))} | "
        f"이격 {fmt_gap_pct(focus.get('gap_pct'))} | {cond_text(focus.get('cond'))}"
    )
    lines = [f"현재가 {fmt_krw_price(current_price)} / 판단종가 {fmt_krw_price(close_price)}", focus_line]
    summary = f"현재가 {fmt_krw_price(current_price)} | {focus_line}"
    return {
        "strategy_label": label, "close_price": close_price,
        "buy_level": buy_level, "sell_level": sell_level,
        "buy_gap_pct": buy_gap, "sell_gap_pct": sell_gap,
        "buy_cond": buy_cond, "sell_cond": sell_cond,
        "focus_side": focus.get("side"), "focus_label": focus.get("label"),
        "focus_level": focus.get("level"), "focus_gap_pct": focus.get("gap_pct"),
        "focus_cond": focus.get("cond"),
        "condition_lines": lines, "condition_summary": summary,
    }


def describe_upbit_signal_reason(analysis: dict) -> str:
    signal = str(analysis.get("signal", "")).upper()
    pos_state = str(analysis.get("position_state", "")).upper()
    prev_state = str(analysis.get("prev_state", "")).upper()
    if signal == "SKIP":
        iv = str(analysis.get("interval", ""))
        iv_label = "1D→09시만" if iv == "day" else f"{iv}→해당 주기만"
        if pos_state in {"BUY", "SELL"}:
            return f"주기 미도래 ({iv_label}, 현재 판단={pos_state})"
        return f"주기 미도래 ({iv_label})"
    if signal in {"BUY", "SELL"}:
        return f"주문조건 충족 ({signal} 전환)"
    if pos_state == "HOLD":
        return "조건미충족 (중립구간)"
    if prev_state and pos_state == prev_state and pos_state in {"BUY", "SELL"}:
        return f"전환조건 미충족 (기존 {pos_state} 상태 유지)"
    return "조건미충족"


def compact_upbit_condition_lines(analysis: dict) -> list[str]:
    """알림 표시는 항상 '현재가 + 가까운 타점 1개'만 유지."""
    raw_lines = [str(x) for x in (analysis.get("condition_lines") or []) if str(x).strip()]
    current_line = next((ln for ln in raw_lines if ln.startswith("현재가 ")), "")
    if not current_line:
        current_line = (
            f"현재가 {fmt_krw_price(analysis.get('current_price'))} / "
            f"판단종가 {fmt_krw_price(analysis.get('close_price'))}"
        )
    focus = pick_focus_target(
        buy_label=(
            f"매수타점(상단 {int(analysis.get('param', 0) or 0)})"
            if str(analysis.get("strategy_name", "")).lower() == "donchian"
            else f"매수타점(SMA_{int(analysis.get('param', 0) or 0)})"
        ),
        buy_level=safe_float(analysis.get("buy_level"), default=0.0),
        buy_gap=analysis.get("buy_gap_pct"),
        buy_cond=analysis.get("buy_cond"),
        sell_label=(
            f"매도타점(하단 {int(analysis.get('sell_param', 0) or 0)})"
            if str(analysis.get("strategy_name", "")).lower() == "donchian"
            else f"매도타점(SMA_{int(analysis.get('param', 0) or 0)})"
        ),
        sell_level=safe_float(analysis.get("sell_level"), default=0.0),
        sell_gap=analysis.get("sell_gap_pct"),
        sell_cond=analysis.get("sell_cond"),
    )
    focus_line = ""
    if focus and valid_price(focus.get("level")):
        focus_line = (
            f"{focus.get('label', '타점')} {fmt_krw_price(focus.get('level'))} | "
            f"이격 {fmt_gap_pct(focus.get('gap_pct'))} | {cond_text(focus.get('cond'))}"
        )
    out = [current_line]
    if focus_line:
        out.append(focus_line)
    else:
        for ln in raw_lines:
            if ln != current_line:
                out.append(ln)
                break
    return out


# ── 포트폴리오 로드 ──────────────────────────────────────

def get_portfolio():
    """Load portfolio from PORTFOLIO env var (JSON) or portfolio.json file."""
    raw = os.getenv("PORTFOLIO")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                portfolio = data.get("portfolio", [])
                if isinstance(portfolio, list) and portfolio:
                    return portfolio
                raise RuntimeError("PORTFOLIO JSON의 portfolio 필드가 비어있습니다.")
            if isinstance(data, list) and data:
                return data
            raise RuntimeError("PORTFOLIO JSON이 비어있습니다.")
        except json.JSONDecodeError:
            raise RuntimeError(f"PORTFOLIO 환경변수가 유효한 JSON이 아닙니다: {raw[:100]}")

    candidate_paths = [
        os.path.join(PROJECT_ROOT, "portfolio.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json"),
    ]
    for p_json_path in candidate_paths:
        if not os.path.exists(p_json_path):
            continue
        with open(p_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            portfolio = data.get("portfolio", [])
        elif isinstance(data, list):
            portfolio = data
        else:
            portfolio = []
        if isinstance(portfolio, list) and portfolio:
            logger.info(f"portfolio.json에서 포트폴리오 로드: {len(portfolio)}개 자산 ({p_json_path})")
            return portfolio

    raise RuntimeError("PORTFOLIO 환경변수 또는 portfolio.json 파일을 찾을 수 없습니다.")


# ── 안전한 OHLCV 조회 ────────────────────────────────────

def safe_get_upbit_ohlcv(ticker: str, interval: str, count: int, max_retries: int = 4, wait_sec: int = 5):
    """OHLCV 데이터 최신성 검증 래퍼."""
    import src.engine.data_cache as dc
    expected_start = dc._expected_latest_candle_start_kst(interval, now_kst=None)

    for attempt in range(max_retries):
        df = get_upbit_ohlcv_local_first(ticker, interval=interval, count=count)
        if df is not None and not df.empty and expected_start is not None:
            last_dt = dc._to_kst_timestamp(df.index[-1])
            if last_dt is not None and last_dt >= expected_start:
                return df
        logger.warning(f"[{ticker}] 캔들 갱신 대기 (시도 {attempt+1}/{max_retries}). {wait_sec}초 후 재조회...")
        time.sleep(wait_sec)

    logger.error(f"[{ticker}] 최신 캔들 수신 실패. 실시간 현재가 기반 가상 캔들 추가 Fallback 실행.")
    df = get_upbit_ohlcv_local_first(ticker, interval=interval, count=count)
    if df is not None and not df.empty and expected_start is not None:
        try:
            current_price = get_upbit_price_local_first(ticker)
            if current_price and current_price > 0:
                new_idx = pd.to_datetime([expected_start])
                new_row = pd.DataFrame({
                    'open': [current_price], 'high': [current_price],
                    'low': [current_price], 'close': [current_price], 'volume': [0.0],
                }, index=new_idx)
                df = pd.concat([df, new_row])
                logger.info(f"[{ticker}] 가상 캔들 추가 완료: {expected_start} close={current_price}")
        except Exception as e:
            logger.error(f"가상 캔들 추가 실패: {e}")
    return df


# ── 자산 분석 ────────────────────────────────────────────

def analyze_asset(trader, item):
    """1단계: 데이터 조회 → 포지션 상태 계산 → 보유 현황 확인."""
    ticker = f"{item['market']}-{item['coin'].upper()}"
    strategy_name = item.get("strategy", "SMA")
    param = item.get("parameter", 20)
    interval_raw = item.get("interval", "day")
    interval = normalize_coin_interval(interval_raw)
    weight = item.get("weight", 100)

    logger.info(f"--- [{ticker}] {strategy_name}({param}), {weight}%, {interval_raw}->{interval} ---")

    count = max(200, param * 3)
    df = safe_get_upbit_ohlcv(ticker, interval=interval, count=count)
    if df is None or len(df) < param + 5:
        logger.error(f"[{ticker}] Insufficient data (got {len(df) if df is not None else 0}, need {param + 5})")
        return None

    last_candle = df.iloc[-2]
    current_price = get_upbit_price_local_first(ticker)

    sell_p = item.get("sell_parameter", 0) or max(5, int(param) // 2)

    if strategy_name == "Donchian":
        strat = DonchianStrategy()
        buy_p = int(param)
        df = strat.create_features(df, buy_period=buy_p, sell_period=sell_p)
        last_candle = df.iloc[-2]
        signal_row = last_candle.copy()
        signal_row['close'] = current_price
        position_state = strat.get_signal(signal_row, buy_period=buy_p, sell_period=sell_p)
        indicator_info = f"Upper={last_candle.get(f'Donchian_Upper_{buy_p}', 'N/A')}, Lower={last_candle.get(f'Donchian_Lower_{sell_p}', 'N/A')}"
    else:
        strat = SMAStrategy()
        df = strat.create_features(df, periods=[param])
        last_candle = df.iloc[-2]
        signal_row = last_candle.copy()
        signal_row['close'] = current_price
        position_state = strat.get_signal(signal_row, strategy_type='SMA_CROSS', ma_period=param)
        indicator_info = f"SMA_{param}={last_candle.get(f'SMA_{param}', 'N/A')}"

    coin_sym = item['coin'].upper()
    _raw_balance = trader.get_balance(coin_sym)
    try:
        coin_balance = float(_raw_balance or 0.0)
    except Exception:
        coin_balance = 0.0
    coin_value = coin_balance * float(current_price or 0.0)
    is_holding = coin_value >= MIN_ORDER_KRW

    condition_info = build_upbit_condition_info(
        strategy_name=str(strategy_name), param=int(param), sell_param=int(sell_p),
        interval=interval, last_candle=last_candle, current_price=current_price,
    )

    logger.info(f"[{ticker}] Close={last_candle['close']}, {indicator_info}, State={position_state}, Price={current_price}")
    logger.info(f"[{ticker}] {coin_sym}={coin_balance:.6f} (≈{coin_value:,.0f} KRW), Holding={is_holding}")
    logger.info(f"[{ticker}] 조건요약: {condition_info['condition_summary']}")

    return {
        'ticker': ticker, 'coin_sym': coin_sym,
        'strategy_name': str(strategy_name),
        'strategy_label': condition_info.get("strategy_label", f"{strategy_name}({param})"),
        'param': int(param), 'sell_param': int(sell_p),
        'position_state': position_state, 'signal': position_state,
        'weight': weight, 'coin_balance': coin_balance, 'coin_value': coin_value,
        'current_price': current_price,
        'close_price': condition_info.get("close_price", 0.0),
        'buy_level': condition_info.get("buy_level", 0.0),
        'sell_level': condition_info.get("sell_level", 0.0),
        'buy_gap_pct': condition_info.get("buy_gap_pct"),
        'sell_gap_pct': condition_info.get("sell_gap_pct"),
        'buy_cond': condition_info.get("buy_cond"),
        'sell_cond': condition_info.get("sell_cond"),
        'condition_summary': condition_info.get("condition_summary", ""),
        'condition_lines': condition_info.get("condition_lines", []),
        'is_holding': is_holding, 'interval': interval,
    }


# ── 메인 자동매매 루틴 ───────────────────────────────────

def run_auto_trade():
    from dotenv import load_dotenv
    load_dotenv()
    now_kst = datetime.now(KST)

    def _send_trade_notice(status: str, details: list[str] | None = None):
        lines = [f"<b>코인 자동매매 {status}</b>"]
        if details:
            lines.extend(details)
        send_telegram("\n".join(lines))

    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("API Keys not found. Set UPBIT_ACCESS_KEY and UPBIT_SECRET_KEY.")
        _send_trade_notice("실패", ["원인: UPBIT API 키 미설정"])
        return

    trader = UpbitTrader(ACCESS_KEY, SECRET_KEY)
    portfolio = get_portfolio()

    _ucfg_mode = load_user_config()
    trading_mode = _ucfg_mode.get("trading_mode", "real").lower()
    is_signal_only = (trading_mode == "signal")
    signal_test_uuids = []
    if is_signal_only:
        logger.info("⚠️ Signal Mode — 테스트 가격으로 지정가 주문 (매수 ×0.5, 매도 ×1.5)")
        cancel_old_signal_test_orders(trader)

    krw_pre = trader.get_balance("KRW")
    logger.info(f"[사전조회] KRW 잔고: {krw_pre:,.0f}원, 전략수: {len(portfolio)}개")

    has_4h = any(normalize_coin_interval(item.get("interval", "day")) == "minute240" for item in portfolio)
    if has_4h:
        now_kst = wait_for_candle_boundary(now_kst)

    signal_state = load_signal_state()
    test_order_notes: list[str] = []

    logger.info(f"=== Portfolio Auto Trade ({len(portfolio)} assets) ===")

    # ── 1단계: 전체 포지션 상태 분석 ──
    analyses = []
    skipped_analyses = []
    analyze_errors = []
    skipped_by_interval = []
    for item in portfolio:
        ticker = f"{item.get('market', 'KRW')}-{str(item.get('coin', '')).upper()}"
        iv = normalize_coin_interval(item.get("interval", "day"))
        interval_due = is_coin_interval_due(iv, now_kst)
        if not interval_due:
            skipped_by_interval.append((ticker, iv))
            logger.info(f"[{ticker}] 주기 미도래({iv}) - 매매 스킵 (분석만 수행)")
            try:
                skip_result = analyze_asset(trader, item)
                if skip_result:
                    key = make_signal_key(item)
                    prev = get_prev_state(signal_state, key)
                    skip_result['signal'] = 'SKIP'
                    skip_result['prev_state'] = prev
                    skip_result['signal_key'] = key
                    skip_result['interval_due'] = False
                    skipped_analyses.append(skip_result)
            except Exception:
                pass
            _strat_label = f"{item.get('strategy', 'SMA')}({item.get('parameter', 20)}, {item.get('interval', 'day')})"
            append_trade_log({
                "mode": "signal", "ticker": ticker, "side": "SKIP",
                "strategy": _strat_label, "reason": f"주기 미도래({iv})",
                "detail": f"interval={iv}, 보충매수/매도만 실행",
            })
            continue
        try:
            result = analyze_asset(trader, item)
            if result:
                key = make_signal_key(item)
                prev = get_prev_state(signal_state, key)
                result['signal'] = determine_signal(result['position_state'], prev)
                result['prev_state'] = prev
                result['signal_key'] = key
                raw_pos = result['position_state']
                if result['position_state'] == 'HOLD' and prev in ('BUY', 'SELL'):
                    result['position_state'] = prev
                logger.info(f"[{ticker}] 전환감지: prev={prev} → raw={raw_pos} → state={result['position_state']} → signal={result['signal']}")
                _log_entry = {
                    "mode": "signal", "ticker": ticker, "side": result['signal'],
                    "strategy": result.get('strategy_label', ''),
                    "detail": f"prev={prev} → raw={raw_pos} → {result['position_state']}",
                    "current_price": result.get('current_price', 0),
                    "buy_target": result.get('buy_level', 0), "sell_target": result.get('sell_level', 0),
                    "buy_gap": result.get('buy_gap_pct', ''), "sell_gap": result.get('sell_gap_pct', ''),
                    "condition": result.get('condition_summary', ''),
                    "reason": describe_upbit_signal_reason(result),
                }
                append_trade_log(_log_entry)
                analyses.append(result)
            else:
                analyze_errors.append(f"{ticker}=분석결과없음")
        except Exception as e:
            logger.error(f"Error analyzing {item.get('coin', '?')}: {e}")
            analyze_errors.append(f"{ticker}=ERROR({e})")

    if not analyses:
        test_order_notes = run_upbit_cycle_test_orders(trader, portfolio=portfolio, analyses=analyses)
        for _note in test_order_notes:
            logger.info(_note)
        if skipped_by_interval and not analyze_errors:
            # 주기 미도래 전략의 목표가/이격도라도 signal_state에 반영
            if skipped_analyses:
                _skip_state = dict(signal_state)
                for _sa in skipped_analyses:
                    _sk = _sa.get("signal_key")
                    if not _sk:
                        continue
                    _prev = str(_sa.get("prev_state", "")).upper() or "-"
                    _pos = str(_sa.get("position_state", "")).upper()
                    _ss = _prev if _prev not in ("-", "") else (_pos or "BUY")
                    _skip_state[_sk] = build_signal_entry(_ss, _sa)
                save_signal_state(_skip_state)
                logger.info(f"signal_state 저장 완료 (주기스킵 {len(skipped_analyses)}개 분석만)")
            logger.info("이번 실행은 주기 미도래 전략만 존재하여 주문 없이 종료: "
                        + ", ".join(f"{t}({iv})" for t, iv in skipped_by_interval[:8]))
            return
        logger.error("No assets analyzed. Exiting.")
        details = [f"원인: 분석 결과 없음 (대상 {len(portfolio)}개)"]
        if skipped_by_interval:
            details.append("주기스킵: " + ", ".join(f"{t}({iv})" for t, iv in skipped_by_interval[:8]))
        if analyze_errors:
            details.append(f"오류: {'; '.join(analyze_errors[:8])}")
        if test_order_notes:
            details.append(f"주문경로테스트: {' | '.join(test_order_notes[:3])}")
        _send_trade_notice("실패", details)
        return

    # ── 총 포트폴리오 가치 계산 ──
    krw_balance = trader.get_balance("KRW")
    seen_coins = set()
    total_coin_value = 0
    for a in analyses:
        if a['coin_sym'] not in seen_coins:
            total_coin_value += a['coin_value']
            seen_coins.add(a['coin_sym'])
    total_portfolio_value = krw_balance + total_coin_value

    coin_weight_sum = {}
    for a in analyses:
        coin_weight_sum[a['coin_sym']] = coin_weight_sum.get(a['coin_sym'], 0) + a['weight']

    logger.info(f"Portfolio Value={total_portfolio_value:,.0f} KRW (현금={krw_balance:,.0f}, 코인={total_coin_value:,.0f})")

    # ── 2단계: 매도 먼저 실행 ──
    exec_results = {}
    decision_notes = {}
    sold_qty_map = {}
    for a in analyses:
        if a['signal'] == 'SELL' and a['is_holding']:
            target_value = total_portfolio_value * (a['weight'] / 100)
            if a['current_price'] and a['current_price'] > 0:
                target_qty = target_value / a['current_price']
            else:
                continue
            already_sold = sold_qty_map.get(a['coin_sym'], 0)
            remaining_balance = a['coin_balance'] - already_sold
            sell_qty = min(target_qty, remaining_balance)
            if sell_qty <= 0:
                continue
            sell_value = sell_qty * a['current_price'] if a['current_price'] else 0
            if sell_value < MIN_ORDER_KRW:
                note = f"SELL 스킵({a.get('strategy_label', '')}): 주문금액 미달 {sell_value:,.0f}원 < {MIN_ORDER_KRW:,.0f}원"
                decision_notes.setdefault(a['ticker'], []).append(note)
                continue
            coin_total_w = coin_weight_sum.get(a['coin_sym'], a['weight'])
            logger.info(f"[{a['ticker']}] SELL {sell_qty:.6f}/{a['coin_balance']:.6f} {a['coin_sym']} (비중={a['weight']}%/{coin_total_w}%)")

            if is_signal_only:
                test_price = int(a['current_price'] * UPBIT_TEST_SELL_MULTIPLIER)
                try:
                    _test_res = trader.sell_limit(a['ticker'], test_price, sell_qty)
                    _test_uuid = _test_res.get('uuid') if isinstance(_test_res, dict) else None
                    if _test_uuid:
                        signal_test_uuids.append({
                            "uuid": _test_uuid, "ticker": a['ticker'], "side": "SELL",
                            "price": test_price, "qty": sell_qty,
                            "created_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
                        })
                except Exception as _te:
                    logger.error(f"[{a['ticker']}] 테스트 매도 오류: {_te}")
                decision_notes.setdefault(a['ticker'], []).append(f"SELL 시그널 (Signal Mode, 테스트 매도 @ {test_price:,})")
                append_trade_log({
                    "mode": "signal", "ticker": a['ticker'], "side": "SELL",
                    "qty": f"{sell_qty:.8g}", "strategy": a.get('strategy_label', ''),
                    "result": "signal_test", "test_price": test_price,
                })
                sold_qty_map[a['coin_sym']] = already_sold + sell_qty
                continue

            try:
                result = trader.smart_sell(a['ticker'], sell_qty, interval=a.get('interval', 'day'))
                logger.info(f"[{a['ticker']}] Sell Result: {result}")
                if isinstance(result, dict):
                    result = dict(result)
                else:
                    result = {"raw_result": result}
                result["_strategy_label"] = a.get("strategy_label", "")
                result["_condition_summary"] = a.get("condition_summary", "")
                exec_results.setdefault(a['ticker'], []).append(result)
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "SELL",
                    "qty": f"{sell_qty:.8g}", "strategy": a.get('strategy_label', ''),
                    "result": "success", "detail": str(result)[:200],
                })
                sold_qty_map[a['coin_sym']] = already_sold + sell_qty
            except Exception as e:
                logger.error(f"[{a['ticker']}] Sell Error: {e}")
                decision_notes.setdefault(a['ticker'], []).append(f"SELL 주문오류({a.get('strategy_label', '')}): {e}")
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "SELL",
                    "qty": f"{sell_qty:.8g}", "strategy": a.get('strategy_label', ''),
                    "result": "error", "detail": str(e)[:200],
                })

        elif a['signal'] == 'SELL' and not a['is_holding']:
            note = f"SELL 스킵({a.get('strategy_label', '')}): 보유수량/평가금액 부족"
            decision_notes.setdefault(a['ticker'], []).append(note)

    time.sleep(1)

    # ── 3단계: 매수 실행 ──
    krw_balance = trader.get_balance("KRW")
    buy_signals = [a for a in analyses if a['signal'] == 'BUY']
    logger.info(f"KRW(매도후)={krw_balance:,.0f} | Buy signals={len(buy_signals)}")

    for a in buy_signals:
        target_value = total_portfolio_value * (a['weight'] / 100)
        current_holding_value = a['coin_value']
        coin_total_w = coin_weight_sum.get(a['coin_sym'], a['weight'])
        my_holding = current_holding_value * (a['weight'] / coin_total_w)
        need_value = target_value - my_holding
        buy_budget = min(need_value, krw_balance) * 0.999

        if buy_budget > MIN_ORDER_KRW:
            logger.info(f"[{a['ticker']}] BUY {buy_budget:,.0f} KRW (Adaptive) | 목표={target_value:,.0f}, 보유={my_holding:,.0f}")

            if is_signal_only:
                test_price = max(1, int(a['current_price'] * UPBIT_TEST_BUY_MULTIPLIER))
                test_vol = buy_budget / test_price
                try:
                    _test_res = trader.buy_limit(a['ticker'], test_price, test_vol)
                    _test_uuid = _test_res.get('uuid') if isinstance(_test_res, dict) else None
                    if _test_uuid:
                        signal_test_uuids.append({
                            "uuid": _test_uuid, "ticker": a['ticker'], "side": "BUY",
                            "price": test_price, "amount": buy_budget,
                            "created_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST"),
                        })
                except Exception as _te:
                    logger.error(f"[{a['ticker']}] 테스트 매수 오류: {_te}")
                decision_notes.setdefault(a['ticker'], []).append(f"BUY 시그널 (Signal Mode, 테스트 매수 @ {test_price:,})")
                append_trade_log({
                    "mode": "signal", "ticker": a['ticker'], "side": "BUY",
                    "amount": f"{buy_budget:,.0f}", "strategy": a.get('strategy_label', ''),
                    "result": "signal_test", "test_price": test_price,
                })
                continue

            try:
                result = trader.adaptive_buy(a['ticker'], buy_budget, interval=a.get('interval', 'day'))
                logger.info(f"[{a['ticker']}] Buy Result: {result}")
                if isinstance(result, dict):
                    result = dict(result)
                else:
                    result = {"raw_result": result}
                result["_strategy_label"] = a.get("strategy_label", "")
                result["_condition_summary"] = a.get("condition_summary", "")
                exec_results.setdefault(a['ticker'], []).append(result)
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "BUY",
                    "amount": f"{buy_budget:,.0f}", "strategy": a.get('strategy_label', ''),
                    "result": "success", "detail": str(result)[:200],
                })
                krw_balance -= buy_budget
            except Exception as e:
                logger.error(f"[{a['ticker']}] Buy Error: {e}")
                decision_notes.setdefault(a['ticker'], []).append(f"BUY 주문오류({a.get('strategy_label', '')}): {e}")
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "BUY",
                    "amount": f"{buy_budget:,.0f}", "strategy": a.get('strategy_label', ''),
                    "result": "error", "detail": str(e)[:200],
                })
        else:
            note = f"BUY 스킵({a.get('strategy_label', '')}): 예산부족 필요={need_value:,.0f}원, 가용={krw_balance:,.0f}원"
            decision_notes.setdefault(a['ticker'], []).append(note)

    # ── 3.5단계: 보충 매수/매도 ──
    _ucfg = load_user_config()
    topup_enabled = _ucfg.get("topup_enabled", False)
    topup_buy_amount = int(_ucfg.get("topup_buy_amount", MIN_ORDER_KRW))
    topup_sell_amount = int(_ucfg.get("topup_sell_amount", MIN_ORDER_KRW))

    if topup_enabled and is_signal_only:
        logger.info("보충 매수/매도: Signal Mode → 테스트 주문으로 대체 (건너뜀)")
    elif topup_enabled:
        krw_balance = trader.get_balance("KRW")
        topup_done = set()

        # 보충 매수
        for a in analyses:
            if a['signal'] != 'HOLD' or a['position_state'] not in ('BUY', 'HOLD'):
                continue
            if a['coin_sym'] in topup_done:
                continue
            target_value = total_portfolio_value * (a['weight'] / 100)
            coin_total_w = coin_weight_sum.get(a['coin_sym'], a['weight'])
            my_holding = a['coin_value'] * (a['weight'] / coin_total_w)
            need_value = target_value - my_holding
            if need_value < MIN_ORDER_KRW:
                continue
            buy_budget = min(topup_buy_amount, need_value, krw_balance) * 0.999
            if buy_budget < MIN_ORDER_KRW * 0.999:
                continue
            logger.info(f"[{a['ticker']}] 보충매수 {buy_budget:,.0f} KRW")
            try:
                result = trader.adaptive_buy(a['ticker'], buy_budget, interval=a.get('interval', 'day'))
                if isinstance(result, dict):
                    result = dict(result)
                else:
                    result = {"raw_result": result}
                result["_strategy_label"] = a.get("strategy_label", "")
                result["_topup"] = True
                exec_results.setdefault(a['ticker'], []).append(result)
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "BUY_TOPUP",
                    "amount": f"{buy_budget:,.0f}", "strategy": a.get('strategy_label', ''),
                    "result": "success", "detail": str(result)[:200],
                })
                krw_balance -= buy_budget
                topup_done.add(a['coin_sym'])
            except Exception as e:
                logger.error(f"[{a['ticker']}] 보충매수 Error: {e}")
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "BUY_TOPUP",
                    "amount": f"{buy_budget:,.0f}", "strategy": a.get('strategy_label', ''),
                    "result": "error", "detail": str(e)[:200],
                })

        # 보충 매도
        topup_sold = set()
        for a in analyses:
            if a['signal'] != 'HOLD' or a['position_state'] != 'SELL':
                continue
            if not a['is_holding'] or a['coin_sym'] in topup_sold:
                continue
            sell_value = a['coin_value']
            if sell_value < MIN_ORDER_KRW:
                continue
            if a['current_price'] and a['current_price'] > 0:
                sell_krw = min(topup_sell_amount, sell_value)
                sell_qty = min(sell_krw / a['current_price'], a['coin_balance'])
            else:
                continue
            act_value = sell_qty * a['current_price']
            if act_value < MIN_ORDER_KRW:
                continue
            logger.info(f"[{a['ticker']}] 보충매도 {sell_qty:.8g} {a['coin_sym']}")
            try:
                result = trader.smart_sell(a['ticker'], sell_qty, interval=a.get('interval', 'day'))
                if isinstance(result, dict):
                    result = dict(result)
                else:
                    result = {"raw_result": result}
                result["_strategy_label"] = a.get("strategy_label", "")
                result["_topup"] = True
                exec_results.setdefault(a['ticker'], []).append(result)
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "SELL_TOPUP",
                    "qty": f"{sell_qty:.8g}", "strategy": a.get('strategy_label', ''),
                    "result": "success", "detail": str(result)[:200],
                })
                topup_sold.add(a['coin_sym'])
            except Exception as e:
                logger.error(f"[{a['ticker']}] 보충매도 Error: {e}")
                append_trade_log({
                    "mode": "real", "ticker": a['ticker'], "side": "SELL_TOPUP",
                    "qty": f"{sell_qty:.8g}", "strategy": a.get('strategy_label', ''),
                    "result": "error", "detail": str(e)[:200],
                })

        # interval 스킵 전략 보충
        if skipped_by_interval:
            for skip_ticker, skip_iv in skipped_by_interval:
                skip_item = next(
                    (p for p in portfolio
                     if f"{p.get('market','KRW')}-{p['coin'].upper()}" == skip_ticker
                     and normalize_coin_interval(p.get('interval', 'day')) == skip_iv),
                    None,
                )
                if not skip_item:
                    continue
                skip_key = make_signal_key(skip_item)
                maint_state = get_prev_state(signal_state, skip_key)
                if maint_state == 'HOLD':
                    maint_state = 'BUY'
                if not maint_state or maint_state not in ('BUY', 'SELL'):
                    continue
                skip_coin = skip_item['coin'].upper()
                skip_weight = float(skip_item.get('weight', 0))
                skip_label = f"{skip_item.get('strategy','SMA')}({skip_item.get('parameter',20)}, {skip_iv})"

                try:
                    skip_bal = float(trader.get_balance(skip_coin) or 0.0)
                except Exception:
                    skip_bal = 0.0
                skip_price = get_upbit_price_local_first(skip_ticker)
                skip_val = skip_bal * float(skip_price or 0)
                skip_holding = skip_val >= MIN_ORDER_KRW

                if maint_state == 'BUY' and skip_coin not in topup_done:
                    target_v = total_portfolio_value * (skip_weight / 100)
                    cw = coin_weight_sum.get(skip_coin, skip_weight)
                    my_hold = skip_val * (skip_weight / cw) if cw > 0 else 0
                    need_v = target_v - my_hold
                    if need_v >= MIN_ORDER_KRW:
                        buy_b = min(topup_buy_amount, need_v, krw_balance) * 0.999
                        if buy_b >= MIN_ORDER_KRW * 0.999:
                            try:
                                result = trader.adaptive_buy(skip_ticker, buy_b, interval=skip_iv)
                                if isinstance(result, dict):
                                    result = dict(result)
                                else:
                                    result = {"raw_result": result}
                                result["_strategy_label"] = skip_label
                                result["_topup"] = True
                                exec_results.setdefault(skip_ticker, []).append(result)
                                append_trade_log({
                                    "mode": "real", "ticker": skip_ticker, "side": "BUY_TOPUP",
                                    "amount": f"{buy_b:,.0f}", "strategy": skip_label,
                                    "result": "success", "detail": str(result)[:200],
                                })
                                krw_balance -= buy_b
                                topup_done.add(skip_coin)
                            except Exception as e:
                                logger.error(f"[{skip_ticker}] 보충매수(스킵) Error: {e}")

                elif maint_state == 'SELL' and skip_holding and skip_coin not in topup_sold:
                    if skip_price and skip_price > 0:
                        sell_krw = min(topup_sell_amount, skip_val)
                        sell_qty = min(sell_krw / skip_price, skip_bal)
                        act_val = sell_qty * skip_price
                        if act_val >= MIN_ORDER_KRW:
                            try:
                                result = trader.smart_sell(skip_ticker, sell_qty, interval=skip_iv)
                                if isinstance(result, dict):
                                    result = dict(result)
                                else:
                                    result = {"raw_result": result}
                                result["_strategy_label"] = skip_label
                                result["_topup"] = True
                                exec_results.setdefault(skip_ticker, []).append(result)
                                append_trade_log({
                                    "mode": "real", "ticker": skip_ticker, "side": "SELL_TOPUP",
                                    "qty": f"{sell_qty:.8g}", "strategy": skip_label,
                                    "result": "success", "detail": str(result)[:200],
                                })
                                topup_sold.add(skip_coin)
                            except Exception as e:
                                logger.error(f"[{skip_ticker}] 보충매도(스킵) Error: {e}")
    else:
        logger.info("보충 매수/매도 비활성 (user_config.topup_enabled=false)")

    # HOLD 로깅
    for a in analyses:
        if a['signal'] == 'HOLD':
            reason = describe_upbit_signal_reason(a)
            logger.info(f"[{a['ticker']}] HOLD (state={a['position_state']}, prev={a.get('prev_state')}) - {reason}")

    # ── 4단계: 포지션 상태 저장 ──
    next_signal_state = dict(signal_state)

    # 주기 미도래(skipped) 전략도 목표가/이격도 업데이트 (상태는 이전값 유지)
    for a in skipped_analyses:
        key = a.get("signal_key")
        if not key:
            continue
        prev = str(a.get("prev_state", "")).upper() or "-"
        pos_state = str(a.get("position_state", "")).upper()
        save_state = prev if prev not in ("-", "") else (pos_state or "BUY")
        next_signal_state[key] = build_signal_entry(save_state, a)

    for a in analyses:
        key = a.get("signal_key")
        if not key:
            continue
        sig = str(a.get("signal", "")).upper()
        prev = str(a.get("prev_state", "")).upper() or "-"
        pos_state = str(a.get("position_state", "")).upper()
        strat_label = str(a.get("strategy_label", ""))
        ticker = str(a.get("ticker", ""))

        if sig == "HOLD":
            if prev == "-" or prev is None:
                init_state = "BUY" if bool(a.get("is_holding")) else "SELL"
                next_signal_state[key] = build_signal_entry(init_state, a)
            else:
                next_signal_state[key] = build_signal_entry(prev, a)
            continue

        if is_signal_only:
            if sig in ("BUY", "SELL"):
                next_signal_state[key] = build_signal_entry(sig, a)
            continue

        if sig == "SELL":
            if not bool(a.get("is_holding")):
                next_signal_state[key] = build_signal_entry("SELL", a)
                continue
            if has_strategy_filled_exec(exec_results, ticker, strat_label, side="sell"):
                next_signal_state[key] = build_signal_entry("SELL", a)
                continue
            next_signal_state[key] = build_signal_entry(prev if prev != "-" else pos_state, a)
            continue

        if sig == "BUY":
            if has_strategy_filled_exec(exec_results, ticker, strat_label, side="buy"):
                next_signal_state[key] = build_signal_entry("BUY", a)
                continue
            next_signal_state[key] = build_signal_entry(prev if prev != "-" else pos_state, a)
            continue

    save_signal_state(next_signal_state)

    if is_signal_only and signal_test_uuids:
        existing = load_signal_test_orders()
        existing.extend(signal_test_uuids)
        save_signal_test_orders(existing)

    test_order_notes = run_upbit_cycle_test_orders(trader, portfolio=portfolio, analyses=analyses)
    for _note in test_order_notes:
        logger.info(_note)

    logger.info("=== Done ===")

    # ── 텔레그램 상세 전송 ──
    updated_bal = trader.get_all_balances() or {}
    krw_after = safe_float(updated_bal.get('KRW', 0))

    _coin_total_after = 0
    _coin_bal_parts = []
    _seen = set()
    for a in analyses:
        sym = a['coin_sym']
        if sym in _seen:
            continue
        _seen.add(sym)
        bal = safe_float(updated_bal.get(sym, 0))
        price = a['current_price'] or 0
        val = bal * price
        _coin_total_after += val
        if bal > 0:
            _coin_bal_parts.append(f"{sym} {bal:.8g} (≈{val:,.0f}원)")

    total_after = krw_after + _coin_total_after
    _trade_prices = {f"KRW-{a['coin_sym']}": a['current_price'] for a in analyses if a.get('current_price')}
    save_balance_cache(updated_bal, _trade_prices)

    _tg_coin_groups = OrderedDict()
    for a in analyses:
        _tg_coin_groups.setdefault(a['coin_sym'], []).append(a)
    for a in skipped_analyses:
        _tg_coin_groups.setdefault(a['coin_sym'], []).append(a)

    _mode_label = "📡 Signal Mode" if is_signal_only else "🔴 Real Trading"
    tg_lines = [f"<b>코인 자동매매 완료 [{_mode_label}]</b>", ""]

    for _sym, _grp in _tg_coin_groups.items():
        _actions = [a['signal'] for a in _grp]
        _has_buy = any(s == 'BUY' for s in _actions)
        _has_sell = any(s == 'SELL' for s in _actions)
        _ticker = _grp[0]['ticker']
        _execs = exec_results.get(_ticker, [])
        _printed_exec = False
        for ex in _execs:
            _type = ex.get('type', '')
            _vol = safe_float(ex.get('filled_volume', 0))
            _avg = safe_float(ex.get('avg_price', 0))
            _krw = safe_float(ex.get('total_krw', 0))
            _strategy = ex.get("_strategy_label", "")
            _cond = ex.get("_condition_summary", "")
            if 'sell' in _type and _vol > 0:
                tg_lines.append(f"{_sym} SELL {_vol:.8g} @ {_avg:,.0f} = {_krw:,.0f}원")
                _printed_exec = True
            elif _vol > 0:
                tg_lines.append(f"{_sym} BUY {_krw:,.0f}원 @ {_avg:,.0f} = {_vol:.8g}")
                _printed_exec = True
            else:
                _act = "SELL" if "sell" in str(_type).lower() else "BUY"
                tg_lines.append(f"{_sym} {_act} 주문 미체결 (체결 0)")
            if _strategy or _cond:
                tg_lines.append(f"  주문조건[{_strategy or '전략'}]: {_cond}")

        if not _execs and not _has_buy and not _has_sell:
            _val = _grp[0]['coin_value']
            tg_lines.append(f"{_sym} HOLD ≈{_val:,.0f}원")
            _printed_exec = True

        for _note in decision_notes.get(_ticker, []):
            tg_lines.append(f"{_sym} {_note}")
            _printed_exec = True

        for _a in _grp:
            _reason = describe_upbit_signal_reason(_a)
            _prev = _a.get("prev_state")
            _prev_text = _prev if _prev not in (None, "") else "-"
            _sig_display = _a.get('signal')
            if _sig_display == 'SKIP':
                _sig_display = f"SKIP (판단={_a.get('position_state')}, 이전={_prev_text})"
            else:
                _sig_display = f"실행={_sig_display} (판단={_a.get('position_state')}, 이전={_prev_text})"
            tg_lines.append(f"{_a.get('strategy_label', '전략')} | {_sig_display}")
            for _line in compact_upbit_condition_lines(_a):
                tg_lines.append(f"  {_line}")
            tg_lines.append(f"  상태: {_reason}")

        if _printed_exec:
            tg_lines.append("")

    if test_order_notes:
        tg_lines.append("<b>주문 경로 테스트</b>")
        for _note in test_order_notes:
            tg_lines.append(_note)
        tg_lines.append("")

    tg_lines.append("")
    tg_lines.append(f"<b>잔고</b>")
    tg_lines.append(f"KRW {krw_after:,.0f}원")
    for bp in _coin_bal_parts:
        tg_lines.append(bp)
    tg_lines.append(f"<b>총자산 {total_after:,.0f}원</b>")
    send_telegram("\n".join(tg_lines))
