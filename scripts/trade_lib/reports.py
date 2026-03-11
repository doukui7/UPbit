"""일일 리포트, 계좌 동기화, 수동 주문, 텔레그램 테스트."""
import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

from .constants import (
    KST, PROJECT_ROOT, MIN_ORDER_KRW,
    ACCOUNT_CACHE_FILE, GOLD_KRX_ETF_CODE,
)
from .utils import safe_float, get_env_any
from .notifier import send_telegram
from .state import append_trade_log, save_balance_cache
from .data import get_upbit_price_local_first, get_gold_price_local_first, get_kis_price_local_first
from .kis_ops import (
    get_kis_balance_with_retry, get_kis_orderable_cash_precise,
    extract_kis_balance_fail_detail,
    normalize_kis_account_fields, normalize_gold_kr_etf,
    format_holdings_brief,
)

from src.strategy.laa import LAAStrategy
from src.trading.upbit_trader import UpbitTrader
from src.engine.kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG
from src.engine.kis_trader import KISTrader
from .data import get_kis_daily_local_first

logger = logging.getLogger(__name__)


# ── 일일 자산 현황 리포트 ────────────────────────────

def run_daily_status_report():
    """매일 아침 전체계좌 자산 잔고/보유현황 텔레그램으로 전송."""
    load_dotenv()
    now = datetime.now(KST)
    lines = [f"<b>일일 자산 현황</b> ({now.strftime('%m-%d %H:%M')} KST)", ""]

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
            coin_details = []
            for sym, qty in coins.items():
                p = get_upbit_price_local_first(f"KRW-{sym}")
                val = qty * float(p) if p else 0
                est_coin_value += val
                coin_details.append(f"{sym} {qty:.8g} ({val:,.0f}원)")
            total = krw + est_coin_value
            lines.append(f"<b>[업비트]</b> 총 {total:,.0f}원")
            lines.append(f"  KRW {krw:,.0f}원 / 코인 {est_coin_value:,.0f}원")
            if coin_details:
                for cd in coin_details[:5]:
                    lines.append(f"  {cd}")
            lines.append("")
        except Exception as e:
            lines.append(f"<b>[업비트]</b> 조회 실패: {e}")
            lines.append("")

    # Kiwoom Gold
    if (os.getenv("Kiwoom_App_Key") or os.getenv("KIWOOM_APP_KEY")) and \
       (os.getenv("Kiwoom_Secret_Key") or os.getenv("KIWOOM_SECRET_KEY")):
        try:
            trader = KiwoomGoldTrader(is_mock=False)
            if trader.auth():
                bal = trader.get_balance() or {}
                cash = float(bal.get("cash_krw", 0.0))
                qty = float(bal.get("gold_qty", 0.0))
                eval_amt = float(bal.get("gold_eval", 0.0))
                if eval_amt <= 0:
                    p = get_gold_price_local_first(trader, GOLD_CODE_1KG) or 0
                    eval_amt = qty * float(p) if p else 0.0
                total = cash + eval_amt
                lines.append(f"<b>[키움 금현물]</b> 총 {total:,.0f}원")
                lines.append(f"  현금 {cash:,.0f}원 / 금 {qty:.4f}g ({eval_amt:,.0f}원)")
                lines.append("")
            else:
                lines.append(f"<b>[키움 금현물]</b> 인증 실패")
                lines.append("")
        except Exception as e:
            lines.append(f"<b>[키움 금현물]</b> 조회 실패: {e}")
            lines.append("")

    # KIS ISA
    isa_key = get_env_any("KIS_ISA_APP_KEY", "KIS_APP_KEY")
    isa_secret = get_env_any("KIS_ISA_APP_SECRET", "KIS_APP_SECRET")
    isa_account_raw = get_env_any("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    isa_prdt_raw = get_env_any("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    isa_account, isa_prdt = normalize_kis_account_fields(isa_account_raw, isa_prdt_raw)
    if isa_key and isa_secret and isa_account:
        try:
            trader = KISTrader(is_mock=False)
            trader.app_key = isa_key
            trader.app_secret = isa_secret
            trader.account_no = isa_account
            trader.acnt_prdt_cd = isa_prdt
            if trader.auth():
                bal = get_kis_balance_with_retry(trader, retries=4, delay_sec=0.9, log_prefix="KIS ISA")
                if bal is None:
                    probe = trader.get_balance()
                    detail = extract_kis_balance_fail_detail(probe)
                    detail_txt = f" ({detail})" if detail else ""
                    lines.append(f"<b>[KIS ISA]</b> 조회 실패: 잔고 조회 실패{detail_txt}")
                    lines.append("")
                else:
                    cash = float(bal.get("cash", 0.0))
                    holdings = bal.get("holdings", []) or []
                    total_eval = float(bal.get("total_eval", 0.0)) or (
                        cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings)
                    )
                    lines.append(f"<b>[KIS ISA]</b> 총 {total_eval:,.0f}원")
                    lines.append(f"  예수금 {cash:,.0f}원 / 보유: {format_holdings_brief(holdings)}")
                    lines.append("")
            else:
                lines.append(f"<b>[KIS ISA]</b> 인증 실패")
                lines.append("")
        except Exception as e:
            lines.append(f"<b>[KIS ISA]</b> 조회 실패: {e}")
            lines.append("")

    # KIS Pension
    pension_acct_raw = get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    pension_prdt_raw = get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    pension_acct, pension_prdt = normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)
    if pension_acct:
        try:
            trader = KISTrader(is_mock=False)
            pension_key = get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
            pension_secret = get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
            if pension_key:
                trader.app_key = pension_key
            if pension_secret:
                trader.app_secret = pension_secret
            if pension_acct:
                trader.account_no = pension_acct
                trader.acnt_prdt_cd = pension_prdt

            if trader.auth():
                bal = get_kis_balance_with_retry(trader, retries=4, delay_sec=0.9, log_prefix="KIS 연금저축")
                if bal is None:
                    probe = trader.get_balance()
                    detail = extract_kis_balance_fail_detail(probe)
                    detail_txt = f" ({detail})" if detail else ""
                    lines.append(f"<b>[KIS 연금저축]</b> 조회 실패: 잔고 조회 실패{detail_txt}")
                    lines.append("")
                else:
                    pension_base_code = get_env_any("KR_ETF_LAA_QQQ", default="133690")
                    cash = get_kis_orderable_cash_precise(
                        trader, bal, base_code=pension_base_code, fallback=0.0
                    )
                    holdings = bal.get("holdings", []) or []
                    stock_eval = float(bal.get("stock_eval", 0.0)) or sum(
                        float(h.get("eval_amt", 0.0) or 0.0) for h in holdings
                    )
                    total_eval = cash + stock_eval
                    lines.append(f"<b>[KIS 연금저축]</b> 총 {total_eval:,.0f}원")
                    lines.append(f"  예수금(주문가능) {cash:,.0f}원 / 보유: {format_holdings_brief(holdings)}")

                    # LAA 전략 시그널 체크 (주문 없이 분석만)
                    try:
                        kr_etf_map = {
                            "SPY": get_env_any("KR_ETF_LAA_SPY", "KR_ETF_SPY", default="360750"),
                            "IWD": get_env_any("KR_ETF_LAA_IWD", "KR_ETF_SPY", default="360750"),
                            "GLD": normalize_gold_kr_etf(get_env_any("KR_ETF_LAA_GLD", "KR_ETF_GOLD", default=GOLD_KRX_ETF_CODE)),
                            "IEF": get_env_any("KR_ETF_LAA_IEF", "KR_ETF_AGG", default="305080"),
                            "QQQ": get_env_any("KR_ETF_LAA_QQQ", default="133690"),
                            "SHY": get_env_any("KR_ETF_LAA_SHY", default="329750"),
                        }
                        tickers_laa = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
                        price_data = {}
                        price_ok = True
                        for t in tickers_laa:
                            code = str(kr_etf_map.get(t, "")).strip()
                            df = get_kis_daily_local_first(trader, code, count=320) if code else None
                            if df is None or len(df) == 0:
                                price_ok = False
                                break
                            price_data[t] = df
                            time.sleep(0.15)

                        if price_ok:
                            strategy = LAAStrategy(settings={"kr_etf_map": kr_etf_map})
                            signal = strategy.analyze(price_data)
                            if signal:
                                risk_label = "공격" if signal.get("risk_on") else "방어"
                                risk_asset = signal.get("selected_risk_asset", "?")
                                risk_kr = signal.get("selected_risk_kr_code", "?")

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
                                    price = get_kis_price_local_first(trader, code) or 0.0
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
                                lines.append(f"  LAA {risk_label} | {risk_asset}->{risk_kr} | {action} (괴리 {max_gap*100:.1f}%p)")
                                if planned_orders:
                                    for po in planned_orders:
                                        lines.append(f"  - {po}")
                            else:
                                lines.append("  LAA 분석 실패")
                        else:
                            lines.append("  ETF 조회 실패 (시그널 생략)")
                    except Exception as sig_e:
                        lines.append(f"  시그널 오류: {sig_e}")

                    lines.append("")
            else:
                lines.append(f"<b>[KIS 연금저축]</b> 인증 실패")
                lines.append("")
        except Exception as e:
            lines.append(f"<b>[KIS 연금저축]</b> 조회 실패: {e}")
            lines.append("")

    if len(lines) <= 2:
        lines.append("조회 가능한 계좌가 없습니다. API 키/계좌 설정을 확인하세요.")

    report = "\n".join(lines)
    logger.info(report.replace("<b>", "").replace("</b>", ""))
    send_telegram(report)


# ── 계좌 동기화 ──────────────────────────────────────

def run_account_sync():
    """VM 경유 계좌 데이터 동기화. Upbit API → account_cache.json."""
    load_dotenv()
    now_kst = datetime.now(KST)

    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("account_sync: UPBIT API 키 미설정")
        return

    trader = UpbitTrader(ACCESS_KEY, SECRET_KEY)

    cache = {
        "updated_at": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "balances": {},
        "orders": [],
        "deposits": [],
        "withdraws": [],
    }

    # 1) 잔고
    try:
        all_bal = trader.get_all_balances()
        if isinstance(all_bal, dict):
            cache["balances"] = {k: float(v) for k, v in all_bal.items()}
    except Exception as e:
        logger.error(f"account_sync 잔고 조회 실패: {e}")

    # 2) 체결 주문 내역
    try:
        orders, err = trader.get_history("order")
        if orders:
            cache["orders"] = orders
        if err:
            logger.warning(f"account_sync 주문 조회 경고: {err}")
    except Exception as e:
        logger.error(f"account_sync 주문 조회 실패: {e}")

    # 3) 입금 내역
    try:
        deposits, err = trader.get_history("deposit")
        if deposits:
            cache["deposits"] = deposits
        if err:
            logger.warning(f"account_sync 입금 조회 경고: {err}")
    except Exception as e:
        logger.error(f"account_sync 입금 조회 실패: {e}")

    # 4) 출금 내역
    try:
        withdraws, err = trader.get_history("withdraw")
        if withdraws:
            cache["withdraws"] = withdraws
        if err:
            logger.warning(f"account_sync 출금 조회 경고: {err}")
    except Exception as e:
        logger.error(f"account_sync 출금 조회 실패: {e}")

    # 5) 미체결 주문
    try:
        pending = trader.get_orders(state="wait") or []
        cache["pending_orders"] = pending if isinstance(pending, list) else []
        logger.info(f"미체결 주문: {len(cache['pending_orders'])}건")
    except Exception as e:
        logger.error(f"account_sync 미체결 조회 실패: {e}")
        cache["pending_orders"] = []

    # 로컬 저장
    try:
        content = json.dumps(cache, indent=2, ensure_ascii=False, default=str)
        with open(ACCOUNT_CACHE_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"account_cache 로컬 저장 완료: {ACCOUNT_CACHE_FILE}")
    except Exception as e:
        logger.error(f"account_cache 저장 실패: {e}")

    # 잔고 캐시도 함께 갱신 (현재가 포함)
    if cache["balances"]:
        try:
            _prices: dict = {}
            for sym in list(cache["balances"].keys()):
                if sym == "KRW":
                    continue
                _ticker = f"KRW-{sym}"
                try:
                    _p = float(trader.get_current_price(_ticker) or 0.0)
                    if _p > 0:
                        _prices[_ticker] = _p
                except Exception:
                    pass
            save_balance_cache(cache["balances"], _prices if _prices else None)
        except Exception as _e:
            logger.warning(f"account_sync 현재가 조회 실패, 가격 없이 저장: {_e}")
            save_balance_cache(cache["balances"])

    pending_cnt = len(cache.get('pending_orders', []))
    logger.info(f"account_sync 완료 - 잔고: {len(cache['balances'])}종, "
                f"주문: {len(cache['orders'])}건, "
                f"미체결: {pending_cnt}건, "
                f"입금: {len(cache['deposits'])}건, "
                f"출금: {len(cache['withdraws'])}건")

    # 포트폴리오 상세 + 전략 정보 구성
    _port_lines = []
    _total_val = 0.0
    _bal = cache.get("balances", {})
    _krw = float(_bal.get("KRW", 0.0))
    _total_val += _krw
    for _sym, _qty in _bal.items():
        if _sym == "KRW":
            continue
        _q = float(_qty)
        _p = float(_prices.get(f"KRW-{_sym}", 0.0)) if _prices else 0.0
        _v = _q * _p
        _total_val += _v
        _port_lines.append(f"  {_sym} {_q:.8g} ({_v:,.0f}원)")
    _port_msg = f"총 자산: {_total_val:,.0f}원 (현금 {_krw:,.0f}원)"
    if _port_lines:
        _port_msg += "\n" + "\n".join(_port_lines)

    # 전략 정보
    _strat_msg = ""
    try:
        _ucfg_path = os.path.join(PROJECT_ROOT, "user_config.json")
        with open(_ucfg_path, "r", encoding="utf-8") as _f:
            _ucfg = json.load(_f)
        _portfolio = _ucfg.get("portfolio", [])
        if _portfolio:
            _parts = []
            for _p_item in _portfolio:
                _s = _p_item.get("strategy", "?")
                _param = _p_item.get("parameter", "")
                _w = _p_item.get("weight", 0)
                _intv = _p_item.get("interval", "")
                _intv_short = {"day": "D", "minute240": "4H", "minute60": "1H"}.get(_intv, _intv)
                _parts.append(f"{_s}({_param},{_intv_short}) {_w:.0f}%")
            _strat_msg = "전략: " + " + ".join(_parts)
    except Exception:
        pass

    # 포지션 상태
    _pos_msg = ""
    try:
        _ss_path = os.path.join(PROJECT_ROOT, "signal_state.json")
        with open(_ss_path, "r", encoding="utf-8") as _f:
            _ss = json.load(_f)
        _pos_parts = []
        for _key, _val in _ss.items():
            if isinstance(_val, dict):
                _st = _val.get("state", "?")
            else:
                _st = str(_val)
            _short_key = _key.replace("KRW-BTC_", "").replace("_", " ")
            _pos_parts.append(f"{_short_key}={_st}")
        if _pos_parts:
            _pos_msg = "포지션: " + " | ".join(_pos_parts)
    except Exception:
        pass

    _tg_parts = [
        f"<b>계좌 동기화 완료</b>",
        f"시각: {now_kst.strftime('%m-%d %H:%M')}",
    ]
    if _strat_msg:
        _tg_parts.append(_strat_msg)
    _tg_parts.append(_port_msg)
    if _pos_msg:
        _tg_parts.append(_pos_msg)
    _tg_parts.append(
        f"주문: {len(cache['orders'])}건"
        + (f" | 미체결: {pending_cnt}건" if pending_cnt > 0 else "")
    )
    send_telegram("\n".join(_tg_parts))


# ── 수동 주문 ────────────────────────────────────────

def run_manual_order():
    """수동 주문 실행 (전략 분석 없이 즉시 매수/매도)."""
    load_dotenv()

    raw = os.getenv("MANUAL_ORDER", "").strip()
    if not raw:
        logger.error("manual_order: MANUAL_ORDER 환경변수 미설정")
        return
    try:
        params = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"manual_order: JSON 파싱 실패 - {e}")
        return

    coin = str(params.get("coin", "")).strip().upper()
    side = str(params.get("side", "")).strip().lower()
    order_type = str(params.get("order_type", "market")).strip().lower()
    pct = safe_float(params.get("pct", 0), default=0.0)
    limit_price = safe_float(params.get("price", 0), default=0.0)
    limit_volume = safe_float(params.get("volume", 0), default=0.0)

    if not coin or side not in ("buy", "sell"):
        logger.error(f"manual_order: 잘못된 파라미터 - coin={coin}, side={side}")
        return

    if order_type == "limit":
        if limit_price <= 0 or limit_volume <= 0:
            logger.error(f"manual_order: 지정가 파라미터 부족 - price={limit_price}, volume={limit_volume}")
            return
    else:
        if pct <= 0:
            logger.error(f"manual_order: 시장가 비율 미설정 - pct={pct}")
            return

    ticker = f"KRW-{coin}"
    logger.info(f"=== Manual Order: {order_type.upper()} {side.upper()} {coin} ===")

    ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
    SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
    if not ACCESS_KEY or not SECRET_KEY:
        logger.error("manual_order: UPBIT API 키 미설정")
        send_telegram("<b>수동 주문 실패</b>\nUPBIT API 키 미설정")
        return

    trader = UpbitTrader(ACCESS_KEY, SECRET_KEY)

    current_price = safe_float(trader.get_current_price(ticker), default=0.0)
    if current_price <= 0:
        logger.error(f"manual_order: 현재가 조회 실패 ({ticker})")
        send_telegram(f"<b>수동 주문 실패</b>\n{ticker} 현재가 조회 불가")
        return

    result = None
    detail = ""

    if order_type == "limit":
        total_krw = limit_price * limit_volume
        if total_krw < MIN_ORDER_KRW:
            msg = f"최소 주문금액 미달: {limit_price:,.0f} × {limit_volume:.8g} = {total_krw:,.0f}원 < {MIN_ORDER_KRW:,.0f}원"
            logger.error(f"manual_order: {msg}")
            send_telegram(f"<b>수동 지정가 실패</b>\n{ticker} {msg}")
            return
        logger.info(f"[{ticker}] 지정가 {side}: {limit_price:,.0f}원 × {limit_volume:.8g}")
        try:
            if side == "buy":
                result = trader.buy_limit(ticker, limit_price, limit_volume)
                detail = f"지정가 매수 {limit_price:,.0f} × {limit_volume:.8g}"
            else:
                result = trader.sell_limit(ticker, limit_price, limit_volume)
                detail = f"지정가 매도 {limit_price:,.0f} × {limit_volume:.8g}"
        except Exception as e:
            logger.error(f"manual_order LIMIT {side} error: {e}")
            send_telegram(f"<b>수동 지정가 오류</b>\n{ticker}: {e}")
            append_trade_log({"mode": "manual", "ticker": ticker, "side": side.upper(),
                              "order_type": "limit", "result": "error", "detail": str(e)[:200]})
            return

    elif side == "buy":
        krw = safe_float(trader.get_balance("KRW"), default=0.0)
        buy_amount = krw * (pct / 100) * 0.999
        if buy_amount < MIN_ORDER_KRW:
            msg = f"KRW 부족: {krw:,.0f}원 × {pct}% = {buy_amount:,.0f}원 < {MIN_ORDER_KRW:,.0f}원"
            logger.error(f"manual_order: {msg}")
            send_telegram(f"<b>수동 매수 실패</b>\n{ticker} {msg}")
            return
        logger.info(f"[{ticker}] 시장가 매수: {buy_amount:,.0f}원 ({pct}% of {krw:,.0f}원)")
        try:
            result = trader.adaptive_buy(ticker, buy_amount, interval="day")
            detail = f"시장가 매수 {buy_amount:,.0f}원"
        except Exception as e:
            logger.error(f"manual_order BUY error: {e}")
            send_telegram(f"<b>수동 매수 오류</b>\n{ticker}: {e}")
            append_trade_log({"mode": "manual", "ticker": ticker, "side": "BUY",
                              "pct": pct, "result": "error", "detail": str(e)[:200]})
            return

    elif side == "sell":
        coin_balance = safe_float(trader.get_balance(coin), default=0.0)
        sell_qty = coin_balance * (pct / 100)
        sell_value = sell_qty * current_price
        if sell_value < MIN_ORDER_KRW:
            msg = f"보유 부족: {coin} {coin_balance:.8g} × {pct}% = {sell_value:,.0f}원 < {MIN_ORDER_KRW:,.0f}원"
            logger.error(f"manual_order: {msg}")
            send_telegram(f"<b>수동 매도 실패</b>\n{ticker} {msg}")
            return
        logger.info(f"[{ticker}] 시장가 매도: {sell_qty:.8g} {coin} ({pct}% of {coin_balance:.8g})")
        try:
            result = trader.smart_sell(ticker, sell_qty, interval="day")
            detail = f"시장가 매도 {sell_qty:.8g} {coin} (≈{sell_value:,.0f}원)"
        except Exception as e:
            logger.error(f"manual_order SELL error: {e}")
            send_telegram(f"<b>수동 매도 오류</b>\n{ticker}: {e}")
            append_trade_log({"mode": "manual", "ticker": ticker, "side": "SELL",
                              "pct": pct, "result": "error", "detail": str(e)[:200]})
            return

    logger.info(f"[{ticker}] Manual Order Result: {result}")
    append_trade_log({
        "mode": "manual", "ticker": ticker, "side": side.upper(),
        "pct": pct, "detail": f"{detail} | {str(result)[:150]}",
        "result": "success",
    })

    # 잔고 갱신 + 캐시
    time.sleep(1)
    updated_bal = trader.get_all_balances() or {}
    prices = {}
    for sym in updated_bal:
        if sym != "KRW":
            p = safe_float(trader.get_current_price(f"KRW-{sym}"), default=0.0)
            if p > 0:
                prices[f"KRW-{sym}"] = p
    save_balance_cache(updated_bal, prices)

    krw_after = safe_float(updated_bal.get("KRW", 0))
    coin_after = safe_float(updated_bal.get(coin, 0))
    coin_value_after = coin_after * current_price
    send_telegram(
        f"<b>수동 주문 완료</b>\n"
        f"{ticker} {side.upper()} | {detail}\n"
        f"현재가: {current_price:,.0f}원\n"
        f"잔고: KRW {krw_after:,.0f}원 / {coin} {coin_after:.8g} (≈{coin_value_after:,.0f}원)"
    )


# ── 텔레그램 테스트 핑 ───────────────────────────────

def run_telegram_test_ping():
    """텔레그램 알림 경로 점검용: 경량 핑 메시지 전송."""
    load_dotenv()
    now = datetime.now(KST)
    msg = f"<b>알림 테스트</b> {now.strftime('%m-%d %H:%M')} KST - 정상"
    logger.info("텔레그램 테스트 핑 전송")
    send_telegram(msg)
