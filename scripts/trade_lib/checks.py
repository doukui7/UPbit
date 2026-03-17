"""헬스체크 / 모니터링 (가상주문 점검)."""
import os
import re
import json
import math
import time
import logging
from datetime import datetime, timedelta
from collections import OrderedDict
from dotenv import load_dotenv

from .constants import (
    KST, PROJECT_ROOT, MIN_ORDER_KRW,
    GOLD_KRX_ETF_CODE,
)
from .utils import safe_float, mask_secret, get_env_any
from .notifier import send_telegram
from .state import load_signal_state, save_signal_state, save_balance_cache
from .data import (
    get_gold_daily_local_first, get_gold_price_local_first,
    get_kis_daily_local_first, get_kis_price_local_first,
)
from .gold_engine import get_gold_portfolio
from .kis_ops import (
    is_kr_market_hours, get_kr_order_phase, is_kr_order_window,
    get_kis_balance_with_retry, extract_order_fail_msg,
    extract_kis_balance_fail_detail, is_non_actionable_kis_order_failure,
    get_kis_orderable_cash_precise,
    normalize_kis_account_fields, sanitize_isa_trade_etf, normalize_gold_kr_etf,
    format_holdings_brief,
)

import src.engine.data_cache as data_cache
from src.strategy.donchian import DonchianStrategy
from src.strategy.widaeri import WDRStrategy
from src.strategy.laa import LAAStrategy
from src.strategy.dual_momentum import DualMomentumStrategy
from src.trading.upbit_trader import UpbitTrader
from src.engine.kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG
from src.engine.kis_trader import KISTrader

logger = logging.getLogger(__name__)


# ── 유틸리티 ─────────────────────────────────────────

def _append_step(result: dict, name: str, status: str, detail: str = ""):
    steps = result.setdefault("steps", [])
    steps.append(f"[{status}] {name}: {detail}" if detail else f"[{status}] {name}")


def _is_ok_or_skip(ok, msg) -> bool:
    txt = str(msg or "").strip().upper()
    return bool(ok) or txt.startswith("SKIP")


def _latest_upbit_slot_kst(now_kst: datetime) -> datetime:
    slots = [
        now_kst.replace(hour=h, minute=0, second=0, microsecond=0)
        for h in (1, 5, 9, 13, 17, 21)
    ]
    passed = [s for s in slots if s <= now_kst]
    if passed:
        return passed[-1]
    yday = now_kst - timedelta(days=1)
    return yday.replace(hour=21, minute=0, second=0, microsecond=0)


def _parse_minute_key_kst(raw: str) -> datetime | None:
    txt = str(raw or "").strip()
    if re.fullmatch(r"\d{12}", txt):
        fmt = "%Y%m%d%H%M"
    elif re.fullmatch(r"\d{10}", txt):
        fmt = "%Y%m%d%H"
    else:
        return None
    try:
        dt = datetime.strptime(txt, fmt)
        return dt.replace(tzinfo=KST)
    except Exception:
        return None


# ── 업비트 스케줄 상태 점검 ──────────────────────────

def _check_upbit_schedule_status(now_kst: datetime | None = None) -> tuple[bool, str]:
    """VM 스케줄러 상태파일 + 로그 파일 기준 업비트 정시 실행 누락 여부 점검."""
    now = now_kst or datetime.now(KST)
    expected = _latest_upbit_slot_kst(now)
    expected_key = expected.strftime("%Y%m%d%H%M")
    expected_label = expected.strftime("%m-%d %H:%M")
    state_path = os.path.join(PROJECT_ROOT, "logs", "vm_scheduler_state.json")
    log_path = os.path.join(PROJECT_ROOT, "logs", "upbit.log")

    # 1차: 상태 파일 확인
    state_ok = False
    state_msg = ""

    if not os.path.exists(state_path):
        state_msg = "상태파일 미존재"
    else:
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                state_msg = "상태파일 형식 오류"
            else:
                last_key = str(state.get("upbit", "")).strip()
                if not last_key:
                    state_msg = "실행 기록 없음"
                else:
                    last_dt = _parse_minute_key_kst(last_key)
                    if last_dt is None:
                        state_msg = f"기록 형식 오류 ({last_key})"
                    elif last_dt < expected:
                        last_label = last_dt.strftime("%m-%d %H:%M")
                        state_msg = f"마지막 {last_label}"
                    else:
                        last_label = last_dt.strftime("%m-%d %H:%M")
                        state_ok = True
                        if last_key != expected_key:
                            state_msg = f"최근 실행 {last_label} (기준 {expected_label})"
                        else:
                            state_msg = f"정시 실행 확인 ({last_label})"
        except Exception as e:
            state_msg = f"상태파일 읽기 오류: {e}"

    if state_ok:
        return True, f"PASS - {state_msg}"

    # 2차: 로그 파일 mtime으로 보조 확인
    log_ok = False
    log_msg = ""
    if os.path.exists(log_path):
        try:
            mtime = os.path.getmtime(log_path)
            log_dt = datetime.fromtimestamp(mtime, tz=KST)
            if log_dt >= expected:
                log_ok = True
                log_msg = f"로그 갱신 {log_dt.strftime('%m-%d %H:%M')}"
        except Exception:
            pass

    if log_ok:
        return True, f"PASS - {log_msg} (상태파일: {state_msg})"

    return False, f"FAIL - 정시 누락 감지 (예상 {expected_label}, {state_msg})"


# ── 키움 금현물 헬스체크 ─────────────────────────────

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
        cash_krw = float(bal.get("cash_krw", 0.0))
        gold_qty = float(bal.get("gold_qty", 0.0))
        result['balance_msg'] = f"현금 {cash_krw:,.0f}원 / 금 {gold_qty}g"

        if cash_krw <= 0 and gold_qty <= 0:
            result['price_msg'] = 'SKIP - 비활성 계좌 (현금/보유 0)'
            result['signal_msg'] = 'SKIP - 비활성 계좌 (시그널 생략)'
            result['order_msg'] = 'SKIP - 비활성 계좌 (주문테스트 생략)'
            return result

        # 3. 시세 조회
        price = get_gold_price_local_first(trader, code)
        if not price or price <= 0:
            if not is_kr_market_hours():
                result['price_msg'] = 'SKIP - 장 시간 외 시세 조회 생략 (장 종료)'
                result['signal_msg'] = 'SKIP - 장 시간 외 시그널 체크 생략'
                result['order_msg'] = 'SKIP - 장 시간 외 주문 테스트 생략'
                return result
            result['price_msg'] = 'FAIL - 시세 조회 실패'
            return result
        result['price'] = True
        result['price_msg'] = f"미니금 현재가 {price:,.0f}원"

        # 4. 시그널 체크
        gold_portfolio = get_gold_portfolio()
        total_weight = sum(g['weight'] for g in gold_portfolio)
        max_period = max(g['buy_period'] for g in gold_portfolio)
        df = get_gold_daily_local_first(trader, code=code, count=max(max_period + 10, 200))

        if df is None or len(df) < max_period + 5:
            import pandas as pd
            csv_path = os.path.join(PROJECT_ROOT, "krx_gold_daily.csv")
            if not os.path.exists(csv_path):
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
            result['signal_msg'] = f"{action_str} (목표 {target_ratio:.0%}, 현재 {current_ratio:.0%})"
        else:
            result['signal_msg'] = 'SKIP - 데이터 부족'

        # 5. 가상주문 왕복 테스트 (하한가 매수 1g → 조회 → 취소)
        if not is_kr_order_window():
            result['order_msg'] = 'SKIP - 주문 가능 시간이 아닙니다 (09:00~15:30, 15:40~16:00 KST)'
            return result

        limit_price = trader._get_limit_price(code, "SELL")
        if limit_price <= 0:
            result['order_msg'] = 'FAIL - 하한가 계산 실패'
            return result

        order_result = trader.send_order("BUY", code, qty=1, price=limit_price, ord_tp="1")
        if not order_result or not order_result.get('success'):
            result['order_msg'] = f"FAIL - 주문 실패: {order_result}"
            return result

        ord_no = order_result.get('ord_no', '')
        time.sleep(2)

        pending = trader.get_pending_orders(code) or []
        found = any(p.get('ord_no') == ord_no for p in pending) if pending else bool(ord_no)

        cancel_result = trader.cancel_order(ord_no, code, qty=1)
        cancel_ok = bool(cancel_result and (cancel_result.get('success') if isinstance(cancel_result, dict) else cancel_result))
        time.sleep(1)

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


# ── KIS ISA 헬스체크 ─────────────────────────────────

def _check_kis_isa() -> dict:
    """KIS ISA 위대리 시스템 헬스체크."""
    import pandas as pd

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
        isa_key = get_env_any("KIS_ISA_APP_KEY", "KIS_APP_KEY")
        isa_secret = get_env_any("KIS_ISA_APP_SECRET", "KIS_APP_SECRET")
        isa_acct_raw = get_env_any("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO")
        isa_prdt_raw = get_env_any("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
        isa_account, isa_prdt = normalize_kis_account_fields(isa_acct_raw, isa_prdt_raw)

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

        etf_code_raw = os.getenv("KIS_ISA_ETF_CODE", "418660")
        etf_code = sanitize_isa_trade_etf(etf_code_raw, default="418660")
        signal_etf_code = os.getenv("KIS_ISA_TREND_ETF_CODE", "133690")
        if str(etf_code_raw).strip() != etf_code:
            _append_step(result, "ETF 코드 보정", "INFO", f"1배/미지원 코드({etf_code_raw}) -> {etf_code}")

        # 2. 잔고 조회
        bal = get_kis_balance_with_retry(trader, retries=3, delay_sec=0.8, log_prefix="KIS ISA")
        if bal is None:
            probe = trader.get_balance()
            detail = extract_kis_balance_fail_detail(probe)
            result['balance_msg'] = f"FAIL - 잔고 조회 실패 ({detail})" if detail else 'FAIL - 잔고 조회 실패'
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
        price = get_kis_price_local_first(trader, etf_code)
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

        signal_df = get_kis_daily_local_first(trader, signal_etf_code, count=1500)
        if signal_df is None or len(signal_df) < 260 * 5:
            from src.engine.data_cache import load_bundled_csv
            signal_df = load_bundled_csv(signal_etf_code)

        if signal_df is not None and len(signal_df) >= 260 * 5:
            sig = strategy.analyze(signal_df)
            if sig:
                bt_msg = ""
                isa_start = os.getenv("KIS_ISA_START_DATE", "2022-03-08")
                trade_df = get_kis_daily_local_first(trader, etf_code, count=1500)
                if trade_df is not None and len(trade_df) >= 60:
                    _eff_start = isa_start
                    _tfd = str(trade_df.index[0].date())
                    if isa_start < _tfd:
                        _eff_start = _tfd

                    _ref_sr = None
                    _ref_info = data_cache.get_wdr_v10_stock_ratio(str(etf_code), _eff_start)
                    if _ref_info:
                        _ref_sr = float(_ref_info.get("stock_ratio", 0.0))
                    elif _eff_start > "2022-03-08":
                        _ref_bt = strategy.run_backtest(
                            signal_daily_df=signal_df,
                            trade_daily_df=trade_df,
                            initial_balance=10_000_000,
                            start_date="2022-03-08",
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

                is_friday = datetime.now(KST).weekday() == 4
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

        # 5. 가상주문 왕복 테스트
        order_phase = get_kr_order_phase()
        if order_phase == "closed":
            result['order_msg'] = 'SKIP - 주문 가능 시간이 아닙니다 (09:00~15:30, 15:40~16:00 KST)'
            return result

        if order_phase == "after_hours":
            order_result = trader.send_order("BUY", etf_code, qty=1, price=0, ord_dvsn="06")
        else:
            limit_price = trader._get_limit_price(etf_code, "SELL")
            if limit_price <= 0:
                result['order_msg'] = 'SKIP - 하한가 계산 실패 (주문테스트 생략)'
                return result
            order_result = trader.send_order("BUY", etf_code, qty=1, price=limit_price, ord_dvsn="00")
        if not order_result:
            result['order_msg'] = 'FAIL - 주문 실패: API 응답 없음(None)'
            return result
        if not isinstance(order_result, dict):
            result['order_msg'] = f"FAIL - 주문 실패: 응답 형식 오류({type(order_result).__name__})"
            return result
        if not order_result.get('success'):
            fail_msg = extract_order_fail_msg(order_result)
            if is_non_actionable_kis_order_failure(fail_msg):
                result['order_msg'] = f"SKIP - 주문 제약: {fail_msg}"
            else:
                result['order_msg'] = f"FAIL - 주문 실패: {fail_msg or order_result}"
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


# ── 업비트 헬스체크 ──────────────────────────────────

def _check_upbit() -> dict:
    """업비트 코인 시스템 헬스체크."""
    # 순환 import 방지: upbit_engine은 여기서만 사용
    from .upbit_engine import get_portfolio, analyze_asset
    from .utils import normalize_coin_interval

    result = {
        'name': '업비트 코인',
        'auth': False, 'auth_msg': '',
        'balance': False, 'balance_msg': '',
        'price': False, 'price_msg': '',
        'signal': False, 'signal_msg': '',
        'order_test': False, 'order_msg': '',
        'schedule_ok': False, 'schedule_msg': '',
        'sync': False, 'sync_msg': '',
        'recovery_run': False, 'recovery_msg': '',
        'steps': [],
    }
    sync_balances = None
    sync_prices = None

    try:
        schedule_ok, schedule_msg = _check_upbit_schedule_status()
        result["schedule_ok"] = schedule_ok
        result["schedule_msg"] = schedule_msg
        _append_step(
            result, "0. 정시 누락 점검",
            "PASS" if _is_ok_or_skip(schedule_ok, schedule_msg) else "FAIL",
            schedule_msg,
        )

        ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY")
        SECRET_KEY = os.getenv("UPBIT_SECRET_KEY")
        _append_step(
            result, "0.1 입력값 확인", "INFO",
            f"UPBIT_ACCESS_KEY={mask_secret(ACCESS_KEY, 5, 3)}, "
            f"UPBIT_SECRET_KEY={mask_secret(SECRET_KEY, 5, 3)}"
        )
        if not ACCESS_KEY or not SECRET_KEY:
            result['auth_msg'] = 'FAIL - API 키 미설정'
            _append_step(result, "1. 인증", "FAIL", result['auth_msg'])
            return result

        trader = UpbitTrader(ACCESS_KEY, SECRET_KEY)
        _append_step(result, "1. 클라이언트 생성", "PASS", "UpbitTrader 인스턴스 생성 완료")

        # 1. 인증 + 2. 잔고 조회
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

        krw = safe_float(all_bal.get('KRW', 0))
        sync_balances = all_bal
        result['auth'] = True
        result['auth_msg'] = 'PASS'
        _currencies = sorted([str(k) for k in all_bal.keys()])
        _append_step(
            result, "2. 인증/잔고 API", "PASS",
            f"통화 {_currencies[:8]}{'...' if len(_currencies) > 8 else ''} (총 {len(_currencies)}개)"
        )

        result['balance'] = True
        coins = {}
        for k, v in all_bal.items():
            if str(k) == 'KRW':
                continue
            fv = safe_float(v, 0.0)
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
        prices = {k: safe_float(v, 0.0) for k, v in prices_raw.items() if safe_float(v, 0.0) > 0}

        if prices:
            result['price'] = True
            result['price_msg'] = ", ".join(f"{t}={p:,.0f}" for t, p in prices.items())
            _append_step(result, "4. 시세 조회", "PASS", f"요청 {len(tickers)}개 / 성공 {len(prices)}개")
            sync_prices = prices
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
                iv = normalize_coin_interval(item.get("interval", "day"))
                if signal_value in {"BUY", "SELL", "HOLD"}:
                    signals.append(f"{ticker}={signal_value}")
                    signal_details.append(f"{ticker}:{signal_value}({iv})")
                else:
                    signal_errors.append(f"{ticker}=ERROR({a.get('signal')})")
            except Exception as e:
                signal_errors.append(f"{_ticker}=ERROR({e})")

        # 같은 코인 시그널 통합
        _coin_sigs = OrderedDict()
        for sd in signal_details:
            _coin_key = sd.split(":")[0]
            _sig_val = sd.split(":")[1].split("(")[0] if ":" in sd else ""
            _coin_sigs.setdefault(_coin_key, []).append(_sig_val)
        unified_signals = []
        unified_details = []
        for _ck, _svs in _coin_sigs.items():
            _unified = "BUY" if all(s == "BUY" for s in _svs) else "SELL"
            unified_signals.append(f"{_ck}={_unified}")
            unified_details.append(f"{_ck}:{_unified}")
        if signals and not signal_errors:
            result['signal'] = True
            result['signal_msg'] = ", ".join(unified_signals)
            _append_step(result, "5. 시그널 분석", "PASS", f"{len(unified_signals)}개 코인: {' | '.join(unified_details)}")
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

        # 5. 가상주문 왕복 테스트
        if krw < MIN_ORDER_KRW:
            result['order_msg'] = f"SKIP - 가상주문 최소금액 부족 (KRW {krw:,.0f}원)"
            _append_step(result, "6. 가상주문", "SKIP", result['order_msg'])
            return result

        test_ticker = tickers[0]
        test_price = safe_float(prices.get(test_ticker), 0.0)
        if test_price <= 0:
            result['order_msg'] = 'SKIP - 테스트 종목 시세 없음'
            _append_step(result, "6. 가상주문", "SKIP", f"{result['order_msg']} (ticker={test_ticker})")
            return result

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

        min_volume = max(5000 / dummy_price, 0.0001) if dummy_price > 0 else 0.001
        min_volume = math.ceil(min_volume * 1e8) / 1e8
        _append_step(
            result, "6.1 가상주문 입력값", "INFO",
            f"ticker={test_ticker}, 현재가={test_price:,.0f}, 테스트가격={dummy_price:,.0f}, 수량={min_volume}"
        )

        order = trader.buy_limit(test_ticker, dummy_price, min_volume)
        if not order:
            result['order_msg'] = "FAIL - 주문 실패: 응답 없음 (API returned None)"
            _append_step(result, "6.2 가상주문 접수", "FAIL", result['order_msg'])
            return result
        if isinstance(order, dict) and 'error' in order:
            _err_detail = order.get('error', str(order))
            result['order_msg'] = f"FAIL - 주문 실패: {_err_detail}"
            _append_step(result, "6.2 가상주문 접수", "FAIL", result['order_msg'])
            return result
        if not isinstance(order, dict):
            result['order_msg'] = f"FAIL - 주문 응답 형식 오류: {type(order).__name__}={str(order)[:200]}"
            _append_step(result, "6.2 가상주문 접수", "FAIL", result['order_msg'])
            return result

        order_uuid = order.get('uuid', '')
        if not order_uuid:
            result['order_msg'] = f"FAIL - 주문 UUID 없음: {order}"
            _append_step(result, "6.2 가상주문 접수", "FAIL", result['order_msg'])
            return result
        _append_step(result, "6.2 가상주문 접수", "PASS", f"uuid={order_uuid}")
        time.sleep(2)

        pending = trader.get_orders(test_ticker, state='wait')
        found = False
        if pending:
            found = any(o.get('uuid') == order_uuid for o in pending)
        _append_step(result, "6.3 미체결 조회", "PASS" if found else "FAIL",
                     f"found={found}, pending_count={len(pending) if pending else 0}")

        cancel = trader.cancel_order(order_uuid)
        cancel_ok = bool(cancel) and ('error' not in str(cancel).lower())
        _append_step(result, "6.4 주문 취소", "PASS" if cancel_ok else "FAIL", f"cancel={cancel}")
        time.sleep(1)

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
    finally:
        try:
            _state = load_signal_state()
            if not isinstance(_state, dict):
                _state = {}
            save_signal_state(_state)
            if isinstance(sync_balances, dict) and sync_balances:
                save_balance_cache(sync_balances, sync_prices if isinstance(sync_prices, dict) else None)
                if isinstance(sync_prices, dict) and sync_prices:
                    result['sync_msg'] = 'balance_cache + signal_state 동기화 완료'
                else:
                    result['sync_msg'] = 'balance_cache(잔고) + signal_state 동기화 완료'
            else:
                result['sync_msg'] = 'signal_state 동기화 완료'
            result['sync'] = True
            _append_step(result, "7. 상태 동기화", "PASS", result['sync_msg'])
        except Exception as e:
            result['sync_msg'] = f"FAIL - 동기화 오류: {e}"
            _append_step(result, "7. 상태 동기화", "FAIL", result['sync_msg'])

    return result


# ── 연금저축 예약주문 요약 ────────────────────────────

def _get_pension_order_summary() -> tuple[str, str]:
    """오늘 예약주문 요약: (대기 요약, 결과 요약)."""
    orders_path = os.path.join(PROJECT_ROOT, "config", "pension_orders.json")
    if not os.path.exists(orders_path):
        return ("", "")
    try:
        with open(orders_path, "r", encoding="utf-8") as f:
            orders = json.load(f)
    except Exception:
        return ("", "")
    if not isinstance(orders, list):
        return ("", "")

    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    pending_parts = []
    result_parts = []
    for o in orders:
        sched = str(o.get("scheduled_kst", ""))
        status = o.get("status", "")
        side = o.get("side", "")
        code = o.get("etf_code", "")
        name = o.get("etf_name", code)
        qty = o.get("qty", 0)
        method = o.get("method", "").split("(")[0].strip()
        time_part = sched[11:16] if len(sched) >= 16 else ""
        label = f"{side} {name}({code}) x{qty}"

        # 대기 주문: 오늘 또는 미래 예정 모두 표시
        if status == "대기":
            sched_date = sched[:10] if len(sched) >= 10 else ""
            if sched_date == today_str:
                pending_parts.append(f"{label} {method} {time_part}")
            else:
                pending_parts.append(f"{label} {method} ({sched_date} {time_part})")
            continue

        # 완료/실패: 오늘 실행된 것만
        if not sched.startswith(today_str):
            continue
        if status == "실패":
            raw = str(o.get("result", ""))
            import re as _re
            m = _re.search(r"'msg':\s*'([^']{1,30})", raw)
            reason = m.group(1) if m else raw[:30]
            result_parts.append(f"{label} -> 실패 ({reason})")
        elif status == "완료":
            result_parts.append(f"{label} -> 완료")

    pending_str = ""
    if pending_parts:
        pending_str = f"오늘 예약주문 {len(pending_parts)}건: " + ", ".join(pending_parts)
    result_str = ""
    if result_parts:
        result_str = "오늘 주문 결과: " + " | ".join(result_parts)
    return (pending_str, result_str)


# ── KIS 연금저축 헬스체크 ────────────────────────────

def _check_kis_pension() -> dict:
    """KIS 연금저축 시스템 헬스체크 (활성 전략 전체)."""
    # user_config에서 활성 전략 목록 로드
    _ucfg_path = os.path.join(PROJECT_ROOT, "user_config.json")
    _pen_portfolio = []
    _ucfg = {}
    try:
        with open(_ucfg_path, "r", encoding="utf-8") as f:
            _ucfg = json.load(f)
        _pen_portfolio = _ucfg.get("pension_portfolio", [])
    except Exception:
        pass
    enabled_strategies = [
        p["strategy"] for p in _pen_portfolio
        if float(p.get("weight", 0)) > 0
    ] or ["LAA"]
    strategy_label = "+".join(enabled_strategies)

    result = {
        "name": f"KIS 연금저축 {strategy_label}",
        "auth": False, "auth_msg": "",
        "balance": False, "balance_msg": "",
        "price": False, "price_msg": "",
        "signal": False, "signal_msg": "",
        "order_test": False, "order_msg": "",
        "pending_orders": "", "order_results": "",
    }
    # 예약주문 요약 (인증 불필요 — 로컬 파일)
    try:
        _pend, _ores = _get_pension_order_summary()
        result["pending_orders"] = _pend
        result["order_results"] = _ores
    except Exception:
        pass
    try:
        trader = KISTrader(is_mock=False)
        pension_key = get_env_any("KIS_PENSION_APP_KEY", "KIS_APP_KEY")
        pension_secret = get_env_any("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET")
        pension_acct_raw = get_env_any("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO")
        pension_prdt_raw = get_env_any("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
        pension_acct, pension_prdt = normalize_kis_account_fields(pension_acct_raw, pension_prdt_raw)
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

        bal = get_kis_balance_with_retry(trader, retries=3, delay_sec=0.8, log_prefix="KIS 연금저축")
        if bal is None:
            probe = trader.get_balance()
            detail = extract_kis_balance_fail_detail(probe)
            result["balance_msg"] = f"FAIL - 잔고 조회 실패 ({detail})" if detail else "FAIL - 잔고 조회 실패"
            return result
        result["balance"] = True
        pension_base_code = get_env_any("KR_ETF_LAA_QQQ", default="133690")
        cash = get_kis_orderable_cash_precise(
            trader, bal, base_code=pension_base_code, fallback=0.0
        )
        holdings = bal.get("holdings", []) or []
        stock_eval = float(bal.get("stock_eval", 0.0)) or sum(
            float(h.get("eval_amt", 0.0) or 0.0) for h in holdings
        )
        total_eval = cash + stock_eval
        holding_brief = format_holdings_brief(holdings)
        result["balance_msg"] = (
            f"예수금(주문가능) {cash:,.0f} / 주식평가 {stock_eval:,.0f} / 총평가 {total_eval:,.0f}"
            f" / 보유 {holding_brief}"
        )

        # ── LAA용 시세 데이터 (LAA 활성 시) ──
        laa_tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
        laa_kr_etf_map = {
            "SPY": get_env_any("KR_ETF_LAA_SPY", "KR_ETF_SPY", default="360750"),
            "IWD": get_env_any("KR_ETF_LAA_IWD", "KR_ETF_SPY", default="360750"),
            "GLD": normalize_gold_kr_etf(get_env_any("KR_ETF_LAA_GLD", "KR_ETF_GOLD", default=GOLD_KRX_ETF_CODE)),
            "IEF": get_env_any("KR_ETF_LAA_IEF", "KR_ETF_AGG", default="305080"),
            "QQQ": get_env_any("KR_ETF_LAA_QQQ", default="133690"),
            "SHY": get_env_any("KR_ETF_LAA_SHY", default="329750"),
        }
        # ── 듀얼모멘텀용 시세 데이터 (듀얼모멘텀 활성 시) ──
        dm_tickers = ["SPY", "EFA", "AGG", "BIL"]
        dm_kr_etf_map = {
            "SPY": _ucfg.get("pen_dm_kr_spy", "360750"),
            "EFA": _ucfg.get("pen_dm_kr_efa", "251350"),
            "AGG": _ucfg.get("pen_dm_kr_agg", "305080"),
            "BIL": _ucfg.get("pen_dm_kr_bil", "329750"),
        }

        # 필요한 코드 통합 조회 (중복 제거)
        all_codes = {}  # {ticker: code}
        if "LAA" in enabled_strategies:
            for t in laa_tickers:
                all_codes[f"LAA_{t}"] = str(laa_kr_etf_map.get(t, "")).strip()
        if "듀얼모멘텀" in enabled_strategies:
            for t in dm_tickers:
                all_codes[f"DM_{t}"] = str(dm_kr_etf_map.get(t, "")).strip()

        fetched = {}  # {code: df} — 중복 코드 1회만 조회
        price_data_laa = {}
        price_data_dm = {}
        for key, code in all_codes.items():
            if not code:
                result["price_msg"] = f"FAIL - {key} 국내 ETF 코드 미설정"
                return result
            if code not in fetched:
                df = get_kis_daily_local_first(trader, code, count=320)
                if df is None or len(df) == 0:
                    result["price_msg"] = f"FAIL - {key}({code}) 국내 데이터 조회 실패"
                    return result
                fetched[code] = df
                time.sleep(0.1)
            if key.startswith("LAA_"):
                price_data_laa[key[4:]] = fetched[code]
            elif key.startswith("DM_"):
                price_data_dm[key[3:]] = fetched[code]

        result["price"] = True
        fetched_codes = list(dict.fromkeys(all_codes.values()))
        result["price_msg"] = f"PASS - 국내 ETF {len(fetched_codes)}종목"

        # ── 전략별 시그널 분석 ──
        signal_parts = []
        signal_ok = True

        if "LAA" in enabled_strategies:
            laa_strategy = LAAStrategy(settings={"kr_etf_map": laa_kr_etf_map})
            laa_sig = laa_strategy.analyze(price_data_laa)
            if laa_sig:
                laa_mode = "공격" if laa_sig.get("risk_on") else "방어"
                laa_target = f"{laa_sig.get('selected_risk_asset')}->{laa_sig.get('selected_risk_kr_code')}"
                signal_parts.append(f"LAA: {laa_mode} | {laa_target}")
            else:
                signal_parts.append("LAA: 분석실패")
                signal_ok = False

        if "듀얼모멘텀" in enabled_strategies:
            dm_settings = {
                "kr_etf_map": dm_kr_etf_map,
                "offensive": _ucfg.get("pen_dm_offensive", "SPY,EFA").split(","),
                "defensive": [_ucfg.get("pen_dm_defensive", "AGG")],
                "canary": [_ucfg.get("pen_dm_canary", "BIL")],
                "lookback": int(_ucfg.get("pen_dm_lookback", 12)),
                "trading_days_per_month": int(_ucfg.get("pen_dm_trading_days", 22)),
                "momentum_weights": {
                    "m1": float(_ucfg.get("pen_dm_w1", 12.0)),
                    "m3": float(_ucfg.get("pen_dm_w3", 4.0)),
                    "m6": float(_ucfg.get("pen_dm_w6", 2.0)),
                    "m12": float(_ucfg.get("pen_dm_w12", 1.0)),
                },
            }
            dm_strategy = DualMomentumStrategy(settings=dm_settings)
            dm_sig = dm_strategy.analyze(price_data_dm)
            if dm_sig:
                dm_mode = "공격" if dm_sig.get("is_offensive") else "방어"
                dm_target = f"{dm_sig.get('target_ticker')}->{dm_sig.get('target_kr_code')}"
                signal_parts.append(f"듀얼모멘텀: {dm_mode} | {dm_target}")
            else:
                signal_parts.append("듀얼모멘텀: 분석실패")
                signal_ok = False

        if not signal_parts:
            result["signal_msg"] = "FAIL - 활성 전략 없음"
            return result
        result["signal"] = signal_ok
        result["signal_msg"] = " / ".join(signal_parts)

        # 주문 테스트
        order_phase = get_kr_order_phase()
        if order_phase == "closed":
            result["order_msg"] = "SKIP - 주문 가능 시간이 아닙니다 (09:00~15:30, 15:40~16:00 KST)"
            return result

        default_bond_code = str(laa_kr_etf_map.get("SHY", "329750")).strip() or "329750"
        test_code = str(get_env_any("KIS_PENSION_TEST_ORDER_CODE", default=default_bond_code)).strip()
        if not test_code:
            result["order_msg"] = "FAIL - 테스트 종목 없음"
            return result

        if order_phase == "after_hours":
            order_result = trader.send_order("BUY", test_code, qty=1, price=0, ord_dvsn="06")
        else:
            limit_price = trader._get_limit_price(test_code, "SELL")
            if limit_price <= 0:
                result["order_msg"] = "SKIP - 가격호가 계산 실패 (주문테스트 생략)"
                return result
            order_result = trader.send_order("BUY", test_code, qty=1, price=limit_price, ord_dvsn="00")
        if not order_result:
            result["order_msg"] = "FAIL - 주문 실패: API 응답 없음(None)"
            return result
        if not isinstance(order_result, dict):
            result["order_msg"] = f"FAIL - 주문 실패: 응답 형식 오류({type(order_result).__name__})"
            return result
        if not order_result.get("success"):
            fail_msg = extract_order_fail_msg(order_result)
            if is_non_actionable_kis_order_failure(fail_msg):
                result["order_msg"] = f"SKIP - 주문 제약: {fail_msg}"
            else:
                result["order_msg"] = f"FAIL - 주문 실패: {fail_msg or order_result}"
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
            result["order_msg"] = f"PASS - 주문/취소 정상 ({test_code})"
        elif found and cancel_ok:
            result["order_test"] = True
            result["order_msg"] = f"PASS - 취소 전파 지연 가능 ({test_code})"
        else:
            result["order_msg"] = f"FAIL - 주문={bool(ord_no)}, 조회={found}, 취소={cancel_ok} ({test_code})"

    except Exception as e:
        result["order_msg"] = result.get("order_msg") or f"ERROR: {e}"
        logger.error(f"KIS 연금저축 LAA 헬스체크 오류: {e}")
    return result


# ── 헬스체크 리포트 ──────────────────────────────────

def _print_health_report(results: dict):
    """헬스체크 종합 리포트 출력 + 텔레그램 전송."""
    now = datetime.now(KST)

    msg_key_map = {
        'auth': 'auth_msg', 'balance': 'balance_msg',
        'price': 'price_msg', 'signal': 'signal_msg',
        'order_test': 'order_msg',
    }
    checks = ['auth', 'balance', 'price', 'signal', 'order_test']

    # ── 콘솔 로그: 상세 출력 ──
    for key, r in results.items():
        name = r['name']
        logger.info(f"=== [{name}] 헬스체크 ===")
        for c in checks:
            logger.info(f"  {c}: {r.get(c, False)} - {r.get(msg_key_map[c], '')}")
        if key == "upbit":
            logger.info(f"  schedule: {r.get('schedule_ok', False)} - {r.get('schedule_msg', '')}")
            logger.info(f"  sync: {r.get('sync', False)} - {r.get('sync_msg', '')}")
            logger.info(f"  recovery: {r.get('recovery_run', False)} - {r.get('recovery_msg', '')}")
        for s in (r.get('steps') or []):
            logger.info(f"  {s}")

    # ── 텔레그램: 간소화 ──
    lines = []
    lines.append(f"<b>헬스체크</b> ({now.strftime('%m-%d %H:%M')} KST)")
    lines.append("")

    pass_count = 0
    total_count = len(results)

    for key, r in results.items():
        name = r['name']
        all_pass = all(_is_ok_or_skip(r.get(c, False), r.get(msg_key_map[c], '')) for c in checks)
        schedule_msg = str(r.get("schedule_msg", "")).strip()
        schedule_ok = _is_ok_or_skip(r.get("schedule_ok", False), schedule_msg) if schedule_msg else True
        if not schedule_ok:
            all_pass = False
        if key == "upbit":
            sync_msg = str(r.get("sync_msg", "")).strip()
            if sync_msg and not _is_ok_or_skip(r.get("sync", False), sync_msg):
                all_pass = False
            recovery_msg = str(r.get("recovery_msg", "")).strip()
            if recovery_msg.upper().startswith("FAIL"):
                all_pass = False
        if all_pass:
            pass_count += 1

        status = "PASS" if all_pass else "WARN"
        lines.append(f"<b>[{name}]</b> {status}")

        bal_msg = r.get('balance_msg', '')
        if bal_msg and not bal_msg.upper().startswith(('FAIL', 'SKIP')):
            lines.append(f"  {bal_msg}")

        sig_msg = r.get('signal_msg', '')
        if sig_msg and not sig_msg.upper().startswith('SKIP'):
            if " / " in sig_msg:
                for _sm in sig_msg.split(" / "):
                    lines.append(f"  시그널: {_sm}")
            else:
                lines.append(f"  시그널: {sig_msg}")
        # 연금저축 예약주문/결과 표시
        if key == "kis_pension":
            _pend_msg = r.get("pending_orders", "")
            if _pend_msg:
                lines.append(f"  {_pend_msg}")
            _ores_msg = r.get("order_results", "")
            if _ores_msg:
                lines.append(f"  {_ores_msg}")
        if key == "upbit" and schedule_msg:
            lines.append(f"  누락여부: {schedule_msg}")
            _sync_msg = str(r.get("sync_msg", "")).strip()
            if _sync_msg:
                lines.append(f"  동기화: {_sync_msg}")
            _recovery_msg = str(r.get("recovery_msg", "")).strip()
            if _recovery_msg:
                lines.append(f"  복구주문: {_recovery_msg}")

        check_labels = {'auth': '인증', 'balance': '잔고', 'price': '시세',
                        'signal': '시그널', 'order_test': '주문테스트'}
        for c in checks:
            ok = r.get(c, False)
            msg = r.get(msg_key_map[c], '')
            if not _is_ok_or_skip(ok, msg):
                lines.append(f"  FAIL {check_labels[c]}: {msg}")
        if key == "upbit" and schedule_msg and not schedule_ok:
            lines.append(f"  FAIL 누락: {schedule_msg}")
        if key == "upbit":
            _sync_msg = str(r.get("sync_msg", "")).strip()
            if _sync_msg and not _is_ok_or_skip(r.get("sync", False), _sync_msg):
                lines.append(f"  FAIL 동기화: {_sync_msg}")
            _recovery_msg = str(r.get("recovery_msg", "")).strip()
            if _recovery_msg.upper().startswith("FAIL"):
                lines.append(f"  FAIL 복구주문: {_recovery_msg}")

        lines.append("")

    summary = "정상" if pass_count == total_count else "점검 필요"
    lines.append(f"<b>종합: {pass_count}/{total_count} {summary}</b>")

    report = "\n".join(lines)
    send_telegram(report)


# ── 전체 헬스체크 실행 ───────────────────────────────

def run_health_check():
    """전체 헬스체크 실행 (매 4시간 - 업비트 + 국내 시스템)."""
    load_dotenv()
    logger.info("=== 전체 헬스체크 시작 ===")

    results = {}

    # 업비트 (24시간)
    if os.getenv("UPBIT_ACCESS_KEY"):
        upbit_result = _check_upbit()
        results['upbit'] = upbit_result

        # 정시 주문 누락/확인불가 시 헬스체크 경로에서 복구 실행
        _recovery_env = str(os.getenv("HEALTHCHECK_UPBIT_RECOVERY", "1")).strip().lower()
        recovery_enabled = _recovery_env not in {"0", "false", "no", "off"}
        schedule_msg = str(upbit_result.get("schedule_msg", "")).strip()
        schedule_ok = _is_ok_or_skip(upbit_result.get("schedule_ok", False), schedule_msg)
        schedule_unknown = schedule_msg.upper().startswith("SKIP")
        need_recovery = (not schedule_ok) or schedule_unknown

        if recovery_enabled and need_recovery:
            try:
                from .upbit_engine import run_auto_trade
                logger.warning(f"[업비트] 누락/미확인 감지로 복구 주문 실행: {schedule_msg}")
                run_auto_trade()
                upbit_result["recovery_run"] = True
                upbit_result["recovery_msg"] = (
                    f"복구 실행 완료 ({'미확인' if schedule_unknown else '누락 감지'}: {schedule_msg})"
                )
            except Exception as e:
                upbit_result["recovery_run"] = False
                upbit_result["recovery_msg"] = f"FAIL - 복구 주문 실행 오류: {e}"
                logger.exception(f"[업비트] 헬스체크 복구 주문 오류: {e}")
        else:
            upbit_result["recovery_run"] = False
            if not recovery_enabled:
                upbit_result["recovery_msg"] = "복구 기능 비활성(HEALTHCHECK_UPBIT_RECOVERY=0)"
            else:
                upbit_result["recovery_msg"] = f"복구 실행 생략 (정상: {schedule_msg})"
        logger.info("--- 업비트 점검 완료 ---")

    # 키움 금현물
    kiwoom_key = os.getenv("Kiwoom_App_Key") or os.getenv("KIWOOM_APP_KEY")
    if kiwoom_key:
        results['kiwoom_gold'] = _check_kiwoom_gold()
        logger.info("--- 키움 금현물 점검 완료 ---")
    else:
        logger.info("[키움 금현물] 감지 안됨 (Kiwoom_App_Key/KIWOOM_APP_KEY 없음)")

    # KIS ISA
    if os.getenv("KIS_ISA_APP_KEY") or os.getenv("KIS_ISA_ACCOUNT_NO"):
        results['kis_isa'] = _check_kis_isa()
        logger.info("--- KIS ISA 점검 완료 ---")

    # KIS 연금저축
    if os.getenv("KIS_PENSION_ACCOUNT_NO"):
        results['kis_pension'] = _check_kis_pension()
        logger.info("--- KIS 연금저축 점검 완료 ---")

    if results:
        _print_health_report(results)
    else:
        logger.info("점검 대상 없음 (API 키 미설정).")

    logger.info("=== 전체 헬스체크 완료 ===")
