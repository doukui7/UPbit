"""KIS ISA 위대리(WDR) 자동매매 엔진."""
import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from .utils import get_env_any
from .notifier import send_telegram
from .data import get_kis_daily_local_first, get_kis_price_local_first
from .kis_ops import (
    get_kr_order_phase, kr_order_phase_label,
    get_kis_balance_with_retry,
    execute_kis_sell_qty_by_phase, execute_kis_buy_qty_by_phase,
    normalize_kis_account_fields, sanitize_isa_trade_etf,
)

import src.engine.data_cache as data_cache
from src.strategy.widaeri import WDRStrategy
from src.engine.kis_trader import KISTrader

logger = logging.getLogger(__name__)


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

    order_phase = get_kr_order_phase()
    if order_phase == "closed":
        logger.info("장 외 시간이라 주문을 건너뜁니다 (09:00~15:30, 16:00~18:00 KST).")
        return
    logger.info(f"현재 주문 구간: {kr_order_phase_label(order_phase)} ({order_phase})")

    isa_key = get_env_any("KIS_ISA_APP_KEY", "KIS_APP_KEY")
    isa_secret = get_env_any("KIS_ISA_APP_SECRET", "KIS_APP_SECRET")
    isa_account_raw = get_env_any("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO")
    isa_prdt_raw = get_env_any("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD", default="01")
    isa_account, isa_prdt = normalize_kis_account_fields(isa_account_raw, isa_prdt_raw)

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

    etf_code_raw = os.getenv("KIS_ISA_ETF_CODE", "418660")
    etf_code = sanitize_isa_trade_etf(etf_code_raw, default="418660")
    if str(etf_code_raw).strip() != etf_code:
        logger.warning(f"KIS_ISA_ETF_CODE={etf_code_raw} 는 1배/미지원 코드라서 {etf_code} 로 보정합니다.")
    signal_etf_code = os.getenv("KIS_ISA_TREND_ETF_CODE", "133690")

    # WDR 전략 설정
    wdr_settings = {}
    if os.getenv("WDR_OVERVALUE_THRESHOLD"):
        wdr_settings['overvalue_threshold'] = float(os.getenv("WDR_OVERVALUE_THRESHOLD"))
    if os.getenv("WDR_UNDERVALUE_THRESHOLD"):
        wdr_settings['undervalue_threshold'] = float(os.getenv("WDR_UNDERVALUE_THRESHOLD"))
    wdr_eval_mode = int(os.getenv("WDR_EVAL_MODE", "3"))

    strategy = WDRStrategy(wdr_settings if wdr_settings else None, evaluation_mode=wdr_eval_mode)

    # ── 1. 시그널 소스(TREND ETF) 일봉 데이터 조회 ──
    logger.info(f"시그널 소스 ETF({signal_etf_code}) 일봉 데이터 조회 중...")
    signal_df = get_kis_daily_local_first(trader, signal_etf_code, count=1500)
    if signal_df is None or len(signal_df) < 260 * 5:
        logger.warning("API 데이터 부족 → 번들 CSV fallback 시도...")
        from src.engine.data_cache import load_bundled_csv
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
    bal = get_kis_balance_with_retry(trader, retries=3, delay_sec=0.8, log_prefix="KIS ISA")
    if bal is None:
        logger.error("잔고 조회 실패. 종료.")
        return

    cash = bal['cash']
    holdings = bal['holdings']

    etf_holding = None
    for h in holdings:
        if h['code'] == etf_code:
            etf_holding = h
            break

    current_shares = etf_holding['qty'] if etf_holding else 0
    current_price = get_kis_price_local_first(trader, etf_code) or 0

    if current_price <= 0:
        logger.error(f"ETF({etf_code}) 현재가 조회 실패. 종료.")
        return

    etf_value = current_shares * current_price
    total_value = cash + etf_value

    logger.info(f"잔고: 예수금={cash:,.0f}원, ETF={current_shares}주 "
                f"(≈{etf_value:,.0f}원), 총자산={total_value:,.0f}원")

    # ── 4. 백테스트 기반 목표 비율 계산 (비중 기반 매매) ──
    isa_start_date = os.getenv("KIS_ISA_START_DATE", "2022-03-08")
    trade_df = get_kis_daily_local_first(trader, etf_code, count=1500)

    bt_stock_ratio = None
    order_action = None
    order_qty = 0

    if trade_df is not None and len(trade_df) >= 60:
        _trade_first_date = str(trade_df.index[0].date())
        _effective_start = isa_start_date
        if _effective_start < _trade_first_date:
            _effective_start = _trade_first_date

        # 초기 비중 참조: 티커별 시작 기준값 우선
        _ref_stock_ratio = None
        _ref_info = data_cache.get_wdr_v10_stock_ratio(str(etf_code), _effective_start)
        if _ref_info:
            _ref_stock_ratio = float(_ref_info.get("stock_ratio", 0.0))
            logger.info(
                f"티커 시작 기준값 비율: {_ref_stock_ratio*100:.1f}% "
                f"({etf_code}, {_ref_info.get('ref_date')})"
            )
        elif _effective_start > "2022-03-08" and signal_df is not None:
            _ref_bt = strategy.run_backtest(
                signal_daily_df=signal_df,
                trade_daily_df=trade_df,
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
                        logger.info(
                            f"참조 백테스트 비율: {_ref_stock_ratio*100:.1f}% "
                            f"({signal_etf_code}→{etf_code}, {_effective_start} 시점)"
                        )

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
        etf_chart = get_kis_daily_local_first(trader, etf_code, count=10)
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
        logger.info(
            f"[SELL] {etf_code} {order_qty}주 "
            f"(≈ {sell_value:,.0f}원, 구간={kr_order_phase_label(order_phase)})"
        )
        result = execute_kis_sell_qty_by_phase(trader, etf_code, order_qty, order_phase)
        logger.info(f"매도 결과: {result}")

    elif order_action == 'BUY' and order_qty > 0:
        buy_value = order_qty * current_price
        logger.info(
            f"[BUY] {etf_code} {order_qty}주 "
            f"(≈ {buy_value:,.0f}원, 구간={kr_order_phase_label(order_phase)})"
        )
        result = execute_kis_buy_qty_by_phase(trader, etf_code, order_qty, order_phase)
        logger.info(f"매수 결과: {result}")

    else:
        logger.info("[HOLD] 매매수량이 없거나 주문금액 미달 - 유지")

    logger.info("=== KIS ISA 위대리(WDR) 자동매매 완료 ===")

    # 텔레그램 요약
    tg = [f"<b>KIS ISA 위대리</b>"]
    tg.append(f"총자산: {total_value:,.0f}원 (현금 {cash:,.0f} / ETF {current_shares}주)")
    tg.append(f"이격도: {signal['divergence']}% | 상태: {signal['state']}")
    if bt_stock_ratio is not None:
        tg.append(f"목표비율: {bt_stock_ratio*100:.0f}% | 판단: {order_action or 'HOLD'} {order_qty}주")
    else:
        tg.append(f"판단(폴백): {order_action or 'HOLD'} {order_qty}주")
    send_telegram("\n".join(tg))
