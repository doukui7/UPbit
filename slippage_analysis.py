"""
슬리피지 분석 스크립트
- 일봉 SMA(29) / 4시간봉 Donchian(115/105) 백테스트 매매 내역 추출
- 각 매매 시점을 소분봉(5분/30분/60분) 데이터와 매칭하여 실제 슬리피지 계산

사용 가능 캐시:
  - minute5:   2026-01-31 ~ 2026-02-16 (4,754건)
  - minute30:  2020-07-10 ~ 2026-02-18 (98,118건) ← 가장 넓은 범위
  - minute60:  2025-01-13 ~ 2026-02-16 (9,562건)
  - minute15:  2025-11-22 ~ 2026-02-16 (8,265건)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine
from data_cache import load_cached

# ================================================================
# 설정
# ================================================================
TICKER = "KRW-BTC"
START_DATE = "2020-01-01"
INITIAL_BALANCE = 1_000_000

STRATEGIES = [
    {
        "name": "SMA(29) 일봉",
        "mode": "SMA Strategy",
        "interval": "day",
        "period": 29,
        "sell_period_ratio": 0.5,
        "exec_config": {"splits": 3, "wait_sec": 60, "timeout_sec": 600},
    },
    {
        "name": "Donchian(115/105) 4시간봉",
        "mode": "Donchian",
        "interval": "minute240",
        "period": 115,
        "sell_period_ratio": 105 / 115,
        "exec_config": {"splits": 3, "wait_sec": 30, "timeout_sec": 300},
    },
]

# 분석에 사용할 소분봉 (넓은 범위 → 좁은 범위 순으로 시도)
DETAIL_INTERVALS = [
    ("minute30", "30분봉"),
    ("minute60", "60분봉"),
    ("minute15", "15분봉"),
    ("minute5",  "5분봉"),
]


# ================================================================
# 소분봉 데이터 로드
# ================================================================
def load_detail_candles():
    """사용 가능한 소분봉 캐시를 모두 로드"""
    loaded = {}
    for interval, label in DETAIL_INTERVALS:
        df = load_cached(TICKER, interval)
        if df is not None and len(df) > 0:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            loaded[interval] = df
            print(f"  ✅ {label:<8} {str(df.index[0])[:19]} ~ {str(df.index[-1])[:19]} ({len(df):,}건)")
        else:
            print(f"  ⬜ {label:<8} 캐시 없음")
    return loaded


def find_best_detail(detail_candles, trade_time, duration_minutes):
    """
    매매 시점에 가장 해상도가 높은 소분봉 데이터를 선택.
    trade_time ~ trade_time + duration_minutes 범위에 데이터가 있는 것 선택.
    우선순위: minute5 > minute15 > minute30 > minute60
    """
    t_start = trade_time
    t_end = trade_time + pd.Timedelta(minutes=duration_minutes)

    # 해상도 높은 것부터 시도
    priority = ["minute5", "minute15", "minute30", "minute60"]
    for iv in priority:
        if iv not in detail_candles:
            continue
        df = detail_candles[iv]
        subset = df[(df.index >= t_start) & (df.index < t_end)]
        if len(subset) >= 1:
            return iv, subset

    return None, None


# ================================================================
# 슬리피지 계산
# ================================================================
def calc_slippage(trade, detail_candles, exec_config):
    """
    매매 시점의 Open(t+1)을 소분봉 데이터와 비교하여 슬리피지 측정.
    """
    trade_time = pd.Timestamp(trade['date'])
    if trade_time.tz is not None:
        trade_time = trade_time.tz_localize(None)

    open_price = trade['open_price']
    trade_type = trade['type']
    timeout_min = exec_config['timeout_sec'] / 60

    # 최대 탐색 범위: 타임아웃 + 여유
    search_minutes = max(timeout_min, 60)

    iv_used, candles = find_best_detail(detail_candles, trade_time, search_minutes)
    if candles is None or len(candles) == 0:
        return None

    result = {
        'trade_date': trade_time,
        'trade_type': trade_type,
        'backtest_open': open_price,
        'detail_interval': iv_used,
        'detail_candles': len(candles),
    }

    # 1) 소분봉 첫 봉의 Open (= 실제 봉 시작 가격)
    first_open = candles.iloc[0]['open']
    result['detail_first_open'] = first_open
    result['open_diff_pct'] = (first_open - open_price) / open_price * 100

    # 2) 소분봉 첫 봉의 Close (단순 지연 체결)
    first_close = candles.iloc[0]['close']
    result['detail_first_close'] = first_close
    if trade_type == 'buy':
        result['slip_first_close_pct'] = (first_close - open_price) / open_price * 100
    else:
        result['slip_first_close_pct'] = (open_price - first_close) / open_price * 100

    # 3) 타임아웃 범위 내 VWAP (지정가 분할 체결 시뮬레이션)
    t_timeout = trade_time + pd.Timedelta(seconds=exec_config['timeout_sec'])
    timeout_candles = candles[candles.index < t_timeout]
    if len(timeout_candles) > 0:
        vol_sum = timeout_candles['volume'].sum()
        if vol_sum > 0:
            vwap = (timeout_candles['close'] * timeout_candles['volume']).sum() / vol_sum
        else:
            vwap = timeout_candles['close'].mean()

        high_max = timeout_candles['high'].max()
        low_min = timeout_candles['low'].min()

        result['vwap'] = vwap
        result['range_high'] = high_max
        result['range_low'] = low_min

        if trade_type == 'buy':
            result['slip_vwap_pct'] = (vwap - open_price) / open_price * 100
            result['slip_worst_pct'] = (high_max - open_price) / open_price * 100
            result['slip_best_pct'] = (low_min - open_price) / open_price * 100
        else:
            result['slip_vwap_pct'] = (open_price - vwap) / open_price * 100
            result['slip_worst_pct'] = (open_price - low_min) / open_price * 100
            result['slip_best_pct'] = (open_price - high_max) / open_price * 100

    return result


# ================================================================
# 리포트 출력
# ================================================================
def print_trade_table(results):
    """개별 매매 슬리피지 테이블"""
    print(f"\n  {'─' * 110}")
    hdr = f"  {'날짜':<22} {'유형':<5} {'BT Open':>14} {'실제Open':>14} {'차이%':>8} {'VWAP':>14} {'VWAP슬립':>8} {'최악':>8} {'소스':<8}"
    print(hdr)
    print(f"  {'─' * 110}")

    for r in results:
        vwap = r.get('vwap', 0)
        slip_vwap = r.get('slip_vwap_pct', 0)
        slip_worst = r.get('slip_worst_pct', 0)
        print(
            f"  {str(r['trade_date']):<22} "
            f"{'매수' if r['trade_type']=='buy' else '매도':<5} "
            f"{r['backtest_open']:>14,.0f} "
            f"{r['detail_first_open']:>14,.0f} "
            f"{r['open_diff_pct']:>+7.3f}% "
            f"{vwap:>14,.0f} "
            f"{slip_vwap:>+7.3f}% "
            f"{slip_worst:>+7.3f}% "
            f"{r.get('detail_interval', '?'):<8}"
        )


def print_statistics(results):
    """통계 요약"""
    df_slip = pd.DataFrame(results)
    buy_df = df_slip[df_slip['trade_type'] == 'buy']
    sell_df = df_slip[df_slip['trade_type'] == 'sell']

    print(f"\n  {'═' * 80}")
    print(f"  📊 슬리피지 통계 요약")
    print(f"  {'═' * 80}")

    for label, sub_df in [("전체", df_slip), ("매수", buy_df), ("매도", sell_df)]:
        if len(sub_df) == 0:
            continue
        print(f"\n  [{label}] ({len(sub_df)}건)")

        metrics = [
            ('open_diff_pct',       'Open 가격 차이    '),
            ('slip_first_close_pct','첫 봉 Close 슬립  '),
            ('slip_vwap_pct',       'VWAP 슬리피지     '),
            ('slip_worst_pct',      '최악 슬리피지     '),
            ('slip_best_pct',       '최선 슬리피지     '),
        ]

        for col, name in metrics:
            if col not in sub_df.columns:
                continue
            vals = sub_df[col].dropna()
            if len(vals) == 0:
                continue
            print(f"    {name}: 평균 {vals.mean():+.4f}% | 중앙값 {vals.median():+.4f}% | std {vals.std():.4f}% | min {vals.min():+.4f}% | max {vals.max():+.4f}%")


# ================================================================
# 메인
# ================================================================
def main():
    engine = BacktestEngine()

    print("📦 소분봉 캐시 로드 중...")
    detail_candles = load_detail_candles()

    if not detail_candles:
        print("❌ 소분봉 캐시 데이터가 없습니다.")
        return

    # 전체 소분봉 범위
    all_starts = [df.index[0] for df in detail_candles.values()]
    all_ends = [df.index[-1] for df in detail_candles.values()]
    overall_start = min(all_starts)
    overall_end = max(all_ends)
    print(f"\n  📅 소분봉 전체 커버 범위: {overall_start} ~ {overall_end}")
    print("=" * 110)

    for strat in STRATEGIES:
        print(f"\n{'=' * 110}")
        print(f"📈 전략: {strat['name']}")
        print(f"{'=' * 110}")

        # 데이터 로드
        df = load_cached(TICKER, strat['interval'])
        if df is None:
            print(f"  ❌ {strat['interval']} 데이터 로드 실패")
            continue

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 백테스트 실행 (슬리피지 0으로 순수 매매내역 추출)
        result = engine.run_backtest(
            ticker=TICKER,
            period=strat['period'],
            interval=strat['interval'],
            fee=0.0005,
            start_date=START_DATE,
            initial_balance=INITIAL_BALANCE,
            df=df,
            strategy_mode=strat['mode'],
            sell_period_ratio=strat['sell_period_ratio'],
            slippage=0.0,
        )

        if "error" in result:
            print(f"  ❌ 백테스트 에러: {result['error']}")
            continue

        trades = result['performance']['trades']
        perf = result['performance']
        print(f"  총 매매: {len(trades)}건 (매도 {perf['trade_count']}회)")
        print(f"  수익률: {perf['total_return']:.2f}%, MDD: {perf['mdd']:.2f}%")

        # 모든 매매에 대해 슬리피지 계산
        matched = []
        unmatched = 0

        for trade in trades:
            slip = calc_slippage(trade, detail_candles, strat['exec_config'])
            if slip:
                matched.append(slip)
            else:
                unmatched += 1

        print(f"\n  소분봉 매칭: {len(matched)}건 성공, {unmatched}건 범위 밖")

        if not matched:
            print("  ⚠️ 소분봉 범위 내 매매가 없습니다.")
            continue

        # 사용된 소분봉 소스 분포
        iv_counts = {}
        for m in matched:
            iv = m.get('detail_interval', '?')
            iv_counts[iv] = iv_counts.get(iv, 0) + 1
        print(f"  소스 분포: {iv_counts}")

        print_trade_table(matched)
        print_statistics(matched)

    print(f"\n{'=' * 110}")
    print("✅ 분석 완료")


if __name__ == "__main__":
    main()
