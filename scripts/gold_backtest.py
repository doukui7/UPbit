"""
gold_backtest.py
================
KRX 금현물 일봉 데이터(krx_gold_daily.csv)를 사용한 백테스트.

수수료: 키움증권 금현물 온라인 0.3% (0.003)
데이터: 2022.01 ~ 2026.02 (약 1001 거래일)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from backtest.engine import BacktestEngine

# ─────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────
CSV_FILE      = os.path.join(os.path.dirname(__file__), "krx_gold_daily.csv")
INITIAL_CAP   = 10_000_000   # 초기 자본: 1,000만원
FEE           = 0.003        # 키움 금현물 수수료 0.3%
START_DATE    = "2022-06-01" # 워밍업 기간 제외 후 평가 시작일


def load_gold_df(csv_path: str) -> pd.DataFrame:
    """krx_gold_daily.csv -> BacktestEngine 호환 OHLCV DataFrame."""
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    if "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    return df


def run_sma_backtest(df: pd.DataFrame, period: int):
    engine = BacktestEngine()
    return engine.run_backtest(
        ticker=None,
        df=df,
        period=period,
        interval="day",
        fee=FEE,
        start_date=START_DATE,
        initial_balance=INITIAL_CAP,
        strategy_mode="SMA Strategy",
        slippage=0.0,
    )


def run_donchian_backtest(df: pd.DataFrame, buy_period: int, sell_period: int):
    engine = BacktestEngine()
    return engine.run_backtest(
        ticker=None,
        df=df,
        period=buy_period,
        interval="day",
        fee=FEE,
        start_date=START_DATE,
        initial_balance=INITIAL_CAP,
        strategy_mode="Donchian",
        sell_period_ratio=sell_period / buy_period,
        slippage=0.0,
    )


def print_result(label: str, result: dict):
    if "error" in result:
        print(f"  [FAIL] {label}: {result['error']}")
        return
    p = result["performance"]
    print(f"\n{'='*55}")
    print(f"  [결과] {label}")
    print(f"{'='*55}")
    print(f"  수익률  : {p['total_return']:+.2f}%")
    print(f"  CAGR    : {p['cagr']:+.2f}%")
    print(f"  MDD     : {p['mdd']:.2f}%")
    print(f"  샤프    : {p['sharpe']:.2f}")
    print(f"  매매횟수: {p['trade_count']}회")
    print(f"  승률    : {p['win_rate']:.1f}%")
    print(f"  최종 자산: {p['final_equity']:,.0f}원")
    print(f"\n  [연도별 성과]")
    ys = p.get("yearly_stats")
    if ys is not None and not ys.empty:
        for yr, row in ys.iterrows():
            print(f"    {yr}: 수익률 {row['Return (%)']:+.1f}%, MDD {row['MDD (%)']:.1f}%")


def optimize_sma(df: pd.DataFrame):
    """SMA 기간 최적화 (5~100)."""
    print("\n\n[SMA 최적화 - 수익률 상위 5개]")
    engine = BacktestEngine()
    results = engine.optimize_sma(
        df=df,
        sma_range=range(5, 101),
        fee=FEE,
        start_date=START_DATE,
        initial_balance=INITIAL_CAP,
    )
    if not results:
        print("  결과 없음")
        return
    results_sorted = sorted(results, key=lambda x: x["total_return"], reverse=True)
    print(f"  {'기간':>4}  {'수익률':>8}  {'CAGR':>7}  {'MDD':>7}  {'샤프':>5}")
    for r in results_sorted[:5]:
        print(f"  SMA {r['SMA Period']:>3}  {r['total_return']:>+8.1f}%  {r['cagr']:>+7.2f}%  "
              f"{r['mdd']:>7.2f}%  {r['sharpe']:>5.2f}")


def optimize_donchian(df: pd.DataFrame):
    """Donchian 기간 최적화 (buy 10~150, sell 5~80)."""
    print("\n\n[Donchian 최적화 - Calmar 상위 5개]")
    engine = BacktestEngine()
    results = engine.optimize_donchian(
        df=df,
        buy_range=range(10, 151, 5),
        sell_range=range(5, 81, 5),
        fee=FEE,
        start_date=START_DATE,
        initial_balance=INITIAL_CAP,
    )
    if not results:
        print("  결과 없음")
        return
    for r in results:
        r["calmar"] = abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0
    results_sorted = sorted(results, key=lambda x: x["calmar"], reverse=True)
    print(f"  {'Buy':>4}  {'Sell':>4}  {'수익률':>8}  {'CAGR':>7}  {'MDD':>7}  {'Calmar':>7}")
    for r in results_sorted[:5]:
        print(f"  {r['Buy Period']:>4}  {r['Sell Period']:>4}  "
              f"{r['total_return']:>+8.1f}%  {r['cagr']:>+7.2f}%  "
              f"{r['mdd']:>7.2f}%  {r['calmar']:>7.2f}")


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  KRX 금현물 백테스트 (키움 수수료 0.3% 적용)")
    print(f"  데이터: {CSV_FILE}")
    print(f"  평가 시작: {START_DATE}  |  초기 자본: {INITIAL_CAP:,}원")
    print("=" * 55)

    # 데이터 로드
    try:
        df = load_gold_df(CSV_FILE)
        print(f"  데이터 로드: {len(df)}행 ({df.index[0].date()} ~ {df.index[-1].date()})")
    except FileNotFoundError:
        print(f"  [FAIL] {CSV_FILE} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # 기본 전략 백테스트
    print("\n[기본 전략 백테스트]")
    print_result("SMA(20) 일봉",           run_sma_backtest(df, 20))
    print_result("SMA(29) 일봉",           run_sma_backtest(df, 29))
    print_result("Donchian(60/30) 일봉",   run_donchian_backtest(df, 60, 30))
    print_result("Donchian(100/50) 일봉",  run_donchian_backtest(df, 100, 50))

    # 최적화
    optimize_sma(df)
    optimize_donchian(df)

    print("\n" + "=" * 55)
    print("  백테스트 완료!")
    print("=" * 55)
