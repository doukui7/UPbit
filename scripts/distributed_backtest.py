import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from backtest.engine import BacktestEngine
from data_cache import load_cached

# 30분봉 데이터 로드 (캐시 활용)
def load_minute30_data(ticker="KRW-BTC"):
    print(f"📦 [Data] {ticker} 30분봉 데이터 로드 중...")
    df = load_cached(ticker, "minute30")
    if df is not None:
        # VWAP 계산
        df['vwap'] = df['value'] / df['volume']
        df = df.sort_index()
        print(f"   -> {len(df)} candles ({df.index.min()} ~ {df.index.max()})")
    return df

def get_distributed_price(minute_df, start_time, duration_minutes):
    """
    특정 시간(start_time)부터 duration_minutes 동안의 분산 체결 평균가(VWAP) 계산
    duration_minutes: 0(Instant), 30, 60, 120
    """
    if duration_minutes == 0:
        # Instant: 해당 시각의 Open 가격 (없으면 다음 봉 Open)
        try:
            # 일치하는 시간이 있으면 그 Open 사용
            if start_time in minute_df.index:
                return minute_df.loc[start_time]['open']
            # 없으면 바로 다음 봉 찾기
            idx = minute_df.index.searchsorted(start_time)
            if idx < len(minute_df):
                return minute_df.iloc[idx]['open']
        except:
            pass
        return None

    # Distributed: start_time ~ start_time + duration
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    # 해당 구간의 30분봉 조회
    mask = (minute_df.index >= start_time) & (minute_df.index < end_time)
    sub_df = minute_df[mask]
    
    if len(sub_df) == 0:
        return None
        
    # 구간 전체의 VWAP 계산 = Total Value / Total Volume
    total_val = sub_df['value'].sum()
    total_vol = sub_df['volume'].sum()
    
    if total_vol == 0:
        return sub_df.iloc[0]['open'] # 거래량 없으면 Open
        
    return total_val / total_vol

def run_distributed_analysis():
    engine = BacktestEngine()
    
    # 전략별 백테스트 실행 (슬리피지 0으로 원본 시그널 확보)
    strategies = [
        {"name": "SMA(29) Daily", "interval": "day", "param": 29, "strategy": "SMA Strategy"},
        {"name": "Donchian(115/105) 4H", "interval": "minute240", "param": 115, "sell_param": 105, "strategy": "Donchian Trend"}
    ]
    
    # 30분봉 데이터 준비 (전체 공유)
    minute_df = load_minute30_data("KRW-BTC")
    if minute_df is None:
        print("❌ 30분봉 데이터가 없습니다. 분석 중단.")
        return

    # User의 5분봉 데이터 정보도 출력
    df5 = load_cached("KRW-BTC", "minute5")
    if df5 is not None:
        print(f"ℹ️ 5분봉 데이터: {len(df5)}개 ({df5.index.min()} ~ {df5.index.max()}) - 기간이 짧아 이번 분석에서는 30분봉을 주력으로 사용합니다.")

    print("\n🔍 분산 주문 백테스트 시작 (비교: Instant vs 30분/60분/120분 분산)\n")
    print("  * 이득(Benefit)이 +면 분산 주문이 유리, -면 즉시 체결이 유리")
    
    for strat in strategies:
        print(f"\n============== {strat['name']} 분석 ==============")
        # 1. Backtest run
        if strat['strategy'] == "Donchian Trend":
            result = engine.run_backtest("KRW-BTC", period=strat['param'], interval=strat['interval'], 
                                       strategy_mode="Donchian Trend", sell_period_ratio=strat['sell_param']/strat['param'],
                                       count=20000)
        else:
            result = engine.run_backtest("KRW-BTC", period=strat['param'], interval=strat['interval'],
                                       strategy_mode="SMA Strategy", count=10000)
            
        trades = result['performance']['trades']
        if not trades:
            print("  -> 매매 없음")
            continue
            
        print(f"  -> 총 실행 횟수: {len(trades)}건")
        
        # 2. 각 Execution에 대해 분산 체결가 계산
        scenarios = [30, 60, 120]
        results = {s: {'buy_benefit': [], 'sell_benefit': []} for s in scenarios}
        
        valid_count = 0
        
        for t in trades:
            exec_time = pd.Timestamp(t['date']) # 체결 시각
            side = t['type'] # 'BUY' or 'SELL'
            base_price = t['open_price'] # Instant Price (Open)

            if not base_price: continue

            # 각 시나리오별 VWAP 계산
            valid_all = True
            prices = {}
            for s in scenarios:
                p = get_distributed_price(minute_df, exec_time, s)
                if p is None:
                    valid_all = False
                    break
                prices[s] = p
            
            if not valid_all:
                continue
            
            valid_count += 1
            
            # Debug Print (First 5)
            if valid_count <= 5:
                print(f"    [Debug] Start: {exec_time}, Side: {side}, Base: {base_price:,.0f}")
                for s in scenarios:
                    print(f"      - {s}min DistPrice: {prices[s]:,.0f} (Diff: {(prices[s]-base_price)/base_price*100:.3f}%)")

            # Benefit 계산
            for s in scenarios:
                dist_price = prices[s]
                diff_pct = (dist_price - base_price) / base_price * 100
                
                if side.upper() == 'BUY':
                    # 매수: 가격이 떨어져야 이득 (Diff < 0 -> Benefit > 0)
                    benefit = -diff_pct
                    results[s]['buy_benefit'].append(benefit)
                elif side.upper() == 'SELL':
                    # 매도: 가격이 올라야 이득 (Diff > 0 -> Benefit > 0)
                    benefit = diff_pct
                    results[s]['sell_benefit'].append(benefit)

        print(f"  -> 분석 가능: {valid_count}건")
        
        # 3. 통계 출력
        print(f"  {'분산시간':<10} | {'매수 이득(Avg)':<15} | {'매도 이득(Avg)':<15} | {'전체 이득(Avg)':<15}")
        print("-" * 70)
        
        for s in scenarios:
            buys = results[s]['buy_benefit']
            sells = results[s]['sell_benefit']
            all_benefits = buys + sells
            
            avg_buy = np.mean(buys) if buys else 0
            avg_sell = np.mean(sells) if sells else 0
            avg_total = np.mean(all_benefits) if all_benefits else 0
            
            print(f"  {s:<2}min{' ':>5} | {avg_buy:>12.3f}%P | {avg_sell:>12.3f}%P | {avg_total:>12.3f}%P")
            
        # Best Scenario
        best_s = max(scenarios, key=lambda s: np.mean(results[s]['buy_benefit'] + results[s]['sell_benefit']) if results[s]['buy_benefit'] else -999)
        best_val = np.mean(results[best_s]['buy_benefit'] + results[best_s]['sell_benefit'])
        if best_val > 0.05:
            print(f"  => ✅ 추천: {best_s}분 분산 주문이 유리함 (+{best_val:.3f}%)")
        elif best_val < -0.05:
            print(f"  => ❌ 비추천: 즉시 체결(Instant)이 유리함 (분산 시 {best_val:.3f}% 손해)")
        else:
            print(f"  => ⚖️ 중립: 큰 차이 없음 ({best_val:.3f}%)")

if __name__ == "__main__":
    run_distributed_analysis()
