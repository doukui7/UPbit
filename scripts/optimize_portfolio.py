"""
포트폴리오 Optuna 최적화 + 인근 파라미터 안정성 검증
"""
import pandas as pd
import numpy as np
from backtest.engine import BacktestEngine
import data_cache

engine = BacktestEngine()

strategies = [
    {'name': 'BTC_DC_D',   'ticker': 'KRW-BTC', 'mode': 'Donchian',     'interval': 'day',       'cur_buy': 20,  'cur_sell': 20},
    {'name': 'BTC_DC_4H',  'ticker': 'KRW-BTC', 'mode': 'Donchian',     'interval': 'minute240', 'cur_buy': 115, 'cur_sell': 105},
    {'name': 'ETH_SMA_D',  'ticker': 'KRW-ETH', 'mode': 'SMA Strategy', 'interval': 'day',       'cur_buy': 40,  'cur_sell': 0},
    {'name': 'SOL_SMA_D',  'ticker': 'KRW-SOL', 'mode': 'SMA Strategy', 'interval': 'day',       'cur_buy': 40,  'cur_sell': 0},
    {'name': 'BTC_SMA_D',  'ticker': 'KRW-BTC', 'mode': 'SMA Strategy', 'interval': 'day',       'cur_buy': 35,  'cur_sell': 0},
]

SEP = '=' * 95

for s in strategies:
    print(f"\n{SEP}")
    print(f"  {s['name']} ({s['ticker']}, {s['interval']})")
    print(SEP)

    df = data_cache.get_ohlcv_cached(s['ticker'], interval=s['interval'], count=5000)
    if df is None or len(df) < 50:
        print("  SKIP - no data")
        continue

    is_dc = 'Donchian' in s['mode']

    # ── 1. 현재 파라미터 성능 ──
    if is_dc:
        result = engine.run_backtest(
            s['ticker'], period=s['cur_buy'], interval=s['interval'],
            count=5000, fee=0.0005, initial_balance=1000000,
            df=df, strategy_mode=s['mode'], sell_period_ratio=s['cur_sell']/s['cur_buy']
        )
    else:
        result = engine.run_backtest(
            s['ticker'], period=s['cur_buy'], interval=s['interval'],
            count=5000, fee=0.0005, initial_balance=1000000,
            df=df, strategy_mode=s['mode']
        )

    if 'error' not in result:
        p = result['performance']
        calmar = abs(p['cagr'] / p['mdd']) if p['mdd'] != 0 else 0
        param_str = f"B{s['cur_buy']}/S{s['cur_sell']}" if is_dc else f"SMA {s['cur_buy']}"
        print(f"  [CURRENT] {param_str:15s}  CAGR={p['cagr']:6.1f}%  MDD={p['mdd']:6.1f}%  Calmar={calmar:.2f}  Sharpe={p['sharpe']:.2f}  Trades={p['trade_count']}")

    # ── 2. Optuna 최적화 ──
    if is_dc:
        opt = engine.optuna_optimize(
            df, strategy_mode='Donchian',
            buy_range=(10, 200), sell_range=(5, 100),
            fee=0.0005, n_trials=300, objective_metric='calmar'
        )
        best_bp = opt['best_params']['buy_period']
        best_sp = opt['best_params']['sell_period']
        param_str = f"B{best_bp}/S{best_sp}"
    else:
        opt = engine.optuna_optimize(
            df, strategy_mode='SMA Strategy',
            buy_range=(10, 200),
            fee=0.0005, n_trials=200, objective_metric='calmar'
        )
        best_bp = opt['best_params']['sma_period']
        best_sp = 0
        param_str = f"SMA {best_bp}"

    best_t = max(opt['trials'], key=lambda t: t['calmar'])
    print(f"  [OPTUNA]  {param_str:15s}  CAGR={best_t['cagr']:6.1f}%  MDD={best_t['mdd']:6.1f}%  Calmar={best_t['calmar']:.2f}  Sharpe={best_t['sharpe']:.2f}  Trades={best_t['trade_count']}")

    # ── 3. 인근 파라미터 안정성 (±10) ──
    print(f"\n  [Robustness: +/-10 around best]")

    close_arr = df['close'].values
    open_arr = df['open'].values

    if is_dc:
        neighbor_data = []
        for bp_t in range(max(5, best_bp - 10), best_bp + 11, 2):
            upper = df['high'].rolling(window=bp_t).max().shift(1).values
            for sp_t in range(max(5, best_sp - 10), best_sp + 11, 2):
                lower = df['low'].rolling(window=sp_t).min().shift(1).values
                sig = np.zeros(len(close_arr), dtype=np.int8)
                sig[close_arr > upper] = 1
                sig[close_arr < lower] = -1
                res = engine._fast_simulate(open_arr, close_arr, sig, 0.0005, 0.0, 1000000)
                res['calmar'] = abs(res['cagr'] / res['mdd']) if res['mdd'] != 0 else 0
                neighbor_data.append({'bp': bp_t, 'sp': sp_t, **res})

        ndf = pd.DataFrame(neighbor_data)
        # Buy period별 평균
        print(f"  Buy Period 변화 (Sell={best_sp} 고정):")
        for bp_t in range(max(5, best_bp - 10), best_bp + 11, 2):
            row = ndf[(ndf['bp'] == bp_t) & (ndf['sp'] == best_sp)]
            if not row.empty:
                r = row.iloc[0]
                marker = " <-- BEST" if bp_t == best_bp else ""
                print(f"    B{bp_t:3d}: Calmar={r['calmar']:.2f}, CAGR={r['cagr']:.1f}%, MDD={r['mdd']:.1f}%{marker}")

        print(f"\n  Sell Period 변화 (Buy={best_bp} 고정):")
        for sp_t in range(max(5, best_sp - 10), best_sp + 11, 2):
            row = ndf[(ndf['bp'] == best_bp) & (ndf['sp'] == sp_t)]
            if not row.empty:
                r = row.iloc[0]
                marker = " <-- BEST" if sp_t == best_sp else ""
                print(f"    S{sp_t:3d}: Calmar={r['calmar']:.2f}, CAGR={r['cagr']:.1f}%, MDD={r['mdd']:.1f}%{marker}")

        calmars = ndf['calmar'].values
    else:
        neighbor_data = []
        for p_t in range(max(5, best_bp - 10), best_bp + 11):
            sma_vals = df['close'].rolling(window=p_t).mean().values
            sig = np.zeros(len(close_arr), dtype=np.int8)
            valid = ~np.isnan(sma_vals)
            sig[valid & (close_arr > sma_vals)] = 1
            sig[valid & (close_arr <= sma_vals)] = -1
            res = engine._fast_simulate(open_arr, close_arr, sig, 0.0005, 0.0, 1000000)
            res['calmar'] = abs(res['cagr'] / res['mdd']) if res['mdd'] != 0 else 0
            neighbor_data.append({'period': p_t, **res})

        for nd in neighbor_data:
            marker = " <-- BEST" if nd['period'] == best_bp else ""
            cur_marker = " (CURRENT)" if nd['period'] == s['cur_buy'] else ""
            print(f"    SMA {nd['period']:3d}: Calmar={nd['calmar']:.2f}, CAGR={nd['cagr']:.1f}%, MDD={nd['mdd']:.1f}%{marker}{cur_marker}")

        calmars = np.array([nd['calmar'] for nd in neighbor_data])

    avg_c = np.mean(calmars)
    std_c = np.std(calmars)
    cv = std_c / avg_c if avg_c > 0 else 999
    print(f"\n  Calmar: avg={avg_c:.2f}, std={std_c:.2f}, CV={cv:.2f}")
    if cv > 0.3:
        print("  >> UNSTABLE (CV > 0.3) - parameter sensitive!")
    else:
        print("  >> STABLE (CV <= 0.3) - robust parameter choice")

print(f"\n{SEP}")
print("  DONE")
print(SEP)
