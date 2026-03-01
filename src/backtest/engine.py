import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.strategy.sma import SMAStrategy
from src.strategy.donchian import DonchianStrategy
import src.engine.data_cache

# CPU 코어 수 기반 워커 수 (최소 2, 최대 물리코어 -1)
_NUM_WORKERS = max(2, min((os.cpu_count() or 4) - 1, 12))


# ── 프로세스 간 전송 가능한 top-level 시뮬레이션 함수 ──────────────
def _simulate_task(open_arr, close_arr, signal_arr, fee, slippage,
                   initial_balance, year_arr, meta: dict) -> dict:
    """단일 파라미터 조합 시뮬레이션 (ProcessPool 워커용)."""
    n = len(open_arr)
    balance = initial_balance
    coin_balance = 0.0
    position = 0
    pending = 0
    buy_price = 0.0
    wins = 0
    sells = 0
    equity = np.empty(n)
    slip_buy = 1 + slippage / 100
    slip_sell = 1 - slippage / 100
    fee_mult = 1 - fee

    for i in range(n):
        op = open_arr[i]
        cl = close_arr[i]
        sig = signal_arr[i]
        if pending == 1:
            ep = op * slip_buy
            coin_balance = balance * fee_mult / ep
            balance = 0.0
            position = 1
            buy_price = ep
            pending = 0
        elif pending == -1:
            ep = op * slip_sell
            sell_ret = (ep - buy_price) / buy_price if buy_price > 0 else 0
            balance = coin_balance * ep * fee_mult
            coin_balance = 0.0
            position = 0
            sells += 1
            if sell_ret > 0:
                wins += 1
            pending = 0
        if position == 0 and sig == 1:
            pending = 1
        elif position == 1 and sig == -1:
            pending = -1
        equity[i] = coin_balance * cl if position == 1 else balance

    final_eq = equity[-1]
    total_ret = (final_eq - initial_balance) / initial_balance * 100
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    mdd = dd.min()
    avg_yearly_mdd = mdd
    if year_arr is not None and len(year_arr) == n:
        unique_years = np.unique(year_arr)
        yearly_mdds = [dd[year_arr == yr].min() for yr in unique_years]
        if yearly_mdds:
            avg_yearly_mdd = float(np.mean(yearly_mdds))
    win_rate = (wins / sells * 100) if sells > 0 else 0
    returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns)
    std = returns.std()
    sharpe = (returns.mean() / std * np.sqrt(365)) if std > 0 else 0
    days_total = max(1, n)
    cagr = 0
    if final_eq > 0 and initial_balance > 0 and days_total > 1:
        cagr = ((final_eq / initial_balance) ** (365 / days_total) - 1) * 100

    res = {
        "total_return": total_ret, "cagr": cagr, "mdd": mdd,
        "avg_yearly_mdd": avg_yearly_mdd, "win_rate": win_rate,
        "trade_count": sells, "sharpe": sharpe, "final_equity": final_eq,
    }
    res.update(meta)
    return res

class BacktestEngine:
    def __init__(self):
        self.strategy = SMAStrategy()

    def run_backtest(self, ticker, period=20, interval="day", count=200, fee=0.0005, start_date=None, initial_balance=1000000, df=None, strategy_mode="SMA Strategy", sell_period_ratio=0.5, slippage=0.0, sell_mode="lower", progress_callback=None):
        """
        Run backtest for a given ticker and parameters.
        Execution Logic: Signal on Close(t) -> Trade on Open(t+1)
        slippage: percentage (e.g. 0.1 = 0.1%). Buy price increases, Sell price decreases.
        """
        def _emit(unit, msg):
            if progress_callback:
                try:
                    progress_callback(int(unit), 100, msg)
                except Exception:
                    pass

        _emit(2, "데이터 준비")

        # 1. Fetch Data
        try:
            if df is None:
                fetch_count = count
                df = data_cache.get_ohlcv_local_first(
                    ticker,
                    interval=interval,
                    count=fetch_count,
                    allow_api_fallback=True,
                )
                if df is None:
                    return {"error": "Failed to fetch data"}
            else:
                df = df.copy()
        except Exception as e:
            return {"error": str(e)}

        _emit(12, "지표 계산")

        # 2. Apply Strategy FIRST (on ALL data for proper warmup)
        if strategy_mode == "Donchian" or strategy_mode == "Donchian Trend":
            self.strategy = DonchianStrategy()
            buy_p = period
            sell_p = max(5, int(buy_p * sell_period_ratio))
            df = self.strategy.create_features(df, buy_period=buy_p, sell_period=sell_p)
            df['signal'] = df.apply(
                lambda row: self.strategy.get_signal(row, buy_period=buy_p, sell_period=sell_p, sell_mode=sell_mode),
                axis=1
            )

        else: # Default SMA
            self.strategy = SMAStrategy()
            calc_periods = [period]
            df = self.strategy.create_features(df, periods=calc_periods)
            df['signal'] = df.apply(lambda row: self.strategy.get_signal(row, strategy_type='SMA_CROSS', ma_period=period), axis=1)

        _emit(20, "기간 필터링")

        # 3. Filter by start_date AFTER features (preserves warmup data)
        if start_date:
            start_ts = pd.to_datetime(start_date)
            if df.index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(df.index.tz)
            df = df[df.index >= start_ts]

        # 4. Check sufficiency
        if len(df) < 5:
            return {"error": f"Insufficient data after filtering (got {len(df)}, need >= 5)"}
        
        balance = initial_balance
        coin_balance = 0
        
        trades = []
        equity_curve = []
        
        # State
        position = 'CASH' # CASH or HOLD
        pending_action = None # 'BUY' or 'SELL' or None
        buy_price = 0
        
        _emit(25, "시뮬레이션")

        # Iterate
        n_rows = len(df)
        step = max(1, n_rows // 100)
        for i in range(n_rows):
            date = df.index[i]
            row = df.iloc[i]
            
            open_price = row['open']
            close_price = row['close']
            signal = row['signal']
            
            # 1. Execute Pending Action at OPEN (with slippage)
            if pending_action == 'BUY':
                # Buy at Open + slippage (worse price for buyer)
                exec_price = open_price * (1 + slippage / 100)
                coin_balance = balance * (1 - fee) / exec_price
                balance = 0
                position = 'HOLD'
                buy_price = exec_price
                
                # Fetch conditions from previous day (Signal Source)
                prev_row = df.iloc[i-1] if i > 0 else row
                p_sma = prev_row.get(f'SMA_{period}', 0)
                
                trades.append({
                    'date': date,
                    'type': 'buy',
                    'price': exec_price,
                    'open_price': open_price,
                    'slippage_pct': slippage,
                    'amount': coin_balance,
                    'balance': coin_balance * exec_price,
                    'sma': p_sma
                })
                pending_action = None
                
            elif pending_action == 'SELL':
                # Sell at Open - slippage (worse price for seller)
                exec_price = open_price * (1 - slippage / 100)
                balance = coin_balance * exec_price * (1 - fee)
                coin_balance = 0
                position = 'CASH'

                # Fetch conditions from previous day
                prev_row = df.iloc[i-1] if i > 0 else row
                p_sma = prev_row.get(f'SMA_{period}', 0)

                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': exec_price,
                    'open_price': open_price,
                    'slippage_pct': slippage,
                    'amount': 0,
                    'balance': balance,
                    'profit': (exec_price - buy_price) / buy_price * 100,
                    'sma': p_sma,
                    'long_sma': 0,
                    'slope': 0
                })
                pending_action = None

            # 2. Check Signal at CLOSE (for NEXT day execution)
            if position == 'CASH' and signal == 'BUY':
                pending_action = 'BUY'
            elif position == 'HOLD' and signal == 'SELL':
                pending_action = 'SELL'

            # Calculate Equity (at Close)
            if position == 'HOLD':
                current_equity = coin_balance * close_price
            else:
                current_equity = balance
            
            equity_curve.append(current_equity)
            if (i == n_rows - 1) or ((i + 1) % step == 0):
                _emit(25 + int((i + 1) / n_rows * 65), "시뮬레이션")

        df['equity'] = equity_curve
        _emit(92, "지표 계산")
        
        # 4. Calculate Metrics
        final_equity = equity_curve[-1]
        total_return = (final_equity - initial_balance) / initial_balance * 100
        
        # CAGR
        days_total = (df.index[-1] - df.index[0]).days
        cagr = 0
        if days_total > 0:
            cagr = (final_equity / initial_balance) ** (365 / days_total) - 1
        cagr *= 100 # percentage

        # MDD
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100
        mdd = df['drawdown'].min()
        
        # Yearly Stats
        df['daily_return'] = df['equity'].pct_change().fillna(0)
        df['year'] = df.index.year
        
        yearly_ret = df.groupby('year')['daily_return'].apply(lambda x: (1 + x).prod() - 1) * 100
        yearly_mdd = df.groupby('year')['drawdown'].min()
        avg_yearly_mdd = yearly_mdd.mean() if len(yearly_mdd) > 0 else mdd
        yearly_df = pd.DataFrame({
            "Return (%)": yearly_ret,
            "MDD (%)": yearly_mdd
        })

        # Win Rate
        winning_trades = [t for t in trades if t['type'] == 'sell' and t.get('profit', 0) > 0]
        total_sell_trades = [t for t in trades if t['type'] == 'sell']
        win_rate = len(winning_trades) / len(total_sell_trades) * 100 if total_sell_trades else 0
        
        # Sharpe Ratio
        sharpe = 0
        if df['daily_return'].std() != 0:
            sharpe = (df['daily_return'].mean() / df['daily_return'].std()) * np.sqrt(365)

        performance = {
            "initial_balance": initial_balance,
            "final_equity": final_equity,
            "total_return": total_return,
            "cagr": cagr,
            "mdd": mdd,
            "avg_yearly_mdd": avg_yearly_mdd,
            "win_rate": win_rate,
            "trade_count": len(total_sell_trades),
            "sharpe": sharpe,
            "yearly_stats": yearly_df, # DataFrame
            "trades": trades,
            "final_status": position, # HOLD or CASH
            "next_action": pending_action # Action for TOMORROW Open
        }
        _emit(100, "완료")

        return {
            "performance": performance,
            "df": df,
            "equity_curve": equity_curve
        }

    # ================================================================
    # 고속 최적화 엔진: 사전계산 + numpy 벡터화
    # ================================================================
    def _fast_simulate(self, open_arr, close_arr, signal_arr, fee, slippage, initial_balance, year_arr=None):
        """
        numpy 배열 기반 고속 시뮬레이션.
        signal_arr: 1=BUY, -1=SELL, 0=HOLD
        year_arr: 연도 배열 (연간 평균 MDD 계산용, optional)
        Returns: dict with performance metrics
        """
        n = len(open_arr)
        balance = initial_balance
        coin_balance = 0.0
        position = 0  # 0=CASH, 1=HOLD
        pending = 0   # 1=BUY, -1=SELL, 0=None
        buy_price = 0.0
        wins = 0
        sells = 0
        equity = np.empty(n)

        slip_buy = 1 + slippage / 100
        slip_sell = 1 - slippage / 100
        fee_mult = 1 - fee

        for i in range(n):
            op = open_arr[i]
            cl = close_arr[i]
            sig = signal_arr[i]

            # Execute pending
            if pending == 1:
                ep = op * slip_buy
                coin_balance = balance * fee_mult / ep
                balance = 0.0
                position = 1
                buy_price = ep
                pending = 0
            elif pending == -1:
                ep = op * slip_sell
                sell_ret = (ep - buy_price) / buy_price if buy_price > 0 else 0
                balance = coin_balance * ep * fee_mult
                coin_balance = 0.0
                position = 0
                sells += 1
                if sell_ret > 0:
                    wins += 1
                pending = 0

            # Check signal
            if position == 0 and sig == 1:
                pending = 1
            elif position == 1 and sig == -1:
                pending = -1

            # Equity
            equity[i] = coin_balance * cl if position == 1 else balance

        final_eq = equity[-1]
        total_ret = (final_eq - initial_balance) / initial_balance * 100

        # MDD
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        mdd = dd.min()

        # Avg Yearly MDD
        avg_yearly_mdd = mdd
        if year_arr is not None and len(year_arr) == n:
            unique_years = np.unique(year_arr)
            yearly_mdds = [dd[year_arr == yr].min() for yr in unique_years]
            if yearly_mdds:
                avg_yearly_mdd = float(np.mean(yearly_mdds))

        # Win rate
        win_rate = (wins / sells * 100) if sells > 0 else 0

        # Sharpe
        returns = np.diff(equity) / equity[:-1]
        returns = np.nan_to_num(returns)
        std = returns.std()
        sharpe = (returns.mean() / std * np.sqrt(365)) if std > 0 else 0

        # CAGR
        days_total = max(1, n)  # approximate
        cagr = 0
        if final_eq > 0 and initial_balance > 0 and days_total > 1:
            cagr = ((final_eq / initial_balance) ** (365 / days_total) - 1) * 100

        return {
            "total_return": total_ret,
            "cagr": cagr,
            "mdd": mdd,
            "avg_yearly_mdd": avg_yearly_mdd,
            "win_rate": win_rate,
            "trade_count": sells,
            "sharpe": sharpe,
            "final_equity": final_eq
        }

    def optimize_donchian(self, df, buy_range, sell_range, fee=0.0005, slippage=0.0,
                          start_date=None, initial_balance=1000000, progress_callback=None,
                          sell_mode="lower"):
        """
        돈치안 고속 최적화: 모든 rolling을 사전계산.
        Returns: list of result dicts
        """
        df = df.copy()

        # start_date 필터용 원본 인덱스 보존
        full_index = df.index

        # 1. 사전계산: 모든 필요한 rolling max/min
        all_buy_periods = sorted(set(buy_range))
        all_sell_periods = sorted(set(sell_range))

        upper_cache = {}
        for bp in all_buy_periods:
            upper_cache[bp] = df['high'].rolling(window=bp).max().shift(1).values

        lower_cache = {}
        for sp in all_sell_periods:
            lower_cache[sp] = df['low'].rolling(window=sp).min().shift(1).values

        close_arr_full = df['close'].values
        open_arr_full = df['open'].values

        # start_date 필터
        start_idx = 0
        if start_date:
            start_ts = pd.to_datetime(start_date)
            if full_index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(full_index.tz)
            mask = full_index >= start_ts
            start_idx = mask.argmax() if mask.any() else 0

        # 연도 배열 (연간 평균 MDD 계산용)
        year_arr_full = full_index.year.values
        year_arr = year_arr_full[start_idx:]

        # 작업 목록 사전 생성
        tasks = []
        o = open_arr_full[start_idx:]
        c = close_arr_full[start_idx:]
        if len(o) < 5:
            return []

        for bp in buy_range:
            upper = upper_cache[bp]
            for sp in sell_range:
                lower = lower_cache[sp]
                signal_arr = np.zeros(len(close_arr_full), dtype=np.int8)
                buy_mask = close_arr_full > upper
                if sell_mode == "midline":
                    midline = (upper + lower) / 2
                    sell_mask = close_arr_full < midline
                else:
                    sell_mask = close_arr_full < lower
                signal_arr[buy_mask] = 1
                signal_arr[sell_mask] = -1
                s = signal_arr[start_idx:]
                tasks.append((o, c, s, fee, slippage, initial_balance, year_arr,
                              {"Buy Period": bp, "Sell Period": sp}))

        total = len(tasks)
        if total == 0:
            return []

        results = []
        with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as pool:
            futures = {pool.submit(_simulate_task, *t): i for i, t in enumerate(tasks)}
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if progress_callback and done % max(1, total // 50) == 0:
                    results[-1]  # ensure no exception
                    progress_callback(done, total, f"병렬 처리 중 ({_NUM_WORKERS} workers)")
        if progress_callback:
            progress_callback(total, total, "완료")

        return results

    def optimize_sma(self, df, sma_range, fee=0.0005, slippage=0.0,
                     start_date=None, initial_balance=1000000, progress_callback=None):
        """
        SMA 고속 최적화: 모든 SMA를 사전계산.
        Returns: list of result dicts
        """
        df = df.copy()
        full_index = df.index

        # 1. 사전계산: 모든 SMA
        all_periods = sorted(set(sma_range))
        sma_cache = {}
        for p in all_periods:
            sma_cache[p] = df['close'].rolling(window=p).mean().values

        close_arr_full = df['close'].values
        open_arr_full = df['open'].values

        # start_date 필터
        start_idx = 0
        if start_date:
            start_ts = pd.to_datetime(start_date)
            if full_index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(full_index.tz)
            mask = full_index >= start_ts
            start_idx = mask.argmax() if mask.any() else 0

        # 연도 배열 (연간 평균 MDD 계산용)
        year_arr_full = full_index.year.values
        year_arr = year_arr_full[start_idx:]

        o = open_arr_full[start_idx:]
        c = close_arr_full[start_idx:]
        if len(o) < 5:
            return []

        tasks = []
        for p in sma_range:
            sma_vals = sma_cache[p]
            signal_arr = np.zeros(len(close_arr_full), dtype=np.int8)
            valid = ~np.isnan(sma_vals)
            signal_arr[valid & (close_arr_full > sma_vals)] = 1
            signal_arr[valid & (close_arr_full <= sma_vals)] = -1
            s = signal_arr[start_idx:]
            tasks.append((o, c, s, fee, slippage, initial_balance, year_arr,
                          {"SMA Period": p}))

        total = len(tasks)
        if total == 0:
            return []

        results = []
        with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as pool:
            futures = {pool.submit(_simulate_task, *t): i for i, t in enumerate(tasks)}
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if progress_callback and done % max(1, total // 20) == 0:
                    progress_callback(done, total, f"병렬 처리 중 ({_NUM_WORKERS} workers)")
        if progress_callback:
            progress_callback(total, total, "완료")

        return results

    # ================================================================
    # Optuna 베이지안 최적화
    # ================================================================
    def optuna_optimize(self, df, strategy_mode="SMA Strategy",
                        buy_range=(5, 200), sell_range=(5, 100),
                        fee=0.0005, slippage=0.0, start_date=None,
                        initial_balance=1000000, n_trials=100,
                        objective_metric="calmar",
                        progress_callback=None, sell_mode="lower"):
        """
        Optuna TPE 기반 베이지안 최적화.
        objective_metric: "calmar", "sharpe", "return", "mdd"
        sell_mode: "lower" (하단선) or "midline" (중심선) — Donchian only
        Returns: dict with best_params, best_value, trials, study
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        df = df.copy()
        full_index = df.index

        # start_date 필터
        start_idx = 0
        if start_date:
            start_ts = pd.to_datetime(start_date)
            if full_index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(full_index.tz)
            mask = full_index >= start_ts
            start_idx = mask.argmax() if mask.any() else 0

        close_arr_full = df['close'].values
        open_arr_full = df['open'].values

        # 사전계산 캐시
        is_donchian = strategy_mode in ("Donchian", "Donchian Trend")
        if is_donchian:
            upper_cache = {}
            lower_cache = {}
            for bp in range(buy_range[0], buy_range[1] + 1):
                upper_cache[bp] = df['high'].rolling(window=bp).max().shift(1).values
            for sp in range(sell_range[0], sell_range[1] + 1):
                lower_cache[sp] = df['low'].rolling(window=sp).min().shift(1).values
        else:
            sma_cache = {}
            for p in range(buy_range[0], buy_range[1] + 1):
                sma_cache[p] = df['close'].rolling(window=p).mean().values

        # 연도 배열 (연간 평균 MDD 계산용)
        year_arr_full = full_index.year.values
        year_arr = year_arr_full[start_idx:]

        trial_results = []

        def _metric_value(res, calmar):
            if objective_metric == "calmar":
                return calmar
            elif objective_metric == "sharpe":
                return res['sharpe']
            elif objective_metric == "return":
                return res['total_return']
            elif objective_metric == "mdd":
                return res['mdd']  # maximize = least negative
            return calmar

        def objective(trial):
            if is_donchian:
                bp = trial.suggest_int("buy_period", buy_range[0], buy_range[1])
                sp = trial.suggest_int("sell_period", sell_range[0], sell_range[1])

                signal_arr = np.zeros(len(close_arr_full), dtype=np.int8)
                signal_arr[close_arr_full > upper_cache[bp]] = 1
                if sell_mode == "midline":
                    midline = (upper_cache[bp] + lower_cache[sp]) / 2
                    signal_arr[close_arr_full < midline] = -1
                else:
                    signal_arr[close_arr_full < lower_cache[sp]] = -1
            else:
                p = trial.suggest_int("sma_period", buy_range[0], buy_range[1])

                sma_vals = sma_cache[p]
                signal_arr = np.zeros(len(close_arr_full), dtype=np.int8)
                valid = ~np.isnan(sma_vals)
                signal_arr[valid & (close_arr_full > sma_vals)] = 1
                signal_arr[valid & (close_arr_full <= sma_vals)] = -1

            o = open_arr_full[start_idx:]
            c = close_arr_full[start_idx:]
            s = signal_arr[start_idx:]

            if len(o) < 5:
                return float('-inf')

            res = self._fast_simulate(o, c, s, fee, slippage, initial_balance, year_arr=year_arr)
            calmar = abs(res['cagr'] / res['mdd']) if res['mdd'] != 0 else 0

            record = {**res, 'calmar': calmar}
            if is_donchian:
                record['Buy Period'] = bp
                record['Sell Period'] = sp
            else:
                record['sma_period'] = p
            trial_results.append(record)

            if progress_callback:
                val = _metric_value(res, calmar)
                progress_callback(len(trial_results), n_trials,
                                  f"Trial {len(trial_results)}: {objective_metric}={val:.2f}")

            return _metric_value(res, calmar)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": trial_results,
            "study": study,
        }

    # ================================================================
    # 보조 전략: MA 이격도 역추세 분할매수 + 익절
    # ================================================================

    def run_aux_backtest(self, df, main_strategy="Donchian", main_buy_p=20,
                         main_sell_p=10, ma_count=2, ma_short=5, ma_long=20,
                         oversold_threshold=-5.0, tp1_pct=5.0, tp2_pct=10.0,
                         fee=0.0005, slippage=0.0, start_date=None,
                         initial_balance=1000000, split_count=2,
                         buy_seed_mode="equal", pyramid_ratio=1.0,
                         use_rsi_filter=False, rsi_period=2, rsi_threshold=10.0,
                         progress_callback=None, main_df=None):
        """보조 전략 단일 백테스트 실행."""
        from src.strategy.aux_mean_reversion import (
            compute_disparity, compute_rsi, generate_main_position, fast_simulate_aux
        )

        def _emit(unit, msg):
            if progress_callback:
                try:
                    progress_callback(int(unit), 100, msg)
                except Exception:
                    pass

        _emit(5, "데이터 준비")

        df = df.copy()
        full_index = df.index
        close_arr = df['close'].values
        open_arr = df['open'].values
        high_arr = df['high'].values
        low_arr = df['low'].values

        # 메인 전략 포지션 생성 (main_df 지정 시 메인 신호를 별도 시간봉에서 계산 후 정렬)
        _emit(20, "메인 전략 신호 계산")
        if main_df is None:
            main_pos = generate_main_position(
                close_arr, high_arr, low_arr,
                main_strategy, main_buy_p, main_sell_p
            )
        else:
            _m_df = main_df.copy()
            _m_close = _m_df['close'].values
            _m_high = _m_df['high'].values
            _m_low = _m_df['low'].values
            _m_pos = generate_main_position(
                _m_close, _m_high, _m_low,
                main_strategy, main_buy_p, main_sell_p
            )

            _exec_idx = pd.DatetimeIndex(full_index)
            _main_idx = pd.DatetimeIndex(_m_df.index)
            if _exec_idx.tz is not None:
                _exec_idx = _exec_idx.tz_localize(None)
            if _main_idx.tz is not None:
                _main_idx = _main_idx.tz_localize(None)

            _m_pos_s = pd.Series(_m_pos, index=_main_idx).sort_index()
            main_pos = _m_pos_s.reindex(_exec_idx, method="ffill").fillna(0).astype(np.int8).values

        # 이격도 계산 (ma_count=1이면 단기MA 이격도만 사용)
        _emit(35, "이격도 계산")
        try:
            ma_count = int(ma_count)
        except Exception:
            ma_count = 2
        ma_count = 1 if ma_count == 1 else 2
        disp_short = compute_disparity(close_arr, ma_short)
        if ma_count == 1:
            disp_long = disp_short
        else:
            disp_long = compute_disparity(close_arr, ma_long)

        try:
            use_rsi_filter = bool(use_rsi_filter)
        except Exception:
            use_rsi_filter = False
        try:
            rsi_period = max(2, int(rsi_period))
        except Exception:
            rsi_period = 2
        try:
            rsi_threshold = float(rsi_threshold)
        except Exception:
            rsi_threshold = 10.0
        rsi_arr = compute_rsi(close_arr, rsi_period) if use_rsi_filter else None

        # start_date 필터
        start_idx = 0
        if start_date:
            start_ts = pd.to_datetime(start_date)
            if full_index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(full_index.tz)
            mask = full_index >= start_ts
            start_idx = mask.argmax() if mask.any() else 0

        year_arr = full_index.year.values[start_idx:]

        _emit(55, "시뮬레이션")
        res = fast_simulate_aux(
            open_arr[start_idx:], close_arr[start_idx:],
            high_arr[start_idx:], low_arr[start_idx:],
            main_pos[start_idx:],
            disp_short[start_idx:], disp_long[start_idx:],
            oversold_threshold, tp1_pct, tp2_pct,
            fee, slippage, initial_balance, year_arr,
            split_count=split_count,
            buy_seed_mode=buy_seed_mode,
            pyramid_ratio=pyramid_ratio,
            use_rsi_filter=use_rsi_filter,
            rsi_arr=(None if rsi_arr is None else rsi_arr[start_idx:]),
            rsi_threshold=rsi_threshold,
            return_series=True,
        )

        # 차트용 시계열(전략 vs Buy&Hold, DD) 추가
        _emit(85, "차트 데이터 정리")
        _dates = full_index[start_idx:]
        _bench_close = close_arr[start_idx:]
        if len(_dates) > 0 and len(_bench_close) > 0 and _bench_close[0] > 0:
            _bench_return = (_bench_close / _bench_close[0] - 1.0) * 100.0
            _bench_peak = np.maximum.accumulate(_bench_close)
            _bench_dd = (_bench_close - _bench_peak) / _bench_peak * 100.0
        else:
            _bench_return = np.array([])
            _bench_dd = np.array([])

        _equity = res.get("equity_curve")
        if _equity is not None and len(_equity) > 0 and initial_balance > 0:
            _strat_return = (_equity / initial_balance - 1.0) * 100.0
        else:
            _strat_return = np.array([])

        res["dates"] = _dates
        res["strategy_return_curve"] = _strat_return
        res["benchmark_return_curve"] = _bench_return
        res["benchmark_dd_curve"] = _bench_dd
        res["disparity_short_curve"] = disp_short[start_idx:]
        res["disparity_long_curve"] = disp_long[start_idx:]
        res["oversold_threshold"] = float(oversold_threshold)
        res["ma_count"] = ma_count
        res["ma_short"] = int(ma_short)
        res["ma_long"] = int(ma_short if ma_count == 1 else ma_long)
        res["use_rsi_filter"] = bool(use_rsi_filter)
        res["rsi_period"] = int(rsi_period)
        res["rsi_threshold"] = float(rsi_threshold)
        if rsi_arr is not None:
            res["rsi_curve"] = rsi_arr[start_idx:]
        _emit(100, "완료")
        return res

    def optimize_aux(self, df, main_strategy="Donchian", main_buy_p=20,
                     main_sell_p=10, ma_count=2, ma_short_range=(3, 30),
                     ma_long_range=(10, 120), threshold_range=(-15.0, -1.0),
                     tp1_range=(2.0, 10.0), tp2_range=(5.0, 20.0),
                     split_count_range=(1, 5),
                     fee=0.0005, slippage=0.0, start_date=None,
                     initial_balance=1000000, n_trials=100,
                     objective_metric="calmar", progress_callback=None,
                     buy_seed_mode="equal", pyramid_ratio=1.0, main_df=None,
                     min_trade_count=0, optimization_method="optuna",
                     ma_short_step=1, ma_long_step=1,
                     threshold_step=0.5, tp_step=0.5, split_step=1,
                     max_grid_evals=30000,
                     use_rsi_filter=False,
                     rsi_period_range=(2, 2), rsi_threshold_range=(5.0, 10.0),
                     rsi_period_step=1, rsi_threshold_step=0.5):
        """보조 전략 최적화 (Optuna/그리드, 분할 매수 횟수 포함)."""
        from src.strategy.aux_mean_reversion import (
            compute_disparity, compute_rsi, generate_main_position, fast_simulate_aux
        )

        df = df.copy()
        full_index = df.index
        close_arr = df['close'].values
        open_arr = df['open'].values
        high_arr = df['high'].values
        low_arr = df['low'].values

        # 메인 전략 포지션 사전계산 (main_df 지정 시 별도 시간봉 신호 정렬)
        if main_df is None:
            main_pos = generate_main_position(
                close_arr, high_arr, low_arr,
                main_strategy, main_buy_p, main_sell_p
            )
        else:
            _m_df = main_df.copy()
            _m_close = _m_df['close'].values
            _m_high = _m_df['high'].values
            _m_low = _m_df['low'].values
            _m_pos = generate_main_position(
                _m_close, _m_high, _m_low,
                main_strategy, main_buy_p, main_sell_p
            )
            _exec_idx = pd.DatetimeIndex(full_index)
            _main_idx = pd.DatetimeIndex(_m_df.index)
            if _exec_idx.tz is not None:
                _exec_idx = _exec_idx.tz_localize(None)
            if _main_idx.tz is not None:
                _main_idx = _main_idx.tz_localize(None)
            _m_pos_s = pd.Series(_m_pos, index=_main_idx).sort_index()
            main_pos = _m_pos_s.reindex(_exec_idx, method="ffill").fillna(0).astype(np.int8).values

        try:
            ma_count = int(ma_count)
        except Exception:
            ma_count = 2
        ma_count = 1 if ma_count == 1 else 2

        opt_method = str(optimization_method or "optuna").strip().lower()
        if opt_method not in {"optuna", "grid"}:
            opt_method = "optuna"

        try:
            ma_short_step = max(1, int(ma_short_step))
        except Exception:
            ma_short_step = 1
        try:
            ma_long_step = max(1, int(ma_long_step))
        except Exception:
            ma_long_step = 1
        try:
            split_step = max(1, int(split_step))
        except Exception:
            split_step = 1
        try:
            threshold_step = float(threshold_step)
        except Exception:
            threshold_step = 0.5
        if threshold_step <= 0:
            threshold_step = 0.5
        try:
            tp_step = float(tp_step)
        except Exception:
            tp_step = 0.5
        if tp_step <= 0:
            tp_step = 0.5
        try:
            max_grid_evals = max(1, int(max_grid_evals))
        except Exception:
            max_grid_evals = 30000

        try:
            use_rsi_filter = bool(use_rsi_filter)
        except Exception:
            use_rsi_filter = False
        try:
            rsi_period_step = max(1, int(rsi_period_step))
        except Exception:
            rsi_period_step = 1
        try:
            rsi_threshold_step = float(rsi_threshold_step)
        except Exception:
            rsi_threshold_step = 0.5
        if rsi_threshold_step <= 0:
            rsi_threshold_step = 0.5
        try:
            _rsi_pmin = max(2, int(rsi_period_range[0]))
            _rsi_pmax = max(2, int(rsi_period_range[1]))
        except Exception:
            _rsi_pmin, _rsi_pmax = 2, 2
        if _rsi_pmax < _rsi_pmin:
            _rsi_pmin, _rsi_pmax = _rsi_pmax, _rsi_pmin
        rsi_period_range = (int(_rsi_pmin), int(_rsi_pmax))
        try:
            _rsi_tmin = float(rsi_threshold_range[0])
            _rsi_tmax = float(rsi_threshold_range[1])
        except Exception:
            _rsi_tmin, _rsi_tmax = 5.0, 10.0
        if _rsi_tmax < _rsi_tmin:
            _rsi_tmin, _rsi_tmax = _rsi_tmax, _rsi_tmin
        rsi_threshold_range = (float(_rsi_tmin), float(_rsi_tmax))

        # 이격도 사전계산 (모든 기간)
        disp_cache = {}
        all_periods = set(range(ma_short_range[0], ma_short_range[1] + 1))
        if ma_count == 2:
            all_periods |= set(range(ma_long_range[0], ma_long_range[1] + 1))
        for p in all_periods:
            disp_cache[p] = compute_disparity(close_arr, p)

        rsi_cache = {}
        if use_rsi_filter:
            rp_vals = list(range(int(rsi_period_range[0]), int(rsi_period_range[1]) + 1, int(rsi_period_step)))
            if rp_vals[-1] != int(rsi_period_range[1]):
                rp_vals.append(int(rsi_period_range[1]))
            for rp in sorted(set(rp_vals)):
                rsi_cache[int(rp)] = compute_rsi(close_arr, int(rp))

        # start_date 필터
        start_idx = 0
        if start_date:
            start_ts = pd.to_datetime(start_date)
            if full_index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(full_index.tz)
            mask = full_index >= start_ts
            start_idx = mask.argmax() if mask.any() else 0

        year_arr = full_index.year.values[start_idx:]
        o = open_arr[start_idx:]
        c = close_arr[start_idx:]
        h = high_arr[start_idx:]
        l = low_arr[start_idx:]
        mp = main_pos[start_idx:]

        trial_results = []

        def _metric(res, calmar):
            if objective_metric == "calmar":
                return calmar
            if objective_metric == "sharpe":
                return res['sharpe']
            if objective_metric == "return":
                return res['total_return']
            if objective_metric == "mdd":
                return res['mdd']
            return calmar

        def _float_grid_values(vmin, vmax, step):
            vmin = float(vmin)
            vmax = float(vmax)
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            step = max(1e-9, float(step))
            out = []
            cur = vmin
            guard = 0
            while cur <= vmax + 1e-9 and guard < 100000:
                out.append(round(float(cur), 6))
                cur += step
                guard += 1
            if not out or out[-1] < vmax - 1e-9:
                out.append(round(float(vmax), 6))
            return out

        def _evaluate_one(ms, ml, thr, t1, t2, sc, rp=None, rt=None):
            if len(o) < 5:
                return float('-inf'), None

            ms = int(ms)
            ml = int(ml)
            sc = int(sc)
            thr = float(thr)
            t1 = float(t1)
            t2 = float(t2)
            rp = int(rp) if rp is not None else int(rsi_period_range[0])
            rt = float(rt) if rt is not None else float(rsi_threshold_range[0])
            if t2 <= t1:
                t2 = t1 + max(0.1, float(tp_step))

            ds = disp_cache[ms][start_idx:]
            dl = ds if ma_count == 1 else disp_cache[ml][start_idx:]
            rsi_slice = None
            if use_rsi_filter:
                _rsi_full = rsi_cache.get(rp, None)
                if _rsi_full is not None:
                    rsi_slice = _rsi_full[start_idx:]

            res = fast_simulate_aux(
                o, c, h, l, mp, ds, dl,
                thr, t1, t2, fee, slippage,
                initial_balance, year_arr,
                split_count=sc,
                buy_seed_mode=buy_seed_mode,
                pyramid_ratio=pyramid_ratio,
                use_rsi_filter=use_rsi_filter,
                rsi_arr=rsi_slice,
                rsi_threshold=rt,
            )

            calmar = abs(res['cagr'] / res['mdd']) if res['mdd'] != 0 else 0
            trade_count = int(res.get('trade_count', 0))
            meets_trade_filter = trade_count >= int(min_trade_count)
            score = _metric(res, calmar) if meets_trade_filter else -1e9

            record = {
                **res, 'calmar': calmar,
                'MA Count': int(ma_count),
                'MA Short': ms, 'MA Long': ml,
                'Threshold': thr, 'TP1 %': t1, 'TP2 %': t2,
                'Split': sc,
                'Use RSI': bool(use_rsi_filter),
                'RSI Period': int(rp),
                'RSI Threshold': float(rt),
                'Buy Seed Mode': buy_seed_mode,
                'Pyramid Ratio': pyramid_ratio,
                'Min Trades': int(min_trade_count),
                'Trade Filter Pass': bool(meets_trade_filter),
                'optimization_method': opt_method,
                'score': float(score),
            }
            return float(score), record

        if opt_method == "optuna":
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                ms = trial.suggest_int("ma_short", ma_short_range[0], ma_short_range[1])
                if ma_count == 1:
                    ml = ms
                else:
                    ml = trial.suggest_int("ma_long", ma_long_range[0], ma_long_range[1])
                    if ml <= ms:
                        ml = ms + 1
                        if ml > ma_long_range[1]:
                            return float('-inf')

                thr = trial.suggest_float("threshold", threshold_range[0], threshold_range[1], step=threshold_step)
                t1 = trial.suggest_float("tp1_pct", tp1_range[0], tp1_range[1], step=tp_step)
                t2 = trial.suggest_float("tp2_pct", tp2_range[0], tp2_range[1], step=tp_step)
                sc = trial.suggest_int("split_count", split_count_range[0], split_count_range[1], step=split_step)
                if use_rsi_filter:
                    rp = trial.suggest_int("rsi_period", rsi_period_range[0], rsi_period_range[1], step=rsi_period_step)
                    rt = trial.suggest_float("rsi_threshold", rsi_threshold_range[0], rsi_threshold_range[1], step=rsi_threshold_step)
                else:
                    rp = int(rsi_period_range[0])
                    rt = float(rsi_threshold_range[0])

                score, record = _evaluate_one(ms, ml, thr, t1, t2, sc, rp, rt)
                if record is not None:
                    trial_results.append(record)

                if progress_callback:
                    progress_callback(
                        len(trial_results), int(n_trials),
                        f"Trial {len(trial_results)}: {objective_metric}={score:.2f}, trades={int(record.get('trade_count', 0)) if record else 0}"
                    )
                return score

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=int(n_trials))

            best_params = dict(study.best_params)
            if ma_count == 1 and "ma_short" in best_params and "ma_long" not in best_params:
                best_params["ma_long"] = int(best_params["ma_short"])
            if use_rsi_filter:
                if "rsi_period" not in best_params:
                    best_params["rsi_period"] = int(rsi_period_range[0])
                if "rsi_threshold" not in best_params:
                    best_params["rsi_threshold"] = float(rsi_threshold_range[0])

            return {
                "best_params": best_params,
                "best_value": float(study.best_value),
                "trials": trial_results,
                "study": study,
                "optimization_method": opt_method,
                "evaluated_count": len(trial_results),
            }

        # grid 방식
        ms_vals = list(range(int(ma_short_range[0]), int(ma_short_range[1]) + 1, int(ma_short_step)))
        if ms_vals[-1] != int(ma_short_range[1]):
            ms_vals.append(int(ma_short_range[1]))

        if ma_count == 1:
            ml_vals = [None]
        else:
            ml_vals = list(range(int(ma_long_range[0]), int(ma_long_range[1]) + 1, int(ma_long_step)))
            if ml_vals[-1] != int(ma_long_range[1]):
                ml_vals.append(int(ma_long_range[1]))

        thr_vals = _float_grid_values(threshold_range[0], threshold_range[1], threshold_step)
        tp1_vals = _float_grid_values(tp1_range[0], tp1_range[1], tp_step)
        tp2_vals = _float_grid_values(tp2_range[0], tp2_range[1], tp_step)
        split_vals = list(range(int(split_count_range[0]), int(split_count_range[1]) + 1, int(split_step)))
        if split_vals[-1] != int(split_count_range[1]):
            split_vals.append(int(split_count_range[1]))

        if use_rsi_filter:
            rp_vals = list(range(int(rsi_period_range[0]), int(rsi_period_range[1]) + 1, int(rsi_period_step)))
            if rp_vals[-1] != int(rsi_period_range[1]):
                rp_vals.append(int(rsi_period_range[1]))
            rp_vals = sorted(set(int(v) for v in rp_vals))
            rt_vals = _float_grid_values(rsi_threshold_range[0], rsi_threshold_range[1], rsi_threshold_step)
        else:
            rp_vals = [int(rsi_period_range[0])]
            rt_vals = [float(rsi_threshold_range[0])]

        total_est = len(ms_vals) * len(ml_vals) * len(thr_vals) * len(tp1_vals) * len(tp2_vals) * len(split_vals) * len(rp_vals) * len(rt_vals)
        total_target = max(1, min(int(total_est), int(max_grid_evals)))

        best_score = float('-inf')
        best_params = {}

        for ms in ms_vals:
            for ml in ml_vals:
                _ml = int(ms) if ma_count == 1 else int(ml)
                if ma_count == 2 and _ml <= int(ms):
                    continue
                for thr in thr_vals:
                    for t1 in tp1_vals:
                        for t2 in tp2_vals:
                            if float(t2) <= float(t1):
                                continue
                            for sc in split_vals:
                                for rp in rp_vals:
                                    for rt in rt_vals:
                                        if len(trial_results) >= int(max_grid_evals):
                                            break

                                        score, record = _evaluate_one(ms, _ml, thr, t1, t2, sc, rp, rt)
                                        if record is None:
                                            continue
                                        trial_results.append(record)

                                        if score > best_score:
                                            best_score = score
                                            best_params = {
                                                "ma_short": int(ms),
                                                "ma_long": int(_ml),
                                                "threshold": float(thr),
                                                "tp1_pct": float(t1),
                                                "tp2_pct": float(t2),
                                                "split_count": int(sc),
                                            }
                                            if use_rsi_filter:
                                                best_params["rsi_period"] = int(rp)
                                                best_params["rsi_threshold"] = float(rt)

                                        if progress_callback:
                                            progress_callback(
                                                len(trial_results), total_target,
                                                f"Grid {len(trial_results)}: {objective_metric}={score:.2f}, trades={int(record.get('trade_count', 0))}"
                                            )
                                    if len(trial_results) >= int(max_grid_evals):
                                        break
                                if len(trial_results) >= int(max_grid_evals):
                                    break
                            if len(trial_results) >= int(max_grid_evals):
                                break
                        if len(trial_results) >= int(max_grid_evals):
                            break
                    if len(trial_results) >= int(max_grid_evals):
                        break
                if len(trial_results) >= int(max_grid_evals):
                    break
            if len(trial_results) >= int(max_grid_evals):
                break

        return {
            "best_params": best_params,
            "best_value": float(best_score if trial_results else float('-inf')),
            "trials": trial_results,
            "study": None,
            "optimization_method": opt_method,
            "evaluated_count": len(trial_results),
            "total_estimated": int(total_est),
            "max_grid_evals": int(max_grid_evals),
        }
