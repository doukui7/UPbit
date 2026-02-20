import pyupbit
import pandas as pd
import numpy as np
from strategy.sma import SMAStrategy
from strategy.donchian import DonchianStrategy

class BacktestEngine:
    def __init__(self):
        self.strategy = SMAStrategy()

    def run_backtest(self, ticker, period=20, interval="day", count=200, fee=0.0005, start_date=None, initial_balance=1000000, df=None, strategy_mode="SMA Strategy", sell_period_ratio=0.5, slippage=0.0, sell_mode="lower"):
        """
        Run backtest for a given ticker and parameters.
        Execution Logic: Signal on Close(t) -> Trade on Open(t+1)
        slippage: percentage (e.g. 0.1 = 0.1%). Buy price increases, Sell price decreases.
        """
        # 1. Fetch Data
        try:
            if df is None:
                fetch_count = count
                df = pyupbit.get_ohlcv(ticker, interval=interval, count=fetch_count)
                if df is None:
                    return {"error": "Failed to fetch data"}
            else:
                df = df.copy()
        except Exception as e:
            return {"error": str(e)}

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
        
        # Iterate
        for i in range(len(df)):
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

        df['equity'] = equity_curve
        
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
            "win_rate": win_rate,
            "trade_count": len(total_sell_trades),
            "sharpe": sharpe,
            "yearly_stats": yearly_df, # DataFrame
            "trades": trades,
            "final_status": position, # HOLD or CASH
            "next_action": pending_action # Action for TOMORROW Open
        }
        
        return {
            "performance": performance,
            "df": df
        }

    # ================================================================
    # 고속 최적화 엔진: 사전계산 + numpy 벡터화
    # ================================================================
    def _fast_simulate(self, open_arr, close_arr, signal_arr, fee, slippage, initial_balance):
        """
        numpy 배열 기반 고속 시뮬레이션.
        signal_arr: 1=BUY, -1=SELL, 0=HOLD
        Returns: (final_equity, total_return, mdd, win_rate, trade_count, sharpe)
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

        results = []
        total = len(list(buy_range)) * len(list(sell_range))
        idx = 0

        for bp in buy_range:
            upper = upper_cache[bp]
            for sp in sell_range:
                idx += 1
                lower = lower_cache[sp]

                # 벡터화 시그널 생성
                signal_arr = np.zeros(len(close_arr_full), dtype=np.int8)
                buy_mask = close_arr_full > upper
                if sell_mode == "midline":
                    midline = (upper + lower) / 2
                    sell_mask = close_arr_full < midline
                else:  # "lower" (기본)
                    sell_mask = close_arr_full < lower
                signal_arr[buy_mask] = 1
                signal_arr[sell_mask] = -1

                # 슬라이스 (start_date 이후)
                o = open_arr_full[start_idx:]
                c = close_arr_full[start_idx:]
                s = signal_arr[start_idx:]

                if len(o) < 5:
                    continue

                res = self._fast_simulate(o, c, s, fee, slippage, initial_balance)
                res["Buy Period"] = bp
                res["Sell Period"] = sp
                results.append(res)

                if progress_callback:
                    progress_callback(idx, total, f"Buy: {bp}, Sell: {sp}")

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

        results = []
        total = len(list(sma_range))
        idx = 0

        for p in sma_range:
            idx += 1
            sma_vals = sma_cache[p]

            # 벡터화 시그널: close > SMA = BUY, else SELL
            signal_arr = np.zeros(len(close_arr_full), dtype=np.int8)
            valid = ~np.isnan(sma_vals)
            signal_arr[valid & (close_arr_full > sma_vals)] = 1
            signal_arr[valid & (close_arr_full <= sma_vals)] = -1

            o = open_arr_full[start_idx:]
            c = close_arr_full[start_idx:]
            s = signal_arr[start_idx:]

            if len(o) < 5:
                continue

            res = self._fast_simulate(o, c, s, fee, slippage, initial_balance)
            res["SMA Period"] = p
            results.append(res)

            if progress_callback:
                progress_callback(idx, total, f"SMA: {p}")

        return results

    # ================================================================
    # Optuna 베이지안 최적화
    # ================================================================
    def optuna_optimize(self, df, strategy_mode="SMA Strategy",
                        buy_range=(5, 200), sell_range=(5, 100),
                        fee=0.0005, slippage=0.0, start_date=None,
                        initial_balance=1000000, n_trials=100,
                        objective_metric="calmar",
                        progress_callback=None):
        """
        Optuna TPE 기반 베이지안 최적화.
        objective_metric: "calmar", "sharpe", "return", "mdd"
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

            res = self._fast_simulate(o, c, s, fee, slippage, initial_balance)
            calmar = abs(res['cagr'] / res['mdd']) if res['mdd'] != 0 else 0

            record = {**res, 'calmar': calmar}
            if is_donchian:
                record['buy_period'] = bp
                record['sell_period'] = sp
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
