import pyupbit
import pandas as pd
import numpy as np
from strategy.sma import SMAStrategy
from strategy.donchian import DonchianStrategy

class BacktestEngine:
    def __init__(self):
        self.strategy = SMAStrategy()

    def run_backtest(self, ticker, period=20, interval="day", count=200, fee=0.0005, start_date=None, initial_balance=1000000, df=None, strategy_mode="SMA Strategy", sell_period_ratio=0.5):
        """
        Run backtest for a given ticker and parameters.
        Execution Logic: Signal on Close(t) -> Trade on Open(t+1)
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
            df['signal'] = df.apply(lambda row: self.strategy.get_signal(row, buy_period=buy_p, sell_period=sell_p), axis=1)

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
            
            # 1. Execute Pending Action at OPEN
            if pending_action == 'BUY':
                # Buy at Open
                coin_balance = balance * (1 - fee) / open_price
                balance = 0
                position = 'HOLD'
                buy_price = open_price
                
                # Fetch conditions from previous day (Signal Source)
                prev_row = df.iloc[i-1] if i > 0 else row
                p_sma = prev_row.get(f'SMA_{period}', 0)
                
                trades.append({
                    'date': date,
                    'type': 'buy',
                    'price': open_price,
                    'amount': coin_balance,
                    'balance': coin_balance * open_price,
                    'sma': p_sma
                })
                pending_action = None
                
            elif pending_action == 'SELL':
                # Sell at Open
                balance = coin_balance * open_price * (1 - fee)
                coin_balance = 0
                position = 'CASH'
                
                # Fetch conditions from previous day
                prev_row = df.iloc[i-1] if i > 0 else row
                p_sma = prev_row.get(f'SMA_{period}', 0)
                
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': open_price,
                    'amount': 0,
                    'balance': balance,
                    'profit': (open_price - buy_price) / buy_price * 100,
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
