import numpy as np
import pandas as pd

class WDRCore:
    """
    WDR (위대리) 전략 핵심 로직 (3단계/5단계).
    """
    DEFAULT_SETTINGS_3TIER = {
        'evaluation_mode': 3, 'overvalue_threshold': 5.0, 'undervalue_threshold': -6.0,
        'sell_ratio_overvalue': 100.0, 'sell_ratio_neutral': 66.7, 'sell_ratio_undervalue': 60.0,
        'buy_ratio_overvalue': 66.7, 'buy_ratio_neutral': 66.7, 'buy_ratio_undervalue': 120.0,
        'trend_period_weeks': 260, 'initial_cash_ratio': 0.16, 'min_cash_ratio': 0.10,
        'min_stock_ratio': 0.10, 'commission_rate': 0.00015,
    }

    DEFAULT_SETTINGS_5TIER = {
        'evaluation_mode': 5, 'super_overvalue_threshold': 10.0, 'overvalue_threshold': 5.0,
        'undervalue_threshold': -6.0, 'super_undervalue_threshold': -10.0,
        'sell_ratio_super_overvalue': 150.0, 'sell_ratio_overvalue': 100.0, 'sell_ratio_neutral': 66.7,
        'sell_ratio_undervalue': 60.0, 'sell_ratio_super_undervalue': 33.0,
        'buy_ratio_super_overvalue': 33.0, 'buy_ratio_overvalue': 66.7, 'buy_ratio_neutral': 66.7,
        'buy_ratio_undervalue': 120.0, 'buy_ratio_super_undervalue': 200.0,
        'trend_period_weeks': 260, 'initial_cash_ratio': 0.16, 'min_cash_ratio': 0.10,
        'min_stock_ratio': 0.10, 'commission_rate': 0.00015,
    }

    def __init__(self, settings: dict = None, evaluation_mode: int = None):
        mode = evaluation_mode or (settings.get('evaluation_mode') if settings else 3)
        base = self.DEFAULT_SETTINGS_5TIER if mode == 5 else self.DEFAULT_SETTINGS_3TIER
        self.settings = {**base, **(settings or {})}
        self.settings['evaluation_mode'] = mode

    @staticmethod
    def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        close_col = 'close' if 'close' in df.columns else 'Close'
        weekly = df[close_col].resample('W-FRI').last().dropna()
        return pd.DataFrame({'close': weekly.values}, index=weekly.index)

    def calc_growth_trend(self, weekly_df: pd.DataFrame) -> np.ndarray:
        prices = weekly_df['close'].values.astype(np.float64)
        n = len(prices)
        window = int(self.settings['trend_period_weeks'])
        x = np.arange(n, dtype=np.float64)
        y = np.log(prices)
        sx_exp, sy_exp, sx2_exp, sxy_exp = np.cumsum(x), np.cumsum(y), np.cumsum(x**2), np.cumsum(x*y)
        w_exp = np.arange(1, n + 1, dtype=np.float64)

        def rolling_sum(arr, win):
            res = arr.copy()
            res[win:] = arr[win:] - arr[:-win]
            return res

        sx, sy, sx2, sxy = rolling_sum(sx_exp, window), rolling_sum(sy_exp, window), rolling_sum(sx2_exp, window), rolling_sum(sxy_exp, window)
        w = np.where(w_exp < window, w_exp, float(window))
        denom = np.where(np.abs(w*sx2 - sx*sx) < 1e-12, 1e-12, w*sx2 - sx*sx)
        slope = (w*sxy - sx*sy) / denom
        intercept = (sy - slope*sx) / w
        return np.exp(slope * x + intercept)

    def calc_divergence(self, current_price: float, trend_value: float) -> float:
        if trend_value <= 0:
            return 0.0
        return (current_price - trend_value) / trend_value * 100.0

    def get_market_state(self, divergence: float) -> tuple:
        s = self.settings
        if s.get('evaluation_mode') == 5:
            if divergence > s.get('super_overvalue_threshold', 10.0): return ('SUPER_OVERVALUE', s['sell_ratio_super_overvalue'], s['buy_ratio_super_overvalue'])
            if divergence > s['overvalue_threshold']: return ('OVERVALUE', s['sell_ratio_overvalue'], s['buy_ratio_overvalue'])
            if divergence < s.get('super_undervalue_threshold', -10.0): return ('SUPER_UNDERVALUE', s['sell_ratio_super_undervalue'], s['buy_ratio_super_undervalue'])
            if divergence < s['undervalue_threshold']: return ('UNDERVALUE', s['sell_ratio_undervalue'], s['buy_ratio_undervalue'])
            return ('NEUTRAL', s['sell_ratio_neutral'], s['buy_ratio_neutral'])
        if divergence > s['overvalue_threshold']: return ('OVERVALUE', s['sell_ratio_overvalue'], s['buy_ratio_overvalue'])
        if divergence < s['undervalue_threshold']: return ('UNDERVALUE', s['sell_ratio_undervalue'], s['buy_ratio_undervalue'])
        return ('NEUTRAL', s['sell_ratio_neutral'], s['buy_ratio_neutral'])

    def get_rebalance_action(self, weekly_pnl: float, divergence: float, current_shares: int, current_price: float, cash: float) -> dict:
        state, sell_ratio, buy_ratio = self.get_market_state(divergence)
        total_value = cash + current_shares * current_price
        action, quantity = None, 0
        min_c, min_s = self.settings['min_cash_ratio'], self.settings['min_stock_ratio']

        if weekly_pnl > 0 and current_shares > 0:
            quantity = int((weekly_pnl * (sell_ratio / 100.0)) / current_price)
            if quantity > 0:
                max_sell = current_shares - max(1, int(total_value * min_s / current_price))
                quantity = max(0, min(quantity, max_sell, current_shares))
            if quantity > 0: action = 'SELL'
        elif weekly_pnl < 0:
            buy_amt = min(abs(weekly_pnl) * (buy_ratio / 100.0), cash - (total_value * min_c), (total_value * self.settings.get('max_stock_ratio', 1.0)) - (current_shares * current_price))
            quantity = int(buy_amt / current_price)
            if quantity > 0: action = 'BUY'
        
        return {'action': action, 'quantity': quantity, 'state': state, 'divergence': round(divergence, 2), 'sell_ratio': sell_ratio, 'buy_ratio': buy_ratio, 'weekly_pnl': round(weekly_pnl, 0)}
