import pandas as pd
import numpy as np

class DonchianStrategy:
    """
    Donchian Channel Strategy (Dual Period / Turtle Style)
    """
    def __init__(self):
        pass

    def create_features(self, df, buy_period=20, sell_period=10):
        """
        Calculate Donchian Channels with separate periods.
        """
        # Avoid modifying original
        df = df.copy()

        # Shift by 1 to use PREVIOUS N days (Traditional Donchian)
        # We want Breakout of HIGH of LAST N days (excluding today)
        
        # Upper Channel (Buy Signal): Max of Highs over buy_period
        df[f'Donchian_Upper_{buy_period}'] = df['high'].rolling(window=buy_period).max().shift(1)
        
        # Lower Channel (Sell Signal): Min of Lows over sell_period
        df[f'Donchian_Lower_{sell_period}'] = df['low'].rolling(window=sell_period).min().shift(1)
        
        # Middle (Optional, for reference)
        # df['Donchian_Middle'] = (df[f'Donchian_Upper_{buy_period}'] + df[f'Donchian_Lower_{sell_period}']) / 2
        
        return df

    def get_signal(self, row, buy_period=20, sell_period=10):
        """
        Generate Buy/Sell signal based on Donchian Breakout.
        """
        close = row.get('close')
        upper = row.get(f'Donchian_Upper_{buy_period}')
        lower = row.get(f'Donchian_Lower_{sell_period}')
        
        if pd.isna(upper) or pd.isna(lower):
            return 'HOLD'
            
        # Buy Condition: Breakout Upper
        if close > upper:
            return 'BUY'
            
        # Sell Condition: Breakdown Lower
        if close < lower:
            return 'SELL'
            
        return 'HOLD'
