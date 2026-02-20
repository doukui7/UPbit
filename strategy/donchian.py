import pandas as pd
import numpy as np


class DonchianStrategy:
    """
    Donchian Channel Strategy (Dual Period / Turtle Style)

    sell_mode:
      "lower"   (기본) - close < Lower(sell_period) 이탈 시 매도
      "midline" - close < Midline((Upper+Lower)/2) 이탈 시 매도
    """
    def __init__(self):
        pass

    def create_features(self, df, buy_period=20, sell_period=10):
        """
        Calculate Donchian Channels with separate periods.
        Upper  = buy_period 고가 채널
        Lower  = sell_period 저가 채널
        Middle = (Upper + Lower) / 2  (중심선)
        """
        df = df.copy()

        # Upper Channel (Buy Signal): Max of Highs over buy_period
        df[f'Donchian_Upper_{buy_period}'] = (
            df['high'].rolling(window=buy_period).max().shift(1)
        )

        # Lower Channel (Sell Signal): Min of Lows over sell_period
        df[f'Donchian_Lower_{sell_period}'] = (
            df['low'].rolling(window=sell_period).min().shift(1)
        )

        # Middle Channel (중심선): (Upper + Lower) / 2
        df[f'Donchian_Middle_{buy_period}_{sell_period}'] = (
            (df[f'Donchian_Upper_{buy_period}'] + df[f'Donchian_Lower_{sell_period}']) / 2
        )

        return df

    def get_signal(self, row, buy_period=20, sell_period=10, sell_mode="lower"):
        """
        Generate Buy/Sell signal based on Donchian Breakout.

        sell_mode:
          "lower"   - 하단 채널 이탈 시 매도 (기본)
          "midline" - 중심선 이탈 시 매도 (빠른 청산)
        """
        close = row.get('close')
        upper = row.get(f'Donchian_Upper_{buy_period}')
        lower = row.get(f'Donchian_Lower_{sell_period}')
        middle = row.get(f'Donchian_Middle_{buy_period}_{sell_period}')

        if pd.isna(upper) or pd.isna(lower):
            return 'HOLD'

        # Buy Condition: Breakout Upper
        if close > upper:
            return 'BUY'

        # Sell Condition
        if sell_mode == "midline":
            # 중심선 이탈 시 매도
            if middle is not None and not pd.isna(middle) and close < middle:
                return 'SELL'
        else:
            # 기본: 하단 채널 이탈 시 매도
            if close < lower:
                return 'SELL'

        return 'HOLD'
