import pandas as pd
import numpy as np

class SMAStrategy:
    """
    SMA Strategy engine adapted from GoldenToilet repository patterns.
    """
    def __init__(self):
        pass

    def calculate_rsi(self, close, period=14):
        """
        Calculate RSI (Relative Strength Index)
        Pattern from GoldenToilet: calculate_rsi
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_features(self, df, periods=None):
        """
        Generate technical indicators and features.
        periods: list of integers for SMA calculation (default: [5, 20, 60])
        """
        if df is None or len(df) < 5:
            return df

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        close = df['close']
        
        # Default periods if None
        if periods is None:
            periods = [5, 20, 60]
            
        # Ensure default Golden/Dead cross periods exist for other logic
        defaults = {5, 20, 60}
        calc_periods = defaults.union(set(periods))

        # SMA Calculation and Slope
        for p in calc_periods:
            col_name = f'SMA_{p}'
            df[col_name] = close.rolling(p).mean()
            # Calculate Slope (Current - Previous)
            df[f'{col_name}_Slope'] = df[col_name].diff()

        # SMA Crossovers (Fixed logic for 5/20/60)
        df['SMA_5_20_Cross'] = (df['SMA_5'] > df['SMA_20']).astype(int)
        df['SMA_20_60_Cross'] = (df['SMA_20'] > df['SMA_60']).astype(int)

        # RSI
        df['RSI_14'] = self.calculate_rsi(close, 14)

        # Disparity
        df['Disparity_20'] = (close - df['SMA_20']) / df['SMA_20'] * 100
        df['Disparity_60'] = (close - df['SMA_60']) / df['SMA_60'] * 100

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = macd - signal
        df['MACD_Cross'] = (macd > signal).astype(int)

        # Bollinger Bands
        std20 = close.rolling(20).std()
        df['BB_Upper'] = df['SMA_20'] + 2 * std20
        df['BB_Lower'] = df['SMA_20'] - 2 * std20
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']

        # Volatility
        df['Volatility'] = close.pct_change().rolling(20).std() * np.sqrt(365) * 100 

        return df

    def calculate_crash_risk(self, row):
        """
        Calculate crash risk score (0-7).
        """
        risk_score = 0
        if row.get('RSI_14', 50) < 40: risk_score += 1
        if row.get('Disparity_20', 0) < -3: risk_score += 1
        if row.get('Disparity_60', 0) < -10: risk_score += 1
        if row.get('Volatility', 0) > 60: risk_score += 1
        if row.get('SMA_20_60_Cross', 1) == 0: risk_score += 1
        return risk_score

    def get_signal(self, row, strategy_type='SMA_CROSS', ma_period=20):
        """
        Generate Buy/Sell signal based on strategy.
        ma_period: Period to compare Close price against (for SMA_CROSS)
        """
        # Basic SMA Strategy: Price > SMA_period
        if strategy_type == 'SMA_CROSS':
            sma_val = row.get(f'SMA_{ma_period}')
            close = row.get('close')
            
            if pd.isna(sma_val):
                return 'HOLD'

            base_signal = 'HOLD'
            if close > sma_val:
                return 'BUY'
            else:
                return 'SELL'
            
            return base_signal
        
        # GoldenToilet-like risk management (Fixed to 20/60 logic)
        elif strategy_type == 'GOLDEN_TOILET_RISK':
            risk_score = self.calculate_crash_risk(row)
            if risk_score >= 4: return 'SELL'
            if row.get('SMA_5_20_Cross') == 1: return 'BUY'
            return 'HOLD'
            
        return 'HOLD'
