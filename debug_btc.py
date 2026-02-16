import pyupbit
import pandas as pd
import numpy as np

# Mock Strategy Logic
def get_signal(row, ma_period, long_ma_period):
    price = row['close']
    sma = row.get(f'SMA_{ma_period}')
    slope = row.get(f'SMA_{long_ma_period}_Slope')
    
    if pd.isna(sma) or pd.isna(slope): return 'N/A'
    
    # Base Condition
    base = 'BUY' if price > sma else 'SELL'
    
    # Filter
    if base == 'BUY' and slope < 0:
        return 'HOLD (Filter)'
        
    return base

# Fetch Data
df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=600) # Approx 2 years

# Calc SMAs
periods = [40, 120, 20]
for p in periods:
    df[f'SMA_{p}'] = df['close'].rolling(window=p).mean()
    df[f'SMA_{p}_Slope'] = df[f'SMA_{p}'].diff()

# Analyze specific range
# Look for: Price > SMA40 AND Slope transitions from Neg to Pos
df = df.dropna()

print("Date | Price | SMA40 | Slope120 | Signal (40/120)")
print("-" * 60)

for index, row in df.iterrows():
    # Filter 2021
    if "2021-07-01" <= str(index) <= "2021-09-15":
        sig = get_signal(row, 40, 120)
        slope = row['SMA_120_Slope']
        
        # Print ALL
        print(f"{index} | Pr:{row['close']:,.0f} | S40:{row['SMA_40']:,.0f} | Slp:{slope:10.2f} | {sig}")
