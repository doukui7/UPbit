import pandas as pd
import numpy as np
from strategy.sma import SMAStrategy

# Setup Strategy
strategy = SMAStrategy()

# Mock Data
# Price > ShortSMA (Condition 1 Met)
# Long SMA: Falling -> Rising (Condition 2 Transitions)
data = {
    'close': [100, 100, 100, 100, 100, 100],
    'SMA_5': [90, 90, 90, 90, 90, 90],  # Price (100) > SMA_5 (90) ALWAYS
    'SMA_20': [200, 199, 198, 198, 199, 200] # Falling -> Flat -> Rising
}
df = pd.DataFrame(data)

# Calculate Slope manually to match strategy logic
# Strategy does: df[f'SMA_{p}_Slope'] = df[f'SMA_{p}'].diff()
# We need to simulate 'create_features' or just add columns manually.
df['SMA_20_Slope'] = df['SMA_20'].diff()

print("Row | Price | ShortSMA | LongSMA | Slope | Signal")
print("-" * 55)

for i in range(1, len(df)):
    row = df.iloc[i]
    # Check signal with Short=5, Long=20
    signal = strategy.get_signal(row, strategy_type='SMA_CROSS', ma_period=5, long_ma_period=20)
    
    slope_val = row['SMA_20_Slope']
    print(f"{i:3} | {row['close']:5} | {row['SMA_5']:8} | {row['SMA_20']:7} | {slope_val:5.1f} | {signal}")
