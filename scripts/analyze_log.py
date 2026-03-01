import pandas as pd

try:
    df = pd.read_csv('KRW-BTC_2020-01-01_daily_log.csv', index_col=0)
    
    print("Row extracted from CSV:")
    # Check columns
    # print(df.columns)
    
    start_date = "2021-07-01"
    end_date = "2021-09-15"
    
    print(f"Date | Price | SMA_40 | Slope_120 | Signal")
    print("-" * 60)
    
    for index, row in df.iterrows():
        d = str(index)[:10]
        if start_date <= d <= end_date:
            price = row['close']
            # Try to find columns. Might be SMA_40 or SMA_20 etc.
            # I will try to find column starting with SMA_ and NOT ending with Slope, and one Ending with Slope
            sma_val = row.get('SMA_40', 0)
            slope_val = row.get('SMA_120_Slope', 0)
            sig = row.get('signal', 'N/A')
            
            # Print only if meaningful
            print(f"{d} | {price:,.0f} | {sma_val:,.0f} | {slope_val:,.1f} | {sig}")

except Exception as e:
    print(e)
