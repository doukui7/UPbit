import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 파일 로드 (한글 인코딩 주의)
file_path = "위대리V1.0 ◈K-2X 레버지리◈ (배포_260204)의 - DB.csv"
df = pd.read_csv(file_path, header=None, skiprows=2)

# Column mapping based on preview:
# Col 5 (index 4): Week Start
# Col 6 (index 5): Week End
# Col 7 (index 6): TIGER 미국나스닥100 주종가 (Weekly Close)
# Col 8 (index 7): 지수 추세 (Index Trend in CSV)
# Col 9 (index 8): 시장 평가 (Divergence in CSV)

# 데이터 정리
weekly_data = df.iloc[:, [5, 6, 7]].copy()
weekly_data.columns = ['date_end', 'close', 'csv_trend']

# 날짜 파싱 (10.10.22(금) 형식)
def parse_korean_date(date_str):
    if pd.isna(date_str) or not isinstance(date_str, str):
        return None
    # "10.10.22(금)" -> "2010.10.22"
    clean_date = date_str.split('(')[0].strip()
    try:
        return datetime.strptime("20" + clean_date, "%Y.%m.%d")
    except:
        return None

weekly_data['date'] = weekly_data['date_end'].apply(parse_korean_date)
weekly_data = weekly_data.dropna(subset=['date'])

# 숫자 파싱 (₩10,220 -> 10220)
def parse_currency(val):
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    return float(val.replace('₩', '').replace(',', '').strip())

weekly_data['close'] = df.iloc[:, 6].apply(parse_currency).dropna()
weekly_data['csv_trend'] = df.iloc[:, 7].apply(parse_currency).dropna()

# 2. WDRStrategy 트렌드 계산 로직 재구현
def calc_growth_trend_validation(prices, dates, window=260):
    excel_base = pd.Timestamp("1899-12-31")
    excel_dates = np.array([(d - excel_base).days for d in dates])
    
    trend = np.full(len(prices), np.nan)
    
    # CSV를 보니 처음부터 값이 있음. 
    # 이는 Expanding window일 가능성이 높음 (최소 N개 데이터 쌓일 때까지)
    # 또는 엑셀의 LOGEST 함수 등은 전체 기반일 수도 있음.
    # 하지만 위대리 원장(엑셀)의 260주 로직을 따라가봄.
    
    for i in range(1, len(prices)):
        start_idx = max(0, i - window)
        current_dates = excel_dates[start_idx:i + 1]
        current_prices = prices[start_idx:i + 1]
        
        # 데이터가 2개 이상일 때만 회귀
        if len(current_prices) < 2:
            trend[i] = current_prices[0]
            continue
            
        log_prices = np.log(current_prices)
        slope, intercept = np.polyfit(current_dates, log_prices, 1)
        trend[i] = np.exp(slope * excel_dates[i] + intercept)
        
    return trend

prices = weekly_data['close'].values
dates = weekly_data['date']
my_trend = calc_growth_trend_validation(prices, dates, window=260)

weekly_data['my_trend'] = my_trend

# 3. 결과 비교
comparison = weekly_data[['date', 'close', 'csv_trend', 'my_trend']].copy()
comparison['diff'] = comparison['csv_trend'] - comparison['my_trend']
comparison['diff_pct'] = (comparison['diff'] / comparison['csv_trend']) * 100

print("--- 상위 10개 데이터 비교 ---")
print(comparison.head(10))

print("\n--- 260주 이후 데이터 비교 (Row 300) ---")
print(comparison.iloc[295:305])

# 오차 요약
valid_diff = comparison.dropna(subset=['diff_pct'])
print(f"\n평균 오차 (%): {valid_diff['diff_pct'].abs().mean():.4f}%")
print(f"최대 오차 (%): {valid_diff['diff_pct'].abs().max():.4f}%")
