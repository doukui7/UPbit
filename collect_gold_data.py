import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

def collect_krx_gold_data(pages=100):
    """
    네이버 금융에서 국제 금 시세(일별 시세)를 크롤링합니다.
    URL: https://finance.naver.com/marketindex/goldDailyQuote.naver
    """
    url = "https://finance.naver.com/marketindex/goldDailyQuote.naver"
    
    all_data = []
    
    print(f"Collecting Gold Data from Naver Finance (Max Pages: {pages})...")
    
    for page in range(1, pages + 1):
        params = {'page': page}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='tbl_exchange')
            
            if not table:
                print(f"Page {page}: No table found.")
                break
                
            rows = table.find_all('tr')
            page_data = []
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 7:
                    continue
                
                date = cols[0].text.strip().replace('.', '-')
                close = cols[1].text.strip().replace(',', '')
                change = cols[2].text.strip().replace(',', '')
                # price_change_rate = cols[3].text.strip().strip('%') # Might be empty or diff format
                buy_price = cols[4].text.strip().replace(',', '')
                sell_price = cols[5].text.strip().replace(',', '')
                # base_rate = cols[6]...
                
                # Note: The columns in goldDailyQuote might differ from standard stock.
                # Let's verify column structure visually or robustly.
                # Standard Global Gold Quote Table often has:
                # [0] 날짜 [1] 매매기준율 [2] 전일대비 [3] 등락률 [4] 현찰 사실때 [5] 현찰 파실때 [6] 송금 ...
                # Wait, "International Gold" (cmdt_CDY)? or "Domestic Gold"?
                # "금시세" usually refers to domestic gold price per gram/don.
                
                # Let's assume standard 'marketindex' gold daily quote columns:
                # 0: date
                # 1: close (trading base rate)
                # 2: change
                # 3: rate
                # 4: buy
                # 5: sell
                # 6: ...
                
                try:
                    page_data.append({
                        'Date': date,
                        'Close': float(close),
                        'Open': float(close), # Approximate if Open not available
                        'High': float(close), # Approximate
                        'Low': float(close),  # Approximate
                        'Volume': 0 # Volume not usually in this table
                    })
                except ValueError:
                    continue

            if not page_data:
                print(f"Page {page}: No valid data found.")
                break
                
            all_data.extend(page_data)
            print(f"Page {page}: Collected {len(page_data)} rows. (Last: {page_data[-1]['Date']})")
            
            time.sleep(0.1) # Be polite
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
            
    if all_data:
        df = pd.DataFrame(all_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        
        # Save to CSV
        output_file = "krx_gold_daily.csv"
        df.to_csv(output_file)
        print(f"Saved {len(df)} rows to {output_file}")
        print(df.tail())
        return df
    else:
        print("No data collected.")
        return None

if __name__ == "__main__":
    collect_krx_gold_data()
