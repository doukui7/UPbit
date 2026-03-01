import os
import sys
from dotenv import load_dotenv
from kis_trader import KISTrader

# Force utf-8 encoding for stdout if possible
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception:
        pass

def test_connection():
    load_dotenv()
    print("--- KIS Connection Test ---")
    
    # Initialize trader (using real account by default)
    trader = KISTrader(is_mock=False)
    
    print("[1] Attempting authentication...")
    if trader.auth():
        print("[OK] Authentication successful!")
        
        print("\n[2] Attempting to fetch current price (TIGER 200 - 102110)...")
        price = trader.get_current_price("102110")
        if price:
            print(f"[OK] Price retrieved: {price:,.0f} KRW")
        else:
            print("[FAIL] Price retrieval failed.")
            
        print("\n[3] Checking account balance...")
        balance = trader.get_balance()
        if balance:
            print("[OK] Balance retrieved successfully.")
            print(f"   - Cash: {balance.get('cash', 0):,.0f} KRW")
        else:
            print("[FAIL] Balance retrieval failed (Check if KIS_ACCOUNT_NO is set in .env)")
    else:
        print("[FAIL] Authentication failed. Please check KIS_APP_KEY and KIS_APP_SECRET in .env")

if __name__ == "__main__":
    test_connection()
