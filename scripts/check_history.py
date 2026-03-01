import os
from dotenv import load_dotenv
import pyupbit
import pprint

load_dotenv()
ak = os.getenv("UPBIT_ACCESS_KEY")
sk = os.getenv("UPBIT_SECRET_KEY")

if not ak or not sk:
    print("Keys missing in .env")
    # Try to load from UpbitTrader logic? No, just rely on .env
    exit()

upbit = pyupbit.Upbit(ak, sk)

print("--- Help for get_deposit_list ---")
try:
    help(upbit.get_deposit_list)
except:
    pass

currencies_to_check = ["KRW", "USDT"]

for curr in currencies_to_check:
    print(f"\n=== Checking {curr} ===")
    
    print(f"--- Deposits ({curr}) ---")
    try:
        d = upbit.get_deposit_list(curr)
        if d:
            print(f"Found {len(d)} records.")
            pprint.pprint(d[0])
        else:
            print("No records.")
    except Exception as e:
        print(f"Error: {e}")

    print(f"--- Withdrawals ({curr}) ---")
    try:
        w = upbit.get_withdraw_list(curr)
        if w:
            print(f"Found {len(w)} records.")
            pprint.pprint(w[0])
        else:
            print("No records.")
    except Exception as e:
        print(f"Error: {e}")




