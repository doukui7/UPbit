import threading
import time
import pandas as pd
import pyupbit
import datetime

class MarketDataWorker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(MarketDataWorker, cls).__new__(cls)
        return cls._instance

    def __init__(self, portfolio_list=None, interval='day'):
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.portfolio_list = portfolio_list if portfolio_list else []
        self.interval = interval
        self.data_map = {} # {ticker: dataframe}
        self.last_update_timestr = "Not Started"
        self.is_running = False
        self.worker_thread = None
        self.initialized = True
        self.status_msg = "Initialized"

    def start_worker(self):
        with self._lock:
            if not self.is_running:
                self.is_running = True
                self.worker_thread = threading.Thread(target=self._job_loop, daemon=True)
                self.worker_thread.start()
                self.status_msg = "Worker Started"

    def update_config(self, portfolio_list):
        """Update the target portfolio dynamically"""
        self.portfolio_list = portfolio_list

    def _job_loop(self):
        while self.is_running:
            self.status_msg = "Fetching Data..."
            if not self.portfolio_list:
                time.sleep(2)
                continue

            # Identify unique (ticker, interval) pairs
            targets = set()
            for item in self.portfolio_list:
                ticker = f"{item['market']}-{item['coin'].upper()}"
                interval = item.get('interval', 'day')
                targets.add((ticker, interval))
            
            for ticker, interval in targets:
                try:
                    # Determine SMA period max to check data length requirements
                    # Simple heuristic: fetch 365 days mostly, or shorter for minutes?
                    # For minute candles, 365 count might be too short for long SMA? 
                    # Upbit limit is 200 per request, but pyupbit handles it.
                    # Default count 200 is often enough for SMA 120, but safety 400.
                    df = pyupbit.get_ohlcv(ticker, interval=interval, count=400)
                    if df is not None:
                        key = f"{ticker}|{interval}"
                        self.data_map[key] = df
                    
                    # Rate limit kindness
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Worker Error {ticker} ({interval}): {e}")
            
            self.last_update_timestr = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_msg = f"Idle (Last: {self.last_update_timestr})"
            
            # Sleep before next cycle
            time.sleep(10) 

    def get_data(self, ticker, interval='day'):
        key = f"{ticker}|{interval}"
        return self.data_map.get(key)
    
    def get_status(self):
        return self.status_msg, self.last_update_timestr
