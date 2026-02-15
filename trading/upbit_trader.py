import pyupbit
import pandas as pd
import time
from strategy.sma import SMAStrategy

class UpbitTrader:
    def __init__(self, access_key, secret_key):
        self.access = access_key
        self.secret = secret_key
        self.upbit = pyupbit.Upbit(access_key, secret_key)
        self.strategy = SMAStrategy()

    def get_orders(self, ticker=None, state='wait'):
        """
        Fetch orders.
        ticker: Optional. If provided, fetches orders for that specific ticker.
        state: 'wait', 'done', 'cancel'. Default is 'wait'.
        """
        try:
            if ticker:
                return self.upbit.get_order(ticker, state=state)
            else:
                # pyupbit.get_orders() fetches all orders for a given state, up to 100.
                return self.upbit.get_orders(state=state)
        except Exception as e:
            print(f"Error getting orders: {e}")
            return None

    def get_history(self, kind="deposit", currency="KRW"):
        """
        Fetch account history.
        kind: 'deposit', 'withdraw'
        currency: 'KRW', 'BTC', 'USDT', etc.
        """
        try:
            if kind == 'deposit':
                return self.upbit.get_deposit_list(currency)
            elif kind == 'withdraw':
                return self.upbit.get_withdraw_list(currency)
            else:
                print(f"Invalid history kind: {kind}")
                return []
        except Exception as e:
            print(f"Error fetching {kind} history ({currency}): {e}")
            return []

    def get_balance(self, ticker="KRW"):
        """
        Get balance of specific ticker.
        """
        try:
            return self.upbit.get_balance(ticker)
        except Exception as e:
            print(f"Error getting balance: {e}")
            return 0

    def get_current_price(self, ticker):
        """
        Get current price of ticker.
        """
        try:
            return pyupbit.get_current_price(ticker)
        except Exception as e:
            print(f"Error getting price: {e}")
            return None

    def buy_market(self, ticker, price_amount):
        """
        Buy coin at market price.
        price_amount: Amount in KRW to buy.
        """
        try:
            return self.upbit.buy_market_order(ticker, price_amount)
        except Exception as e:
            return {"error": str(e)}

    def sell_market(self, ticker, volume):
        """
        Sell coin at market price.
        volume: Amount of coin to sell.
        """
        try:
            return self.upbit.sell_market_order(ticker, volume)
        except Exception as e:
            today_date = None # Not used in simple check

    def check_and_trade(self, ticker, interval="day", sma_period=20):
        """
        Check signal and execute trade.
        Returns trade result or status message.
        """
        # 1. Get Data
        # Ensure we fetch enough data for long_period
        count = 200
        if long_period and long_period > 200:
             count = long_period + 50
             
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
        if df is None:
            return "Failed to fetch data"

        # 2. Analyze
        calc_periods = [sma_period]
        if long_period:
            calc_periods.append(long_period)
            
        df = self.strategy.create_features(df, periods=calc_periods)
        last_row = df.iloc[-1]
        
        # Using row[-2] (previous closed candle) is safer for backtest logic consistency,
        # but for live trading, we might want to react to 'current' signal or strict 'close' signal.
        # GoldenToilet usually uses 'current' row for indicators but makes decision based on logic.
        # Let's use the standard approach: Check signal on the just-closed candle or current developing candle?
        # Standard: Use *previous* completed candle to avoid repainting.
        # However, for simplicity here, we'll check the *last complete* row if data is historical, 
        # but get_ohlcv includes current incomplete candle as last row?
        # pyupbit get_ohlcv includes current candle as last row.
        # So df.iloc[-2] is the last *completed* candle.
        
        previous_row = df.iloc[-2] # Last completed candle
        current_signal = self.strategy.get_signal(previous_row, strategy_type='SMA_CROSS', ma_period=sma_period, long_ma_period=long_period)
        
        # 3. Check Position
        krw_balance = self.get_balance("KRW")
        coin_currency = ticker.split('-')[1] # KRW-BTC -> BTC
        coin_balance = self.get_balance(coin_currency)
        current_price = self.get_current_price(ticker)
        
        if current_price is None:
            return "Failed to get current price"

        min_order_amount = 5000 # KRW

        # Logic
        if current_signal == 'BUY':
            # Buy if we have KRW and don't hold much coin? 
            # Simple logic: If we have > 5000 KRW, buy.
            if krw_balance > min_order_amount:
                # Buy 99% of balance to account for fees
                amount_to_buy = krw_balance * 0.99
                result = self.buy_market(ticker, amount_to_buy)
                return f"BUY EXECUTED: {result}"
            else:
                return "BUY SIGNAL but Insufficient KRW"
        
        elif current_signal == 'SELL':
            # Sell if we have coin value > 5000 KRW
            coin_value = coin_balance * current_price
            if coin_value > min_order_amount:
                result = self.sell_market(ticker, coin_balance)
                return f"SELL EXECUTED: {result}"
            else:
                return "SELL SIGNAL but Insufficient Coin"
        
        return f"HOLD (Signal: {current_signal})"
