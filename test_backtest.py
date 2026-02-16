from backtest.engine import BacktestEngine
import pandas as pd

def test_backtest():
    engine = BacktestEngine()
    print("Running backtest for KRW-BTC...")
    # Fetch small amount of data for quick test
    result = engine.run_backtest("KRW-BTC", period=20, interval="day", count=100)
    
    if "error" in result:
        print(f"Backtest Failed: {result['error']}")
    else:
        perf = result["performance"]
        print("Backtest Successful!")
        print(f"Total Return: {perf['total_return']:.2f}%")
        print(f"Win Rate: {perf['win_rate']:.2f}%")
        print(f"Trades Count: {perf['trade_count']}")
        print(f"Final Equity: {perf['final_equity']:.2f}")

if __name__ == "__main__":
    test_backtest()
