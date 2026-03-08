import pandas as pd
from .wdr.core import WDRCore
from .wdr.backtest import WDRBacktest

class WDRStrategy:
    """
    WDR (위대리) 3-Stage Strategy - ISA 계좌용 (Wrapper)
    기존 코드와의 하위 호환성을 유지하면서 내부 로직을 분리함.
    """
    def __init__(self, settings: dict = None, evaluation_mode: int = None):
        self.core = WDRCore(settings, evaluation_mode)
        self.settings = self.core.settings
        self.backtest_engine = WDRBacktest(self.core)

    def analyze(self, df: pd.DataFrame) -> dict | None:
        """현재 시점에서 이격도 및 시장 상태 분석 (자동매매용)."""
        if df is None or len(df) < 260: return None
        weekly = self.core.daily_to_weekly(df)
        trend = self.core.calc_growth_trend(weekly)
        if len(trend) == 0: return None
        
        last_p = weekly['close'].iloc[-1]
        last_t = trend[-1]
        div = self.core.calc_divergence(last_p, last_t)
        state, sell_r, buy_r = self.core.get_market_state(div)
        
        return {
            'divergence': round(div, 2), 'state': state,
            'sell_ratio': sell_r, 'buy_ratio': buy_r,
            'qqq_price': round(last_p, 2), 'trend': round(last_t, 2),
            'date': weekly.index[-1].strftime('%Y-%m-%d'),
        }

    def daily_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.core.daily_to_weekly(df)

    def calc_growth_trend(self, weekly_df: pd.DataFrame):
        return self.core.calc_growth_trend(weekly_df)

    def get_rebalance_action(self, weekly_pnl: float, divergence: float, current_shares: int, current_price: float, cash: float) -> dict:
        return self.core.get_rebalance_action(weekly_pnl, divergence, current_shares, current_price, cash)

    def run_backtest(self, signal_df=None, trade_df=None, initial_balance=10_000_000, 
                     start_date=None, fee_rate=None, initial_stock_ratio=None, 
                     precomputed_merged=None, signal_daily_df=None, trade_daily_df=None):
        """백테스트 실행."""
        sig = signal_daily_df if signal_daily_df is not None else signal_df
        trd = trade_daily_df if trade_daily_df is not None else trade_df
        return self.backtest_engine.run(sig, trd, initial_balance, 
                                        start_date, fee_rate, initial_stock_ratio, 
                                        precomputed_merged)

    def backtest(self, df, initial_balance=10_000_000):
        """기존 UI 호환 래퍼."""
        res = self.run_backtest(df, initial_balance=initial_balance)
        if not res: return None
        m = res["metrics"]
        return {
            "equity_df": res["equity_df"], "total_return": m["total_return"],
            "cagr": m["cagr"], "mdd": m["mdd"], "trade_count": m["trade_count"],
        }


def _wdr_opt_task(signal_df, trade_df, settings_override: dict, eval_mode: int,
                  cap: float, start_date: str, fee_rate: float,
                  initial_stock_ratio: float) -> dict | None:
    """ProcessPool 워커용 WDR 단일 조합 백테스트.
    settings_override: overvalue_threshold, undervalue_threshold,
                       sell_ratio_*, buy_ratio_* 등을 포함하는 dict.
    """
    strat = WDRStrategy(settings=settings_override, evaluation_mode=eval_mode)
    bt = strat.run_backtest(
        signal_df=signal_df,
        trade_df=trade_df,
        initial_balance=cap,
        start_date=start_date,
        fee_rate=fee_rate,
        initial_stock_ratio=initial_stock_ratio / 100.0 if initial_stock_ratio > 0 else None,
    )
    if bt and bt.get("metrics"):
        m = bt["metrics"]
        _key_map = {
            "overvalue_threshold": "고평가(%)",
            "undervalue_threshold": "저평가(%)",
            "sell_ratio_overvalue": "매도_고평가",
            "sell_ratio_neutral": "매도_중립",
            "sell_ratio_undervalue": "매도_저평가",
            "buy_ratio_overvalue": "매수_고평가",
            "buy_ratio_neutral": "매수_중립",
            "buy_ratio_undervalue": "매수_저평가",
        }
        result = {
            "고평가(%)": settings_override.get("overvalue_threshold"),
            "저평가(%)": settings_override.get("undervalue_threshold"),
            "초기비중(%)": initial_stock_ratio,
        }
        # 비율 파라미터가 있으면 추가
        for eng_key, kor_key in _key_map.items():
            if eng_key in settings_override and kor_key not in result:
                result[kor_key] = settings_override[eng_key]
        result.update({
            "CAGR(%)": round(m["cagr"], 2),
            "MDD(%)": round(m["mdd"], 2),
            "Calmar": round(m.get("calmar", 0), 3),
            "Sharpe": round(m.get("sharpe", 0), 3),
            "수익률(%)": round(m["total_return"], 2),
            "최종자산": round(m.get("final_equity", 0), 0),
        })
        return result
    return None
