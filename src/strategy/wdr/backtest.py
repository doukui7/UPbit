import numpy as np
import pandas as pd

class WDRBacktest:
    """
    WDR 전략 백테스트 전용 엔진.
    """
    def __init__(self, core):
        self.core = core

    def run(self, signal_df: pd.DataFrame, trade_df: pd.DataFrame | None = None, initial_balance: float = 10_000_000, start_date=None, fee_rate: float | None = None, initial_stock_ratio: float | None = None, precomputed_merged: pd.DataFrame | None = None) -> dict | None:
        if precomputed_merged is not None and len(precomputed_merged) >= 5:
            merged = precomputed_merged
        else:
            if signal_df is None or len(signal_df) < 60: return None
            trade_df = trade_df if trade_df is not None else signal_df
            w_sig, w_trd = self.core.daily_to_weekly(signal_df), self.core.daily_to_weekly(trade_df)
            trend = self.core.calc_growth_trend(w_sig)
            w_sig["trend"] = trend
            w_sig = w_sig.dropna(subset=["trend"])
            if w_sig.empty: return None
            w_sig["divergence"] = (w_sig["close"] - w_sig["trend"]) / w_sig["trend"] * 100.0
            merged = w_sig[["close", "divergence"]].rename(columns={"close": "signal_close"}).join(w_trd.rename(columns={"close": "trade_close"}), how="inner").dropna(subset=["trade_close"])

        if start_date: merged = merged[merged.index >= pd.Timestamp(start_date)]
        if len(merged) < 5: return None

        comm = float(fee_rate if fee_rate is not None else self.core.settings.get("commission_rate", 0.00015))
        f_prc, f_div = float(merged["trade_close"].iloc[0]), float(merged["divergence"].iloc[0])
        
        # Initial ratio
        if initial_stock_ratio is not None:
            buy_r = max(0.0, min(1.0, float(initial_stock_ratio)))
        else:
            mode = self.core.settings.get("evaluation_mode", 3)
            buy_r = 0.5 if mode == 3 else (0.35 if f_div > 5 else 0.65 if f_div < -6 else 0.5)
        
        shares = int((initial_balance * buy_r) / f_prc)
        cash = initial_balance - shares * f_prc
        
        hist, trades = [], []
        _prc, _sig_c, _divs = merged["trade_close"].values, merged["signal_close"].values, merged["divergence"].values
        _dates = merged.index

        for i in range(len(merged)):
            cur_p = _prc[i]
            pnl = (cur_p - _prc[i-1]) * shares if i > 0 else 0
            act = self.core.get_rebalance_action(pnl, _divs[i], shares, cur_p, cash)
            side, qty = act["action"], act["quantity"]
            
            if side == "SELL" and qty > 0:
                cash += qty * cur_p * (1 - comm)
                shares -= qty
                trades.append({"date": _dates[i], "action": "SELL", "price": cur_p, "qty": qty})
            elif side == "BUY" and qty > 0:
                cost = qty * cur_p * (1 + comm)
                if cost <= cash:
                    cash -= cost
                    shares += qty
                    trades.append({"date": _dates[i], "action": "BUY", "price": cur_p, "qty": qty})

            hist.append({
                "date": _dates[i], "equity": cash + shares * cur_p, 
                "cash": cash, "shares": shares, "price": cur_p, 
                "divergence": _divs[i]
            })

        res_df = pd.DataFrame(hist).set_index("date")
        equity = res_df["equity"].values
        total_ret = (equity[-1] / initial_balance - 1) * 100
        days = (res_df.index[-1] - res_df.index[0]).days
        cagr = ((equity[-1] / initial_balance) ** (365/days) - 1) * 100 if days > 0 else 0
        peak = np.maximum.accumulate(equity)
        mdd = ((equity - peak) / peak * 100).min() if len(equity) > 0 else 0

        # 상세 지표 계산
        sharpe, calmar = 0.0, 0.0
        returns = res_df["equity"].pct_change().dropna()
        if len(returns) > 1:
            std = returns.std()
            if std > 0: sharpe = float((returns.mean() / std) * np.sqrt(52))
        if mdd != 0: calmar = abs(cagr / mdd)

        return {
            "equity_df": res_df, 
            "metrics": {
                "total_return": total_ret, "cagr": cagr, "mdd": mdd, 
                "sharpe": sharpe, "calmar": calmar, "trade_count": len(trades),
                "final_equity": equity[-1] if len(equity) > 0 else initial_balance
            }
        }
