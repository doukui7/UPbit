import numpy as np
import pandas as pd


class LAAStrategy:
    """
    LAA (Lethargic Asset Allocation) 전략.

    핵심 자산(기본 75%):
      - IWD 25%
      - GLD 25%
      - IEF 25%
    리스크 자산(기본 25%):
      - SPY가 200일선 위면 QQQ
      - 아니면 SHY
    """

    DEFAULT_SETTINGS = {
        "risk_signal_ticker": "SPY",
        "risk_signal_sma_days": 200,
        "core_assets": ["IWD", "GLD", "IEF"],
        "risk_on_asset": "QQQ",
        "risk_off_asset": "SHY",
        "core_weight": 0.75,
        "risk_weight": 0.25,
        "kr_etf_map": {
            "IWD": "360750",   # 대체: S&P500
            "GLD": "411060",   # ACE KRX금현물
            "IEF": "453540",   # TIGER 미국채10년선물
            "QQQ": "133690",   # TIGER 미국나스닥100
            "SHY": "114470",   # KODEX 국고채3년
        },
    }

    def __init__(self, settings: dict | None = None):
        self.settings = {**self.DEFAULT_SETTINGS, **(settings or {})}

    @staticmethod
    def _get_close(df: pd.DataFrame) -> pd.Series:
        close_col = "close" if "close" in df.columns else "Close"
        s = df[close_col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna()

    def _compute_risk_on(self, price_data: dict[str, pd.DataFrame]) -> tuple[bool, float, float]:
        signal_ticker = self.settings["risk_signal_ticker"]
        sma_days = int(self.settings["risk_signal_sma_days"])
        spy_close = self._get_close(price_data[signal_ticker])
        last_price = float(spy_close.iloc[-1])

        if len(spy_close) >= sma_days:
            sma = float(spy_close.tail(sma_days).mean())
        else:
            # 데이터가 부족하면 보수적으로 사용 가능한 구간 평균 사용
            sma = float(spy_close.mean())

        return last_price >= sma, last_price, sma

    def analyze(self, price_data: dict[str, pd.DataFrame]) -> dict | None:
        s = self.settings
        required = set(s["core_assets"] + [s["risk_on_asset"], s["risk_off_asset"], s["risk_signal_ticker"]])

        for ticker in required:
            if ticker not in price_data or price_data[ticker] is None:
                return None
            close = self._get_close(price_data[ticker])
            if close.empty:
                return None

        risk_on, signal_price, signal_sma = self._compute_risk_on(price_data)
        selected_risk_asset = s["risk_on_asset"] if risk_on else s["risk_off_asset"]

        core_assets = list(s["core_assets"])
        core_each = float(s["core_weight"]) / max(len(core_assets), 1)
        target_weights = {a: 0.0 for a in set(core_assets + [s["risk_on_asset"], s["risk_off_asset"]])}
        for a in core_assets:
            target_weights[a] = core_each
        target_weights[selected_risk_asset] = float(s["risk_weight"])

        # 합계 1.0 정규화
        total_w = sum(target_weights.values())
        if total_w > 0:
            target_weights = {k: v / total_w for k, v in target_weights.items()}

        kr_map = s["kr_etf_map"]
        target_weights_kr = {}
        for us_ticker, w in target_weights.items():
            code = str(kr_map.get(us_ticker, "")).strip()
            if code:
                target_weights_kr[code] = target_weights_kr.get(code, 0.0) + float(w)

        reason = (
            f"SPY 종가({signal_price:.2f})가 {s['risk_signal_sma_days']}일선({signal_sma:.2f}) "
            f"{'상단' if risk_on else '하단'}이므로 리스크 자산을 {selected_risk_asset}로 배치"
        )

        return {
            "risk_on": risk_on,
            "selected_risk_asset": selected_risk_asset,
            "selected_risk_kr_code": kr_map.get(selected_risk_asset, ""),
            "target_weights": target_weights,
            "target_weights_kr": target_weights_kr,
            "signal_price": signal_price,
            "signal_sma": signal_sma,
            "reason": reason,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        }

    def run_backtest(
        self,
        price_data: dict[str, pd.DataFrame],
        initial_balance: float = 1_000_000.0,
        fee: float = 0.0002,
    ) -> dict | None:
        s = self.settings
        assets = list(set(s["core_assets"] + [s["risk_on_asset"], s["risk_off_asset"], s["risk_signal_ticker"]]))

        merged = None
        for ticker in assets:
            if ticker not in price_data or price_data[ticker] is None:
                return None
            ser = self._get_close(price_data[ticker]).rename(ticker)
            if merged is None:
                merged = ser.to_frame()
            else:
                merged = merged.join(ser, how="inner")

        if merged is None or merged.empty:
            return None

        monthly = merged.resample("ME").last().dropna()
        if len(monthly) < 14:
            return None

        balance = float(initial_balance)
        equity_rows = []
        alloc_rows = []
        current_weights = None

        start_idx = max(12, int(np.ceil(s["risk_signal_sma_days"] / 22)))
        for i in range(start_idx, len(monthly)):
            cur_date = monthly.index[i]
            prev_date = monthly.index[i - 1]

            # 직전 월 가중치로 이번 달 수익 반영
            if current_weights is not None:
                r = 0.0
                for a, w in current_weights.items():
                    p0 = float(monthly.loc[prev_date, a])
                    p1 = float(monthly.loc[cur_date, a])
                    if p0 > 0:
                        r += float(w) * ((p1 / p0) - 1.0)
                balance *= (1.0 + r)
                balance *= (1.0 - fee)

            # 이번 월말 기준으로 다음 달 가중치 결정
            sub_data = {}
            for a in assets:
                sub_data[a] = pd.DataFrame({"close": merged.loc[:cur_date, a]})
            sig = self.analyze(sub_data)
            if not sig:
                continue

            current_weights = sig["target_weights"]
            equity_rows.append({"date": cur_date, "equity": balance})
            alloc_rows.append({
                "date": cur_date,
                "risk_on": sig["risk_on"],
                "risk_asset": sig["selected_risk_asset"],
            })

        if not equity_rows:
            return None

        eq = pd.DataFrame(equity_rows).set_index("date")
        eq["peak"] = eq["equity"].cummax()
        eq["drawdown"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100.0

        final_equity = float(eq["equity"].iloc[-1])
        total_return = (final_equity / initial_balance - 1.0) * 100.0
        days = int((eq.index[-1] - eq.index[0]).days)
        cagr = ((final_equity / initial_balance) ** (365.0 / max(days, 1)) - 1.0) * 100.0
        mdd = float(eq["drawdown"].min())

        rets = eq["equity"].pct_change().dropna()
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(12)) if len(rets) > 1 and rets.std() > 0 else 0.0

        return {
            "equity_df": eq,
            "allocations": pd.DataFrame(alloc_rows),
            "metrics": {
                "total_return": total_return,
                "cagr": cagr,
                "mdd": mdd,
                "sharpe": sharpe,
                "final_equity": final_equity,
            },
        }
