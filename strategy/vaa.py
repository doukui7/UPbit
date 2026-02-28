import numpy as np
import pandas as pd


class VAAStrategy:
    """
    VAA (Vigilant Asset Allocation) 전략.

    Wouter Keller의 VAA 전략을 기반으로 한 공격/방어 모멘텀 전략.

    핵심 원리:
      - 13612W 모멘텀 스코어: (m1*12 + m3*4 + m6*2 + m12*1) / 19
      - 공격자산 4개 중 모멘텀 양수인 것 중 최고 1개 선택
      - 모든 공격자산 모멘텀 음수 → 방어자산 중 최고 모멘텀 1개

    자산 구성 (US 원본):
      - 공격: SPY(미국), EFA(선진국), EEM(이머징), AGG(채권)
      - 방어: LQD(회사채), IEF(중기채), SHY(단기채)

    한국 ETF 대체 (연금저축/ISA):
      - SPY → 379800 KODEX 미국S&P500TR
      - EFA → 195930 TIGER 선진국MSCI World
      - EEM → 295820 TIGER MSCI EM
      - AGG → 305080 TIGER 미국채10년선물
      - LQD → 329750 TIGER 미국달러선물
      - IEF → 305080 TIGER 미국채10년선물
      - SHY → 329750 TIGER 미국달러선물

    리밸런싱: 월 1회 (월말 기준)
    """

    DEFAULT_SETTINGS = {
        'offensive': ['SPY', 'EFA', 'EEM', 'AGG'],
        'defensive': ['LQD', 'IEF', 'SHY'],
        'lookback': 12,
        'top_n': 1,
        'trading_days_per_month': 22,
        'momentum_weights': {'m1': 12.0, 'm3': 4.0, 'm6': 2.0, 'm12': 1.0},
        'kr_etf_map': {
            'SPY': '379800',
            'EFA': '195930',
            'EEM': '295820',
            'AGG': '305080',
            'LQD': '329750',
            'IEF': '305080',
            'SHY': '329750',
        },
    }

    def __init__(self, settings: dict = None):
        self.settings = {**self.DEFAULT_SETTINGS, **(settings or {})}

    @staticmethod
    def _get_close(df: pd.DataFrame) -> pd.Series:
        close_col = "close" if "close" in df.columns else "Close"
        s = df[close_col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna()

    @staticmethod
    def calc_monthly_return(prices: np.ndarray, months: int,
                            trading_days: int = 22) -> float:
        lookback = trading_days * months
        if len(prices) < lookback + 1:
            return 0.0
        current = prices[-1]
        past = prices[-(lookback + 1)]
        if past <= 0:
            return 0.0
        return (current - past) / past

    @staticmethod
    def calc_momentum_score(prices: np.ndarray,
                            trading_days: int = 22,
                            weights: dict | None = None) -> float:
        """13612W 모멘텀 스코어: (m1*12 + m3*4 + m6*2 + m12*1) / 19"""
        m1 = VAAStrategy.calc_monthly_return(prices, 1, trading_days)
        m3 = VAAStrategy.calc_monthly_return(prices, 3, trading_days)
        m6 = VAAStrategy.calc_monthly_return(prices, 6, trading_days)
        m12 = VAAStrategy.calc_monthly_return(prices, 12, trading_days)
        w = weights or {}
        w1 = float(w.get('m1', 12.0))
        w3 = float(w.get('m3', 4.0))
        w6 = float(w.get('m6', 2.0))
        w12 = float(w.get('m12', 1.0))
        denom = (w1 + w3 + w6 + w12) / 4.0
        if denom <= 0:
            denom = 1.0
        return (m1 * w1 + m3 * w3 + m6 * w6 + m12 * w12) / (w1 + w3 + w6 + w12) * 4.0 / 4.0

    def analyze(self, price_data: dict[str, pd.DataFrame]) -> dict | None:
        """VAA 시그널 분석 — 공격자산 모멘텀 양수 최고 vs 방어자산 최고."""
        s = self.settings
        td = s['trading_days_per_month']
        min_days = td * 12 + 5

        all_tickers = list(set(s['offensive'] + s['defensive']))
        for ticker in all_tickers:
            if ticker not in price_data or price_data[ticker] is None:
                return None
            if len(price_data[ticker]) < min_days:
                return None

        # 공격자산 모멘텀 스코어
        off_scores = {}
        for ticker in s['offensive']:
            close = self._get_close(price_data[ticker]).values
            off_scores[ticker] = round(self.calc_momentum_score(
                close, td, s.get('momentum_weights')), 6)

        # 방어자산 모멘텀 스코어
        def_scores = {}
        for ticker in s['defensive']:
            close = self._get_close(price_data[ticker]).values
            def_scores[ticker] = round(self.calc_momentum_score(
                close, td, s.get('momentum_weights')), 6)

        # 양수 모멘텀 공격자산 중 최고 선택
        positive_off = {k: v for k, v in off_scores.items() if v > 0}
        kr_map = s['kr_etf_map']

        if positive_off:
            sorted_off = sorted(positive_off.items(), key=lambda x: x[1], reverse=True)
            top_n = sorted_off[:s['top_n']]
            target_tickers = [t for t, _ in top_n]
            is_offensive = True
            reason_parts = [f"{t}({sc:.4f})" for t, sc in top_n]
            reason = f"공격자산 모멘텀 양수 → {', '.join(reason_parts)} 선택"
        else:
            # 모든 공격자산 음수 → 방어자산 최고 1개
            sorted_def = sorted(def_scores.items(), key=lambda x: x[1], reverse=True)
            target_tickers = [sorted_def[0][0]]
            is_offensive = False
            reason = (f"모든 공격자산 모멘텀 음수 → "
                      f"방어자산 {target_tickers[0]}({sorted_def[0][1]:.4f}) 선택")

        # target_weights 계산
        weight_each = 1.0 / len(target_tickers)
        target_weights = {}
        for t in all_tickers:
            target_weights[t] = 0.0
        for t in target_tickers:
            target_weights[t] = weight_each

        # kr_etf_map 변환
        target_weights_kr = {}
        for us_ticker, w in target_weights.items():
            if w <= 0:
                continue
            code = str(kr_map.get(us_ticker, "")).strip()
            if code:
                target_weights_kr[code] = target_weights_kr.get(code, 0.0) + float(w)

        return {
            'target_tickers': target_tickers,
            'target_kr_codes': [kr_map.get(t, '') for t in target_tickers],
            'is_offensive': is_offensive,
            'offensive_scores': off_scores,
            'defensive_scores': def_scores,
            'target_weights': target_weights,
            'target_weights_kr': target_weights_kr,
            'reason': reason,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        }

    def run_backtest(
        self,
        price_data: dict[str, pd.DataFrame],
        initial_balance: float = 1_000_000.0,
        fee: float = 0.0002,
    ) -> dict | None:
        """월별 리밸런싱 백테스트."""
        s = self.settings
        td = s['trading_days_per_month']
        all_tickers = list(set(s['offensive'] + s['defensive']))

        # 데이터 통합
        merged = None
        for ticker in all_tickers:
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
        current_ticker = None
        current_shares = 0.0
        equity_rows = []
        position_rows = []

        for i in range(12, len(monthly)):
            date = monthly.index[i]

            # 해당 시점까지 데이터로 시그널 계산
            sub_data = {}
            for t in all_tickers:
                sub_data[t] = pd.DataFrame({"close": merged.loc[:date, t]})

            sig = self.analyze(sub_data)
            if not sig:
                continue

            target = sig['target_tickers'][0]
            price_now = float(monthly.loc[date, target])

            # 포지션 변경
            if target != current_ticker:
                if current_ticker and current_shares > 0:
                    sell_price = float(monthly.loc[date, current_ticker])
                    balance = current_shares * sell_price * (1 - fee)
                current_shares = (balance * (1 - fee)) / price_now if price_now > 0 else 0
                balance = 0
                current_ticker = target

            current_eval = current_shares * price_now
            equity_rows.append({"date": date, "equity": current_eval})
            position_rows.append({
                "date": date, "ticker": current_ticker,
                "is_offensive": sig['is_offensive'],
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
        sharpe = 0.0
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float((rets.mean() / rets.std()) * np.sqrt(12))

        return {
            "equity_df": eq,
            "positions": pd.DataFrame(position_rows),
            "metrics": {
                "total_return": total_return,
                "cagr": cagr,
                "mdd": mdd,
                "sharpe": sharpe,
                "final_equity": final_equity,
            },
        }

    def should_rebalance(self, last_rebalance_date: str = None) -> bool:
        today = pd.Timestamp.now()
        if last_rebalance_date:
            last = pd.Timestamp(last_rebalance_date)
            if last.year == today.year and last.month == today.month:
                return False
        return True
