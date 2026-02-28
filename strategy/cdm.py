import numpy as np
import pandas as pd


class CDMStrategy:
    """
    CDM (Composite Dual Momentum) 전략.

    4모듈 듀얼모멘텀 전략.

    핵심 원리:
      - 공격자산 8개를 2개씩 4모듈로 구성
      - 각 모듈에서 12개월 수익률 기준 상대모멘텀으로 승자 선택
      - 승자의 12개월 수익률 > 방어자산 수익률 → 승자에 투자 (절대모멘텀)
      - 그렇지 않으면 → 방어자산에 투자
      - 각 모듈 25% 균등 배분 (총 100%)

    모듈 구성 (US 원본):
      Module 1: SPY vs VEU (미국 vs 해외)
      Module 2: VNQ vs REM (부동산 vs 리츠)
      Module 3: LQD vs HYG (회사채 vs 하이일드)
      Module 4: TLT vs GLD (장기채 vs 금)
      방어자산:  BIL (초단기채)

    한국 ETF 대체 (연금저축/ISA):
      - SPY → 379800 KODEX 미국S&P500TR
      - VEU → 195930 TIGER 선진국MSCI World
      - VNQ → 352560 TIGER 리츠부동산인프라
      - REM → 352560 TIGER 리츠부동산인프라 (중복)
      - LQD → 305080 TIGER 미국채10년선물
      - HYG → 305080 TIGER 미국채10년선물 (중복)
      - TLT → 304660 KODEX 미국채울트라30년선물(H)
      - GLD → 132030 KODEX 골드선물(H)
      - BIL → 329750 TIGER 미국달러선물

    리밸런싱: 월 1회 (월말 기준)
    """

    DEFAULT_SETTINGS = {
        'offensive': ['SPY', 'VEU', 'VNQ', 'REM', 'LQD', 'HYG', 'TLT', 'GLD'],
        'defensive': ['BIL'],
        'lookback': 12,
        'trading_days_per_month': 22,
        'kr_etf_map': {
            'SPY': '379800',
            'VEU': '195930',
            'VNQ': '352560',
            'REM': '352560',
            'LQD': '305080',
            'HYG': '305080',
            'TLT': '304660',
            'GLD': '132030',
            'BIL': '329750',
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

    def _get_modules(self) -> list[list[str]]:
        """공격자산을 2개씩 묶어 모듈 리스트 반환."""
        off = self.settings['offensive']
        modules = []
        for i in range(0, len(off), 2):
            pair = off[i:i + 2]
            if len(pair) == 2:
                modules.append(pair)
        return modules

    def analyze(self, price_data: dict[str, pd.DataFrame]) -> dict | None:
        """CDM 시그널 분석 — 4모듈 듀얼모멘텀."""
        s = self.settings
        td = s['trading_days_per_month']
        lb = s['lookback']
        min_days = td * lb + 5

        all_tickers = list(set(s['offensive'] + s['defensive']))
        for ticker in all_tickers:
            if ticker not in price_data or price_data[ticker] is None:
                return None
            if len(price_data[ticker]) < min_days:
                return None

        modules = self._get_modules()
        if not modules:
            return None

        # 방어자산 12개월 수익률
        def_ticker = s['defensive'][0]
        def_prices = self._get_close(price_data[def_ticker]).values
        def_return = self.calc_monthly_return(def_prices, lb, td)

        module_weight = 1.0 / len(modules)  # 각 모듈 25%
        target_weights = {t: 0.0 for t in all_tickers}
        module_results = []

        for idx, pair in enumerate(modules):
            # 모듈 내 상대모멘텀: 12개월 수익률 비교
            scores = []
            for ticker in pair:
                close = self._get_close(price_data[ticker]).values
                ret = self.calc_monthly_return(close, lb, td)
                scores.append((ticker, ret))
            scores.sort(key=lambda x: x[1], reverse=True)
            winner, winner_ret = scores[0]

            # 절대모멘텀: 승자 vs 방어자산
            if winner_ret > def_return:
                target_weights[winner] = target_weights.get(winner, 0.0) + module_weight
                module_results.append({
                    'module': idx + 1,
                    'pair': pair,
                    'winner': winner,
                    'winner_return': round(winner_ret * 100, 2),
                    'is_offensive': True,
                })
            else:
                target_weights[def_ticker] = target_weights.get(def_ticker, 0.0) + module_weight
                module_results.append({
                    'module': idx + 1,
                    'pair': pair,
                    'winner': def_ticker,
                    'winner_return': round(def_return * 100, 2),
                    'is_offensive': False,
                })

        # 0인 항목 제거
        target_weights = {k: v for k, v in target_weights.items() if v > 0}

        # kr_etf_map 변환
        kr_map = s['kr_etf_map']
        target_weights_kr = {}
        for us_ticker, w in target_weights.items():
            code = str(kr_map.get(us_ticker, "")).strip()
            if code:
                target_weights_kr[code] = target_weights_kr.get(code, 0.0) + float(w)

        # 요약
        offensive_count = sum(1 for m in module_results if m['is_offensive'])
        reason_parts = [f"M{m['module']}:{m['winner']}({m['winner_return']:+.1f}%)"
                        for m in module_results]
        reason = f"{offensive_count}/{len(modules)} 모듈 공격 | " + " | ".join(reason_parts)

        return {
            'target_weights': target_weights,
            'target_weights_kr': target_weights_kr,
            'module_results': module_results,
            'defensive_return': round(def_return * 100, 2),
            'offensive_count': offensive_count,
            'total_modules': len(modules),
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
        prev_weights = None
        equity_rows = []
        alloc_rows = []

        for i in range(12, len(monthly)):
            date = monthly.index[i]
            prev_date = monthly.index[i - 1]

            # 이전 가중치로 이번 달 수익 반영
            if prev_weights is not None:
                r = 0.0
                for a, w in prev_weights.items():
                    p0 = float(monthly.loc[prev_date, a])
                    p1 = float(monthly.loc[date, a])
                    if p0 > 0:
                        r += float(w) * ((p1 / p0) - 1.0)
                balance *= (1.0 + r)
                balance *= (1.0 - fee)

            # 시그널 계산
            sub_data = {}
            for t in all_tickers:
                sub_data[t] = pd.DataFrame({"close": merged.loc[:date, t]})

            sig = self.analyze(sub_data)
            if not sig:
                continue

            prev_weights = sig['target_weights']
            equity_rows.append({"date": date, "equity": balance})
            alloc_rows.append({
                "date": date,
                "offensive_count": sig['offensive_count'],
                "weights": dict(sig['target_weights']),
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
            "allocations": pd.DataFrame(alloc_rows),
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
