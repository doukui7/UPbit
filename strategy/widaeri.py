import numpy as np
import pandas as pd


class WDRStrategy:
    """
    WDR (위대리) 3-Stage Strategy - ISA 계좌용

    나스닥 성장 추세선 대비 이격도에 따른 3단계 매매:
      고평가  (>5%)  : 매도비율 100%, 매수비율 66.7%
      중립  (-6%~5%) : 매도비율 66.7%, 매수비율 66.7%
      저평가 (<-6%)  : 매도비율 60%,  매수비율 120%

    시그널 소스: TIGER 미국나스닥100 (133690, 2010.10.18 상장)
      - 국내 최초 미국주식 ETF → 가장 긴 가격 이력 (15년+)
      - QQQ 해외 데이터 대신 사용하여 해외 시세 API 불필요
    추세선: 133690 주간 종가의 Rolling Window(260주) 로그 선형 회귀
    리밸런싱: 주간 (금요일 기준)
    매매 대상: 국내 레버리지 ETF (ISA 계좌에서 해외 ETF 직접 매매 불가)
      - 실제 매매는 TQQQ/SOXL 대응 국내 2x 레버리지 ETF로 수행

    ═══════════════════════════════════════════════════════════
    [수수료 정보] ISA 계좌 국내 ETF 매매
    ═══════════════════════════════════════════════════════════

    ┌─────────────┬──────────────┬──────────────┬──────────────────────┐
    │ 증권사       │ 매매 수수료   │ ETF 보수(별도)│ 비고                  │
    ├─────────────┼──────────────┼──────────────┼──────────────────────┤
    │ 한국투자증권  │ 0.0140396%   │ 상품별 상이   │ 나무 온라인 (현재 사용) │
    │ 키움증권     │ 0.015%       │ 상품별 상이   │ 영웅문 온라인          │
    │ 미래에셋     │ 0.014%       │ 상품별 상이   │ m.Stock 온라인        │
    │ 삼성증권     │ 0.015%       │ 상품별 상이   │ mPOP 온라인           │
    │ NH투자      │ 0.0140396%   │ 상품별 상이   │ 나무 온라인            │
    │ 토스증권     │ 무료~0.015%  │ 상품별 상이   │ 신규 이벤트 시 무료     │
    │ 카카오페이   │ 0.015%       │ 상품별 상이   │                      │
    └─────────────┴──────────────┴──────────────┴──────────────────────┘

    [시그널 소스 ETF]
    ┌─────────────────────────┬──────────┬──────────┬──────────────────────┐
    │ ETF명                    │ 총보수    │ 종목코드  │ 비고                  │
    ├─────────────────────────┼──────────┼──────────┼──────────────────────┤
    │ TIGER 미국나스닥100       │ 0.07%    │ 133690   │ 시그널용 (2010.10 상장)│
    └─────────────────────────┴──────────┴──────────┴──────────────────────┘
    * 133690은 국내 최초 미국주식 ETF (15년+ 이력)
    * 추세선·이격도 계산에만 사용, 실제 매매는 레버리지 ETF로 수행

    [매매 대상 ETF 보수 비교 (레버리지/액티브)]
    ┌───────────────────────────────────────────┬──────────┬──────────┬─────────────┐
    │ ETF명                                      │ 총보수    │ 종목코드  │ 구분          │
    ├───────────────────────────────────────────┼──────────┼──────────┼─────────────┤
    │ TIGER 미국나스닥100레버리지(합성)            │ 0.25%    │ 418660   │ 나스닥 2x    │
    │ KODEX 미국나스닥100레버리지(합성 H)          │ 0.30%    │ 409820   │ 나스닥 2x(H) │
    │ TIGER 미국필라델피아반도체레버리지(합성)       │ 0.58%    │ 423920   │ 반도체 2x    │
    │ ACE 미국빅테크TOP7 Plus레버리지(합성)        │ 0.60%    │ 465610   │ 빅테크 2x    │
    │ PLUS 미국테크TOP10레버리지(합성)             │ 0.50%    │ 461910   │ 테크 2x      │
    │ TIMEFOLIO 미국나스닥100액티브                │ 0.80%    │ 426030   │ 나스닥 액티브 │
    └───────────────────────────────────────────┴──────────┴──────────┴─────────────┘
    * 한국은 2x까지만 허용 (미국 TQQQ 3x, SOXL 3x와 차이)
    * (H) = 환헤지, 무표기 = 환노출
    * 액티브 = 운용사 재량으로 나스닥100 초과수익 추구
    * 레버리지 ETF는 보수가 높으므로 장기보유보다 단기매매에 적합

    [ISA 세제혜택]
    * 비과세 한도: 일반형 200만원 / 서민·청년형 400만원
    * 한도 초과분: 9.9% 분리과세 (일반 15.4% 대비 유리)
    * 의무가입기간: 3년
    * 납입한도: 연 2,000만원 (총 1억원)
    * → 위대리 전략의 주간 리밸런싱 시 매매차익이 비과세 한도
    *   내에서 면세 처리되므로 절세 효과 극대화

    [증권사 선택 가이드 - ISA]
    * API 자동매매 필수 → 한국투자증권(KIS Open API) 또는 키움증권
    * 최저 수수료 → 미래에셋(0.014%), 한국투자(0.014%)
    * 수동 매매 OK → 토스증권(이벤트 시 무료)
    * 현재 선택: 한국투자증권 (API 지원 + 저수수료 + 안정성)
    """

    DEFAULT_SETTINGS = {
        'overvalue_threshold': 5.0,
        'undervalue_threshold': -6.0,
        'sell_ratio_overvalue': 100.0,
        'sell_ratio_neutral': 66.7,
        'sell_ratio_undervalue': 60.0,
        'buy_ratio_overvalue': 66.7,
        'buy_ratio_neutral': 66.7,
        'buy_ratio_undervalue': 120.0,
        'trend_period_weeks': 260,  # 5년
        'initial_cash_ratio': 0.50,
        'min_cash_ratio': 0.10,
        'min_stock_ratio': 0.10,
        'commission_rate': 0.00015,  # ISA ETF 수수료 (~0.015%)
    }

    def __init__(self, settings: dict = None):
        self.settings = {**self.DEFAULT_SETTINGS, **(settings or {})}

    # ─────────────────────────────────────────────────────
    # 데이터 변환
    # ─────────────────────────────────────────────────────
    @staticmethod
    def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """일봉 → 주봉 변환 (금요일 종가 기준)."""
        close_col = 'close' if 'close' in df.columns else 'Close'
        weekly = df[close_col].resample('W-FRI').last().dropna()
        return pd.DataFrame({'close': weekly.values}, index=weekly.index)

    # ─────────────────────────────────────────────────────
    # 성장 추세선 계산
    # ─────────────────────────────────────────────────────
    def calc_growth_trend(self, weekly_df: pd.DataFrame) -> np.ndarray:
        """
        성장 추세선 계산 (초기 260주는 Expanding Window, 이후 Rolling Window 260주).
        최소 2개 이상의 데이터가 있을 때부터 로그 선형 회귀 수행.

        Args:
            weekly_df: 주봉 DataFrame (index=datetime, columns=['close'])

        Returns:
            numpy array: 추세선 값
        """
        prices = weekly_df['close'].values
        dates = weekly_df.index
        window = self.settings['trend_period_weeks']

        # 엑셀 날짜 형식 (1899-12-31 기준 일수)
        excel_dates = np.array([
            (pd.Timestamp(d) - pd.Timestamp("1899-12-31")).days
            for d in dates
        ])

        trend = np.full(len(prices), np.nan)

        # 인덱스 1(두 번째 데이터)부터 계산 시작 (expanding -> rolling)
        for i in range(1, len(prices)):
            # 현재까지의 데이터 개수와 설정된 윈도우 크기 중 작은 것을 선택 (+1은 현재 포인트 포함)
            current_window_size = min(i + 1, window + 1)
            start_idx = i - (current_window_size - 1)
            
            current_dates = excel_dates[start_idx:i + 1]
            current_prices = prices[start_idx:i + 1]

            if len(current_prices) < 2:
                continue

            # 로그 변환 후 선형 회귀
            try:
                log_prices = np.log(current_prices)
                slope, intercept = np.polyfit(current_dates, log_prices, 1)
                trend[i] = np.exp(slope * excel_dates[i] + intercept)
            except Exception:
                continue

        return trend

    # ─────────────────────────────────────────────────────
    # 이격도 & 시장 상태
    # ─────────────────────────────────────────────────────
    @staticmethod
    def calc_divergence(price: float, trend: float) -> float:
        """이격도 계산 (%)."""
        if trend == 0 or pd.isna(trend):
            return 0.0
        return (price - trend) / trend * 100

    def get_market_state(self, divergence: float) -> tuple:
        """
        3단계 시장 상태 판단.
        Returns: (state, sell_ratio, buy_ratio)
        """
        s = self.settings
        if divergence > s['overvalue_threshold']:
            return ('OVERVALUE', s['sell_ratio_overvalue'], s['buy_ratio_overvalue'])
        elif divergence < s['undervalue_threshold']:
            return ('UNDERVALUE', s['sell_ratio_undervalue'], s['buy_ratio_undervalue'])
        else:
            return ('NEUTRAL', s['sell_ratio_neutral'], s['buy_ratio_neutral'])

    # ─────────────────────────────────────────────────────
    # 리밸런싱 액션 계산
    # ─────────────────────────────────────────────────────
    def get_rebalance_action(self, weekly_pnl: float, divergence: float,
                              current_shares: int, current_price: float,
                              cash: float) -> dict:
        """
        주간 리밸런싱 액션 계산.

        Args:
            weekly_pnl: 이번 주 손익 (주가변동 × 보유수량)
            divergence: QQQ 이격도 (%)
            current_shares: 현재 보유 주수
            current_price: 현재 ETF 가격
            cash: 현재 예수금

        Returns:
            dict: {action, quantity, state, divergence, sell_ratio, buy_ratio}
        """
        state, sell_ratio, buy_ratio = self.get_market_state(divergence)
        total_value = cash + current_shares * current_price

        action = None
        quantity = 0

        min_cash_ratio = self.settings['min_cash_ratio']
        min_stock_ratio = self.settings['min_stock_ratio']

        if weekly_pnl > 0 and current_shares > 0:
            # 수익 → 매도 (이익 실현)
            sell_amount = weekly_pnl * (sell_ratio / 100.0)
            quantity = int(sell_amount / current_price)

            # 최소 주식 비율 확인
            if quantity > 0:
                new_shares = current_shares - quantity
                new_stock_value = new_shares * current_price
                if total_value > 0 and new_stock_value / total_value < min_stock_ratio:
                    max_sell = current_shares - max(1, int(total_value * min_stock_ratio / current_price))
                    quantity = max(0, min(quantity, max_sell))

            if quantity > current_shares:
                quantity = current_shares

            if quantity > 0:
                action = 'SELL'

        elif weekly_pnl < 0:
            # 손실 → 매수 (저가 매수)
            buy_amount = abs(weekly_pnl) * (buy_ratio / 100.0)

            # 최소 현금 비율 확인
            max_buy_cash = cash - max(0, total_value * min_cash_ratio)
            buy_amount = min(buy_amount, max(0, max_buy_cash))

            quantity = int(buy_amount / current_price)

            if quantity > 0 and (quantity * current_price) <= cash:
                action = 'BUY'
            else:
                quantity = 0

        return {
            'action': action,
            'quantity': quantity,
            'state': state,
            'divergence': round(divergence, 2),
            'sell_ratio': sell_ratio,
            'buy_ratio': buy_ratio,
            'weekly_pnl': round(weekly_pnl, 0),
        }

    # ─────────────────────────────────────────────────────
    # 현재 시그널 계산 (자동매매용)
    # ─────────────────────────────────────────────────────
    def analyze(self, qqq_daily_df: pd.DataFrame) -> dict | None:
        """
        QQQ 일봉 데이터로 현재 이격도 및 시장 상태 분석.

        Args:
            qqq_daily_df: QQQ 일봉 DataFrame (columns: [open, high, low, close, volume])

        Returns:
            dict: {divergence, state, sell_ratio, buy_ratio, qqq_price, trend}
        """
        if qqq_daily_df is None or len(qqq_daily_df) < 260 * 5:
            return None

        weekly = self.daily_to_weekly(qqq_daily_df)
        trend = self.calc_growth_trend(weekly)

        # 마지막 유효 데이터
        valid_idx = ~np.isnan(trend)
        if not valid_idx.any():
            return None

        last_price = weekly['close'].values[-1]
        last_trend = trend[-1]

        if np.isnan(last_trend):
            return None

        divergence = self.calc_divergence(last_price, last_trend)
        state, sell_ratio, buy_ratio = self.get_market_state(divergence)

        return {
            'divergence': round(divergence, 2),
            'state': state,
            'sell_ratio': sell_ratio,
            'buy_ratio': buy_ratio,
            'qqq_price': round(last_price, 2),
            'trend': round(last_trend, 2),
            'date': weekly.index[-1].strftime('%Y-%m-%d'),
        }

    # ─────────────────────────────────────────────────────
    # 백테스트
    # ─────────────────────────────────────────────────────
    def run_backtest(
        self,
        signal_daily_df: pd.DataFrame,
        trade_daily_df: pd.DataFrame | None = None,
        initial_balance: float = 10_000_000,
        start_date=None,
    ) -> dict | None:
        """
        WDR 주간 리밸런싱 백테스트.

        Args:
            signal_daily_df: 시그널 계산용 일봉 (예: QQQ, 133690)
            trade_daily_df: 실제 매매 대상 일봉 (없으면 signal_daily_df 사용)
            initial_balance: 초기자본
            start_date: 백테스트 시작일 (None이면 전체)

        Returns:
            dict:
              - equity_df: date index, columns=[equity, cash, shares, price, divergence, action]
              - trades: 거래 로그 리스트
              - benchmark_df: 벤치마크(매수후보유) 수익률(%)
              - metrics: 성과 지표
        """
        if signal_daily_df is None or len(signal_daily_df) < 60:
            return None
        if trade_daily_df is None:
            trade_daily_df = signal_daily_df
        if trade_daily_df is None or len(trade_daily_df) < 60:
            return None

        weekly_signal = self.daily_to_weekly(signal_daily_df)
        weekly_trade = self.daily_to_weekly(trade_daily_df)
        if weekly_signal is None or weekly_trade is None:
            return None
        if len(weekly_signal) < 10 or len(weekly_trade) < 10:
            return None

        trend = self.calc_growth_trend(weekly_signal)
        sig_df = weekly_signal.copy()
        sig_df["trend"] = trend
        sig_df = sig_df.dropna(subset=["trend"])
        if sig_df.empty:
            return None

        sig_df["divergence"] = (
            (sig_df["close"] - sig_df["trend"]) / sig_df["trend"] * 100.0
        )

        merged = sig_df[["divergence"]].join(
            weekly_trade.rename(columns={"close": "trade_close"}),
            how="inner",
        ).dropna(subset=["trade_close"])

        if start_date is not None:
            merged = merged[merged.index >= pd.Timestamp(start_date)]

        if len(merged) < 5:
            return None

        commission_rate = float(self.settings.get("commission_rate", 0.00015))
        cash_ratio = float(self.settings.get("initial_cash_ratio", 0.5))
        first_price = float(merged["trade_close"].iloc[0])
        if first_price <= 0:
            return None

        init_balance = float(initial_balance)
        cash = init_balance * cash_ratio
        shares = int((init_balance * (1.0 - cash_ratio)) / first_price)
        cash = init_balance - shares * first_price

        equity_records = [{
            "date": merged.index[0],
            "equity": cash + shares * first_price,
            "cash": cash,
            "shares": shares,
            "price": first_price,
            "divergence": float(merged["divergence"].iloc[0]),
            "action": "INIT",
            "quantity": shares,
        }]
        trades = [{
            "date": merged.index[0].strftime("%Y-%m-%d"),
            "action": "INIT",
            "price": first_price,
            "quantity": shares,
            "cash": cash,
            "shares": shares,
        }]

        sell_count = 0
        win_count = 0

        for i in range(1, len(merged)):
            cur_price = float(merged["trade_close"].iloc[i])
            prev_price = float(merged["trade_close"].iloc[i - 1])
            divergence = float(merged["divergence"].iloc[i])
            weekly_pnl = (cur_price - prev_price) * shares

            action = self.get_rebalance_action(
                weekly_pnl=weekly_pnl,
                divergence=divergence,
                current_shares=shares,
                current_price=cur_price,
                cash=cash,
            )

            if action["action"] == "SELL" and action["quantity"] > 0:
                qty = int(action["quantity"])
                amount = qty * cur_price
                fee = amount * commission_rate
                cash += amount - fee
                shares -= qty
                sell_count += 1
                if weekly_pnl > 0:
                    win_count += 1
                trades.append({
                    "date": merged.index[i].strftime("%Y-%m-%d"),
                    "action": "SELL",
                    "price": cur_price,
                    "quantity": qty,
                    "cash": cash,
                    "shares": shares,
                })
            elif action["action"] == "BUY" and action["quantity"] > 0:
                qty = int(action["quantity"])
                amount = qty * cur_price
                fee = amount * commission_rate
                total_cost = amount + fee
                if total_cost <= cash:
                    cash -= total_cost
                    shares += qty
                    trades.append({
                        "date": merged.index[i].strftime("%Y-%m-%d"),
                        "action": "BUY",
                        "price": cur_price,
                        "quantity": qty,
                        "cash": cash,
                        "shares": shares,
                    })

            equity_records.append({
                "date": merged.index[i],
                "equity": cash + shares * cur_price,
                "cash": cash,
                "shares": shares,
                "price": cur_price,
                "divergence": divergence,
                "action": action["action"] or "HOLD",
                "quantity": int(action["quantity"]),
            })

        equity_df = pd.DataFrame(equity_records).set_index("date")
        if equity_df.empty:
            return None

        equity = equity_df["equity"]
        final_equity = float(equity.iloc[-1])
        total_return = (final_equity / init_balance - 1.0) * 100.0

        days = (equity.index[-1] - equity.index[0]).days
        cagr = ((final_equity / init_balance) ** (365.0 / days) - 1.0) * 100.0 if days > 0 else 0.0

        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100.0
        mdd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        yearly_mdds = drawdown.groupby(drawdown.index.year).min() if len(drawdown) > 0 else pd.Series(dtype=float)
        avg_yearly_mdd = float(yearly_mdds.mean()) if len(yearly_mdds) > 0 else mdd

        weekly_returns = equity.pct_change().dropna()
        sharpe = 0.0
        if len(weekly_returns) > 1 and weekly_returns.std() > 0:
            sharpe = float((weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52))

        calmar = abs(cagr / mdd) if mdd != 0 else 0.0
        win_rate = (win_count / sell_count * 100.0) if sell_count > 0 else 0.0

        benchmark_pct = (merged["trade_close"] / merged["trade_close"].iloc[0] - 1.0) * 100.0
        benchmark_df = pd.DataFrame({"benchmark_return_pct": benchmark_pct}, index=merged.index)

        return {
            "equity_df": equity_df,
            "trades": trades,
            "benchmark_df": benchmark_df,
            "metrics": {
                "total_return": round(total_return, 4),
                "cagr": round(cagr, 4),
                "mdd": round(mdd, 4),
                "avg_yearly_mdd": round(avg_yearly_mdd, 4),
                "sharpe": round(sharpe, 4),
                "calmar": round(calmar, 4),
                "trade_count": max(0, len(trades) - 1),
                "win_rate": round(win_rate, 2),
                "final_equity": round(final_equity, 2),
            },
        }

    def backtest(self, qqq_daily_df: pd.DataFrame, initial_balance: float = 10_000_000) -> dict | None:
        """
        기존 UI 호환용 단순 백테스트 래퍼.
        """
        result = self.run_backtest(
            signal_daily_df=qqq_daily_df,
            trade_daily_df=qqq_daily_df,
            initial_balance=initial_balance,
            start_date=None,
        )
        if not result:
            return None
        m = result["metrics"]
        return {
            "equity_df": result["equity_df"],
            "total_return": m["total_return"],
            "cagr": m["cagr"],
            "mdd": m["mdd"],
            "avg_yearly_mdd": m["avg_yearly_mdd"],
            "sharpe": m["sharpe"],
            "calmar": m["calmar"],
            "final_equity": m["final_equity"],
        }
