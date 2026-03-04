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

    # ── 3단계 기본 설정 ──
    DEFAULT_SETTINGS_3TIER = {
        'evaluation_mode': 3,
        'overvalue_threshold': 5.0,
        'undervalue_threshold': -6.0,
        'sell_ratio_overvalue': 100.0,
        'sell_ratio_neutral': 66.7,
        'sell_ratio_undervalue': 60.0,
        'buy_ratio_overvalue': 66.7,
        'buy_ratio_neutral': 66.7,
        'buy_ratio_undervalue': 120.0,
        'trend_period_weeks': 260,
        'initial_cash_ratio': 0.16,
        'min_cash_ratio': 0.10,
        'min_stock_ratio': 0.10,
        'commission_rate': 0.00015,
    }

    # ── 5단계 기본 설정 ──
    DEFAULT_SETTINGS_5TIER = {
        'evaluation_mode': 5,
        'super_overvalue_threshold': 10.0,
        'overvalue_threshold': 5.0,
        'undervalue_threshold': -6.0,
        'super_undervalue_threshold': -10.0,
        'sell_ratio_super_overvalue': 150.0,
        'sell_ratio_overvalue': 100.0,
        'sell_ratio_neutral': 66.7,
        'sell_ratio_undervalue': 60.0,
        'sell_ratio_super_undervalue': 33.0,
        'buy_ratio_super_overvalue': 33.0,
        'buy_ratio_overvalue': 66.7,
        'buy_ratio_neutral': 66.7,
        'buy_ratio_undervalue': 120.0,
        'buy_ratio_super_undervalue': 200.0,
        'trend_period_weeks': 260,
        'initial_cash_ratio': 0.16,
        'min_cash_ratio': 0.10,
        'min_stock_ratio': 0.10,
        'commission_rate': 0.00015,
    }

    # 하위 호환용
    DEFAULT_SETTINGS = DEFAULT_SETTINGS_3TIER

    def __init__(self, settings: dict = None, evaluation_mode: int = None):
        # evaluation_mode 결정: 인자 > settings > 기본(3)
        mode = evaluation_mode
        if mode is None and settings:
            mode = settings.get('evaluation_mode')
        if mode is None:
            mode = 3
        base = self.DEFAULT_SETTINGS_5TIER if mode == 5 else self.DEFAULT_SETTINGS_3TIER
        self.settings = {**base, **(settings or {})}
        self.settings['evaluation_mode'] = mode

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
        완전 벡터화된 NumPy 구현 (for 루프 제거).
        """
        prices = weekly_df['close'].values.astype(np.float64)
        n = len(prices)
        window = int(self.settings['trend_period_weeks'])

        # x값 (상대적 인덱스 또는 날짜 기준 일수)
        # O(n) 처리를 위해 인덱스 기반으로 우선 계산 (성능 최적화)
        x = np.arange(n, dtype=np.float64)
        y = np.log(prices)

        # 1) Expanding Window (1 ~ window)
        sx_exp  = np.cumsum(x)
        sy_exp  = np.cumsum(y)
        sx2_exp = np.cumsum(x**2)
        sxy_exp = np.cumsum(x * y)
        w_exp   = np.arange(1, n + 1, dtype=np.float64)
        
        # 2) Rolling Window (size=window)
        # 윈도우 합산: sum[i] = cumsum[i] - cumsum[i-window]
        def rolling_sum(arr, win):
            res = arr.copy()
            res[win:] = arr[win:] - arr[:-win]
            return res

        sx  = rolling_sum(sx_exp, window)
        sy  = rolling_sum(sy_exp, window)
        sx2 = rolling_sum(sx2_exp, window)
        sxy = rolling_sum(sxy_exp, window)
        
        # 윈도우 크기 결정: i < window 인 구간은 i+1, 이후는 window
        w = np.where(w_exp < window, w_exp, float(window))
        
        denom = w * sx2 - sx * sx
        # denom이 0인 경우 방지 (최소 1e-12)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        
        slope = (w * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / w
        
        log_trend = slope * x + intercept
        return np.exp(log_trend)

    # ─────────────────────────────────────────────────────
    # 이격도 & 시장 상태
    # ─────────────────────────────────────────────────────
    @staticmethod
    def calc_divergence(price: float, trend: float) -> float:
        """이격도 계산 (%): (가격 - 추세) / 추세 × 100.
        양수 = 가격이 추세 위 (고평가), 음수 = 가격이 추세 아래 (저평가)."""
        if trend == 0 or pd.isna(trend):
            return 0.0
        return (price - trend) / trend * 100

    def get_market_state(self, divergence: float) -> tuple:
        """
        시장 상태 판단 (3단계 또는 5단계).
        Returns: (state, sell_ratio, buy_ratio)
        """
        s = self.settings
        mode = s.get('evaluation_mode', 3)

        if mode == 5:
            sov = s.get('super_overvalue_threshold', 10.0)
            ov = s['overvalue_threshold']
            un = s['undervalue_threshold']
            sun = s.get('super_undervalue_threshold', -10.0)

            if divergence > sov:
                return ('SUPER_OVERVALUE', s['sell_ratio_super_overvalue'], s['buy_ratio_super_overvalue'])
            elif divergence > ov:
                return ('OVERVALUE', s['sell_ratio_overvalue'], s['buy_ratio_overvalue'])
            elif divergence < sun:
                return ('SUPER_UNDERVALUE', s['sell_ratio_super_undervalue'], s['buy_ratio_super_undervalue'])
            elif divergence < un:
                return ('UNDERVALUE', s['sell_ratio_undervalue'], s['buy_ratio_undervalue'])
            else:
                return ('NEUTRAL', s['sell_ratio_neutral'], s['buy_ratio_neutral'])
        else:
            # 3단계
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

            # 최대 주식 비율 확인
            max_stock_ratio = self.settings.get('max_stock_ratio', 1.0)
            current_stock_value = current_shares * current_price
            max_stock_buy_cash = (total_value * max_stock_ratio) - current_stock_value
            buy_amount = min(buy_amount, max(0, max_stock_buy_cash))

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
        fee_rate: float | None = None,
        initial_stock_ratio: float | None = None,
        precomputed_merged: pd.DataFrame | None = None,
    ) -> dict | None:
        """
        WDR 주간 리밸런싱 백테스트.
        precomputed_merged: 미리 계산된 merged DataFrame (signal_close, divergence, trade_close 컬럼).
                            제공 시 trend 계산/데이터 변환 과정 생략 → 배치 사전 계산 시 대폭 고속화.
        """
        # 1) precomputed_merged가 있으면 바로 사용
        if precomputed_merged is not None:
            merged = precomputed_merged
            if start_date is not None:
                merged = merged[merged.index >= pd.Timestamp(start_date)]
            if len(merged) < 5:
                return None
        else:
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

            merged = sig_df[["close", "divergence"]].rename(columns={"close": "signal_close"}).join(
                weekly_trade.rename(columns={"close": "trade_close"}),
                how="inner",
            ).dropna(subset=["trade_close"])

            if start_date is not None:
                merged = merged[merged.index >= pd.Timestamp(start_date)]

            if len(merged) < 5:
                return None

        commission_rate = float(fee_rate) if fee_rate is not None else float(self.settings.get("commission_rate", 0.00015))
        first_price = float(merged["trade_close"].iloc[0])
        if first_price <= 0:
            return None

        # 초기 매수 비율 결정
        init_balance = float(initial_balance)
        first_div = float(merged["divergence"].iloc[0])

        if initial_stock_ratio is not None:
            # 다이얼에서 직접 지정한 비율 사용 (0.0 = 전액 현금도 유효)
            buy_ratio = max(0.0, min(1.0, float(initial_stock_ratio)))
        else:
            mode = self.settings.get("evaluation_mode", 3)
            if mode == 5:
                sov = self.settings.get("super_overvalue_threshold", 10.0)
                ov = self.settings["overvalue_threshold"]
                un = self.settings["undervalue_threshold"]
                sun = self.settings.get("super_undervalue_threshold", -10.0)
                if first_div > sov:
                    buy_ratio = 0.20
                elif first_div > ov:
                    buy_ratio = 0.35
                elif first_div < sun:
                    buy_ratio = 0.80
                elif first_div < un:
                    buy_ratio = 0.65
                else:
                    buy_ratio = 0.50
            else:
                buy_ratio = 1.0 - float(self.settings.get("initial_cash_ratio", 0.16))

        cash = init_balance * (1.0 - buy_ratio)
        shares = int((init_balance * buy_ratio) / first_price)
        cash = init_balance - shares * first_price

        equity_records = [{
            "date": merged.index[0],
            "equity": cash + shares * first_price,
            "cash": cash,
            "shares": shares,
            "price": first_price,
            "signal_close": float(merged["signal_close"].iloc[0]),
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

        # 벡터화된 사전 추출 (pandas .iloc 반복 호출 제거)
        _trade_prices   = merged["trade_close"].to_numpy(dtype=np.float64)
        _signal_closes  = merged["signal_close"].to_numpy(dtype=np.float64)
        _divergences    = merged["divergence"].to_numpy(dtype=np.float64)
        _dates          = merged.index

        for i in range(1, len(merged)):
            cur_price  = _trade_prices[i]
            prev_price = _trade_prices[i - 1]
            divergence = _divergences[i]
            weekly_pnl = (cur_price - prev_price) * shares

            action = self.get_rebalance_action(
                weekly_pnl=weekly_pnl,
                divergence=divergence,
                current_shares=shares,
                current_price=cur_price,
                cash=cash,
            )

            act = action["action"]
            qty = int(action["quantity"])

            if act == "SELL" and qty > 0:
                amount = qty * cur_price
                fee = amount * commission_rate
                cash += amount - fee
                shares -= qty
                sell_count += 1
                if weekly_pnl > 0:
                    win_count += 1
                trades.append({
                    "date": _dates[i].strftime("%Y-%m-%d"),
                    "action": "SELL",
                    "price": cur_price,
                    "quantity": qty,
                    "cash": cash,
                    "shares": shares,
                })
            elif act == "BUY" and qty > 0:
                amount = qty * cur_price
                fee = amount * commission_rate
                total_cost = amount + fee
                if total_cost <= cash:
                    cash -= total_cost
                    shares += qty
                    trades.append({
                        "date": _dates[i].strftime("%Y-%m-%d"),
                        "action": "BUY",
                        "price": cur_price,
                        "quantity": qty,
                        "cash": cash,
                        "shares": shares,
                    })

            equity_records.append({
                "date": _dates[i],
                "equity": cash + shares * cur_price,
                "cash": cash,
                "shares": shares,
                "price": cur_price,
                "signal_close": _signal_closes[i],
                "divergence": divergence,
                "action": act or "HOLD",
                "quantity": qty,
            })

        equity_df = pd.DataFrame(equity_records).set_index("date")
        if equity_df.empty:
            return None

        equity = equity_df["equity"].to_numpy(dtype=np.float64)
        equity_idx = equity_df.index
        final_equity = equity[-1]
        total_return = (final_equity / init_balance - 1.0) * 100.0

        days = (equity_idx[-1] - equity_idx[0]).days
        cagr = ((final_equity / init_balance) ** (365.0 / days) - 1.0) * 100.0 if days > 0 else 0.0

        peak = np.maximum.accumulate(equity)
        drawdown_arr = (equity - peak) / peak * 100.0
        mdd = float(drawdown_arr.min()) if len(drawdown_arr) > 0 else 0.0
        drawdown = pd.Series(drawdown_arr, index=equity_idx)

        yearly_mdds = drawdown.groupby(drawdown.index.year).min() if len(drawdown) > 0 else pd.Series(dtype=float)
        avg_yearly_mdd = float(yearly_mdds.mean()) if len(yearly_mdds) > 0 else mdd

        eq_series = pd.Series(equity, index=equity_idx)
        weekly_returns = eq_series.pct_change().dropna().to_numpy(dtype=np.float64)
        sharpe = 0.0
        if len(weekly_returns) > 1:
            std = weekly_returns.std()
            if std > 0:
                sharpe = float((weekly_returns.mean() / std) * np.sqrt(52))

        calmar = abs(cagr / mdd) if mdd != 0 else 0.0
        win_rate = (win_count / sell_count * 100.0) if sell_count > 0 else 0.0

        sig_close_arr = merged["signal_close"].to_numpy(dtype=np.float64)
        benchmark_pct = (sig_close_arr / sig_close_arr[0] - 1.0) * 100.0
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


def _wdr_opt_task(signal_df, trade_df, settings_override: dict, eval_mode: int,
                  cap: float, start_date: str, fee_rate: float,
                  initial_stock_ratio: float) -> dict | None:
    """ProcessPool 워커용 WDR 단일 조합 백테스트.
    settings_override: overvalue_threshold, undervalue_threshold,
                       sell_ratio_*, buy_ratio_* 등을 포함하는 dict.
    """
    strat = WDRStrategy(settings=settings_override, evaluation_mode=eval_mode)
    bt = strat.run_backtest(
        signal_daily_df=signal_df,
        trade_daily_df=trade_df,
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
            "Calmar": round(m["calmar"], 3),
            "Sharpe": round(m["sharpe"], 3),
            "수익률(%)": round(m["total_return"], 2),
            "최종자산": round(m["final_equity"], 0),
        })
        return result
    return None
