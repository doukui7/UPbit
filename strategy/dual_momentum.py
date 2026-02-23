import numpy as np
import pandas as pd


class DualMomentumStrategy:
    """
    Dual Momentum (GEM) Strategy - 연금저축 계좌용

    절대+상대 모멘텀 기반 자산 배분:
      - Offensive: SPY(미국), EFA(선진국) → 모멘텀 점수로 상대 비교
      - Defensive: AGG(채권) → 모멘텀 약세 시 방어 자산
      - Canary: BIL(단기채) → 절대 모멘텀 기준선

    모멘텀 점수: (m1*12 + m3*4 + m6*2 + m12) / 4
      m1 = 1개월 수익률, m3 = 3개월, m6 = 6개월, m12 = 12개월
      (거래일 기준: 1개월 = 22일)

    리밸런싱: 월 1회 (월말 기준)
    매매 대상: 국내 상장 ETF (연금저축 계좌에서 해외 ETF 직접 매매 불가)

    ═══════════════════════════════════════════════════════════
    [수수료 정보] 연금저축 계좌 국내 ETF 매매
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

    [매매 대상 ETF 보수 비교]
    ┌─────────────────────────────┬──────────┬──────────┬─────────────┐
    │ ETF명                        │ 총보수    │ 종목코드  │ 추종 지수     │
    ├─────────────────────────────┼──────────┼──────────┼─────────────┤
    │ TIGER 미국S&P500             │ 0.07%    │ 360750   │ SPY 대용     │
    │ KODEX 미국S&P500TR           │ 0.045%   │ 379800   │ SPY 대용(TR) │
    │ TIGER 선진국MSCI World       │ 0.19%    │ 453850   │ EFA 대용     │
    │ KODEX 선진국MSCI World       │ 0.18%    │ 251350   │ EFA 대용     │
    │ TIGER 미국채10년선물          │ 0.10%    │ 453540   │ AGG 대용     │
    │ KODEX 미국채10년선물          │ 0.09%    │ 308620   │ AGG 대용     │
    │ ACE 미국30년국채액티브        │ 0.05%    │ 471460   │ 장기채 대안   │
    └─────────────────────────────┴──────────┴──────────┴─────────────┘
    * TR(Total Return): 배당 재투자형 → 장기 복리 효과 우수
    * 듀얼모멘텀은 월 1회 리밸런싱이므로 매매 수수료 영향 최소

    [연금저축 세제혜택]
    * 세액공제: 연 최대 600만원 납입 (IRP 합산 900만원)
      - 총급여 5,500만원 이하: 16.5% 공제 (최대 99만원 환급)
      - 총급여 5,500만원 초과: 13.2% 공제 (최대 79.2만원 환급)
    * 과세이연: 매매차익·배당에 대한 세금을 인출 시까지 이연
    * 연금 수령 시: 3.3~5.5% 연금소득세 (일반 15.4% 대비 유리)
    * 중도 인출 시: 16.5% 기타소득세 (불이익)
    * → 듀얼모멘텀의 월간 리밸런싱 매매차익이 모두 과세이연되어
    *   복리 효과 극대화 (일반 계좌 대비 연 1~2% 추가 수익 효과)

    [증권사 선택 가이드 - 연금저축]
    * API 자동매매 필수 → 한국투자증권(KIS Open API) 또는 키움증권
    * 최저 수수료 → 미래에셋(0.014%), 한국투자(0.014%)
    * ETF 라인업 풍부 → 미래에셋(TIGER), 삼성(KODEX)
    * 현재 선택: 한국투자증권 (API 지원 + 저수수료 + ISA와 통합 관리)

    ═══════════════════════════════════════════════════════════
    [전략 상세 설명]
    ═══════════════════════════════════════════════════════════
    1. 기본 원리 (GEM: Global Equity Momentum)
       - 세계적인 퀀트 투자자 Gary Antonacci가 고안한 전략으로, '상대 모멘텀'과 '절대 모멘텀'을 결합합니다.
       - 상대 모멘텀: 미국(SPY)과 선진국(EFA) 중 더 강한 자산을 선택하여 수익률을 극대화합니다.
       - 절대 모멘텀: 선택된 자산의 수익률이 현금(BIL)보다 낮을 경우, 하락장으로 판단하여 방어 자산(AGG)으로 대피합니다.

    2. 미국 시장 (Original)
       - 공격 자산: SPY (S&P 500), EFA (MSCI EAFE - 미국 제외 선진국)
       - 방어 자산: AGG (Total Bond Market)
       - 기준(카나리아): BIL (1-3 Month T-Bill) - 과거 12개월 수익률 기준

    3. 한국 시장 변경 (Adaptation for KR)
       - 연금저축/ISA 계좌는 해외 ETF 직접 매매가 불가능하므로 지수 수익률이 유사한 국내 상장 ETF로 대체합니다.
       - SPY 대용: TIGER 미국S&P500 (360750) 또는 KODEX 미국S&P500TR (379800)
       - EFA 대용: TIGER 선진국MSCI World (453850) 또는 KODEX 선진국MSCI World (251350)
       - AGG 대용: TIGER 미국채10년선물 (453540) 또는 KODEX 미국채10년선물 (308620)
       - 시그널: 해외 원지수(SPY, EFA, BIL) 시세를 그대로 사용하여 판단하되, 실제 집행만 국내 ETF로 수행합니다.
    """

    DEFAULT_SETTINGS = {
        'offensive': ['SPY', 'EFA'],
        'defensive': ['AGG'],
        'canary': ['BIL'],
        'lookback': 12,       # 카나리아 룩백 (개월)
        'top_n': 1,           # 공격 자산 중 선택 수
        'trading_days_per_month': 22,

        # 한국 ETF 매핑 (연금저축/ISA에서 실제 매매할 종목코드)
        'kr_etf_map': {
            'SPY': '360750',   # TIGER 미국S&P500
            'EFA': '453850',   # TIGER 선진국MSCI World
            'AGG': '453540',   # TIGER 미국채10년선물
            'BIL': None,       # 카나리아 시그널용
        },
    }

    def __init__(self, settings: dict = None):
        self.settings = {**self.DEFAULT_SETTINGS, **(settings or {})}

    # ─────────────────────────────────────────────────────
    # 수익률 계산
    # ─────────────────────────────────────────────────────
    @staticmethod
    def calc_monthly_return(prices: np.ndarray, months: int,
                            trading_days: int = 22) -> float:
        """
        N개월 수익률 계산.
        """
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
                            trading_days: int = 22) -> float:
        """
        복합 모멘텀 점수 계산.
        score = (m1*12 + m3*4 + m6*2 + m12) / 4
        """
        m1 = DualMomentumStrategy.calc_monthly_return(prices, 1, trading_days)
        m3 = DualMomentumStrategy.calc_monthly_return(prices, 3, trading_days)
        m6 = DualMomentumStrategy.calc_monthly_return(prices, 6, trading_days)
        m12 = DualMomentumStrategy.calc_monthly_return(prices, 12, trading_days)
        return (m1 * 12 + m3 * 4 + m6 * 2 + m12) / 4

    # ─────────────────────────────────────────────────────
    # 시그널 분석
    # ─────────────────────────────────────────────────────
    def analyze(self, price_data: dict[str, pd.DataFrame]) -> dict | None:
        """
        듀얼모멘텀 시그널 분석.
        """
        s = self.settings
        td = s['trading_days_per_month']
        min_days = td * 12 + 5  # 12개월 + 여유분

        # 데이터 충분성 확인
        for ticker in s['offensive'] + s['canary']:
            if ticker not in price_data:
                return None
            df = price_data[ticker]
            if df is None or len(df) < min_days:
                return None

        # 공격 자산 모멘텀 점수 계산
        scores = {}
        for ticker in s['offensive']:
            df = price_data[ticker]
            close_col = 'close' if 'close' in df.columns else 'Close'
            prices = df[close_col].values
            scores[ticker] = round(self.calc_momentum_score(prices, td), 6)

        # 카나리아 수익률 (절대 모멘텀 기준선)
        canary_ticker = s['canary'][0]
        canary_df = price_data[canary_ticker]
        close_col = 'close' if 'close' in canary_df.columns else 'Close'
        canary_prices = canary_df[close_col].values
        canary_return = self.calc_monthly_return(
            canary_prices, s['lookback'], td
        )

        # 상대 모멘텀: 최고 점수 공격 자산 선택
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_ticker, top_score = sorted_scores[0]

        kr_etf_map = s['kr_etf_map']

        # 의사결정: 절대 모멘텀 (카나리아 비교)
        if top_score > canary_return:
            # 공격 자산 승리 → 공격 자산에 100% 배분
            target = top_ticker
            is_offensive = True
            reason = (f"{top_ticker} 모멘텀({top_score:.4f}) > "
                      f"BIL({canary_return:.4f}) → 공격 자산 배분")
        else:
            # 카나리아 승리 → 방어 자산에 100% 배분
            target = s['defensive'][0]
            is_offensive = False
            reason = (f"모든 공격자산 모멘텀 < BIL({canary_return:.4f}) "
                      f"→ 방어 자산({target}) 배분")

        return {
            'target_ticker': target,
            'target_kr_code': kr_etf_map.get(target, ''),
            'is_offensive': is_offensive,
            'scores': scores,
            'canary_return': round(canary_return, 6),
            'reason': reason,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        }

    # ─────────────────────────────────────────────────────
    # 백테스트 수행
    # ─────────────────────────────────────────────────────
    def run_backtest(self, price_data: dict[str, pd.DataFrame], initial_balance=1000000, fee=0.0002):
        """
        월별 리밸런싱 백테스트 수행.
        Execution: 월말 종가 분석 -> 익월 초 시가 매매 (단, 여기서는 월말 종가 데이터로 단순화)
        """
        s = self.settings
        td = s['trading_days_per_month']
        
        # 1. 월별 데이터 통합 (공통 인덱스 추출)
        combined_df = None
        for ticker, df in price_data.items():
            df = df[['close']].rename(columns={'close': ticker})
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.join(df, how='inner')
        
        if combined_df is None or len(combined_df) < td * 12:
            return None

        # 2. 월말 샘플링
        monthly_df = combined_df.resample('ME').last().dropna()
        daily_for_momentum = combined_df.copy()
        
        balance = initial_balance
        equity_curve = []
        positions = []
        
        current_ticker = None
        current_shares = 0
        
        total_months = len(monthly_df)
        
        for i in range(12, total_months): # 룩백 기간 이후부터
            date = monthly_df.index[i]
            # 시그널 판단 (해당 월말 시점의 과거 일봉 데이터 사용)
            # momentum_data = 리밸런싱 시점까지의 일봉
            target_date = monthly_df.index[i]
            # 해당 날짜까지의 가격 데이터를 잘라서 analyze에 맞는 형식으로 전달
            sub_price_data = {}
            for t in price_data.keys():
                sub_price_data[t] = price_data[t][price_data[t].index <= target_date]
            
            signal = self.analyze(sub_price_data)
            if not signal: continue
            
            target_ticker = signal['target_ticker']
            price_current = monthly_df.iloc[i][target_ticker]
            
            # 포지션 변경
            if target_ticker != current_ticker:
                # 전량 매도 후 전량 매수
                if current_ticker:
                    # 매도 (이전 티커 가격)
                    prev_price = monthly_df.iloc[i][current_ticker]
                    balance = current_shares * prev_price * (1 - fee)
                
                # 매수
                current_shares = (balance * (1 - fee)) / price_current
                balance = 0
                current_ticker = target_ticker
            
            # 자산 평가
            current_eval = current_shares * price_current
            equity_curve.append({'date': date, 'equity': current_eval, 'ticker': current_ticker})
            positions.append({'date': date, 'ticker': current_ticker, 'score': signal['scores'].get(current_ticker, 0), 'is_offensive': signal['is_offensive']})

        # 결과 데이터프레임
        res_df = pd.DataFrame(equity_curve).set_index('date')
        if res_df.empty: return None
        
        # 성과 지표 계산
        final_equity = res_df['equity'].iloc[-1]
        total_return = (final_equity / initial_balance - 1) * 100
        
        days = (res_df.index[-1] - res_df.index[0]).days
        cagr = ((final_equity / initial_balance) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        res_df['peak'] = res_df['equity'].cummax()
        res_df['drawdown'] = (res_df['equity'] - res_df['peak']) / res_df['peak'] * 100
        mdd = res_df['drawdown'].min()
        
        sharpe = 0
        monthly_returns = res_df['equity'].pct_change().dropna()
        if not monthly_returns.empty:
            sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)

        return {
            'equity_df': res_df,
            'positions': pd.DataFrame(positions),
            'metrics': {
                'total_return': total_return,
                'cagr': cagr,
                'mdd': mdd,
                'sharpe': sharpe,
                'final_equity': final_equity
            }
        }

    def should_rebalance(self, last_rebalance_date: str = None) -> bool:
        """
        월말 리밸런싱 여부 확인.
        """
        today = pd.Timestamp.now()
        if last_rebalance_date:
            last = pd.Timestamp(last_rebalance_date)
            if last.year == today.year and last.month == today.month:
                return False
        return True
