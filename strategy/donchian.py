import pandas as pd
import numpy as np


class DonchianStrategy:
    """
    Donchian Channel Strategy (Dual Period / Turtle Style)

    sell_mode:
      "lower"   (기본) - close < Lower(sell_period) 이탈 시 매도
      "midline" - close < Midline((Upper+Lower)/2) 이탈 시 매도

    ═══════════════════════════════════════════════════════════
    [수수료 정보] KRX 금현물 매매
    ═══════════════════════════════════════════════════════════

    ┌─────────────┬──────────────┬──────────────┬──────────────────────┐
    │ 증권사       │ 매매 수수료   │ 최소 수수료   │ 비고                  │
    ├─────────────┼──────────────┼──────────────┼──────────────────────┤
    │ 키움증권     │ 0.33%        │ 1,000원      │ 현재 사용 중           │
    │ 한국투자증권  │ 0.33%        │ 1,000원      │ 나무 온라인 기준       │
    │ 미래에셋     │ 0.33%        │ 1,000원      │                      │
    │ 삼성증권     │ 0.33%        │ 1,000원      │ 지점 0.5%+            │
    │ NH투자      │ 0.33%        │ 1,000원      │                      │
    │ 한화투자     │ 0.33%        │ 1,000원      │                      │
    └─────────────┴──────────────┴──────────────┴──────────────────────┘

    * KRX 금시장은 증권사 수수료가 대부분 동일 (0.33%, VAT 포함 약 0.363%)
    * 증권사 선택 시 수수료보다 API 지원 여부가 핵심 기준
      - 키움증권: REST API 지원 (현재 사용)
      - 한국투자증권: REST API 지원 (KIS Open API)
      - 미래에셋/삼성/NH: API 미지원 또는 제한적

    [세금 - KRX 금현물의 최대 장점]
    * 양도소득세: 비과세 (금융투자소득세 대상 아님)
    * 배당소득세: 해당 없음
    * 부가세(VAT): 면제 (KRX 금시장 거래 시)
    * 실물 인출 시: 부가세 10% 부과 (인출하지 않으면 면세)
    * → 금 ETF(예: KODEX 골드선물) 대비 세금 우위
    *   금 ETF는 매매차익에 15.4% 배당소득세 부과
    """
    def __init__(self):
        pass

    def create_features(self, df, buy_period=20, sell_period=10):
        """
        Calculate Donchian Channels with separate periods.
        Upper  = buy_period 고가 채널
        Lower  = sell_period 저가 채널
        Middle = (Upper + Lower) / 2  (중심선)
        """
        df = df.copy()

        # Upper Channel (Buy Signal): Max of Highs over buy_period
        df[f'Donchian_Upper_{buy_period}'] = (
            df['high'].rolling(window=buy_period).max().shift(1)
        )

        # Lower Channel (Sell Signal): Min of Lows over sell_period
        df[f'Donchian_Lower_{sell_period}'] = (
            df['low'].rolling(window=sell_period).min().shift(1)
        )

        # Middle Channel (중심선): (Upper + Lower) / 2
        df[f'Donchian_Middle_{buy_period}_{sell_period}'] = (
            (df[f'Donchian_Upper_{buy_period}'] + df[f'Donchian_Lower_{sell_period}']) / 2
        )

        return df

    def get_signal(self, row, buy_period=20, sell_period=10, sell_mode="lower"):
        """
        Generate Buy/Sell signal based on Donchian Breakout.

        sell_mode:
          "lower"   - 하단 채널 이탈 시 매도 (기본)
          "midline" - 중심선 이탈 시 매도 (빠른 청산)
        """
        close = row.get('close')
        upper = row.get(f'Donchian_Upper_{buy_period}')
        lower = row.get(f'Donchian_Lower_{sell_period}')
        middle = row.get(f'Donchian_Middle_{buy_period}_{sell_period}')

        if pd.isna(upper) or pd.isna(lower):
            return 'HOLD'

        # Buy Condition: Breakout Upper
        if close > upper:
            return 'BUY'

        # Sell Condition
        if sell_mode == "midline":
            # 중심선 이탈 시 매도
            if middle is not None and not pd.isna(middle) and close < middle:
                return 'SELL'
        else:
            # 기본: 하단 채널 이탈 시 매도
            if close < lower:
                return 'SELL'

        return 'HOLD'
