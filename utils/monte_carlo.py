"""몬테카를로 시뮬레이션 및 켈리 기준 분석

최적 파라미터의 일별 수익률을 부트스트랩하여
미래 성과 분포를 추정하고, 켈리 기준으로
최적 투자 비중을 산출합니다.
"""

import numpy as np
import pandas as pd


# ════════════════════════ 몬테카를로 시뮬레이션 ════════════════════════

def monte_carlo_equity(daily_returns: np.ndarray, n_sims: int = 1000,
                       n_days: int = 252, initial_value: float = 10000.0,
                       seed: int = 42) -> dict:
    """일별 수익률 부트스트랩으로 합성 자산 경로 생성.

    Parameters
    ----------
    daily_returns : 일별 수익률 배열 (예: 0.01 = +1%)
    n_sims : 시뮬레이션 횟수
    n_days : 시뮬레이션 기간 (거래일)
    initial_value : 초기 자산
    seed : 난수 시드

    Returns
    -------
    dict with keys:
      - "paths": np.ndarray (n_sims, n_days+1) 자산 경로
      - "final_values": np.ndarray (n_sims,) 최종 자산
      - "cagr_dist": np.ndarray (n_sims,) CAGR 분포 (%)
      - "mdd_dist": np.ndarray (n_sims,) MDD 분포 (%)
      - "percentiles": dict {5, 25, 50, 75, 95} → 최종자산
    """
    rng = np.random.default_rng(seed)
    n_ret = len(daily_returns)
    if n_ret < 10:
        return {"paths": np.array([]), "final_values": np.array([]),
                "cagr_dist": np.array([]), "mdd_dist": np.array([]),
                "percentiles": {}}

    # 부트스트랩 인덱스 생성
    idx = rng.integers(0, n_ret, size=(n_sims, n_days))
    sampled = daily_returns[idx]  # (n_sims, n_days)

    # 자산 경로 계산
    cum_ret = np.cumprod(1 + sampled, axis=1)
    paths = np.zeros((n_sims, n_days + 1))
    paths[:, 0] = initial_value
    paths[:, 1:] = initial_value * cum_ret

    final_values = paths[:, -1]

    # CAGR 분포 (%)
    years = n_days / 252
    cagr_dist = ((final_values / initial_value) ** (1 / years) - 1) * 100

    # MDD 분포 (%)
    mdd_dist = np.zeros(n_sims)
    for i in range(n_sims):
        peak = np.maximum.accumulate(paths[i])
        dd = (paths[i] - peak) / peak * 100
        mdd_dist[i] = dd.min()

    # 백분위
    pcts = {p: float(np.percentile(final_values, p))
            for p in [5, 25, 50, 75, 95]}

    return {
        "paths": paths,
        "final_values": final_values,
        "cagr_dist": cagr_dist,
        "mdd_dist": mdd_dist,
        "percentiles": pcts,
    }


# ════════════════════════ 켈리 기준 ════════════════════════

def kelly_criterion(daily_returns: np.ndarray) -> dict:
    """일별 수익률로 켈리 기준 최적 투자 비중 계산.

    Parameters
    ----------
    daily_returns : 일별 수익률 배열 (예: 0.01 = +1%)

    Returns
    -------
    dict with keys:
      - "win_rate": float (승률, 0~1)
      - "avg_win": float (평균 수익, %)
      - "avg_loss": float (평균 손실 절대값, %)
      - "payoff_ratio": float (손익비 = avg_win / avg_loss)
      - "kelly_full": float (풀 켈리, %)
      - "kelly_half": float (하프 켈리, %)
      - "kelly_quarter": float (쿼터 켈리, %)
      - "recommendation": str (투자 비중 권고)
      - "grade": str (등급)
    """
    if len(daily_returns) < 20:
        return _empty_kelly()

    # 포지션이 없는 날(수익률 0)은 제외하고 실제 거래일만 분석
    active_returns = daily_returns[daily_returns != 0]
    if len(active_returns) < 10:
        return _empty_kelly()

    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return _empty_kelly()

    win_rate = len(wins) / len(active_returns)
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(np.abs(losses)))

    if avg_loss == 0:
        return _empty_kelly()

    # 손익비 (payoff ratio)
    payoff = avg_win / avg_loss

    # 켈리 공식: f* = (bp - q) / b
    # b = payoff ratio, p = win probability, q = 1-p
    p = win_rate
    q = 1 - p
    kelly_f = (payoff * p - q) / payoff

    kelly_full = max(kelly_f * 100, 0)  # 음수면 0으로 (투자 X)
    kelly_half = kelly_full / 2
    kelly_quarter = kelly_full / 4

    # 등급 및 권고
    grade, recommendation = _kelly_recommendation(kelly_full, payoff, win_rate)

    return {
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win * 100, 4),
        "avg_loss": round(avg_loss * 100, 4),
        "payoff_ratio": round(payoff, 3),
        "kelly_full": round(kelly_full, 2),
        "kelly_half": round(kelly_half, 2),
        "kelly_quarter": round(kelly_quarter, 2),
        "recommendation": recommendation,
        "grade": grade,
    }


def kelly_from_trades(trades: list) -> dict:
    """거래 목록에서 켈리 기준 계산 (pnl_pct 기반).

    trades: list of dict, 각 dict에 "pnl_pct" 키 필요
    """
    pnl_list = [t["pnl_pct"] for t in trades
                if "pnl_pct" in t and t.get("type", "SELL") == "SELL"]
    if not pnl_list:
        pnl_list = [t["pnl_pct"] for t in trades if "pnl_pct" in t]
    if len(pnl_list) < 5:
        return _empty_kelly()

    arr = np.array(pnl_list) / 100  # % → 비율
    return kelly_criterion(arr)


def _kelly_recommendation(kelly_pct: float, payoff: float,
                           win_rate: float) -> tuple:
    """켈리 비중에 따른 등급과 투자 권고."""
    if kelly_pct <= 0:
        return "투자 불가", (
            "켈리 기준에 따르면 기대값이 음수입니다. "
            "이 전략에 자본을 투입하는 것은 권장되지 않습니다."
        )

    if kelly_pct > 50:
        grade = "매우 공격적"
        rec = (
            f"풀 켈리 {kelly_pct:.1f}%는 매우 높아 파산 위험이 큽니다.\n"
            f"실전에서는 **쿼터 켈리({kelly_pct / 4:.1f}%)** 이하를 권장합니다.\n"
            f"승률 {win_rate * 100:.1f}%, 손익비 {payoff:.2f}로 기대값은 양수이나, "
            f"변동성이 극심할 수 있습니다."
        )
    elif kelly_pct > 25:
        grade = "공격적"
        rec = (
            f"풀 켈리 {kelly_pct:.1f}%는 공격적입니다.\n"
            f"**하프 켈리({kelly_pct / 2:.1f}%)** 정도가 적절합니다.\n"
            f"승률 {win_rate * 100:.1f}%, 손익비 {payoff:.2f}로 "
            f"양호한 전략이지만, 자산의 절반 이상을 한 전략에 배분하는 것은 위험합니다."
        )
    elif kelly_pct > 10:
        grade = "적정"
        rec = (
            f"풀 켈리 {kelly_pct:.1f}%는 적정 수준입니다.\n"
            f"**하프 켈리({kelly_pct / 2:.1f}%)~풀 켈리** 사이에서 "
            f"본인의 리스크 허용도에 따라 조절하세요.\n"
            f"승률 {win_rate * 100:.1f}%, 손익비 {payoff:.2f}."
        )
    else:
        grade = "보수적"
        rec = (
            f"풀 켈리 {kelly_pct:.1f}%로, 소규모 비중만 권장됩니다.\n"
            f"전체 포트폴리오의 **{kelly_pct:.1f}% 이하**를 배분하고, "
            f"나머지는 다른 전략이나 현금으로 분산하세요.\n"
            f"승률 {win_rate * 100:.1f}%, 손익비 {payoff:.2f}."
        )

    return grade, rec


def _empty_kelly() -> dict:
    """데이터 부족 시 빈 켈리 결과."""
    return {
        "win_rate": 0, "avg_win": 0, "avg_loss": 0,
        "payoff_ratio": 0, "kelly_full": 0, "kelly_half": 0,
        "kelly_quarter": 0,
        "recommendation": "데이터가 부족하여 분석할 수 없습니다.",
        "grade": "분석 불가",
    }
