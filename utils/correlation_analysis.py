"""전략 간 상관관계 분석 및 포트폴리오 조합

여러 전략의 equity 곡선을 받아서
상관행렬, 동일비중/리스크패리티 포트폴리오,
분산투자 효과를 계산합니다.
"""

import numpy as np
import pandas as pd


# ════════════════════════ 데이터 정렬 ════════════════════════

def align_equities(equities: dict) -> dict:
    """여러 전략의 equity Series를 공통 날짜 범위로 정렬.

    Parameters
    ----------
    equities : {전략명: pd.Series(DatetimeIndex → 자산)}

    Returns
    -------
    dict with keys:
      - "aligned": {전략명: pd.Series} 공통 날짜만
      - "start": 공통 시작일
      - "end": 공통 종료일
      - "n_days": 공통 거래일 수
    """
    if len(equities) < 2:
        return {"aligned": equities, "start": None, "end": None, "n_days": 0}

    # 공통 인덱스 = 모든 전략의 교집합
    common_idx = None
    for eq in equities.values():
        idx = eq.index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    common_idx = common_idx.sort_values()
    if len(common_idx) < 2:
        return {"aligned": {}, "start": None, "end": None, "n_days": 0}

    aligned = {}
    for name, eq in equities.items():
        s = eq.reindex(common_idx).dropna()
        aligned[name] = s

    return {
        "aligned": aligned,
        "start": common_idx[0],
        "end": common_idx[-1],
        "n_days": len(common_idx),
    }


def daily_returns_df(equities: dict) -> pd.DataFrame:
    """equity dict → 일별 수익률 DataFrame."""
    df = pd.DataFrame({name: eq for name, eq in equities.items()})
    return df.pct_change().dropna()


# ════════════════════════ 상관행렬 ════════════════════════

def correlation_matrix(equities: dict) -> pd.DataFrame:
    """일별 수익률 기반 상관행렬."""
    ret_df = daily_returns_df(equities)
    return ret_df.corr()


# ════════════════════════ 포트폴리오 구성 ════════════════════════

def equal_weight_portfolio(equities: dict, initial_capital: float = 10000.0) -> pd.Series:
    """동일비중 포트폴리오 equity 곡선.

    각 전략의 수익률을 균등 가중 평균하여 합성.
    """
    ret_df = daily_returns_df(equities)
    if ret_df.empty:
        return pd.Series(dtype=float)

    n = len(ret_df.columns)
    port_ret = ret_df.mean(axis=1)  # 동일비중 = 단순 평균

    equity = (1 + port_ret).cumprod() * initial_capital
    # 첫 날 추가
    first_date = ret_df.index[0] - pd.Timedelta(days=1)
    equity = pd.concat([pd.Series([initial_capital], index=[first_date]), equity])
    return equity


def risk_parity_portfolio(equities: dict, initial_capital: float = 10000.0,
                          lookback: int = 60) -> pd.Series:
    """리스크패리티(역변동성 가중) 포트폴리오 equity 곡선.

    각 전략의 최근 lookback일 변동성의 역수로 비중 결정.
    """
    ret_df = daily_returns_df(equities)
    if ret_df.empty or len(ret_df) < lookback + 1:
        return pd.Series(dtype=float)

    port_returns = []
    dates = []

    for i in range(lookback, len(ret_df)):
        window = ret_df.iloc[i - lookback:i]
        vols = window.std()

        # 변동성 0인 자산은 매우 작은 값으로 대체
        vols = vols.replace(0, 1e-10)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()

        day_ret = ret_df.iloc[i]
        port_returns.append((weights * day_ret).sum())
        dates.append(ret_df.index[i])

    if not port_returns:
        return pd.Series(dtype=float)

    port_ret = pd.Series(port_returns, index=dates)
    equity = (1 + port_ret).cumprod() * initial_capital
    first_date = dates[0] - pd.Timedelta(days=1)
    equity = pd.concat([pd.Series([initial_capital], index=[first_date]), equity])
    return equity


# ════════════════════════ 성과 지표 ════════════════════════

def portfolio_metrics(equity: pd.Series) -> dict:
    """포트폴리오 성과 지표 계산."""
    if equity.empty or len(equity) < 2:
        return {"CAGR": 0, "MDD": 0, "Calmar": 0, "Sharpe": 0, "변동성": 0}

    start_v = equity.iloc[0]
    end_v = equity.iloc[-1]
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1e-6)

    cagr = ((end_v / start_v) ** (1 / years) - 1) * 100 if start_v > 0 else 0

    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    mdd = dd.min()

    daily_ret = equity.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100 if len(daily_ret) > 1 else 0
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    calmar = abs(cagr / mdd) if mdd != 0 else 0

    return {
        "CAGR": round(cagr, 2),
        "MDD": round(mdd, 2),
        "Calmar": round(calmar, 2),
        "Sharpe": round(sharpe, 2),
        "변동성": round(vol, 2),
    }


# ════════════════════════ 분산투자 효과 ════════════════════════

def diversification_analysis(individual_equities: dict,
                             combined_equity: pd.Series) -> dict:
    """분산투자 효과 분석.

    Returns
    -------
    dict with keys:
      - "individual_metrics": {전략명: {CAGR, MDD, ...}}
      - "combined_metrics": {CAGR, MDD, ...}
      - "vol_reduction": 변동성 감소율 (%)
      - "mdd_improvement": MDD 개선율 (%)
      - "diversification_ratio": 분산투자비율
      - "evaluation": 종합 평가 텍스트
    """
    ind_metrics = {name: portfolio_metrics(eq) for name, eq in individual_equities.items()}
    comb_metrics = portfolio_metrics(combined_equity)

    # 개별 전략 평균 변동성
    avg_vol = np.mean([m["변동성"] for m in ind_metrics.values()])
    comb_vol = comb_metrics["변동성"]
    vol_reduction = ((avg_vol - comb_vol) / avg_vol * 100) if avg_vol > 0 else 0

    # 개별 전략 평균 MDD
    avg_mdd = np.mean([m["MDD"] for m in ind_metrics.values()])
    comb_mdd = comb_metrics["MDD"]
    mdd_improvement = ((abs(avg_mdd) - abs(comb_mdd)) / abs(avg_mdd) * 100) if avg_mdd != 0 else 0

    # 분산투자비율 = 개별 가중평균 변동성 / 포트폴리오 변동성
    div_ratio = avg_vol / comb_vol if comb_vol > 0 else 1.0

    # 종합 평가
    evaluation = _generate_evaluation(ind_metrics, comb_metrics,
                                       vol_reduction, mdd_improvement, div_ratio)

    return {
        "individual_metrics": ind_metrics,
        "combined_metrics": comb_metrics,
        "vol_reduction": round(vol_reduction, 1),
        "mdd_improvement": round(mdd_improvement, 1),
        "diversification_ratio": round(div_ratio, 2),
        "evaluation": evaluation,
    }


def _generate_evaluation(ind_metrics, comb_metrics, vol_red, mdd_imp, div_ratio):
    """종합 평가 텍스트 생성."""
    lines = []

    if div_ratio >= 1.5:
        lines.append(f"분산투자비율 {div_ratio:.2f}로 **매우 우수한** 분산 효과를 보입니다.")
    elif div_ratio >= 1.2:
        lines.append(f"분산투자비율 {div_ratio:.2f}로 **양호한** 분산 효과를 보입니다.")
    elif div_ratio >= 1.05:
        lines.append(f"분산투자비율 {div_ratio:.2f}로 **미미한** 분산 효과를 보입니다.")
    else:
        lines.append(f"분산투자비율 {div_ratio:.2f}로 분산 효과가 거의 없습니다.")

    if vol_red > 20:
        lines.append(f"변동성이 {vol_red:.1f}% 감소하여 리스크가 크게 줄었습니다.")
    elif vol_red > 10:
        lines.append(f"변동성이 {vol_red:.1f}% 감소하여 리스크가 줄었습니다.")
    elif vol_red > 0:
        lines.append(f"변동성이 {vol_red:.1f}% 소폭 감소했습니다.")
    else:
        lines.append(f"변동성이 감소하지 않았습니다 ({vol_red:.1f}%).")

    if mdd_imp > 20:
        lines.append(f"MDD가 {mdd_imp:.1f}% 개선되어 최대 손실폭이 크게 줄었습니다.")
    elif mdd_imp > 10:
        lines.append(f"MDD가 {mdd_imp:.1f}% 개선되었습니다.")
    elif mdd_imp > 0:
        lines.append(f"MDD가 {mdd_imp:.1f}% 소폭 개선되었습니다.")
    else:
        lines.append(f"MDD가 개선되지 않았습니다 ({mdd_imp:.1f}%).")

    comb_calmar = comb_metrics["Calmar"]
    avg_calmar = np.mean([m["Calmar"] for m in ind_metrics.values()])
    if comb_calmar > avg_calmar * 1.1:
        lines.append(f"포트폴리오 Calmar({comb_calmar:.2f})가 개별 전략 평균({avg_calmar:.2f})보다 높아 **조합이 효과적**입니다.")
    elif comb_calmar > avg_calmar * 0.9:
        lines.append(f"포트폴리오 Calmar({comb_calmar:.2f})가 개별 전략 평균({avg_calmar:.2f})과 유사합니다.")
    else:
        lines.append(f"포트폴리오 Calmar({comb_calmar:.2f})가 개별 전략 평균({avg_calmar:.2f})보다 낮습니다. 조합을 재검토하세요.")

    return "\n".join(lines)
