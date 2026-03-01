import pandas as pd
import numpy as np

def _normalize_numeric_series(series_obj, preferred_cols=("equity", "close", "Close")) -> pd.Series:
    """Series/DataFrame/array를 숫자 Series로 정규화한다."""
    if series_obj is None:
        return pd.Series(dtype=float)

    if isinstance(series_obj, pd.DataFrame):
        if series_obj.empty:
            return pd.Series(dtype=float)
        pick_col = None
        for col in preferred_cols:
            if col in series_obj.columns:
                pick_col = col
                break
        s = series_obj[pick_col] if pick_col else series_obj.iloc[:, 0]
    elif isinstance(series_obj, pd.Series):
        s = series_obj.copy()
    else:
        try:
            s = pd.Series(series_obj)
        except Exception:
            return pd.Series(dtype=float)

    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)

    if isinstance(s.index, pd.DatetimeIndex):
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()

def _infer_periods_per_year(index_like) -> int:
    """DatetimeIndex 간격으로 연환산 주기를 추정한다."""
    if not isinstance(index_like, pd.DatetimeIndex) or len(index_like) < 2:
        return 252
    try:
        deltas = index_like.to_series().diff().dropna().dt.total_seconds()
        if deltas.empty:
            return 252
        med = float(deltas.median())
        if med <= 0:
            return 252
        annual = int(round((365.25 * 24 * 3600) / med))
        return max(1, min(annual, 365))
    except Exception:
        return 252

def _calc_equity_metrics(equity_series: pd.Series, periods_per_year: int = 252) -> dict:
    """equity Series로 기본 성과 지표를 계산한다."""
    eq = _normalize_numeric_series(equity_series)
    if len(eq) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "avg_yearly_mdd": 0.0,
            "final_equity": float(eq.iloc[-1]) if len(eq) else 0.0,
        }

    init_val = float(eq.iloc[0])
    final_val = float(eq.iloc[-1])
    total_return = ((final_val / init_val) - 1.0) * 100.0 if init_val > 0 else 0.0

    if isinstance(eq.index, pd.DatetimeIndex):
        days = max((eq.index[-1] - eq.index[0]).days, 1)
        cagr = ((final_val / init_val) ** (365.0 / days) - 1.0) * 100.0 if init_val > 0 and final_val > 0 else 0.0
    else:
        years = max((len(eq) - 1) / max(periods_per_year, 1), 1 / max(periods_per_year, 1))
        cagr = ((final_val / init_val) ** (1.0 / years) - 1.0) * 100.0 if init_val > 0 and final_val > 0 else 0.0

    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    mdd = float(dd.min()) if len(dd) else 0.0

    rets = eq.pct_change().dropna()
    if len(rets) > 1 and float(rets.std()) > 0:
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(max(periods_per_year, 1)))
    else:
        sharpe = 0.0

    if isinstance(dd.index, pd.DatetimeIndex) and len(dd) > 0:
        yearly_mdd = dd.groupby(dd.index.year).min()
        avg_yearly_mdd = float(yearly_mdd.mean()) if len(yearly_mdd) > 0 else mdd
    else:
        avg_yearly_mdd = mdd

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "sharpe": float(sharpe),
        "avg_yearly_mdd": float(avg_yearly_mdd),
        "final_equity": float(final_val),
    }
