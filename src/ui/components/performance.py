import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.engine.backtest_core import _normalize_numeric_series, _infer_periods_per_year, _calc_equity_metrics
from src.utils.formatting import _safe_float

def _render_analysis_title(title: str):
    st.markdown(f"#### 📊 {title}")
    st.divider()

def _apply_dd_hover_format(fig):
    fig.update_traces(hovertemplate="날짜: %{x}<br>낙폭: %{y:.2f}%<extra></extra>")
    return fig

def _apply_return_hover_format(fig, apply_all: bool = False):
    if apply_all:
        fig.update_traces(hovertemplate="날짜: %{x}<br>수익률: %{y:.2f}%<extra></extra>")
    return fig

def _render_performance_analysis(
    equity_series,
    benchmark_series=None,
    strategy_metrics: dict | None = None,
    strategy_label: str = "전략",
    benchmark_label: str = "벤치마크",
    monte_carlo_sims: int = 400,
):
    """공통 성과 분석 UI 렌더러."""
    eq = _normalize_numeric_series(equity_series, preferred_cols=("equity",))
    if len(eq) < 2:
        st.info("분석에 필요한 전략 자산 데이터가 부족합니다.")
        return

    annual_n = _infer_periods_per_year(eq.index)
    calc_metrics = _calc_equity_metrics(eq, periods_per_year=annual_n)

    strategy_metrics = strategy_metrics or {}
    strat_total = _safe_float(strategy_metrics.get("total_return", calc_metrics["total_return"]), calc_metrics["total_return"])
    strat_cagr = _safe_float(strategy_metrics.get("cagr", calc_metrics["cagr"]), calc_metrics["cagr"])
    strat_mdd = _safe_float(strategy_metrics.get("mdd", calc_metrics["mdd"]), calc_metrics["mdd"])
    strat_sharpe = _safe_float(strategy_metrics.get("sharpe", calc_metrics["sharpe"]), calc_metrics["sharpe"])
    strat_avg_yearly_mdd = _safe_float(strategy_metrics.get("avg_yearly_mdd", calc_metrics["avg_yearly_mdd"]), calc_metrics["avg_yearly_mdd"])

    bench_metrics = None
    bench = _normalize_numeric_series(benchmark_series, preferred_cols=("close", "Close", "equity"))
    if len(bench) > 1:
        try:
            if isinstance(eq.index, pd.DatetimeIndex) and isinstance(bench.index, pd.DatetimeIndex):
                bench = bench[(bench.index >= eq.index.min()) & (bench.index <= eq.index.max())]
                if len(bench) > 1:
                    bench = bench.reindex(eq.index, method="ffill").dropna()
            if len(bench) > 1:
                bench_metrics = _calc_equity_metrics(bench, periods_per_year=annual_n)
        except Exception:
            bench_metrics = None

    _render_analysis_title("벤치마크 vs 전략 성과")
    perf_table = {
        "지표": ["총 수익률 (%)", "CAGR (%)", "MDD (%)", "샤프비율", "연평균MDD (%)"],
        strategy_label: [f"{strat_total:.2f}", f"{strat_cagr:.2f}", f"{strat_mdd:.2f}", f"{strat_sharpe:.2f}", f"{strat_avg_yearly_mdd:.2f}"],
        benchmark_label: ["-", "-", "-", "-", "-"],
    }
    if bench_metrics is not None:
        perf_table[benchmark_label] = [
            f"{bench_metrics['total_return']:.2f}", f"{bench_metrics['cagr']:.2f}", 
            f"{bench_metrics['mdd']:.2f}", f"{bench_metrics['sharpe']:.2f}", f"{bench_metrics['avg_yearly_mdd']:.2f}"
        ]
    st.table(pd.DataFrame(perf_table))

    _render_analysis_title("월별 수익률 (%)")
    if isinstance(eq.index, pd.DatetimeIndex):
        monthly_eq = eq.resample("ME").last().dropna()
        monthly_ret = monthly_eq.pct_change().dropna() * 100.0
        if len(monthly_ret) > 0:
            mr_df = pd.DataFrame({"연도": monthly_ret.index.year, "월": monthly_ret.index.month, "수익률": monthly_ret.values})
            pivot = mr_df.pivot_table(index="연도", columns="월", values="수익률", aggfunc="mean")
            if not pivot.empty:
                pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                pivot.columns = [f"{int(m)}월" for m in pivot.columns]
                yearly_comp = mr_df.groupby("연도")["수익률"].apply(lambda x: (np.prod(1.0 + x / 100.0) - 1.0) * 100.0)
                pivot["연간"] = yearly_comp
                def _color_ret(v):
                    if pd.isna(v): return ""
                    fg = "#D32F2F" if v >= 0 else "#1565C0"
                    bg = "#FFF3F3" if v >= 0 else "#E3F2FD"
                    return f"color: {fg}; background-color: {bg}; font-weight: 600"
                st.dataframe(pivot.style.format("{:.2f}").map(_color_ret), use_container_width=True)
    
    _render_analysis_title("몬테카를로 시뮬레이션")
    rets = eq.pct_change().dropna()
    if len(rets) > 15:
        n_sims = int(max(100, min(int(monte_carlo_sims), 2000)))
        n_steps = int(min(len(rets), 750))
        init_cap = float(eq.iloc[-(n_steps+1)])
        rng = np.random.default_rng(42)
        sampled = rng.choice(rets.values, size=(n_sims, n_steps), replace=True)
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = init_cap
        paths[:, 1:] = init_cap * np.cumprod(1.0 + sampled, axis=1)
        
        final_values = paths[:, -1]
        p5, p50, p95 = np.percentile(final_values, [5, 50, 95])
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("하위 5%", f"{p5:,.0f}원")
        mc2.metric("중앙값", f"{p50:,.0f}원")
        mc3.metric("상위 5%", f"{p95:,.0f}원")
        
        fig_mc = go.Figure()
        x_vals = list(range(n_steps + 1))
        fig_mc.add_trace(go.Scatter(x=x_vals, y=np.percentile(paths, 95, axis=0), mode="lines", line=dict(width=0), showlegend=False))
        fig_mc.add_trace(go.Scatter(x=x_vals, y=np.percentile(paths, 5, axis=0), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(255,193,7,0.15)", showlegend=False))
        fig_mc.add_trace(go.Scatter(x=x_vals, y=np.percentile(paths, 50, axis=0), mode="lines", name="중앙 경로", line=dict(color="goldenrod", width=2)))
        fig_mc.update_layout(height=340, margin=dict(l=0, r=0, t=30, b=30))
        st.plotly_chart(fig_mc, use_container_width=True)
    
    _render_analysis_title("켈리 공식")
    if len(rets) > 15:
        w = rets[rets > 0]
        l = rets[rets < 0]
        if len(w) > 0 and len(l) > 0:
            win_p = len(w) / len(rets)
            payoff = w.mean() / abs(l.mean()) if l.mean() != 0 else 0
            kelly = max(0, min(win_p - ( (1-win_p) / payoff if payoff > 0 else 0), 1))
            st.metric("권장 비중 (Kelly)", f"{kelly*100:.1f}%")

def render_performance_table(
    equity_series,
    benchmark_series=None,
    strategy_metrics: dict | None = None,
    strategy_label: str = "전략",
    benchmark_label: str = "벤치마크",
):
    """성과 분석 테이블을 렌더링하는 공개 함수."""
    return _render_performance_analysis(
        equity_series,
        benchmark_series=benchmark_series,
        strategy_metrics=strategy_metrics,
        strategy_label=strategy_label,
        benchmark_label=benchmark_label
    )
