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
    show_drawdown: bool = True,
    show_weight: bool = False,
    equity_df: pd.DataFrame | None = None, # 주식 비중 계산용 (shares, price 포함 시)
    xaxis_range: list | None = None, # 차트 X축 범위 강제 지정 (선택)
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
    bench = pd.Series(dtype=float)
    if benchmark_series is not None:
        try:
            # 벤치마크 입력 타입에 따라 안전하게 Series로 정규화
            if isinstance(benchmark_series, pd.DataFrame):
                if "benchmark_return_pct" in benchmark_series.columns:
                    _ret = pd.to_numeric(benchmark_series["benchmark_return_pct"], errors="coerce").dropna()
                    # 수익률(%) 시계열 -> 지수(1.0 기준) 시계열
                    bench = (1.0 + _ret / 100.0)
                else:
                    bench = _normalize_numeric_series(
                        benchmark_series,
                        preferred_cols=("close", "Close", "equity"),
                    )
            else:
                bench = _normalize_numeric_series(
                    benchmark_series,
                    preferred_cols=("close", "Close", "equity", "benchmark_return_pct"),
                )
                if isinstance(benchmark_series, pd.Series) and benchmark_series.name == "benchmark_return_pct":
                    bench = (1.0 + bench / 100.0)

            if len(bench) > 1 and isinstance(eq.index, pd.DatetimeIndex) and isinstance(bench.index, pd.DatetimeIndex):
                bench = bench[(bench.index >= eq.index.min()) & (bench.index <= eq.index.max())]
                if len(bench) > 1:
                    bench = bench.reindex(eq.index, method="ffill").dropna()

            if len(bench) > 1:
                bench_metrics = _calc_equity_metrics(bench, periods_per_year=annual_n)
        except Exception as e:
            st.error(f"벤치마크 계산 오류: {e}")
            bench_metrics = None

    _render_analysis_title("벤치마크 vs 전략 성과")
    
    # --- Cumulative Performance Chart ---
    try:
        fig_perf = go.Figure()
        
        # 전략 누적 수익률 (%)
        if len(eq) > 0:
            eq_norm = (eq / eq.iloc[0] - 1.0) * 100.0
            
            _now_ts = pd.Timestamp.now().normalize()
            _is_virtual = eq_norm.index[-1] > _now_ts
            
            if _is_virtual and len(eq_norm) > 1:
                # 확정 데이터와 가상 데이터 분리
                eq_confirmed = eq_norm.iloc[:-1]
                eq_virtual = eq_norm.iloc[-1:]
                
                fig_perf.add_trace(go.Scatter(
                    x=eq_confirmed.index, y=eq_confirmed.values,
                    name=f"{strategy_label}(확정)",
                    line=dict(color='#FFD700', width=2.5),
                    hovertemplate="날짜: %{x}<br>전략(확정): %{y:.2f}%<extra></extra>"
                ))
                fig_perf.add_trace(go.Scatter(
                    x=eq_virtual.index, y=eq_virtual.values,
                    name=f"{strategy_label}(실시간/가상)",
                    mode='lines+markers',
                    marker=dict(symbol='star', size=10, color='#FFD700'),
                    line=dict(color='#FFD700', width=2.5, dash='dot'),
                    hovertemplate="날짜: %{x}<br>전략(가상): %{y:.2f}%<extra></extra>"
                ))
            else:
                fig_perf.add_trace(go.Scatter(
                    x=eq_norm.index, y=eq_norm.values,
                    name=strategy_label,
                    line=dict(color='#FFD700', width=2.5), # Gold
                    hovertemplate="날짜: %{x}<br>전략: %{y:.2f}%<extra></extra>"
                ))

            
            # 매수/매도 마커 추가
            trades = strategy_metrics.get('trades', [])
            if trades:
                try:
                    buy_trades = [t for t in trades if t.get('type') == 'buy']
                    sell_trades = [t for t in trades if t.get('type') == 'sell']
                    
                    if buy_trades:
                        b_dates = []
                        b_vals = []
                        for t in buy_trades:
                            d = pd.Timestamp(t['date'])
                            if d in eq_norm.index:
                                b_dates.append(d)
                                b_vals.append(eq_norm.loc[d])
                        
                        if b_dates:
                            fig_perf.add_trace(go.Scatter(
                                x=b_dates, y=b_vals, mode='markers',
                                name='매수',
                                marker=dict(symbol='triangle-up', size=12, color='#00C853', line=dict(width=1, color='white')),
                                hovertemplate="날짜: %{x}<br>매입가 부근<extra></extra>"
                            ))
                            
                    if sell_trades:
                        s_dates = []
                        s_vals = []
                        for t in sell_trades:
                            d = pd.Timestamp(t['date'])
                            if d in eq_norm.index:
                                s_dates.append(d)
                                s_vals.append(eq_norm.loc[d])
                        
                        if s_dates:
                            fig_perf.add_trace(go.Scatter(
                                x=s_dates, y=s_vals, mode='markers',
                                name='매도',
                                marker=dict(symbol='triangle-down', size=12, color='#FF1744', line=dict(width=1, color='white')),
                                hovertemplate="날짜: %{x}<br>매도가 부근<extra></extra>"
                            ))
                except Exception as ex:
                    # 마커 생성 실패해도 차트는 출력되도록 예외 처리
                    pass
            
        # 벤치마크 누적 수익률 (%)
        if len(bench) > 0:
            # 벤치마크 기간을 전략 기간에 맞춤 (이미 위에서 reindex됨)
            bench_norm = (bench / bench.iloc[0] - 1.0) * 100.0
            fig_perf.add_trace(go.Scatter(
                x=bench_norm.index, y=bench_norm.values,
                name=benchmark_label,
                line=dict(color='rgba(150, 150, 150, 0.6)', width=1.5, dash='dot'),
                hovertemplate="날짜: %{x}<br>벤치마크: %{y:.2f}%<extra></extra>"
            ))
            
        fig_perf.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=20),
            yaxis_title="누적 수익률 (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        #그리드 추가
        fig_perf.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.1)')
        fig_perf.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.1)')
        
        if xaxis_range:
            fig_perf.update_xaxes(range=xaxis_range)
        
        st.plotly_chart(fig_perf, use_container_width=True)
    except Exception as e:
        st.error(f"성과 차트 생성 실패: {e}")

    perf_table = {
        "지표": ["총 수익률 (%)", "CAGR (%)", "MDD (%)", "샤프비율", "연평균MDD (%)"],
        strategy_label: [f"{strat_total:.2f}", f"{strat_cagr:.2f}", f"{strat_mdd:.2f}", f"{strat_sharpe:.2f}", f"{strat_avg_yearly_mdd:.2f}"],
        benchmark_label: ["•", "•", "•", "•", "•"],
    }
    if bench_metrics is not None:
        perf_table[benchmark_label] = [
            f"{bench_metrics['total_return']:.2f}", f"{bench_metrics['cagr']:.2f}", 
            f"{bench_metrics['mdd']:.2f}", f"{bench_metrics['sharpe']:.2f}", f"{bench_metrics['avg_yearly_mdd']:.2f}"
        ]
    st.table(pd.DataFrame(perf_table))

    # --- Drawdown Chart ---
    if show_drawdown:
        _render_analysis_title("Drawdown (낙폭)")
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100.0
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, 
            fill='tozeroy', name="Drawdown", 
            line=dict(color='#E53935', width=1),
            fillcolor='rgba(229, 57, 53, 0.2)'
        ))
        fig_dd.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=10), yaxis=dict(ticksuffix="%"))
        if xaxis_range:
            fig_dd.update_xaxes(range=xaxis_range)
        fig_dd = _apply_dd_hover_format(fig_dd)
        st.plotly_chart(fig_dd, use_container_width=True)

    # --- Stock Weight Chart ---
    if show_weight and equity_df is not None:
        _render_analysis_title("전략 자산 비중 (Stock vs Cash)")
        try:
            sdf = equity_df.copy()
            # shares, price, equity 컬럼이 있어야 함
            if "shares" in sdf.columns and "price" in sdf.columns and "equity" in sdf.columns:
                sdf["stock_val"] = sdf["shares"] * sdf["price"]
                sdf["stock_weight"] = (sdf["stock_val"] / sdf["equity"] * 100.0).clip(0, 100)
                sdf["cash_weight"] = 100.0 - sdf["stock_weight"]
                
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(
                    x=sdf.index, y=sdf["stock_weight"], 
                    name="주식(ETF)", stackgroup='one',
                    line=dict(width=0.5, color='#FFC107'),
                    fillcolor='rgba(255, 193, 7, 0.7)'
                ))
                fig_w.add_trace(go.Scatter(
                    x=sdf.index, y=sdf["cash_weight"], 
                    name="현금", stackgroup='one',
                    line=dict(width=0.5, color='#B0BEC5'),
                    fillcolor='rgba(176, 190, 197, 0.5)'
                ))
                fig_w.update_layout(
                    height=220, margin=dict(l=0, r=0, t=10, b=10),
                    yaxis=dict(ticksuffix="%", range=[0, 100]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                if xaxis_range:
                    fig_w.update_xaxes(range=xaxis_range)
                st.plotly_chart(fig_w, use_container_width=True)
        except Exception as e:
            st.error(f"비중 차트 생성 실패: {e}")

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
    
def render_performance_table(
    equity_series,
    benchmark_series=None,
    strategy_metrics: dict | None = None,
    strategy_label: str = "전략",
    benchmark_label: str = "벤치마크",
    show_drawdown: bool = True,
    show_weight: bool = False,
    equity_df: pd.DataFrame | None = None,
):
    """성과 분석 테이블을 렌더링하는 공개 함수."""
    return _render_performance_analysis(
        equity_series,
        benchmark_series=benchmark_series,
        strategy_metrics=strategy_metrics,
        strategy_label=strategy_label,
        benchmark_label=benchmark_label,
        show_drawdown=show_drawdown,
        show_weight=show_weight,
        equity_df=equity_df
    )
