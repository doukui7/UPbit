# -*- coding: utf-8 -*-
"""
벤치마크 이름 자동 변경 + 분석 섹션 추가:
1. 벤치마크 이름을 트렌드 ETF에서 자동으로 가져오기
2. 벤치마크 vs 전략 성과 비교 테이블
3. 월별 수익률 히트맵
4. 몬테카를로 시뮬레이션
5. 켈리 공식
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(r'c:\Users\박두규\Desktop\upbit\app.py', 'r', encoding='utf-8-sig') as f:
    content = f.read()

# ─── 1. Drawdown 차트 뒤에 분석 섹션 추가 ───
# Find the end of Drawdown chart section
marker = '                        st.plotly_chart(fig_dd, use_container_width=True)\n\n    # ══════════════════════════════════════════════════════════════\n    # Tab 2: 수동 주문\n    # ══════════════════════════════════════════════════════════════'

if marker not in content:
    print("❌ Marker not found!")
    # try alternative with \r\n
    marker = '                        st.plotly_chart(fig_dd, use_container_width=True)\r\n\r\n    # ══════════════════════════════════════════════════════════════\r\n    # Tab 2: 수동 주문\r\n    # ══════════════════════════════════════════════════════════════'

if marker not in content:
    print("❌ Marker still not found! Trying partial match...")
    idx = content.find('st.plotly_chart(fig_dd, use_container_width=True)')
    positions = []
    start = 0
    while True:
        pos = content.find('st.plotly_chart(fig_dd, use_container_width=True)', start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    print(f"  Found 'st.plotly_chart(fig_dd, use_container_width=True)' at positions: {positions}")
    
    # Find the one in ISA section
    for pos in positions:
        after = content[pos:pos+200]
        if '수동 주문' in after:
            end_of_line = content.index('\n', pos)
            # Find the Tab 2 comment
            tab2_pos = content.index('# Tab 2: 수동 주문', pos)
            # Get the divider line before it
            divider_start = content.rfind('\n', 0, tab2_pos)
            divider_start = content.rfind('\n', 0, divider_start)  # go back 2 lines for the ═══
            
            marker = content[pos:tab2_pos + len('# Tab 2: 수동 주문')]
            # Also include the ═══ line
            marker_start = content.rfind('══════════', pos, tab2_pos)
            marker_start = content.rfind('\n', 0, marker_start) + 1
            marker = content[pos:content.index('\n', tab2_pos) + 1]
            print(f"  Found ISA section marker at pos {pos}")
            break
    else:
        print("  Could not find ISA section, checking all positions")
        sys.exit(1)

if marker in content:
    analysis_section = '''                        st.plotly_chart(fig_dd, use_container_width=True)

                        # ── 벤치마크 vs 전략 성과 비교 ──
                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:1.5rem 0 1.1rem 0;'>벤치마크 vs 전략 성과</div>",
                            unsafe_allow_html=True,
                        )
                        _metrics = bt.get("metrics", {})
                        _bm_total = float(_bm_df["benchmark_return_pct"].iloc[-1]) if _bm_df is not None and len(_bm_df) > 0 else 0
                        _bm_days = (_bm_df.index[-1] - _bm_df.index[0]).days if _bm_df is not None and len(_bm_df) > 1 else 1
                        _bm_cagr = ((1 + _bm_total / 100) ** (365.0 / max(_bm_days, 1)) - 1) * 100 if _bm_days > 0 else 0
                        _bm_eq_line = _bm_df["benchmark_return_pct"] / 100 + 1 if _bm_df is not None else pd.Series(dtype=float)
                        _bm_pk = _bm_eq_line.cummax() if len(_bm_eq_line) > 0 else pd.Series(dtype=float)
                        _bm_dd_line = ((_bm_eq_line - _bm_pk) / _bm_pk * 100) if len(_bm_pk) > 0 else pd.Series(dtype=float)
                        _bm_mdd = float(_bm_dd_line.min()) if len(_bm_dd_line) > 0 else 0

                        _bm_weekly_ret = _bm_eq_line.pct_change().dropna() if len(_bm_eq_line) > 0 else pd.Series(dtype=float)
                        _bm_sharpe = float((_bm_weekly_ret.mean() / _bm_weekly_ret.std()) * np.sqrt(52)) if len(_bm_weekly_ret) > 1 and _bm_weekly_ret.std() > 0 else 0

                        _perf_data = {
                            "": ["총 수익률 (%)", "CAGR (%)", "MDD (%)", "샤프비율", "연평균MDD (%)"],
                            "위대리 전략": [
                                f"{_metrics.get('total_return', 0):.2f}",
                                f"{_metrics.get('cagr', 0):.2f}",
                                f"{_metrics.get('mdd', 0):.2f}",
                                f"{_metrics.get('sharpe', 0):.2f}",
                                f"{_metrics.get('avg_yearly_mdd', 0):.2f}",
                            ],
                            f"{_trend_bm_label} B&H": [
                                f"{_bm_total:.2f}",
                                f"{_bm_cagr:.2f}",
                                f"{_bm_mdd:.2f}",
                                f"{_bm_sharpe:.2f}",
                                "-",
                            ],
                        }
                        st.table(pd.DataFrame(_perf_data))

                        # ── 연도별/월별 수익률 히트맵 ──
                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:1.5rem 0 1.1rem 0;'>월별 수익률 (%)</div>",
                            unsafe_allow_html=True,
                        )
                        _eq_series = eq_df["equity"]
                        _monthly_eq = _eq_series.resample("ME").last().dropna()
                        _monthly_ret = _monthly_eq.pct_change().dropna() * 100
                        if len(_monthly_ret) > 0:
                            _mr_df = pd.DataFrame({
                                "year": _monthly_ret.index.year,
                                "month": _monthly_ret.index.month,
                                "return": _monthly_ret.values,
                            })
                            _pivot = _mr_df.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
                            _pivot.columns = [f"{m}월" for m in _pivot.columns]
                            # 연간 합계
                            _pivot["연간"] = _pivot.sum(axis=1)

                            def _color_ret(val):
                                if pd.isna(val):
                                    return ""
                                color = "#D32F2F" if val >= 0 else "#1565C0"
                                bg = "#FFF3F3" if val >= 0 else "#E3F2FD"
                                return f"color: {color}; background: {bg}; font-weight: 600"

                            st.dataframe(
                                _pivot.style.format("{:.1f}").map(_color_ret),
                                use_container_width=True,
                            )
                        else:
                            st.info("월별 수익률 데이터가 부족합니다.")

                        # ── 몬테카를로 시뮬레이션 ──
                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:1.5rem 0 1.1rem 0;'>몬테카를로 시뮬레이션</div>",
                            unsafe_allow_html=True,
                        )
                        _weekly_rets = _eq_series.pct_change().dropna()
                        if len(_weekly_rets) > 10:
                            _mc_mu = float(_weekly_rets.mean())
                            _mc_sigma = float(_weekly_rets.std())
                            _mc_n_sims = 500
                            _mc_n_weeks = len(_weekly_rets)
                            np.random.seed(42)
                            _mc_paths = np.zeros((_mc_n_sims, _mc_n_weeks))
                            for _s in range(_mc_n_sims):
                                _rand_rets = np.random.normal(_mc_mu, _mc_sigma, _mc_n_weeks)
                                _mc_paths[_s] = np.cumprod(1 + _rand_rets) * float(_eq_series.iloc[0])

                            _mc_final = _mc_paths[:, -1]
                            _mc_p5 = np.percentile(_mc_final, 5)
                            _mc_p25 = np.percentile(_mc_final, 25)
                            _mc_p50 = np.percentile(_mc_final, 50)
                            _mc_p75 = np.percentile(_mc_final, 75)
                            _mc_p95 = np.percentile(_mc_final, 95)

                            _mc_c1, _mc_c2, _mc_c3, _mc_c4, _mc_c5 = st.columns(5)
                            _mc_c1.metric("5%ile", f"{_mc_p5:,.0f}원")
                            _mc_c2.metric("25%ile", f"{_mc_p25:,.0f}원")
                            _mc_c3.metric("중앙값", f"{_mc_p50:,.0f}원")
                            _mc_c4.metric("75%ile", f"{_mc_p75:,.0f}원")
                            _mc_c5.metric("95%ile", f"{_mc_p95:,.0f}원")

                            fig_mc = go.Figure()
                            # 5~95% 영역
                            _mc_p5_path = np.percentile(_mc_paths, 5, axis=0)
                            _mc_p95_path = np.percentile(_mc_paths, 95, axis=0)
                            _mc_p50_path = np.percentile(_mc_paths, 50, axis=0)
                            _mc_x = list(range(_mc_n_weeks))

                            fig_mc.add_trace(go.Scatter(
                                x=_mc_x, y=_mc_p95_path, mode="lines",
                                name="95%ile", line=dict(width=0), showlegend=False,
                            ))
                            fig_mc.add_trace(go.Scatter(
                                x=_mc_x, y=_mc_p5_path, mode="lines",
                                name="5%ile", line=dict(width=0), showlegend=False,
                                fill="tonexty", fillcolor="rgba(255,193,7,0.15)",
                            ))
                            fig_mc.add_trace(go.Scatter(
                                x=_mc_x, y=_mc_p50_path, mode="lines",
                                name="중앙값", line=dict(color="goldenrod", width=2),
                            ))
                            # 실제 equity
                            fig_mc.add_trace(go.Scatter(
                                x=_mc_x, y=_eq_series.values, mode="lines",
                                name="실제 전략", line=dict(color="gold", width=2),
                            ))
                            fig_mc.update_layout(
                                yaxis_title="자산 (원)", xaxis_title="주 (weeks)",
                                height=350,
                                margin=dict(l=0, r=0, t=30, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            )
                            st.plotly_chart(fig_mc, use_container_width=True)

                            st.caption(f"시뮬레이션 조건: {_mc_n_sims}회 | 주간 평균 수익률 {_mc_mu*100:.3f}% | 표준편차 {_mc_sigma*100:.3f}%")
                        else:
                            st.info("시뮬레이션에 필요한 데이터가 부족합니다.")

                        # ── 켈리 공식 ──
                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:1.5rem 0 1.1rem 0;'>켈리 공식 (Kelly Criterion)</div>",
                            unsafe_allow_html=True,
                        )
                        if len(_weekly_rets) > 10:
                            _win_rets = _weekly_rets[_weekly_rets > 0]
                            _loss_rets = _weekly_rets[_weekly_rets < 0]
                            _win_prob = len(_win_rets) / len(_weekly_rets) if len(_weekly_rets) > 0 else 0
                            _loss_prob = 1 - _win_prob
                            _avg_win = float(_win_rets.mean()) if len(_win_rets) > 0 else 0
                            _avg_loss = float(abs(_loss_rets.mean())) if len(_loss_rets) > 0 else 0.001

                            # Kelly % = W - (1-W) / R  where W=win_prob, R=avg_win/avg_loss
                            _kelly_r = _avg_win / _avg_loss if _avg_loss > 0 else 1
                            _kelly_full = _win_prob - _loss_prob / _kelly_r if _kelly_r > 0 else 0
                            _kelly_half = _kelly_full / 2  # Half Kelly (보수적)
                            _kelly_quarter = _kelly_full / 4  # Quarter Kelly

                            _kc1, _kc2, _kc3, _kc4 = st.columns(4)
                            _kc1.metric("승률", f"{_win_prob*100:.1f}%")
                            _kc2.metric("손익비 (W/L)", f"{_kelly_r:.2f}")
                            _kc3.metric("Full Kelly", f"{_kelly_full*100:.1f}%")
                            _kc4.metric("Half Kelly", f"{_kelly_half*100:.1f}%")

                            st.markdown(
                                f"<div style='background:#F3F8FF;border:1px solid #BBDEFB;border-radius:8px;padding:12px 16px;margin:8px 0;font-size:0.92rem'>"
                                f"<b>해석:</b> 켈리 공식에 따르면 총 자산의 <b>{_kelly_full*100:.1f}%</b>를 투자에 배분하는 것이 "
                                f"기대 자산 성장률을 최대화합니다. 실전에서는 <b>Half Kelly ({_kelly_half*100:.1f}%)</b> 이하를 권장합니다.<br>"
                                f"<span style='color:#666'>평균 수익: +{_avg_win*100:.3f}% | 평균 손실: -{_avg_loss*100:.3f}% | "
                                f"승률: {_win_prob*100:.1f}% ({len(_win_rets)}/{len(_weekly_rets)}주)</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info("켈리 공식 계산에 필요한 데이터가 부족합니다.")

    # ══════════════════════════════════════════════════════════════
    # Tab 2: 수동 주문
    # ══════════════════════════════════════════════════════════════'''

    content = content.replace(marker, analysis_section, 1)
    print("✅ Analysis sections added after Drawdown chart")
else:
    print("❌ Could not find marker to insert analysis sections")

# ─── 2. numpy import 확인 (analysis uses np) ───
# Check if np is imported in the ISA function
if 'import numpy as np' not in content[:5000]:
    # It's likely imported at the top of the file
    pass
# The ISA function uses go.Figure so it already has go imported
# Check if pd and np are available
check_np = "import plotly.graph_objects as go\n    from plotly.subplots import make_subplots"
if check_np in content:
    if 'import numpy as np' not in content.split('def render_kis_isa_mode')[1][:500]:
        # Add numpy import
        content = content.replace(check_np, check_np + "\n    import numpy as np\n    import pandas as pd", 1)
        print("✅ Added numpy/pandas import to ISA function")

with open(r'c:\Users\박두규\Desktop\upbit\app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ Saved ({len(content)} chars)")
