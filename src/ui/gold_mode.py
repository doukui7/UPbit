import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import plotly.graph_objects as go

from src.constants import IS_CLOUD
import src.engine.data_cache as data_cache
from src.ui.components.performance import _apply_return_hover_format, _apply_dd_hover_format
from src.ui.components.triggers import render_strategy_trigger_tab

def render_gold_mode(config, save_config):
    """금(Gold) 현물 거래 모드 - 키움증권 KRX 금시장 (코인 탭과 동일한 구조)"""
    from src.utils.helpers import load_mode_config, save_mode_config
    from src.engine.kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG
    from src.backtest.engine import BacktestEngine

    st.title("🥇 Gold Trading System (키움증권 KRX)")

    # ── 사이드바: 설정 ─────────────────────────────────────
    st.sidebar.header("Gold 설정")

    # API Keys
    try:
        kiwoom_ak      = st.secrets.get("Kiwoom_App_Key", "")
        kiwoom_sk      = st.secrets.get("Kiwoom_Secret_Key", "")
        kiwoom_account = st.secrets.get("KIWOOM_ACCOUNT", "")
    except Exception:
        kiwoom_ak      = os.getenv("Kiwoom_App_Key", "")
        kiwoom_sk      = os.getenv("Kiwoom_Secret_Key", "")
        kiwoom_account = os.getenv("KIWOOM_ACCOUNT", "")

    if IS_CLOUD:
        st.sidebar.info("📱 조회 전용 모드 (Cloud)")
    else:
        with st.sidebar.expander("키움 API Keys", expanded=False):
            kiwoom_ak      = st.text_input("App Key",    value=kiwoom_ak,      type="password", key="kiwoom_ak")
            kiwoom_sk      = st.text_input("Secret Key", value=kiwoom_sk,      type="password", key="kiwoom_sk")
            kiwoom_account = st.text_input("계좌번호",    value=kiwoom_account, key="kiwoom_acc")

    # 전략 설정 (코인 포트폴리오와 동일하게 다중 전략 지원)
    st.sidebar.subheader("전략 설정")
    st.sidebar.caption("여러 전략을 추가하여 포트폴리오를 구성할 수 있습니다.")

    _gold_mode_cfg = load_mode_config("gold")
    _gold_cfg_default = [{"strategy": "Donchian", "buy": 90, "sell": 55, "weight": 100}]
    _gold_cfg = _gold_mode_cfg.get("gold_strategy", None) or config.get("gold_strategy", _gold_cfg_default)

    df_gold_strat = pd.DataFrame(_gold_cfg)
    if IS_CLOUD:
        st.sidebar.dataframe(df_gold_strat, use_container_width=True, hide_index=True)
        edited_gold_strat = df_gold_strat
    else:
        edited_gold_strat = st.sidebar.data_editor(
            df_gold_strat, num_rows="dynamic", use_container_width=True, hide_index=True,
            key="gold_strat_editor",
            column_config={
                "strategy": st.column_config.SelectboxColumn("전략", options=["Donchian", "SMA"], required=True),
                "buy":      st.column_config.NumberColumn("매수", min_value=5, max_value=300, step=1, required=True),
                "sell":     st.column_config.NumberColumn("매도", min_value=0, max_value=300, step=1, required=True, help="Donchian 매도 채널 (SMA는 무시됨, 0=매수의 절반)"),
                "weight":   st.column_config.NumberColumn("비중 %", min_value=1, max_value=100, step=1, required=True),
            },
        )

    # 비중 검증
    gold_total_weight = int(edited_gold_strat["weight"].sum())
    if gold_total_weight > 100:
        st.sidebar.error(f"총 비중이 {gold_total_weight}% 입니다. (100% 이하로 설정해주세요)")
    else:
        gold_cash_weight = 100 - gold_total_weight
        st.sidebar.info(f"투자 비중: {gold_total_weight}% | 현금: {gold_cash_weight}%")

    # 골드 포트폴리오 리스트 생성
    gold_portfolio_list = []
    for _, row in edited_gold_strat.iterrows():
        bp = int(row.get("buy", 90))
        sp = int(row.get("sell", 0) or 0)
        if sp == 0:
            sp = max(5, bp // 2)
        gold_portfolio_list.append({
            "strategy": str(row.get("strategy", "Donchian")),
            "buy_period": bp,
            "sell_period": sp,
            "weight": int(row.get("weight", 100)),
        })

    # 첫 번째 전략 (기본값)
    if gold_portfolio_list:
        _g_first = gold_portfolio_list[0]
        buy_period = _g_first["buy_period"]
        sell_period = _g_first["sell_period"]
    else:
        buy_period = 90
        sell_period = 55

    # 공통 설정
    st.sidebar.subheader("공통 설정")
    _gold_start_default = _gold_mode_cfg.get("gold_start_date", None) or config.get("gold_start_date", "2022-06-01")
    gold_start_date = st.sidebar.date_input(
        "기준 시작일", value=pd.to_datetime(_gold_start_default).date(),
        help="백테스트 평가 시작일", disabled=IS_CLOUD, key="gold_start_date"
    )
    _gold_cap_default = _gold_mode_cfg.get("gold_initial_cap", None) or config.get("gold_initial_cap", 10_000_000)
    gold_initial_cap = st.sidebar.number_input(
        "초기 자본금 (KRW)", value=_gold_cap_default, step=100_000, format="%d",
        disabled=IS_CLOUD, key="gold_initial_cap"
    )
    st.sidebar.caption(f"설정: **{gold_initial_cap:,.0f} KRW**")

    if not IS_CLOUD:
        if st.sidebar.button("💾 Gold 설정 저장", key="gold_save_btn"):
            gold_data = {
                "gold_strategy": edited_gold_strat.to_dict("records"),
                "gold_start_date": str(gold_start_date),
                "gold_initial_cap": gold_initial_cap,
            }
            save_mode_config("gold", gold_data)
            # 전역 config에도 반영 (하위호환)
            new_gold_cfg = config.copy()
            new_gold_cfg.update(gold_data)
            save_config(new_gold_cfg)
            st.sidebar.success("저장 완료!")

    # ── 트레이더 + 백그라운드 워커 초기화 ────────────────────
    from src.engine.data_manager import GoldDataWorker

    @st.cache_resource
    def _get_gold_trader(ak, sk, acct):
        t = KiwoomGoldTrader(is_mock=False)
        t.app_key = ak
        t.app_secret = sk
        t.account_no = acct
        return t

    @st.cache_resource
    def _get_gold_worker(_trader):
        """백그라운드 워커: 잔고/시세/호가를 3초마다 병렬 갱신"""
        w = GoldDataWorker()
        w.configure(_trader)
        w.start()
        return w

    gold_trader = None
    gold_worker = None
    if kiwoom_ak and kiwoom_sk:
        gold_trader = _get_gold_trader(kiwoom_ak, kiwoom_sk, kiwoom_account)
        gold_worker = _get_gold_worker(gold_trader)

    # ── 데이터 로드 헬퍼 (parquet 캐시 → 워커 → CSV 폴백) ──
    def load_gold_data(buy_p: int) -> pd.DataFrame | None:
        """일봉 데이터: parquet 캐시 → 워커 차트 → CSV 폴백."""
        import src.engine.data_cache as data_cache
        # 1순위: parquet 캐시 (사전 다운로드된 대량 데이터)
        cached = data_cache.load_cached_gold()
        if cached is not None and len(cached) >= buy_p + 5:
            return cached
        # 2순위: 백그라운드 워커 차트 데이터
        if gold_worker:
            df_w = gold_worker.get_chart()
            if df_w is not None and len(df_w) >= buy_p + 5:
                return df_w
        # 3순위: CSV 파일 (오프라인 폴백)
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "krx_gold_daily.csv")
        if os.path.exists(csv_path):
            df_csv = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df_csv.columns = [c.lower() for c in df_csv.columns]
            if "open" not in df_csv.columns: df_csv["open"] = df_csv["close"]
            if "high" not in df_csv.columns: df_csv["high"] = df_csv["close"]
            if "low"  not in df_csv.columns: df_csv["low"]  = df_csv["close"]
            return df_csv
        return None

    # ── 탭 구성 ───────────────────────────────────────────
    tab_g1, tab_g2, tab_g3, tab_g4, tab_g5 = st.tabs(
        ["🚀 실시간 모니터링", "🛒 수동 주문", "📊 백테스트", "💳 수수료/세금", "⏰ 트리거"]
    )

    # ══════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링 (코인 탭1과 동일 구조)
    # ══════════════════════════════════════════════════════
    with tab_g1:
        st.header("실시간 금 모니터링")
        _strat_labels = [f"{g['strategy']}({g['buy_period']}/{g['sell_period']}) {g['weight']}%" for g in gold_portfolio_list]
        st.caption(f"전략: {', '.join(_strat_labels)} | 초기자본: {gold_initial_cap:,.0f}원")

        # 새로고침
        col_r1, col_r2 = st.columns([1, 5])
        with col_r1:
            if st.button("🔄 새로고침", key="gold_refresh"):
                for k in list(st.session_state.keys()):
                    if k.startswith("__gc_") or k.startswith("__gt_"):
                        del st.session_state[k]
                st.cache_data.clear()
                st.rerun()

        # 계좌 잔고 (워커에서 읽기 — 블로킹 없음)
        with st.expander("💰 계좌 현황", expanded=True):
            if not gold_worker:
                st.warning("사이드바에서 키움 API Key를 입력해주세요.")
            else:
                bal = gold_worker.get('balance')
                if bal:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("예수금", f"{bal['cash_krw']:,.0f}원")
                    b2.metric("금 보유량", f"{bal['gold_qty']:.4f}g")
                    b3.metric("금 평가금액", f"{bal['gold_eval']:,.0f}원")
                    total_asset = bal['cash_krw'] + bal['gold_eval']
                    pnl = total_asset - gold_initial_cap
                    b4.metric("총 평가", f"{total_asset:,.0f}원", delta=f"{pnl:+,.0f}원")
                elif not gold_worker.is_ready():
                    st.info("데이터 로딩 중... (백그라운드 갱신)")
                    bal = None
                else:
                    st.warning("잔고 조회 실패 (API 인증 확인)")
                    bal = None

        # ── 백테스트 자산현황 vs 실제 자산 비교 ──
        with st.expander("📊 백테스트 vs 실제 자산 비교", expanded=True):
            _bt_max_p = max((g['buy_period'] for g in gold_portfolio_list), default=90)
            _bt_df_gold = load_gold_data(_bt_max_p)

            if _bt_df_gold is None or len(_bt_df_gold) < _bt_max_p + 5:
                st.warning("백테스트용 데이터가 부족합니다. 백테스트 탭에서 사전 다운로드를 실행하세요.")
            else:
                # 캐시 키 생성
                _bt_ck = f"__gt_bt_cmp_{gold_initial_cap}_{gold_start_date}_{len(_bt_df_gold)}_{_bt_df_gold.index[-1]}"
                for _gp in gold_portfolio_list:
                    _bt_ck += f"_{_gp['strategy']}_{_gp['buy_period']}_{_gp['sell_period']}_{_gp['weight']}"

                if _bt_ck not in st.session_state:
                    _total_theo = 0.0
                    _strat_res = []
                    _total_w = sum(g['weight'] for g in gold_portfolio_list)
                    _cash_w = max(0, 100 - _total_w)
                    _reserved = gold_initial_cap * _cash_w / 100

                    for _gp in gold_portfolio_list:
                        _per_cap = gold_initial_cap * _gp['weight'] / 100
                        _eng = BacktestEngine()
                        _sr = _gp['sell_period'] / _gp['buy_period'] if _gp['buy_period'] > 0 else 0.5
                        _bt_r = _eng.run_backtest(
                            ticker=None, df=_bt_df_gold,
                            period=_gp['buy_period'], interval="day",
                            fee=0.003, start_date=str(gold_start_date),
                            initial_balance=_per_cap,
                            strategy_mode=_gp['strategy'],
                            sell_period_ratio=_sr, slippage=0.0,
                        )
                        if "error" not in _bt_r:
                            _perf = _bt_r['performance']
                            _total_theo += _perf['final_equity']
                            _strat_res.append({
                                "label": f"{_gp['strategy']}({_gp['buy_period']}/{_gp['sell_period']})",
                                "weight": _gp['weight'], "initial": _per_cap,
                                "equity": _perf['final_equity'],
                                "return_pct": _perf['total_return'],
                                "status": _perf.get('final_status', 'UNKNOWN'),
                            })
                        else:
                            _total_theo += _per_cap
                            _strat_res.append({
                                "label": f"{_gp['strategy']}({_gp['buy_period']}/{_gp['sell_period']})",
                                "weight": _gp['weight'], "initial": _per_cap,
                                "equity": _per_cap, "return_pct": 0, "status": "ERROR",
                            })

                    _total_theo += _reserved
                    st.session_state[_bt_ck] = {
                        "theo": _total_theo, "strats": _strat_res, "cash": _reserved,
                    }

                _bt_cmp = st.session_state[_bt_ck]
                _theo_total = _bt_cmp["theo"]
                _theo_return = (_theo_total - gold_initial_cap) / gold_initial_cap * 100 if gold_initial_cap > 0 else 0

                # 실제 자산
                _g_bal = gold_worker.get('balance') if gold_worker else None
                _actual_total = (_g_bal['cash_krw'] + _g_bal['gold_eval']) if _g_bal else 0.0
                _actual_return = (_actual_total - gold_initial_cap) / gold_initial_cap * 100 if gold_initial_cap > 0 else 0
                _diff_val = _actual_total - _theo_total

                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("초기 자본", f"{gold_initial_cap:,.0f}원")
                ac2.metric("이론 총자산", f"{_theo_total:,.0f}원", delta=f"{_theo_return:+.2f}%")
                ac3.metric("실제 총자산",
                           f"{_actual_total:,.0f}원" if _g_bal else "조회 불가",
                           delta=f"{_actual_return:+.2f}%" if _g_bal else None)
                ac4.metric("차이 (실제-이론)",
                           f"{_diff_val:+,.0f}원" if _g_bal else "-",
                           delta_color="off" if abs(_diff_val) < 1000 else "inverse")

                # 전략별 상세
                _detail = []
                for _sr in _bt_cmp["strats"]:
                    _detail.append({
                        "전략": _sr["label"], "비중": f"{_sr['weight']}%",
                        "배분 자본": f"{_sr['initial']:,.0f}원",
                        "이론 자산": f"{_sr['equity']:,.0f}원",
                        "수익률": f"{_sr['return_pct']:+.2f}%",
                        "포지션": _sr["status"],
                    })
                if _bt_cmp["cash"] > 0:
                    _cw = max(0, 100 - sum(g['weight'] for g in gold_portfolio_list))
                    _detail.append({
                        "전략": "현금 예비", "비중": f"{_cw}%",
                        "배분 자본": f"{_bt_cmp['cash']:,.0f}원",
                        "이론 자산": f"{_bt_cmp['cash']:,.0f}원",
                        "수익률": "0.00%", "포지션": "CASH",
                    })
                st.dataframe(pd.DataFrame(_detail), use_container_width=True, hide_index=True)

        # 시그널 차트 (다중 전략 지원)
        with st.expander("📊 시그널 모니터링", expanded=True):
            # 가장 큰 buy_period로 데이터 로드
            max_buy_p = max((g['buy_period'] for g in gold_portfolio_list), default=90)
            df_gold = load_gold_data(max_buy_p)

            if df_gold is None or len(df_gold) < max_buy_p + 5:
                st.warning("일봉 데이터 부족. API 연결 또는 krx_gold_daily.csv를 확인하세요.")
            else:
                close_now = float(df_gold['close'].iloc[-1])
                gold_signal_rows = []

                # 전략별 차트 렌더링
                n_strats = len(gold_portfolio_list)
                if n_strats > 0:
                    chart_cols = st.columns(n_strats)

                for gi, gp in enumerate(gold_portfolio_list):
                    g_strat = gp['strategy']
                    g_bp = gp['buy_period']
                    g_sp = gp['sell_period']
                    g_wt = gp['weight']

                    if g_strat == "Donchian":
                        g_upper = df_gold['high'].rolling(window=g_bp).max().shift(1)
                        g_lower = df_gold['low'].rolling(window=g_sp).min().shift(1)
                        g_buy_target = float(g_upper.iloc[-1])
                        g_sell_target = float(g_lower.iloc[-1])
                        g_buy_dist = (close_now - g_buy_target) / g_buy_target * 100 if g_buy_target else 0
                        g_sell_dist = (close_now - g_sell_target) / g_sell_target * 100 if g_sell_target else 0
                        in_pos = False
                        for i in range(len(df_gold)):
                            u = g_upper.iloc[i]; l = g_lower.iloc[i]; c = float(df_gold['close'].iloc[i])
                            if not pd.isna(u) and c > u: in_pos = True
                            elif not pd.isna(l) and c < l: in_pos = False
                        g_signal = ("SELL" if close_now < g_sell_target else "HOLD") if in_pos else \
                                   ("BUY" if close_now > g_buy_target else "WAIT")
                        g_pos_label = "보유" if in_pos else "현금"
                    else:
                        g_sma = df_gold['close'].rolling(window=g_bp).mean()
                        g_buy_target = float(g_sma.iloc[-1])
                        g_sell_target = g_buy_target
                        g_buy_dist = (close_now - g_buy_target) / g_buy_target * 100 if g_buy_target else 0
                        g_sell_dist = g_buy_dist
                        g_signal = "BUY" if close_now > g_buy_target else "SELL"
                        g_pos_label = "보유" if close_now > g_buy_target else "현금"

                    gold_signal_rows.append({
                        "전략": f"{g_strat} {g_bp}/{g_sp}",
                        "비중": f"{g_wt}%",
                        "현재가": f"{close_now:,.0f}",
                        "매수목표": f"{g_buy_target:,.0f}",
                        "매도목표": f"{g_sell_target:,.0f}",
                        "매수이격도": f"{g_buy_dist:+.2f}%",
                        "매도이격도": f"{g_sell_dist:+.2f}%",
                        "포지션": g_pos_label,
                        "시그널": g_signal,
                    })

                    # 차트 렌더링
                    g_sig_color = "green" if g_signal == "BUY" else ("red" if g_signal == "SELL" else ("blue" if g_signal == "WAIT" else "gray"))
                    df_chart = df_gold.iloc[-120:]
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Candlestick(
                        x=df_chart.index, open=df_chart['open'],
                        high=df_chart['high'], low=df_chart['low'],
                        close=df_chart['close'], name='금 일봉',
                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                    ))
                    if g_strat == "Donchian":
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_upper.loc[df_chart.index],
                            name=f'상단({g_bp})', line=dict(color='green', width=1.5, dash='dot')))
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_lower.loc[df_chart.index],
                            name=f'하단({g_sp})', line=dict(color='red', width=1.5, dash='dot')))
                    else:
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_sma.loc[df_chart.index],
                            name=f'SMA({g_bp})', line=dict(color='orange', width=2)))
                    fig_g.update_layout(
                        title=f"KRX 금현물 {g_strat}({g_bp}/{g_sp}) [{g_pos_label}] [{g_buy_dist:+.1f}%]",
                        title_font_color=g_sig_color,
                        height=400, margin=dict(l=0, r=0, t=70, b=30),
                        xaxis_rangeslider_visible=False, showlegend=True,
                        xaxis=dict(showticklabels=True, tickformat='%Y/%m/%d', tickangle=-45, nticks=10),
                        yaxis_title="가격 (원/g)",
                    )
                    with chart_cols[gi]:
                        st.plotly_chart(fig_g, use_container_width=True)

                # 시그널 요약 테이블
                if gold_signal_rows:
                    st.dataframe(pd.DataFrame(gold_signal_rows), use_container_width=True, hide_index=True)

        # 자동매매 규칙
        with st.expander("⚖️ 자동매매 규칙", expanded=False):
            rules_lines = [
                "**실행 시점**: GitHub Actions - 매 평일 KST 09:05",
                "**실행 경로**: 로컬 직접 주문 미사용 → GitHub Actions → 키움 API\n",
            ]
            for gp in gold_portfolio_list:
                if gp['strategy'] == "Donchian":
                    rules_lines.append(f"**{gp['strategy']}({gp['buy_period']}/{gp['sell_period']})** 비중 {gp['weight']}%")
                    rules_lines.append(f"- 매수: 종가 > {gp['buy_period']}일 최고가 → 시장가 매수")
                    rules_lines.append(f"- 매도: 종가 < {gp['sell_period']}일 최저가 → 시장가 매도\n")
                else:
                    rules_lines.append(f"**{gp['strategy']}({gp['buy_period']})** 비중 {gp['weight']}%")
                    rules_lines.append(f"- 매수: 종가 > SMA({gp['buy_period']}) → 시장가 매수")
                    rules_lines.append(f"- 매도: 종가 < SMA({gp['buy_period']}) → 시장가 매도\n")
            rules_lines.append("**수수료**: 키움증권 0.165% (왕복 ~0.34%)")
            st.markdown("\n".join(rules_lines))

    # ══════════════════════════════════════════════════════
    # Tab 2: 수동 주문 — HTS 스타일 (코인 트레이딩 패널과 동일 구조)
    # ══════════════════════════════════════════════════════
    with tab_g2:
        st.header("수동 주문")

        if not gold_trader:
            st.warning("API Key를 사이드바에서 입력해주세요.")
        else:
            GOLD_TICK = 10  # KRX 금현물 호가단위: 10원
            GOLD_MIN_QTY = 1.0  # 1KG 종목 최소 수량: 1g

            def _gold_align_price(price, tick=GOLD_TICK):
                return round(price / tick) * tick

            # ── 호가 선택 콜백 ──
            def _on_gold_ob_select():
                sel = st.session_state.get('_gold_ob_sel', '')
                try:
                    price_str = sel.split(' ', 1)[1].replace(',', '')
                    chosen = int(float(price_str))
                    st.session_state['gold_buy_price'] = chosen
                    st.session_state['gold_sell_price'] = chosen
                except (IndexError, ValueError):
                    pass

            # ═══ 트레이딩 패널 (3초 자동갱신, 워커에서 읽기만 → 블로킹 없음) ═══
            @st.fragment
            def gold_trading_panel():
                # ── 워커에서 즉시 읽기 (API 호출 없음) ──
                g_bal = gold_worker.get('balance') if gold_worker else None
                g_price = gold_worker.get('price', 0) if gold_worker else 0

                g_cash = g_bal['cash_krw'] if g_bal else 0.0
                g_qty  = g_bal['gold_qty'] if g_bal else 0.0
                g_eval = g_bal['gold_eval'] if g_bal else 0.0
                g_hold_val = g_qty * g_price if g_price > 0 else g_eval

                # ── 상단 정보 바 ──
                gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                gc1.metric("현재가 (원/g)", f"{g_price:,.0f}")
                gc2.metric("금 보유", f"{g_qty:.2f}g")
                gc3.metric("평가금액", f"{g_hold_val:,.0f}원")
                gc4.metric("예수금", f"{g_cash:,.0f}원")
                gc5.metric("호가단위", f"{GOLD_TICK}원")

                # ── 최근 거래 알림 바 ──
                g_last_trade = st.session_state.get('_gold_last_trade')
                if g_last_trade:
                    gt_type = g_last_trade.get('type', '')
                    gt_time = g_last_trade.get('time', '')
                    gt_detail = g_last_trade.get('detail', '')
                    is_buy = '매수' in gt_type
                    g_color = '#D32F2F' if is_buy else '#1976D2'
                    gnc1, gnc2 = st.columns([6, 1])
                    gnc1.markdown(
                        f'<div style="padding:6px 12px;border-radius:6px;background:{g_color}22;'
                        f'border-left:4px solid {g_color};font-size:14px;">'
                        f'<b style="color:{g_color}">{gt_type}</b> | {gt_detail} | {gt_time}</div>',
                        unsafe_allow_html=True
                    )
                    if gnc2.button("✕", key="_gold_dismiss"):
                        del st.session_state['_gold_last_trade']
                        st.rerun()

                st.divider()

                # ═══ 메인 레이아웃: 호가창(좌) + 주문(우) ═══
                ob_col, order_col = st.columns([2, 3])

                # ── 좌: 호가창 (HTML 테이블) ──
                with ob_col:
                    price_labels = []

                    ob = gold_worker.get('orderbook') if gold_worker else None

                    if ob and ob.get('asks') and ob.get('bids'):
                        asks = ob['asks']  # 매도호가 (낮→높)
                        bids = ob['bids']  # 매수호가 (높→낮)

                        all_qtys = [a['qty'] for a in asks] + [b['qty'] for b in bids]
                        max_qty = max(all_qtys) if all_qtys else 1

                        html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                        html.append(
                            '<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666">'
                            '<td>구분</td><td style="text-align:right">잔량(g)</td>'
                            '<td style="text-align:right">가격(원)</td>'
                            '<td style="text-align:right">등락</td><td>비율</td></tr>'
                        )

                        ask_prices = []
                        bid_prices = []

                        # 매도호가 (높→낮 순서로 표시)
                        for a in reversed(asks):
                            ap = a['price']
                            aq = a['qty']
                            diff = ((ap / g_price) - 1) * 100 if g_price > 0 else 0
                            bar_w = int(aq / max_qty * 100) if max_qty > 0 else 0
                            ask_prices.append(ap)
                            html.append(
                                f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                                f'<td>매도</td>'
                                f'<td style="text-align:right">{aq:.2f}</td>'
                                f'<td style="text-align:right;font-weight:bold">{ap:,.0f}</td>'
                                f'<td style="text-align:right">{diff:+.2f}%</td>'
                                f'<td><div style="background:#1976D233;height:14px;width:{bar_w}%"></div></td></tr>'
                            )

                        # 현재가 구분선
                        html.append(
                            f'<tr style="background:#FFF3E0;border:2px solid #FF9800;height:32px;font-weight:bold">'
                            f'<td colspan="2" style="color:#E65100">현재가</td>'
                            f'<td style="text-align:right;color:#E65100;font-size:15px">{g_price:,.0f}</td>'
                            f'<td colspan="2"></td></tr>'
                        )

                        # 매수호가 (높→낮 순서)
                        for b in bids:
                            bp = b['price']
                            bq = b['qty']
                            diff = ((bp / g_price) - 1) * 100 if g_price > 0 else 0
                            bar_w = int(bq / max_qty * 100) if max_qty > 0 else 0
                            bid_prices.append(bp)
                            html.append(
                                f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                                f'<td>매수</td>'
                                f'<td style="text-align:right">{bq:.2f}</td>'
                                f'<td style="text-align:right;font-weight:bold">{bp:,.0f}</td>'
                                f'<td style="text-align:right">{diff:+.2f}%</td>'
                                f'<td><div style="background:#D32F2F33;height:14px;width:{bar_w}%"></div></td></tr>'
                            )

                        html.append('</table>')
                        st.markdown(''.join(html), unsafe_allow_html=True)

                        # 호가 선택 selectbox
                        ask_prices.reverse()  # 낮→높 → 높→낮으로 복원
                        for ap in ask_prices:
                            price_labels.append(f"매도 {ap:,.0f}")
                        price_labels.append(f"── {g_price:,.0f} ──")
                        for bp in bid_prices:
                            price_labels.append(f"매수 {bp:,.0f}")

                        st.selectbox(
                            "호가 선택 → 주문가 반영", price_labels,
                            index=len(ask_prices),
                            key="_gold_ob_sel", on_change=_on_gold_ob_select
                        )

                        # 스프레드 정보
                        if asks and bids:
                            spread = asks[0]['price'] - bids[0]['price']
                            spread_pct = (spread / g_price * 100) if g_price > 0 else 0
                            total_ask_qty = sum(a['qty'] for a in asks)
                            total_bid_qty = sum(b['qty'] for b in bids)
                            st.caption(
                                f"스프레드: {spread:,.0f}원 ({spread_pct:.3f}%) | "
                                f"매도잔량: {total_ask_qty:.2f}g | 매수잔량: {total_bid_qty:.2f}g"
                            )
                    else:
                        st.info("호가 데이터를 불러오는 중...")

                # ── 우: 주문 패널 ──
                with order_col:
                    buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                    with buy_tab:
                        gb_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="g_buy_type")

                        if gb_type == "시장가":
                            gb_amount = st.number_input(
                                "매수 금액 (원)", min_value=10_000,
                                value=min(int(g_cash * 0.99), 1_000_000) if g_cash > 10_000 else 10_000,
                                step=10_000, key="g_buy_amt"
                            )
                            # % 버튼
                            gqb1, gqb2, gqb3, gqb4 = st.columns(4)
                            if gqb1.button("10%", key="gb10"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.1)
                                st.rerun()
                            if gqb2.button("25%", key="gb25"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.25)
                                st.rerun()
                            if gqb3.button("50%", key="gb50"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.5)
                                st.rerun()
                            if gqb4.button("100%", key="gb100"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.999)
                                st.rerun()

                            if g_price > 0:
                                st.caption(f"예상 수량: ~{gb_amount / g_price:.2f}g")

                            if st.button("시장가 매수", type="primary", key="g_buy_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gb_amount < 10_000:
                                    st.toast("최소 주문금액: 10,000원", icon="⚠️")
                                elif gb_amount > g_cash:
                                    st.toast(f"예수금 부족 ({g_cash:,.0f}원)", icon="⚠️")
                                else:
                                    with st.spinner("매수 주문 중..."):
                                        if gold_trader.auth():
                                            cur_p = data_cache.get_gold_current_price_local_first(
                                                trader=gold_trader,
                                                code=GOLD_CODE_1KG,
                                                allow_api_fallback=True,
                                                ttl_sec=8.0,
                                            ) or g_price or 1
                                            buy_qty = round(gb_amount / cur_p, 2)
                                            result = gold_trader.send_order("BUY", GOLD_CODE_1KG, qty=buy_qty, ord_tp="3")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 시장가 매수! {gb_amount:,.0f}원 ≈ {buy_qty:.2f}g", icon="🟢")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "시장가 매수", "detail": f"{gb_amount:,.0f}원 ≈ {buy_qty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"매수 실패: {result}", icon="🔴")

                        else:  # 지정가
                            gbc1, gbc2 = st.columns(2)
                            gb_price = gbc1.number_input(
                                "매수 가격 (원/g)", min_value=1,
                                value=_gold_align_price(g_price * 0.99) if g_price > 0 else 100_000,
                                step=GOLD_TICK, key="gold_buy_price"
                            )
                            gb_qty = gbc2.number_input(
                                "매수 수량 (g)", min_value=0.01, value=max(GOLD_MIN_QTY, 1.0),
                                step=0.01, format="%.2f", key="g_buy_qty"
                            )
                            gb_total = gb_price * gb_qty
                            st.caption(f"총액: **{gb_total:,.0f}원** | 호가: {GOLD_TICK}원 | 최소: {GOLD_MIN_QTY}g")

                            gqbc1, gqbc2, gqbc3, gqbc4 = st.columns(4)
                            if gb_price > 0:
                                if gqbc1.button("10%", key="glb10"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.1 / gb_price, 2)
                                    st.rerun()
                                if gqbc2.button("25%", key="glb25"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.25 / gb_price, 2)
                                    st.rerun()
                                if gqbc3.button("50%", key="glb50"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.5 / gb_price, 2)
                                    st.rerun()
                                if gqbc4.button("100%", key="glb100"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.999 / gb_price, 2)
                                    st.rerun()

                            if st.button("지정가 매수", type="primary", key="g_lbuy_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gb_total < 10_000:
                                    st.toast("최소 주문금액: 10,000원", icon="⚠️")
                                elif gb_total > g_cash:
                                    st.toast(f"예수금 부족 ({g_cash:,.0f}원)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매수 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("BUY", GOLD_CODE_1KG, qty=gb_qty, price=gb_price, ord_tp="1")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 지정가 매수 등록! {gb_price:,.0f}원 × {gb_qty:.2f}g", icon="🟢")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "지정가 매수", "detail": f"{gb_price:,.0f}원 × {gb_qty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

                    with sell_tab:
                        gs_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="g_sell_type")

                        if gs_type == "시장가":
                            gs_qty = st.number_input(
                                f"매도 수량 (g)", min_value=0.01,
                                value=g_qty if g_qty > 0 else 1.0,
                                step=0.01, format="%.2f", key="g_sell_qty"
                            )
                            gqs1, gqs2, gqs3, gqs4 = st.columns(4)
                            if gqs1.button("25%", key="gs25"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.25, 2)
                                st.rerun()
                            if gqs2.button("50%", key="gs50"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.5, 2)
                                st.rerun()
                            if gqs3.button("75%", key="gs75"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.75, 2)
                                st.rerun()
                            if gqs4.button("100%", key="gs100"):
                                st.session_state['g_sell_qty'] = round(g_qty, 2)
                                st.rerun()

                            if g_price > 0:
                                st.caption(f"예상 금액: ~{gs_qty * g_price:,.0f}원")

                            if st.button("시장가 매도", type="primary", key="g_sell_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gs_qty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif gs_qty > g_qty:
                                    st.toast(f"보유량 초과 ({g_qty:.2f}g)", icon="⚠️")
                                else:
                                    with st.spinner("매도 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("SELL", GOLD_CODE_1KG, qty=gs_qty, ord_tp="3")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 시장가 매도! {gs_qty:.2f}g", icon="🔴")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "시장가 매도", "detail": f"{gs_qty:.2f}g ≈ {gs_qty * g_price:,.0f}원",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"매도 실패: {result}", icon="🔴")

                        else:  # 지정가
                            gsc1, gsc2 = st.columns(2)
                            gs_price = gsc1.number_input(
                                "매도 가격 (원/g)", min_value=1,
                                value=_gold_align_price(g_price * 1.01) if g_price > 0 else 100_000,
                                step=GOLD_TICK, key="gold_sell_price"
                            )
                            gs_lqty = gsc2.number_input(
                                "매도 수량 (g)", min_value=0.01,
                                value=g_qty if g_qty > 0.01 else 1.0,
                                step=0.01, format="%.2f", key="g_sell_lqty"
                            )
                            gs_total = gs_price * gs_lqty
                            st.caption(f"총액: **{gs_total:,.0f}원** | 호가: {GOLD_TICK}원")

                            gqsc1, gqsc2, gqsc3, gqsc4 = st.columns(4)
                            if g_qty > 0:
                                if gqsc1.button("25%", key="gls25"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.25, 2)
                                    st.rerun()
                                if gqsc2.button("50%", key="gls50"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.5, 2)
                                    st.rerun()
                                if gqsc3.button("75%", key="gls75"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.75, 2)
                                    st.rerun()
                                if gqsc4.button("100%", key="gls100"):
                                    st.session_state['g_sell_lqty'] = round(g_qty, 2)
                                    st.rerun()

                            if st.button("지정가 매도", type="primary", key="g_lsell_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gs_lqty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif gs_lqty > g_qty:
                                    st.toast(f"보유량 초과 ({g_qty:.2f}g)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매도 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("SELL", GOLD_CODE_1KG, qty=gs_lqty, price=gs_price, ord_tp="1")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 지정가 매도 등록! {gs_price:,.0f}원 × {gs_lqty:.2f}g", icon="🔴")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "지정가 매도", "detail": f"{gs_price:,.0f}원 × {gs_lqty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

            gold_trading_panel()

    # ══════════════════════════════════════════════════════
    # Tab 3: 백테스트 (코인 탭4와 동일 구조 - 3개 서브탭)
    # ══════════════════════════════════════════════════════
    with tab_g3:
        gbt1, gbt2 = st.tabs(["📈 단일 백테스트", "🛠️ 파라미터 최적화"])

        # ── 서브탭1: 단일 백테스트 ──────────────────────────
        with gbt1:
            st.header("금현물 단일 백테스트")

            # ── 데이터 가용 범위 + 사전 다운로드 ──
            import src.engine.data_cache as _dc_gold
            _gold_info = _dc_gold.gold_cache_info()
            if _gold_info["exists"]:
                _gi_start = _gold_info["start"]
                _gi_end = _gold_info["end"]
                _gi_start_str = _gi_start.strftime('%Y-%m-%d') if hasattr(_gi_start, 'strftime') else str(_gi_start)[:10]
                _gi_end_str = _gi_end.strftime('%Y-%m-%d') if hasattr(_gi_end, 'strftime') else str(_gi_end)[:10]
                st.info(f"사용 가능 데이터: **{_gold_info['rows']:,}**개 캔들 ({_gi_start_str} ~ {_gi_end_str})")
            else:
                st.warning("캐시된 Gold 데이터가 없습니다. 아래 버튼으로 사전 다운로드하세요.")

            if gold_trader and st.button("Gold 일봉 전체 다운로드 (2014~ 전체)", key="gold_predownload"):
                with st.status("Gold 일봉 다운로드 중...", expanded=True) as dl_status:
                    prog_dl = st.progress(0)
                    log_dl = st.empty()
                    def _dl_progress(fetched, total, msg):
                        pct = min(fetched / total, 1.0) if total > 0 else 1.0
                        prog_dl.progress(pct)
                        log_dl.text(msg)
                    df_dl = _dc_gold.fetch_and_cache_gold(gold_trader, count=5000, progress_callback=_dl_progress)
                    if df_dl is not None and len(df_dl) > 0:
                        dl_status.update(label=f"완료! {len(df_dl):,}개 캔들 다운로드됨", state="complete")
                        st.rerun()
                    else:
                        dl_status.update(label="다운로드 실패", state="error")

            st.divider()

            bt_col1, bt_col2, bt_col3 = st.columns(3)
            with bt_col1:
                bt_strategy = st.selectbox("전략", ["Donchian", "SMA"], key="gold_bt_strat")
            with bt_col2:
                bt_buy_p  = st.number_input("매수 기간", min_value=5, max_value=300, value=buy_period,  step=1, key="gold_bt_buy")
            with bt_col3:
                bt_sell_p = st.number_input("매도 기간", min_value=5, max_value=300, value=sell_period, step=1, key="gold_bt_sell",
                                            help="Donchian 매도 채널 (SMA는 무시됨)")

            bt_start = st.date_input("백테스트 시작일", value=gold_start_date, key="gold_bt_start")
            bt_cap   = st.number_input("초기 자본 (원)", value=gold_initial_cap, step=100_000, format="%d", key="gold_bt_cap")

            if st.button("🚀 백테스트 실행", key="gold_bt_run", type="primary"):
                df_bt = load_gold_data(bt_buy_p)
                if df_bt is None or len(df_bt) < bt_buy_p + 5:
                    st.error("데이터 부족. 사전 다운로드를 실행하세요.")
                else:
                    st.caption(f"조회된 캔들: {len(df_bt):,}개 ({df_bt.index[0].strftime('%Y-%m-%d')} ~ {df_bt.index[-1].strftime('%Y-%m-%d')})")
                    with st.spinner("백테스트 실행 중..."):
                        engine = BacktestEngine()
                        result = engine.run_backtest(
                            ticker=None, df=df_bt,
                            period=bt_buy_p,
                            interval="day",
                            fee=0.003,
                            start_date=str(bt_start),
                            initial_balance=bt_cap,
                            strategy_mode=bt_strategy,
                            sell_period_ratio=(bt_sell_p / bt_buy_p) if bt_strategy == "Donchian" else 1.0,
                            slippage=0.0,
                        )

                    if "error" in result:
                        st.error(f"백테스트 오류: {result['error']}")
                    else:
                        p = result["performance"]
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("총 수익률",  f"{p['total_return']:+.2f}%")
                        r2.metric("CAGR",      f"{p['cagr']:+.2f}%")
                        r3.metric("MDD",       f"{p['mdd']:.2f}%")
                        r4.metric("샤프 비율",  f"{p['sharpe']:.2f}")
                        r5, r6, r7, r8 = st.columns(4)
                        r5.metric("매매 횟수",  f"{p['trade_count']}회")
                        r6.metric("승률",      f"{p['win_rate']:.1f}%")
                        r7.metric("최종 자산",  f"{p['final_equity']:,.0f}원")
                        calmar = abs(p['cagr'] / p['mdd']) if p['mdd'] != 0 else 0
                        r8.metric("Calmar",    f"{calmar:.2f}")

                        # ── B&H 성과 비교 ─────────────────────────────
                        df_result_full = result.get("df")
                        equity_curve   = result.get("equity_curve")
                        df_bt_chart_bh = df_bt.loc[df_bt.index >= pd.Timestamp(str(bt_start))]

                        if not df_bt_chart_bh.empty:
                            bh_close = df_bt_chart_bh["close"]
                            bh_start_val = bh_close.iloc[0]
                            bh_end_val = bh_close.iloc[-1]
                            bh_total_ret = (bh_end_val / bh_start_val - 1) * 100
                            bh_days = (bh_close.index[-1] - bh_close.index[0]).days
                            bh_cagr = ((bh_end_val / bh_start_val) ** (365 / bh_days) - 1) * 100 if bh_days > 0 else 0
                            bh_peak = bh_close.cummax()
                            bh_dd_all = (bh_close - bh_peak) / bh_peak * 100
                            bh_mdd = bh_dd_all.min()
                            bh_daily_ret = bh_close.pct_change().dropna()
                            bh_sharpe = (bh_daily_ret.mean() / bh_daily_ret.std() * np.sqrt(365)) if bh_daily_ret.std() > 0 else 0
                            bh_calmar = abs(bh_cagr / bh_mdd) if bh_mdd != 0 else 0

                            st.subheader("전략 vs Buy & Hold")
                            _cmp_df = pd.DataFrame({
                                "": ["전략", "Buy & Hold"],
                                "총수익률(%)": [f"{p['total_return']:+.2f}", f"{bh_total_ret:+.2f}"],
                                "CAGR(%)": [f"{p['cagr']:+.2f}", f"{bh_cagr:+.2f}"],
                                "MDD(%)": [f"{p['mdd']:.2f}", f"{bh_mdd:.2f}"],
                                "Sharpe": [f"{p['sharpe']:.2f}", f"{bh_sharpe:.2f}"],
                                "Calmar": [f"{calmar:.2f}", f"{bh_calmar:.2f}"],
                            })
                            st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

                        # ── 연도별 성과 (상세 컬럼) ──────────────────
                        if df_result_full is not None and equity_curve is not None:
                            eq_series = pd.Series(equity_curve, index=df_result_full.index[-len(equity_curve):])
                            yearly_rows_g = []
                            years = eq_series.index.year.unique()
                            for yr in sorted(years):
                                yr_eq = eq_series[eq_series.index.year == yr]
                                if yr_eq.empty:
                                    continue
                                yr_start_eq = yr_eq.iloc[0]
                                yr_end_eq   = yr_eq.iloc[-1]
                                yr_ret      = (yr_end_eq / yr_start_eq - 1) * 100
                                peak_yr     = yr_eq.cummax()
                                yr_mdd      = ((yr_eq - peak_yr) / peak_yr * 100).min()
                                # B&H 연도별 비교
                                bh_yr_ret = 0.0
                                bh_yr_mdd = 0.0
                                if not df_bt_chart_bh.empty:
                                    bh_yr = bh_close[bh_close.index.year == yr]
                                    if not bh_yr.empty:
                                        bh_yr_ret = (bh_yr.iloc[-1] / bh_yr.iloc[0] - 1) * 100
                                        bh_yr_pk = bh_yr.cummax()
                                        bh_yr_mdd = ((bh_yr - bh_yr_pk) / bh_yr_pk * 100).min()
                                # 상태: 연도 마지막 일의 포지션
                                yr_trades = [t for t in p.get("trades", []) if pd.Timestamp(t["date"]).year <= yr]
                                last_type = yr_trades[-1]["type"] if yr_trades else "—"
                                yr_state = "보유" if last_type == "buy" else "현금"
                                yearly_rows_g.append({
                                    "연도": yr,
                                    "전략(%)": f"{yr_ret:+.2f}",
                                    "B&H(%)": f"{bh_yr_ret:+.2f}",
                                    "전략MDD": f"{yr_mdd:.2f}",
                                    "B&H MDD": f"{bh_yr_mdd:.2f}",
                                    "상태": yr_state,
                                })
                            if yearly_rows_g:
                                st.subheader("연도별 성과")
                                st.dataframe(pd.DataFrame(yearly_rows_g), use_container_width=True, hide_index=True)

                        # ── 에쿼티 커브 차트 ──────────────────────────
                        if equity_curve is not None and len(equity_curve) > 0:
                            df_eq = pd.DataFrame({"equity": equity_curve})
                            df_eq.index = df_result_full.index[-len(equity_curve):]
                            df_eq["return_pct"] = (df_eq["equity"] / bt_cap - 1) * 100
                            # Buy & Hold 비교
                            df_bt_chart = df_bt.loc[df_bt.index >= pd.Timestamp(str(bt_start))]
                            if not df_bt_chart.empty:
                                bh_base = df_bt_chart["close"].iloc[0]
                                bh_pct  = (df_bt_chart["close"] / bh_base - 1) * 100

                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                x=df_eq.index, y=df_eq["return_pct"], mode="lines",
                                name="전략", line=dict(color="gold", width=2)
                            ))
                            if not df_bt_chart.empty:
                                fig_eq.add_trace(go.Scatter(
                                    x=bh_pct.index, y=bh_pct.values, mode="lines",
                                    name="Buy & Hold", line=dict(color="gray", width=1, dash="dot")
                                ))
                            fig_eq.update_layout(
                                title=f"누적 수익률 ({bt_strategy} {bt_buy_p}/{bt_sell_p})",
                                yaxis_title="수익률 (%)", height=350,
                                margin=dict(l=0, r=0, t=70, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            fig_eq = _apply_return_hover_format(fig_eq, apply_all=True)
                            st.plotly_chart(fig_eq, use_container_width=True)

                            # ── DD 차트 ──────────────────────────────
                            dd_idx = df_result_full.index[-len(equity_curve):]
                            strat_dd = df_result_full.loc[dd_idx, "drawdown"] if "drawdown" in df_result_full.columns else None
                            if strat_dd is not None:
                                fig_dd = go.Figure()
                                fig_dd.add_trace(go.Scatter(
                                    x=dd_idx, y=strat_dd.values, mode="lines",
                                    name="전략 DD", line=dict(color="crimson", width=2),
                                    fill="tozeroy", fillcolor="rgba(220,20,60,0.15)"
                                ))
                                if not df_bt_chart.empty:
                                    bh_peak = df_bt_chart["close"].cummax()
                                    bh_dd = (df_bt_chart["close"] - bh_peak) / bh_peak * 100
                                    fig_dd.add_trace(go.Scatter(
                                        x=bh_dd.index, y=bh_dd.values, mode="lines",
                                        name="Buy & Hold DD", line=dict(color="gray", width=1, dash="dot")
                                    ))
                                fig_dd.update_layout(
                                    title="Drawdown",
                                    yaxis_title="DD (%)", height=280,
                                    margin=dict(l=0, r=0, t=70, b=30),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                                )
                                fig_dd = _apply_dd_hover_format(fig_dd)
                                st.plotly_chart(fig_dd, use_container_width=True)

                        # ── 거래 내역 ─────────────────────────────────
                        with st.expander("거래 내역"):
                            trades_list = p.get("trades", [])
                            if trades_list:
                                st.dataframe(pd.DataFrame(trades_list), use_container_width=True, hide_index=True)
                            else:
                                st.info("실행된 거래가 없습니다.")

        # ── 서브탭2: 파라미터 최적화 ────────────────────────
        with gbt2:
            st.header("파라미터 최적화")

            # 데이터 가용 범위 표시
            _gold_info_opt = _dc_gold.gold_cache_info()
            if _gold_info_opt["exists"]:
                _gio_s = _gold_info_opt["start"]
                _gio_e = _gold_info_opt["end"]
                _gio_s_str = _gio_s.strftime('%Y-%m-%d') if hasattr(_gio_s, 'strftime') else str(_gio_s)[:10]
                _gio_e_str = _gio_e.strftime('%Y-%m-%d') if hasattr(_gio_e, 'strftime') else str(_gio_e)[:10]
                st.info(f"사용 가능 데이터: **{_gold_info_opt['rows']:,}**개 캔들 ({_gio_s_str} ~ {_gio_e_str})")
            else:
                st.warning("캐시된 Gold 데이터가 없습니다. 백테스트 탭에서 사전 다운로드를 실행하세요.")

            opt_strat_g = st.selectbox("전략", ["Donchian", "SMA"], key="gold_opt_strat")

            with st.form("gold_optimization_form"):
                gopt_method = st.radio("최적화 방법", ["Grid Search (전수 탐색)", "Optuna (베이지안 최적화)"], horizontal=True, key="gold_opt_method")
                use_optuna_g = "Optuna" in gopt_method

                if opt_strat_g == "Donchian":
                    st.markdown("##### 매수 채널 기간")
                    goc1, goc2, goc3 = st.columns(3)
                    g_buy_start = goc1.number_input("시작", 5, 300, 30,  key="gold_opt_buy_start")
                    g_buy_end   = goc2.number_input("끝",   5, 300, 150, key="gold_opt_buy_end")
                    g_buy_step  = goc3.number_input("간격", 1, 50,  5,   key="gold_opt_buy_step")
                    st.markdown("##### 매도 채널 기간")
                    goc4, goc5, goc6 = st.columns(3)
                    g_sell_start = goc4.number_input("시작", 5, 300, 10, key="gold_opt_sell_start")
                    g_sell_end   = goc5.number_input("끝",   5, 300, 80, key="gold_opt_sell_end")
                    g_sell_step  = goc6.number_input("간격", 1, 50,  5,  key="gold_opt_sell_step")
                    g_sell_mode_label = st.radio("매도 방식", ["하단선 (Lower)", "중심선 (Midline)", "두 방법 비교"], horizontal=True, key="gold_opt_sell_mode")
                else:
                    st.markdown("##### SMA 기간")
                    goc1, goc2, goc3 = st.columns(3)
                    g_sma_start = goc1.number_input("시작", 5, 300, 10, key="gold_opt_sma_start")
                    g_sma_end   = goc2.number_input("끝",   5, 300, 100, key="gold_opt_sma_end")
                    g_sma_step  = goc3.number_input("간격", 1, 50,  5,   key="gold_opt_sma_step")

                if use_optuna_g:
                    st.divider()
                    st.markdown("##### Optuna 설정")
                    uoc1, uoc2 = st.columns(2)
                    g_n_trials  = uoc1.number_input("탐색 횟수", 50, 2000, 200, step=50, key="gold_optuna_trials")
                    g_obj_label = uoc2.selectbox("목적함수", ["Calmar (CAGR/|MDD|)", "Sharpe", "수익률 (Return)", "MDD 최소"], key="gold_optuna_obj")

                st.divider()
                g_opt_start = st.date_input("시작일", value=gold_start_date, key="gold_opt_start")
                _gfee_col, _gslip_col = st.columns(2)
                g_opt_fee   = _gfee_col.number_input("수수료 (%)", value=0.3, format="%.2f", key="gold_opt_fee") / 100
                g_opt_slip  = _gslip_col.number_input("슬리피지 (%)", value=0.0, min_value=0.0, max_value=2.0, step=0.05, format="%.2f", key="gold_opt_slip")
                g_opt_cap   = st.number_input("초기 자본 (원)", value=gold_initial_cap, step=100_000, format="%d", key="gold_opt_cap")
                _gsort_col, _gtrade_col = st.columns(2)
                g_sort_label = _gsort_col.selectbox("정렬 기준", ["Calmar", "Sharpe", "CAGR", "MDD 최소", "수익률"], key="gold_opt_sort")
                g_min_trades = _gtrade_col.number_input("최소 매매 횟수", value=0, min_value=0, max_value=100, step=1, key="gold_opt_min_trades")

                gopt_submitted = st.form_submit_button("최적화 시작", type="primary")

            if gopt_submitted:
                import plotly.express as px
                df_opt_src = load_gold_data(max(
                    g_buy_end if opt_strat_g == "Donchian" else g_sma_end, 300
                ))
                if df_opt_src is None or df_opt_src.empty:
                    st.error("데이터 로드 실패. 백테스트 탭에서 사전 다운로드를 실행하세요.")
                else:
                    st.caption(f"조회된 캔들: {len(df_opt_src):,}개 ({df_opt_src.index[0].strftime('%Y-%m-%d')} ~ {df_opt_src.index[-1].strftime('%Y-%m-%d')})")
                    with st.status("최적화 진행 중...", expanded=True) as gopt_status:
                        prog_bar_g  = st.progress(0)
                        log_area_g  = st.empty()

                        def g_opt_progress(idx, total, msg):
                            pct = min(idx / total, 1.0) if total > 0 else 0
                            prog_bar_g.progress(pct)
                            log_area_g.text(f"{msg} ({idx}/{total})")

                        # sell_mode 결정
                        _g_sell_modes = ["lower"]
                        if opt_strat_g == "Donchian":
                            if g_sell_mode_label == "중심선 (Midline)":
                                _g_sell_modes = ["midline"]
                            elif g_sell_mode_label == "두 방법 비교":
                                _g_sell_modes = ["lower", "midline"]

                        engine_g = BacktestEngine()
                        gopt_results = []
                        try:
                            for _gsm in _g_sell_modes:
                                if use_optuna_g:
                                    obj_map_g = {"Calmar (CAGR/|MDD|)": "calmar", "Sharpe": "sharpe", "수익률 (Return)": "return", "MDD 최소": "mdd"}
                                    obj_key_g = obj_map_g.get(g_obj_label, "calmar")
                                    if opt_strat_g == "Donchian":
                                        opt_res_g = engine_g.optuna_optimize(
                                            df_opt_src, strategy_mode="Donchian",
                                            buy_range=(g_buy_start, g_buy_end),
                                            sell_range=(g_sell_start, g_sell_end),
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            n_trials=g_n_trials,
                                            objective_metric=obj_key_g,
                                            progress_callback=g_opt_progress,
                                            sell_mode=_gsm,
                                        )
                                    else:
                                        opt_res_g = engine_g.optuna_optimize(
                                            df_opt_src, strategy_mode="SMA",
                                            buy_range=(g_sma_start, g_sma_end),
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            n_trials=g_n_trials,
                                            objective_metric=obj_key_g,
                                            progress_callback=g_opt_progress,
                                        )
                                    _trials = opt_res_g.get("trials", [])
                                    for _t in _trials:
                                        _t["sell_mode"] = _gsm
                                    gopt_results.extend(_trials)
                                    best_params  = opt_res_g.get("best_params", {})
                                    gopt_status.update(label=f"Optuna 완료! ({_gsm})", state="complete")
                                    st.success(f"[{_gsm}] 최적 파라미터: {best_params} | 목적함수 값: {opt_res_g['best_value']:.2f}")
                                else:
                                    if opt_strat_g == "Donchian":
                                        buy_r  = range(g_buy_start,  g_buy_end  + 1, g_buy_step)
                                        sell_r = range(g_sell_start, g_sell_end + 1, g_sell_step)
                                        _grid_res = engine_g.optimize_donchian(
                                            df_opt_src, buy_range=buy_r, sell_range=sell_r,
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            progress_callback=g_opt_progress,
                                            sell_mode=_gsm,
                                        )
                                    else:
                                        sma_r = range(g_sma_start, g_sma_end + 1, g_sma_step)
                                        _grid_res = engine_g.optimize_sma(
                                            df_opt_src, sma_range=sma_r,
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            progress_callback=g_opt_progress,
                                        )
                                    for _t in _grid_res:
                                        _t["sell_mode"] = _gsm
                                    gopt_results.extend(_grid_res)
                                    gopt_status.update(label=f"Grid Search 완료! ({_gsm})", state="complete")

                        except Exception as e:
                            gopt_status.update(label=f"오류: {e}", state="error")

                    if gopt_results:
                        df_opt_res = pd.DataFrame(gopt_results)
                        df_opt_res["calmar"] = df_opt_res.apply(
                            lambda r: abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0, axis=1
                        )

                        # 최소 매매 횟수 필터
                        if g_min_trades > 0:
                            df_opt_res = df_opt_res[df_opt_res["trade_count"] >= g_min_trades]

                        # 정렬 기준 적용
                        _sort_map = {"Calmar": "calmar", "Sharpe": "sharpe", "CAGR": "cagr", "MDD 최소": "mdd", "수익률": "total_return"}
                        _sort_col = _sort_map.get(g_sort_label, "calmar")
                        _sort_asc = True if _sort_col == "mdd" else False
                        df_opt_res_sorted = df_opt_res.sort_values(_sort_col, ascending=_sort_asc)

                        st.subheader(f"상위 20개 파라미터 (정렬: {g_sort_label})")
                        _base_cols = ["Buy Period", "Sell Period"] if opt_strat_g == "Donchian" else ["sma_period"]
                        if len(_g_sell_modes) > 1:
                            _base_cols.append("sell_mode")
                        disp_cols = _base_cols + ["total_return", "cagr", "mdd", "sharpe", "calmar", "win_rate", "trade_count"]
                        disp_cols = [c for c in disp_cols if c in df_opt_res_sorted.columns]
                        st.dataframe(
                            df_opt_res_sorted[disp_cols].head(20).style.format({
                                "total_return": "{:.2f}%", "cagr": "{:.2f}%",
                                "mdd": "{:.2f}%", "sharpe": "{:.2f}",
                                "calmar": "{:.2f}", "win_rate": "{:.1f}%",
                            }),
                            use_container_width=True, hide_index=True
                        )

                        # 히트맵 (Donchian Grid Search일 때)
                        if opt_strat_g == "Donchian" and not use_optuna_g and \
                                "Buy Period" in df_opt_res.columns and "Sell Period" in df_opt_res.columns:
                            st.subheader("Calmar 히트맵 (Buy × Sell)")
                            try:
                                df_heat = df_opt_res.pivot_table(index="Buy Period", columns="Sell Period", values="calmar")
                                fig_heat = px.imshow(
                                    df_heat, color_continuous_scale="RdYlGn",
                                    labels=dict(x="Sell Period", y="Buy Period", color="Calmar"),
                                    aspect="auto"
                                )
                                fig_heat.update_layout(height=500, margin=dict(l=0, r=0, t=70, b=30))
                                st.plotly_chart(fig_heat, use_container_width=True)
                            except Exception:
                                pass

                        # 전체 결과 다운로드
                        csv_data_g = df_opt_res_sorted.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 전체 결과 다운로드 (CSV)", data=csv_data_g,
                                           file_name="gold_optimization_results.csv", mime="text/csv")

                        # ── 파라미터 선택 백테스트 ─────────────────────
                        st.divider()
                        st.subheader("파라미터 백테스트")

                        # 상위 결과 목록 생성 (최대 20개)
                        _top_n = min(20, len(df_opt_res_sorted))
                        _sel_options = []
                        for _ri in range(_top_n):
                            _r = df_opt_res_sorted.iloc[_ri]
                            if opt_strat_g == "Donchian":
                                _lbl = f"#{_ri+1}  Donchian {int(_r['Buy Period'])}/{int(_r['Sell Period'])}"
                            else:
                                _sp = int(_r.get("sma_period", _r.get("SMA Period", 20)))
                                _lbl = f"#{_ri+1}  SMA {_sp}"
                            _lbl += f"  |  Calmar {_r['calmar']:.2f}  |  CAGR {_r['cagr']:.2f}%  |  MDD {_r['mdd']:.2f}%"
                            _sel_options.append(_lbl)

                        _sel_idx = st.selectbox("백테스트할 파라미터 선택", range(_top_n),
                                                format_func=lambda x: _sel_options[x], index=0,
                                                key="gold_opt_bt_select")

                        best_row = df_opt_res_sorted.iloc[_sel_idx]
                        if opt_strat_g == "Donchian":
                            best_buy_p = int(best_row["Buy Period"])
                            best_sell_p = int(best_row["Sell Period"])
                            st.info(f"선택 파라미터: **Donchian {best_buy_p}/{best_sell_p}**  |  Calmar {best_row['calmar']:.2f}  |  CAGR {best_row['cagr']:.2f}%  |  MDD {best_row['mdd']:.2f}%")
                        else:
                            best_buy_p = int(best_row.get("sma_period", best_row.get("SMA Period", 20)))
                            best_sell_p = 0
                            st.info(f"선택 파라미터: **SMA {best_buy_p}**  |  Calmar {best_row['calmar']:.2f}  |  CAGR {best_row['cagr']:.2f}%  |  MDD {best_row['mdd']:.2f}%")

                        try:
                            _sell_ratio = (best_sell_p / best_buy_p) if opt_strat_g == "Donchian" and best_buy_p > 0 else 1.0
                            _strat_mode = "Donchian" if opt_strat_g == "Donchian" else "SMA Strategy"
                            best_result = engine_g.run_backtest(
                                ticker=None, df=df_opt_src,
                                period=best_buy_p,
                                interval="day",
                                fee=g_opt_fee,
                                start_date=str(g_opt_start),
                                initial_balance=g_opt_cap,
                                strategy_mode=_strat_mode,
                                sell_period_ratio=_sell_ratio,
                                slippage=0.0,
                            )
                            if "error" not in best_result:
                                bp = best_result["performance"]
                                bc1, bc2, bc3, bc4 = st.columns(4)
                                bc1.metric("총 수익률", f"{bp['total_return']:+.2f}%")
                                bc2.metric("CAGR", f"{bp['cagr']:+.2f}%")
                                bc3.metric("MDD", f"{bp['mdd']:.2f}%")
                                bc4.metric("샤프 비율", f"{bp['sharpe']:.2f}")
                                bc5, bc6, bc7, bc8 = st.columns(4)
                                bc5.metric("매매 횟수", f"{bp['trade_count']}회")
                                bc6.metric("승률", f"{bp['win_rate']:.1f}%")
                                bc7.metric("최종 자산", f"{bp['final_equity']:,.0f}원")
                                _bcalmar = abs(bp['cagr'] / bp['mdd']) if bp['mdd'] != 0 else 0
                                bc8.metric("Calmar", f"{_bcalmar:.2f}")

                                # 에쿼티 + DD 차트
                                _best_df = best_result.get("df")
                                _best_eq = best_result.get("equity_curve")
                                if _best_df is not None and _best_eq is not None and len(_best_eq) > 0:
                                    _beq_idx = _best_df.index[-len(_best_eq):]
                                    _beq_ret = (np.array(_best_eq) / g_opt_cap - 1) * 100

                                    # Buy & Hold 비교
                                    _bh_chart = df_opt_src.loc[df_opt_src.index >= pd.Timestamp(str(g_opt_start))]
                                    _bh_base = _bh_chart["close"].iloc[0] if not _bh_chart.empty else 1
                                    _bh_ret = ((_bh_chart["close"] / _bh_base - 1) * 100) if not _bh_chart.empty else pd.Series()

                                    # 에쿼티(수익률) 차트
                                    _title_lbl = f"Donchian {best_buy_p}/{best_sell_p}" if opt_strat_g == "Donchian" else f"SMA {best_buy_p}"
                                    fig_best_eq = go.Figure()
                                    fig_best_eq.add_trace(go.Scatter(
                                        x=_beq_idx, y=_beq_ret, mode="lines",
                                        name="전략", line=dict(color="gold", width=2)
                                    ))
                                    if not _bh_chart.empty:
                                        fig_best_eq.add_trace(go.Scatter(
                                            x=_bh_ret.index, y=_bh_ret.values, mode="lines",
                                            name="Buy & Hold", line=dict(color="gray", width=1, dash="dot")
                                        ))
                                    fig_best_eq.update_layout(
                                        title=f"누적 수익률 ({_title_lbl})",
                                        yaxis_title="수익률 (%)", height=350,
                                        margin=dict(l=0, r=0, t=70, b=30),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                                    )
                                    fig_best_eq = _apply_return_hover_format(fig_best_eq, apply_all=True)
                                    st.plotly_chart(fig_best_eq, use_container_width=True)

                                    # DD 차트
                                    _best_dd = _best_df.loc[_beq_idx, "drawdown"] if "drawdown" in _best_df.columns else None
                                    if _best_dd is not None:
                                        fig_best_dd = go.Figure()
                                        fig_best_dd.add_trace(go.Scatter(
                                            x=_beq_idx, y=_best_dd.values, mode="lines",
                                            name="전략 DD", line=dict(color="crimson", width=2),
                                            fill="tozeroy", fillcolor="rgba(220,20,60,0.15)"
                                        ))
                                        if not _bh_chart.empty:
                                            _bh_peak = _bh_chart["close"].cummax()
                                            _bh_dd = (_bh_chart["close"] - _bh_peak) / _bh_peak * 100
                                            fig_best_dd.add_trace(go.Scatter(
                                                x=_bh_dd.index, y=_bh_dd.values, mode="lines",
                                                name="Buy & Hold DD", line=dict(color="gray", width=1, dash="dot")
                                            ))
                                        fig_best_dd.update_layout(
                                            title="Drawdown",
                                            yaxis_title="DD (%)", height=280,
                                            margin=dict(l=0, r=0, t=70, b=30),
                                            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                                        )
                                        fig_best_dd = _apply_dd_hover_format(fig_best_dd)
                                        st.plotly_chart(fig_best_dd, use_container_width=True)
                        except Exception as e:
                            st.warning(f"최적 파라미터 백테스트 오류: {e}")

    # ══════════════════════════════════════════════════════
    # Tab 4: 수수료/세금 (기존 내용 유지)
    # ══════════════════════════════════════════════════════
    with tab_g4:
        st.header("KRX 금현물 수수료 및 세금 안내")
        st.caption("키움증권 기준 | 수수료는 변경될 수 있으니 공식 홈페이지에서 최신 정보를 확인하세요.")

        st.subheader("1. 거래 수수료")
        fee_data = pd.DataFrame([
            {"구분": "매매 수수료", "요율/금액": "약 0.165% (온라인)", "비고": "매수/매도 각각 부과, 부가세 포함"},
            {"구분": "유관기관 수수료", "요율/금액": "0.0046396%", "비고": "거래소/예탁원 등 (매매금액 기준)"},
            {"구분": "합계 (편도)", "요율/금액": "약 0.17%", "비고": "매수 또는 매도 1회당"},
            {"구분": "합계 (왕복)", "요율/금액": "약 0.34%", "비고": "매수 + 매도 1세트"},
        ])
        st.dataframe(fee_data, use_container_width=True, hide_index=True)

        st.subheader("2. 세금")
        tax_data = pd.DataFrame([
            {"구분": "양도소득세", "세율": "비과세", "비고": "KRX 금시장 매매차익은 양도세 면제"},
            {"구분": "배당소득세", "세율": "15.4%", "비고": "보관료 환급금(이자) 발생 시"},
            {"구분": "부가가치세 (매매)", "세율": "면세", "비고": "KRX 장내 거래 시 부가세 없음"},
            {"구분": "부가가치세 (실물 인출)", "세율": "10%", "비고": "평균 매수단가 x 인출 수량 기준"},
        ])
        st.dataframe(tax_data, use_container_width=True, hide_index=True)
        st.info("KRX 금시장의 최대 장점: **매매차익 비과세 + 부가세 면세** (실물 인출 시에만 부가세 10% 부과)")

        st.subheader("3. 보관료")
        st.markdown(
            "| 항목 | 내용 |\n"
            "|------|------|\n"
            "| **보관료율** | 매일 잔량의 시가 환산 금액 x **0.02% (연율)** |\n"
            "| **일할 계산** | 시가 x 보유수량 x 0.0002% / 365일 |\n"
            "| **부가세** | 별도 (보관료의 10%) |\n"
            "| **부과 주기** | 매월 말 정산 |\n"
            "\n"
            "> 예시: 금 100g 보유, 시가 13만원/g → 연간 보관료 약 **2,600원** (부가세 별도)\n"
        )

        st.subheader("4. 금 투자 방법별 비교")
        compare_data = pd.DataFrame([
            {"투자 방법": "KRX 금현물", "매매차익 세금": "비과세", "부가세": "면세 (인출시 10%)", "거래 수수료": "~0.17%", "실물 인출": "가능 (100g/1kg)"},
            {"투자 방법": "금 ETF", "매매차익 세금": "15.4%", "부가세": "해당없음", "거래 수수료": "~0.015%+보수", "실물 인출": "불가"},
            {"투자 방법": "골드뱅킹 (은행)", "매매차익 세금": "15.4%", "부가세": "매입시 면세, 인출시 10%", "거래 수수료": "~1%", "실물 인출": "가능"},
            {"투자 방법": "금 실물 (귀금속점)", "매매차익 세금": "비과세", "부가세": "10% (매입 시)", "거래 수수료": "5~15% (스프레드)", "실물 인출": "즉시"},
        ])
        st.dataframe(compare_data, use_container_width=True, hide_index=True)
        st.caption("출처: 키움증권 금현물 수수료 안내, KRX 금시장 안내서 | 수수료는 변경될 수 있습니다.")

    with tab_g5:
        render_strategy_trigger_tab("GOLD")





