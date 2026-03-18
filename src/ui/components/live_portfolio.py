"""실시간 포트폴리오 탭 — 트레이딩 모드, 자산 현황, 시그널 모니터링, 합산 성과."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.strategy.sma import SMAStrategy
from src.strategy.donchian import DonchianStrategy
import src.engine.data_cache as data_cache
from src.ui.components.performance import (
    _render_performance_analysis,
    _apply_return_hover_format,
    _apply_dd_hover_format,
)
from src.ui.coin_utils import (
    load_balance_cache, load_signal_state,
    normalize_coin_interval, make_signal_key,
    determine_signal, resolve_effective_state,
    state_to_position_label, ttl_cache, clear_cache,
    get_signal_entry,
)


def render_live_portfolio_tab(
    trader, worker, portfolio_list, initial_cap, config, save_config,
    start_date, backtest_engine, INTERVAL_REV_MAP,
):
    # ── 트레이딩 모드 선택 ──
    _mode_col1, _mode_col2 = st.columns([3, 1])
    with _mode_col1:
        st.header("실시간 포트폴리오 대시보드")
    with _mode_col2:
        _cur_mode = config.get("trading_mode", "real")
        _mode_options = ["Signal Mode", "Real Trading"]
        _mode_idx = 0 if _cur_mode == "signal" else 1
        _selected_mode = st.selectbox(
            "트레이딩 모드", _mode_options, index=_mode_idx,
            key="coin_trading_mode",
        )
        _new_mode = "signal" if _selected_mode == "Signal Mode" else "real"
        if _new_mode != _cur_mode:
            config["trading_mode"] = _new_mode
            save_config(config)
            st.rerun()
    if _cur_mode == "signal":
        st.info("📡 **Signal Mode** — 시그널 분석만 수행하고 실제 주문은 실행하지 않습니다.")
    else:
        st.success("🔴 **Real Trading Mode** — 시그널에 따라 실제 주문이 실행됩니다.")

    if not trader:
        st.warning("사이드바에서 API 키를 설정해주세요.")
        return

    # Configure and Start Worker
    worker.update_config(portfolio_list)
    worker.start_worker()

    # Control Bar
    if st.button("🔄 새로고침"):
        clear_cache("krw_bal_t1", "prices_t1", "balances_t1")
        st.session_state["coin_force_price_refresh_once"] = True
        st.rerun()

    if not portfolio_list:
        st.warning("사이드바에서 포트폴리오에 코인을 추가해주세요.")
        return

    # 자산관리 투입원금 로드
    from src.ui.components.asset_mgmt import load_asset_mgmt, calc_invested_capital
    _am_data = load_asset_mgmt()
    _am_coin_strats = _am_data.get("coin", {}).get("strategies", {})

    count = len(portfolio_list)

    # ── 일괄 API 호출 (TTL 캐시) ──
    unique_coins = list(dict.fromkeys(item['coin'].upper() for item in portfolio_list))
    unique_tickers = list(dict.fromkeys(f"{item['market']}-{item['coin'].upper()}" for item in portfolio_list))

    def _fetch_all_prices():
        _force_refresh = bool(st.session_state.pop("coin_force_price_refresh_once", False))
        _ttl_sec = 0.0 if _force_refresh else 5.0
        return data_cache.get_current_prices_local_first(
            unique_tickers, ttl_sec=_ttl_sec, allow_api_fallback=True,
        )

    all_prices = ttl_cache("prices_t1", _fetch_all_prices, ttl=5)

    def _fetch_all_balances():
        if hasattr(trader, 'get_all_balances'):
            raw = trader.get_all_balances()
            if raw and isinstance(raw, dict) and len(raw) > 0:
                return raw
        return None

    _live_bal = ttl_cache("balances_t1", _fetch_all_balances, ttl=10)

    if _live_bal and isinstance(_live_bal, dict) and len(_live_bal) > 0:
        krw_bal = float(_live_bal.get('KRW', 0) or 0)
        all_balances = {c: float(_live_bal.get(c, 0) or 0) for c in unique_coins}
    else:
        _cached = load_balance_cache()
        if _cached and _cached.get("balances"):
            _bal = _cached["balances"]
            krw_bal = float(_bal.get('KRW', 0) or 0)
            all_balances = {c: float(_bal.get(c, 0) or 0) for c in unique_coins}
        else:
            krw_bal = 0
            all_balances = {c: 0 for c in unique_coins}

    # --- Total Summary Container ---
    st.subheader("🏁 포트폴리오 요약")

    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

    # 자산관리 투입원금 합산 (없으면 사이드바 초기자본 fallback)
    _am_total = sum(
        calc_invested_capital(_am_coin_strats.get(make_signal_key(it), {}))
        for it in portfolio_list
    )
    if _am_total > 0:
        total_init_val = _am_total
        reserved_cash = 0
    else:
        total_init_val = initial_cap
        total_weight_alloc = sum([item.get('weight', 0) for item in portfolio_list])
        cash_ratio = max(0, 100 - total_weight_alloc) / 100.0
        reserved_cash = initial_cap * cash_ratio
    total_theo_val = reserved_cash

    # --- 전체 자산 현황 테이블 ---
    asset_summary_rows = [{"자산": "KRW (현금)", "보유량": f"{krw_bal:,.0f}", "현재가": "-", "평가금액(KRW)": f"{krw_bal:,.0f}", "상태": "-"}]
    seen_coins_summary = set()
    for s_item in portfolio_list:
        s_coin = s_item['coin'].upper()
        if s_coin in seen_coins_summary:
            continue
        seen_coins_summary.add(s_coin)
        s_ticker = f"{s_item['market']}-{s_coin}"
        s_bal = all_balances.get(s_coin, 0)
        s_price = all_prices.get(s_ticker, 0) or 0
        s_val = s_bal * s_price
        is_holding = s_val >= 5000
        asset_summary_rows.append({
            "자산": s_coin,
            "보유량": (f"{s_bal:.8f}" if s_bal < 1 else f"{s_bal:,.4f}") if s_bal > 0 else "0",
            "현재가": f"{s_price:,.0f}",
            "평가금액(KRW)": f"{s_val:,.0f}",
            "상태": "보유중" if is_holding else "미보유",
        })
    # 포트폴리오에 없지만 계좌에 보유 중인 코인 추가
    if _live_bal and isinstance(_live_bal, dict):
        for _extra_c, _extra_v in _live_bal.items():
            _extra_c = str(_extra_c).upper()
            if _extra_c == 'KRW' or _extra_c in seen_coins_summary:
                continue
            _extra_bal = float(_extra_v or 0)
            if _extra_bal <= 0:
                continue
            _extra_ticker = f"KRW-{_extra_c}"
            _extra_price = all_prices.get(_extra_ticker, 0) or 0
            if _extra_price == 0:
                try:
                    _extra_prices = data_cache.get_current_prices_local_first(
                        [_extra_ticker], ttl_sec=5.0, allow_api_fallback=True,
                    )
                    _extra_price = (_extra_prices or {}).get(_extra_ticker, 0) or 0
                except Exception:
                    pass
            _extra_val = _extra_bal * _extra_price
            if _extra_val < 100:
                continue
            seen_coins_summary.add(_extra_c)
            asset_summary_rows.append({
                "자산": f"{_extra_c} *",
                "보유량": (f"{_extra_bal:.8f}" if _extra_bal < 1 else f"{_extra_bal:,.4f}"),
                "현재가": f"{_extra_price:,.0f}" if _extra_price > 0 else "-",
                "평가금액(KRW)": f"{_extra_val:,.0f}",
                "상태": "미등록 보유",
            })
    total_real_summary = krw_bal + sum(
        float(all_balances.get(c, 0) or 0) * (all_prices.get(f"KRW-{c}", 0) or 0)
        for c in seen_coins_summary
    )
    if _live_bal and isinstance(_live_bal, dict):
        for _sc in seen_coins_summary:
            if _sc not in all_balances:
                _sb = float(_live_bal.get(_sc, 0) or 0)
                _sp = all_prices.get(f"KRW-{_sc}", 0) or 0
                total_real_summary += _sb * _sp
    asset_summary_rows.append({
        "자산": "합계", "보유량": "", "현재가": "",
        "평가금액(KRW)": f"{total_real_summary:,.0f}", "상태": "",
    })
    with st.expander(f"💰 전체 자산 현황 (Total: {total_real_summary:,.0f} KRW)", expanded=True):
        st.dataframe(pd.DataFrame(asset_summary_rows), use_container_width=True, hide_index=True)
        st.caption("* 표시 = 포트폴리오 미등록 코인 (계좌에 보유 중)")

    # --- 단기 모니터링 차트 (60봉) ---
    with st.expander("📊 단기 시그널 모니터링 (60봉)", expanded=True):
        signal_rows = []
        signal_state = load_signal_state()
        if not signal_state:
            st.caption("signal_state.json이 없어 이전 상태는 '-' 및 포지션 '미확인'으로 표시됩니다.")

        interval_order = {'day': 0, 'minute240': 1, 'minute60': 2, 'minute30': 3, 'minute15': 4, 'minute10': 5}
        btc_items = sorted(
            [x for x in portfolio_list if x.get('coin', '').upper() == 'BTC'],
            key=lambda x: interval_order.get(x.get('interval', 'day'), 99)
        )
        other_items = sorted(
            [x for x in portfolio_list if x.get('coin', '').upper() != 'BTC'],
            key=lambda x: interval_order.get(x.get('interval', 'day'), 99)
        )

        def _to_float(val, default=0.0):
            try:
                num = float(val)
                return default if pd.isna(num) else num
            except Exception:
                return default

        def render_chart_row(items):
            if not items:
                return
            cols = st.columns(len(items))
            for ci, item in enumerate(items):
                p_ticker = f"{item['market']}-{item['coin'].upper()}"
                p_strategy = item.get('strategy', 'SMA')
                try:
                    p_param = int(float(item.get('parameter', item.get('sma', 20)) or 20))
                except Exception:
                    p_param = 20
                try:
                    _sp = item.get('sell_parameter', 0)
                    p_sell_param = int(float(_sp)) if _sp not in (None, "", 0, "0") else max(5, p_param // 2)
                except Exception:
                    p_sell_param = max(5, p_param // 2)
                p_interval = normalize_coin_interval(item.get('interval', 'day'))
                iv_label = INTERVAL_REV_MAP.get(p_interval, p_interval)

                try:
                    df_60 = worker.get_data(p_ticker, p_interval)
                    if df_60 is None or len(df_60) < p_param + 5:
                        df_60 = ttl_cache(
                            f"ohlcv_{p_ticker}_{p_interval}",
                            lambda t=p_ticker, iv=p_interval, pp=p_param: data_cache.get_ohlcv_local_first(
                                t, interval=iv, count=max(60 + pp, 200), allow_api_fallback=True,
                            ),
                            ttl=30
                        )
                    if df_60 is None or len(df_60) < p_param + 5:
                        continue

                    latest_close = _to_float(df_60['close'].iloc[-1], default=0.0)
                    close_now = _to_float(all_prices.get(p_ticker, latest_close), default=latest_close)

                    # 실시간 가격으로 마지막(진행 중) 캔들 갱신
                    if close_now > 0 and close_now != latest_close:
                        df_60 = df_60.copy()
                        df_60.at[df_60.index[-1], 'close'] = close_now
                        cur_high = _to_float(df_60['high'].iloc[-1], default=close_now)
                        cur_low = _to_float(df_60['low'].iloc[-1], default=close_now)
                        if close_now > cur_high:
                            df_60.at[df_60.index[-1], 'high'] = close_now
                        if close_now < cur_low:
                            df_60.at[df_60.index[-1], 'low'] = close_now

                    strategy_name = str(p_strategy).strip().lower()
                    if strategy_name.startswith("donchian"):
                        strat = DonchianStrategy()
                        df_feat = strat.create_features(df_60, buy_period=p_param, sell_period=p_sell_param)
                        if df_feat is None or len(df_feat) < 2:
                            continue
                        eval_candle = df_feat.iloc[-2]
                        close_eval = _to_float(eval_candle.get("close"), default=latest_close)
                        upper_key = f"Donchian_Upper_{p_param}"
                        lower_key = f"Donchian_Lower_{p_sell_param}"
                        buy_target = _to_float(eval_candle.get(upper_key), default=0.0)
                        sell_target = _to_float(eval_candle.get(lower_key), default=0.0)
                        position_state = str(
                            strat.get_signal(eval_candle, buy_period=p_param, sell_period=p_sell_param)
                        ).upper()
                        upper_vals = df_feat[upper_key]
                        lower_vals = df_feat[lower_key]
                    else:
                        strat = SMAStrategy()
                        df_feat = strat.create_features(df_60, periods=[p_param])
                        if df_feat is None or len(df_feat) < 2:
                            continue
                        eval_candle = df_feat.iloc[-2]
                        close_eval = _to_float(eval_candle.get("close"), default=latest_close)
                        sma_key = f"SMA_{p_param}"
                        sma_val = _to_float(eval_candle.get(sma_key), default=0.0)
                        buy_target = sma_val
                        sell_target = sma_val
                        position_state = str(
                            strat.get_signal(eval_candle, strategy_type='SMA_CROSS', ma_period=p_param)
                        ).upper()
                        sma_vals = df_feat[sma_key]

                    # VM signal_state에서 목표가/상태 읽기 (통합 데이터 소스)
                    signal_key = make_signal_key({
                        "market": item.get("market", "KRW"),
                        "coin": item.get("coin", ""),
                        "strategy": p_strategy,
                        "parameter": p_param,
                        "interval": p_interval,
                    })
                    sig_entry = get_signal_entry(signal_state, signal_key)
                    # VM 데이터가 있으면 VM 기준 사용, 없으면 로컬 계산 fallback
                    if sig_entry.get("buy_target"):
                        buy_target = _to_float(sig_entry.get("buy_target"), default=buy_target)
                        sell_target = _to_float(sig_entry.get("sell_target"), default=sell_target)

                    buy_dist = ((close_now - buy_target) / buy_target * 100) if buy_target > 0 else 0
                    sell_dist = ((close_now - sell_target) / sell_target * 100) if sell_target > 0 else 0

                    prev_state = sig_entry.get("state", "").upper() or None
                    if prev_state not in {"BUY", "SELL", "HOLD"}:
                        prev_state = None
                    exec_signal = determine_signal(position_state, prev_state)
                    effective_state = resolve_effective_state(position_state, prev_state, exec_signal)
                    position_label = state_to_position_label(effective_state)

                    coin_sym = item['coin'].upper()
                    coin_bal = _to_float(all_balances.get(coin_sym, 0), default=0.0)
                    coin_val = coin_bal * close_now
                    is_holding = coin_val >= 5000

                    # Backtest vs Real 비교
                    bt_signal = position_state  # 백테스트: 현재 가격 기반 전략 시그널
                    real_signal = prev_state if prev_state else "-"  # 실제: signal_state.json (VM 실행 결과)
                    # 일치 여부 판단
                    if real_signal == "-":
                        match_label = "미확인"
                    elif bt_signal == "HOLD" or real_signal == "HOLD":
                        match_label = "중립"
                    elif bt_signal == real_signal:
                        match_label = "일치"
                    else:
                        match_label = f"불일치"

                    signal_rows.append({
                        "종목": p_ticker.replace("KRW-", ""),
                        "전략": f"{p_strategy} {p_param}",
                        "시간봉": iv_label,
                        "Backtest": bt_signal,
                        "Real": real_signal,
                        "일치": match_label,
                        "실보유": "보유" if is_holding else "미보유",
                        "실행": exec_signal,
                        "현재가": f"{close_now:,.0f}",
                        "매수목표": f"{buy_target:,.0f}",
                        "매도목표": f"{sell_target:,.0f}",
                        "매수이격도": f"{buy_dist:+.2f}%",
                        "매도이격도": f"{sell_dist:+.2f}%",
                    })

                    df_chart = df_feat.iloc[-60:]
                    with cols[ci]:
                        fig_m = go.Figure()
                        fig_m.add_trace(go.Candlestick(
                            x=df_chart.index, open=df_chart['open'],
                            high=df_chart['high'], low=df_chart['low'],
                            close=df_chart['close'], name='가격',
                            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                        ))

                        if strategy_name.startswith("donchian"):
                            upper_chart = upper_vals.loc[df_chart.index]
                            lower_chart = lower_vals.loc[df_chart.index]
                            fig_m.add_trace(go.Scatter(
                                x=df_chart.index, y=upper_chart,
                                name=f'상단({p_param})', line=dict(color='green', width=1, dash='dot')
                            ))
                            fig_m.add_trace(go.Scatter(
                                x=df_chart.index, y=lower_chart,
                                name=f'하단({p_sell_param})', line=dict(color='red', width=1, dash='dot')
                            ))
                        else:
                            sma_chart = sma_vals.loc[df_chart.index]
                            fig_m.add_trace(go.Scatter(
                                x=df_chart.index, y=sma_chart,
                                name=f'SMA({p_param})', line=dict(color='orange', width=2)
                            ))

                        if exec_signal == "BUY":
                            sig_color = "green"
                        elif exec_signal == "SELL":
                            sig_color = "red"
                        elif effective_state == "BUY":
                            sig_color = "green"
                        elif effective_state == "SELL":
                            sig_color = "red"
                        else:
                            sig_color = "gray"
                        if strategy_name.startswith("donchian"):
                            if effective_state in ("BUY", "HOLD"):
                                _chart_dist = f"매도이격:{sell_dist:+.1f}%"
                            else:
                                _chart_dist = f"매수이격:{buy_dist:+.1f}%"
                        else:
                            _chart_dist = f"{buy_dist:+.1f}%"
                        fig_m.update_layout(
                            title=f"{p_ticker.replace('KRW-','')} {p_strategy}{p_param} ({iv_label}) [{position_label}] [실행:{exec_signal}] [{_chart_dist}]",
                            title_font_color=sig_color,
                            height=300, margin=dict(l=0, r=0, t=35, b=30),
                            xaxis_rangeslider_visible=False,
                            showlegend=False,
                            xaxis=dict(showticklabels=True, tickformat='%m/%d %H:%M', tickangle=-45, nticks=6),
                        )
                        st.plotly_chart(fig_m, use_container_width=True)

                except Exception as chart_err:
                    with cols[ci]:
                        st.warning(f"{p_ticker} 데이터 로드 실패: {chart_err}")
                    continue

        render_chart_row(btc_items)
        render_chart_row(other_items)

        if signal_rows:
            df_sig = pd.DataFrame(signal_rows)
            st.dataframe(df_sig, use_container_width=True, hide_index=True)

    # 합산 포트폴리오 자리
    combined_portfolio_container = st.container()

    st.write(f"### 📋 자산 상세 (현금 예비: {reserved_cash:,.0f} KRW)")

    # 같은 코인을 공유하는 전략의 시그널 상태별 비중 합산 (잔고 분할용)
    # BUY/HOLD 전략만 코인을 "소유", SELL 전략은 현금 보유로 간주
    _sig_state_for_split = load_signal_state() or {}
    _coin_buy_weight: dict[str, float] = {}   # BUY/HOLD 전략 비중 합
    _coin_total_weight: dict[str, float] = {}  # 전체 전략 비중 합
    for _it in portfolio_list:
        _csym = _it['coin'].upper()
        _w = float(_it.get('weight', 0) or 0)
        _coin_total_weight[_csym] = _coin_total_weight.get(_csym, 0) + _w
        _skey = make_signal_key(_it)
        _sentry = get_signal_entry(_sig_state_for_split, _skey)
        _sstate = _sentry.get("state", "").upper()
        # SELL이 아닌 상태(BUY/HOLD/미확인) → 코인 보유 전략
        if _sstate != "SELL":
            _coin_buy_weight[_csym] = _coin_buy_weight.get(_csym, 0) + _w

    # 포트폴리오 합산용 에쿼티 수집
    portfolio_equity_data = []

    def _cached_backtest(ticker, period, interval, count, start_date_str, initial_balance, strategy_mode, sell_period_ratio, _df_hash):
        df_bt_local = data_cache.load_cached(ticker, interval)
        if df_bt_local is None or len(df_bt_local) < period:
            return None
        return backtest_engine.run_backtest(
            ticker, period=period, interval=interval, count=count,
            start_date=start_date_str, initial_balance=initial_balance,
            df=df_bt_local, strategy_mode=strategy_mode,
            sell_period_ratio=sell_period_ratio
        )

    for asset_idx, item in enumerate(portfolio_list):
        ticker = f"{item['market']}-{item['coin'].upper()}"

        strategy_mode = item.get('strategy', 'SMA Strategy')
        try:
            param_val = int(float(item.get('parameter', item.get('sma', 20)) or 20))
        except Exception:
            param_val = 20
        if param_val <= 0:
            param_val = 20
        try:
            sell_param_val = int(float(item.get('sell_parameter', 0) or 0))
        except Exception:
            sell_param_val = 0
        if sell_param_val < 0:
            sell_param_val = 0
        try:
            weight = float(item.get('weight', 0) or 0)
        except Exception:
            weight = 0.0
        interval = item.get('interval', 'day')

        _am_key = make_signal_key(item)
        _am_invested = calc_invested_capital(_am_coin_strats.get(_am_key, {}))
        per_coin_cap = _am_invested if _am_invested > 0 else initial_cap * (weight / 100.0)

        with st.expander(f"**{ticker}** ({strategy_mode} {param_val}, {weight}%, {interval})", expanded=False):
            try:
                df_curr = worker.get_data(ticker, interval)

                if df_curr is None or len(df_curr) < param_val:
                    st.warning(f"데이터 대기 중... ({ticker}, {interval})")
                    total_theo_val += per_coin_cap
                    continue

                if strategy_mode == "Donchian":
                    strategy_eng = DonchianStrategy()
                    buy_p = param_val
                    sell_p = sell_param_val or max(5, buy_p // 2)
                    df_curr = strategy_eng.create_features(df_curr, buy_period=buy_p, sell_period=sell_p)
                    last_candle = df_curr.iloc[-2]
                    curr_upper = last_candle.get(f'Donchian_Upper_{buy_p}', 0)
                    curr_lower = last_candle.get(f'Donchian_Lower_{sell_p}', 0)
                    curr_sma = (curr_upper + curr_lower) / 2
                else:
                    strategy_eng = SMAStrategy()
                    calc_periods = [param_val]
                    df_curr = strategy_eng.create_features(df_curr, periods=calc_periods)
                    last_candle = df_curr.iloc[-2]
                    curr_sma = last_candle[f'SMA_{param_val}']

                _cp = pd.to_numeric(all_prices.get(ticker, 0), errors="coerce")
                curr_price = float(_cp) if pd.notna(_cp) else 0.0
                coin_sym = item['coin'].upper()
                _cb = pd.to_numeric(all_balances.get(coin_sym, 0), errors="coerce")
                coin_bal_total = float(_cb) if pd.notna(_cb) else 0.0
                # 같은 코인 다전략 → 시그널 상태 기반 잔고 분할
                # SELL 전략은 코인 0, BUY/HOLD 전략만 비중 비율로 분배
                _this_skey = make_signal_key(item)
                _this_sentry = get_signal_entry(_sig_state_for_split, _this_skey)
                _this_state = _this_sentry.get("state", "").upper()
                buy_w = _coin_buy_weight.get(coin_sym, 0)
                if _this_state == "SELL":
                    # SELL 전략 → 코인 미보유 (현금 보유)
                    share_ratio = 0.0
                    coin_bal = 0.0
                elif buy_w > 0:
                    # BUY/HOLD 전략 → BUY 전략들 비중 비율로 분배
                    share_ratio = weight / buy_w
                    coin_bal = coin_bal_total * share_ratio
                else:
                    # 모든 전략이 SELL → 잔여 코인은 균등 분배
                    total_w = _coin_total_weight.get(coin_sym, weight)
                    share_ratio = weight / total_w if total_w > 0 else 1.0
                    coin_bal = coin_bal_total * share_ratio
                coin_value_krw = coin_bal * curr_price

                # Theo Backtest
                sell_ratio = (sell_param_val or max(5, param_val // 2)) / param_val if param_val > 0 else 0.5
                df_bt = data_cache.load_cached(ticker, interval)
                if df_bt is not None and len(df_bt) >= param_val:
                    req_count = len(df_bt)
                    df_hash = f"{len(df_bt)}_{df_bt.index[-1]}"
                else:
                    df_bt = df_curr
                    req_count = len(df_bt)
                    df_hash = f"{len(df_bt)}_{df_bt.index[-1]}"
                bt_res = _cached_backtest(
                    ticker, param_val, interval, req_count,
                    str(start_date), per_coin_cap, strategy_mode,
                    sell_ratio, df_hash
                )
                if bt_res is None:
                    bt_res = backtest_engine.run_backtest(
                        ticker, period=param_val, interval=interval,
                        count=req_count, start_date=start_date,
                        initial_balance=per_coin_cap, df=df_bt,
                        strategy_mode=strategy_mode,
                        sell_period_ratio=sell_ratio
                    )

                expected_eq = 0
                theo_status = "UNKNOWN"

                if "error" not in bt_res:
                    perf = bt_res['performance']
                    theo_status = perf['final_status']
                    expected_eq = perf['final_equity']
                    total_theo_val += expected_eq
                    hist_df_tmp = bt_res['df']
                    label = f"{ticker} ({strategy_mode} {param_val}, {interval})"
                    portfolio_equity_data.append({
                        "ticker": ticker,
                        "label": label,
                        "equity": hist_df_tmp['equity'],
                        "close": hist_df_tmp['close'],
                        "cap": per_coin_cap,
                        "perf": perf,
                    })
                else:
                    total_theo_val += per_coin_cap

                real_status = "HOLD" if coin_value_krw >= 5000 else "CASH"

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("가격 / SMA", f"{curr_price:,.0f}", delta=f"{curr_price - curr_sma:,.0f}")

                if strategy_mode == "Donchian":
                    c2.metric("채널", f"{buy_p}/{sell_p}")
                else:
                    c2.metric("SMA 기간", f"{param_val}")

                roi_theo = (expected_eq - per_coin_cap) / per_coin_cap * 100
                c3.metric(f"이론 자산", f"{expected_eq:,.0f}", delta=f"{roi_theo:.2f}%")

                match = (real_status == theo_status)
                match_color = "green" if match else "red"
                c4.markdown(f"**동기화**: :{match_color}[{'일치' if match else '불일치'}]")
                _share_pct = f" ({share_ratio:.0%})" if share_ratio < 1.0 else ""
                c4.caption(f"실제: {coin_bal:,.4f} {coin_sym}{_share_pct} ({real_status})")

                st.divider()

                p_tab1, p_tab2 = st.tabs(["📈 분석 & 벤치마크", "📋 체결 내역"])

                with p_tab1:
                    if "error" not in bt_res:
                        hist_df = bt_res['df']

                        if 'yearly_stats' in perf:
                            st.caption("📅 연도별 성과")
                            ys = perf['yearly_stats'].copy()
                            ys.index.name = "연도"
                            st.dataframe(ys.style.format("{:.2f}"), use_container_width=True)

                        _render_performance_analysis(
                            equity_series=hist_df.get("equity"),
                            benchmark_series=hist_df.get("close"),
                            strategy_metrics=perf,
                            strategy_label=f"{strategy_mode} 전략",
                            benchmark_label=f"{ticker} 단순보유",
                        )

                with p_tab2:
                    st.markdown("**가상 체결 (백테스트)**")
                    if "error" not in bt_res and perf.get('trades'):
                        vt_rows = []
                        for t in perf['trades']:
                            t_date = t.get('date', '')
                            if hasattr(t_date, 'strftime'):
                                t_date = t_date.strftime('%Y-%m-%d')
                            t_type = t.get('type', '')
                            t_side = "매수" if t_type == 'buy' else ("매도" if t_type == 'sell' else t_type)
                            t_price = t.get('price', 0)
                            t_amount = t.get('amount', 0)
                            t_equity = t.get('equity', 0)
                            vt_rows.append({
                                "일시": t_date,
                                "구분": f"🔴 {t_side}" if t_type == 'buy' else f"🔵 {t_side}",
                                "체결가": f"{t_price:,.0f}",
                                "수량": f"{t_amount:.6f}" if t_amount < 1 else f"{t_amount:,.4f}",
                                "자산(KRW)": f"{t_equity:,.0f}",
                            })
                        if vt_rows:
                            st.dataframe(pd.DataFrame(vt_rows[-20:]), use_container_width=True, hide_index=True)
                            st.caption(f"최근 {min(20, len(vt_rows))}건 / 총 {len(vt_rows)}건")
                        else:
                            st.info("백테스트 체결 기록 없음")
                    else:
                        st.info("백테스트 데이터 없음")

                    st.markdown("**실제 체결 (거래소)**")
                    try:
                        done = ttl_cache(
                            f"done_{ticker}",
                            lambda t=ticker: trader.get_done_orders(t),
                            ttl=30
                        )
                        if done:
                            rt_rows = []
                            for r in done[:20]:
                                side = r.get('side', '')
                                side_kr = "매수" if side == 'bid' else ("매도" if side == 'ask' else side)
                                price_r = float(r.get('price', 0) or 0)
                                exec_vol = float(r.get('executed_volume', 0) or 0)
                                if price_r > 0 and exec_vol > 0:
                                    total_k = price_r * exec_vol
                                elif 'trades' in r and r['trades']:
                                    total_k = sum(float(tr.get('funds', 0)) for tr in r['trades'])
                                else:
                                    total_k = price_r
                                created = r.get('created_at', '')
                                if pd.notna(created):
                                    try: created = pd.to_datetime(created).strftime('%Y-%m-%d %H:%M')
                                    except: pass
                                rt_rows.append({
                                    "일시": created,
                                    "구분": f"🔴 {side_kr}" if side == 'bid' else f"🔵 {side_kr}",
                                    "체결가": f"{price_r:,.0f}" if price_r > 0 else "-",
                                    "수량": f"{exec_vol:.6f}" if exec_vol < 1 else f"{exec_vol:,.4f}",
                                    "금액(KRW)": f"{total_k:,.0f}",
                                })
                            if rt_rows:
                                st.dataframe(pd.DataFrame(rt_rows), use_container_width=True, hide_index=True)
                            else:
                                st.info("체결 완료 주문 없음")
                        else:
                            st.info("체결 완료 주문 없음")
                    except Exception:
                        st.info("체결 내역 조회 불가 (API 권한 확인)")

            except Exception as e:
                st.error(f"{ticker} 처리 오류: {e}")

    # --- Populate Total Summary ---
    total_roi = (total_theo_val - total_init_val) / total_init_val * 100 if total_init_val else 0
    real_roi = (total_real_summary - total_init_val) / total_init_val * 100 if total_init_val else 0
    diff_val = total_real_summary - total_theo_val

    sum_col1.metric("초기 자본", f"{total_init_val:,.0f} KRW")
    sum_col2.metric("이론 총자산", f"{total_theo_val:,.0f} KRW", delta=f"{total_roi:.2f}%")
    sum_col3.metric("실제 총자산", f"{total_real_summary:,.0f} KRW", delta=f"{real_roi:.2f}%")
    sum_col4.metric("차이 (실제-이론)", f"{diff_val:,.0f} KRW", delta_color="off" if abs(diff_val)<1000 else "inverse")

    # --- 합산 포트폴리오 성과 ---
    if portfolio_equity_data:
        with combined_portfolio_container:
            with st.expander("📊 합산 포트폴리오 성과", expanded=True):
                equity_dfs = []
                bench_dfs = []
                for ed in portfolio_equity_data:
                    eq = ed['equity'].copy()
                    cl = ed['close'].copy()
                    cap = ed['cap']

                    if hasattr(eq.index, 'tz') and eq.index.tz is not None:
                        eq.index = eq.index.tz_localize(None)
                        cl.index = cl.index.tz_localize(None)
                    eq_daily = eq.resample('D').last().dropna()
                    cl_daily = cl.resample('D').last().dropna()

                    bench_daily = (cl_daily / cl_daily.iloc[0]) * cap

                    eq_daily.name = ed['label']
                    bench_daily.name = ed['label']
                    equity_dfs.append(eq_daily)
                    bench_dfs.append(bench_daily)

                combined_eq = pd.concat(equity_dfs, axis=1).sort_index()
                combined_bench = pd.concat(bench_dfs, axis=1).sort_index()

                combined_eq = combined_eq.ffill().bfill()
                combined_bench = combined_bench.ffill().bfill()

                combined_eq['cash_reserve'] = reserved_cash
                combined_bench['cash_reserve'] = reserved_cash

                total_eq = combined_eq.sum(axis=1)
                total_bench = combined_bench.sum(axis=1)

                norm_eq = total_eq / total_eq.iloc[0] * 100
                norm_bench = total_bench / total_bench.iloc[0] * 100

                port_final = total_eq.iloc[-1]
                port_init = total_eq.iloc[0]
                port_return = (port_final - port_init) / port_init * 100

                port_days = (total_eq.index[-1] - total_eq.index[0]).days
                port_cagr = 0
                if port_days > 0 and port_final > 0:
                    port_cagr = ((port_final / port_init) ** (365 / port_days) - 1) * 100

                port_peak = total_eq.cummax()
                port_dd = (total_eq - port_peak) / port_peak * 100
                port_mdd = port_dd.min()

                port_returns = total_eq.pct_change().dropna()
                port_sharpe = 0
                if port_returns.std() > 0:
                    port_sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(365)

                bench_final = total_bench.iloc[-1]
                bench_init = total_bench.iloc[0]
                bench_return = (bench_final - bench_init) / bench_init * 100

                pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                pm1.metric("총 수익률", f"{port_return:.2f}%")
                pm2.metric("CAGR", f"{port_cagr:.2f}%")
                pm3.metric("MDD", f"{port_mdd:.2f}%")
                pm4.metric("Sharpe", f"{port_sharpe:.2f}")
                pm5.metric("vs 단순보유", f"{port_return - bench_return:+.2f}%p")

                st.caption(f"기간: {total_eq.index[0].strftime('%Y-%m-%d')} ~ {total_eq.index[-1].strftime('%Y-%m-%d')} ({port_days}일) | 초기자금: {port_init:,.0f} → 최종: {port_final:,.0f} KRW")

                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(
                    x=norm_eq.index, y=norm_eq.values,
                    name='포트폴리오 (전략)', line=dict(color='blue', width=2)
                ))
                fig_port.add_trace(go.Scatter(
                    x=norm_bench.index, y=norm_bench.values,
                    name='포트폴리오 (단순보유)', line=dict(color='gray', dash='dot')
                ))

                # 실제 체결 마커
                _mk_start = norm_eq.index.min()
                _mk_end = norm_eq.index.max() + pd.Timedelta(days=1)
                all_buy_dates = []
                all_sell_dates = []
                for ed in portfolio_equity_data:
                    _mk_ticker = str(ed.get("ticker", "") or "")
                    if not _mk_ticker:
                        continue
                    try:
                        _done_orders = ttl_cache(
                            f"done_combined_{_mk_ticker}",
                            lambda t=_mk_ticker: trader.get_done_orders(t),
                            ttl=20,
                        )
                    except Exception:
                        _done_orders = []
                    if not isinstance(_done_orders, list):
                        continue

                    for _od in _done_orders:
                        _side = str(_od.get("side", "") or "").lower()
                        _raw_ts = (
                            _od.get("created_at")
                            or _od.get("done_at")
                            or _od.get("created")
                            or _od.get("timestamp")
                        )
                        if not _raw_ts:
                            continue
                        _d_ts = pd.to_datetime(_raw_ts, errors="coerce", utc=True)
                        if pd.isna(_d_ts):
                            _d_ts = pd.to_datetime(_raw_ts, errors="coerce")
                        if pd.isna(_d_ts):
                            continue
                        try:
                            if getattr(_d_ts, "tzinfo", None) is not None:
                                _d_ts = _d_ts.tz_convert("Asia/Seoul").tz_localize(None)
                        except Exception:
                            try:
                                _d_ts = _d_ts.tz_localize(None)
                            except Exception:
                                pass

                        if _d_ts < _mk_start or _d_ts > _mk_end:
                            continue
                        if _side == "bid":
                            all_buy_dates.append(_d_ts)
                        elif _side == "ask":
                            all_sell_dates.append(_d_ts)

                def _add_real_trade_markers(_date_list, _name, _symbol, _color):
                    if not _date_list:
                        return 0
                    _vals = []
                    _valid_dates = []
                    _seen_days = set()
                    for _d in _date_list:
                        _d_ts = pd.Timestamp(_d)
                        if getattr(_d_ts, "tzinfo", None) is not None:
                            _d_ts = _d_ts.tz_localize(None)
                        _idx = norm_eq.index.get_indexer([_d_ts], method='nearest')
                        if _idx[0] < 0:
                            continue
                        _x = norm_eq.index[_idx[0]]
                        _day_key = _x.strftime("%Y-%m-%d") + _name
                        if _day_key in _seen_days:
                            continue
                        _seen_days.add(_day_key)
                        _valid_dates.append(_x)
                        _vals.append(norm_eq.iloc[_idx[0]])
                    if _valid_dates:
                        fig_port.add_trace(go.Scatter(
                            x=_valid_dates, y=_vals, mode='markers', name=_name,
                            marker=dict(symbol=_symbol, size=8, color=_color, opacity=0.8)
                        ))
                    return len(_valid_dates)

                _buy_mk_cnt = _add_real_trade_markers(all_buy_dates, "매수", "triangle-up", "green")
                _sell_mk_cnt = _add_real_trade_markers(all_sell_dates, "매도", "triangle-down", "red")
                if _buy_mk_cnt or _sell_mk_cnt:
                    st.caption(f"마커 기준: 실제 체결(거래소) | 매수 {_buy_mk_cnt}건 / 매도 {_sell_mk_cnt}건")
                else:
                    st.caption("마커 기준: 실제 체결(거래소) | 해당 기간 체결 없음")

                fig_port.update_layout(
                    height=350,
                    title="합산 포트폴리오: 전략 vs 단순보유 (정규화, 마커=실제 체결)",
                    yaxis_title="정규화 (%)",
                    yaxis_type="log",
                    margin=dict(l=0, r=0, t=80, b=0),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                )
                fig_port = _apply_return_hover_format(fig_port, apply_all=True)
                st.plotly_chart(fig_port, use_container_width=True)

                # 포트폴리오 MDD 차트
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=port_dd.index, y=port_dd.values,
                    name='낙폭', fill='tozeroy',
                    line=dict(color='red', width=1)
                ))
                fig_dd.update_layout(
                    height=200,
                    title="포트폴리오 낙폭 (%)",
                    yaxis_title="낙폭 (%)",
                    margin=dict(l=0, r=0, t=80, b=0),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                )
                fig_dd = _apply_dd_hover_format(fig_dd)
                st.plotly_chart(fig_dd, use_container_width=True)

                # 개별 자산 성과 테이블
                asset_perf_rows = []
                for ed in portfolio_equity_data:
                    p = ed['perf']
                    asset_perf_rows.append({
                        "자산": ed['label'],
                        "배분자본": f"{ed['cap']:,.0f}",
                        "최종자산": f"{p['final_equity']:,.0f}",
                        "수익률(%)": f"{p['total_return']:.2f}",
                        "CAGR(%)": f"{p['cagr']:.2f}",
                        "MDD(%)": f"{p['mdd']:.2f}",
                        "승률(%)": f"{p['win_rate']:.1f}",
                        "거래수": p['trade_count'],
                        "Sharpe": f"{p['sharpe']:.2f}",
                        "상태": p['final_status'],
                    })
                st.dataframe(pd.DataFrame(asset_perf_rows), use_container_width=True, hide_index=True)

                # 연도별 성과
                st.caption("📅 합산 포트폴리오 연도별 성과")
                port_daily_ret = total_eq.pct_change().fillna(0)
                port_year = total_eq.index.year
                port_dd_series = port_dd

                yearly_rows = []
                for yr in sorted(port_year.unique()):
                    yr_mask = port_year == yr
                    yr_ret = (1 + port_daily_ret[yr_mask]).prod() - 1
                    yr_mdd = port_dd_series[yr_mask].min()
                    yr_eq_start = total_eq[yr_mask].iloc[0]
                    yr_eq_end = total_eq[yr_mask].iloc[-1]

                    yr_bench_start = total_bench[yr_mask].iloc[0]
                    yr_bench_end = total_bench[yr_mask].iloc[-1]
                    yr_bench_ret = (yr_bench_end - yr_bench_start) / yr_bench_start * 100

                    yearly_rows.append({
                        "연도": yr,
                        "수익률(%)": f"{yr_ret * 100:.2f}",
                        "MDD(%)": f"{yr_mdd:.2f}",
                        "시작자산": f"{yr_eq_start:,.0f}",
                        "최종자산": f"{yr_eq_end:,.0f}",
                        "Buy&Hold(%)": f"{yr_bench_ret:.2f}",
                        "초과수익(%p)": f"{yr_ret * 100 - yr_bench_ret:.2f}",
                    })
                st.dataframe(pd.DataFrame(yearly_rows), use_container_width=True, hide_index=True)
