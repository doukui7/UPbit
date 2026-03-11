"""거래 내역 탭 — 실제 거래 내역, VM 매매 로그, 슬리피지 분석."""
import json
import os
import subprocess
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import src.engine.data_cache as data_cache
from src.ui.coin_utils import load_balance_cache, ttl_cache, PROJECT_ROOT

INTERVAL_MAP = {
    "1D": "day",
    "4H": "minute240",
    "1H": "minute60",
    "30m": "minute30",
    "15m": "minute15",
    "5m": "minute5",
    "1m": "minute1",
}


def render_trade_history_tab(trader, portfolio_list):
    st.header("거래 내역")

    hist_tab1, hist_tab2, hist_tab3 = st.tabs(["💸 실제 거래 내역 (거래소)", "📋 VM 매매 로그", "📊 슬리피지 분석"])

    with hist_tab1:
        st.subheader("실제 거래 내역")

        c_h1, c_h2 = st.columns(2)
        h_type = c_h1.selectbox("조회 유형", ["전체", "입금", "출금", "체결 주문"])
        h_curr = c_h2.selectbox("화폐", ["전체", "KRW", "BTC", "ETH", "XRP", "SOL", "USDT", "DOGE", "ADA", "AVAX", "LINK"])

        d_h1, d_h2 = st.columns(2)
        h_date_start = d_h1.date_input("조회 시작일", value=datetime.now().date() - timedelta(days=90), key="hist_start")
        h_date_end = d_h2.date_input("조회 종료일", value=datetime.now().date(), key="hist_end")

        def _parse_deposit_withdraw(raw, type_label):
            rows = []
            for r in raw:
                done = r.get('done_at', r.get('created_at', ''))
                if pd.notna(done):
                    try: done = pd.to_datetime(done).strftime('%Y-%m-%d %H:%M')
                    except: pass
                amount = float(r.get('amount', 0))
                fee_val = float(r.get('fee', 0))
                state = r.get('state', '')
                state_kr = {"ACCEPTED": "완료", "REJECTED": "거부", "CANCELLED": "취소", "PROCESSING": "처리중", "WAITING": "대기중"}.get(state, state)
                rows.append({
                    "거래일시": done, "유형": type_label,
                    "화폐/코인": r.get('currency', ''),
                    "구분": type_label,
                    "수량": f"{amount:,.4f}" if amount < 100 else f"{amount:,.0f}",
                    "단가": "-",
                    "체결금액(KRW)": "-",
                    "수수료": f"{fee_val:,.4f}" if fee_val > 0 else "-",
                    "상태": state_kr,
                    "_sort_dt": done,
                })
            return rows

        def _parse_orders(raw):
            rows = []
            for r in raw:
                market = r.get('market', '')
                coin = market.split('-')[1] if '-' in str(market) else market
                side = r.get('side', '')
                side_kr = "매수" if side == 'bid' else ("매도" if side == 'ask' else side)
                state = r.get('state', '')
                executed_vol = float(r.get('executed_volume', 0) or 0)
                total_vol = float(r.get('volume', 0) or 0)
                # 부분체결 구분: cancel이지만 체결수량이 있으면 "부분체결"
                if state == 'cancel' and executed_vol > 0:
                    fill_pct = (executed_vol / total_vol * 100) if total_vol > 0 else 0
                    state_kr = f"부분체결({fill_pct:.0f}%)"
                elif state == 'cancel':
                    state_kr = "취소"
                else:
                    state_kr = {"done": "체결완료", "wait": "대기"}.get(state, state)
                price = float(r.get('price', 0) or 0)
                avg_price = float(r.get('avg_price', 0) or 0)
                paid_fee = float(r.get('paid_fee', 0) or 0)
                unit_price = avg_price if avg_price > 0 else price
                if unit_price > 0 and executed_vol > 0:
                    total_krw = unit_price * executed_vol
                elif 'trades' in r and r['trades']:
                    total_krw = sum(float(t.get('funds', 0)) for t in r['trades'])
                else:
                    total_krw = price
                ord_type = r.get('ord_type', '')
                type_kr = {"limit": "지정가", "price": "시장가(매수)", "market": "시장가(매도)"}.get(ord_type, ord_type)
                created = r.get('created_at', '')
                if pd.notna(created):
                    try: created = pd.to_datetime(created).strftime('%Y-%m-%d %H:%M')
                    except: pass
                # 주문수량 vs 체결수량 표시
                if total_vol > 0 and executed_vol != total_vol and executed_vol > 0:
                    exec_str = f"{executed_vol:,.8f}" if executed_vol < 1 else f"{executed_vol:,.4f}"
                    ord_str = f"{total_vol:,.8f}" if total_vol < 1 else f"{total_vol:,.4f}"
                    vol_str = f"{exec_str} / {ord_str}"
                else:
                    vol_str = f"{executed_vol:,.8f}" if executed_vol < 1 else f"{executed_vol:,.4f}"
                price_str = f"{unit_price:,.0f}" if unit_price > 0 else "-"
                rows.append({
                    "거래일시": created, "유형": f"체결({type_kr})",
                    "화폐/코인": coin,
                    "구분": side_kr,
                    "수량": vol_str,
                    "단가": price_str,
                    "체결금액(KRW)": f"{total_krw:,.0f}",
                    "수수료": f"{paid_fee:,.2f}",
                    "상태": state_kr,
                    "_sort_dt": created,
                })
            return rows

        def _display_history_df(all_rows):
            if all_rows:
                result_df = pd.DataFrame(all_rows)
                try:
                    result_df['_dt'] = pd.to_datetime(result_df['_sort_dt'], errors='coerce')
                    mask = (result_df['_dt'].dt.date >= h_date_start) & (result_df['_dt'].dt.date <= h_date_end)
                    result_df = result_df[mask].sort_values('_dt', ascending=False)
                except Exception:
                    pass
                result_df = result_df.drop(columns=['_sort_dt', '_dt'], errors='ignore')
                if len(result_df) > 0:
                    st.success(f"{len(result_df)}건 조회됨")
                    def _color_side(val):
                        if val == "매수": return "color: #e74c3c"
                        elif val == "매도": return "color: #2980b9"
                        elif val == "입금": return "color: #27ae60"
                        elif val == "출금": return "color: #8e44ad"
                        return ""
                    st.dataframe(
                        result_df.style.map(_color_side, subset=["구분"]),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.warning("해당 기간에 내역이 없습니다.")
            else:
                st.warning(f"조회 결과 없음. (유형: {h_type}, 화폐: {h_curr})")

        def _get_rows_from_cache(acct, h_type, h_curr):
            api_curr = None if h_curr == "전체" else h_curr
            rows = []
            if h_type in ("전체", "입금"):
                rows.extend(_parse_deposit_withdraw(acct.get("deposits", []), "입금"))
            if h_type in ("전체", "출금"):
                rows.extend(_parse_deposit_withdraw(acct.get("withdraws", []), "출금"))
            if h_type in ("전체", "체결 주문"):
                rows.extend(_parse_orders(acct.get("orders", [])))
            if api_curr and rows:
                rows = [r for r in rows if api_curr.upper() in r.get("화폐/코인", "").upper()]
            return rows

        # ── 조회 ──
        _hist_btn_col1, _hist_btn_col2 = st.columns([1, 3])
        with _hist_btn_col1:
            _do_query = st.button("조회", key="hist_query", type="primary")
        with _hist_btn_col2:
            _acct_cache = load_balance_cache()
            _cache_time = _acct_cache.get("updated_at", "")
            if _cache_time:
                st.caption(f"마지막 동기화: {_cache_time}")

        if _do_query and trader:
            api_curr = None if h_curr == "전체" else h_curr
            query_types = []
            if h_type == "전체":
                query_types = [("deposit", "입금"), ("withdraw", "출금"), ("order", "체결")]
            elif "입금" in h_type:
                query_types = [("deposit", "입금")]
            elif "출금" in h_type:
                query_types = [("withdraw", "출금")]
            elif "체결" in h_type:
                query_types = [("order", "체결")]

            all_rows = []
            ip_blocked = False
            with st.spinner("API 조회 중..."):
                for api_type, label in query_types:
                    try:
                        data, err = trader.get_history(api_type, api_curr)
                        if err and ("authorization_ip" in err or "verified IP" in err or "401" in str(err)):
                            ip_blocked = True
                            break
                        if data:
                            if api_type in ("deposit", "withdraw"):
                                all_rows.extend(_parse_deposit_withdraw(data, label))
                            else:
                                all_rows.extend(_parse_orders(data))
                    except Exception as _he:
                        st.warning(f"API 오류: {_he}")
                        ip_blocked = True
                        break

            if all_rows and not ip_blocked:
                st.session_state["_hist_rows"] = all_rows
                st.session_state["_hist_source"] = "API 실시간"
            elif ip_blocked:
                if _acct_cache.get("orders") or _acct_cache.get("deposits"):
                    st.session_state["_hist_rows"] = _get_rows_from_cache(_acct_cache, h_type, h_curr)
                    st.session_state["_hist_source"] = f"캐시 ({_cache_time})"
            else:
                st.info("조회 결과가 없습니다.")

        # 세션에 저장된 결과 표시
        if st.session_state.get("_hist_rows"):
            _src = st.session_state.get("_hist_source", "")
            if _src:
                st.caption(f"데이터 출처: {_src}")
            _display_history_df(st.session_state["_hist_rows"])
        elif _acct_cache.get("orders") or _acct_cache.get("deposits"):
            st.caption(f"캐시 데이터 ({_cache_time})")
            _display_history_df(_get_rows_from_cache(_acct_cache, h_type, h_curr))

        st.caption("Upbit API 제한: 최근 100건까지 조회 가능")

    with hist_tab2:
        st.subheader("VM 매매 로그")

        _tl_path = os.path.join(PROJECT_ROOT, "trade_log.json")
        if st.button("동기화", key="tl_sync"):
            try:
                subprocess.run(
                    ["git", "fetch", "origin", "--quiet"],
                    cwd=PROJECT_ROOT, capture_output=True, timeout=10,
                )
                subprocess.run(
                    ["git", "checkout", "origin/master", "--", "trade_log.json"],
                    cwd=PROJECT_ROOT, capture_output=True, timeout=5,
                )
                st.toast("동기화 완료")
            except Exception:
                pass

        _tl_entries = []
        if os.path.exists(_tl_path):
            try:
                with open(_tl_path, "r", encoding="utf-8") as _f:
                    _tl_entries = json.load(_f)
                if not isinstance(_tl_entries, list):
                    _tl_entries = []
            except Exception:
                _tl_entries = []

        if not _tl_entries:
            st.info("VM 매매 로그가 없습니다. (trade_log.json 미동기화)")
        else:
            _tl_modes = ["전체", "auto", "manual", "signal"]
            _tl_mode = st.selectbox("모드 필터", _tl_modes, key="tl_mode_filter")

            _tl_rows = []
            for e in _tl_entries:
                mode = e.get("mode", "")
                if _tl_mode != "전체" and mode != _tl_mode:
                    continue
                side = e.get("side", "")
                side_kr = {"BUY": "매수", "SELL": "매도", "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도"}.get(side, side)
                mode_kr = {"real": "실매매", "auto": "실매매", "manual": "수동", "signal": "시그널"}.get(mode, mode)
                amount_str = e.get("amount", e.get("qty", ""))
                _tl_rows.append({
                    "시간": e.get("time", ""),
                    "모드": mode_kr,
                    "코인": e.get("ticker", ""),
                    "구분": side_kr,
                    "전략": e.get("strategy", ""),
                    "금액/수량": str(amount_str),
                    "결과": e.get("result", ""),
                    "상세": str(e.get("detail", ""))[:80],
                })

            if _tl_rows:
                _tl_df = pd.DataFrame(_tl_rows)
                st.success(f"{len(_tl_df)}건")

                def _color_trade_side(val):
                    if val in ("매수", "보충매수"):
                        return "color: #e74c3c"
                    elif val in ("매도", "보충매도"):
                        return "color: #2980b9"
                    elif val == "시그널":
                        return "color: #f39c12"
                    return ""

                def _color_result(val):
                    if val == "success":
                        return "color: #27ae60"
                    elif val == "error":
                        return "color: #e74c3c"
                    return ""

                st.dataframe(
                    _tl_df.style
                        .map(_color_trade_side, subset=["구분"])
                        .map(_color_result, subset=["결과"]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("필터 조건에 해당하는 로그 없음")

    with hist_tab3:
        st.subheader("슬리피지 분석 (실제 체결 vs 백테스트)")

        if not trader:
            st.warning("API Key가 필요합니다.")
        else:
            sa_col1, sa_col2 = st.columns(2)
            sa_ticker_list = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
            sa_ticker = sa_col1.selectbox("코인 선택", sa_ticker_list, key="sa_ticker")
            sa_interval = sa_col2.selectbox("시간봉", list(INTERVAL_MAP.keys()), key="sa_interval")

            if st.button("슬리피지 분석", key="sa_run"):
                with st.spinner("체결 데이터 조회 중..."):
                    done_orders = trader.get_done_orders(sa_ticker)

                    if not done_orders:
                        st.info("체결 완료된 주문이 없습니다.")
                    else:
                        df_orders = pd.DataFrame(done_orders)

                        if 'created_at' in df_orders.columns:
                            df_orders['date'] = pd.to_datetime(df_orders['created_at'])
                        if 'price' in df_orders.columns:
                            df_orders['exec_price'] = pd.to_numeric(df_orders['price'], errors='coerce')
                        if 'executed_volume' in df_orders.columns:
                            df_orders['exec_volume'] = pd.to_numeric(df_orders['executed_volume'], errors='coerce')

                        api_interval = INTERVAL_MAP.get(sa_interval, "day")
                        df_ohlcv = data_cache.get_ohlcv_local_first(
                            sa_ticker,
                            interval=api_interval,
                            count=200,
                            allow_api_fallback=True,
                        )

                        if df_ohlcv is not None and 'date' in df_orders.columns and 'exec_price' in df_orders.columns:
                            df_ohlcv['open_price'] = df_ohlcv['open']

                            slip_data = []
                            for _, order in df_orders.iterrows():
                                order_date = order.get('date')
                                exec_price = order.get('exec_price', 0)
                                side = order.get('side', '')

                                if pd.isna(order_date) or exec_price == 0:
                                    continue

                                if df_ohlcv.index.tz is not None and order_date.tzinfo is None:
                                    order_date = order_date.tz_localize(df_ohlcv.index.tz)

                                idx = df_ohlcv.index.searchsorted(order_date)
                                if idx < len(df_ohlcv):
                                    candle_open = df_ohlcv.iloc[idx]['open']
                                    slippage_pct = (exec_price - candle_open) / candle_open * 100
                                    if side == 'ask':
                                        slippage_pct = -slippage_pct

                                    slip_data.append({
                                        'date': order_date,
                                        'side': 'BUY' if side == 'bid' else 'SELL',
                                        'exec_price': exec_price,
                                        'candle_open': candle_open,
                                        'slippage_pct': slippage_pct,
                                        'volume': order.get('exec_volume', 0)
                                    })

                            if slip_data:
                                df_slip = pd.DataFrame(slip_data)

                                avg_slip = df_slip['slippage_pct'].mean()
                                max_slip = df_slip['slippage_pct'].max()
                                min_slip = df_slip['slippage_pct'].min()

                                sc1, sc2, sc3, sc4 = st.columns(4)
                                sc1.metric("평균 슬리피지", f"{avg_slip:.3f}%")
                                sc2.metric("최대 (불리)", f"{max_slip:.3f}%")
                                sc3.metric("최소 (유리)", f"{min_slip:.3f}%")
                                sc4.metric("거래 수", f"{len(df_slip)}건")

                                buy_slip = df_slip[df_slip['side'] == 'BUY']
                                sell_slip = df_slip[df_slip['side'] == 'SELL']

                                if not buy_slip.empty:
                                    st.caption(f"매수 평균 슬리피지: {buy_slip['slippage_pct'].mean():.3f}% ({len(buy_slip)}건)")
                                if not sell_slip.empty:
                                    st.caption(f"매도 평균 슬리피지: {sell_slip['slippage_pct'].mean():.3f}% ({len(sell_slip)}건)")

                                fig_slip = go.Figure()
                                fig_slip.add_trace(go.Bar(
                                    x=df_slip['date'], y=df_slip['slippage_pct'],
                                    marker_color=['red' if s > 0 else 'green' for s in df_slip['slippage_pct']],
                                    name='슬리피지 %'
                                ))
                                fig_slip.add_hline(y=avg_slip, line_dash="dash", line_color="blue",
                                                   annotation_text=f"Avg: {avg_slip:.3f}%")
                                fig_slip.update_layout(title="거래 슬리피지 (+ = 불리)", height=350, margin=dict(t=80),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                                st.plotly_chart(fig_slip, use_container_width=True)

                                st.dataframe(
                                    df_slip.style.format({
                                        'exec_price': '{:,.0f}',
                                        'candle_open': '{:,.0f}',
                                        'slippage_pct': '{:.3f}%',
                                        'volume': '{:.6f}'
                                    }).background_gradient(cmap='RdYlGn_r', subset=['slippage_pct']),
                                    use_container_width=True
                                )

                                st.info(
                                    f"권장 백테스트 슬리피지: **{abs(avg_slip):.2f}%** "
                                    f"(실제 평균 기반, 백테스트 탭에서 설정)"
                                )
                            else:
                                st.info("매칭 가능한 체결-캔들 데이터가 없습니다.")
                        else:
                            st.dataframe(df_orders)
                            st.caption("OHLCV 매칭 불가 - 원본 주문 데이터 표시")
