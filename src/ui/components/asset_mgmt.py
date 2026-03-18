"""코인 자산관리 탭 — 백테스트+입출금 통합 일지 (일별)."""
import json
import os
import streamlit as st
import pandas as pd
from datetime import date

_ASSET_MGMT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "asset_mgmt.json")
_TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "trade_log.json")


# ═════════════════════════════════════════════════════════════
# 공통 유틸 (코인/연금 공용)
# ═════════════════════════════════════════════════════════════

def load_asset_mgmt() -> dict:
    try:
        with open(_ASSET_MGMT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"coin": {"strategies": {}}, "pension": {"strategies": {}}}


def save_asset_mgmt(data: dict):
    os.makedirs(os.path.dirname(_ASSET_MGMT_PATH), exist_ok=True)
    with open(_ASSET_MGMT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calc_invested_capital(strategy_cfg: dict) -> float:
    """투입원금 = 입금 합계 - 출금 합계."""
    total = 0.0
    for tx in strategy_cfg.get("transactions", []):
        amt = float(tx.get("amount", 0) or 0)
        if tx.get("type") == "deposit":
            total += amt
        elif tx.get("type") == "withdraw":
            total -= amt
    return total


def get_earliest_start_date(section: str) -> str | None:
    """전략별 최초 거래일 중 가장 빠른 날짜."""
    data = load_asset_mgmt()
    strategies = data.get(section, {}).get("strategies", {})
    dates = []
    for cfg in strategies.values():
        sd = cfg.get("start_date", "")
        if sd:
            dates.append(sd)
        txs = cfg.get("transactions", [])
        if txs:
            dates.append(txs[0].get("date", ""))
    dates = [d for d in dates if d]
    return min(dates) if dates else None


def get_total_invested(section: str) -> float:
    """지정 섹션의 전체 투입원금 합계."""
    data = load_asset_mgmt()
    strategies = data.get(section, {}).get("strategies", {})
    return sum(calc_invested_capital(cfg) for cfg in strategies.values())


def _running_balance(transactions: list) -> list[float]:
    """거래 목록에서 잔액 누적 계산."""
    balance = 0.0
    result = []
    for tx in transactions:
        amt = float(tx.get("amount", 0) or 0)
        if tx.get("type") == "deposit":
            balance += amt
        elif tx.get("type") == "withdraw":
            balance -= amt
        result.append(balance)
    return result


# ═════════════════════════════════════════════════════════════
# 거래 데이터 헬퍼
# ═════════════════════════════════════════════════════════════

def _load_trade_log() -> list:
    try:
        with open(_TRADE_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _detect_krw_deposits() -> list:
    """Upbit API에서 KRW 입금 내역 조회 → [{date, amount}, ...]."""
    try:
        from src.trading.upbit_trader import UpbitTrader
        _cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "user_config.json")
        with open(_cfg_path, "r", encoding="utf-8") as f:
            _ucfg = json.load(f)
        ak = _ucfg.get("upbit_access_key", "")
        sk = _ucfg.get("upbit_secret_key", "")
        if not ak or not sk:
            return []
        trader = UpbitTrader(ak, sk)
        data, err = trader.get_history("deposit", "KRW")
        if err or not data:
            return []
        result = []
        for d in data:
            if d.get("state", "").lower() != "accepted":
                continue
            done = d.get("done_at", d.get("created_at", ""))
            if not done:
                continue
            amt = float(d.get("amount", 0))
            if amt > 0:
                result.append({"date": done[:10], "amount": round(amt)})
        return result
    except Exception:
        return []


def _get_daily_prices(ticker: str, interval: str) -> dict:
    """데이터 캐시에서 일별 종가(09시 기준) 추출."""
    import src.engine.data_cache as data_cache
    df = data_cache.load_cached(ticker, interval)
    if df is None or df.empty:
        return {}
    prices = {}
    col = "close" if "close" in df.columns else "Close"
    if col not in df.columns:
        return {}
    for idx in df.index:
        ds = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
        prices[ds] = float(df.at[idx, col])
    return prices


def _extract_bt_data(item: dict, start_str: str, cap: float):
    """백테스트 → (bt_trades, bt_equity_raw)."""
    import src.engine.data_cache as data_cache
    from src.backtest.engine import BacktestEngine

    bt_trades: dict = {}
    bt_equity: dict = {}
    ticker = f"KRW-{item['coin'].upper()}"
    interval = item.get("interval", "day")
    period = int(item.get("parameter", 20))
    sell_p = int(item.get("sell_parameter", 0) or 0)
    strategy = item.get("strategy", "SMA")
    sell_ratio = (sell_p / period) if (strategy == "Donchian" and sell_p > 0 and period > 0) else 0

    df = data_cache.load_cached(ticker, interval)
    if df is None or len(df) < period:
        return bt_trades, bt_equity

    result = BacktestEngine().run_backtest(
        ticker, period=period, interval=interval,
        count=len(df), start_date=start_str or "",
        initial_balance=cap, df=df,
        strategy_mode=strategy, sell_period_ratio=sell_ratio,
    )
    if not result:
        return bt_trades, bt_equity

    for t in result.get("performance", {}).get("trades", []):
        d = t["date"]
        ds = str(d.date()) if hasattr(d, "date") else str(d)[:10]
        daily = bt_trades.setdefault(ds, {"buy": 0.0, "sell": 0.0})
        if t["type"] == "buy":
            daily["buy"] += float(t.get("equity", 0) or t.get("balance", 0))
        elif t["type"] == "sell":
            daily["sell"] += float(t.get("balance", 0))

    bt_df = result.get("df")
    if bt_df is not None and "equity" in bt_df.columns:
        for idx in bt_df.index:
            ds = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
            bt_equity[ds] = float(bt_df.at[idx, "equity"])

    return bt_trades, bt_equity


def _compute_bt_asset(item, cfg, initial_cap_fallback):
    """전략의 현재 BT자산 계산 (입출금 + NAV 기반)."""
    txs = cfg.get("transactions", [])
    first_dep = next((tx for tx in txs if tx["type"] == "deposit"), None)
    bt_seed = float(first_dep["amount"]) if first_dep else (initial_cap_fallback or 100000)
    bt_start = cfg.get("start_date", "")
    if not bt_start and first_dep:
        bt_start = first_dep["date"]
    if not bt_start:
        return calc_invested_capital(cfg)

    bt_trades, bt_eq_raw = _extract_bt_data(item, bt_start, bt_seed)
    if not bt_eq_raw:
        return calc_invested_capital(cfg)

    bt_eq_sorted = sorted(bt_eq_raw.items())
    tx_by_date: dict[str, list] = {}
    for tx in txs:
        tx_by_date.setdefault(tx.get("date", ""), []).append(tx)

    try:
        tbl_start = pd.to_datetime(bt_start).date()
    except Exception:
        return calc_invested_capital(cfg)

    all_days = pd.date_range(start=tbl_start, end=date.today(), freq="D")
    bt_shares = 0.0
    bt_cash = 0.0
    last_raw = bt_seed
    eq_idx = 0

    for day in all_days:
        ds = str(day.date())
        while eq_idx < len(bt_eq_sorted) and bt_eq_sorted[eq_idx][0] <= ds:
            last_raw = bt_eq_sorted[eq_idx][1]
            eq_idx += 1
        nav = last_raw / bt_seed if bt_seed > 0 else 1.0

        for tx in tx_by_date.get(ds, []):
            amt = float(tx.get("amount", 0) or 0)
            if tx["type"] == "deposit":
                bt_cash += amt
            elif tx["type"] == "withdraw":
                bt_cash -= amt
        if bt_cash < 0 and nav > 0:
            bt_shares += bt_cash / nav
            bt_cash = 0.0

        bt = bt_trades.get(ds, {})
        if bt.get("buy", 0) > 0 and bt_cash > 0 and nav > 0:
            bt_shares += bt_cash / nav
            bt_cash = 0.0

    final_nav = last_raw / bt_seed if bt_seed > 0 else 1.0
    return bt_shares * final_nav + bt_cash


# ═════════════════════════════════════════════════════════════
# 코인 자산관리 탭
# ═════════════════════════════════════════════════════════════

def render_asset_mgmt_tab(portfolio_list: list, start_date=None, initial_cap: float = 0):
    """코인 자산관리 탭 — 일별 BT+입출금 통합."""
    from src.ui.coin_utils import make_signal_key

    st.header("자산관리")
    am = load_asset_mgmt()
    coin_strategies = am.setdefault("coin", {}).setdefault("strategies", {})

    # 전략 키 목록
    strategy_keys = []
    for item in portfolio_list:
        key = make_signal_key(item)
        label = f"{item.get('coin','')} {item.get('strategy','')}({item.get('parameter','')}, {item.get('interval','')})"
        strategy_keys.append((key, label, item))

    _labels = [lb for _, lb, _ in strategy_keys]
    if not _labels:
        st.info("포트폴리오에 전략이 없습니다.")
        return

    # ── 전략 선택 + BT 시작일 + 저장 (한 행) ──
    sc1, sc2, sc3 = st.columns([4, 2, 1])
    with sc1:
        _sel_label = st.selectbox("전략 선택", _labels, key="am_strategy_sel")
    _sel_idx = _labels.index(_sel_label)
    _sel_key, _sel_label, _sel_item = strategy_keys[_sel_idx]

    cfg = coin_strategies.get(_sel_key, {})
    if not cfg:
        cfg = {"label": _sel_label, "transactions": []}
        coin_strategies[_sel_key] = cfg
    cfg.setdefault("transactions", [])
    cfg["label"] = _sel_label

    with sc2:
        _cfg_sd = cfg.get("start_date", "")
        _def_sd = pd.to_datetime(_cfg_sd).date() if _cfg_sd else (start_date or date.today())
        strat_start = st.date_input("BT 시작일", value=_def_sd, key=f"am_start_{_sel_key}")
    with sc3:
        st.write("")
        st.write("")
        if st.button("저장", key=f"am_cfg_save_{_sel_key}"):
            cfg["start_date"] = str(strat_start)
            save_asset_mgmt(am)
            st.rerun()

    # ── 통합 거래 일지 (일별) ──
    st.divider()
    st.subheader(f"{_sel_label} — 통합 거래 일지")
    txs = cfg.get("transactions", [])
    saved_ov = cfg.get("overrides", {})

    # ── 마이그레이션: overrides의 입금/출금 → transactions ──
    _mig = False
    for _ov_ds, _ov_v in list(saved_ov.items()):
        _ov_dep = _ov_v.get("입금", 0)
        _ov_wd = _ov_v.get("출금", 0)
        if _ov_dep > 0 or _ov_wd > 0:
            _ex = [tx for tx in txs if tx.get("date") == _ov_ds]
            if _ov_dep > 0 and not any(tx.get("type") == "deposit" for tx in _ex):
                txs.append({"date": _ov_ds, "type": "deposit", "amount": _ov_dep, "memo": ""})
                _mig = True
            if _ov_wd > 0 and not any(tx.get("type") == "withdraw" for tx in _ex):
                txs.append({"date": _ov_ds, "type": "withdraw", "amount": _ov_wd, "memo": ""})
                _mig = True
            _ov_v.pop("입금", None)
            _ov_v.pop("출금", None)
            if not _ov_v:
                saved_ov.pop(_ov_ds, None)
    if _mig:
        txs.sort(key=lambda x: x.get("date", ""))
        cfg["overrides"] = saved_ov
        save_asset_mgmt(am)

    # BT 시드: 백테스트 엔진용 (NAV 비율 계산 기준, 실제 자산은 입금으로만 투입)
    _first_dep = next((tx for tx in txs if tx["type"] == "deposit"), None)
    _bt_seed = float(_first_dep["amount"]) if _first_dep else (initial_cap or 100000)

    # BT 시작일
    _bt_start_str = cfg.get("start_date", "")
    if not _bt_start_str:
        _bt_start_str = _first_dep["date"] if _first_dep else str(start_date or "2026-01-01")

    # 1. 백테스트 실행
    bt_trades, bt_equity_raw = _extract_bt_data(_sel_item, _bt_start_str, _bt_seed)

    # 2. 룩업 테이블
    tx_by_date: dict[str, list] = {}
    for tx in txs:
        tx_by_date.setdefault(tx.get("date", ""), []).append(tx)

    bt_eq_sorted = sorted(bt_equity_raw.items()) if bt_equity_raw else []

    # 3. 일별 테이블 생성
    try:
        _tbl_start = pd.to_datetime(_bt_start_str).date()
    except Exception:
        _tbl_start = start_date or date.today()
    all_days = pd.date_range(start=_tbl_start, end=date.today(), freq="D")

    if len(all_days) == 0:
        st.info("시작일을 설정하세요.")
    else:
        bt_shares = 0.0
        bt_cash = 0.0
        last_bt_raw = _bt_seed
        bt_cum_fee = 0.0
        bt_eq_idx = 0
        bt_in_coin = False
        rows = []
        auto_vals = {}
        bt_asset_series = []
        cum_deposit = 0.0

        # 자동 입금 감지
        if not txs and not cfg.get("_auto_detect_done"):
            cfg["_auto_detect_done"] = True
            _api_deps = _detect_krw_deposits()
            if _api_deps:
                for dep in _api_deps:
                    cfg["transactions"].append(
                        {"date": dep["date"], "type": "deposit",
                         "amount": dep["amount"], "memo": "API자동감지"})
                txs = cfg["transactions"]
                txs.sort(key=lambda x: x.get("date", ""))
                tx_by_date.clear()
                for tx in txs:
                    tx_by_date.setdefault(tx.get("date", ""), []).append(tx)
            save_asset_mgmt(am)

        for day in all_days:
            ds = str(day.date())
            ov = saved_ov.get(ds, {})

            # BT equity fill-forward
            while bt_eq_idx < len(bt_eq_sorted) and bt_eq_sorted[bt_eq_idx][0] <= ds:
                last_bt_raw = bt_eq_sorted[bt_eq_idx][1]
                bt_eq_idx += 1
            bt_nav = last_bt_raw / _bt_seed if _bt_seed > 0 else 1.0

            # Auto 입출금 (from transactions)
            auto_dep = 0.0
            auto_wd = 0.0
            for tx in tx_by_date.get(ds, []):
                amt = float(tx.get("amount", 0) or 0)
                if tx["type"] == "deposit":
                    auto_dep += amt
                elif tx["type"] == "withdraw":
                    auto_wd += amt

            # Effective 입출금 (override > auto)
            eff_dep = ov.get("입금", auto_dep)
            eff_wd = ov.get("출금", auto_wd)
            auto_vals[ds] = {"입금": round(auto_dep), "출금": round(auto_wd)}

            # 입출금 → BT 현금
            bt_cash += eff_dep - eff_wd
            cum_deposit += eff_dep - eff_wd
            if bt_cash < 0 and bt_nav > 0:
                bt_shares += bt_cash / bt_nav
                bt_cash = 0.0

            # BT 매매 (user-level 스케일링)
            bt = bt_trades.get(ds, {})
            pre_user_eq = bt_shares * bt_nav + bt_cash
            _scale = pre_user_eq / last_bt_raw if last_bt_raw > 0 else 0
            user_buy = round(bt.get("buy", 0) * _scale) if bt.get("buy") else 0
            user_sell = round(bt.get("sell", 0) * _scale) if bt.get("sell") else 0

            if bt.get("sell", 0) > 0:
                if user_sell > 0:
                    bt_cum_fee += user_sell * 0.0005 / 0.9995
                bt_in_coin = False

            if bt.get("buy", 0) > 0:
                if bt_cash > 0 and bt_nav > 0:
                    bt_shares += bt_cash / bt_nav
                    bt_cash = 0.0
                bt_in_coin = True

            # 자산 계산
            strat_val = bt_shares * bt_nav
            bt_eq = strat_val + bt_cash
            if bt_in_coin:
                bt_coin_val = round(strat_val)
                bt_cash_val = round(bt_cash)
            else:
                bt_coin_val = 0
                bt_cash_val = round(strat_val + bt_cash)

            rows.append({
                "수정": "●" if ov else "",
                "날짜": ds,
                "입금": round(eff_dep),
                "출금": round(eff_wd),
                "BT매수": f"{user_buy:,}" if user_buy else "",
                "BT매도": f"{user_sell:,}" if user_sell else "",
                "BT수수료": f"{round(bt_cum_fee):,}" if bt_cum_fee > 0 else "",
                "BT코인": f"{bt_coin_val:,}",
                "BT현금": f"{bt_cash_val:,}",
                "BT자산": f"{round(bt_eq):,}",
            })
            bt_asset_series.append({"날짜": ds, "BT자산": round(bt_eq), "투입원금": round(cum_deposit)})

        rows.reverse()
        tbl = pd.DataFrame(rows)

        _ncol = st.column_config.NumberColumn
        _tcol = st.column_config.TextColumn
        edited = st.data_editor(
            tbl,
            column_config={
                "수정": _tcol("수정", width="small"),
                "날짜": _tcol("날짜", width="small"),
                "입금": _ncol("입금", format="%.0f", min_value=0, step=10000),
                "출금": _ncol("출금", format="%.0f", min_value=0, step=10000),
                "BT매수": _tcol("BT매수"),
                "BT매도": _tcol("BT매도"),
                "BT수수료": _tcol("BT수수료"),
                "BT코인": _tcol("BT코인"),
                "BT현금": _tcol("BT현금"),
                "BT자산": _tcol("BT자산"),
            },
            disabled=["수정", "날짜", "BT매수", "BT매도", "BT수수료", "BT코인", "BT현금", "BT자산"],
            use_container_width=True,
            hide_index=True,
            height=500,
            key=f"am_journal_{_sel_key}",
        )

        # ── 변경 감지 ──
        _editable = ["입금", "출금"]
        changed_dates = set()
        for idx in edited.index:
            for col in _editable:
                _ov = round(float(tbl.at[idx, col] or 0))
                _nv = round(float(edited.at[idx, col] or 0))
                if _ov != _nv:
                    changed_dates.add(edited.at[idx, "날짜"])

        # ── 저장 / 초기화 버튼 ──
        bc1, bc2, bc3 = st.columns([1.5, 1.5, 7])
        with bc1:
            if changed_dates:
                st.caption(f"미저장 {len(changed_dates)}건")
            if st.button("저장", key=f"am_save_{_sel_key}", type="primary",
                         disabled=not changed_dates):
                for idx2 in edited.index:
                    ds2 = edited.at[idx2, "날짜"]
                    if ds2 not in changed_dates:
                        continue
                    new_dep = round(float(edited.at[idx2, "입금"] or 0))
                    new_wd = round(float(edited.at[idx2, "출금"] or 0))
                    txs[:] = [tx for tx in txs if not (
                        tx.get("date") == ds2 and tx.get("type") in ("deposit", "withdraw")
                    )]
                    if new_dep > 0:
                        txs.append({"date": ds2, "type": "deposit", "amount": new_dep, "memo": ""})
                    if new_wd > 0:
                        txs.append({"date": ds2, "type": "withdraw", "amount": new_wd, "memo": ""})
                txs.sort(key=lambda x: x.get("date", ""))
                save_asset_mgmt(am)
                st.rerun()
        with bc2:
            if st.button("초기화", key=f"am_reset_{_sel_key}",
                         disabled=not saved_ov):
                cfg.pop("overrides", None)
                save_asset_mgmt(am)
                st.rerun()

        # ── 전체 자산 차트 ──
        if bt_asset_series:
            st.divider()
            st.subheader("자산 추이")
            import plotly.graph_objects as go
            chart_df = pd.DataFrame(bt_asset_series)
            chart_df["날짜"] = pd.to_datetime(chart_df["날짜"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=chart_df["날짜"], y=chart_df["BT자산"],
                mode="lines", name="BT 자산",
                line=dict(color="#2196F3", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=chart_df["날짜"], y=chart_df["투입원금"],
                mode="lines", name="투입원금 (누적)",
                line=dict(color="#9E9E9E", width=1, dash="dash"),
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                yaxis_title="KRW",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── 입출금 관리 ──
    st.divider()
    st.subheader("입출금 관리")

    if txs:
        _bal = _running_balance(txs)
        _tx_rows = []
        for i, tx in enumerate(txs):
            _tx_rows.append({
                "날짜": tx.get("date", ""),
                "유형": "입금" if tx.get("type") == "deposit" else "출금",
                "금액": f"{float(tx.get('amount', 0)):,.0f}",
                "메모": tx.get("memo", ""),
                "잔액": f"{_bal[i]:,.0f}",
            })
        st.dataframe(pd.DataFrame(_tx_rows), use_container_width=True, hide_index=True)

        # 삭제
        _del_opts = [f"{i+1}. {tx.get('date','')} {tx.get('type','')} {float(tx.get('amount',0)):,.0f}"
                     for i, tx in enumerate(txs)]
        dc1, dc2 = st.columns([3, 1])
        with dc1:
            _del_sel = st.selectbox("삭제할 항목", _del_opts, key=f"am_tx_del_sel_{_sel_key}")
        with dc2:
            st.write("")
            st.write("")
            if st.button("삭제", key=f"am_tx_del_{_sel_key}"):
                _del_idx = _del_opts.index(_del_sel)
                txs.pop(_del_idx)
                save_asset_mgmt(am)
                st.rerun()
    else:
        st.info("등록된 입출금 내역이 없습니다.")

    # 새 입출금 등록
    st.markdown("**새 입출금 등록**")
    tc1, tc2, tc3, tc4, tc5 = st.columns([2, 1.5, 2, 2, 1])
    with tc1:
        _tx_date = st.date_input("날짜", value=date.today(), key=f"am_tx_date_{_sel_key}")
    with tc2:
        _tx_type = st.selectbox("유형", ["입금", "출금"], key=f"am_tx_type_{_sel_key}")
    with tc3:
        _tx_amt = st.number_input("금액 (KRW)", min_value=0, value=0, step=10000, key=f"am_tx_amt_{_sel_key}")
    with tc4:
        _tx_memo = st.text_input("메모", key=f"am_tx_memo_{_sel_key}")
    with tc5:
        st.write("")
        st.write("")
        if st.button("등록", key=f"am_tx_add_{_sel_key}", type="primary"):
            if _tx_amt > 0:
                txs.append({
                    "date": str(_tx_date),
                    "type": "deposit" if _tx_type == "입금" else "withdraw",
                    "amount": _tx_amt,
                    "memo": _tx_memo.strip(),
                })
                txs.sort(key=lambda x: x.get("date", ""))
                save_asset_mgmt(am)
                st.rerun()
            else:
                st.warning("금액을 입력하세요.")

    # ── 자산 통합 (BT 기반) ──
    st.divider()
    st.subheader("자산 통합")

    summary_rows = []
    total_invested = 0.0
    total_bt = 0.0

    for key, label, item in strategy_keys:
        s_cfg = coin_strategies.get(key, {})
        s_inv = calc_invested_capital(s_cfg)
        total_invested += s_inv
        bt_asset = _compute_bt_asset(item, s_cfg, initial_cap)
        total_bt += bt_asset
        bt_pnl = bt_asset - s_inv
        bt_pct = (bt_pnl / s_inv * 100) if s_inv > 0 else 0.0
        summary_rows.append({
            "전략": label,
            "투입원금": f"{s_inv:,.0f}",
            "BT자산": f"{bt_asset:,.0f}",
            "수익금": f"{bt_pnl:+,.0f}",
            "수익률": f"{bt_pct:+.2f}%",
        })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    total_pnl = total_bt - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 투입원금", f"{total_invested:,.0f} KRW")
    c2.metric("총 BT자산", f"{total_bt:,.0f} KRW")
    c3.metric("총 수익금", f"{total_pnl:+,.0f} KRW")
    c4.metric("총 수익률", f"{total_pnl_pct:+.2f}%")

    st.session_state["_asset_mgmt_total_invested_coin"] = total_invested
