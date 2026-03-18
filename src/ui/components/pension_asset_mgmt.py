"""연금저축 자산관리 탭 — 전략별 매매일지+입출금 통합 테이블."""
import streamlit as st
import pandas as pd
from datetime import date

from src.ui.components.asset_mgmt import (
    load_asset_mgmt, save_asset_mgmt, calc_invested_capital, _running_balance,
)


def render_pension_asset_mgmt_tab(
    pen_port_edited,
    pen_bal_key: str,
):
    """연금저축 자산관리 탭."""
    st.header("자산관리")

    am = load_asset_mgmt()
    pen_strategies = am.setdefault("pension", {}).setdefault("strategies", {})

    # 전략 키 목록
    strategy_keys = []
    if pen_port_edited is not None and not pen_port_edited.empty:
        for _, row in pen_port_edited.iterrows():
            name = str(row.get("strategy", ""))
            weight = float(row.get("weight", 0))
            if not name or weight <= 0:
                continue
            strategy_keys.append((name, f"{name} ({weight:.0f}%)", weight))

    if not strategy_keys:
        st.info("포트폴리오에 활성 전략이 없습니다.")
        return

    # ── 전략 선택 ──
    _labels = [label for _, label, _ in strategy_keys]
    _sel_label = st.selectbox("전략 선택", _labels, key="pam_strategy_sel")
    _sel_idx = _labels.index(_sel_label)
    _sel_key, _sel_label, _sel_weight = strategy_keys[_sel_idx]

    cfg = pen_strategies.get(_sel_key, {})
    if not cfg:
        cfg = {"label": _sel_label, "transactions": []}
        pen_strategies[_sel_key] = cfg
    cfg.setdefault("transactions", [])
    cfg["label"] = _sel_label

    # ── 거래 추가 폼 ──
    st.subheader("거래 입력")
    fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 2, 3])
    with fc1:
        tx_date = st.date_input("날짜", value=date.today(), key=f"pam_tx_date_{_sel_key}")
    with fc2:
        tx_type = st.selectbox("구분", ["입금", "출금"], key=f"pam_tx_type_{_sel_key}")
    with fc3:
        tx_amount = st.number_input("금액 (KRW)", min_value=0.0, step=100000.0, format="%.0f", key=f"pam_tx_amt_{_sel_key}")
    with fc4:
        tx_memo = st.text_input("메모", key=f"pam_tx_memo_{_sel_key}")

    if st.button("추가", key=f"pam_tx_add_{_sel_key}") and tx_amount > 0:
        cfg["transactions"].append({
            "date": str(tx_date),
            "type": "deposit" if tx_type == "입금" else "withdraw",
            "amount": tx_amount,
            "memo": tx_memo.strip(),
        })
        cfg["transactions"].sort(key=lambda x: x.get("date", ""))
        save_asset_mgmt(am)
        st.rerun()

    # ── 거래 일지 테이블 ──
    st.divider()
    st.subheader(f"{_sel_label} — 거래 일지")

    txs = cfg.get("transactions", [])
    if not txs:
        st.info("거래 내역이 없습니다. 위에서 입금/출금을 추가하세요.")
    else:
        balances = _running_balance(txs)
        rows = []
        for i, tx in enumerate(txs):
            t = tx.get("type", "")
            amt = float(tx.get("amount", 0) or 0)
            rows.append({
                "날짜": tx.get("date", ""),
                "구분": "입금" if t == "deposit" else "출금",
                "금액": f"{amt:,.0f}",
                "잔액": f"{balances[i]:,.0f}",
                "메모": tx.get("memo", ""),
            })

        df = pd.DataFrame(rows)

        def _style_type(val):
            if val == "입금":
                return "color: #2196F3; font-weight: bold"
            elif val == "출금":
                return "color: #F44336; font-weight: bold"
            return ""

        styled = df.style.applymap(_style_type, subset=["구분"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        dc1, dc2 = st.columns([1, 5])
        with dc1:
            del_idx = st.number_input("삭제할 행 번호", min_value=1, max_value=len(txs), value=len(txs), step=1, key=f"pam_del_idx_{_sel_key}")
        with dc2:
            st.write("")
            st.write("")
            if st.button("해당 행 삭제", key=f"pam_del_btn_{_sel_key}"):
                txs.pop(int(del_idx) - 1)
                save_asset_mgmt(am)
                st.rerun()

    # ── 현재 잔고 (KIS) ──
    bal = st.session_state.get(pen_bal_key) or {}
    total_eval = float(bal.get("total_eval", 0) or 0)
    if total_eval <= 0:
        cash = float(bal.get("cash", 0) or bal.get("buyable_cash", 0) or 0)
        stock = float(bal.get("stock_eval", 0) or 0)
        if stock <= 0:
            stock = sum(float(h.get("eval_amt", 0) or 0) for h in (bal.get("holdings", []) or []))
        total_eval = cash + stock

    # ── 전략별 현황 테이블 ──
    st.divider()
    st.subheader("전략별 현황")

    total_weight = max(sum(w for _, _, w in strategy_keys), 1.0)
    summary_rows = []
    total_invested = 0.0
    total_current = 0.0

    for key, label, weight in strategy_keys:
        s_cfg = pen_strategies.get(key, {})
        invested = calc_invested_capital(s_cfg)
        total_invested += invested

        current_eval = total_eval * (weight / total_weight)
        total_current += current_eval

        pnl = current_eval - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0.0

        _txs = s_cfg.get("transactions", [])
        _dep_sum = sum(float(t.get("amount", 0)) for t in _txs if t.get("type") == "deposit")
        _wd_sum = sum(float(t.get("amount", 0)) for t in _txs if t.get("type") == "withdraw")

        summary_rows.append({
            "전략": label,
            "입금합계": f"{_dep_sum:,.0f}",
            "출금합계": f"{_wd_sum:,.0f}",
            "투입원금": f"{invested:,.0f}",
            "현재평가": f"{current_eval:,.0f}",
            "수익금": f"{pnl:+,.0f}",
            "수익률": f"{pnl_pct:+.2f}%",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── 전체 포트폴리오 합산 ──
    st.divider()
    st.subheader("전체 포트폴리오 합산")
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 투입원금", f"{total_invested:,.0f} KRW")
    c2.metric("총 현재평가", f"{total_current:,.0f} KRW")
    c3.metric("총 수익금", f"{total_pnl:+,.0f} KRW")
    c4.metric("총 수익률", f"{total_pnl_pct:+.2f}%")

    st.session_state["_asset_mgmt_total_invested_pension"] = total_invested
