import streamlit as st
import time
import pandas as pd
from src.utils.formatting import _safe_float

def _get_kis_token(trader, acct_key, ak):
    """ISA와 연금저축에서 공용으로 사용하는 KIS 토큰 관리 로직 통합."""
    _token_key = f"kis_token_{acct_key}"
    _shared_key = f"kis_token_shared_{str(ak or '')[-8:]}"
    
    _cached = st.session_state.get(_token_key) or st.session_state.get(_shared_key)
    
    if _cached and (_cached.get("expiry", 0) - time.time()) > 300:
        trader.access_token = _cached.get("token")
        trader.token_expiry = float(_cached.get("expiry", 0))
        return True
    
    if trader.auth():
        _new_tok = {"token": trader.access_token, "expiry": trader.token_expiry}
        st.session_state[_token_key] = _new_tok
        st.session_state[_shared_key] = _new_tok
        return True
    return False

def _compute_kis_balance_summary(bal):
    """KIS 계좌 잔고 데이터를 공통 형식으로 요약."""
    if not bal or "output1" not in bal:
        return {"buyable_cash": 0.0, "holdings": [], "stock_eval": 0.0, "total_eval": 0.0}
    
    out1 = bal.get("output1", [])
    out2 = bal.get("output2", [{}])[0]
    
    holdings = []
    for h in out1:
        qty = _safe_float(h.get("hldg_qty", 0))
        if qty > 0:
            holdings.append({
                "code": h.get("pdno"),
                "name": h.get("prdt_name"),
                "qty": qty,
                "price": _safe_float(h.get("prpr")),
                "eval_amt": _safe_float(h.get("evl_amt_smtl_amt")),
                "pnl_pct": _safe_float(h.get("evl_pnl_rt"))
            })
            
    return {
        "buyable_cash": _safe_float(out2.get("dnca_tot_amt")),
        "holdings": holdings,
        "stock_eval": _safe_float(out2.get("scts_evl_amt")),
        "total_eval": _safe_float(out2.get("tot_evl_amt"))
    }
