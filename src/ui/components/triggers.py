import streamlit as st
import pandas as pd

def render_strategy_trigger_tab(mode: str, coin_portfolio: list | None = None):
    """전략별 주문/알림 트리거 안내 탭 공통 렌더러."""
    st.header("전략 트리거")
    st.caption("기준: GitHub Actions 스케줄 + 현재 실행 로직")
    
    def _iv_label(iv: str) -> str:
        x = str(iv or "day").strip().lower()
        if x in {"1d", "d", "day"}: return "1D"
        if x in {"4h", "240"}: return "4H"
        if x in {"1h", "60"}: return "1H"
        return iv

    if mode == "COIN":
        st.subheader("코인 전략 트리거")
        rows = []
        for i, p in enumerate(coin_portfolio or [], start=1):
            ticker = f"{p.get('market', 'KRW')}-{str(p.get('coin', '')).upper()}"
            rows.append({
                "전략": f"{i}. {ticker} {p.get('strategy', 'SMA')}",
                "주기": _iv_label(p.get("interval", "day")),
                "주문 트리거": "전략 스케줄에 따름",
                "경로": "GitHub Actions → Upbit API"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif mode == "ISA":
        st.subheader("ISA 위대리 전략 트리거")
        st.info("매주 금요일 15:20 KST 주문 실행")
    elif mode == "PENSION":
        st.subheader("연금저축 전략 트리거")
        st.info("매월 25~31일 평일 15:20 KST 주문 실행")
