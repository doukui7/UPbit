import pandas as pd
import streamlit as st


def _iv_label(iv: str) -> str:
    x = str(iv or "day").strip().lower()
    if x in {"1d", "d", "day"}:
        return "1D"
    if x in {"4h", "240", "240m", "minute240"}:
        return "4H"
    if x in {"1h", "60", "60m", "minute60"}:
        return "1H"
    return str(iv or "")


def _render_coin_logic_guide():
    st.markdown("### 매수/매도 로직 설명")
    with st.expander("⚖️ 리밸런싱 규칙", expanded=True):
        st.markdown(
            """
**실행 시점**: VM cron 자동 실행 + GitHub Actions 수동 실행

- 자동: KST 01:00, 05:00, 09:00, 13:00, 17:00, 21:00
- 수동: GitHub Actions `Auto Trade` → `run_job=trade`

**실행 순서**: 전체 시그널 분석 → 매도 먼저 실행 (현금 확보) → 현금 비례 배분 매수

**매매 판단** (전일 종가 기준)

| 현재 상태 | 시그널 | 실행 내용 |
|-----------|--------|-----------|
| 코인 미보유 | 매수 시그널 | **매수** — 현금에서 비중 비례 배분 |
| 코인 미보유 | 매도/중립 | **대기** — 현금 보존 (비중만큼 예비) |
| 코인 보유 중 | 매도 시그널 | **매도** — 전량 시장가 매도 |
| 코인 보유 중 | 매수/중립 | **유지** — 계속 보유 (추가 매수 없음) |

**매수 금액 계산**: 보유 중인 자산은 무시, 현금을 미보유 자산 비중끼리 비례 배분

> 예) BTC 40%(보유중), ETH 30%(미보유), SOL 30%(미보유)  
> 미보유 비중 합계 = 60%  
> ETH 매수액 = 현금 × 30/60, SOL 매수액 = 현금 × 30/60

**시그널 발생 조건**

| | 매수 시그널 | 매도 시그널 |
|---|---------|---------|
| **SMA** | 종가 > 이동평균선 | 종가 < 이동평균선 |
| **Donchian** | 종가 > N일 최고가 돌파 | 종가 < M일 최저가 이탈 |
"""
        )


def render_strategy_trigger_tab(mode: str, coin_portfolio: list | None = None):
    """전략별 주문/알림 트리거 안내 탭 공통 렌더러."""
    st.header("전략 트리거")
    st.caption("기준: 자동 스케줄 + 현재 실행 로직")

    if mode == "COIN":
        st.subheader("코인 전략 트리거")
        rows = []
        for i, p in enumerate(coin_portfolio or [], start=1):
            ticker = f"{p.get('market', 'KRW')}-{str(p.get('coin', '')).upper()}"
            rows.append(
                {
                    "전략": f"{i}. {ticker} {p.get('strategy', 'SMA')}",
                    "주기": _iv_label(p.get("interval", "day")),
                    "주문 트리거": "자동(4시간 간격) / 수동 실행 가능",
                    "경로": "VM cron → github_action_trade.py → Upbit API",
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("등록된 코인 전략이 없습니다.")
        _render_coin_logic_guide()
    elif mode == "ISA":
        st.subheader("ISA 위대리 전략 트리거")
        st.info("매주 금요일 15:10 KST 주문 실행")
    elif mode == "PENSION":
        st.subheader("연금저축 전략 트리거")
        st.info("매월 25~31일 평일 15:10 KST 주문 실행")
