import json
import os
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


def _load_trade_log() -> list:
    """trade_log.json 로드. 없으면 빈 리스트."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(project_root, "trade_log.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _render_trade_log():
    """최근 주문 시도/결과 로그 표시."""
    st.markdown("### 주문 로그")

    # GitHub에서 최신 로그 pull
    import subprocess
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        subprocess.run(
            ["git", "fetch", "origin", "--quiet"],
            cwd=project_root, capture_output=True, timeout=10,
        )
        subprocess.run(
            ["git", "checkout", "origin/master", "--", "trade_log.json"],
            cwd=project_root, capture_output=True, timeout=10,
        )
    except Exception:
        pass

    logs = _load_trade_log()
    if not logs:
        st.info("주문 로그가 없습니다. 자동/수동 주문 실행 시 기록됩니다.")
        return

    rows = []
    for entry in logs[:50]:
        rows.append({
            "시각": entry.get("time", ""),
            "모드": entry.get("mode", ""),
            "종목": entry.get("ticker", ""),
            "방향": entry.get("side", ""),
            "전략": entry.get("strategy", entry.get("pct", "")),
            "결과": entry.get("result", ""),
            "상세": entry.get("detail", "")[:80],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_coin_logic_guide():
    st.markdown("### 매수/매도 로직 규칙")
    with st.expander("자동매매 규칙 (변경 금지)", expanded=True):
        st.markdown(
            """
**실행 시점**: VM cron 자동 실행 + GitHub Actions 수동 실행

- 자동: KST 01:00, 05:00, 09:00, 13:00, 17:00, 21:00
- 수동 주문: Streamlit UI → GitHub Actions → VM → Upbit API

**실행 순서**: 전체 시그널 분석 → 매도 먼저 실행 (현금 확보) → 현금 비례 배분 매수

**매매 판단** (마지막 완성 봉 기준)

| 현재 상태 | 시그널 | 실행 내용 |
|-----------|--------|-----------|
| 코인 미보유 | 매수 시그널 | **매수** — 현금에서 비중 비례 배분 |
| 코인 미보유 | 매도/중립 | **대기** — 현금 보존 |
| 코인 보유 중 | 매도 시그널 | **매도** — 전략 비중 비례 매도 |
| 코인 보유 중 | 매수/중립 | **유지** — 계속 보유 |

**전환 감지**: 이전 상태(signal_state.json)와 비교하여 상태 전환 시에만 주문 실행

**시그널 발생 조건**

| | 매수 시그널 | 매도 시그널 |
|---|---------|---------|
| **SMA** | 종가 > 이동평균선 | 종가 < 이동평균선 |
| **Donchian** | 종가 > N일 최고가 돌파 | 종가 < M일 최저가 이탈 |

**수동 주문** (전략 분석 없이 즉시 실행)
- 코인/방향/비율 지정 → VM 경유 즉시 매수/매도
- 최소 주문 금액: 5,000원
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
        st.divider()
        _render_trade_log()
    elif mode == "ISA":
        st.subheader("ISA 위대리 전략 트리거")
        st.info("매주 금요일 15:10 KST 주문 실행")
    elif mode == "PENSION":
        st.subheader("연금저축 전략 트리거")
        st.info("매월 25~31일 평일 15:10 KST 주문 실행")
