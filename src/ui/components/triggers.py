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


# ─── 현재 작동 아키텍처 상세 ───────────────────────────────────

def _render_architecture_detail():
    """현재 정상 작동 중인 시스템 아키텍처 상세 안내."""

    st.markdown("### 현재 작동 아키텍처")

    # 1) 매매 실행 경로
    st.markdown("#### 1. 코인 자동매매 (VM 스케줄러 — Primary)")
    st.markdown(
        "```\n"
        "Google VM (vm_scheduler.py)\n"
        "  ↓ 5초 루프, minute==0 and hour in (1,5,9,13,17,21) KST\n"
        "  ↓ vm_run_job.sh upbit\n"
        "  ↓ github_action_trade.py (최신 코드 — ensure가 5분마다 git reset)\n"
        "  ↓ 전략 분석 → 시그널 전환 감지 → 매도 → 매수 → 보충매수/매도\n"
        "  ↓ trade_log.json에 결과 기록\n"
        "  → Upbit API 주문 실행\n"
        "```"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**실행 시각 (KST 정시)**")
        st.dataframe(pd.DataFrame([
            {"시각": "01:00", "4H 전략": "✅", "1D 전략": "❌"},
            {"시각": "05:00", "4H 전략": "✅", "1D 전략": "❌"},
            {"시각": "09:00", "4H 전략": "✅", "1D 전략": "✅"},
            {"시각": "13:00", "4H 전략": "✅", "1D 전략": "❌"},
            {"시각": "17:00", "4H 전략": "✅", "1D 전략": "❌"},
            {"시각": "21:00", "4H 전략": "✅", "1D 전략": "❌"},
        ]), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**실행 윈도우**")
        st.markdown(
            "- 정시 + **55분 이내** 실행 허용\n"
            "- 예: 01:00~01:55 사이 도착 → 실행\n"
            "- 55분 초과 시 SKIP (보충매수/매도만 실행)\n"
            "- 1D 전략: hour==9 일 때만 실행\n"
        )

    # 2) 수동 주문
    st.markdown("#### 2. 수동 주문 (GitHub Actions — workflow_dispatch)")
    st.markdown(
        "```\n"
        "Streamlit UI → GitHub Actions (coin_trade.yml) → workflow_dispatch\n"
        "  → trade 또는 manual_order job\n"
        "  → SSH → VM → Upbit API\n"
        "  → SCP + git push (잔고/시그널/로그 동기화)\n"
        "```"
    )
    st.info("coin_trade.yml의 schedule cron은 **trade job을 실행하지 않음** (VM 스케줄러 전용). "
            "schedule은 account_sync만 트리거.")

    # 3) 코드 업데이트 경로
    st.markdown("#### 3. VM 코드 업데이트 (vm_scheduler.yml — 5분 cron)")
    st.markdown(
        "```\n"
        "GitHub Actions (vm_scheduler.yml) → */5 cron\n"
        "  → SSH → VM: git fetch + git reset --hard origin/master\n"
        "     (상태 파일 백업/복원: balance_cache, signal_state, trade_log)\n"
        "  → .vm_runtime_env 갱신 (API 키, 시크릿)\n"
        "  → 스케줄러 ensure (프로세스 생존 확인, heartbeat 체크)\n"
        "  → SCP: VM → Runner (balance_cache, signal_state, trade_log)\n"
        "  → git commit + push (Streamlit UI용 동기화)\n"
        "```"
    )

    # 4) 잔고 동기화
    st.markdown("#### 4. 잔고/시그널 동기화 경로")
    st.dataframe(pd.DataFrame([
        {"경로": "VM 로컬 저장", "주기": "30초", "내용": "balance_cache.json (잔고+현재가)", "방식": "Upbit API → 파일 저장"},
        {"경로": "VM → GitHub (Primary)", "주기": "trade 후 즉시", "내용": "balance_cache + signal_state + trade_log", "방식": "vm_run_job.sh → GH_PAT HTTPS push"},
        {"경로": "VM → GitHub (Primary)", "주기": "30분", "내용": "balance_cache + signal_state + trade_log", "방식": "vm_scheduler.py → GH_PAT git push"},
        {"경로": "VM → GitHub (보조)", "주기": "15분 (cron)", "내용": "balance_cache + signal_state + trade_log", "방식": "vm_scheduler.yml ensure → SCP → push"},
        {"경로": "GitHub → Streamlit", "주기": "페이지 로드 시", "내용": "모든 캐시 파일", "방식": "git fetch + checkout origin/master"},
    ]), use_container_width=True, hide_index=True)

    st.info(
        "**VM 직접 push (2026-03-08 변경)**: GH_PAT HTTPS로 VM에서 직접 push.\n"
        "GH Actions cron 스로틀링(무료 플랜에서 수시간 정지 가능) 의존 해소."
    )


def _render_error_history():
    """과거 에러 및 수정 이력."""
    st.markdown("### 에러 수정 이력")

    errors = [
        {
            "날짜": "2026-03-08",
            "문제": "잔고 캐시 510분(8.5시간) 미갱신",
            "원인": "GH Actions */5 cron 스로틀링으로 vm_scheduler.yml ensure 완전 정지\n"
                    "잔고 push가 100% GH Actions에 의존 → cron 멈추면 push도 멈춤\n"
                    "vm_run_job.sh의 SSH push도 deploy key 없어 항상 실패 (deadcode)",
            "수정": "1) vm_run_job.sh: SSH URL → GH_PAT HTTPS push (trade 후 즉시)\n"
                    "2) vm_scheduler.py: workflow_dispatch → 직접 git push (30분마다)\n"
                    "3) vm_scheduler.yml: */5 → */15 (보조 역할, 무료분수 절약)",
            "파일": "vm_run_job.sh, vm_scheduler.py, vm_scheduler.yml",
        },
        {
            "날짜": "2026-03-08",
            "문제": "보충매수 실행 안 됨 (21시)",
            "원인": "1) _is_coin_interval_due() 10분 윈도우 → 지연 도착 시 SKIP\n"
                    "2) Donchian HOLD 시그널 → 보충매수 코드가 BUY/SELL만 인식",
            "수정": "1) 실행 윈도우 10분 → 55분 확대\n"
                    "2) HOLD → BUY 매핑 추가 (Path A + Path B)",
            "파일": "scripts/github_action_trade.py",
        },
        {
            "날짜": "2026-03-08",
            "문제": "GH Actions trade job이 VM과 중복 실행 (20분 지연)",
            "원인": "coin_trade.yml schedule cron + VM 스케줄러 동시 실행\n"
                    "GH Actions cron 스로틀링으로 20분+ 지연",
            "수정": "coin_trade.yml trade job에서 schedule 트리거 제거\n"
                    "→ VM 스케줄러만 정시 실행 담당",
            "파일": ".github/workflows/coin_trade.yml",
        },
        {
            "날짜": "2026-03-08",
            "문제": "잔고 캐시 30분+ 미갱신",
            "원인": "vm_scheduler.yml ensure의 git reset --hard가\n"
                    "VM의 balance_cache.json을 stale 버전으로 덮어씀\n"
                    "→ SCP 직후 stale 파일 다운로드",
            "수정": "git reset 전후 상태 파일 백업/복원 패턴 추가\n"
                    "(balance_cache, signal_state, trade_log)",
            "파일": ".github/workflows/vm_scheduler.yml",
        },
        {
            "날짜": "2026-03-07",
            "문제": "trade_log.json이 Streamlit에 미표시",
            "원인": "SCP sync에 trade_log.json 미포함",
            "수정": "ensure, balance_sync, trade job에 trade_log.json SCP 추가",
            "파일": "vm_scheduler.yml, coin_trade.yml",
        },
        {
            "날짜": "2026-03-06",
            "문제": "signal_state.json이 git reset으로 덮어써짐 → 반복 매수",
            "원인": "account_sync SSH의 git reset --hard가 signal_state를\n"
                    "GitHub 버전(SELL)으로 복귀 → 다음 trade에서 동일 전환 반복 감지",
            "수정": "모든 SSH 스크립트에 cp backup/restore 패턴 추가",
            "파일": "coin_trade.yml, vm_scheduler.yml, vm_run_job.sh",
        },
        {
            "날짜": "2026-03-05",
            "문제": "VM → GitHub push 실패 (Permission denied)",
            "원인": "github.token (ghs_)은 runner 전용, VM에서 사용 불가",
            "수정": "VM에서 직접 push 포기 → SCP → runner → push 패턴 도입",
            "파일": "coin_trade.yml, vm_scheduler.yml",
        },
        {
            "날짜": "2026-03-05",
            "문제": "Upbit Orders API 401 (invalid_query_payload)",
            "원인": "pyupbit JWT 인증 방식 불일치\n"
                    "(requests.get(params=...) 사용 → pyupbit은 data= 사용)",
            "수정": "pyupbit._request_headers(params) + data=params 방식 사용",
            "파일": "scripts/github_action_trade.py",
        },
    ]

    for e in errors:
        with st.expander(f"[{e['날짜']}] {e['문제']}", expanded=False):
            st.markdown(f"**원인:**\n```\n{e['원인']}\n```")
            st.markdown(f"**수정:**\n```\n{e['수정']}\n```")
            st.markdown(f"**파일:** `{e['파일']}`")


def _render_coin_logic_guide():
    st.markdown("### 매수/매도 로직 규칙")
    with st.expander("자동매매 규칙", expanded=True):
        st.markdown(
            """
**실행 주체**: Google VM 스케줄러 (정시 실행, 지연 없음)

- 자동: KST 01:00, 05:00, 09:00, 13:00, 17:00, 21:00
- 수동 주문: Streamlit UI → GitHub Actions workflow_dispatch → VM → Upbit API

**실행 순서**: 전체 시그널 분석 → 매도 먼저 실행 (현금 확보) → 현금 비례 배분 매수 → 보충매수/매도

**매매 판단** (마지막 완성 봉 기준)

| 현재 상태 | 시그널 | 실행 내용 |
|-----------|--------|-----------|
| 코인 미보유 | 매수 시그널 | **매수** — 현금에서 비중 비례 배분 |
| 코인 미보유 | 매도/중립 | **대기** — 현금 보존 |
| 코인 보유 중 | 매도 시그널 | **매도** — 전략 비중 비례 매도 |
| 코인 보유 중 | 매수/중립 | **유지** + 보충매수 실행 |

**전환 감지**: signal_state.json과 비교, 상태 전환 시에만 주문 실행

**보충매수/매도**: BUY 상태(HOLD 포함) 유지 시 목표 금액 대비 부족분 보충

**시그널 발생 조건**

| | 매수 시그널 | 매도 시그널 |
|---|---------|---------|
| **SMA** | 종가 > 이동평균선 | 종가 < 이동평균선 |
| **Donchian** | 종가 > N일 최고가 돌파 | 종가 < M일 최저가 이탈 |
| **Donchian HOLD** | 상단/하단 사이 → 이전 상태 유지 | (보충매수 대상) |
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
            iv = _iv_label(p.get("interval", "day"))
            strat = p.get("strategy", "SMA")
            param = p.get("parameter", "")
            sell_param = p.get("sell_parameter", "")
            weight = p.get("weight", 0)

            if iv == "4H":
                schedule = "01/05/09/13/17/21시 (6회/일)"
            elif iv == "1D":
                schedule = "09시 (1회/일)"
            else:
                schedule = "01/05/09/13/17/21시"

            trigger_detail = f"VM 스케줄러 정시 → 55분 윈도우"
            rows.append({
                "#": i,
                "종목": ticker,
                "전략": f"{strat}({param})",
                "매도파라미터": sell_param if sell_param else "-",
                "주기": iv,
                "비중": f"{weight}%",
                "실행 스케줄": schedule,
                "트리거": trigger_detail,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("등록된 코인 전략이 없습니다.")

        # 상세 아키텍처
        _render_architecture_detail()

        st.divider()
        _render_coin_logic_guide()

        st.divider()
        _render_error_history()

        st.divider()
        _render_trade_log()

    elif mode == "ISA":
        st.subheader("ISA 위대리 전략 트리거")
        st.info("매주 금요일 15:10 KST 주문 실행")
    elif mode == "PENSION":
        st.subheader("연금저축 전략 트리거")
        st.info("매월 25~31일 평일 15:10 KST 주문 실행")
