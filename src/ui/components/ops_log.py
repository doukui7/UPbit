"""운영 로그 탭 — GH Actions 실행, VM 매매 로그, Git 동기화, 시스템 상태, 실행 검증."""
import json
import os
import subprocess
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_KST = timezone(timedelta(hours=9))

# ── 워크플로우 목록 ──
_WORKFLOWS = [
    ("coin_trade.yml", "Coin Trade"),
    ("pension_trade.yml", "Pension Trade"),
    ("monitoring.yml", "Monitoring"),
]


def _run_gh(args: list[str], timeout: int = 15) -> str | None:
    """gh CLI 실행 후 stdout 반환. 실패 시 None."""
    try:
        r = subprocess.run(
            ["gh"] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=_PROJECT_ROOT,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def _run_git(args: list[str], timeout: int = 10) -> str | None:
    """git 명령 실행 후 stdout 반환."""
    try:
        r = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=_PROJECT_ROOT,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def _load_json(filename: str) -> dict | list | None:
    """프로젝트 루트의 JSON 파일 로드."""
    path = os.path.join(_PROJECT_ROOT, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sync_from_github():
    """GitHub에서 최신 상태 파일 동기화."""
    _run_git(["fetch", "origin", "--quiet"])
    files = ["trade_log.json", "balance_cache.json", "signal_state.json", "account_cache.json"]
    for f in files:
        _run_git(["checkout", "origin/master", "--", f])


# ═══════════════════════════════════════════
# 서브탭 1: GH Actions 실행 내역
# ═══════════════════════════════════════════
def _render_gh_actions():
    st.subheader("GH Actions 실행 내역")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("새로고침", key="ops_gh_refresh"):
            st.rerun()

    all_runs = []
    for wf_file, wf_name in _WORKFLOWS:
        out = _run_gh([
            "run", "list", f"--workflow={wf_file}", "--limit", "10",
            "--json", "databaseId,status,conclusion,event,updatedAt,name",
        ])
        if out:
            try:
                runs = json.loads(out)
                for r in runs:
                    r["workflow"] = wf_name
                all_runs.extend(runs)
            except Exception:
                pass

    if not all_runs:
        st.warning("GH Actions 실행 내역을 가져올 수 없습니다. (gh CLI 인증 필요)")
        return

    # 시간순 정렬 (최신 먼저)
    all_runs.sort(key=lambda x: x.get("updatedAt", ""), reverse=True)

    rows = []
    for r in all_runs:
        updated = r.get("updatedAt", "")
        if updated:
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                kst_str = dt.astimezone(_KST).strftime("%m-%d %H:%M")
            except Exception:
                kst_str = updated[:16]
        else:
            kst_str = ""

        status = r.get("status", "")
        conclusion = r.get("conclusion", "")
        if status == "completed":
            state_str = conclusion or "completed"
        else:
            state_str = status

        rows.append({
            "시간(KST)": kst_str,
            "워크플로우": r.get("workflow", ""),
            "이벤트": r.get("event", ""),
            "상태": state_str,
            "ID": r.get("databaseId", ""),
        })

    df = pd.DataFrame(rows)

    def _color_state(val):
        if val == "success":
            return "color: #27ae60; font-weight: bold"
        elif val == "failure":
            return "color: #e74c3c; font-weight: bold"
        elif val in ("in_progress", "queued"):
            return "color: #f39c12; font-weight: bold"
        return ""

    st.dataframe(
        df.style.map(_color_state, subset=["상태"]),
        use_container_width=True, hide_index=True,
    )

    # 요약 카드
    success_cnt = sum(1 for r in rows if r["상태"] == "success")
    fail_cnt = sum(1 for r in rows if r["상태"] == "failure")
    total = len(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("총 실행", f"{total}건")
    c2.metric("성공", f"{success_cnt}건")
    c3.metric("실패", f"{fail_cnt}건", delta=f"-{fail_cnt}" if fail_cnt > 0 else None,
              delta_color="inverse" if fail_cnt > 0 else "off")


# ═══════════════════════════════════════════
# 서브탭 2: VM 매매 로그
# ═══════════════════════════════════════════
def _render_trade_log():
    st.subheader("VM 매매 로그")

    if st.button("GitHub에서 동기화", key="ops_tl_sync"):
        _sync_from_github()
        st.toast("동기화 완료")

    entries = _load_json("trade_log.json")
    if not isinstance(entries, list) or not entries:
        st.info("VM 매매 로그가 없습니다. (trade_log.json 미동기화)")
        return

    # 필터
    modes = ["전체", "auto", "manual", "signal"]
    sel_mode = st.selectbox("모드 필터", modes, key="ops_tl_mode")

    rows = []
    for e in entries:
        mode = e.get("mode", "")
        if sel_mode != "전체" and mode != sel_mode:
            continue
        side = e.get("side", "")
        side_kr = {"BUY": "매수", "SELL": "매도", "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도"}.get(side, side)
        mode_kr = {"auto": "자동", "manual": "수동", "signal": "시그널"}.get(mode, mode)
        amount_str = e.get("amount", e.get("qty", ""))
        rows.append({
            "시간": e.get("time", ""),
            "모드": mode_kr,
            "코인": e.get("ticker", ""),
            "구분": side_kr,
            "전략": e.get("strategy", ""),
            "금액/수량": str(amount_str),
            "결과": e.get("result", ""),
            "상세": str(e.get("detail", ""))[:80],
        })

    if not rows:
        st.info("필터 조건에 해당하는 로그 없음")
        return

    df = pd.DataFrame(rows)
    st.success(f"{len(df)}건")

    def _color_side(val):
        if val in ("매수", "보충매수"):
            return "color: #e74c3c"
        elif val in ("매도", "보충매도"):
            return "color: #2980b9"
        return ""

    def _color_result(val):
        if val == "success":
            return "color: #27ae60"
        elif val == "error":
            return "color: #e74c3c"
        return ""

    st.dataframe(
        df.style.map(_color_side, subset=["구분"]).map(_color_result, subset=["결과"]),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════
# 서브탭 3: Git 동기화 내역
# ═══════════════════════════════════════════
def _render_git_sync():
    st.subheader("Git 동기화 내역")

    if st.button("최신 커밋 Pull", key="ops_git_pull"):
        _run_git(["pull", "--rebase", "origin", "master"])
        st.toast("Pull 완료")

    out = _run_git([
        "log", "--oneline", "--format=%H|%aI|%s", "-30",
    ])
    if not out:
        st.warning("git log를 가져올 수 없습니다.")
        return

    rows = []
    for line in out.strip().split("\n"):
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        hash_short = parts[0][:7]
        time_str = parts[1]
        msg = parts[2]

        try:
            dt = datetime.fromisoformat(time_str)
            kst_str = dt.astimezone(_KST).strftime("%m-%d %H:%M")
        except Exception:
            kst_str = time_str[:16]

        is_auto = msg.startswith("auto:")
        rows.append({
            "시간(KST)": kst_str,
            "유형": "자동" if is_auto else "수동",
            "메시지": msg,
            "해시": hash_short,
        })

    if not rows:
        st.info("커밋 내역 없음")
        return

    df = pd.DataFrame(rows)

    def _color_type(val):
        if val == "자동":
            return "color: #27ae60; font-weight: bold"
        return ""

    st.dataframe(
        df.style.map(_color_type, subset=["유형"]),
        use_container_width=True, hide_index=True,
    )

    # 자동 커밋 간격 분석
    auto_commits = [r for r in rows if r["유형"] == "자동"]
    if len(auto_commits) >= 2:
        st.caption(f"최근 자동 커밋 {len(auto_commits)}건 (총 {len(rows)}건 중)")
    else:
        st.caption(f"전체 {len(rows)}건")


# ═══════════════════════════════════════════
# 서브탭 4: 시스템 상태 요약
# ═══════════════════════════════════════════
def _render_system_status():
    st.subheader("시스템 상태 요약")

    if st.button("전체 동기화", key="ops_sys_sync"):
        _sync_from_github()
        st.toast("동기화 완료")

    # ── balance_cache ──
    st.markdown("#### 잔고 캐시")
    bal = _load_json("balance_cache.json")
    if isinstance(bal, dict):
        updated = bal.get("updated_at", "알 수 없음")
        balances = bal.get("balances", {})
        prices = bal.get("prices", {})

        c1, c2 = st.columns(2)
        c1.metric("갱신 시각", updated)

        total_val = balances.get("KRW", 0)
        for sym, qty in balances.items():
            if sym == "KRW":
                continue
            ticker = f"KRW-{sym}"
            price = prices.get(ticker, 0)
            total_val += qty * price
        c2.metric("총 자산", f"{total_val:,.0f}원")

        bal_rows = []
        for sym, qty in balances.items():
            if sym == "KRW":
                bal_rows.append({"자산": "KRW", "수량": f"{qty:,.0f}", "평가액": f"{qty:,.0f}원"})
            else:
                ticker = f"KRW-{sym}"
                price = prices.get(ticker, 0)
                val = qty * price
                bal_rows.append({"자산": sym, "수량": f"{qty:.8g}", "평가액": f"{val:,.0f}원"})
        st.dataframe(pd.DataFrame(bal_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("balance_cache.json 없음")

    # ── signal_state ──
    st.markdown("#### 시그널 상태")
    sig = _load_json("signal_state.json")
    if isinstance(sig, dict):
        sig_rows = []
        for key, val in sig.items():
            if key.startswith("__"):
                continue
            sig_rows.append({"전략키": key, "포지션": val})
        if sig_rows:
            df_sig = pd.DataFrame(sig_rows)

            def _color_pos(val):
                if val == "BUY":
                    return "color: #e74c3c; font-weight: bold"
                elif val == "SELL":
                    return "color: #2980b9; font-weight: bold"
                return ""

            st.dataframe(
                df_sig.style.map(_color_pos, subset=["포지션"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("시그널 상태 없음")
    else:
        st.warning("signal_state.json 없음")

    # ── trade_log 최근 5건 ──
    st.markdown("#### 최근 매매 로그 (5건)")
    tl = _load_json("trade_log.json")
    if isinstance(tl, list) and tl:
        recent = tl[:5]
        for e in recent:
            side = e.get("side", "")
            side_kr = {"BUY": "매수", "SELL": "매도", "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도"}.get(side, side)
            mode_kr = {"auto": "자동", "manual": "수동", "signal": "시그널"}.get(e.get("mode", ""), "")
            result = e.get("result", "")
            icon = "✅" if result == "success" else ("❌" if result == "error" else "ℹ️")
            st.caption(f"{icon} [{e.get('time', '')}] {mode_kr} {e.get('ticker', '')} {side_kr} | {result}")
    else:
        st.info("매매 로그 없음")


# ═══════════════════════════════════════════
# 서브탭 5: 실행 검증
# ═══════════════════════════════════════════
def _render_verification():
    st.subheader("실행 검증")
    st.caption("특정 날짜의 자동매매가 정상 실행되었는지 4단계로 검증합니다.")

    now_kst = datetime.now(_KST)
    check_date = st.date_input("검증 날짜", value=now_kst.date(), key="ops_verify_date")
    date_str = check_date.strftime("%Y-%m-%d")

    if st.button("검증 실행", key="ops_verify_run"):
        results = []

        # 1단계: GH Actions run 존재 확인
        st.markdown("##### 1단계: GH Actions 실행 확인")
        out = _run_gh([
            "run", "list", "--workflow=coin_trade.yml", "--limit", "30",
            "--json", "databaseId,status,conclusion,event,updatedAt",
        ])
        gh_found = False
        if out:
            try:
                runs = json.loads(out)
                for r in runs:
                    updated = r.get("updatedAt", "")
                    if date_str in updated:
                        gh_found = True
                        break
            except Exception:
                pass
        if gh_found:
            st.success(f"✅ GH Actions에서 {date_str} 실행 기록 발견")
        else:
            st.error(f"❌ GH Actions에서 {date_str} 실행 기록 없음")
        results.append(gh_found)

        # 2단계: trade_log.json에 해당 날짜 기록 확인
        st.markdown("##### 2단계: VM 매매 로그 확인")
        tl = _load_json("trade_log.json")
        tl_found = False
        if isinstance(tl, list):
            for e in tl:
                if date_str in e.get("time", ""):
                    tl_found = True
                    break
        if tl_found:
            st.success(f"✅ trade_log.json에 {date_str} 기록 발견")
        else:
            st.warning(f"⚠️ trade_log.json에 {date_str} 기록 없음 (HOLD시 기록 안될 수 있음)")
        results.append(tl_found)

        # 3단계: git log에 auto commit 확인
        st.markdown("##### 3단계: Git 자동 커밋 확인")
        git_out = _run_git(["log", "--oneline", "--after", f"{date_str}T00:00:00", "--before", f"{date_str}T23:59:59", "--grep=auto:"])
        git_found = bool(git_out and git_out.strip())
        if git_found:
            st.success(f"✅ {date_str} 자동 커밋 발견")
            st.code(git_out[:200])
        else:
            st.warning(f"⚠️ {date_str} 자동 커밋 없음")
        results.append(git_found)

        # 4단계: balance_cache 갱신 시각 확인
        st.markdown("##### 4단계: 잔고 캐시 갱신 확인")
        bal = _load_json("balance_cache.json")
        bal_ok = False
        if isinstance(bal, dict):
            updated = bal.get("updated_at", "")
            if date_str in updated:
                bal_ok = True
                st.success(f"✅ balance_cache 갱신 확인 ({updated})")
            else:
                st.warning(f"⚠️ balance_cache 갱신 시각: {updated} ({date_str} 아님)")
        else:
            st.error("❌ balance_cache.json 없음")
        results.append(bal_ok)

        # 종합 결과
        st.divider()
        passed = sum(results)
        if passed == 4:
            st.success(f"🎉 검증 완료: {passed}/4 통과 — 정상 실행 확인")
        elif passed >= 2:
            st.warning(f"⚠️ 검증 결과: {passed}/4 통과 — 부분 확인 (HOLD 시 로그 미기록 가능)")
        else:
            st.error(f"❌ 검증 실패: {passed}/4 통과 — 실행 확인 불가")


# ═══════════════════════════════════════════
# 메인 렌더 함수
# ═══════════════════════════════════════════
def render_ops_log_tab():
    """운영 로그 탭 메인 렌더 함수."""
    st.header("운영 로그")
    st.caption("GH Actions 실행, VM 매매, Git 동기화 내역을 통합 조회합니다.")

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "🔄 GH Actions",
        "📋 VM 매매 로그",
        "🔧 Git 동기화",
        "📊 시스템 상태",
        "🔍 실행 검증",
    ])

    with sub1:
        _render_gh_actions()
    with sub2:
        _render_trade_log()
    with sub3:
        _render_git_sync()
    with sub4:
        _render_system_status()
    with sub5:
        _render_verification()
