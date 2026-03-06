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


def _run_git(args: list[str], timeout: int = 30) -> str | None:
    """git 명령 실행 후 stdout 반환. 실패 시 None."""
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


def _sync_from_github() -> list[str]:
    """GitHub에서 최신 상태 파일 동기화. 성공한 파일 목록 반환."""
    fetch_out = _run_git(["fetch", "origin", "--quiet"], timeout=15)
    if fetch_out is None:
        # fetch 자체가 실패할 수도 있으므로 재시도
        _run_git(["fetch", "origin"], timeout=20)
    files = ["trade_log.json", "balance_cache.json", "signal_state.json", "account_cache.json"]
    synced = []
    for f in files:
        result = _run_git(["checkout", "origin/master", "--", f])
        if result is not None:
            synced.append(f)
    return synced


# ═══════════════════════════════════════════
# 서브탭 1: GH Actions 실행 내역
# ═══════════════════════════════════════════
def _get_failure_note(run_id) -> str:
    """실패한 run의 실패 step 이름을 한글로 조회."""
    out = _run_gh([
        "run", "view", str(run_id),
        "--json", "jobs",
    ], timeout=10)
    if not out:
        return "로그 조회 실패"
    try:
        data = json.loads(out)
        jobs = data.get("jobs", [])
        for job in jobs:
            if job.get("conclusion") != "failure":
                continue
            steps = job.get("steps", [])
            for step in steps:
                if step.get("conclusion") == "failure":
                    name = step.get("name", "?")
                    # 한글 변환
                    if "SSH" in name or "Cloud VM" in name:
                        return f"VM SSH 연결 실패 또는 타임아웃 ({name})"
                    if "Sync" in name or "push" in name.lower():
                        return f"동기화/Push 단계 실패 ({name})"
                    if "Checkout" in name:
                        return f"코드 체크아웃 실패 ({name})"
                    return f"실패 단계: {name}"
            return f"실패 job: {job.get('name', '?')}"
    except Exception:
        pass
    return "원인 불명"


def _analyze_success_run(run_kst_str: str, workflow: str) -> str:
    """
    성공한 run에서 실제 매매가 없었을 때 이유를 분석.
    trade_log와 signal_state를 교차 확인.
    """
    if "Coin" not in workflow:
        return ""  # 코인 외 워크플로우는 분석 생략

    # trade_log에서 해당 시간대 기록 존재 확인
    tl = _load_json("trade_log.json")
    # run_kst_str: "03-06 21:39" 형태 → 날짜+시 추출
    date_part = run_kst_str[:5]   # "03-06"
    hour_part = run_kst_str[6:8]  # "21"

    has_trade = False
    trade_summary = ""
    if isinstance(tl, list):
        for e in tl:
            t = e.get("time", "")
            # "2026-03-06 09:02:19 KST" 형태
            if f"-{date_part}" in t:
                t_hour = t[11:13] if len(t) > 13 else ""
                # ±1시간 범위로 매칭 (GH Actions 시작 ~ 실행 완료)
                try:
                    rh = int(hour_part)
                    th = int(t_hour)
                    if abs(rh - th) <= 1 or abs(rh - th) >= 23:
                        has_trade = True
                        side = {"BUY": "매수", "SELL": "매도",
                                "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도"
                                }.get(e.get("side", ""), e.get("side", ""))
                        result = "성공" if e.get("result") == "success" else "실패"
                        trade_summary += f"{side}({result}) "
                except (ValueError, TypeError):
                    pass

    if has_trade:
        return f"매매 실행됨: {trade_summary.strip()}"

    # 매매 없음 → signal_state로 원인 분석
    sig = _load_json("signal_state.json")
    if not isinstance(sig, dict):
        return "매매 없음 (signal_state 미확인)"

    reasons = []
    for key, val in sig.items():
        if key.startswith("__"):
            continue
        if val == "BUY":
            reasons.append(f"{key}: BUY 유지 → 전환 없어 매매 불필요")
        elif val == "SELL":
            reasons.append(f"{key}: SELL 유지 → 전환 없어 매매 불필요")
        elif val == "HOLD":
            reasons.append(f"{key}: HOLD → 돈치안 채널 내부, 매매 없음")

    if reasons:
        return "매매 없음 — " + "; ".join(reasons)
    return "매매 없음 (시그널 전환 없음)"


# 수동 기록: 특정 run ID에 대한 실패 원인 + 수정 내용 (한글)
_MANUAL_NOTES: dict[int, str] = {
    22763810376: (
        "[실패 원인] VM SSH 타임아웃 — coin_trade.yml의 schedule 조건에 "
        "'github.event_name == schedule' 누락으로 trade job이 아예 실행되지 않음.\n"
        "[수정] schedule 조건 추가 (커밋 5779277)"
    ),
    22756233176: (
        "[실패 원인] VM SSH 타임아웃 — 보충매수 로직에서 Donchian HOLD 상태를 "
        "매수 대상으로 인식하지 못해 무한 대기.\n"
        "[수정] HOLD 해소 로직 추가: prev state가 BUY면 보충매수 실행 (커밋 e5871ea)"
    ),
}


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

    # 실패 run의 원인 조회 (캐싱)
    notes_cache = st.session_state.get("_ops_run_notes", {})
    for r in all_runs:
        run_id = r.get("databaseId")
        if not run_id:
            continue
        conclusion = r.get("conclusion", "")
        if run_id in _MANUAL_NOTES:
            notes_cache[run_id] = _MANUAL_NOTES[run_id]
        elif run_id not in notes_cache:
            if conclusion == "failure":
                notes_cache[run_id] = _get_failure_note(run_id)
    st.session_state["_ops_run_notes"] = notes_cache

    rows = []
    for r in all_runs:
        updated = r.get("updatedAt", "")
        kst_str = ""
        if updated:
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                kst_str = dt.astimezone(_KST).strftime("%m-%d %H:%M")
            except Exception:
                kst_str = updated[:16]

        status = r.get("status", "")
        conclusion = r.get("conclusion", "")
        if status == "completed":
            state_str = conclusion or "completed"
        else:
            state_str = status

        run_id = r.get("databaseId", "")
        wf_name = r.get("workflow", "")
        note = notes_cache.get(run_id, "")

        # 성공 run: 매매 여부 분석
        if state_str == "success" and not note and kst_str:
            note = _analyze_success_run(kst_str, wf_name)

        rows.append({
            "시간(KST)": kst_str,
            "워크플로우": wf_name,
            "이벤트": r.get("event", ""),
            "상태": state_str,
            "ID": run_id,
            "비고": note,
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

    # 상세 내역 expander
    noted_rows = [r for r in rows if r["비고"]]
    fail_rows = [r for r in rows if r["상태"] == "failure"]
    no_trade_rows = [r for r in rows if r["상태"] == "success" and "매매 없음" in r.get("비고", "")]

    if fail_rows:
        with st.expander(f"실패 {len(fail_rows)}건 상세", expanded=True):
            for fr in fail_rows:
                st.markdown(f"**{fr['시간(KST)']} | {fr['워크플로우']}** (ID: {fr['ID']})")
                note = fr.get("비고", "")
                if note:
                    # 수정 내용이 포함된 경우 분리 표시
                    if "[수정]" in note:
                        parts = note.split("[수정]")
                        st.error(parts[0].strip())
                        st.success(f"[수정] {parts[1].strip()}")
                    else:
                        st.error(note)
                else:
                    st.warning("원인 미확인 — 로그 직접 확인 필요")

    if no_trade_rows:
        with st.expander(f"성공했지만 매매 없음 {len(no_trade_rows)}건", expanded=False):
            for nr in no_trade_rows:
                st.markdown(f"**{nr['시간(KST)']} | {nr['워크플로우']}**")
                note = nr.get("비고", "")
                # 전략별 상태를 줄바꿈으로 표시
                if ";" in note:
                    parts = note.split("—", 1)
                    st.caption(parts[0].strip())
                    if len(parts) > 1:
                        for reason in parts[1].split(";"):
                            reason = reason.strip()
                            if reason:
                                st.caption(f"  • {reason}")
                else:
                    st.caption(note)

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
        synced = _sync_from_github()
        if synced:
            st.toast(f"동기화 완료: {', '.join(synced)}")
        else:
            st.toast("동기화 실패 — fetch 또는 checkout 오류")

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

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("최신 커밋 Pull", key="ops_git_pull"):
            # stash → pull → stash pop 으로 로컬 변경 보호
            _run_git(["stash", "--include-untracked"])
            pull_out = _run_git(["pull", "--rebase", "origin", "master"], timeout=20)
            _run_git(["stash", "pop"])
            if pull_out is not None:
                st.toast("Pull 완료")
            else:
                st.toast("Pull 실패 — 네트워크 또는 충돌 확인 필요")
    with col_b:
        if st.button("상태 파일만 동기화", key="ops_git_sync_files"):
            synced = _sync_from_github()
            if synced:
                st.toast(f"동기화 완료: {', '.join(synced)}")
            else:
                st.toast("동기화 실패")

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
        synced = _sync_from_github()
        if synced:
            st.toast(f"동기화 완료: {', '.join(synced)}")
        else:
            st.toast("동기화 실패 — git fetch 확인 필요")

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
    st.caption("특정 날짜의 자동매매가 정상 실행되었는지 4단계로 검증합니다. 각 단계별 상세 내역을 출력합니다.")

    now_kst = datetime.now(_KST)
    check_date = st.date_input("검증 날짜", value=now_kst.date(), key="ops_verify_date")
    date_str = check_date.strftime("%Y-%m-%d")

    v_col1, v_col2 = st.columns(2)
    with v_col2:
        if st.button("GitHub 동기화 후 검증", key="ops_verify_sync"):
            synced = _sync_from_github()
            st.toast(f"동기화: {', '.join(synced)}" if synced else "동기화 실패")

    with v_col1:
        run_verify = st.button("검증 실행", key="ops_verify_run")

    if run_verify:
        results = []

        # ── 1단계: GH Actions 실행 확인 ──
        st.markdown("##### 1단계: GH Actions 실행 확인 🔗")
        gh_day_runs = []
        for wf_file, wf_name in _WORKFLOWS:
            out = _run_gh([
                "run", "list", f"--workflow={wf_file}", "--limit", "50",
                "--json", "databaseId,status,conclusion,event,updatedAt,createdAt,displayTitle",
            ])
            if not out:
                continue
            try:
                runs = json.loads(out)
                for r in runs:
                    created = r.get("createdAt", r.get("updatedAt", ""))
                    if date_str in created:
                        try:
                            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            kst_str = dt.astimezone(_KST).strftime("%H:%M:%S")
                        except Exception:
                            kst_str = created[11:19]
                        updated_raw = r.get("updatedAt", "")
                        try:
                            dt_u = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
                            dur = dt_u - dt
                            dur_str = f"{int(dur.total_seconds())}s"
                        except Exception:
                            dur_str = "-"
                        status = r.get("status", "")
                        conclusion = r.get("conclusion", "")
                        state_str = conclusion if status == "completed" else status
                        gh_day_runs.append({
                            "시작(KST)": kst_str,
                            "워크플로우": wf_name,
                            "이벤트": r.get("event", ""),
                            "상태": state_str,
                            "소요": dur_str,
                            "ID": r.get("databaseId", ""),
                            "제목": (r.get("displayTitle", "") or "")[:40],
                        })
            except Exception:
                pass

        gh_found = len(gh_day_runs) > 0
        if gh_found:
            st.success(f"✅ GH Actions에서 {date_str} 실행 기록 {len(gh_day_runs)}건 발견")
            df_gh = pd.DataFrame(gh_day_runs)

            def _color_gh(val):
                if val == "success":
                    return "color: #27ae60; font-weight: bold"
                elif val == "failure":
                    return "color: #e74c3c; font-weight: bold"
                elif val in ("in_progress", "queued"):
                    return "color: #f39c12; font-weight: bold"
                return ""

            st.dataframe(
                df_gh.style.map(_color_gh, subset=["상태"]),
                use_container_width=True, hide_index=True,
            )
            # 요약
            success_cnt = sum(1 for r in gh_day_runs if r["상태"] == "success")
            fail_cnt = sum(1 for r in gh_day_runs if r["상태"] == "failure")
            if fail_cnt > 0:
                st.error(f"실패 {fail_cnt}건 / 성공 {success_cnt}건 / 총 {len(gh_day_runs)}건")
            else:
                st.caption(f"성공 {success_cnt}건 / 총 {len(gh_day_runs)}건")
        else:
            st.error(f"❌ GH Actions에서 {date_str} 실행 기록 없음")
        results.append(gh_found)

        # ── 2단계: VM 매매 로그 확인 ──
        st.markdown("##### 2단계: VM 매매 로그 확인")
        tl = _load_json("trade_log.json")
        tl_entries = []
        if isinstance(tl, list):
            for e in tl:
                if date_str in e.get("time", ""):
                    side = e.get("side", "")
                    side_kr = {"BUY": "매수", "SELL": "매도", "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도"}.get(side, side)
                    mode_kr = {"auto": "자동", "manual": "수동", "signal": "시그널"}.get(e.get("mode", ""), e.get("mode", ""))
                    result = e.get("result", "")
                    tl_entries.append({
                        "시간": e.get("time", "")[11:19] if len(e.get("time", "")) > 11 else e.get("time", ""),
                        "모드": mode_kr,
                        "코인": e.get("ticker", ""),
                        "구분": side_kr,
                        "전략": e.get("strategy", ""),
                        "금액/수량": str(e.get("amount", e.get("qty", ""))),
                        "결과": result,
                        "상세": str(e.get("detail", ""))[:60],
                    })
        tl_found = len(tl_entries) > 0
        if tl_found:
            st.success(f"✅ trade_log.json에 {date_str} 기록 {len(tl_entries)}건 발견")
            df_tl = pd.DataFrame(tl_entries)

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
                df_tl.style.map(_color_side, subset=["구분"]).map(_color_result, subset=["결과"]),
                use_container_width=True, hide_index=True,
            )
            # 매수/매도/보충 건수 요약
            buy_cnt = sum(1 for e in tl_entries if e["구분"] in ("매수", "보충매수"))
            sell_cnt = sum(1 for e in tl_entries if e["구분"] in ("매도", "보충매도"))
            err_cnt = sum(1 for e in tl_entries if e["결과"] == "error")
            summary_parts = [f"매수 {buy_cnt}건", f"매도 {sell_cnt}건"]
            if err_cnt > 0:
                summary_parts.append(f"오류 {err_cnt}건")
            st.caption(" | ".join(summary_parts))
        else:
            st.warning(f"⚠️ trade_log.json에 {date_str} 기록 없음 (HOLD시 매매 기록 안될 수 있음)")
        results.append(tl_found)

        # ── 3단계: Git 자동 커밋 확인 ──
        st.markdown("##### 3단계: Git 자동 커밋 확인")
        # 해당 날짜의 모든 커밋 조회 (auto 커밋 + 일반 커밋)
        git_all = _run_git([
            "log", "--format=%H|%aI|%s",
            "--after", f"{date_str}T00:00:00",
            "--before", f"{date_str}T23:59:59",
        ])
        git_entries = []
        git_auto_count = 0
        if git_all and git_all.strip():
            for line in git_all.strip().split("\n"):
                parts = line.split("|", 2)
                if len(parts) < 3:
                    continue
                hash_short = parts[0][:7]
                time_str = parts[1]
                msg = parts[2]
                try:
                    dt = datetime.fromisoformat(time_str)
                    kst_str = dt.astimezone(_KST).strftime("%H:%M:%S")
                except Exception:
                    kst_str = time_str[11:19]
                is_auto = msg.startswith("auto:")
                if is_auto:
                    git_auto_count += 1
                git_entries.append({
                    "시간(KST)": kst_str,
                    "유형": "자동" if is_auto else "수동",
                    "메시지": msg[:60],
                    "해시": hash_short,
                })

        git_found = git_auto_count > 0
        if git_entries:
            if git_found:
                st.success(f"✅ {date_str} 자동 커밋 {git_auto_count}건 발견 (전체 {len(git_entries)}건)")
            else:
                st.warning(f"⚠️ {date_str} 자동 커밋 없음 (수동 커밋 {len(git_entries)}건 있음)")
            df_git = pd.DataFrame(git_entries)

            def _color_type(val):
                if val == "자동":
                    return "color: #27ae60; font-weight: bold"
                return "color: #888"

            st.dataframe(
                df_git.style.map(_color_type, subset=["유형"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.warning(f"⚠️ {date_str} 커밋 기록 없음")
        results.append(git_found)

        # ── 4단계: 잔고 캐시 갱신 확인 ──
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
            # 잔고 상세 출력
            balances = bal.get("balances", {})
            prices = bal.get("prices", {})
            if balances:
                bal_rows = []
                total_val = 0.0
                for sym, qty in balances.items():
                    if sym == "KRW":
                        bal_rows.append({"자산": "KRW", "수량": f"{qty:,.0f}", "평가액": f"{qty:,.0f}원"})
                        total_val += qty
                    else:
                        ticker = f"KRW-{sym}"
                        price = prices.get(ticker, 0)
                        val = qty * price
                        bal_rows.append({"자산": sym, "수량": f"{qty:.8g}", "평가액": f"{val:,.0f}원"})
                        total_val += val
                st.dataframe(pd.DataFrame(bal_rows), use_container_width=True, hide_index=True)
                st.caption(f"총 자산: {total_val:,.0f}원 (갱신: {updated})")
        else:
            st.error("❌ balance_cache.json 없음")
        results.append(bal_ok)

        # ── 5단계: signal_state 확인 ──
        st.markdown("##### 5단계: 시그널 상태 확인")
        sig = _load_json("signal_state.json")
        sig_ok = False
        if isinstance(sig, dict):
            sig_updated = sig.get("__updated_at", "")
            if date_str in sig_updated:
                sig_ok = True
                st.success(f"✅ signal_state 갱신 확인 ({sig_updated})")
            elif sig_updated:
                st.warning(f"⚠️ signal_state 갱신 시각: {sig_updated} ({date_str} 아님)")
            else:
                st.info("signal_state에 갱신 시각 없음")
            # 상태 상세 출력
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
            st.warning("signal_state.json 없음")
        results.append(sig_ok)

        # ── 종합 결과 ──
        st.divider()
        passed = sum(results)
        total_steps = len(results)
        if passed == total_steps:
            st.success(f"🎉 검증 완료: {passed}/{total_steps} 통과 — 정상 실행 확인")
        elif passed >= 3:
            st.warning(f"⚠️ 검증 결과: {passed}/{total_steps} 통과 — 대부분 정상 (HOLD 시 일부 미기록 가능)")
        elif passed >= 1:
            st.warning(f"⚠️ 검증 결과: {passed}/{total_steps} 통과 — 부분 확인")
        else:
            st.error(f"❌ 검증 실패: {passed}/{total_steps} 통과 — 실행 확인 불가")

        # 실패 항목 요약
        step_names = ["GH Actions", "VM 매매 로그", "Git 자동 커밋", "잔고 캐시", "시그널 상태"]
        failed_steps = [step_names[i] for i, r in enumerate(results) if not r]
        if failed_steps:
            st.caption(f"미통과 항목: {', '.join(failed_steps)}")


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
