"""코인 트레이딩 UI 공유 유틸리티."""
import json
import os
import time
import subprocess
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOP_20_TICKERS = [
    "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
    "KRW-ADA", "KRW-SHIB", "KRW-TRX", "KRW-AVAX", "KRW-LINK",
    "KRW-BCH", "KRW-DOT", "KRW-NEAR", "KRW-POL", "KRW-ETC",
    "KRW-XLM", "KRW-STX", "KRW-HBAR", "KRW-EOS", "KRW-SAND",
]


def ttl_cache(key, fn, ttl=5):
    """세션 기반 TTL 캐시. ttl초 이내 재호출시 캐시 반환."""
    now = time.time()
    ck, tk = f"__c_{key}", f"__t_{key}"
    if ck in st.session_state and (now - st.session_state.get(tk, 0)) < ttl:
        return st.session_state[ck]
    val = fn()
    st.session_state[ck] = val
    st.session_state[tk] = now
    return val


def clear_cache(*keys):
    """거래 후 캐시 무효화"""
    for k in keys:
        st.session_state.pop(f"__c_{k}", None)
        st.session_state.pop(f"__t_{k}", None)


def _auto_sync_from_github():
    """자동 동기화: 2분마다 GitHub에서 캐시 파일 pull."""
    _key = "__last_auto_sync"
    now = time.time()
    if now - st.session_state.get(_key, 0) > 120:
        sync_account_cache_from_github()
        st.session_state[_key] = now


def load_balance_cache():
    """통합 캐시 로드: balance_cache(잔고/가격) + account_cache(주문/체결).

    - 잔고·가격·updated_at → balance_cache.json (VM 5분 push, 최신)
    - orders·pending_orders·deposits·withdraws → account_cache.json (GH Actions)
    두 파일을 합쳐 단일 dict 반환. 잔고는 더 최신 소스 우선.
    """
    _auto_sync_from_github()
    bal = {}
    acct = {}
    try:
        bf = os.path.join(PROJECT_ROOT, "balance_cache.json")
        if os.path.exists(bf):
            with open(bf, "r", encoding="utf-8") as f:
                bal = json.load(f) or {}
    except Exception:
        pass
    try:
        af = os.path.join(PROJECT_ROOT, "account_cache.json")
        if os.path.exists(af):
            with open(af, "r", encoding="utf-8") as f:
                acct = json.load(f) or {}
    except Exception:
        pass

    if not bal.get("balances"):
        bal = acct  # balance_cache 없으면 account_cache 폴백

    # account_cache의 주문/체결 데이터를 병합
    for key in ("orders", "pending_orders", "deposits", "withdraws"):
        if key in acct and key not in bal:
            bal[key] = acct[key]

    return bal


def load_account_cache():
    """하위호환 래퍼 — load_balance_cache()로 통합."""
    return load_balance_cache()


def load_signal_state():
    """최근 전략 포지션 상태(signal_state.json) 로드. 2분마다 GitHub 자동 동기화."""
    _auto_sync_from_github()
    try:
        state_file = os.path.join(PROJECT_ROOT, "signal_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                return state
    except Exception:
        pass
    return {}


def trigger_gh_workflow(job_name: str, extra_inputs: dict | None = None) -> tuple[bool, str]:
    """GitHub Actions workflow를 트리거하고 결과 반환."""
    try:
        _wf_map = {
            "trade": "coin_trade.yml", "manual_order": "coin_trade.yml",
            "account_sync": "coin_trade.yml",
            "kiwoom_gold": "gold_trade.yml",
            "kis_isa": "isa_trade.yml", "kis_pension": "pension_trade.yml",
            "health_check": "monitoring.yml", "daily_status": "monitoring.yml",
            "vm_once_add": "monitoring.yml", "vm_once_show": "monitoring.yml",
        }
        _wf = _wf_map.get(job_name, "coin_trade.yml")
        cmd = ["gh", "workflow", "run", _wf, "-f", f"run_job={job_name}"]
        for k, v in (extra_inputs or {}).items():
            cmd.extend(["-f", f"{k}={v}"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True, f"{job_name} 실행 요청 완료"
        return False, f"실행 실패: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "gh CLI가 설치되어 있지 않습니다. GitHub CLI를 설치해주세요."
    except Exception as e:
        return False, f"오류: {e}"


def sync_account_cache_from_github():
    """GitHub에서 캐시 파일 pull + 로컬 브랜치 fast-forward."""
    try:
        r1 = subprocess.run(
            ["git", "fetch", "origin", "--quiet"],
            cwd=PROJECT_ROOT, capture_output=True, timeout=15,
        )
        if r1.returncode != 0:
            print(f"[sync] git fetch failed: {r1.stderr.decode()[:200]}")
            return False
        # 각 파일을 개별 checkout — 일부 파일이 없어도 나머지는 정상 동기화
        _files = [
            "balance_cache.json", "signal_state.json",
            "trade_log.json", "account_cache.json",
            "logs/vm_scheduler_state.json",
        ]
        ok_count = 0
        for f in _files:
            r = subprocess.run(
                ["git", "checkout", "origin/master", "--", f],
                cwd=PROJECT_ROOT, capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                ok_count += 1
        # 로컬 브랜치도 fast-forward (코드 업데이트)
        subprocess.run(
            ["git", "merge", "--ff-only", "origin/master"],
            cwd=PROJECT_ROOT, capture_output=True, timeout=15,
        )
        if ok_count == 0:
            print(f"[sync] 모든 파일 checkout 실패")
            return False
        return True
    except Exception as e:
        print(f"[sync] exception: {e}")
        return False


def trigger_and_wait_gh(job_name: str, status_placeholder=None, extra_inputs: dict | None = None) -> tuple[bool, str]:
    """workflow 트리거 → 완료 대기 → 캐시 pull. 한 번에 처리."""
    _wf_map = {
        "trade": "coin_trade.yml", "manual_order": "coin_trade.yml",
        "account_sync": "coin_trade.yml",
        "kiwoom_gold": "gold_trade.yml",
        "kis_isa": "isa_trade.yml", "kis_pension": "pension_trade.yml",
        "health_check": "monitoring.yml", "daily_status": "monitoring.yml",
        "vm_once_add": "monitoring.yml", "vm_once_show": "monitoring.yml",
    }
    wf_file = _wf_map.get(job_name, "coin_trade.yml")

    # 0) 트리거 전 최신 run ID 기록
    prev_run_id = None
    try:
        r = subprocess.run(
            ["gh", "run", "list", f"--workflow={wf_file}", "--limit", "1",
             "--json", "databaseId"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            _runs = json.loads(r.stdout)
            if _runs:
                prev_run_id = _runs[0].get("databaseId")
    except Exception:
        pass

    # 1) 트리거
    if status_placeholder:
        status_placeholder.text("워크플로우 트리거 중...")
    ok, msg = trigger_gh_workflow(job_name, extra_inputs=extra_inputs)
    if not ok:
        return False, msg

    # 2) 새 run ID 찾기 (최대 20초 대기)
    run_id = None
    for _ in range(10):
        time.sleep(2)
        try:
            r = subprocess.run(
                ["gh", "run", "list", f"--workflow={wf_file}", "--limit", "5",
                 "--json", "databaseId,status"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                for run in json.loads(r.stdout):
                    rid = run.get("databaseId")
                    if rid and rid != prev_run_id:
                        run_id = rid
                        break
                if run_id:
                    break
        except Exception:
            pass
    if not run_id:
        sync_account_cache_from_github()
        return False, "실행 ID를 찾을 수 없습니다."

    # 3) 완료 대기 (최대 5분)
    if status_placeholder:
        status_placeholder.text(f"실행 중... (run #{run_id})")
    for i in range(60):
        time.sleep(5)
        if status_placeholder:
            status_placeholder.text(f"실행 중... ({(i+1)*5}초 경과)")
        try:
            r = subprocess.run(
                ["gh", "run", "view", str(run_id), "--json", "status,conclusion"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                info = json.loads(r.stdout)
                if info.get("status") == "completed":
                    success = info.get("conclusion") == "success"
                    if status_placeholder:
                        status_placeholder.text("결과 가져오는 중...")
                    sync_account_cache_from_github()
                    return success, f"{'성공' if success else '실패'} ({(i+1)*5}초)"
        except Exception:
            pass

    sync_account_cache_from_github()
    return False, "타임아웃 (300초) - 캐시는 업데이트 시도했습니다."


def normalize_coin_interval(interval):
    """github_action_trade.py와 동일한 interval 정규화."""
    iv = str(interval or "day").strip().lower()
    if iv in {"1d", "d", "day", "daily"}:
        return "day"
    if iv in {"4h", "240", "240m", "minute240"}:
        return "minute240"
    if iv in {"1h", "60", "60m", "minute60"}:
        return "minute60"
    return iv


def make_signal_key(item):
    """github_action_trade.py와 동일한 시그널 키 생성."""
    ticker = f"{item.get('market', 'KRW')}-{str(item.get('coin', '')).upper()}"
    strategy = item.get("strategy", "SMA")
    try:
        param = int(float(item.get("parameter", 20) or 20))
    except Exception:
        param = 20
    interval = normalize_coin_interval(item.get("interval", "day"))
    return f"{ticker}_{strategy}_{param}_{interval}"


def determine_signal(position_state, prev_state):
    """github_action_trade.py와 동일한 전환 감지 로직."""
    pos = str(position_state or "").upper()
    prev = str(prev_state or "").upper() or None
    if pos == "HOLD":
        return "HOLD"
    if prev is None:
        return pos
    if pos == prev:
        return "HOLD"
    return pos


def resolve_effective_state(position_state, prev_state, signal):
    """UI 표시용 최종 상태."""
    prev = str(prev_state or "").upper()
    if prev in {"BUY", "SELL"}:
        return prev
    if prev in {"HOLD"}:
        return "HOLD"
    return "UNKNOWN"


def state_to_position_label(state):
    s = str(state or "").upper()
    if s == "BUY":
        return "보유"
    if s == "HOLD":
        return "보유"
    if s == "SELL":
        return "현금"
    if s == "UNKNOWN":
        return "미확인"
    return "현금"


def get_signal_entry(signal_state, key):
    """signal_state에서 항목 읽기 (string/dict 호환).

    VM이 저장한 확장 포맷(dict) 또는 기존 포맷(string) 모두 지원.
    """
    raw = signal_state.get(key)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.upper() in {"BUY", "SELL", "HOLD"}:
        return {"state": raw.upper()}
    return {}


def get_signal_state_value(signal_state, key):
    """signal_state에서 상태값만 추출 (string/dict 호환)."""
    entry = get_signal_entry(signal_state, key)
    s = entry.get("state", "").upper()
    return s if s in {"BUY", "SELL", "HOLD"} else None
