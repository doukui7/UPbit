"""프로젝트 로직 탭 — 시스템 아키텍처, 매매 전략, 구축 가이드, 진행 내역."""
import json
import os
import time
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta


# ─── 진행 내역 파일 경로 ───
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROGRESS_FILE = os.path.join(_PROJECT_ROOT, "config", "project_progress.json")


def _load_progress() -> list:
    """프로젝트 진행 내역 로드."""
    try:
        if os.path.exists(_PROGRESS_FILE):
            with open(_PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return _default_progress()


def _save_progress(items: list):
    """프로젝트 진행 내역 저장."""
    os.makedirs(os.path.dirname(_PROGRESS_FILE), exist_ok=True)
    with open(_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _default_progress() -> list:
    """기본 진행 내역 템플릿."""
    return [
        {"phase": "1. 인프라 구축", "task": "GCP VM 인스턴스 생성 (e2-micro)", "status": "완료", "note": "Google Cloud VM (GCP) 사용 중"},
        {"phase": "1. 인프라 구축", "task": "고정 IP 할당 및 업비트 등록", "status": "완료", "note": "VM 고정 IP → 업비트 API 화이트리스트"},
        {"phase": "1. 인프라 구축", "task": "방화벽 포트 개방 (8501, 61208)", "status": "완료", "note": "Streamlit + Glances 포트"},
        {"phase": "2. 보안 설정", "task": "API 키 환경변수 분리 (.env)", "status": "완료", "note": "GitHub Secrets + VM .env"},
        {"phase": "2. 보안 설정", "task": "텔레그램 알림 봇 연동", "status": "완료", "note": "체결/에러 알림 발송 중"},
        {"phase": "2. 보안 설정", "task": "Deploy Key (SSH) 설정", "status": "진행중", "note": "VM SSH 키 생성 완료, GitHub 등록 필요"},
        {"phase": "3. 매매 엔진", "task": "SMA 이동평균선 전략 구현", "status": "완료", "note": "strategy/sma.py"},
        {"phase": "3. 매매 엔진", "task": "Donchian 채널 돌파 전략 구현", "status": "완료", "note": "strategy/donchian.py"},
        {"phase": "3. 매매 엔진", "task": "시그널 전환 감지 (signal_state.json)", "status": "완료", "note": "상태 변경 시에만 주문 실행"},
        {"phase": "3. 매매 엔진", "task": "자동매매 스케줄러 (Python 단일 로직)", "status": "완료", "note": "vm_scheduler.py 상주 + cron watchdog"},
        {"phase": "3. 매매 엔진", "task": "수동 주문 기능 (manual_order)", "status": "완료", "note": "전략 분석 없이 즉시 매수/매도"},
        {"phase": "3. 매매 엔진", "task": "주문 로그 기록 (trade_log.json)", "status": "완료", "note": "모든 주문 시도/결과 기록"},
        {"phase": "4. 대시보드", "task": "Streamlit 메인 UI 구성", "status": "완료", "note": "코인/골드/ISA/연금저축 4모드"},
        {"phase": "4. 대시보드", "task": "실시간 잔고/시세 표시", "status": "완료", "note": "10초 갱신 (balance_cache.json)"},
        {"phase": "4. 대시보드", "task": "전략 트리거 및 주문 로그 탭", "status": "완료", "note": "triggers.py — 규칙 + 로그 표시"},
        {"phase": "4. 대시보드", "task": "프로젝트 로직 탭", "status": "완료", "note": "시스템 문서 + 진행 내역"},
        {"phase": "5. 운영 관리", "task": "스케줄링 단일 로직 통합", "status": "완료", "note": "Python 스케줄러 = 유일 실행주체, cron = watchdog 전용"},
        {"phase": "5. 운영 관리", "task": "헬스체크 오탐 방지", "status": "완료", "note": "상태파일 + 로그 mtime 이중 확인"},
        {"phase": "5. 운영 관리", "task": "VS Code Remote-SSH 개발 환경", "status": "완료", "note": "로컬 → VM 원격 개발"},
        {"phase": "6. ISA/연금저축", "task": "위대리(WDR) ISA 전략", "status": "완료", "note": "매주 금요일 15:10 KST"},
        {"phase": "6. ISA/연금저축", "task": "LAA/DM/VAA/CDM 연금 전략", "status": "완료", "note": "매월 25~31일 리밸런싱"},
    ]


def _render_architecture():
    """시스템 아키텍처 섹션."""
    st.markdown("### 시스템 아키텍처")
    st.markdown("""
```
┌─────────────────────────────────────────────────────────────┐
│                    로컬 PC (개발/관리)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  VS Code     │  │  브라우저     │  │  텔레그램 앱     │   │
│  │  Remote-SSH  │  │  Streamlit   │  │  알림 수신       │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
└─────────┼─────────────────┼───────────────────┼─────────────┘
          │ SSH              │ HTTP:8501          │ Telegram API
          ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                 클라우드 VM (고정 IP)                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Python 스케줄러 (vm_scheduler.py) — 상주 프로세스     │  │
│  │    ├─ 자동매매: upbit (4H 간격, 정시)                  │  │
│  │    ├─ 헬스체크: health_check (4H 간격, +5분)           │  │
│  │    ├─ 일일현황: daily_status (평일 09:00)              │  │
│  │    ├─ 금현물:   kiwoom_gold (평일 15:05)               │  │
│  │    ├─ ISA:      kis_isa (금요일 15:10)                 │  │
│  │    ├─ 연금저축: kis_pension (25~31일 평일 15:20)       │  │
│  │    └─ 예약주문: vm_reserved_orders.json (1회성)        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  감시 체계 (Watchdog)                                  │  │
│  │    ├─ VM cron: 5분마다 vm_scheduler_manager.sh ensure  │  │
│  │    └─ GitHub Actions: 5분마다 vm_scheduler_ensure      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  GitHub Actions → SSH → vm_run_job.sh (수동 실행)      │  │
│  │    ├─ 수동주문: run_manual_order() (즉시 실행)         │  │
│  │    └─ 계좌동기화: account_sync                         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │  Streamlit   │  │  Glances     │                         │
│  │  대시보드    │  │  리소스 감시 │                         │
│  │  (:8501)     │  │  (:61208)    │                         │
│  └──────────────┘  └──────────────┘                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ API (고정 IP 인증)
                           ▼
          ┌────────────┬────────────┬──────────┐
          │ 업비트     │ KIS 증권   │ 키움증권 │
          │ 코인 매매  │ ISA/연금   │ 금현물   │
          └────────────┴────────────┴──────────┘
```
""")

    st.markdown("""
**핵심 구성 요소:**

| 구성 요소 | 역할 | 기술 |
|-----------|------|------|
| 클라우드 VM | 고정 IP로 API 통신 | Google Cloud Platform (GCP) |
| Python 스케줄러 | **유일한 스케줄 실행 주체** (상주 프로세스) | vm_scheduler.py |
| VM cron | 스케줄러 watchdog (5분마다 생존 확인) | crontab |
| GitHub Actions | 스케줄러 watchdog + 수동 실행 | cron + workflow_dispatch |
| vm_run_job.sh | 작업 실행 단위 (flock 중복방지 + 상태기록) | bash |
| Streamlit | 웹 대시보드 (설정/모니터링) | Python Streamlit |
| 텔레그램 | 체결/에러 실시간 알림 | Telegram Bot API |
""")


def _render_trading_strategy():
    """매매 전략 섹션."""
    st.markdown("### 매매 전략")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### SMA (단순이동평균선) 전략
- **매수**: 종가 > N일 이동평균선
- **매도**: 종가 < N일 이동평균선
- **기준 봉**: 마지막 완성 봉 (iloc[-2])
- **예시**: SMA(29) → 29일 이평선 기준

```python
# 핵심 로직
ma = df['close'].rolling(window=period).mean()
if close > ma:  # 매수 시그널
    signal = "BUY"
elif close < ma:  # 매도 시그널
    signal = "SELL"
```
""")

    with col2:
        st.markdown("""
#### Donchian 채널 돌파 전략
- **매수**: 종가 > N일 최고가 돌파
- **매도**: 종가 < M일 최저가 이탈
- **HOLD**: 돌파/이탈 없으면 유지
- **예시**: Donchian(115/105) → 매수 115일, 매도 105일

```python
# 핵심 로직
upper = df['high'].rolling(window=n).max()
lower = df['low'].rolling(window=m).min()
if close > upper:  signal = "BUY"
elif close < lower: signal = "SELL"
else: signal = "HOLD"
```
""")

    st.markdown("""
#### 매매 판단 규칙 (자동매매)

| 현재 상태 | 시그널 | 실행 내용 |
|-----------|--------|-----------|
| 코인 미보유 | 매수 시그널 | **매수** — 현금에서 비중 비례 배분 |
| 코인 미보유 | 매도/중립 | **대기** — 현금 보존 |
| 코인 보유 중 | 매도 시그널 | **매도** — 전략 비중 비례 매도 |
| 코인 보유 중 | 매수/중립 | **유지** — 계속 보유 |

**전환 감지**: `signal_state.json`에 이전 상태 저장 → 상태 전환 시에만 주문 실행
""")


def _render_execution_flow():
    """실행 흐름 섹션."""
    st.markdown("### 실행 흐름")

    st.markdown("""
#### 스케줄링 구조 (단일 로직)
```
┌─────────────────────────────────────────────────────┐
│  Python 스케줄러 (vm_scheduler.py) — 상주 프로세스   │
│    5초 간격 체크 → RULES 매칭 → vm_run_job.sh 실행   │
│    → vm_scheduler_state.json 상태 기록               │
└─────────────────────────┬───────────────────────────┘
                          │
    ┌─────────────────────┼──────────────────────┐
    ▼                     ▼                      ▼
  VM cron              GitHub Actions         수동 실행
  (watchdog)           (watchdog)             (dispatch)
  5분마다 ensure       5분마다 ensure         vm_run_job.sh
```

**핵심 원칙**: Python 스케줄러가 유일한 스케줄 실행 주체.
cron과 GitHub Actions는 스케줄러 생존 감시(watchdog)만 담당.

#### 자동매매 실행 순서
```
Python 스케줄러 (4시간 간격 정시)
    │ _run_mode("upbit")
    ▼
vm_run_job.sh upbit
    │ flock 중복방지 → python 실행 → 상태파일 기록
    ▼
github_action_trade.py (TRADING_MODE=upbit)
    │
    ├─ 1. portfolio.json 로드 (전략별 코인 목록)
    ├─ 2. signal_state.json 로드 (이전 상태)
    ├─ 3. 캔들 마감 대기 (4H 경계 시각 전 도착 시)
    ├─ 4. 전략 분석 (SMA/Donchian → BUY/SELL/HOLD)
    ├─ 5. 시그널 전환 감지 (이전 상태 ↔ 현재 시그널)
    ├─ 6. 매도 먼저 실행 (현금 확보)
    ├─ 7. 현금 비례 배분 매수
    ├─ 8. signal_state.json 업데이트
    ├─ 9. trade_log.json 기록 + GitHub push
    └─ 10. 텔레그램 알림 발송
```

#### 수동 주문 실행 순서
```
Streamlit UI (코인/방향/비율 선택)
    │
    ▼
GitHub Actions workflow_dispatch
    │  (run_job=manual_order, manual_order_params=JSON)
    ▼
SSH → VM → vm_run_job.sh manual_order
    │
    ├─ MANUAL_ORDER 환경변수 파싱
    ├─ 전략 분석 없이 즉시 실행
    ├─ BUY: 원화잔고 × (비율/100) × 0.999 → adaptive_buy
    ├─ SELL: 코인잔고 × (비율/100) → smart_sell
    ├─ trade_log.json 기록
    └─ 텔레그램 알림 발송
```
""")

    st.markdown("""
#### 실행 스케줄

| 모드 | 스케줄 | 실행 주체 |
|------|--------|-----------|
| 코인 자동매매 | KST 01/05/09/13/17/21시 정시 | Python 스케줄러 → vm_run_job.sh |
| 헬스체크 | KST 01/05/09/13/17/21시 +5분 | Python 스케줄러 → vm_run_job.sh |
| 일일 자산 현황 | 평일 09:00 KST | Python 스케줄러 → vm_run_job.sh |
| 골드 (금현물) | 평일 15:05 KST | Python 스케줄러 → vm_run_job.sh |
| ISA (위대리) | 매주 금요일 15:10 KST | Python 스케줄러 → vm_run_job.sh |
| 연금저축 | 매월 25~31일 평일 15:20 KST | Python 스케줄러 → vm_run_job.sh |
| 예약 주문 | vm_reserved_orders.json | Python 스케줄러 (1회성) |
| 코인 수동주문 | Streamlit에서 즉시 | GitHub Actions → vm_run_job.sh |
| 스케줄러 감시 | 5분마다 | VM cron + GitHub Actions |
""")


def _render_file_structure():
    """프로젝트 파일 구조 섹션."""
    st.markdown("### 프로젝트 파일 구조")
    st.markdown("""
```
upbit/
├── app.py                          # Streamlit 메인 앱
├── portfolio.json                  # 코인 포트폴리오 설정
├── signal_state.json               # 전략별 포지션 상태
├── trade_log.json                  # 주문 시도/결과 로그
│
├── scripts/
│   ├── github_action_trade.py      # 자동매매/수동주문 실행 스크립트
│   ├── vm_run_job.sh               # VM 작업 실행기 (flock + 상태 기록)
│   ├── vm_scheduler.py             # Python 스케줄러 (유일한 스케줄 주체)
│   ├── vm_scheduler_manager.sh     # 스케줄러 시작/중지/감시
│   └── vm_cron_manager.sh          # cron watchdog 설치/관리
│
├── src/
│   ├── constants.py                # 상수 정의
│   ├── strategy/
│   │   ├── sma.py                  # SMA 이동평균선 전략
│   │   ├── donchian.py             # Donchian 채널 돌파 전략
│   │   ├── widaeri.py              # 위대리 전략 (ISA)
│   │   ├── vaa.py                  # VAA 전략 (연금저축)
│   │   ├── cdm.py                  # CDM 전략 (연금저축)
│   │   ├── laa.py                  # LAA 전략 (연금저축)
│   │   └── dual_momentum.py        # 듀얼모멘텀 전략 (연금저축)
│   │
│   ├── engine/
│   │   ├── data_cache.py           # OHLCV 데이터 캐싱
│   │   └── data_manager.py         # 데이터 관리
│   │
│   ├── trading/
│   │   └── upbit_trader.py         # 업비트 주문 실행 (smart_buy/sell)
│   │
│   ├── ui/
│   │   ├── coin_mode.py            # 코인 트레이딩 UI
│   │   ├── gold_mode.py            # 골드 트레이딩 UI
│   │   ├── isa_mode.py             # ISA 트레이딩 UI
│   │   ├── pension_mode.py         # 연금저축 트레이딩 UI
│   │   ├── project_logic.py        # 프로젝트 로직 (이 파일)
│   │   └── components/
│   │       ├── triggers.py         # 전략 트리거 + 주문 로그
│   │       └── performance.py      # 성과 분석
│   │
│   └── utils/
│       ├── helpers.py              # 설정 load/save + 유틸
│       ├── kis.py                  # KIS 증권 API 트레이더
│       └── db_manager.py           # SQLite DB 관리
│
├── config/
│   ├── common.json                 # 공통 설정 (텔레그램 등)
│   ├── coin.json                   # 코인 모드 설정
│   ├── gold.json                   # 골드 모드 설정
│   ├── isa.json                    # ISA 모드 설정
│   └── pension.json                # 연금저축 모드 설정
│
└── .github/workflows/
    └── auto_trade.yml              # GitHub Actions 워크플로우
```
""")


def _render_security_guide():
    """보안 및 설정 가이드."""
    st.markdown("### 보안 및 설정 가이드")

    st.markdown("""
#### API 키 관리 원칙
1. **코드에 키 하드코딩 절대 금지** → `.env` 또는 GitHub Secrets 사용
2. **업비트 출금 권한 비활성화** → 주문/조회 권한만 활성화
3. **고정 IP 등록 필수** → VM IP를 업비트 API 화이트리스트에 등록
4. **방화벽 IP 제한** → Streamlit/Glances 포트는 본인 IP만 허용 권장

#### 환경 변수 (GitHub Secrets)

| 변수명 | 용도 | 필수 |
|--------|------|------|
| `UPBIT_ACCESS_KEY` | 업비트 API 액세스 키 | O |
| `UPBIT_SECRET_KEY` | 업비트 API 시크릿 키 | O |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 봇 토큰 | O |
| `TELEGRAM_CHAT_ID` | 텔레그램 채팅 ID | O |
| `VM_HOST` | VM 접속 주소 | O |
| `VM_USER` | VM 사용자명 | O |
| `VM_SSH_KEY` | VM SSH 비밀키 | O |
| `GH_TOKEN` | GitHub PAT (push용) | O |
| `KIS_*` | KIS 증권 API 키 (ISA/연금) | 선택 |

#### 주요 사이트 바로가기
- **업비트 API 관리**: https://upbit.com/mypage/open_api_management
- **업비트 API 문서**: https://docs.upbit.com
- **GCP 콘솔**: https://console.cloud.google.com
- **GitHub Actions**: 워크플로우 설정 (`.github/workflows/auto_trade.yml`)
""")


def _render_test_guide():
    """테스트 가이드."""
    st.markdown("### 테스트 가이드")

    st.markdown("""
#### 1단계: API 통신 검증
```bash
python3 -c "
import pyupbit, os
from dotenv import load_dotenv
load_dotenv()
upbit = pyupbit.Upbit(os.getenv('UPBIT_ACCESS_KEY'), os.getenv('UPBIT_SECRET_KEY'))
print(f'잔고: {upbit.get_balance(\"KRW\"):,.0f}원')
"
```
- ⬜ 원화 잔고가 정확히 출력되는지 확인
- ⬜ `IP_ADDRESS_NOT_REGISTERED` 에러가 없는지 확인

#### 2단계: 전략 로직 검증
```bash
python3 scripts/github_action_trade.py upbit  # DRY_RUN 모드
```
- ⬜ SMA/Donchian 시그널이 업비트 차트와 일치하는지 확인
- ⬜ signal_state.json 상태 전환이 정상 동작하는지 확인

#### 3단계: 소액 주문 테스트
```bash
# 수동 주문 (5,000원 테스트)
MANUAL_ORDER='{"coin":"BTC","side":"buy","pct":1}' python3 scripts/github_action_trade.py manual_order
```
- ⬜ 업비트 앱에서 체결 확인
- ⬜ 텔레그램 알림 수신 확인
- ⬜ trade_log.json에 기록 확인

#### 4단계: 시스템 생존 테스트
- ⬜ `pm2 start` 후 VS Code 종료 → 1시간 후 `pm2 logs` 확인
- ⬜ VM 재부팅 후 `pm2 status`로 자동 복구 확인
- ⬜ Streamlit 대시보드 (`http://VM_IP:8501`) 접속 확인
""")


_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
_STATE_FILE = os.path.join(_LOG_DIR, "vm_scheduler_state.json")

# 스케줄 규칙 (vm_scheduler.py RULES 미러)
_SCHEDULE_RULES = [
    {"mode": "upbit", "label": "코인 자동매매", "hours": [1, 5, 9, 13, 17, 21], "minute": 0, "days": "매일"},
    {"mode": "health_check", "label": "헬스체크", "hours": [1, 5, 9, 13, 17, 21], "minute": 5, "days": "매일"},
    {"mode": "daily_status", "label": "일일 자산 현황", "hours": [9], "minute": 0, "days": "평일"},
    {"mode": "kiwoom_gold", "label": "키움 금현물", "hours": [15], "minute": 5, "days": "평일"},
    {"mode": "kis_isa", "label": "KIS ISA", "hours": [15], "minute": 10, "days": "금요일"},
    {"mode": "kis_pension", "label": "KIS 연금저축", "hours": [15], "minute": 20, "days": "25~31 평일"},
]


def _read_tail(filepath: str, n: int = 30) -> list[str]:
    """파일 끝에서 n줄 읽기."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []


def _next_schedule_kst(rule: dict, now: datetime) -> datetime | None:
    """다음 실행 예정 시각 계산."""
    for day_offset in range(8):
        dt = now + timedelta(days=day_offset)
        for h in rule["hours"]:
            candidate = dt.replace(hour=h, minute=rule["minute"], second=0, microsecond=0)
            if candidate <= now:
                continue
            wd = candidate.weekday()
            days = rule["days"]
            if days == "매일":
                return candidate
            elif days == "평일" and wd < 5:
                return candidate
            elif days == "금요일" and wd == 4:
                return candidate
            elif days == "25~31 평일" and wd < 5 and 25 <= candidate.day <= 31:
                return candidate
    return None


@st.fragment(run_every=10)
def _render_scheduler_live():
    """스케줄러 실시간 모니터링 (10초 갱신)."""
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # 상태 파일 읽기
    state = {}
    if os.path.exists(_STATE_FILE):
        try:
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            pass

    # ── 스케줄러 상태 ──
    hb_epoch = state.get("__heartbeat_epoch", "")
    hb_kst = state.get("__heartbeat_kst", "")
    started_at = state.get("__started_at_kst", "")
    last_error = state.get("__last_error", "")

    hb_age = ""
    hb_status = "알 수 없음"
    if hb_epoch:
        try:
            age_sec = int(time.time() - float(hb_epoch))
            hb_age = f"{age_sec}초 전"
            hb_status = "정상" if age_sec < 180 else "경고 (stale)"
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("스케줄러", hb_status, delta=hb_age if hb_age else None,
              delta_color="normal" if "정상" in hb_status else "off")
    c2.metric("Heartbeat", hb_kst or "-")
    c3.metric("시작 시각", started_at or "-")
    c4.metric("현재 시각", now_str)

    if last_error:
        st.warning(f"최근 오류: {last_error}")

    # ── 모드별 실행 기록 + 다음 예정 ──
    rows = []
    for rule in _SCHEDULE_RULES:
        mode = rule["mode"]
        last_key = str(state.get(mode, "")).strip()
        last_label = "-"
        if last_key and len(last_key) == 12:
            try:
                dt = datetime.strptime(last_key, "%Y%m%d%H%M")
                last_label = dt.strftime("%m-%d %H:%M")
            except Exception:
                last_label = last_key

        next_dt = _next_schedule_kst(rule, now)
        next_label = next_dt.strftime("%m-%d %H:%M") if next_dt else "-"

        # 남은 시간
        remaining = ""
        if next_dt:
            delta = next_dt - now
            total_min = int(delta.total_seconds() // 60)
            if total_min < 60:
                remaining = f"{total_min}분"
            elif total_min < 1440:
                remaining = f"{total_min // 60}시간 {total_min % 60}분"
            else:
                remaining = f"{total_min // 1440}일 {(total_min % 1440) // 60}시간"

        rows.append({
            "모드": rule["label"],
            "스케줄": rule["days"],
            "마지막 실행": last_label,
            "다음 예정": next_label,
            "남은 시간": remaining,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 로그 뷰어 ──
    log_tabs = st.tabs(["스케줄러 로그", "Watchdog 로그", "업비트 로그"])

    log_files = [
        os.path.join(_LOG_DIR, "vm_scheduler.log"),
        os.path.join(_LOG_DIR, "vm_cron_watchdog.log"),
        os.path.join(_LOG_DIR, "upbit.log"),
    ]
    for tab, log_path in zip(log_tabs, log_files):
        with tab:
            lines = _read_tail(log_path, 40)
            if lines:
                st.code("".join(lines), language="log")
            else:
                fname = os.path.basename(log_path)
                st.info(f"{fname} 파일 없음 또는 비어있음")


def _render_operations_guide():
    """운영 관리 가이드."""
    st.markdown("### 운영 관리")

    # ── 실시간 스케줄러 모니터 ──
    st.markdown("#### 스케줄러 실시간 모니터")
    _render_scheduler_live()

    st.markdown("---")

    # ── 명령어 가이드 ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
#### 스케줄러 관리
```bash
# 스케줄러 시작
bash scripts/vm_scheduler_manager.sh start

# 스케줄러 상태 확인
bash scripts/vm_scheduler_manager.sh status

# 스케줄러 재시작
bash scripts/vm_scheduler_manager.sh restart

# 스케줄러 중지
bash scripts/vm_scheduler_manager.sh stop

# watchdog cron 설치/확인/제거
bash scripts/vm_cron_manager.sh install
bash scripts/vm_cron_manager.sh show
bash scripts/vm_cron_manager.sh remove
```
""")

    with col2:
        st.markdown("""
#### VM 관리
```bash
# 스케줄러 프로세스 확인
pgrep -f vm_scheduler

# 자동매매 로그 확인
tail -f ~/upbit/logs/upbit.log

# 스케줄러 로그 확인
tail -f ~/upbit/logs/vm_scheduler.log

# 상태 파일 확인
cat ~/upbit/logs/vm_scheduler_state.json

# watchdog cron 확인
crontab -l
```

#### 수동 실행
```bash
# 특정 모드 수동 실행
bash scripts/vm_run_job.sh upbit
bash scripts/vm_run_job.sh health_check

# 예약 주문 관리
bash scripts/vm_oneoff_manager.sh show
bash scripts/vm_oneoff_manager.sh add <mode> <kst>
```
""")

    st.markdown("---")
    st.markdown("""
#### 로컬 실행 (Windows)

**실행 파일**: `run.bat`
```bat
@echo off
start http://localhost:8501
streamlit run app.py --server.address localhost --server.port 8501 --server.headless true
pause
```

**Streamlit 설정**: `.streamlit/config.toml`
```toml
[server]
headless = false        # false = 실행 시 브라우저 자동 열림
runOnSave = false

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

> `headless = true`이면 브라우저가 자동으로 열리지 않음.
> `run.bat`의 `start http://...` 명령은 streamlit보다 먼저 실행되어 빈 페이지가 뜰 수 있으므로,
> `headless = false`로 설정하여 streamlit이 준비 완료 후 브라우저를 여는 것을 권장.
""")


def _render_audit_log():
    """점검 결과 기록 — 기능별 체크리스트."""
    st.markdown("### 시스템 점검 결과")
    st.caption("2026-03-05 전수 점검 — 모드별 탭별 기능 단위 체크리스트")

    # ══════════════════════════════════════════
    # 코인 모드 (5탭)
    # ══════════════════════════════════════════
    with st.expander("코인 모드 — 5탭 기능별 체크리스트", expanded=True):

        st.markdown("#### 탭1. 실시간 포트폴리오")
        st.markdown("""
- ✅ 새로고침 버튼 — 가격/잔고 수동 갱신 (TTL 캐시 우회)
- ✅ 워커 상태 표시 — 백그라운드 데이터 수집 상태
- ✅ 💰 전체 자산 현황 (확장기) — 자산 요약 테이블 (현금+코인별 보유량/현재가/평가금액)
- ✅ 보유 현금 (KRW) 메트릭 — 현재 현금 잔고
- ✅ CASH/미확인 메트릭 — CASH 상태 자산 수
- ✅ BUY 시그널 메트릭 — BUY 상태 자산 수
- ✅ 리밸런싱 상태 테이블 — 종목/전략/비중/포지션/시그널/현재가치/목표/액션
- ✅ 리밸런싱 매수 실행 버튼 — BUY 시그널 종목 일괄 매수
- ✅ 리밸런싱 진행바 + 실시간 로그
- ✅ 📊 단기 시그널 모니터링 (60봉) (확장기)
- ✅ BTC 전략별 캔들차트 — SMA/Donchian 지표 오버레이
- ✅ 기타 코인 전략별 캔들차트
- ✅ 시그널 요약 테이블 — 종목/전략/시간봉/포지션/실보유/실행/판단/이전/현재가/매수목표/매도목표/이격도
- ✅ ⚖️ 리밸런싱 규칙 (확장기) — 매수/매도 조건표 설명
- ✅ 📋 자산 상세 — 종목별 확장기 (전략/비중/시간봉)
- ✅ 종목별 가격/SMA 메트릭 — 현재가 vs SMA, 이격도
- ✅ 종목별 채널/SMA 기간 메트릭
- ✅ 종목별 이론 자산 메트릭 — 백테스트 수익률
- ✅ 종목별 동기화 상태 — 백테스트 vs 실보유 일치/불일치
- ✅ 종목별 📈 분석 & 벤치마크 서브탭
- ✅ 종목별 📋 체결 내역 서브탭 — 가상(백테스트) + 실제(거래소)
- ✅ 종목별 연도별 성과 테이블
- ✅ 종목별 성과 분석 차트
- ✅ 📊 합산 포트폴리오 성과 (확장기)
- ✅ 합산 포트폴리오 누적 수익률 차트
- ✅ 낙폭(DD) 차트
- ✅ 스택 차트 — 자산별 기여도
- ✅ 자산별 성과 테이블
- ✅ 연도별 성과 테이블
""")

        st.markdown("#### 탭2. 수동 주문")
        st.markdown("""
**VM 경유 즉시 주문:**
- ✅ 잔고 표시 — 최근 잔고 캡션
- ✅ 코인 선택 — selectbox (BTC/ETH/XRP/SOL/DOGE)
- ✅ 방향 선택 — selectbox (매수/매도)
- ✅ 비율 슬라이더 — 10~100%
- ✅ 주문 실행 버튼 (VM 경유) — GitHub Actions 트리거

**직접 주문 (로컬 API, 3초 fragment 자동 갱신):**
- ✅ 코인 선택 — selectbox (포트폴리오 + TOP20 + 직접입력)
- ✅ 현재가 메트릭
- ✅ 보유량 메트릭
- ✅ 평가금액 메트릭
- ✅ 보유 KRW 메트릭
- ✅ 호가단위 메트릭
- ✅ 최근 거래 알림 바 — 매수(빨강)/매도(파랑) 색상 구분
- ✅ 알림 닫기(✕) 버튼
- ✅ 30분봉 캔들 차트 — MA5, MA20, 거래량
- ✅ 호가창 HTML 테이블 — 매도/매수 10호가, 잔량
- ✅ 스프레드/매수비율 정보
- ✅ 호가 선택 → 주문가 반영 selectbox

**매수 패널:**
- ✅ 주문 유형 라디오 — 시장가/지정가
- ✅ 시장가 매수: 금액 입력 + 빠른배분(10/25/50/100%) + 예상수량 + 실행 버튼
- ✅ 지정가 매수: 가격 입력 + 수량 입력 + 총액 표시 + 빠른배분(10/25/50/100%) + 실행 버튼

**매도 패널:**
- ✅ 주문 유형 라디오 — 시장가/지정가
- ✅ 시장가 매도: 수량 입력 + 빠른배분(25/50/75/100%) + 예상금액 + 실행 버튼
- ✅ 지정가 매도: 가격 입력 + 수량 입력 + 총액 표시 + 빠른배분(25/50/75/100%) + 실행 버튼

**미체결 주문:**
- ✅ 미체결 주문 조회 버튼
- ✅ 미체결 주문 목록 + 개별 취소 버튼
""")

        st.markdown("#### 탭3. 거래 내역")
        st.markdown("""
**서브탭: 💸 실제 거래 내역 (거래소)**
- ✅ 조회 유형 selectbox — 전체/입금/출금/체결
- ✅ 화폐 필터 selectbox
- ✅ 조회 기간 — 시작일/종료일 date_input
- ✅ 조회 버튼
- ✅ 거래 내역 테이블 — 거래일시/유형/화폐/구분/금액/체결금액/수수료/상태
- ✅ 매수/매도 색상 스타일링

**서브탭: 🧪 가상 로그 (페이퍼)**
- ✅ 금액 입력 — 가상 입출금
- ✅ 입출금 버튼
- ✅ 누적 가상 조정액 표시

**서브탭: 📊 슬리피지 분석**
- ✅ 코인/시간봉 selectbox
- ✅ 슬리피지 분석 버튼
- ✅ 평균/최대/최소 슬리피지 메트릭
- ✅ 거래 수 메트릭
- ✅ 슬리피지 바 차트
- ✅ 슬리피지 상세 테이블
- ✅ 권장 슬리피지 안내
""")

        st.markdown("#### 탭4. 백테스트")
        st.markdown("""
**서브탭: 📈 개별 백테스트**
- ✅ 백테스트 대상 selectbox — 포트폴리오 + TOP20 + 직접입력
- ✅ 전략 선택 — SMA / Donchian
- ✅ SMA 파라미터: 기간 입력
- ✅ Donchian 파라미터: 매수 채널/매도 채널 기간 + 매도 기준선 라디오
- ✅ 시간봉 selectbox
- ✅ 백테스트 기간 — 시작일/종료일
- ✅ 수수료(%) / 슬리피지(%) 입력
- ✅ 편도/왕복 비용 표시
- ✅ 백테스트 실행 버튼
- ✅ 결과 메트릭: 총수익률/CAGR/승률/MDD/샤프비율
- ✅ 최종 잔고 성공 메시지
- ✅ 슬리피지 영향 분석
- ✅ 가격 & 전략 성과 차트 — 캔들+전략지표+equity+매매마커+DD
- ✅ 연도별 성과 테이블
- ✅ 성과 분석 컴포넌트
- ✅ 전략 상태 + 다음 행동 안내
- ✅ 거래 내역 확장기 + 테이블
- ✅ 일별 로그 CSV 다운로드 버튼
- ✅ 하단선 vs 중심선 비교 (Donchian 두 방법 비교)

**서브탭: 🛠️ 파라미터 최적화**
- ✅ 데이터 캐시 관리 확장기 — 캐시 목록 + 전체 삭제 버튼
- ✅ 최적화 대상 selectbox
- ✅ 전략 selectbox (SMA/Donchian)
- ✅ 최적화 방법 라디오 — Grid Search / Optuna
- ✅ Donchian 범위: 매수/매도 채널 시작/끝/간격 + 매도 방식 라디오
- ✅ SMA 범위: SMA 시작/끝/간격
- ✅ Optuna 설정: 탐색 횟수 + 목적함수 selectbox
- ✅ 기간/수수료/슬리피지 설정
- ✅ 최적화 시작 폼 제출 버튼
- ✅ 진행 상태 표시 + progress bar
- ✅ 최적 파라미터 요약 메시지
- ✅ 결과 테이블 + gradient 강조
- ✅ 파라미터 히트맵 차트
- ✅ 종합 평가 (CV 안정성) + 종합 안정성 등급 + 실전 권고

**서브탭: 🧩 보조 전략 (역추세)**
- ✅ 대상 티커 selectbox
- ✅ 메인 전략/기준 이평선/과매도 임계값 설정
- ✅ RSI 필터 체크박스
- ✅ 분할 매수 횟수 / 피라미딩 배율 설정
- ✅ 보조 전략 백테스트 실행 버튼
- ✅ 보조 전략 성과 차트 + RSI 차트
- ✅ 보조 전략 최적화 — 실행 버튼 + 결과 테이블 + 차트
- ✅ 최적 파라미터 반영 버튼

**서브탭: 📡 전체 종목 스캔**
- ✅ 매도 채널 비율 슬라이더
- ✅ 스캔 시작 버튼
- ✅ 스캔 결과 테이블
- ✅ 시간봉별 요약 테이블
- ✅ 종목별 최적 구성 테이블
""")

        st.markdown("#### 탭5. 트리거")
        st.markdown("""
- ✅ 전략 트리거 테이블 — 포트폴리오별 전략/주기/경로
- ✅ 매수/매도 로직 규칙 확장기 (변경 금지)
- ✅ 주문 로그 — trade_log.json 최근 50건 표시
""")

    # ══════════════════════════════════════════
    # 골드 모드 (5탭)
    # ══════════════════════════════════════════
    with st.expander("골드 모드 — 5탭 기능별 체크리스트", expanded=False):

        st.markdown("#### 탭1. 실시간 모니터링")
        st.markdown("""
**사이드바 설정:**
- ✅ 키움 API Keys 확장기 — 앱키/시크릿키/계좌번호 입력
- ✅ 전략 Data Editor — 전략명/전략/비중/간격/파라미터 편집
- ✅ 투자 비중 정보 — 비중 합계 표시
- ✅ 기준 시작일 date_input
- ✅ 초기 자본금 입력
- ✅ Gold 설정 저장 버튼

**메인 영역:**
- ✅ 새로고침 버튼
- ✅ 💰 계좌 현황 확장기
- ✅ 예수금 메트릭
- ✅ 금 보유량 메트릭 (g)
- ✅ 금 평가금액 메트릭
- ✅ 총 평가 메트릭 (손익 delta)
- ✅ 📊 백테스트 vs 실제 자산 비교 확장기
- ✅ 초기 자본 / 이론 총자산 / 실제 총자산 / 차이 메트릭
- ✅ 전략별 상세 테이블 — 전략/비중/배분자본/이론자산/수익률/포지션
- ✅ 📊 시그널 모니터링 확장기
- ✅ KRX 금현물 전략별 캔들차트 (Donchian/SMA 지표)
- ✅ 시그널 요약 테이블 — 전략/비중/현재가/매수목표/매도목표/이격도/포지션/시그널
- ✅ ⚖️ 자동매매 규칙 확장기 — 실행 시점/경로/전략별 규칙
""")

        st.markdown("#### 탭2. 수동 주문")
        st.markdown("""
**상단 정보 바 (3초 fragment 갱신):**
- ✅ 현재가 (원/g) 메트릭
- ✅ 금 보유 메트릭
- ✅ 평가금액 메트릭
- ✅ 예수금 메트릭
- ✅ 호가단위 메트릭
- ✅ 최근 거래 알림 바 + 해제 버튼

**호가창:**
- ✅ 호가창 HTML 테이블 — 매도/매수 호가, 잔량, 등락, 비율
- ✅ 호가 선택 → 주문가 반영 selectbox
- ✅ 스프레드/잔량 정보

**매수 패널:**
- ✅ 주문 유형 라디오 — 시장가/지정가
- ✅ 시장가 매수: 금액 입력 + 빠른배분(10/25/50/100%) + 예상수량 + 실행 버튼
- ✅ 지정가 매수: 가격 입력 + 수량 입력 + 총액 + 빠른배분(10/25/50/100%) + 실행 버튼

**매도 패널:**
- ✅ 주문 유형 라디오 — 시장가/지정가
- ✅ 시장가 매도: 수량 입력 + 빠른배분(25/50/75/100%) + 예상금액 + 실행 버튼
- ✅ 지정가 매도: 가격 입력 + 수량 입력 + 총액 + 빠른배분(25/50/75/100%) + 실행 버튼
""")

        st.markdown("#### 탭3. 백테스트")
        st.markdown("""
**서브탭: 📈 단일 백테스트**
- ✅ 사용 가능 데이터 범위 표시
- ✅ Gold 일봉 전체 다운로드 버튼 + 진행바
- ✅ 전략 selectbox (Donchian/SMA)
- ✅ 매수/매도 기간 입력
- ✅ 시작일 / 초기 자본 설정
- ✅ 백테스트 실행 버튼
- ✅ 결과 메트릭: 총수익률/CAGR/MDD/샤프/매매횟수/승률/최종자산/Calmar
- ✅ 전략 vs Buy & Hold 비교 테이블
- ✅ 연도별 성과 테이블
- ✅ 누적 수익률 차트 + Drawdown 차트
- ✅ 거래 내역 확장기 + 테이블

**서브탭: 🛠️ 파라미터 최적화**
- ✅ 전략/최적화 방법(Grid/Optuna) 선택
- ✅ 파라미터 범위 설정 (Donchian: 매수/매도 채널, SMA: 기간)
- ✅ 매도 방식 라디오 (하단선/중심선/비교)
- ✅ Optuna 설정: 탐색 횟수 + 목적함수
- ✅ 시작일/수수료/슬리피지/초기자본/정렬기준/최소매매횟수
- ✅ 최적화 시작 버튼 + 진행바
- ✅ 최적 파라미터 성공 메시지
- ✅ 상위 20개 결과 테이블
- ✅ Calmar 히트맵 차트
- ✅ CSV 다운로드 버튼
- ✅ 파라미터 선택 백테스트 selectbox + 상세 메트릭 + 차트
""")

        st.markdown("#### 탭4. 수수료/세금")
        st.markdown("""
- ✅ KRX 금현물 수수료 테이블 — 구분/요율/비고
- ✅ 세금 테이블 — 구분/세율/비고
- ✅ 매매차익 비과세 안내
- ✅ 보관료 정보 — 0.02%/년, 일할 계산
- ✅ 금 투자 방법별 비교 테이블 — KRX/ETF/은행/실물
""")

        st.markdown("#### 탭5. 트리거")
        st.markdown("""
- ✅ 전략 트리거 테이블 — 골드 전략 스케줄
""")

    # ══════════════════════════════════════════
    # ISA 모드 (8탭)
    # ══════════════════════════════════════════
    with st.expander("ISA 모드 — 8탭 기능별 체크리스트", expanded=False):

        st.markdown("#### 탭1. 실시간 모니터링")
        st.markdown("""
**사이드바 설정:**
- ✅ KIS API Keys 확장기 — 앱키/시크릿키/계좌번호/상품코드
- ✅ TREND ETF selectbox — 시그널용 ETF
- ✅ 매매 ETF selectbox — 실제 매매 ETF
- ✅ 평가 시스템 selectbox — 3단계/5단계
- ✅ 고평가/저평가 임계값 입력
- ✅ 상세 매수/매도 비중 설정 확장기 (상태별 매도/매수 비중)
- ✅ 초기자본(시드) 입력
- ✅ 초기 주식 비중 슬라이더
- ✅ 시작일 date_input
- ✅ ISA 설정 저장 버튼
- ✅ ETF별 권장 시작일/비중 테이블 확장기
- ✅ 라이브 전략 요약 — 총수익률/CAGR/MDD/Calmar 메트릭

**메인 영역:**
- ✅ 잔고 새로고침 버튼
- ✅ 매수 가능금액 메트릭
- ✅ 주식 평가 메트릭
- ✅ 총 평가 메트릭
- ✅ 보유 종목 수 메트릭
- ✅ ISA 잔고 보유 종목 테이블
- ✅ 이격도 메트릭
- ✅ 시장 상태 메트릭
- ✅ 매도 비율 메트릭
- ✅ 매수 비율 메트릭
- ✅ 백테스트 목표 비중 정보
- ✅ 권장 주문 (매수/매도/홀드)
- ✅ 백테스트 성과 표시
- ✅ 시그널 가격 차트 (확정 + 실시간/가상)
- ✅ 성장 추세선
- ✅ 고평가/저평가 임계선
- ✅ WDR 시그널 OHLC 차트 (plotly)
- ✅ 전략 수익률 추이 차트
- ✅ 성과 분석 컴포넌트
""")

        st.markdown("#### 탭2. 수동 주문")
        st.markdown("""
- ⬜ 수동 주문 기능 — **준비중** (placeholder)
""")

        st.markdown("#### 탭3. 주문방식")
        st.markdown("""
- ⬜ 주문 방식 가이드 — **준비중** (placeholder)
""")

        st.markdown("#### 탭4. 수수료/세금")
        st.markdown("""
- ⬜ 수수료/세금 정보 — **준비중** (placeholder)
""")

        st.markdown("#### 탭5. 미국 위대리 백테스트")
        st.markdown("""
- ⬜ 미국 백테스트 상세 뷰 — **준비중** (placeholder)
""")

        st.markdown("#### 탭6. 위대리 최적화")
        st.markdown("""
- ✅ 평가 시스템 selectbox — 3단계/5단계
- ✅ 탐색 시작일/종료일
- ✅ 매매 ETF 상장일 자동 감지
- ✅ 탐색 방식 selectbox — 그리드/랜덤/Optuna
- ✅ 정렬 기준 selectbox — Calmar/Sharpe/CAGR/수익률
- ✅ 초기 자본/수수료/최대 평가 수/탐색 횟수 설정
- ✅ 고평가 범위 (최소/최대/스텝) 입력
- ✅ 저평가 범위 (최소/최대/스텝) 입력
- ✅ 초기 주식 비중 범위 (최소/최대/스텝) 입력
- ✅ 평가별 매수/매도 비율 기본값 확장기
- ✅ 예상 조합 수 표시
- ✅ 최적화 시작 버튼
- ✅ 최적화 결과 테이블 (최대 300행)
- ✅ 1등 조합 상세 정보
- ✅ 1등 조합 → 라이브 백테스트 적용 버튼
""")

        st.markdown("#### 탭7. 라이브 백테스트")
        st.markdown("""
**설정:**
- ✅ 평가 시스템 selectbox — 3단계/5단계
- ✅ 시작일/종료일 date_input
- ✅ 초기 자본/수수료 설정
- ✅ 탐색 방식 selectbox — 그리드/랜덤/Optuna
- ✅ 라이브 애니메이션 체크박스 + 프레임 단위/지연 설정
- ✅ 라이브 계산 실행 버튼

**사전 계산 범위:**
- ✅ 고평가/저평가/초기 비중 범위 (각 최소/최대/스텝)
- ✅ 평가별 매수/매도 비율 범위 (고/중/저/초고/초저 각각)
- ✅ 주식 비중 제한 설정 (최소/최대)
- ✅ 사전 계산 시작 버튼
- ✅ 세션 캐시 초기화 / 디스크 캐시 삭제 버튼
- ✅ 사전 계산 진행률 바

**라이브 다이얼:**
- ✅ 고평가/저평가 임계값 슬라이더
- ✅ 초기 주식 비중 슬라이더
- ✅ 상태별 매도/매수 비율 슬라이더 (고/중/저/초고/초저)

**결과:**
- ✅ 최적화 성과 Top 30 확장기 + 테이블
- ✅ 다이얼 적용 순위 선택 + 설정 적용 버튼
- ✅ 트렌드 차트: 가격 vs 성장 추세 vs 임계 밴드
- ✅ 이격도(%) 차트
- ✅ 자동 재생 체크박스 + 라이브 프레임 슬라이더
- ✅ 현재 상태/보유 비중 표시
- ✅ 성과 분석 컴포넌트
- ✅ 상세 성과 지표 확장기 — 매매횟수/승률/최종자산
- ✅ 거래 기록 테이블
""")

        st.markdown("#### 탭8. 트리거")
        st.markdown("""
- ✅ ISA 전략 트리거 — 매주 금요일 15:10 KST
""")

    # ══════════════════════════════════════════
    # 연금저축 모드 (8탭)
    # ══════════════════════════════════════════
    with st.expander("연금저축 모드 — 8탭 기능별 체크리스트", expanded=False):

        st.markdown("#### 탭1. 실시간 모니터링")
        st.markdown("""
**잔고:**
- ✅ 잔고 새로고침 버튼
- ✅ 잔고 기준시각 표시
- ✅ 매수 가능금액 메트릭
- ✅ 주식 평가 메트릭
- ✅ 총 평가 메트릭
- ✅ 보유 종목 수 메트릭
- ✅ 현재 보유 종목 테이블

**전체 포트폴리오 합산:**
- ✅ 활성 전략별 목표 수량 동적 컬럼
- ✅ 결합 포트폴리오 합산 테이블

**LAA 전략:**
- ✅ LAA 시그널 자동 계산
- ✅ 전략 가이드 이동 버튼
- ✅ 리스크 상태 메트릭 — Risk-On / Risk-Off
- ✅ 리스크 자산 메트릭 — QQQ 또는 SHY
- ✅ 국내 ETF 메트릭
- ✅ 권장 동작 메트릭 — HOLD / REBALANCE
- ✅ LAA 분석 이유 설명
- ✅ TIGER 미국S&P500 현재가/200일선/이격도 메트릭
- ✅ TIGER 미국S&P500 + 200일선 차트
- ✅ 이격도 차트
- ✅ 목표 배분 vs 현재 보유 테이블
- ✅ LAA 백테스트 자산/실제 총자산/최대 비중 차이/포지션 동기화 메트릭
- ✅ LAA 성과 분석 컴포넌트

**듀얼모멘텀 전략:**
- ✅ DM 시그널 자동 계산
- ✅ 전략 가이드 이동 버튼
- ✅ 선택 자산 / 국내 ETF / 카나리아 수익률 / 권장 동작 메트릭
- ✅ DM 분석 이유 설명
- ✅ 시그널/전략 선택 로직 설명
- ✅ 요약 점수 테이블
- ✅ 모멘텀 계산 과정 테이블 (1M/3M/6M/12M)
- ✅ 리밸런싱 예상 포트폴리오 — 비중/배정금액/잔여현금
- ✅ DM 백테스트/실제 자산/포지션 비교 메트릭
- ✅ DM 성과 분석 컴포넌트

**VAA 전략:**
- ✅ VAA 시그널 자동 계산
- ✅ 선택 자산 / 포지션 / 선택 ETF / 권장 동작 메트릭
- ✅ VAA 분석 이유 설명
- ✅ 모멘텀 스코어 테이블
- ✅ 목표 배분 vs 현재 보유 테이블

**CDM 전략:**
- ✅ CDM 시그널 자동 계산
- ✅ 공격 모듈 수 / 방어자산 12M수익률 / 권장 동작 / 최대 비중 ETF 메트릭
- ✅ CDM 분석 이유 설명
- ✅ 모듈별 결과 테이블
- ✅ 목표 배분 vs 현재 보유 테이블
""")

        st.markdown("#### 탭2. 백테스트")
        st.markdown("""
**공통 설정:**
- ✅ 전략 selectbox — LAA/듀얼모멘텀/VAA/CDM
- ✅ 시작일 date_input
- ✅ 초기 자본 / 수수료 입력

**LAA 백테스트:**
- ✅ LAA 백테스트 실행 버튼
- ✅ 총수익률/CAGR/MDD/샤프 메트릭
- ✅ 최종 자산 메트릭
- ✅ 자산 곡선 차트
- ✅ 연도별 수익률 테이블
- ✅ 월별 자산 배분 이력 테이블
- ✅ LAA 성과 분석 컴포넌트

**듀얼모멘텀 백테스트:**
- ✅ DM 백테스트 실행 버튼
- ✅ DM 성과 분석 컴포넌트

**VAA 백테스트:**
- ✅ VAA 백테스트 실행 버튼
- ✅ VAA 성과 분석 컴포넌트
- ✅ 월별 포지션 이력 테이블

**CDM 백테스트:**
- ✅ CDM 백테스트 실행 버튼
- ✅ CDM 성과 분석 컴포넌트
- ✅ 월별 배분 이력 테이블
""")

        st.markdown("#### 탭3. 수동 주문")
        st.markdown("""
- ✅ 매매 ETF selectbox
- ✅ 주문 코드 대체 경고/설명
- ✅ 현재가 / 보유수량 / 평가금액 / 매수 가능금액 메트릭
- ✅ ETF 일봉 차트 — Candlestick + SMA(5/20/60/120/200) + 거래량
- ✅ 호가창 HTML — 매도호가/매수호가/현재가
- ✅ 호가 통계 — 스프레드/잔량

**매수 탭:**
- ✅ 주문 방식 라디오 — 시장가/지정가/동시호가/시간외 종가
- ✅ 매수 지정가/수량 입력
- ✅ 주문 계획 표시 (단가/수량/총금액)
- ✅ 매수 실행 버튼

**매도 탭:**
- ✅ 주문 방식 라디오 — 시장가/지정가/동시호가/시간외 종가
- ✅ 매도 지정가/수량 입력
- ✅ 전량 매도 체크박스
- ✅ 주문 계획 표시
- ✅ 매도 실행 버튼
""")

        st.markdown("#### 탭4. 거래내역")
        st.markdown("""
- ✅ 조회 기간 — 시작일/종료일
- ✅ 매매구분 selectbox — 전체/매수/매도
- ✅ 체결구분 selectbox — 전체/체결/미체결
- ✅ 종목코드/주문번호 필터 (선택)
- ✅ 최대 조회 건수 selectbox
- ✅ 입출금/정산 내역 조회 버튼 + 결과 테이블
- ✅ 입출금 원본 응답 보기 확장기
- ✅ 주문/체결 내역 조회 버튼
- ✅ 조회건수/체결금액 합계/체결건수 메트릭
- ✅ 거래내역 상세 테이블
- ✅ 원본 응답 보기 확장기
""")

        st.markdown("#### 탭5. 전략 가이드")
        st.markdown("""
**LAA 가이드:**
- ✅ 전략 개요 — 4자산 25% 균등배분
- ✅ 의사결정 흐름 — 월별 리밸런싱 로직
- ✅ 국내 ETF 매핑 테이블

**듀얼모멘텀 가이드:**
- ✅ 전략 개요 — 가중 모멘텀 점수
- ✅ 의사결정 흐름 — 월별 전환 로직
- ✅ 국내 ETF 매핑 테이블
- ✅ 운용 특성 — 시장 대응/리스크 관리

**VAA 가이드:**
- ✅ 전략 개요 — 13612W 모멘텀
- ✅ 의사결정 흐름 — 공격/방어 전환 로직
- ✅ 국내 ETF 매핑 테이블

**CDM 가이드:**
- ✅ 전략 개요 — 4모듈 이중 모멘텀
- ✅ 의사결정 흐름 — 4모듈 상대+절대 모멘텀
- ✅ 국내 ETF 매핑 테이블
""")

        st.markdown("#### 탭6. 주문방식")
        st.markdown("""
- ✅ 주문방식 설명 테이블 — 시장가/지정가/동시호가/시간외 종가
- ✅ 호가단위 테이블 — 가격대별 최소 단위
- ✅ 자동매매 흐름 설명 — 매월 25~31일 평일 KST 15:20
""")

        st.markdown("#### 탭7. 수수료/세금")
        st.markdown("""
- ✅ 증권사별 매매 수수료 비교 테이블
- ✅ ETF 보수 테이블
- ✅ 연금저축 세제혜택 — 세액공제/과세이연/연금소득세
- ✅ LAA 복리 효과 설명
""")

        st.markdown("#### 탭8. 트리거")
        st.markdown("""
- ✅ 연금저축 전략 트리거 — 매월 25~31일 평일 15:10 KST
""")

    # ══════════════════════════════════════════
    # 매매 엔진
    # ══════════════════════════════════════════
    with st.expander("매매 엔진 — 기능별 체크리스트", expanded=False):
        st.markdown("""
- ✅ SMA 전략 — `strategy/sma.py` BUY/SELL/HOLD 반환, RSI/MACD/볼린저 보조지표
- ✅ Donchian 전략 — `strategy/donchian.py` upper/lower 채널, midline 매도 모드
- ✅ 시그널 전환 감지 — `_determine_signal()` 이전 상태↔현재 비교, 전환 시에만 주문
- ✅ signal_state 관리 — load/save 구현, 키: `{MARKET}-{COIN}_{STRATEGY}_{PARAM}_{IV}`
- ✅ run_auto_trade — 분석→매도→매수→상태저장→알림 전체 사이클
- ✅ run_manual_order — MANUAL_ORDER JSON 파싱, 전략 없이 즉시 실행
- ✅ smart_buy — 분할 지정가 + 시장가 폴백, pyupbit None 체크
- ✅ smart_sell — 분할 지정가 + 시장가 폴백, pyupbit None 체크
- ✅ adaptive_buy — 슬리피지 감시, 0.1% 이하 시 전량 매수
- ✅ _append_trade_log — trade_log.json 기록 + GitHub push, 200건 제한
- ✅ 캔들 마감 대기 — `_wait_for_candle_boundary()` 4H 경계 시각 대기
- ✅ WDR 위대리 전략 — 3/5단계 평가, 260주 로그선형회귀
- ✅ LAA 전략 — 월간 리밸런싱, 4자산 균등배분
- ✅ 듀얼모멘텀 전략 — SPY/EFA 상대모멘텀 + BIL 절대모멘텀
- ✅ VAA 전략 — 13612W 모멘텀, 공격/방어 전환
- ✅ CDM 전략 — 4모듈 듀얼모멘텀
""")

    # ══════════════════════════════════════════
    # 인프라/보안
    # ══════════════════════════════════════════
    with st.expander("인프라 및 보안 — 기능별 체크리스트", expanded=False):
        st.markdown("""
**스케줄링 구조 (단일 로직):**
- ✅ Python 스케줄러 (`vm_scheduler.py`) — **유일한 스케줄 실행 주체**, 상주 프로세스
- ✅ 스케줄러 RULES — upbit(정시), health_check(+5분), daily(평일09:00), gold(평일15:05), isa(금15:10), pension(25~31평일15:20)
- ✅ VM cron — 스케줄러 watchdog 전용 (5분마다 `vm_scheduler_manager.sh ensure`)
- ✅ GitHub Actions — 스케줄러 watchdog (5분마다 `vm_scheduler_ensure`) + 수동 dispatch
- ✅ vm_run_job.sh — 8개 모드, `flock` 중복 방지, 성공 시 `vm_scheduler_state.json` 기록
- ✅ 상태 추적 — `vm_scheduler_state.json` (실행 기록 + heartbeat)
- ✅ 예약 주문 — `vm_reserved_orders.json` (1회성 예약)

**헬스체크 누락 감지:**
- ✅ 1차: `vm_scheduler_state.json` 상태파일 기준 확인
- ✅ 2차: `logs/upbit.log` mtime 보조 확인 (상태파일 미갱신 시 보완)
- ✅ 복구: 누락 감지 시 `run_auto_trade()` 자동 실행 (시그널 전환 감지로 중복 주문 방지)

**인프라:**
- ✅ GCP VM 인스턴스 — 가동 중, 고정 IP 할당
- ✅ portfolio.json — BTC Donchian(115/105, 4H) + SMA(29, 1D), 50:50 비중
- ✅ API 키 관리 — GitHub Secrets, `.env` 분리, 하드코딩 없음
- ✅ 텔레그램 봇 — HTML 포맷, 4096자 청킹, 에러 핸들링
- ⬜ Deploy Key — VM 키 생성 완료, **GitHub 등록 대기 중**
- ✅ 모드별 설정 분리 — `config/{mode}.json` load/save_mode_config 함수
""")

    # ── 발견된 이슈 ──
    st.markdown("---")
    st.markdown("#### 발견된 이슈 및 조치사항")
    issues = pd.DataFrame([
        {"우선순위": "🟢 해결", "이슈": "이중 스케줄링 (cron + scheduler) 충돌",
         "상태": "완료", "조치": "Python 스케줄러를 유일한 실행 주체로 통합, cron은 watchdog 전용"},
        {"우선순위": "🟢 해결", "이슈": "헬스체크 오탐 (상태파일 미갱신)",
         "상태": "완료", "조치": "vm_run_job.sh에 상태파일 기록 추가 + 로그 mtime 보조 확인"},
        {"우선순위": "🔴 높음", "이슈": "Deploy Key GitHub 등록 미완료",
         "상태": "대기", "조치": "공개키를 GitHub Deploy Keys에 등록 (Allow write access)"},
        {"우선순위": "🟡 중간", "이슈": "signal_state.json 파일 미생성",
         "상태": "정상", "조치": "코드 구현 완료, 첫 자동매매 실행 시 자동 생성"},
        {"우선순위": "🟡 중간", "이슈": "trade_log.json 파일 미생성",
         "상태": "정상", "조치": "코드 구현 완료, 첫 주문 시 자동 생성"},
    ])
    st.dataframe(issues, use_container_width=True, hide_index=True)


def _render_progress(progress_items: list):
    """진행 내역 표시."""
    st.markdown("### 프로젝트 진행 내역")

    # 통계 계산
    total = len(progress_items)
    done = sum(1 for p in progress_items if p["status"] == "완료")
    in_prog = sum(1 for p in progress_items if p["status"] == "진행중")
    pending = sum(1 for p in progress_items if p["status"] == "대기")

    # 진행률 바
    pct = done / total if total > 0 else 0
    st.progress(pct, text=f"전체 진행률: {done}/{total} ({pct:.0%})")

    # 상태 요약
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("완료", f"{done}건")
    mc2.metric("진행중", f"{in_prog}건")
    mc3.metric("대기", f"{pending}건")

    # Phase별 그룹핑 표시
    phases = []
    for item in progress_items:
        if item["phase"] not in phases:
            phases.append(item["phase"])

    for phase in phases:
        phase_items = [p for p in progress_items if p["phase"] == phase]
        phase_done = sum(1 for p in phase_items if p["status"] == "완료")
        phase_total = len(phase_items)

        status_icon = ""
        if phase_done == phase_total:
            status_icon = " ✅"
        elif any(p["status"] == "진행중" for p in phase_items):
            status_icon = " 🔄"

        with st.expander(f"{phase} ({phase_done}/{phase_total}){status_icon}", expanded=(phase_done < phase_total)):
            rows = []
            for item in phase_items:
                status_emoji = {"완료": "✅", "진행중": "🔄", "대기": "⏳"}.get(item["status"], "❓")
                rows.append({
                    "상태": f"{status_emoji} {item['status']}",
                    "작업": item["task"],
                    "비고": item.get("note", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── 메인 렌더 함수 ───
def render_project_logic():
    """프로젝트 로직 탭 전체 렌더링."""
    st.header("프로젝트 로직")
    st.caption("24시간 클라우드 기반 업비트 자동매매 시스템 — 아키텍처, 전략, 가이드")

    # 탭 구성
    tabs = st.tabs([
        "아키텍처",
        "매매 전략",
        "실행 흐름",
        "파일 구조",
        "보안/설정",
        "테스트",
        "운영 관리",
        "점검 결과",
        "진행 내역",
    ])

    with tabs[0]:
        _render_architecture()

    with tabs[1]:
        _render_trading_strategy()

    with tabs[2]:
        _render_execution_flow()

    with tabs[3]:
        _render_file_structure()

    with tabs[4]:
        _render_security_guide()

    with tabs[5]:
        _render_test_guide()

    with tabs[6]:
        _render_operations_guide()

    with tabs[7]:
        _render_audit_log()

    with tabs[8]:
        progress = _load_progress()
        _render_progress(progress)
