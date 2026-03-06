# 프로젝트 작업 규칙

- 사용자가 명시적으로 요청하지 않으면 UI 레이블/제목/설명 문구를 영어로 바꾸지 않는다.
- 기존에 표시되던 한국 종목명(코드 + 한글명) 표기를 임의로 삭제하거나 축소하지 않는다.
- 기존 동작을 리팩터링할 때는 화면 표시 요소(텍스트/라벨/종목명 병기)가 유지되는지 먼저 확인한다.
- 요청한 것 이외의 내용은 임의로 수정하지 않는다.
- 요청 범위를 벗어나는 수정이 필요하면 작업 전에 반드시 사용자 확인을 받는다.
- **오류가 없는 코드는 임의로 수정하지 않는다.** 정상 작동하는 기존 코드(ISA/연금저축/골드/코인 등)를 리팩터링, 구조 변경, 스타일 변경 등의 이유로 건드리지 않는다. 버그 수정이나 기능 추가 요청이 있을 때만 해당 범위 내에서 수정한다.
- 한글로 작성된 화면 문구/레이블/설명/가이드를 영어로 변경하지 않는다.
- 소스 파일은 EUC-KR(CP949) 인코딩으로 저장한다.

# 주문 실행 규칙 (변경 금지)

아래 규칙은 자동매매/수동매매 코드 수정 시 반드시 준수해야 하며, 사용자의 명시적 승인 없이 변경할 수 없다.

## 자동매매 (run_auto_trade)
1. **전환 감지 기반**: 이전 포지션 상태(signal_state.json)와 현재 전략 시그널을 비교하여 전환이 발생한 경우에만 주문 실행
2. **매도 우선**: 매도 시그널을 먼저 실행한 뒤 현금 확보 후 매수 진행
3. **비중 비례 배분**: 매수 금액 = 총 포트폴리오 가치 × (전략 비중 / 100) - 현재 보유분
4. **주기 필터**: 4H 전략은 KST 01/05/09/13/17/21시, 1D 전략은 KST 09시에만 실행
5. **캔들 마감 대기**: 4H 전략 포함 시 정각까지 대기 (최대 12분)
6. **최소 주문 금액**: 5,000원 미만 주문 스킵
7. **상태 저장**: 실제 체결 확인 후에만 signal_state 업데이트
8. **잔고 캐시**: 매 실행 후 balance_cache.json + signal_state.json을 GitHub에 push

## 수동 주문 (run_manual_order)
1. **전략 분석 없음**: 사용자가 지정한 코인/방향/비율로 즉시 실행
2. **매수**: 가용 KRW × (비율/100) × 0.999 금액으로 adaptive_buy
3. **매도**: 보유 수량 × (비율/100) 수량으로 smart_sell
4. **최소 주문 금액**: 5,000원 미만 주문 거부
5. **결과 알림**: 텔레그램으로 주문 결과 + 잔고 현황 전송
6. **잔고 갱신**: 주문 후 balance_cache.json 즉시 갱신 + GitHub push

## 연금저축 예약주문 (run_kis_pension_trade)
1. **예약주문 기반**: config/pension_orders.json의 '대기' 상태 주문을 전부 실행 (LAA 전략 분석 없음)
2. **날짜 무시**: scheduled_kst 시간에 관계없이 대기 주문이면 모두 실행
3. **주문 방식**: 동시호가 (장마감) / 시간외 종가 / 시장가 / 지정가
4. **동시호가 실패 fallback**: 동시호가 실패 시 시간외 종가(ord_dvsn="06")로 자동 재주문
5. **결과 저장**: pension_orders.json 상태를 완료/실패로 업데이트
6. **텔레그램 알림**: 처리 건수 + 각 주문 결과 전송

### 자동매매 흐름 (GitHub Actions → VM → 주문 실행)

```
[1] 사용자: Streamlit UI → 예약 주문 관리에서 주문 추가
         ↓ (pension_orders.json 생성/수정)
[2] 사용자: pension_orders.json을 GitHub에 커밋/푸시
         ↓
[3] GitHub Actions: pension_trade.yml schedule 트리거
    - 매월 25~31일 평일 KST 15:20 (UTC 06:20)
    - 또는 workflow_dispatch로 수동 실행
         ↓
[4] Runner: actions/checkout@v4 (master 브랜치 체크아웃)
         ↓
[5] Runner → VM SSH (appleboy/ssh-action):
    a. cd ~/upbit
    b. git fetch origin + git reset --hard origin/master
       → 최신 pension_orders.json 반영
    c. bash scripts/vm_run_job.sh kis_pension
         ↓
[6] VM: github_action_trade.py (TRADING_MODE=kis_pension)
    → run_kis_pension_trade() 실행:
    a. config/pension_orders.json 로드
    b. status="대기" 주문 필터링
    c. KIS 인증 (KIS_APP_KEY/SECRET/PENSION_ACCOUNT_NO)
    d. 각 주문 실행:
       - 동시호가: execute_closing_auction_buy/sell()
         → 실패 시 send_order(ord_dvsn="06") 시간외 종가 재주문
       - 시간외: send_order(ord_dvsn="06")
       - 시장가: smart_buy_krw() / smart_sell_all()
       - 지정가: send_order(ord_dvsn="00")
    e. pension_orders.json 상태 업데이트 (완료/실패)
    f. 텔레그램 알림 전송
         ↓
[7] Runner: SCP로 VM에서 pension_orders.json 다운로드
         ↓
[8] Runner: git add + commit + push
    → 실행 결과가 GitHub에 동기화
         ↓
[9] 사용자: Streamlit UI에서 실행 결과 확인 (git pull 후)
```

### 상세 체크리스트

#### A. 사전 준비 (1회)
- [ ] GitHub Secrets 등록: `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_PENSION_ACCOUNT_NO`
- [ ] GitHub Secrets 등록: `VM_HOST`, `VM_USER`, `VM_SSH_KEY`
- [ ] GitHub Secrets/Vars 등록: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- [ ] VM에 `~/upbit` 리포 클론 + venv 설정
- [ ] VM에 deploy key 등록 (push용, 선택사항 — 없어도 Runner에서 push)

#### B. 주문 등록 (매월)
- [ ] Streamlit UI → 연금저축 → 예약 주문 관리 → 매수/매도 주문 추가
- [ ] 주문 정보 확인: ETF코드, 수량, 주문방식(동시호가/시간외/시장가/지정가)
- [ ] pension_orders.json이 로컬에 정상 저장되었는지 확인
- [ ] `git add config/pension_orders.json && git commit && git push`

#### C. 스케줄 실행 검증
- [ ] pension_trade.yml에 schedule cron 존재: `"20 6 25-31 * 1-5"`
- [ ] pension_trade.yml에 Checkout 스텝 존재 (push를 위해 필수)
- [ ] SSH 스크립트에서 `git reset --hard` → 최신 pension_orders.json 반영
- [ ] vm_run_job.sh에서 TRADING_MODE=kis_pension 지원 확인
- [ ] run_kis_pension_trade()가 대기 주문 전부 처리 확인 (날짜 무시)

#### D. 주문 실행 검증
- [ ] KIS 인증 성공 여부 (텔레그램 알림 or VM 로그 확인)
- [ ] 동시호가 주문 → 체결 확인
- [ ] 동시호가 실패 시 시간외 종가 자동 재주문 작동 확인
- [ ] pension_orders.json 상태가 "완료" 또는 "실패"로 업데이트
- [ ] 텔레그램에 주문 결과 알림 수신

#### E. 결과 동기화 검증
- [ ] Runner가 SCP로 pension_orders.json 다운로드 성공
- [ ] git push로 실행 결과가 GitHub에 반영
- [ ] Streamlit UI에서 git pull 후 결과 확인 가능

#### F. 장애 대응
- [ ] KIS 인증 실패 → 텔레그램 알림 + VM 로그 확인
- [ ] 대기 주문 0건 → "대기 중인 예약주문 없음" 텔레그램 알림
- [ ] SCP 실패 → Runner 로그에 WARN 출력 (치명적이지 않음)
- [ ] 주문 실패 → pension_orders.json에 실패 사유 기록, 텔레그램 알림

## 주문 경로
- **코인 자동매매**: VM cron → github_action_trade.py → Upbit API (고정 IP)
- **코인 수동 주문**: Streamlit UI → GitHub Actions (workflow_dispatch) → VM SSH → Upbit API
- **코인 계좌 조회**: Streamlit UI → GitHub Actions → VM SSH → Upbit API → account_cache.json push
- **연금저축 예약주문**: Streamlit UI → pension_orders.json 커밋 → GitHub Actions schedule → VM SSH → KIS API → 결과 push

## GitHub Actions ↔ VM 연동 참고사항 (2026-03-05 검증 완료)

### 동작하지 않는 방법 (실패 기록)
1. **`github.token` (ghs_) VM 전달**: Actions runner 전용 토큰이라 외부 VM에서 GitHub API 호출 불가 (Bad credentials)
2. **OAuth PAT (gho_) 등록**: 동작하지만 ~8시간 후 만료, 영구 솔루션 아님
3. **VM에서 SSH git fetch**: GitHub SSH 호스트키 변경으로 `Host key verification failed` 발생
4. **`ssh-keyscan` 후 SSH fetch**: VM 환경에 따라 known_hosts 갱신 불완전
5. **pyupbit 커스텀 JWT 해시 (`unquote()` 사용)**: pyupbit은 `urlencode().replace("%5B%5D=","[]=")` 방식
6. **`requests.get(params=...)` 로 Upbit API 호출**: pyupbit은 `data=` 파라미터 사용

### 동작하는 방법 (현재 적용됨)
1. **VM git fetch → HTTPS remote**: `git remote set-url origin https://github.com/doukui7/UPbit.git` 사용
2. **GitHub push → Runner에서 push**: VM에서 SCP로 파일 다운로드 → runner에서 git push (github.token 자동 적용)
3. **Upbit API 인증 → pyupbit 네이티브**: `self.upbit._request_headers(params)` + `requests.get(data=params)` 사용
4. **GH_TOKEN → 전면 제거**: 모든 workflow에서 VM으로 GH_TOKEN 전달하지 않음

## 로깅 규칙
- 모든 주문 시도와 결과를 trade_log.json에 기록
- 로그 항목: 시각, 모드(auto/manual), 코인, 방향, 금액/수량, 결과(성공/실패/스킵), 사유
- trade_log.json은 GitHub에 push하여 Streamlit UI에서 조회 가능

# Project Rules

- Do not change UI labels, titles, subtitles, or help text to English unless the user explicitly asks.
- Do not remove Korean instrument name displays (code + Korean name) unless explicitly requested.
- Do not change anything outside the user's explicit request without permission.
- If out-of-scope changes are required, get user confirmation before editing.
- Set source file encoding to EUC-KR (CP949).
