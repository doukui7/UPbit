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

## 주문 경로
- **자동매매**: VM cron → github_action_trade.py → Upbit API (고정 IP)
- **수동 주문**: Streamlit UI → GitHub Actions (workflow_dispatch) → VM SSH → Upbit API
- **계좌 조회**: Streamlit UI → GitHub Actions → VM SSH → Upbit API → account_cache.json push

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
