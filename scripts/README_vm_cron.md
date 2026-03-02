# VM Cron 운영 가이드

## 개요
- 자동매매 정시 실행은 GitHub `schedule` 대신 VM `cron`으로 수행합니다.
- GitHub Actions는 수동 실행과 VM 크론 제어 용도로 사용합니다.

## GitHub Actions에서 사용
- 워크플로우: `Auto Trade`
- 입력값(`run_job`) 선택:
  - `vm_cron_install`: VM 크론 설치/갱신
  - `vm_cron_show`: VM 크론 블록 확인
  - `vm_cron_remove`: VM 크론 블록 제거
  - `trade`, `health_check`, `daily_status`, `kiwoom_gold`, `kis_isa`, `kis_pension`: VM에서 단건 실행

## VM 직접 명령
```bash
cd ~/upbit
bash scripts/vm_cron_manager.sh install
bash scripts/vm_cron_manager.sh show
bash scripts/vm_cron_manager.sh remove
```

## 설치되는 기본 스케줄 (KST)
- `01:00,05:00,09:00,13:00,17:00,21:00` 코인 자동매매 (`upbit`)
- `01:05,05:05,09:05,13:05,17:05,21:05` 헬스체크 (`health_check`)
- 평일 `09:00` 일일 현황 (`daily_status`)
- 평일 `15:05` 키움 금현물 (`kiwoom_gold`)
- 금요일 `15:10` KIS ISA (`kis_isa`)
- 매월 25~31일 평일 `15:10` KIS 연금저축 (`kis_pension`)
