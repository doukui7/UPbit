# VM Python 스케줄러 가이드

## 개요
- `cron` 설치 권한이 없는 VM에서도 자동 실행이 가능하도록 파이썬 상시 스케줄러를 사용합니다.
- 스케줄러 프로세스는 `scripts/vm_scheduler.py` 입니다.
- 시작/중지/상태 확인은 `scripts/vm_scheduler_manager.sh`로 수행합니다.

## GitHub Actions에서 사용
- 워크플로우: `Auto Trade`
- `run_job` 선택:
  - `vm_scheduler_start`: VM 스케줄러 시작
  - `vm_scheduler_status`: VM 스케줄러 상태/최근 로그 확인
  - `vm_scheduler_stop`: VM 스케줄러 중지

## VM 직접 명령
```bash
cd ~/upbit
bash scripts/vm_scheduler_manager.sh start
bash scripts/vm_scheduler_manager.sh status
bash scripts/vm_scheduler_manager.sh stop
```

## 기본 스케줄 (KST)
- `01:00,05:00,09:00,13:00,17:00,21:00` 코인 자동매매 (`upbit`)
- `01:05,05:05,09:05,13:05,17:05,21:05` 헬스체크 (`health_check`)
- 평일 `09:00` 일일 현황 (`daily_status`)
- 평일 `15:05` 키움 금현물 (`kiwoom_gold`)
- 금요일 `15:10` KIS ISA (`kis_isa`)
- 매월 25~31일 평일 `15:10` KIS 연금저축 (`kis_pension`)

## 로그/상태 파일
- 로그: `logs/vm_scheduler.log`, `logs/vm_scheduler.out.log`
- PID: `logs/vm_scheduler.pid`
- 실행 중복 방지 lock: `logs/vm_scheduler.lock`
- 마지막 실행 상태: `logs/vm_scheduler_state.json`
