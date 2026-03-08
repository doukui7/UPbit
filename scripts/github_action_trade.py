"""자동매매 디스패처.

TRADING_MODE 환경변수로 실행할 엔진을 선택한다.
각 엔진 로직은 trade_lib/ 패키지에 모듈별로 분리되어 있다.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# scripts/ 상위(프로젝트 루트)를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


if __name__ == "__main__":
    load_dotenv()
    mode = os.getenv("TRADING_MODE", "upbit").lower()

    try:
        if mode == "kiwoom_gold":
            from trade_lib.gold_engine import run_kiwoom_gold_trade
            run_kiwoom_gold_trade()

        elif mode == "kis_isa":
            from trade_lib.kis_isa_engine import run_kis_isa_trade
            run_kis_isa_trade()

        elif mode == "kis_pension":
            from trade_lib.kis_pension_engine import run_kis_pension_trade
            run_kis_pension_trade()

        elif mode in ("health_check", "health_check_upbit"):
            from trade_lib.checks import run_health_check
            run_health_check()

        elif mode == "daily_status":
            from trade_lib.reports import run_daily_status_report
            run_daily_status_report()

        elif mode == "account_sync":
            from trade_lib.reports import run_account_sync
            run_account_sync()

        elif mode == "manual_order":
            from trade_lib.reports import run_manual_order
            run_manual_order()

        elif mode == "telegram_test_ping":
            from trade_lib.reports import run_telegram_test_ping
            run_telegram_test_ping()

        else:
            from trade_lib.upbit_engine import run_auto_trade
            run_auto_trade()

    except Exception as e:
        logger.exception(f"치명적 예외 발생(mode={mode}): {e}")
        try:
            from trade_lib.notifier import send_telegram
            send_telegram(
                f"<b>자동매매 실행 실패</b>\n{mode}: {type(e).__name__} - {str(e)[:200]}"
            )
        except Exception:
            pass
        raise
