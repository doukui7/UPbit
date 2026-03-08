"""공유 상수 및 파일 경로."""
import os
from datetime import timezone, timedelta

# ── 프로젝트 루트 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 타임존 ──
KST = timezone(timedelta(hours=9))

# ── 주문 금액 제한 ──
MIN_ORDER_KRW = 5000
UPBIT_TEST_MIN_KRW = 5100
UPBIT_TEST_BUY_MULTIPLIER = 0.5    # 현재가 -50%
UPBIT_TEST_SELL_MULTIPLIER = 2.5   # 현재가 +150%

# ── ETF 코드 ──
ISA_WDR_TRADE_ETF_CODES = {"418660", "409820", "423920", "426030", "465610", "461910"}
GOLD_KRX_ETF_CODE = "411060"
GOLD_LEGACY_ETF_CODES = {"132030"}

# ── 상태/캐시 파일 경로 ──
SIGNAL_STATE_FILE = os.path.join(PROJECT_ROOT, "signal_state.json")
BALANCE_CACHE_FILE = os.path.join(PROJECT_ROOT, "balance_cache.json")
ACCOUNT_CACHE_FILE = os.path.join(PROJECT_ROOT, "account_cache.json")
TRADE_LOG_FILE = os.path.join(PROJECT_ROOT, "trade_log.json")
SIGNAL_TEST_ORDERS_FILE = os.path.join(PROJECT_ROOT, "signal_test_orders.json")

# ── 설정 ──
SIGNAL_TEST_CANCEL_AFTER_MIN = 30
TRADE_LOG_MAX_ENTRIES = 200
