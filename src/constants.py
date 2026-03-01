import os
from dotenv import load_dotenv

load_dotenv()

# --- ETF 및 거래 관련 상수 (Source of Truth) ---
ETF_NAME_KR = {
    "418660": "TIGER 미국나스닥100레버리지(합성)",
    "409820": "KODEX 미국나스닥100레버리지(합성 H)",
    "423920": "TIGER 미국필라델피아반도체레버리지(합성)",
    "465610": "ACE 미국빅테크TOP7 Plus레버리지(합성)",
    "461910": "PLUS 미국테크TOP10레버리지(합성)",
    "426030": "TIMEFOLIO 미국나스닥100액티브",
    "133690": "TIGER 미국나스닥100",
    "360750": "TIGER 미국S&P500",
    "132030": "KODEX Gold선물(H)",
    "453540": "TIGER 미국채10년선물",
    "114470": "KODEX 국고채3년",
    "453850": "TIGER 선진국MSCI World",
    "251350": "KODEX 선진국MSCI World",
    "308620": "KODEX 미국채10년선물",
    "471460": "ACE 미국30년국채액티브",
}

ISA_WDR_TRADE_ETF_CODES = ("418660", "409820", "423920", "426030", "465610", "461910")
ISA_WDR_TRADE_ETF_SET = set(ISA_WDR_TRADE_ETF_CODES)

# --- 업비트 관련 상수 ---
TOP_20_TICKERS = [
    "KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-ZEREBRO", 
    "KRW-DOGE", "KRW-VTHO", "KRW-STX", "KRW-SEI", "KRW-TRX",
    "KRW-ADA", "KRW-AVAX", "KRW-ETC", "KRW-SHIB", "KRW-ALGO", 
    "KRW-LINK", "KRW-SUI", "KRW-NEO", "KRW-HBAR", "KRW-ANKR"
]

# --- 기술적 분석 및 백테스트 상수 ---
INTERVAL_MAP = {
    "1분": "1m", "3분": "3m", "5분": "5m", "10분": "10m", "15분": "15m",
    "30분": "30m", "1시간": "1h", "4시간": "4h", "1일": "1D", "1주": "1W"
}

# UI용 역방향 맵
INTERVAL_REV_MAP = {
    "minute1": "1m", "minute3": "3m", "minute5": "5m", "minute10": "10m", "minute15": "15m",
    "minute30": "30m", "minute60": "1h", "minute240": "4h", "day": "1D", "week": "1W",
    "1m": "1m", "3m": "3m", "5m": "5m", "10m": "10m", "15m": "15m",
    "30m": "30m", "1h": "1h", "4h": "4h", "1D": "1D", "1W": "1W"
}

CANDLES_PER_DAY = {
    "1m": 1440, "3m": 480, "5m": 288, "10m": 144, "15m": 96,
    "30m": 48, "1h": 24, "4h": 6, "1D": 1, "1W": 0.14
}

# --- 경로 및 환경 설정 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
DATA_DIR = os.path.join(BASE_DIR, "data")

CONFIG_FILE = "user_config.json"
IS_CLOUD = os.path.exists("/mount/src") or "streamlit.app" in os.getenv("HOSTNAME", "")

GOLD_TICK = 5  # KRX 금시장 호가 단위
