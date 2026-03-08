"""공용 유틸리티 함수."""
import os
import json
import math
import html
import re
import time
import logging
from datetime import datetime, timedelta

from .constants import (
    PROJECT_ROOT, KST,
    ISA_WDR_TRADE_ETF_CODES, GOLD_KRX_ETF_CODE, GOLD_LEGACY_ETF_CODES,
)

logger = logging.getLogger(__name__)


# ── 타입 변환 / 포맷팅 ──────────────────────────────────

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def fmt_krw_price(value: float | None) -> str:
    if value is None:
        return "N/A"
    v = safe_float(value, default=0.0)
    if not math.isfinite(v) or v <= 0:
        return "N/A"
    return f"{v:,.0f}원"


def calc_gap_pct(current_price: float | None, target_price: float | None) -> float | None:
    cur = safe_float(current_price, default=0.0)
    tgt = safe_float(target_price, default=0.0)
    if not math.isfinite(cur) or not math.isfinite(tgt) or cur <= 0 or tgt <= 0:
        return None
    return (cur / tgt - 1.0) * 100.0


def fmt_gap_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    v = safe_float(value, default=float("nan"))
    if not math.isfinite(v):
        return "N/A"
    return f"{v:+.2f}%"


def valid_price(value: float | None) -> bool:
    v = safe_float(value, default=0.0)
    return math.isfinite(v) and v > 0


def abs_gap(value: float | None) -> float:
    if value is None:
        return float("inf")
    v = safe_float(value, default=float("nan"))
    if not math.isfinite(v):
        return float("inf")
    return abs(v)


# ── 업비트 호가 단위 ────────────────────────────────────

def upbit_tick_size(price: float) -> float:
    p = safe_float(price, default=0.0)
    if p >= 2000000: return 1000.0
    if p >= 1000000: return 500.0
    if p >= 500000:  return 100.0
    if p >= 100000:  return 50.0
    if p >= 10000:   return 10.0
    if p >= 1000:    return 5.0
    if p >= 100:     return 1.0
    if p >= 10:      return 0.1
    if p >= 1:       return 0.01
    return 0.001


def round_upbit_price(price: float, mode: str = "floor") -> float:
    p = safe_float(price, default=0.0)
    if not math.isfinite(p) or p <= 0:
        return 0.0
    tick = upbit_tick_size(p)
    ratio = p / tick
    if mode == "ceil":
        q = math.ceil(ratio - 1e-12)
    elif mode == "nearest":
        q = round(ratio)
    else:
        q = math.floor(ratio + 1e-12)
    out = q * tick
    if tick >= 1:   return float(int(round(out)))
    if tick >= 0.1:  return round(out, 1)
    if tick >= 0.01: return round(out, 2)
    return round(out, 3)


def ceil_volume_8(volume: float) -> float:
    v = safe_float(volume, default=0.0)
    if not math.isfinite(v) or v <= 0:
        return 0.0
    return math.ceil(v * 1e8) / 1e8


# ── 기타 유틸 ────────────────────────────────────────────

def extract_order_error_text(order_resp) -> str:
    if order_resp is None:
        return "응답 없음(None)"
    if isinstance(order_resp, dict):
        err = order_resp.get("error")
        if err:
            return str(err)
    return ""


def mask_secret(value: str, left: int = 4, right: int = 4) -> str:
    s = str(value or "").strip()
    if not s:
        return "(empty)"
    if len(s) <= left + right:
        return "*" * len(s)
    return f"{s[:left]}...{s[-right:]}"


def sanitize_html(text: str) -> str:
    escaped = html.escape(str(text), quote=False)
    escaped = escaped.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    escaped = escaped.replace("&lt;code&gt;", "<code>").replace("&lt;/code&gt;", "</code>")
    escaped = escaped.replace("&lt;pre&gt;", "<pre>").replace("&lt;/pre&gt;", "</pre>")
    return escaped


def sanitize_isa_trade_etf(code: str, default: str = "418660") -> str:
    raw = str(code or "").strip()
    c = raw.split()[0] if raw else ""
    return c if c in ISA_WDR_TRADE_ETF_CODES else str(default)


def normalize_gold_kr_etf(code: str, default: str = GOLD_KRX_ETF_CODE) -> str:
    raw = str(code or "").strip()
    c = raw.split()[0] if raw else ""
    if c in GOLD_LEGACY_ETF_CODES:
        return str(default)
    return c or str(default)


def load_user_config():
    cfg_path = os.path.join(PROJECT_ROOT, "user_config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_env_any(*keys, default=""):
    for key in keys:
        val = os.getenv(key, "")
        if str(val).strip():
            return val
    return default


def normalize_kis_account_fields(account_no: str, prdt_cd: str = "01") -> tuple[str, str]:
    raw_acct = "".join(ch for ch in str(account_no or "") if ch.isdigit())
    raw_prdt = "".join(ch for ch in str(prdt_cd or "") if ch.isdigit())
    if len(raw_acct) >= 10:
        cano = raw_acct[:8]
        acnt_prdt_cd = raw_acct[8:10]
    else:
        cano = raw_acct[:8] if len(raw_acct) > 8 else raw_acct
        acnt_prdt_cd = raw_prdt
    if not acnt_prdt_cd:
        acnt_prdt_cd = "01"
    return cano, acnt_prdt_cd.zfill(2)[:2]


# ── 코인 주기 관련 ──────────────────────────────────────

def normalize_coin_interval(interval: str) -> str:
    iv = str(interval or "day").strip().lower()
    if iv in {"1d", "d", "day", "daily"}:    return "day"
    if iv in {"4h", "240", "240m", "minute240"}: return "minute240"
    if iv in {"1h", "60", "60m", "minute60"}:   return "minute60"
    return iv


def is_coin_interval_due(interval: str, now_kst) -> bool:
    if os.getenv("FORCE_TRADE", "").strip() in ("1", "true", "yes"):
        return True
    iv = normalize_coin_interval(interval)
    hour = int(now_kst.hour)
    minute = int(now_kst.minute)
    due_window_min = 55
    if iv == "day":
        return hour == 9 and minute <= due_window_min
    if iv == "minute240":
        return hour in (1, 5, 9, 13, 17, 21) and minute <= due_window_min
    return True


def wait_for_candle_boundary(now_kst, max_wait_sec: int = 720):
    boundary_hours = [1, 5, 9, 13, 17, 21]
    for bh in boundary_hours:
        target = now_kst.replace(hour=bh, minute=0, second=5, microsecond=0)
        if target > now_kst:
            wait_sec = (target - now_kst).total_seconds()
            if 0 < wait_sec <= max_wait_sec:
                logger.info(f"캔들 마감 대기: {wait_sec:.0f}초 후 {bh:02d}:00 KST 실행")
                time.sleep(wait_sec)
                return datetime.now(KST)
            break
    if now_kst.hour >= 21:
        tomorrow = now_kst + timedelta(days=1)
        target = tomorrow.replace(hour=1, minute=0, second=5, microsecond=0)
        wait_sec = (target - now_kst).total_seconds()
        if 0 < wait_sec <= max_wait_sec:
            logger.info(f"캔들 마감 대기: {wait_sec:.0f}초 후 01:00 KST 실행")
            time.sleep(wait_sec)
            return datetime.now(KST)
    logger.info(f"캔들 마감 경과, 즉시 실행 (현재: {now_kst.strftime('%H:%M')} KST)")
    return datetime.now(KST)


# ── 포맷 / 헬스체크 헬퍼 ────────────────────────────────

def fmt_interval_label(interval: str) -> str:
    iv = normalize_coin_interval(interval)
    return {"day": "1D", "minute240": "4H", "minute60": "1H",
            "minute30": "30m", "minute15": "15m", "minute5": "5m", "minute1": "1m"}.get(iv, iv)


def cond_text(flag: bool | None) -> str:
    if flag is None: return "판단불가"
    return "조건충족" if flag else "조건미충족"


def append_step(result: dict, step: str, status: str, detail: str):
    st = str(status or "INFO").upper()
    detail_text = str(detail or "")
    if len(detail_text) > 280:
        detail_text = detail_text[:280] + "...(truncated)"
    line = f"{step} [{st}] {detail_text}"
    result.setdefault("steps", []).append(line)
    logger.info(f"[{result.get('name', '헬스체크')}] {line}")
