"""텔레그램 알림 전송."""
import os
import json
import re
import logging

from .constants import PROJECT_ROOT
from .utils import sanitize_html, load_user_config, get_env_any

logger = logging.getLogger(__name__)


def _normalize_bot_token(raw: str) -> str:
    """텔레그램 봇 토큰 정규화 (URL/따옴표/공백 등 정리)."""
    t = str(raw or "").strip().strip('"').strip("'")
    t = re.sub(r"\s+", "", t)
    if "api.telegram.org" in t and "/bot" in t:
        m = re.search(r"/bot([^/\\s]+)", t)
        if m:
            t = m.group(1).strip()
    if t.lower().startswith("bot"):
        t = t[3:]
    if ":" in t:
        left, right = t.split(":", 1)
        left = "".join(ch for ch in left if ch.isdigit())
        right = "".join(ch for ch in right if re.match(r"[A-Za-z0-9_-]", ch))
        t = f"{left}:{right}" if left and right else t
    m2 = re.search(r"([0-9]{6,}:[A-Za-z0-9_-]{20,})", t)
    if m2:
        t = m2.group(1)
    return t.strip().strip('"').strip("'")


def send_telegram(message: str):
    """텔레그램 봇으로 메시지 전송. 토큰/챗ID 없으면 무시."""
    token = get_env_any("TELEGRAM_BOT_TOKEN", "telegram_bot_token")
    chat_id = get_env_any("TELEGRAM_CHAT_ID", "telegram_chat_id")
    if not token or not chat_id:
        cfg = load_user_config()
        token = token or str(cfg.get("telegram_bot_token", "")).strip()
        chat_id = chat_id or str(cfg.get("telegram_chat_id", "")).strip()
    if not token or not chat_id:
        _common_path = os.path.join(PROJECT_ROOT, "config", "common.json")
        if os.path.exists(_common_path):
            try:
                with open(_common_path, "r", encoding="utf-8") as _f:
                    _common = json.load(_f)
                token = token or str(_common.get("telegram_bot_token", "")).strip()
                chat_id = chat_id or str(_common.get("telegram_chat_id", "")).strip()
            except Exception:
                pass
    if not token or not chat_id:
        logger.warning("텔레그램 설정이 없어 알림 전송을 생략합니다. (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)")
        return

    token = _normalize_bot_token(token)
    chat_id = str(chat_id).strip().strip('"').strip("'")
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        import requests
        for i in range(0, len(message), 4000):
            chunk = message[i:i+4000]
            safe_chunk = sanitize_html(chunk)
            resp = requests.post(url, json={
                "chat_id": chat_id,
                "text": safe_chunk,
                "parse_mode": "HTML",
            }, timeout=10)
            ok = False
            desc = ""
            try:
                body = resp.json()
                ok = bool(resp.ok and body.get("ok", False))
                desc = str(body.get("description", ""))
            except Exception:
                ok = bool(resp.ok)
                desc = resp.text[:200] if getattr(resp, "text", "") else ""

            if ok:
                logger.info("텔레그램 전송 성공")
            else:
                logger.warning(f"텔레그램 전송 실패: HTTP {resp.status_code} {desc}")
    except Exception as e:
        logger.warning(f"텔레그램 전송 실패: {e}")
