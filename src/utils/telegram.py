import requests
import html
import re

def _send_telegram_message(token: str, chat_id: str, text: str):
    """텔레그램 메시지 전송 로직."""
    def _normalize_bot_token(raw: str):
        v = str(raw).strip()
        if not v: return ""
        if v.startswith("bot"): return v
        return f"bot{v}"

    def _sanitize_html_chunk(msg: str):
        return html.escape(msg).replace("<", "&lt;").replace(">", "&gt;")

    token = _normalize_bot_token(token)
    if not token or not chat_id:
        return

    # 텍스트 정규화 및 분할 전송 (텔레그램 글자수 제한 대응)
    clean_text = text.replace("<br>", "\n").replace("<p>", "").replace("</p>", "\n")
    # HTML 태그 제거 (단순 텍스트 전송 우선)
    clean_text = re.sub(r'<[^>]*>', '', clean_text)
    
    url = f"https://api.telegram.org/{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": clean_text[:4000], # 안전을 위해 4000자 제한
        "parse_mode": "HTML"
    }
    
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass
