import os
import time
import logging
import requests
from dotenv import load_dotenv

logger = logging.getLogger("KISAuth")

class KISAuth:
    """
    한국투자증권 Open API 인증 및 토큰 관리.
    """
    def __init__(self, app_key: str, app_secret: str, base_url: str):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url
        self.access_token = None
        self.token_expiry = 0.0
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json;charset=UTF-8"})

    def _is_token_valid(self) -> bool:
        """토큰 만료 5분 전이면 갱신 필요."""
        return self.access_token is not None and (self.token_expiry - time.time()) > 300

    def ensure_token(self) -> bool:
        if self._is_token_valid():
            return True
        return self.auth()

    def auth(self) -> bool:
        """액세스 토큰 발급. 1분당 1회 제한 처리."""
        if self._is_token_valid():
            return True

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }

        for attempt in range(3):
            try:
                res = self._session.post(url, json=payload, timeout=10)
                data = res.json()

                if data.get("error_code") == "EGW00133":
                    logger.warning("토큰 발급 대기(1분당 1회 제한)... 65초 후 재시도")
                    time.sleep(65)
                    continue

                self.access_token = data.get("access_token")
                if not self.access_token:
                    logger.error(f"토큰 발급 실패: {data}")
                    return False

                expires_in = int(data.get("expires_in", 86400))
                self.token_expiry = time.time() + expires_in
                logger.info(f"KIS 토큰 발급 성공 (만료: {expires_in // 3600}시간)")
                return True

            except Exception as e:
                logger.error(f"KIS 인증 오류: {e}")
                return False

        return False

    def get_headers(self, tr_id: str, extra: dict = None) -> dict:
        h = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }
        if extra:
            h.update(extra)
        return h

    def hashkey(self, body: dict) -> str:
        """주문 요청 전 필요한 hashkey 생성."""
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        try:
            res = self._session.post(url, json=body, headers=headers, timeout=5)
            data = res.json()
            return data.get("HASH", "")
        except Exception as e:
            logger.error(f"hashkey 생성 오류: {e}")
            return ""
