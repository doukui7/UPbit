import os
import logging
from dotenv import load_dotenv

from .kis.auth import KISAuth
from .kis.quota import KISQuota
from .kis.account import KISAccount
from .kis.trade import KISTrade

logger = logging.getLogger("KISTrader")

class KISTrader:
    """
    한국투자증권(KIS) 통합 트레이더 클래스.
    인증, 시세, 계좌, 주문 기능을 각 전문 모듈에 위임합니다.
    """
    def __init__(self, is_mock: bool = False, app_key="", app_secret="", cano="", prdt=""):
        load_dotenv()
        
        # 환경 변수 로드
        self.is_mock = is_mock
        app_key = app_key or (os.getenv("KIS_MOCK_APP_KEY") if is_mock else os.getenv("KIS_APP_KEY")) or ""
        app_secret = app_secret or (os.getenv("KIS_MOCK_APP_SECRET") if is_mock else os.getenv("KIS_APP_SECRET")) or ""
        cano = cano or (os.getenv("KIS_MOCK_CANO") if is_mock else os.getenv("KIS_CANO")) or ""
        prdt = prdt or (os.getenv("KIS_MOCK_ACNT_PRDT_CD", "01") if is_mock else os.getenv("KIS_ACNT_PRDT_CD", "01")) or "01"

        self.base_url = "https://openapivts.koreainvestment.com:29443" if is_mock else "https://openapi.koreainvestment.com:9443"
        
        # 1. 인증 모듈
        self._auth = KISAuth(app_key, app_secret, self.base_url)
        
        # 2. 시세 모듈
        self.quota = KISQuota(self._auth, self.base_url)
        
        # 3. 계좌 모듈
        self.account_module = KISAccount(self._auth, self.base_url, cano, prdt) # avoid naming conflict with account_no
        self.account = self.account_module # alias
        
        # 4. 주문 모듈
        self.trade = KISTrade(self._auth, self.base_url, cano, prdt)

    def auth(self) -> bool:
        """기존 코드 호환용: 토큰 유효성 확인."""
        return self._auth.ensure_token()

    @property
    def access_token(self):
        return self._auth.access_token

    @access_token.setter
    def access_token(self, value):
        self._auth.access_token = value

    @property
    def token_expiry(self):
        return self._auth.token_expiry

    @token_expiry.setter
    def token_expiry(self, value):
        self._auth.token_expiry = value

    @property
    def app_key(self): return self._auth.app_key
    @app_key.setter
    def app_key(self, value): self._auth.app_key = value

    @property
    def app_secret(self): return self._auth.app_secret
    @app_secret.setter
    def app_secret(self, value): self._auth.app_secret = value

    @property
    def account_no(self): return self.account_module.account_no
    @account_no.setter
    def account_no(self, value):
        self.account_module.account_no = value
        self.trade.account_no = value

    @property
    def acnt_prdt_cd(self): return self.account_module.acnt_prdt_cd
    @acnt_prdt_cd.setter
    def acnt_prdt_cd(self, value):
        self.account_module.acnt_prdt_cd = value
        self.trade.acnt_prdt_cd = value

    # --- 구버전 호환용 래퍼 메서드 ---
    def get_current_price(self, stock_code: str) -> float | None:
        return self.quota.get_current_price(stock_code)

    def get_balance(self) -> dict | None:
        return self.account.get_balance()

    def send_order(self, order_type: str, stock_code: str, qty: int,
                   price: int = 0, ord_dvsn: str = "01") -> dict | None:
        return self.trade.send_order(order_type, stock_code, qty, price, ord_dvsn)

    def cancel_order(self, org_ord_no: str, stock_code: str, qty: int = 0) -> dict | None:
        return self.trade.cancel_order(org_ord_no, stock_code, qty)

    # 차트 등 기타는 필요시 직접 quota 호출
    def get_daily_chart(self, stock_code: str, count: int = 200):
        return self.quota.get_daily_chart(stock_code, count=count)
    
    def get_pending_orders(self, stock_code: str = ""):
        return self.trade.get_pending_orders(stock_code)

    def execute_closing_auction_buy(self, stock_code: str, qty: int) -> dict | None:
        """동시호가 시장가 매수 (01)"""
        return self.send_order("BUY", stock_code, qty, price=0, ord_dvsn="01")

    def execute_closing_auction_sell(self, stock_code: str, qty: int) -> dict | None:
        """동시호가 시장가 매도 (01)"""
        return self.send_order("SELL", stock_code, qty, price=0, ord_dvsn="01")

    def get_orderable_cash(self, stock_code: str, price: int = 0, ord_dvsn: str = "01") -> float | None:
        """주문 가능 현금 조회."""
        return self.account.get_orderable_cash(stock_code, price, ord_dvsn)

    def _get_limit_price(self, stock_code: str, order_type: str) -> int:
        """동시호가 체결 보장을 위한 상한가(BUY)/하한가(SELL) 계산."""
        price = self.get_current_price(stock_code)
        if not price or price <= 0:
            return 0
        # 호가 단위 (KRX 규정)
        def _tick(p):
            if p < 2000: return 1
            if p < 5000: return 5
            if p < 20000: return 10
            if p < 50000: return 50
            if p < 200000: return 100
            if p < 500000: return 500
            return 1000
        if order_type == "BUY":
            raw = int(price * 1.30)
            t = _tick(raw)
            return (raw // t) * t
        else:
            raw = int(price * 0.70)
            t = _tick(raw)
            return ((raw + t - 1) // t) * t

    def smart_buy_krw(self, stock_code: str, target_krw: float) -> dict | None:
        """목표 금액(KRW) 만큼 시장가 매수"""
        cur_price = self.get_current_price(stock_code)
        if not cur_price or cur_price <= 0:
            return {"success": False, "msg": "현재가 조회 실패"}
        qty = int(target_krw // cur_price)
        if qty <= 0:
            return {"success": False, "msg": f"목표 금액({target_krw}원)이 너무 적습니다 (현재가 {cur_price}원)"}
        return self.send_order("BUY", stock_code, qty, price=0, ord_dvsn="01")

    def smart_sell_all(self, stock_code: str) -> dict | None:
        """전량 시장가 매도"""
        bal = self.get_balance()
        if not bal:
            return {"success": False, "msg": "잔고 조회 실패"}
        qty = 0
        for h in bal.get("holdings", []):
            if h["code"] == stock_code:
                qty = h["qty"]
                break
        if qty <= 0:
            return {"success": False, "msg": "보유 수량 없음"}
        return self.send_order("SELL", stock_code, qty, price=0, ord_dvsn="01")
