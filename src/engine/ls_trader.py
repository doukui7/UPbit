"""LS증권 미국주식 트레이더 (ebest OpenAPI)."""
import os
import asyncio
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class LSTrader:
    """LS증권 해외주식 매매 클래스."""

    # 거래소 코드
    MARKET_NYSE = "81"
    MARKET_NASDAQ = "82"

    def __init__(self):
        load_dotenv()
        self.app_key = os.getenv("LS_APP_KEY", "")
        self.secret_key = os.getenv("LS_SECRET_KEY", "")
        self.account_no = os.getenv("LS_ACCOUNT_NO", "")
        self.account_pwd = os.getenv("LS_ACCOUNT_PWD", "")
        self.api = None
        self._authenticated = False

    # ── async → sync 래퍼 ──

    @staticmethod
    def _run_async(coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # ── 인증 ──

    def auth(self) -> bool:
        """LS증권 OpenAPI 로그인."""
        if self._authenticated and self.api is not None:
            return True
        try:
            import ebest
            self.api = ebest.OpenApi()
            result = self._run_async(self.api.login(self.app_key, self.secret_key))
            if result:
                self._authenticated = True
                _sim = "모의" if getattr(self.api, 'is_simulation', False) else "실전"
                logger.info(f"[LS] 로그인 성공 ({_sim})")
                return True
            else:
                msg = getattr(self.api, 'last_message', 'unknown')
                logger.error(f"[LS] 로그인 실패: {msg}")
                return False
        except Exception as e:
            logger.error(f"[LS] 로그인 오류: {e}")
            return False

    def _ensure_auth(self) -> bool:
        if not self._authenticated:
            return self.auth()
        return True

    # ── 잔고 조회 ──

    def get_balance(self) -> dict | None:
        """해외증권 잔고 조회 (COSOQ00201)."""
        if not self._ensure_auth():
            return None
        try:
            async def _req():
                return await self.api.request("COSOQ00201", {
                    "COSOQ00201InBlock1": {
                        "RecCnt": 1,
                        "BaseDt": "",
                        "CrcyCode": "USD",
                        "AstkBalTpCode": "00",
                    }
                })
            resp = self._run_async(_req())
            if resp is None:
                return None
            body = resp.body if hasattr(resp, 'body') else resp

            # 원화 예수금
            block2 = body.get("COSOQ00201OutBlock2", {})
            cash_krw = float(block2.get("WonDpsBalAmt", 0) or 0)

            # 보유 종목
            block4 = body.get("COSOQ00201OutBlock4", [])
            if isinstance(block4, dict):
                block4 = [block4]
            holdings = []
            for h in block4:
                qty = float(h.get("AstkBalQty", 0) or 0)
                if qty <= 0:
                    continue
                holdings.append({
                    "code": str(h.get("ShtnIsuNo", "")).strip(),
                    "qty": qty,
                    "sellable_qty": float(h.get("AstkSellAbleQty", 0) or 0),
                    "avg_price": float(h.get("FcstckUprc", 0) or 0),
                    "cur_price": float(h.get("OvrsScrtsCurpri", 0) or 0),
                    "pnl_rate": float(h.get("PnlRat", 0) or 0),
                    "currency": str(h.get("CrcyCode", "USD")).strip(),
                })

            return {"cash_krw": cash_krw, "holdings": holdings}
        except Exception as e:
            logger.error(f"[LS] 잔고 조회 오류: {e}")
            return None

    def get_deposit(self) -> dict | None:
        """해외증권 예수금 상세 (COSOQ02701)."""
        if not self._ensure_auth():
            return None
        try:
            async def _req():
                return await self.api.request("COSOQ02701", {
                    "COSOQ02701InBlock1": {
                        "RecCnt": 1,
                        "CrcyCode": "ALL",
                    }
                })
            resp = self._run_async(_req())
            if resp is None:
                return None
            body = resp.body if hasattr(resp, 'body') else resp

            block3 = body.get("COSOQ02701OutBlock3", [])
            if isinstance(block3, dict):
                block3 = [block3]
            usd_info = next((b for b in block3 if str(b.get("CrcyCode", "")).strip() == "USD"), {})
            usd_available = float(usd_info.get("FcurrOrdAbleAmt", 0) or 0)
            exchange_rate = float(usd_info.get("BaseXchrat", 0) or 0)

            block4 = body.get("COSOQ02701OutBlock4", {})
            krw_deposit = float(block4.get("WonDpsBalAmt", 0) or 0)

            return {
                "usd_available": usd_available,
                "krw_deposit": krw_deposit,
                "exchange_rate": exchange_rate,
            }
        except Exception as e:
            logger.error(f"[LS] 예수금 조회 오류: {e}")
            return None

    # ── 시세 조회 ──

    def get_price(self, symbol: str, market: str = "82") -> dict | None:
        """해외주식 현재가 (g3101)."""
        if not self._ensure_auth():
            return None
        try:
            keysymbol = f"{market}{symbol.upper()}"
            async def _req():
                return await self.api.request("g3101", {
                    "g3101InBlock": {
                        "delaygb": "R",
                        "keysymbol": keysymbol,
                        "exchcd": market,
                        "symbol": symbol.upper(),
                    }
                })
            resp = self._run_async(_req())
            if resp is None:
                return None
            body = resp.body if hasattr(resp, 'body') else resp
            out = body.get("g3101OutBlock", {})
            return {
                "symbol": str(out.get("symbol", symbol)).strip(),
                "name": str(out.get("korname", "")).strip(),
                "price": float(out.get("price", 0) or 0),
                "open": float(out.get("open", 0) or 0),
                "high": float(out.get("high", 0) or 0),
                "low": float(out.get("low", 0) or 0),
                "volume": float(out.get("volume", 0) or 0),
                "change": float(out.get("diff", 0) or 0),
                "change_rate": float(out.get("rate", 0) or 0),
                "currency": str(out.get("currency", "USD")).strip(),
            }
        except Exception as e:
            logger.error(f"[LS] 시세 조회 오류 ({symbol}): {e}")
            return None

    # ── 주문 ──

    def send_order(self, side: str, symbol: str, qty: int, price: float = 0,
                   market: str = "82", price_type: str = "00") -> dict | None:
        """해외주식 주문 (COSAT00301).

        Args:
            side: "BUY" or "SELL"
            symbol: 종목코드 (예: "TSLA")
            qty: 수량
            price: 가격 (시장가=0)
            market: "81"=NYSE, "82"=NASDAQ
            price_type: "00"=지정가, "03"=시장가
        """
        if not self._ensure_auth():
            return None
        ord_ptn = "02" if side.upper() == "BUY" else "01"
        try:
            async def _req():
                return await self.api.request("COSAT00301", {
                    "COSAT00301InBlock1": {
                        "RecCnt": 1,
                        "OrdPtnCode": ord_ptn,
                        "OrgOrdNo": 0,
                        "OrdMktCode": market,
                        "IsuNo": symbol.upper(),
                        "OrdQty": qty,
                        "OvrsOrdPrc": price,
                        "OrdprcPtnCode": price_type,
                        "BrkTpCode": "",
                    }
                })
            resp = self._run_async(_req())
            if resp is None:
                return {"success": False, "msg": "API 응답 없음"}
            body = resp.body if hasattr(resp, 'body') else resp
            rsp_cd = str(body.get("rsp_cd", "")).strip()
            rsp_msg = str(body.get("rsp_msg", "")).strip()
            success = rsp_cd == "0" or rsp_cd.startswith("0")
            logger.info(f"[LS] 주문 {side} {symbol} x{qty} @{price}: {rsp_cd} {rsp_msg}")
            return {"success": success, "msg": rsp_msg, "rsp_cd": rsp_cd, "raw": body}
        except Exception as e:
            logger.error(f"[LS] 주문 오류 ({side} {symbol}): {e}")
            return {"success": False, "msg": str(e)}

    def cancel_order(self, ord_no: int, symbol: str = "",
                     market: str = "82") -> dict | None:
        """해외주식 주문 취소 (COSAT00301, OrdPtnCode=08)."""
        if not self._ensure_auth():
            return None
        try:
            async def _req():
                return await self.api.request("COSAT00301", {
                    "COSAT00301InBlock1": {
                        "RecCnt": 1,
                        "OrdPtnCode": "08",
                        "OrgOrdNo": ord_no,
                        "OrdMktCode": market,
                        "IsuNo": symbol.upper() if symbol else "",
                        "OrdQty": 0,
                        "OvrsOrdPrc": 0,
                        "OrdprcPtnCode": "",
                        "BrkTpCode": "",
                    }
                })
            resp = self._run_async(_req())
            if resp is None:
                return {"success": False, "msg": "API 응답 없음"}
            body = resp.body if hasattr(resp, 'body') else resp
            rsp_cd = str(body.get("rsp_cd", "")).strip()
            rsp_msg = str(body.get("rsp_msg", "")).strip()
            success = rsp_cd == "0" or rsp_cd.startswith("0")
            logger.info(f"[LS] 취소 #{ord_no}: {rsp_cd} {rsp_msg}")
            return {"success": success, "msg": rsp_msg}
        except Exception as e:
            logger.error(f"[LS] 취소 오류 (#{ord_no}): {e}")
            return {"success": False, "msg": str(e)}

    def modify_order(self, ord_no: int, new_price: float) -> dict | None:
        """해외주식 주문 정정 (COSAT00311)."""
        if not self._ensure_auth():
            return None
        try:
            async def _req():
                return await self.api.request("COSAT00311", {
                    "COSAT00311InBlock1": {
                        "RecCnt": 1,
                        "OrdPtnCode": "07",
                        "OrgOrdNo": ord_no,
                        "OrdMktCode": "",
                        "IsuNo": "",
                        "OrdQty": 0,
                        "OvrsOrdPrc": new_price,
                        "OrdprcPtnCode": "",
                        "BrkTpCode": "",
                    }
                })
            resp = self._run_async(_req())
            if resp is None:
                return {"success": False, "msg": "API 응답 없음"}
            body = resp.body if hasattr(resp, 'body') else resp
            rsp_cd = str(body.get("rsp_cd", "")).strip()
            rsp_msg = str(body.get("rsp_msg", "")).strip()
            success = rsp_cd == "0" or rsp_cd.startswith("0")
            return {"success": success, "msg": rsp_msg}
        except Exception as e:
            logger.error(f"[LS] 정정 오류 (#{ord_no}): {e}")
            return {"success": False, "msg": str(e)}

    # ── 미체결 조회 ──

    def get_pending_orders(self, market: str = "82") -> list:
        """해외주식 미체결 조회 (COSAQ00102)."""
        if not self._ensure_auth():
            return []
        try:
            async def _req():
                return await self.api.request("COSAQ00102", {
                    "COSAQ00102InBlock1": {
                        "RecCnt": 1,
                        "QryTpCode": "1",
                        "BkseqTpCode": "1",
                        "OrdMktCode": market,
                        "BnsTpCode": "0",
                        "IsuNo": "",
                        "SrtOrdNo": 9999999999,
                        "OrdDt": "",
                        "ExecYn": "2",
                        "CrcyCode": "USD",
                        "ThdayBnsAppYn": "0",
                        "LoanBalHldYn": "0",
                    }
                })
            resp = self._run_async(_req())
            if resp is None:
                return []
            body = resp.body if hasattr(resp, 'body') else resp
            block3 = body.get("COSAQ00102OutBlock3", [])
            if isinstance(block3, dict):
                block3 = [block3]
            results = []
            for o in block3:
                unfilled = int(o.get("UnercQty", 0) or 0)
                if unfilled <= 0:
                    continue
                ptn = str(o.get("OrdPtnCode", "")).strip()
                results.append({
                    "ord_no": int(o.get("OrdNo", 0) or 0),
                    "symbol": str(o.get("ShtnIsuNo", "")).strip(),
                    "side": "BUY" if ptn == "02" else "SELL",
                    "qty": int(o.get("OrdQty", 0) or 0),
                    "price": float(o.get("OvrsOrdPrc", 0) or 0),
                    "unfilled_qty": unfilled,
                    "ord_time": str(o.get("OrdTime", "")).strip(),
                })
            return results
        except Exception as e:
            logger.error(f"[LS] 미체결 조회 오류: {e}")
            return []

    # ── 정리 ──

    def close(self):
        """연결 종료."""
        if self.api is not None:
            try:
                self._run_async(self.api.close())
            except Exception:
                pass
            self.api = None
            self._authenticated = False
