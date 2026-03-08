import logging
import time

logger = logging.getLogger("KISTrade")

class KISTrade:
    """
    Koreainvestment Trading API (Buy, Sell, Cancel, Orders).
    """
    def __init__(self, auth, base_url: str, account_no: str, acnt_prdt_cd: str):
        self.auth = auth
        self.base_url = base_url
        self.account_no = account_no
        self.acnt_prdt_cd = acnt_prdt_cd
        self._session = auth._session

    def send_order(self, order_type: str, stock_code: str, qty: int,
                   price: int = 0, ord_dvsn: str = "01") -> dict | None:
        """
        국내주식/ETF 매수/매도 주문.
        MCP 가이드에 따라 TR_ID 및 파라미터를 최신화함.
        """
        if not self.auth.ensure_token(): return None
        
        is_mock = "VT" in self.base_url
        if order_type.upper() == "BUY":
            tr_id = "VTTC0012U" if is_mock else "TTTC0012U"
        else:
            tr_id = "VTTC0011U" if is_mock else "TTTC0011U"

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        
        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": stock_code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(int(price)) if int(price) > 0 else "0",
            "EXCG_ID_DVSN_CD": "KRX",  # 필수 필드 추가
            "SLL_TYPE": "01",
        }

        # hashkey 생성
        hashkey = self.auth.hashkey(body)
        headers = self.auth.get_headers(tr_id, {"hashkey": hashkey})

        try:
            res = self._session.post(url, json=body, headers=headers, timeout=10)
            data = res.json()
            if data.get("rt_cd") == "0":
                ord_no = data.get("output", {}).get("ODNO", "")
                logger.info(f"[{order_type}] 주문 성공: {stock_code} {qty}주 번호={ord_no}")
                return {"success": True, "ord_no": ord_no, "msg": data.get("msg1")}
            else:
                logger.error(f"[{order_type}] 주문 실패: {data.get('msg1')}")
                return {"success": False, "msg": data.get("msg1")}
        except Exception as e:
            logger.error(f"주문 예외: {e}")
            return None

    def cancel_order(self, org_ord_no: str, stock_code: str, qty: int = 0,
                     cancel_all: bool = True) -> dict | None:
        """주문 정정/취소 (TTTC0803U)."""
        if not self.auth.ensure_token(): return None
        tr_id = "VTTC0803U" if "VT" in self.base_url else "TTTC0803U"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-rvsecncl"
        body = {
            "CANO": self.account_no, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "KRX_FWDG_ORD_ORGNO": "", "ORGN_ODNO": org_ord_no,
            "ORD_DVSN": "00", "RVSE_CNCL_DVSN_CD": "02",
            "ORD_QTY": "0" if cancel_all else str(qty), "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y" if cancel_all else "N",
        }
        headers = self.auth.get_headers(tr_id, {"hashkey": self.auth.hashkey(body)})
        try:
            res = self._session.post(url, json=body, headers=headers, timeout=10)
            rt_cd = res.json().get("rt_cd", "")
            return {"success": rt_cd == "0", "msg": res.json().get("msg1")}
        except Exception as e:
            logger.error(f"취소/정정 예외: {e}")
            return None

    def get_pending_orders(self, stock_code: str = "") -> list:
        """미체결 정정/취소 가능 주문 조회 (TTTC8036R)."""
        if not self.auth.ensure_token(): return []
        tr_id = "VTTC8036R" if "VT" in self.base_url else "TTTC8036R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
        params = {
            "CANO": self.account_no, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "INQR_DVSN_1": "1", "INQR_DVSN_2": "0",
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
        }
        try:
            res = self._session.get(url, params=params, headers=self.auth.get_headers(tr_id), timeout=10)
            data = res.json()
            if data.get("rt_cd") != "0": return []
            
            result = []
            for r in data.get("output", []):
                if stock_code and r.get("pdno") != stock_code: continue
                qty = int(r.get("psbl_qty", 0))
                if qty <= 0: continue
                result.append({
                    "ord_no": r.get("odno"), "side": "SELL" if r.get("sll_buy_dvsn_cd") == "01" else "BUY",
                    "stock_code": r.get("pdno"), "remaining_qty": qty,
                    "price": int(r.get("ord_unpr", 0)), "time": r.get("ord_tmd"),
                })
            return result
        except Exception as e:
            logger.error(f"미체결 조회 오류: {e}")
            return []

    def get_history(self, stock_code: str = "", days: int = 1) -> list:
        """당일 주식 일별 주문 체결 조회 (TTTC8001R)."""
        if not self.auth.ensure_token(): return []
        tr_id = "VTTC8001R" if "VT" in self.base_url else "TTTC8001R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
        
        from datetime import datetime
        dt = datetime.now().strftime("%Y%m%d")
        params = {
            "CANO": self.account_no, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "INQR_STRT_DT": dt, "INQR_END_DT": dt, "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00", "PDNO": stock_code, "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        try:
            res = self._session.get(url, params=params, headers=self.auth.get_headers(tr_id), timeout=10)
            data = res.json()
            if data.get("rt_cd") != "0": return []
            
            result = []
            for r in data.get("output1", []):
                qty = int(r.get("ord_qty", 0))
                ccld_qty = int(r.get("tot_ccld_qty", 0))
                if qty <= 0: continue
                result.append({
                    "ord_no": r.get("odno"), "side": "BUY" if r.get("sll_buy_dvsn_cd") == "02" else "SELL",
                    "stock_code": r.get("pdno"), "qty": qty, "ccld_qty": ccld_qty,
                    "price": int(r.get("ord_unpr", 0)), "time": r.get("ord_tmd"),
                })
            return result
        except Exception as e:
            logger.error(f"체결 내역 조회 오류: {e}")
            return []
