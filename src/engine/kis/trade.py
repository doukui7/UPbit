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

    def get_trade_history(self, start_date: str = "", end_date: str = "",
                          side: str = "00", ccld: str = "00",
                          stock_code: str = "", order_no: str = "",
                          max_rows: int = 50) -> dict:
        """주식 일별 주문 체결 조회 — 기간/필터/페이징 지원 (TTTC8001R).

        Args:
            start_date: 조회 시작일 (YYYYMMDD)
            end_date: 조회 종료일 (YYYYMMDD)
            side: 매도매수 ("00" 전체, "01" 매도, "02" 매수)
            ccld: 체결구분 ("00" 전체, "01" 체결, "02" 미체결)
            stock_code: 종목번호 (빈값=전체)
            order_no: 주문번호 (빈값=전체)
            max_rows: 최대 조회 건수

        Returns:
            {"success": bool, "rows": list[dict], "msg": str, "source": str}
        """
        if not self.auth.ensure_token():
            return {"success": False, "rows": [], "msg": "인증 실패", "source": ""}

        from datetime import datetime
        if not start_date:
            start_date = datetime.now().strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        tr_id = "VTTC8001R" if "VT" in self.base_url else "TTTC8001R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"

        all_rows = []
        ctx_fk = ""
        ctx_nk = ""

        for page in range(20):  # 최대 20페이지
            params = {
                "CANO": self.account_no,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "INQR_STRT_DT": start_date,
                "INQR_END_DT": end_date,
                "SLL_BUY_DVSN_CD": side,
                "INQR_DVSN": "00",
                "PDNO": stock_code,
                "CCLD_DVSN": ccld,
                "ORD_GNO_BRNO": "",
                "ODNO": order_no,
                "INQR_DVSN_3": "00",
                "INQR_DVSN_1": "",
                "CTX_AREA_FK100": ctx_fk,
                "CTX_AREA_NK100": ctx_nk,
            }
            try:
                res = self._session.get(
                    url, params=params,
                    headers=self.auth.get_headers(tr_id),
                    timeout=10,
                )
                data = res.json()
                if data.get("rt_cd") != "0":
                    if page == 0:
                        return {
                            "success": False, "rows": [],
                            "msg": data.get("msg1", "조회 실패"),
                            "source": "KIS API",
                        }
                    break

                rows = data.get("output1", [])
                all_rows.extend(rows)

                # 페이징 종료 조건
                if len(all_rows) >= max_rows:
                    all_rows = all_rows[:max_rows]
                    break
                tr_cont = data.get("tr_cont", "")
                ctx_fk = data.get("ctx_area_fk100", "")
                ctx_nk = data.get("ctx_area_nk100", "")
                if tr_cont not in ("F", "M") or (not ctx_fk and not ctx_nk):
                    break
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"거래내역 조회 오류 (page {page}): {e}")
                if page == 0:
                    return {
                        "success": False, "rows": [],
                        "msg": str(e), "source": "KIS API",
                    }
                break

        return {
            "success": True,
            "rows": all_rows,
            "msg": f"{len(all_rows)}건 조회",
            "source": "KIS API",
        }
