import logging
import requests

logger = logging.getLogger("KISAccount")

class KISAccount:
    """
    Koreainvestment Account Balance and Holdings API.
    """
    def __init__(self, auth, base_url: str, account_no: str, acnt_prdt_cd: str):
        self.auth = auth
        self.base_url = base_url
        self.account_no = account_no
        self.acnt_prdt_cd = acnt_prdt_cd
        self._session = auth._session

    def get_balance(self) -> dict | None:
        """종합계좌 잔고 조회 (TTTC8434R)."""
        if not self.auth.ensure_token(): return None
        tr_id = "VTTC8434R" if "VT" in self.base_url else "TTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        params = {
            "CANO": self.account_no, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
            "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
        }

        try:
            res = self._session.get(url, params=params, headers=self.auth.get_headers(tr_id), timeout=10)
            data = res.json()
            if data.get("rt_cd") != "0":
                logger.error(f"잔고 조회 오류: {data.get('msg1')}")
                return None

            holdings = []
            for item in data.get("output1", []):
                qty = int(float(str(item.get("hldg_qty", 0)).replace(",", "")))
                if qty <= 0: continue
                holdings.append({
                    "code": item.get("pdno"), "name": item.get("prdt_name"), "qty": qty,
                    "avg_price": float(str(item.get("pchs_avg_pric", 0)).replace(",", "")),
                    "cur_price": float(str(item.get("prpr", 0)).replace(",", "")),
                    "eval_amt": float(str(item.get("evlu_amt", 0)).replace(",", "")),
                    "pnl_rate": float(str(item.get("evlu_pfls_rt", 0)).replace(",", "")),
                })

            summary = data.get("output2", [{}])[0]
            cash = float(str(summary.get("dnca_tot_amt", 0)).replace(",", ""))
            buyable_cash = float(str(summary.get("ord_psbl_cash", cash)).replace(",", ""))
            total_eval = float(str(summary.get("tot_asst_amt", 0)).replace(",", ""))
            stock_eval = float(str(summary.get("scts_evlu_amt", sum(h["eval_amt"] for h in holdings))).replace(",", ""))
            if total_eval <= 0:
                total_eval = cash + stock_eval

            return {
                "cash": cash, "buyable_cash": buyable_cash, "total_eval": total_eval,
                "stock_eval": stock_eval, "holdings": holdings,
            }
        except Exception as e:
            logger.error(f"잔고 조회 예외: {e}")
            return None

    def get_orderable_cash(self, stock_code: str, price: int = 0, ord_dvsn: str = "01") -> float | None:
        """주문 가능 현금 조회 (TTTC8908R)."""
        if not self.auth.ensure_token(): return None
        tr_id = "VTTC8908R" if "VT" in self.base_url else "TTTC8908R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        params = {
            "CANO": self.account_no, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": stock_code, "ORD_UNPR": str(int(price)), "ORD_DVSN": ord_dvsn,
            "CMA_EVLU_AMT_ICLD_YN": "N", "OVRS_ICLD_YN": "N",
        }
        try:
            res = self._session.get(url, params=params, headers=self.auth.get_headers(tr_id), timeout=10)
            out = res.json().get("output", {})
            return float(str(out.get("ord_psbl_cash", 0)).replace(",", ""))
        except Exception as e:
            logger.warning(f"주문가능금액 조회 예외 ({stock_code}): {e}")
            return None
