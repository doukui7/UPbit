import time
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger("KISQuota")

class KISQuota:
    """
    Koreainvestment Quotations API handling.
    """
    def __init__(self, auth, base_url: str):
        self.auth = auth
        self.base_url = base_url
        self._session = auth._session

    # --- Domestic Stocks ---
    def get_current_price(self, stock_code: str) -> float | None:
        """국내주식/ETF 현재가 조회 (FHKST01010100)."""
        if not self.auth.ensure_token():
            return None

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }

        for _retry in range(3):
            try:
                res = self._session.get(
                    url,
                    params=params,
                    headers=self.auth.get_headers("FHKST01010101" if "VT" in self.base_url else "FHKST01010100"),
                    timeout=5,
                )
                data = res.json()
                price = float(data.get("output", {}).get("stck_prpr", 0))
                return price if price > 0 else None
            except Exception as e:
                logger.warning(f"현재가 조회 시시도 {_retry + 1}/3 ({stock_code}): {e}")
                time.sleep(0.5)
        return None

    def get_price_info(self, stock_code: str) -> dict | None:
        """국내주식/ETF 현재가 상세 조회."""
        if not self.auth.ensure_token():
            return None

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }

        try:
            res = self._session.get(url, params=params,
                                    headers=self.auth.get_headers("FHKST01010100"), timeout=5)
            o = res.json().get("output", {})
            return {
                'cur_prc': float(o.get("stck_prpr", 0)),
                'open_pric': float(o.get("stck_oprc", 0)),
                'high_pric': float(o.get("stck_hgpr", 0)),
                'low_pric': float(o.get("stck_lwpr", 0)),
                'prev_close': float(o.get("stck_sdpr", 0)),
                'volume': float(o.get("acml_vol", 0)),
                'change_rate': float(o.get("prdy_ctrt", 0)),
            }
        except Exception as e:
            logger.error(f"주변정보 조회 오류 ({stock_code}): {e}")
            return None

    def get_daily_chart(self, stock_code: str, start_date: str = None,
                        end_date: str = None, count: int = 200) -> pd.DataFrame | None:
        """국내주식/ETF 일봉 차트 (FHKST03010100)."""
        if not self.auth.ensure_token():
            return None

        if end_date is None: end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None: start_date = (datetime.now() - timedelta(days=count*2)).strftime("%Y%m%d")

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        all_rows = []
        current_end = end_date.replace("-", "")
        start_date = start_date.replace("-", "")

        for _ in range(6): # 최대 600일 분량
            try:
                res = self._session.get(url, params={
                    "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code,
                    "FID_INPUT_DATE_1": start_date, "FID_INPUT_DATE_2": current_end,
                    "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0",
                }, headers=self.auth.get_headers("FHKST03010100"), timeout=10)
                rows = res.json().get("output2", [])
                if not rows: break

                for r in rows:
                    dt = r.get("stck_bsop_date", "")
                    if dt < start_date: continue
                    all_rows.append({
                        "date": dt, "open": float(r.get("stck_oprc", 0)),
                        "high": float(r.get("stck_hgpr", 0)), "low": float(r.get("stck_lwpr", 0)),
                        "close": float(r.get("stck_clpr", 0)), "volume": int(r.get("acml_vol", 0)),
                    })
                
                if len(all_rows) >= count: break
                oldest = min(r.get("stck_bsop_date", "99999999") for r in rows)
                if oldest <= start_date: break
                current_end = str(int(oldest) - 1).zfill(8)
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"차트 조회 오류: {e}")
                break

        if not all_rows: return None
        df = pd.DataFrame(all_rows).sort_values("date").drop_duplicates("date").tail(count)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        return df.set_index("date")

    # --- Overseas Stocks ---
    def get_overseas_price(self, symbol: str, exchange: str = "NAS") -> float | None:
        """해외주식 현재가 조회 (HHDFS76200200)."""
        if not self.auth.ensure_token(): return None
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price-detail"
        try:
            res = self._session.get(url, params={"AUTH": "", "EXCD": exchange, "SYMB": symbol},
                                    headers=self.auth.get_headers("HHDFS76200200"), timeout=5)
            o = res.json().get("output", {})
            return float(o.get("last", 0) or o.get("base", 0))
        except Exception as e:
            logger.error(f"해외 주가 오류 ({symbol}): {e}")
            return None
