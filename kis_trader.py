import os
import time
import requests
import json
import hashlib
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KISTrader")

load_dotenv()


class KISTrader:
    """
    한국투자증권 Open API 트레이더 (ISA / 연금저축 계좌 지원)

    지원 기능:
      - 국내주식/ETF 현재가 조회, 일봉 차트, 매수/매도 주문
      - 해외주식 현재가/일봉 조회 (시그널 참조용)
      - 계좌 잔고 조회, 미체결 주문 조회/취소

    TR_ID 목록 (실전/모의):
      FHKST01010100       : 국내주식 현재가
      FHKST03010100       : 국내주식 일봉 차트
      TTTC0802U / VTTC0802U : 국내주식 매수
      TTTC0801U / VTTC0801U : 국내주식 매도
      TTTC0803U / VTTC0803U : 국내주식 정정/취소
      TTTC8434R / VTTC8434R : 국내주식 잔고 조회
      HHDFS76200200       : 해외주식 현재가
      HHDFS76240000       : 해외주식 일봉 차트
    """
    REAL_URL = "https://openapi.koreainvestment.com:9443"
    MOCK_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(self, is_mock=False):
        self.app_key = os.getenv("KIS_APP_KEY")
        self.app_secret = os.getenv("KIS_APP_SECRET")
        self.account_no = os.getenv("KIS_ACCOUNT_NO", "")        # 계좌번호 앞 8자리
        self.acnt_prdt_cd = os.getenv("KIS_ACNT_PRDT_CD", "01")  # 계좌상품코드 (ISA: 보통 01)
        self.is_mock = is_mock
        self.base_url = self.MOCK_URL if is_mock else self.REAL_URL

        self.access_token = None
        self.token_expiry = 0.0  # unix timestamp

        # HTTP 커넥션 풀링
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json;charset=UTF-8"})

        if not (self.app_key and self.app_secret):
            logger.warning("KIS_APP_KEY 또는 KIS_APP_SECRET 가 .env에 없습니다.")

    # ─────────────────────────────────────────────────────
    # 내부 유틸
    # ─────────────────────────────────────────────────────
    def _is_token_valid(self) -> bool:
        """토큰 만료 5분 전부터 갱신."""
        return self.access_token is not None and (self.token_expiry - time.time()) > 300

    def _ensure_token(self) -> bool:
        if self._is_token_valid():
            return True
        return self.auth()

    def _headers(self, tr_id: str, extra: dict = None) -> dict:
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

    def _normalize_account_fields(self, account_no: str | None = None, acnt_prdt_cd: str | None = None) -> tuple[str, str]:
        """
        계좌번호/CANO, 상품코드를 안전하게 정규화.
        - 10자리 입력 시: 앞 8자리(CANO) + 뒤 2자리(상품코드)
        - 8자리 입력 시: 상품코드는 인자/기존값/기본값(01) 사용
        """
        raw_acct = "".join(ch for ch in str(account_no if account_no is not None else self.account_no) if ch.isdigit())
        raw_prdt = "".join(ch for ch in str(acnt_prdt_cd if acnt_prdt_cd is not None else self.acnt_prdt_cd) if ch.isdigit())

        if len(raw_acct) >= 10:
            cano = raw_acct[:8]
            prdt = raw_acct[8:10]
        else:
            cano = raw_acct[:8] if len(raw_acct) > 8 else raw_acct
            prdt = raw_prdt

        if not prdt:
            prdt = "01"

        prdt = prdt.zfill(2)[:2]
        return cano, prdt

    def _account_params(self) -> tuple[str, str]:
        """요청 전 계좌 필드를 정규화하고 객체 상태도 동기화."""
        cano, prdt = self._normalize_account_fields()
        self.account_no = cano
        self.acnt_prdt_cd = prdt
        return cano, prdt

    def _hashkey(self, body: dict) -> str:
        """주문 요청 시 필요한 hashkey 생성."""
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

    # ─────────────────────────────────────────────────────
    # 인증 (OAuth2)
    # ─────────────────────────────────────────────────────
    def auth(self) -> bool:
        """접근 토큰 발급. 1분당 1회 제한 (EGW00133 시 대기)."""
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

                # 1분당 1회 제한 처리
                if data.get("error_code") == "EGW00133":
                    logger.warning("토큰 발급 대기 (1분당 1회 제한)... 65초 후 재시도")
                    time.sleep(65)
                    continue

                self.access_token = data.get("access_token")
                if not self.access_token:
                    logger.error(f"토큰 발급 실패: {data}")
                    return False

                # 만료 시간 (보통 24시간)
                expires_in = int(data.get("expires_in", 86400))
                self.token_expiry = time.time() + expires_in

                logger.info(f"KIS 토큰 발급 성공 (만료: {expires_in // 3600}시간)")
                return True

            except Exception as e:
                logger.error(f"KIS 인증 오류: {e}")
                return False

        return False

    # ─────────────────────────────────────────────────────
    # 국내주식/ETF 시세 조회
    # ─────────────────────────────────────────────────────
    def get_current_price(self, stock_code: str) -> float | None:
        """국내주식/ETF 현재가 조회 (FHKST01010100)."""
        if not self._ensure_token():
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
                    headers=self._headers("FHKST01010100"),
                    timeout=5,
                )
                data = res.json()
                output = data.get("output", {})
                price = float(output.get("stck_prpr", 0))
                return price if price > 0 else None
            except Exception as e:
                if _retry < 2:
                    logger.warning(f"현재가 조회 재시도 {_retry + 1}/2 ({stock_code}): {e}")
                    time.sleep(0.25 + (0.25 * _retry))
                else:
                    logger.error(f"현재가 조회 오류 ({stock_code}): {e}")
        return None

    def get_price_info(self, stock_code: str) -> dict | None:
        """국내주식/ETF 현재가 상세 조회."""
        if not self._ensure_token():
            return None

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }

        try:
            res = self._session.get(url, params=params,
                                    headers=self._headers("FHKST01010100"), timeout=5)
            data = res.json()
            o = data.get("output", {})
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
            logger.error(f"현재가 상세 오류 ({stock_code}): {e}")
            return None

    def get_orderbook(self, stock_code: str) -> dict | None:
        """
        국내주식/ETF 호가 조회 (FHKST01010200).

        Returns: {
            'asks': [{'price': float, 'qty': int}, ...],  # 매도호가 (낮→높)
            'bids': [{'price': float, 'qty': int}, ...],  # 매수호가 (높→낮)
            'cur_prc': float
        }
        """
        if not self._ensure_token():
            return None

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
        }

        try:
            res = self._session.get(url, params=params,
                                    headers=self._headers("FHKST01010200"), timeout=5)
            data = res.json()
            o = data.get("output1", {})

            if not o:
                logger.debug(f"호가 응답 없음 ({stock_code})")
                return None

            asks = []
            bids = []

            for i in range(1, 11):
                ask_p = float(o.get(f"askp{i}", 0))
                ask_q = int(o.get(f"askp_rsqn{i}", 0))
                bid_p = float(o.get(f"bidp{i}", 0))
                bid_q = int(o.get(f"bidp_rsqn{i}", 0))
                if ask_p > 0:
                    asks.append({'price': ask_p, 'qty': ask_q})
                if bid_p > 0:
                    bids.append({'price': bid_p, 'qty': bid_q})

            cur_prc = float(o.get("stck_prpr", 0))
            if not cur_prc and asks and bids:
                cur_prc = (asks[0]['price'] + bids[0]['price']) / 2

            if asks or bids:
                return {'asks': asks, 'bids': bids, 'cur_prc': cur_prc}

            return None

        except Exception as e:
            logger.error(f"호가 조회 오류 ({stock_code}): {e}")
            return None

    def get_daily_chart(self, stock_code: str, start_date: str = None,
                        end_date: str = None, count: int = 200) -> pd.DataFrame | None:
        """
        국내주식/ETF 일봉 차트 (FHKST03010100).
        Returns: DataFrame columns: [open, high, low, close, volume] index=datetime
        """
        if not self._ensure_token():
            return None

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=count * 2)).strftime("%Y%m%d")

        # 날짜 형식 통일
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        all_rows = []

        # 페이지네이션 (최대 100건씩)
        current_end = end_date
        for _ in range(60): # 최대 6000건 (100건 x 60회)
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": stock_code,
                "FID_INPUT_DATE_1": start_date,
                "FID_INPUT_DATE_2": current_end,
                "FID_PERIOD_DIV_CODE": "D",
                "FID_ORG_ADJ_PRC": "0",  # 수정주가
            }

            rows = None
            for _retry in range(3):
                try:
                    res = self._session.get(
                        url,
                        params=params,
                        headers=self._headers("FHKST03010100"),
                        timeout=10,
                    )
                    data = res.json()
                    rows = data.get("output2", [])
                    break
                except Exception as e:
                    if _retry < 2:
                        logger.warning(f"일봉 조회 재시도 {_retry + 1}/2 ({stock_code}): {e}")
                        time.sleep(0.7 + (0.4 * _retry))
                    else:
                        logger.error(f"일봉 조회 오류 ({stock_code}): {e}")

            if rows is None:
                break
            if not rows:
                break

            for r in rows:
                try:
                    dt = r.get("stck_bsop_date", "")
                    if not dt or dt < start_date:
                        continue
                    all_rows.append({
                        "date": dt,
                        "open": float(r.get("stck_oprc", 0)),
                        "high": float(r.get("stck_hgpr", 0)),
                        "low": float(r.get("stck_lwpr", 0)),
                        "close": float(r.get("stck_clpr", 0)),
                        "volume": int(r.get("acml_vol", 0)),
                    })
                except (ValueError, TypeError):
                    continue

            if len(all_rows) >= count:
                break

            # 다음 페이지 (가장 오래된 날짜 - 1)
            oldest = min(r.get("stck_bsop_date", "99999999") for r in rows)
            if oldest <= start_date:
                break
            current_end = str(int(oldest) - 1).zfill(8)

            time.sleep(0.2)

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df.tail(count)

    # ─────────────────────────────────────────────────────
    # 해외주식 시세 조회 (시그널 참조용)
    # ─────────────────────────────────────────────────────
    def get_overseas_price(self, symbol: str, exchange: str = "NAS") -> float | None:
        """
        해외주식 현재가 조회 (HHDFS76200200).
        exchange: NAS(나스닥), NYS(뉴욕), AMS(아멕스)
        """
        if not self._ensure_token():
            return None

        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price-detail"
        params = {"AUTH": "", "EXCD": exchange, "SYMB": symbol}

        try:
            res = self._session.get(url, params=params,
                                    headers=self._headers("HHDFS76200200"), timeout=5)
            data = res.json()
            output = data.get("output", {})
            price = float(output.get("last", 0) or output.get("base", 0))
            return price if price > 0 else None
        except Exception as e:
            logger.error(f"해외 현재가 오류 ({symbol}): {e}")
            return None

    def get_overseas_daily_chart(self, symbol: str, exchange: str = "NAS",
                                 count: int = 1500) -> pd.DataFrame | None:
        """
        해외주식 일봉 차트 (HHDFS76240000).
        Returns: DataFrame columns: [open, high, low, close, volume] index=datetime
        """
        if not self._ensure_token():
            return None

        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice"
        all_rows = []
        bymd = ""  # 빈 문자열 = 최신부터

        for _ in range(100):  # 최대 10000일 (~27년)
            params = {
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": symbol,
                "GUBN": "0",   # 0=일별
                "BYMD": bymd,
                "MODP": "0",   # 수정주가
            }

            try:
                res = self._session.get(url, params=params,
                                        headers=self._headers("HHDFS76240000"), timeout=10)
                data = res.json()
                rows = data.get("output2", [])

                if not rows:
                    break

                for r in rows:
                    try:
                        dt = r.get("xymd", "")
                        if not dt:
                            continue
                        close = float(r.get("clos", 0))
                        if close <= 0:
                            continue
                        all_rows.append({
                            "date": dt,
                            "open": float(r.get("open", 0)),
                            "high": float(r.get("high", 0)),
                            "low": float(r.get("low", 0)),
                            "close": close,
                            "volume": int(float(r.get("tvol", 0))),
                        })
                    except (ValueError, TypeError):
                        continue

                if len(all_rows) >= count:
                    break

                # 다음 페이지: 가장 오래된 날짜
                oldest = min(r.get("xymd", "99999999") for r in rows)
                bymd = oldest

                time.sleep(0.2)

            except Exception as e:
                logger.error(f"해외 일봉 오류 ({symbol}): {e}")
                break

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df.tail(count)

    # ─────────────────────────────────────────────────────
    # 계좌
    # ─────────────────────────────────────────────────────
    def get_balance(self) -> dict | None:
        """
        계좌 잔고 조회.
        Returns: {
            "cash": float,          # 예수금 (주문가능현금)
            "total_eval": float,    # 총 평가금액
            "holdings": [           # 보유 종목 리스트
                {"code": str, "name": str, "qty": int, "avg_price": float,
                 "cur_price": float, "eval_amt": float, "pnl_rate": float}
            ]
        }
        """
        if not self._ensure_token():
            return None
        cano, prdt = self._account_params()

        tr_id = "VTTC8434R" if self.is_mock else "TTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        try:
            res = self._session.get(url, params=params,
                                    headers=self._headers(tr_id), timeout=10)
            data = res.json()

            # API 응답 코드 확인
            rt_cd = data.get("rt_cd", "")
            if rt_cd != "0":
                msg_cd = data.get("msg_cd", "")
                msg1 = data.get("msg1", "")
                logger.error(f"잔고 조회 API 오류: rt_cd={rt_cd}, msg_cd={msg_cd}, msg1={msg1}")
                return {"error": True, "msg_cd": msg_cd, "msg1": msg1, "rt_cd": rt_cd,
                        "cash": 0.0, "total_eval": 0.0, "holdings": []}

            # 보유 종목
            holdings = []
            for item in data.get("output1", []):
                qty = int(item.get("hldg_qty", 0))
                if qty <= 0:
                    continue
                holdings.append({
                    "code": item.get("pdno", ""),
                    "name": item.get("prdt_name", ""),
                    "qty": qty,
                    "avg_price": float(item.get("pchs_avg_pric", 0)),
                    "cur_price": float(item.get("prpr", 0)),
                    "eval_amt": float(item.get("evlu_amt", 0)),
                    "pnl_rate": float(item.get("evlu_pfls_rt", 0)),
                })

            # 계좌 요약
            output2 = data.get("output2", [{}])
            summary = output2[0] if output2 else {}

            return {
                "cash": float(summary.get("dnca_tot_amt", 0)),
                "total_eval": float(summary.get("tot_evlu_amt", 0)),
                "holdings": holdings,
            }

        except Exception as e:
            logger.error(f"잔고 조회 오류: {e}")
            return None

    def get_holding_qty(self, stock_code: str) -> int:
        """특정 종목 보유 수량 조회."""
        bal = self.get_balance()
        if not bal:
            return 0
        for h in bal["holdings"]:
            if h["code"] == stock_code:
                return h["qty"]
        return 0

    # ─────────────────────────────────────────────────────
    # 주문
    # ─────────────────────────────────────────────────────
    def send_order(self, order_type: str, stock_code: str, qty: int,
                   price: int = 0, ord_dvsn: str = "01") -> dict | None:
        """
        국내주식/ETF 매수/매도 주문.

        Args:
            order_type: "BUY" 또는 "SELL"
            stock_code: 종목코드 (6자리)
            qty: 주문 수량
            price: 주문 가격 (시장가일 때 0)
            ord_dvsn: "00"=지정가, "01"=시장가
        """
        if not self._ensure_token():
            return None
        cano, prdt = self._account_params()

        if order_type == "BUY":
            tr_id = "VTTC0802U" if self.is_mock else "TTTC0802U"
        else:
            tr_id = "VTTC0801U" if self.is_mock else "TTTC0801U"

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "PDNO": stock_code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price) if price > 0 else "0",
        }

        # hashkey 생성
        hashkey = self._hashkey(body)
        headers = self._headers(tr_id, {"hashkey": hashkey})

        try:
            res = self._session.post(url, json=body, headers=headers, timeout=10)
            data = res.json()

            rt_cd = data.get("rt_cd", "")
            msg = data.get("msg1", "")

            if rt_cd == "0":
                output = data.get("output", {})
                ord_no = output.get("ODNO", "")
                logger.info(f"[{order_type}] 주문 성공: {stock_code} {qty}주, 주문번호={ord_no}")
                return {"success": True, "ord_no": ord_no, "msg": msg}
            else:
                logger.error(f"[{order_type}] 주문 실패: {msg}")
                return {"success": False, "msg": msg}

        except Exception as e:
            logger.error(f"주문 오류: {e}")
            return None

    def cancel_order(self, org_ord_no: str, stock_code: str, qty: int = 0,
                     cancel_all: bool = True) -> dict | None:
        """국내주식 주문 취소."""
        if not self._ensure_token():
            return None
        cano, prdt = self._account_params()

        tr_id = "VTTC0803U" if self.is_mock else "TTTC0803U"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-rvsecncl"

        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "KRX_FWDG_ORD_ORGNO": "",
            "ORGN_ODNO": org_ord_no,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02=취소
            "ORD_QTY": "0" if cancel_all else str(qty),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y" if cancel_all else "N",
        }

        hashkey = self._hashkey(body)
        headers = self._headers(tr_id, {"hashkey": hashkey})

        try:
            res = self._session.post(url, json=body, headers=headers, timeout=10)
            data = res.json()
            rt_cd = data.get("rt_cd", "")
            msg = data.get("msg1", "")
            logger.info(f"취소 결과: rt_cd={rt_cd}, msg={msg}")
            return {"success": rt_cd == "0", "msg": msg}
        except Exception as e:
            logger.error(f"취소 오류: {e}")
            return None

    # ─────────────────────────────────────────────────────
    # 스마트 매수/매도 (간단 래퍼)
    # ─────────────────────────────────────────────────────
    def smart_buy_krw(self, stock_code: str, krw_amount: float) -> dict | None:
        """KRW 금액 기준 시장가 매수."""
        price = self.get_current_price(stock_code)
        if not price or price <= 0:
            logger.error(f"매수 실패: {stock_code} 현재가 조회 불가")
            return None

        qty = int(krw_amount / price)
        if qty <= 0:
            logger.info(f"매수 불가: {stock_code} 금액 부족 ({krw_amount:,.0f}원, 현재가={price:,.0f}원)")
            return None

        logger.info(f"시장가 매수: {stock_code} {qty}주 (≈{qty * price:,.0f}원)")
        return self.send_order("BUY", stock_code, qty, ord_dvsn="01")

    def smart_sell_all(self, stock_code: str) -> dict | None:
        """보유 수량 전량 시장가 매도."""
        qty = self.get_holding_qty(stock_code)
        if qty <= 0:
            logger.info(f"매도 불가: {stock_code} 보유 없음")
            return None

        logger.info(f"전량 시장가 매도: {stock_code} {qty}주")
        return self.send_order("SELL", stock_code, qty, ord_dvsn="01")

    def smart_sell_qty(self, stock_code: str, qty: int) -> dict | None:
        """지정 수량 시장가 매도."""
        holding = self.get_holding_qty(stock_code)
        qty = min(qty, holding)
        if qty <= 0:
            return None

        logger.info(f"시장가 매도: {stock_code} {qty}/{holding}주")
        return self.send_order("SELL", stock_code, qty, ord_dvsn="01")

    # ─────────────────────────────────────────────────────
    # 미체결 조회
    # ─────────────────────────────────────────────────────
    def get_pending_orders(self, stock_code: str = "") -> list:
        """
        미체결(정정/취소 가능) 주문 조회.
        TR: TTTC8036R(실전) / VTTC8036R(모의)
        """
        if not self._ensure_token():
            return []
        cano, prdt = self._account_params()

        tr_id = "VTTC8036R" if self.is_mock else "TTTC8036R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"

        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "INQR_DVSN_1": "1",
            "INQR_DVSN_2": "0",
        }
        headers = self._headers(tr_id)

        try:
            res = self._session.get(url, params=params, headers=headers, timeout=10)
            data = res.json()
            if data.get("rt_cd") != "0":
                logger.debug(f"미체결조회: {data.get('msg1', '')}")
                return []

            result = []
            for r in data.get("output", []):
                code = r.get("pdno", "")
                if stock_code and code != stock_code:
                    continue
                remaining = int(r.get("psbl_qty", 0))
                if remaining <= 0:
                    continue
                result.append({
                    "ord_no": r.get("odno", ""),
                    "side": "SELL" if r.get("sll_buy_dvsn_cd") == "01" else "BUY",
                    "stock_code": code,
                    "ord_qty": int(r.get("ord_qty", 0)),
                    "remaining_qty": remaining,
                    "ord_price": int(r.get("ord_unpr", 0)),
                    "ord_time": r.get("ord_tmd", ""),
                })
            logger.info(f"미체결 주문: {len(result)}건" + (f" ({stock_code})" if stock_code else ""))
            return result

        except Exception as e:
            logger.error(f"미체결조회 오류: {e}")
            return []

    # ─────────────────────────────────────────────────────
    # 장마감 동시호가 + 시간외 재주문
    # ─────────────────────────────────────────────────────
    def _get_limit_price(self, stock_code: str, order_type: str) -> int:
        """동시호가 체결 보장을 위한 상한가(BUY)/하한가(SELL) 계산."""
        price = self.get_current_price(stock_code)
        if not price or price <= 0:
            return 0

        if order_type == "BUY":
            raw = int(price * 1.30)
        else:
            raw = int(price * 0.70)

        # ETF 호가단위 정렬
        if raw < 5000:
            tick = 5
        elif raw < 10000:
            tick = 10
        elif raw < 50000:
            tick = 50
        else:
            tick = 100

        if order_type == "BUY":
            return (raw // tick) * tick
        else:
            return ((raw + tick - 1) // tick) * tick

    def execute_closing_auction_buy(self, stock_code: str, qty: int) -> dict:
        """
        장마감 동시호가 매수 + 미체결 시 시간외 재주문.
        1. 상한가 지정가 주문 (ord_dvsn="00") → 동시호가 참여
        2. 60초 대기 후 미체결 확인
        3. 미체결 → 취소 → 시간외(ord_dvsn="06") 재주문
        """
        import time as _time

        if qty <= 0:
            return {"success": False, "msg": "수량 0"}

        limit_price = self._get_limit_price(stock_code, "BUY")
        if limit_price <= 0:
            return {"success": False, "msg": "현재가 조회 실패"}

        logger.info(f"[동시호가 매수] {stock_code} {qty}주 @ 상한가 {limit_price:,}원")
        phase1 = self.send_order("BUY", stock_code, qty, price=limit_price, ord_dvsn="00")

        if not phase1 or not phase1.get("success"):
            logger.error(f"[동시호가 매수] 주문 실패: {phase1}")
            return {"success": False, "phase1_result": phase1, "filled_qty": 0,
                    "remaining_qty": qty, "method": "closing_auction_failed"}

        ord_no = phase1.get("ord_no", "")
        logger.info(f"[동시호가 매수] 주문 접수. 주문번호={ord_no}. 체결 대기 60초...")
        _time.sleep(60)

        # 미체결 확인
        pending = self.get_pending_orders(stock_code)
        unfilled = [p for p in pending if p["ord_no"] == ord_no and p["remaining_qty"] > 0]

        if not unfilled:
            logger.info(f"[동시호가 매수] 전량 체결 완료: {stock_code} {qty}주")
            return {"success": True, "phase1_result": phase1, "phase2_result": None,
                    "filled_qty": qty, "remaining_qty": 0, "method": "closing_auction"}

        remaining = unfilled[0]["remaining_qty"]
        filled = qty - remaining
        logger.info(f"[동시호가 매수] 미체결 {remaining}주 (체결 {filled}주). 취소 후 시간외 재주문...")

        cancel_result = self.cancel_order(ord_no, stock_code)
        _time.sleep(3)

        if not cancel_result or not cancel_result.get("success"):
            logger.error(f"[동시호가 매수] 취소 실패: {cancel_result}")
            return {"success": False, "phase1_result": phase1, "phase2_result": None,
                    "filled_qty": filled, "remaining_qty": remaining, "method": "cancel_failed"}

        # 시간외 종가 재주문 (ord_dvsn="06", price=0)
        logger.info(f"[시간외 매수] {stock_code} {remaining}주 (ord_dvsn=06)")
        phase2 = self.send_order("BUY", stock_code, remaining, price=0, ord_dvsn="06")
        logger.info(f"[시간외 매수] 결과: {phase2}")

        return {"success": True, "phase1_result": phase1, "phase2_result": phase2,
                "filled_qty": filled, "remaining_qty": remaining,
                "method": "closing_auction+after_hours"}

    def execute_closing_auction_sell(self, stock_code: str, qty: int) -> dict:
        """
        장마감 동시호가 매도 + 미체결 시 시간외 재주문.
        1. 하한가 지정가 주문 (ord_dvsn="00") → 동시호가 참여
        2. 60초 대기 후 미체결 확인
        3. 미체결 → 취소 → 시간외(ord_dvsn="06") 재주문
        """
        import time as _time

        if qty <= 0:
            return {"success": False, "msg": "수량 0"}

        limit_price = self._get_limit_price(stock_code, "SELL")
        if limit_price <= 0:
            return {"success": False, "msg": "현재가 조회 실패"}

        logger.info(f"[동시호가 매도] {stock_code} {qty}주 @ 하한가 {limit_price:,}원")
        phase1 = self.send_order("SELL", stock_code, qty, price=limit_price, ord_dvsn="00")

        if not phase1 or not phase1.get("success"):
            logger.error(f"[동시호가 매도] 주문 실패: {phase1}")
            return {"success": False, "phase1_result": phase1, "filled_qty": 0,
                    "remaining_qty": qty, "method": "closing_auction_failed"}

        ord_no = phase1.get("ord_no", "")
        logger.info(f"[동시호가 매도] 주문 접수. 주문번호={ord_no}. 체결 대기 60초...")
        _time.sleep(60)

        pending = self.get_pending_orders(stock_code)
        unfilled = [p for p in pending if p["ord_no"] == ord_no and p["remaining_qty"] > 0]

        if not unfilled:
            logger.info(f"[동시호가 매도] 전량 체결 완료: {stock_code} {qty}주")
            return {"success": True, "phase1_result": phase1, "phase2_result": None,
                    "filled_qty": qty, "remaining_qty": 0, "method": "closing_auction"}

        remaining = unfilled[0]["remaining_qty"]
        filled = qty - remaining
        logger.info(f"[동시호가 매도] 미체결 {remaining}주 (체결 {filled}주). 취소 후 시간외 재주문...")

        cancel_result = self.cancel_order(ord_no, stock_code)
        _time.sleep(3)

        if not cancel_result or not cancel_result.get("success"):
            logger.error(f"[동시호가 매도] 취소 실패: {cancel_result}")
            return {"success": False, "phase1_result": phase1, "phase2_result": None,
                    "filled_qty": filled, "remaining_qty": remaining, "method": "cancel_failed"}

        logger.info(f"[시간외 매도] {stock_code} {remaining}주 (ord_dvsn=06)")
        phase2 = self.send_order("SELL", stock_code, remaining, price=0, ord_dvsn="06")
        logger.info(f"[시간외 매도] 결과: {phase2}")

        return {"success": True, "phase1_result": phase1, "phase2_result": phase2,
                "filled_qty": filled, "remaining_qty": remaining,
                "method": "closing_auction+after_hours"}

    # ─────────────────────────────────────────────────────
    # 동시호가 스마트 래퍼
    # ─────────────────────────────────────────────────────
    def smart_buy_krw_closing(self, stock_code: str, krw_amount: float) -> dict | None:
        """KRW 금액 기준 동시호가 매수."""
        price = self.get_current_price(stock_code)
        if not price or price <= 0:
            logger.error(f"동시호가 매수 실패: {stock_code} 현재가 조회 불가")
            return None
        qty = int(krw_amount / price)
        if qty <= 0:
            logger.info(f"동시호가 매수 불가: 금액 부족 ({krw_amount:,.0f}원)")
            return None
        return self.execute_closing_auction_buy(stock_code, qty)

    def smart_sell_qty_closing(self, stock_code: str, qty: int) -> dict | None:
        """지정 수량 동시호가 매도."""
        holding = self.get_holding_qty(stock_code)
        qty = min(qty, holding)
        if qty <= 0:
            return None
        return self.execute_closing_auction_sell(stock_code, qty)

    def smart_sell_all_closing(self, stock_code: str) -> dict | None:
        """전량 동시호가 매도."""
        qty = self.get_holding_qty(stock_code)
        if qty <= 0:
            logger.info(f"동시호가 매도 불가: {stock_code} 보유 없음")
            return None
        return self.execute_closing_auction_sell(stock_code, qty)


# ─────────────────────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_dotenv()
    trader = KISTrader(is_mock=False)

    # 1. 인증
    print("[1] 인증...")
    if not trader.auth():
        print("  인증 실패")
        exit(1)
    print("  인증 성공")

    # 2. 국내 ETF 현재가
    print("\n[2] TIGER 미국나스닥100 (133690) 현재가...")
    price = trader.get_current_price("133690")
    print(f"  현재가: {price:,.0f}원" if price else "  조회 실패")

    # 3. 해외주식 현재가
    print("\n[3] QQQ 현재가...")
    qqq_price = trader.get_overseas_price("QQQ", "NAS")
    print(f"  QQQ: ${qqq_price:.2f}" if qqq_price else "  조회 실패")

    # 4. 잔고
    print("\n[4] 잔고 조회...")
    bal = trader.get_balance()
    if bal:
        print(f"  예수금: {bal['cash']:,.0f}원")
        print(f"  총 평가: {bal['total_eval']:,.0f}원")
        for h in bal["holdings"]:
            print(f"  {h['name']}({h['code']}): {h['qty']}주 @ {h['cur_price']:,.0f}원")
    else:
        print("  잔고 조회 실패")
