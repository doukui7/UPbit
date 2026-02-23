import os
import time
import requests
import json
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KiwoomGold")

load_dotenv()

# ─────────────────────────────────────────────────────────
# 키움 금현물 종목코드
# M04020000 = 금99.99 1kg
# M04020100 = 미니금99.99 100g
# ─────────────────────────────────────────────────────────
GOLD_CODE_1KG   = "M04020000"
GOLD_CODE_MINI  = "M04020100"

class KiwoomGoldTrader:
    """
    키움증권 금현물(KRX) REST API 트레이더

    API 문서 TR 목록:
      ka50081 : 금현물 일봉차트 조회
      kt50000 : 금현물 매수주문
      kt50001 : 금현물 매도주문
      kt50002 : 금현물 정정주문
      kt50003 : 금현물 취소주문
    """
    REAL_URL = "https://api.kiwoom.com"
    MOCK_URL = "https://mockapi.kiwoom.com"

    def __init__(self, is_mock=False):
        # .env 키 이름: Kiwoom_App_Key / Kiwoom_Secret_Key / KIWOOM_ACCOUNT
        self.app_key    = os.getenv("Kiwoom_App_Key")
        self.app_secret = os.getenv("Kiwoom_Secret_Key")
        self.account_no = os.getenv("KIWOOM_ACCOUNT", "")
        self.is_mock    = is_mock
        self.base_url   = self.MOCK_URL if is_mock else self.REAL_URL

        self.access_token  = None
        self.token_expiry  = 0.0  # unix timestamp

        # HTTP 커넥션 풀링 (매 요청마다 TCP 핸드셰이크 방지)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json;charset=UTF-8"})

        if not (self.app_key and self.app_secret):
            logger.warning("Kiwoom_App_Key 또는 Kiwoom_Secret_Key 가 .env에 없습니다.")

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

    def _headers(self, api_id: str, extra: dict = None) -> dict:
        h = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.access_token}",
            "api-id":        api_id,
        }
        if extra:
            h.update(extra)
        return h

    # ─────────────────────────────────────────────────────
    # 인증 (OAuth2 Client Credentials)
    # ─────────────────────────────────────────────────────
    def auth(self) -> bool:
        """액세스 토큰 발급. 성공 시 True 반환."""
        if not (self.app_key and self.app_secret):
            logger.error("Auth 실패: API 키가 없습니다.")
            return False

        url     = f"{self.base_url}/oauth2/token"
        payload = {
            "grant_type": "client_credentials",
            "appkey":     self.app_key,
            "secretkey":  self.app_secret,
        }
        try:
            res  = self._session.post(url, json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()

            self.access_token = data.get("token") or data.get("access_token")

            # 만료 시각 파싱 ("20260220222955" 형식 또는 expires_in 초)
            expires_dt = data.get("expires_dt", "")
            if expires_dt and len(expires_dt) == 14:
                exp = datetime.strptime(expires_dt, "%Y%m%d%H%M%S")
                self.token_expiry = exp.timestamp()
            else:
                self.token_expiry = time.time() + int(data.get("expires_in", 86400))

            logger.info(f"키움 인증 성공. 만료: {datetime.fromtimestamp(self.token_expiry)}")
            return True

        except Exception as e:
            logger.error(f"키움 인증 오류: {e}")
            return False

    # ─────────────────────────────────────────────────────
    # 시세 조회
    # ─────────────────────────────────────────────────────
    def get_current_price(self, code: str = GOLD_CODE_1KG) -> float | None:
        """
        금현물 현재가 조회.
        1순위: 세션 캐시 (5초 TTL)
        2순위: 일봉 최신 데이터
        3순위: 현재가 API (ka50070)
        """
        # 세션 캐시 확인 (5초 TTL)
        now = time.time()
        cache_key = f"_price_{code}"
        if hasattr(self, '_price_cache') and cache_key in self._price_cache:
            cached_price, cached_time = self._price_cache[cache_key]
            if (now - cached_time) < 5 and cached_price > 0:
                return cached_price

        if not hasattr(self, '_price_cache'):
            self._price_cache = {}

        # 일봉 (가장 안정적)
        df = self.get_daily_chart(code=code, count=1)
        if df is not None and not df.empty:
            price = float(df.iloc[-1]["close"])
            self._price_cache[cache_key] = (price, now)
            return price

        # 현재가 API 폴백
        price_data = self.get_price_info(code)
        if price_data and price_data.get('cur_prc', 0) > 0:
            price = price_data['cur_prc']
            self._price_cache[cache_key] = (price, now)
            return price

        return None

    def get_market_price(self, code: str = GOLD_CODE_1KG) -> float | None:
        """get_current_price 의 alias (app.py 호환성)."""
        return self.get_current_price(code)

    def get_price_info(self, code: str = GOLD_CODE_1KG) -> dict | None:
        """
        금현물 현재가 상세 조회 (ka50070).
        Returns: {cur_prc, open_pric, high_pric, low_pric, prev_close, change, change_rate, volume}
        """
        if not self._ensure_token():
            return None

        url  = f"{self.base_url}/api/dostk/stkinfo"
        body = {"stk_cd": code}

        try:
            res  = self._session.post(url, json=body, headers=self._headers("ka50070"), timeout=5)
            res.raise_for_status()
            data = res.json()

            return {
                'cur_prc':     float(data.get("cur_prc",     0)),
                'open_pric':   float(data.get("open_pric",   0)),
                'high_pric':   float(data.get("high_pric",   0)),
                'low_pric':    float(data.get("low_pric",    0)),
                'prev_close':  float(data.get("prev_cls_prc", 0)),
                'change':      float(data.get("vs_prev",     0)),
                'change_rate': float(data.get("vs_prev_rate", 0)),
                'volume':      float(data.get("acc_trde_qty", 0)),
            }
        except Exception as e:
            logger.debug(f"현재가 API 오류 (폴백 사용): {e}")
            return None

    def get_orderbook(self, code: str = GOLD_CODE_1KG) -> dict | None:
        """
        금현물 호가 조회 (ka50072).
        API 실패 시 현재가 기반 시뮬레이션 호가 생성.

        Returns: {
            'asks': [{'price': float, 'qty': float}, ...],  # 매도호가 (낮→높)
            'bids': [{'price': float, 'qty': float}, ...],  # 매수호가 (높→낮)
            'cur_prc': float
        }
        """
        if not self._ensure_token():
            return self._simulated_orderbook(code)

        url  = f"{self.base_url}/api/dostk/stkinfo"
        body = {"stk_cd": code}

        try:
            res  = self._session.post(url, json=body, headers=self._headers("ka50072"), timeout=5)
            res.raise_for_status()
            data = res.json()

            asks = []
            bids = []
            for i in range(1, 11):
                ask_p = float(data.get(f"sell_prc_{i}", 0))
                ask_q = float(data.get(f"sell_qty_{i}", 0))
                bid_p = float(data.get(f"buy_prc_{i}", 0))
                bid_q = float(data.get(f"buy_qty_{i}", 0))
                if ask_p > 0:
                    asks.append({'price': ask_p, 'qty': ask_q})
                if bid_p > 0:
                    bids.append({'price': bid_p, 'qty': bid_q})

            cur_prc = float(data.get("cur_prc", 0))
            if not cur_prc and asks and bids:
                cur_prc = (asks[0]['price'] + bids[0]['price']) / 2

            if asks or bids:
                return {'asks': asks, 'bids': bids, 'cur_prc': cur_prc}

        except Exception as e:
            logger.debug(f"호가 API 오류 (시뮬레이션 사용): {e}")

        return self._simulated_orderbook(code)

    def _simulated_orderbook(self, code: str = GOLD_CODE_1KG) -> dict | None:
        """현재가 기반 시뮬레이션 호가 (API 미지원 시 폴백)."""
        import random

        cur_price = self.get_current_price(code)
        if not cur_price or cur_price <= 0:
            return None

        tick = 10  # KRX 금현물 호가단위: 10원
        asks = []
        bids = []

        for i in range(10):
            ask_p = cur_price + tick * (i + 1)
            bid_p = cur_price - tick * i
            ask_q = round(random.uniform(0.5, 50.0), 2)
            bid_q = round(random.uniform(0.5, 50.0), 2)
            asks.append({'price': ask_p, 'qty': ask_q})
            if bid_p > 0:
                bids.append({'price': bid_p, 'qty': bid_q})

        return {'asks': asks, 'bids': bids, 'cur_prc': cur_price}

    def get_daily_chart(self, code: str = GOLD_CODE_1KG, count: int = 200) -> pd.DataFrame | None:
        """
        금현물 일봉 차트 조회 (ka50081).
        
        Returns:
            DataFrame columns: [open, high, low, close, volume]  index=datetime
        """
        if not self._ensure_token():
            return None

        url  = f"{self.base_url}/api/dostk/chart"
        body = {
            "stk_cd":        code,
            "base_dt":       datetime.now().strftime("%Y%m%d"),
            "upd_stkpc_tp":  "0",   # 0=원주가
        }

        all_rows = []
        cont_yn  = "N"
        next_key = ""

        for _ in range(20):  # 최대 20번 연속 조회 (페이지당 ~100일)
            extra = {}
            if cont_yn == "Y" and next_key:
                extra["cont-yn"]  = "Y"
                extra["next-key"] = next_key

            try:
                res  = self._session.post(url, json=body, headers=self._headers("ka50081", extra), timeout=10)
                data = res.json()
                logger.debug(f"일봉 응답 키: {list(data.keys())}")
            except Exception as e:
                logger.error(f"일봉 조회 오류: {e}")
                break

            # 응답 키 후보: gds_day_chart_qry, stk_dt_pole_chart_qry 등
            rows = (data.get("gds_day_chart_qry")
                    or data.get("stk_dt_pole_chart_qry")
                    or data.get("output")
                    or [])
            if not rows:
                break

            for r in rows:
                try:
                    all_rows.append({
                        "date":   r.get("dt", ""),
                        "open":   float(r.get("open_pric",  0)),
                        "high":   float(r.get("high_pric",  0)),
                        "low":    float(r.get("low_pric",   0)),
                        "close":  float(r.get("cur_prc",    0)),
                        "volume": float(r.get("acc_trde_qty", 0)),
                    })
                except (ValueError, TypeError):
                    continue

            cont_yn  = res.headers.get("cont-yn",  "N")
            next_key = res.headers.get("next-key", "")

            if cont_yn != "Y" or len(all_rows) >= count:
                break

        if not all_rows:
            logger.warning(f"[{code}] 일봉 데이터 없음.")
            return None

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        return df.tail(count)

    # ─────────────────────────────────────────────────────
    # 잔고 조회
    # ─────────────────────────────────────────────────────
    def get_balance(self) -> dict | None:
        """
        계좌 잔고 조회 (예수금 + 금 보유량).
        Returns: {"cash_krw": float, "gold_qty": float, "gold_eval": float}
        """
        if not self._ensure_token():
            return None

        url  = f"{self.base_url}/api/dostk/acnt"
        body = {"acnt_no": self.account_no}

        try:
            res  = self._session.post(url, json=body, headers=self._headers("kt50020"), timeout=10)
            res.raise_for_status()
            data = res.json()
            logger.debug(f"잔고 응답: {data}")

            # 응답 구조는 API 문서 확인 후 조정
            cash_krw  = float(data.get("dsps_psbl_amt", 0))   # 출금가능금액
            holdings  = data.get("acnt_evlt_amt_list", [])
            gold_qty  = 0.0
            gold_eval = 0.0
            for h in holdings:
                stk_cd = h.get("stk_cd", "")
                if stk_cd.startswith("M0402"):
                    gold_qty  += float(h.get("rmnd_qty",     0))
                    gold_eval += float(h.get("evlt_amt",     0))

            result = {"cash_krw": cash_krw, "gold_qty": gold_qty, "gold_eval": gold_eval}
            logger.info(f"잔고: 예수금={cash_krw:,.0f}원, 금={gold_qty}g, 평가={gold_eval:,.0f}원")
            return result

        except Exception as e:
            logger.error(f"잔고 조회 오류: {e}")
            return None

    # ─────────────────────────────────────────────────────
    # 주문
    # ─────────────────────────────────────────────────────
    def send_order(self, order_type: str, code: str, qty: float, price: float = 0, ord_tp: str = "3") -> dict | None:
        """
        금현물 주문 전송.
        
        Args:
            order_type : "BUY" | "SELL"
            code       : 종목코드 (예: M04020000)
            qty        : 주문 수량 (그램 단위)
            price      : 주문 가격 (0이면 시장가(ord_tp=3))
            ord_tp     : "1"=지정가, "3"=시장가 (기본: 시장가)
        """
        if not self._ensure_token():
            return None

        api_id = "kt50000" if order_type == "BUY" else "kt50001"
        url    = f"{self.base_url}/api/dostk/acnt"

        body = {
            "acnt_no":  self.account_no,
            "stk_cd":   code,
            "ord_qty":  str(qty),
            "ord_uv":   str(int(price)) if price > 0 else "0",
            "ord_tp":   ord_tp,          # 1=지정가, 3=시장가
            "buy_sell": "1" if order_type == "BUY" else "2",
        }

        try:
            res  = self._session.post(url, json=body, headers=self._headers(api_id), timeout=10)
            res.raise_for_status()
            data = res.json()
            rc   = data.get("return_code", -1)
            msg  = data.get("return_msg",  "")

            if str(rc) == "0":
                ord_no = data.get("ord_no", "")
                logger.info(f"[{order_type}] 주문 성공: 종목={code}, 수량={qty}, 주문번호={ord_no}")
                return {"success": True, "ord_no": ord_no, "data": data}
            else:
                logger.error(f"[{order_type}] 주문 실패: rc={rc}, msg={msg}")
                return {"success": False, "rc": rc, "msg": msg}

        except Exception as e:
            logger.error(f"주문 오류: {e}")
            return None

    def cancel_order(self, org_ord_no: str, code: str, qty: float) -> dict | None:
        """금현물 주문 취소 (kt50003)."""
        if not self._ensure_token():
            return None
        url  = f"{self.base_url}/api/dostk/acnt"
        body = {
            "acnt_no":      self.account_no,
            "stk_cd":       code,
            "org_ord_no":   org_ord_no,
            "cncl_ord_qty": str(qty),
        }
        try:
            res  = self._session.post(url, json=body, headers=self._headers("kt50003"), timeout=10)
            data = res.json()
            logger.info(f"취소 결과: {data}")
            return data
        except Exception as e:
            logger.error(f"취소 오류: {e}")
            return None

    def get_pending_orders(self, code: str = GOLD_CODE_1KG) -> list:
        """
        금현물 미체결 주문 조회 (kt50075).
        Returns: list of dicts with keys: ord_no, side, price, qty, remaining_qty, ord_time
        """
        if not self._ensure_token():
            return []
        url  = f"{self.base_url}/api/dostk/acnt"
        body = {
            "acnt_no": self.account_no,
            "stk_cd":  code,
        }
        try:
            res  = self._session.post(url, json=body, headers=self._headers("kt50075"), timeout=10)
            data = res.json()
            rc   = data.get("return_code", -1)
            if str(rc) != "0":
                logger.debug(f"미체결조회: rc={rc}, msg={data.get('return_msg', '')}")
                return []

            # 응답 키 후보 탐색
            rows = (data.get("gds_unfilled_qry")
                    or data.get("gds_ncls_ord_qry")
                    or data.get("output")
                    or [])

            result = []
            for r in rows:
                try:
                    ord_no = r.get("ord_no", "")
                    buy_sell = r.get("buy_sell", "")
                    side = "BUY" if buy_sell == "1" else "SELL"
                    side_kr = "매수" if side == "BUY" else "매도"
                    price = float(r.get("ord_uv", 0) or r.get("ord_prc", 0))
                    qty = float(r.get("ord_qty", 0))
                    remaining = float(r.get("ncls_qty", 0) or r.get("rmn_qty", 0) or qty)
                    ord_time = r.get("ord_tm", "") or r.get("ord_time", "")

                    result.append({
                        "ord_no": ord_no,
                        "side": side,
                        "side_kr": side_kr,
                        "price": price,
                        "qty": qty,
                        "remaining_qty": remaining,
                        "ord_time": ord_time,
                        "code": code,
                    })
                except (ValueError, TypeError):
                    continue

            logger.info(f"미체결 주문: {len(result)}건")
            return result

        except Exception as e:
            logger.error(f"미체결조회 오류: {e}")
            return []

    # ─────────────────────────────────────────────────────
    # 스마트 매수 / 매도 (간단 래퍼)
    # ─────────────────────────────────────────────────────
    def smart_buy_krw(self, code: str = GOLD_CODE_1KG, krw_amount: float = 0) -> dict | None:
        """
        KRW 금액 기준 금현물 시장가 매수.
        qty = krw_amount / current_price  (그램 단위, 소수점 2자리)
        """
        price = self.get_current_price(code)
        if not price or price <= 0:
            logger.error("현재가 조회 실패로 매수 취소.")
            return None
        qty = round(krw_amount / price, 2)
        if qty <= 0:
            logger.error(f"수량 계산 오류: {qty}")
            return None
        logger.info(f"시장가 매수: {qty}g @ ≈{price:,.0f}원 (총 {krw_amount:,.0f}원)")
        return self.send_order("BUY", code, qty, ord_tp="3")

    def smart_sell_all(self, code: str = GOLD_CODE_1KG) -> dict | None:
        """보유 금 전량 시장가 매도."""
        bal = self.get_balance()
        if not bal or bal["gold_qty"] <= 0:
            logger.warning("매도 실패: 보유 금 없음.")
            return None
        qty = bal["gold_qty"]
        logger.info(f"전량 매도: {qty}g")
        return self.send_order("SELL", code, qty, ord_tp="3")

    # ─────────────────────────────────────────────────────
    # 장마감 동시호가 + 미체결 처리
    # ─────────────────────────────────────────────────────
    def _get_limit_price(self, code: str, order_type: str) -> int:
        """동시호가 체결 보장을 위한 상한가(BUY)/하한가(SELL) 계산. 금 호가단위: 10원."""
        price = self.get_current_price(code)
        if not price or price <= 0:
            return 0
        tick = 10
        if order_type == "BUY":
            raw = int(price * 1.30)
            return (raw // tick) * tick
        else:
            raw = int(price * 0.70)
            return ((raw + tick - 1) // tick) * tick

    def execute_closing_auction_buy(self, code: str, qty: float) -> dict:
        """
        장마감 동시호가 매수 (금현물).
        1. 상한가 지정가 주문 (ord_tp="1") → 동시호가 참여
        2. 90초 대기 후 미체결 확인
        3. 미체결 → 취소 + 로그 (금현물은 시간외 미지원)
        """
        import time as _time

        if qty <= 0:
            return {"success": False, "msg": "수량 0"}

        limit_price = self._get_limit_price(code, "BUY")
        if limit_price <= 0:
            return {"success": False, "msg": "현재가 조회 실패"}

        logger.info(f"[동시호가 매수] {code} {qty}g @ 상한가 {limit_price:,}원")
        phase1 = self.send_order("BUY", code, qty, price=limit_price, ord_tp="1")

        if not phase1 or not phase1.get("success"):
            logger.error(f"[동시호가 매수] 주문 실패: {phase1}")
            return {"success": False, "phase1_result": phase1, "filled_qty": 0,
                    "remaining_qty": qty, "method": "closing_auction_failed"}

        ord_no = phase1.get("ord_no", "")
        logger.info(f"[동시호가 매수] 주문 접수. 주문번호={ord_no}. 체결 대기 90초...")
        _time.sleep(90)

        pending = self.get_pending_orders(code)
        unfilled = [p for p in pending if p["ord_no"] == ord_no and p["remaining_qty"] > 0]

        if not unfilled:
            logger.info(f"[동시호가 매수] 전량 체결 완료: {qty}g")
            return {"success": True, "phase1_result": phase1,
                    "filled_qty": qty, "remaining_qty": 0, "method": "closing_auction"}

        remaining = unfilled[0]["remaining_qty"]
        filled = qty - remaining
        logger.warning(f"[동시호가 매수] 미체결 {remaining}g (체결 {filled}g). "
                       f"금현물은 시간외 미지원 → 취소")
        self.cancel_order(ord_no, code, remaining)
        _time.sleep(3)

        return {"success": filled > 0, "phase1_result": phase1,
                "filled_qty": filled, "remaining_qty": remaining,
                "method": "closing_auction_partial",
                "note": "금현물 시간외 매매 미지원. 미체결분은 다음 거래일 처리 필요."}

    def execute_closing_auction_sell(self, code: str, qty: float) -> dict:
        """
        장마감 동시호가 매도 (금현물).
        1. 하한가 지정가 주문 (ord_tp="1") → 동시호가 참여
        2. 90초 대기 후 미체결 확인
        3. 미체결 → 취소 + 로그
        """
        import time as _time

        if qty <= 0:
            return {"success": False, "msg": "수량 0"}

        limit_price = self._get_limit_price(code, "SELL")
        if limit_price <= 0:
            return {"success": False, "msg": "현재가 조회 실패"}

        logger.info(f"[동시호가 매도] {code} {qty}g @ 하한가 {limit_price:,}원")
        phase1 = self.send_order("SELL", code, qty, price=limit_price, ord_tp="1")

        if not phase1 or not phase1.get("success"):
            logger.error(f"[동시호가 매도] 주문 실패: {phase1}")
            return {"success": False, "phase1_result": phase1, "filled_qty": 0,
                    "remaining_qty": qty, "method": "closing_auction_failed"}

        ord_no = phase1.get("ord_no", "")
        logger.info(f"[동시호가 매도] 주문 접수. 주문번호={ord_no}. 체결 대기 90초...")
        _time.sleep(90)

        pending = self.get_pending_orders(code)
        unfilled = [p for p in pending if p["ord_no"] == ord_no and p["remaining_qty"] > 0]

        if not unfilled:
            logger.info(f"[동시호가 매도] 전량 체결 완료: {qty}g")
            return {"success": True, "phase1_result": phase1,
                    "filled_qty": qty, "remaining_qty": 0, "method": "closing_auction"}

        remaining = unfilled[0]["remaining_qty"]
        filled = qty - remaining
        logger.warning(f"[동시호가 매도] 미체결 {remaining}g (체결 {filled}g). "
                       f"금현물은 시간외 미지원 → 취소")
        self.cancel_order(ord_no, code, remaining)
        _time.sleep(3)

        return {"success": filled > 0, "phase1_result": phase1,
                "filled_qty": filled, "remaining_qty": remaining,
                "method": "closing_auction_partial",
                "note": "금현물 시간외 매매 미지원. 미체결분은 다음 거래일 처리 필요."}

    # ─────────────────────────────────────────────────────
    # 동시호가 스마트 래퍼
    # ─────────────────────────────────────────────────────
    def smart_buy_krw_closing(self, code: str = GOLD_CODE_1KG, krw_amount: float = 0) -> dict | None:
        """KRW 금액 기준 동시호가 매수."""
        price = self.get_current_price(code)
        if not price or price <= 0:
            logger.error("현재가 조회 실패로 동시호가 매수 취소.")
            return None
        qty = round(krw_amount / price, 2)
        if qty <= 0:
            return None
        return self.execute_closing_auction_buy(code, qty)

    def smart_sell_all_closing(self, code: str = GOLD_CODE_1KG) -> dict | None:
        """전량 동시호가 매도."""
        bal = self.get_balance()
        if not bal or bal["gold_qty"] <= 0:
            logger.warning("동시호가 매도 실패: 보유 금 없음.")
            return None
        return self.execute_closing_auction_sell(code, bal["gold_qty"])


# ─────────────────────────────────────────────────────────
# 테스트 실행
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    trader = KiwoomGoldTrader(is_mock=False)

    print("=== 키움 금현물 API 테스트 ===")
    print(f"App Key: {trader.app_key[:8]}..." if trader.app_key else "App Key: 없음")

    # 1. 인증 테스트
    print("\n[1] 인증 테스트...")
    if trader.auth():
        print(f"  ✅ 인증 성공! 토큰: {trader.access_token[:20]}...")
    else:
        print("  ❌ 인증 실패")

    # 2. 일봉 조회 테스트
    print("\n[2] 금현물 일봉 조회 (최근 10거래일)...")
    df = trader.get_daily_chart(count=10)
    if df is not None:
        print(df.tail(5).to_string())
    else:
        print("  ❌ 일봉 데이터 없음 (API 연결 필요)")

    # 3. 잔고 조회 테스트
    print("\n[3] 잔고 조회...")
    bal = trader.get_balance()
    if bal:
        print(f"  예수금: {bal['cash_krw']:,.0f}원")
        print(f"  금 보유: {bal['gold_qty']}g (평가 {bal['gold_eval']:,.0f}원)")
    else:
        print("  ❌ 잔고 조회 실패")
