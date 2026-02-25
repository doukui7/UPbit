import pyupbit
import time
import logging
from strategy.sma import SMAStrategy
import data_cache

logger = logging.getLogger(__name__)

# 시간봉별 실행 설정
INTERVAL_EXEC_CONFIG = {
    "day":       {"splits": 3, "wait_sec": 60,  "timeout_sec": 600, "fallback_market": True},
    "minute240": {"splits": 3, "wait_sec": 30,  "timeout_sec": 300, "fallback_market": True},
    "minute60":  {"splits": 2, "wait_sec": 20,  "timeout_sec": 120, "fallback_market": True},
    "minute30":  {"splits": 2, "wait_sec": 10,  "timeout_sec": 60,  "fallback_market": True},
    "minute15":  {"splits": 1, "wait_sec": 5,   "timeout_sec": 30,  "fallback_market": True},
    "minute5":   {"splits": 1, "wait_sec": 3,   "timeout_sec": 15,  "fallback_market": True},
    "minute1":   {"splits": 1, "wait_sec": 2,   "timeout_sec": 10,  "fallback_market": True},
}

# Upbit 호가 단위 (KRW 마켓)
def get_tick_size(price):
    """Upbit KRW 마켓 호가 단위 반환"""
    if price >= 2000000: return 1000
    elif price >= 1000000: return 500
    elif price >= 500000: return 100
    elif price >= 100000: return 50
    elif price >= 10000: return 10
    elif price >= 1000: return 5
    elif price >= 100: return 1
    elif price >= 10: return 0.1
    elif price >= 1: return 0.01
    else: return 0.001

def round_to_tick(price, tick_size):
    """호가 단위에 맞게 반올림"""
    return round(price / tick_size) * tick_size


class UpbitTrader:
    def __init__(self, access_key, secret_key):
        self.access = access_key
        self.secret = secret_key
        self.upbit = pyupbit.Upbit(access_key, secret_key)
        self.strategy = SMAStrategy()
        self.execution_log = []  # 체결 로그 기록

    def get_orders(self, ticker=None, state='wait'):
        try:
            if ticker:
                return self.upbit.get_order(ticker, state=state)
            else:
                return self.upbit.get_orders(state=state)
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return None

    def get_history(self, kind="deposit", currency=None):
        """입출금/체결 내역 조회. Upbit API 직접 호출로 안정성 확보.
        currency=None이면 전체 화폐 조회.
        Returns: (data_list, error_string_or_None)
        """
        import jwt
        import uuid as _uuid
        import hashlib
        from urllib.parse import urlencode
        import requests

        server_url = "https://api.upbit.com"

        def _auth_header(query_params=None):
            payload = {
                'access_key': self.upbit.access if hasattr(self.upbit, 'access') else '',
                'nonce': str(_uuid.uuid4()),
            }
            if query_params:
                query_string = urlencode(query_params, doseq=True)
                m = hashlib.sha512()
                m.update(query_string.encode())
                payload['query_hash'] = m.hexdigest()
                payload['query_hash_alg'] = 'SHA512'
            token = jwt.encode(payload, self.upbit.secret if hasattr(self.upbit, 'secret') else '')
            return {"Authorization": f"Bearer {token}"}

        try:
            if kind == 'deposit':
                params = {"limit": 100, "order_by": "desc"}
                if currency:
                    params["currency"] = currency
                res = requests.get(f"{server_url}/v1/deposits", params=params, headers=_auth_header(params))
                if res.status_code == 200:
                    return res.json(), None
                err = f"Deposit API {res.status_code}: {res.text}"
                logger.error(err)
                return [], err

            elif kind == 'withdraw':
                params = {"limit": 100, "order_by": "desc"}
                if currency:
                    params["currency"] = currency
                res = requests.get(f"{server_url}/v1/withdraws", params=params, headers=_auth_header(params))
                if res.status_code == 200:
                    return res.json(), None
                err = f"Withdraw API {res.status_code}: {res.text}"
                logger.error(err)
                return [], err

            elif kind == 'order':
                # 체결 완료 주문 조회 - states[] 배열 형식 사용 (Upbit API 필수)
                params = {"states[]": ["done", "cancel"], "limit": 100, "order_by": "desc"}
                if currency and currency != "KRW":
                    params["market"] = f"KRW-{currency}"
                res = requests.get(f"{server_url}/v1/orders/closed", params=params, headers=_auth_header(params))
                if res.status_code == 200:
                    return res.json(), None
                # fallback: /v1/orders 엔드포인트
                res2 = requests.get(f"{server_url}/v1/orders", params=params, headers=_auth_header(params))
                if res2.status_code == 200:
                    return res2.json(), None
                err = f"Order API {res.status_code}: {res.text} / fallback {res2.status_code}: {res2.text}"
                logger.error(err)
                return [], err

            else:
                return [], f"Unknown kind: {kind}"
        except Exception as e:
            err = f"Error fetching {kind} history: {str(e)}"
            logger.error(err)
            return [], err

    def get_balance(self, ticker="KRW"):
        try:
            return self.upbit.get_balance(ticker)
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0

    def get_all_balances(self):
        """모든 잔고를 한 번의 API 호출로 조회. Returns: {currency: balance}"""
        try:
            balances = self.upbit.get_balances()
            result = {}
            if balances:
                for b in balances:
                    currency = b.get('currency', '')
                    balance = float(b.get('balance', 0)) + float(b.get('locked', 0))
                    result[currency] = balance
            return result
        except Exception as e:
            logger.error(f"Error getting all balances: {e}")
            return {}

    def get_current_price(self, ticker):
        try:
            return data_cache.get_current_price_local_first(ticker, ttl_sec=3.0, allow_api_fallback=True)
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None

    def get_orderbook(self, ticker):
        """호가창 조회"""
        try:
            return data_cache.get_orderbook_cached(ticker, ttl_sec=2.0, allow_api_fallback=True)
        except Exception as e:
            logger.error(f"Error getting orderbook: {e}")
            return None

    def buy_market(self, ticker, price_amount):
        try:
            return self.upbit.buy_market_order(ticker, price_amount)
        except Exception as e:
            return {"error": str(e)}

    def sell_market(self, ticker, volume):
        try:
            return self.upbit.sell_market_order(ticker, volume)
        except Exception as e:
            return {"error": str(e)}

    def buy_limit(self, ticker, price, volume):
        """지정가 매수"""
        try:
            return self.upbit.buy_limit_order(ticker, price, volume)
        except Exception as e:
            return {"error": str(e)}

    def sell_limit(self, ticker, price, volume):
        """지정가 매도"""
        try:
            return self.upbit.sell_limit_order(ticker, price, volume)
        except Exception as e:
            return {"error": str(e)}

    def cancel_order(self, uuid):
        """주문 취소"""
        try:
            return self.upbit.cancel_order(uuid)
        except Exception as e:
            return {"error": str(e)}

    def get_order_detail(self, uuid):
        """주문 상세 조회"""
        try:
            return self.upbit.get_individual_order(uuid)
        except Exception as e:
            return {"error": str(e)}

    def _wait_for_fill(self, uuid, timeout_sec=60, check_interval=3):
        """주문 체결 대기. 체결되면 True, 타임아웃이면 False"""
        elapsed = 0
        while elapsed < timeout_sec:
            time.sleep(check_interval)
            elapsed += check_interval
            detail = self.get_order_detail(uuid)
            if detail and isinstance(detail, dict):
                state = detail.get('state', '')
                if state == 'done':
                    return True, detail
                elif state == 'cancel':
                    return False, detail
        return False, None

    # ========================================================
    # 스마트 주문 실행: 지정가 분할 주문 + 시간봉 대응
    # ========================================================
    def smart_buy(self, ticker, krw_amount, interval="day", price_offset_pct=0.02):
        """
        지정가 분할 매수.
        - ticker: KRW-BTC 등
        - krw_amount: 매수할 KRW 금액
        - interval: 시간봉 (실행 전략 결정)
        - price_offset_pct: 현재가 대비 지정가 오프셋 (0.02 = 0.02% 위)
        Returns: {"filled_amount": 총 체결 수량, "avg_price": 평균 체결가, "logs": [...]}
        """
        config = INTERVAL_EXEC_CONFIG.get(interval, INTERVAL_EXEC_CONFIG["day"])
        splits = config["splits"]
        wait_sec = config["wait_sec"]
        timeout_sec = config["timeout_sec"]
        fallback = config["fallback_market"]

        per_split = krw_amount / splits
        total_filled_volume = 0
        total_filled_krw = 0
        logs = []
        remaining_krw = krw_amount

        for i in range(splits):
            if remaining_krw < 5000:
                break

            amount = min(per_split, remaining_krw)
            current_price = self.get_current_price(ticker)
            if not current_price:
                logs.append({"split": i+1, "status": "error", "msg": "가격 조회 실패"})
                break

            # 호가 1호가 위 지정가 (매수 유리 가격)
            ob = self.get_orderbook(ticker)
            if ob and ob.get('orderbook_units'):
                best_ask = ob['orderbook_units'][0]['ask_price']
                limit_price = best_ask  # 매도 1호가에 맞춤
            else:
                # Fallback: 현재가 + offset
                tick = get_tick_size(current_price)
                limit_price = round_to_tick(current_price * (1 + price_offset_pct / 100), tick)

            volume = amount / limit_price
            logger.info(f"[{ticker}] Split {i+1}/{splits}: Limit BUY {volume:.6f} @ {limit_price:,.0f}")

            result = self.buy_limit(ticker, limit_price, volume)
            if isinstance(result, dict) and result.get('error'):
                logs.append({"split": i+1, "status": "error", "msg": result['error']})
                break

            uuid = result.get('uuid') if isinstance(result, dict) else None
            if not uuid:
                logs.append({"split": i+1, "status": "error", "msg": "주문 UUID 없음"})
                break

            # 체결 대기
            per_timeout = timeout_sec // splits
            filled, detail = self._wait_for_fill(uuid, timeout_sec=per_timeout, check_interval=wait_sec)

            if filled and detail:
                exec_vol = float(detail.get('executed_volume', 0))
                exec_price = float(detail.get('price', limit_price)) if detail.get('trades') is None else limit_price
                # trades에서 실제 체결 평균가 계산
                trades = detail.get('trades', [])
                if trades:
                    t_krw = sum(float(t['funds']) for t in trades)
                    t_vol = sum(float(t['volume']) for t in trades)
                    exec_price = t_krw / t_vol if t_vol > 0 else limit_price
                    total_filled_krw += t_krw
                else:
                    total_filled_krw += exec_vol * limit_price

                total_filled_volume += exec_vol
                remaining_krw -= exec_vol * limit_price
                logs.append({
                    "split": i+1, "status": "filled", "volume": exec_vol,
                    "price": exec_price, "limit_price": limit_price
                })
            else:
                # 미체결 → 취소
                self.cancel_order(uuid)
                # 잔여 확인
                detail2 = self.get_order_detail(uuid)
                partial_vol = float(detail2.get('executed_volume', 0)) if detail2 else 0
                if partial_vol > 0:
                    total_filled_volume += partial_vol
                    total_filled_krw += partial_vol * limit_price
                    remaining_krw -= partial_vol * limit_price

                logs.append({
                    "split": i+1, "status": "timeout_cancelled",
                    "partial_volume": partial_vol, "limit_price": limit_price
                })

        # 잔여금이 있으면 시장가 마무리
        if remaining_krw >= 5000 and fallback:
            logger.info(f"[{ticker}] Fallback market buy: {remaining_krw:,.0f} KRW")
            result = self.buy_market(ticker, remaining_krw * 0.999)
            if isinstance(result, dict) and not result.get('error'):
                # 시장가는 즉시 체결
                time.sleep(2)
                current_price = self.get_current_price(ticker)
                est_vol = remaining_krw / current_price if current_price else 0
                total_filled_volume += est_vol
                total_filled_krw += remaining_krw
                logs.append({"split": "market_fallback", "status": "filled", "krw": remaining_krw})

        avg_price = total_filled_krw / total_filled_volume if total_filled_volume > 0 else 0

        exec_result = {
            "type": "buy",
            "ticker": ticker,
            "filled_volume": total_filled_volume,
            "avg_price": avg_price,
            "total_krw": total_filled_krw,
            "splits_used": len(logs),
            "logs": logs
        }
        self.execution_log.append(exec_result)
        return exec_result

    def smart_sell(self, ticker, volume, interval="day", price_offset_pct=0.02):
        """
        지정가 분할 매도.
        - ticker: KRW-BTC 등
        - volume: 매도할 코인 수량
        - interval: 시간봉 (실행 전략 결정)
        - price_offset_pct: 현재가 대비 지정가 오프셋
        Returns: {"filled_krw": 총 체결 금액, "avg_price": 평균 체결가, "logs": [...]}
        """
        config = INTERVAL_EXEC_CONFIG.get(interval, INTERVAL_EXEC_CONFIG["day"])
        splits = config["splits"]
        wait_sec = config["wait_sec"]
        timeout_sec = config["timeout_sec"]
        fallback = config["fallback_market"]

        per_split = volume / splits
        total_filled_volume = 0
        total_filled_krw = 0
        logs = []
        remaining_volume = volume

        for i in range(splits):
            current_price = self.get_current_price(ticker)
            if not current_price or remaining_volume * current_price < 5000:
                break

            sell_vol = min(per_split, remaining_volume)

            # 호가 1호가 아래 지정가 (매도 유리 가격)
            ob = self.get_orderbook(ticker)
            if ob and ob.get('orderbook_units'):
                best_bid = ob['orderbook_units'][0]['bid_price']
                limit_price = best_bid  # 매수 1호가에 맞춤
            else:
                tick = get_tick_size(current_price)
                limit_price = round_to_tick(current_price * (1 - price_offset_pct / 100), tick)

            logger.info(f"[{ticker}] Split {i+1}/{splits}: Limit SELL {sell_vol:.6f} @ {limit_price:,.0f}")

            result = self.sell_limit(ticker, limit_price, sell_vol)
            if isinstance(result, dict) and result.get('error'):
                logs.append({"split": i+1, "status": "error", "msg": result['error']})
                break

            uuid = result.get('uuid') if isinstance(result, dict) else None
            if not uuid:
                logs.append({"split": i+1, "status": "error", "msg": "주문 UUID 없음"})
                break

            per_timeout = timeout_sec // splits
            filled, detail = self._wait_for_fill(uuid, timeout_sec=per_timeout, check_interval=wait_sec)

            if filled and detail:
                exec_vol = float(detail.get('executed_volume', 0))
                trades = detail.get('trades', [])
                if trades:
                    t_krw = sum(float(t['funds']) for t in trades)
                    t_vol = sum(float(t['volume']) for t in trades)
                    exec_price = t_krw / t_vol if t_vol > 0 else limit_price
                    total_filled_krw += t_krw
                else:
                    exec_price = limit_price
                    total_filled_krw += exec_vol * limit_price

                total_filled_volume += exec_vol
                remaining_volume -= exec_vol
                logs.append({
                    "split": i+1, "status": "filled", "volume": exec_vol,
                    "price": exec_price, "limit_price": limit_price
                })
            else:
                self.cancel_order(uuid)
                detail2 = self.get_order_detail(uuid)
                partial_vol = float(detail2.get('executed_volume', 0)) if detail2 else 0
                if partial_vol > 0:
                    total_filled_volume += partial_vol
                    total_filled_krw += partial_vol * limit_price
                    remaining_volume -= partial_vol

                logs.append({
                    "split": i+1, "status": "timeout_cancelled",
                    "partial_volume": partial_vol, "limit_price": limit_price
                })

        # 잔여 수량 시장가 매도
        current_price = self.get_current_price(ticker) or 0
        if remaining_volume > 0 and remaining_volume * current_price >= 5000 and fallback:
            logger.info(f"[{ticker}] Fallback market sell: {remaining_volume:.6f}")
            result = self.sell_market(ticker, remaining_volume)
            if isinstance(result, dict) and not result.get('error'):
                time.sleep(2)
                est_krw = remaining_volume * current_price
                total_filled_volume += remaining_volume
                total_filled_krw += est_krw
                logs.append({"split": "market_fallback", "status": "filled", "volume": remaining_volume})

        avg_price = total_filled_krw / total_filled_volume if total_filled_volume > 0 else 0

        exec_result = {
            "type": "sell",
            "ticker": ticker,
            "filled_volume": total_filled_volume,
            "avg_price": avg_price,
            "total_krw": total_filled_krw,
            "splits_used": len(logs),
            "logs": logs
        }
        self.execution_log.append(exec_result)
        return exec_result

    def adaptive_buy(self, ticker, total_amount_krw, interval="minute240"):
        """
        Adaptive Buy Strategy:
        - Gets Target Price (Current Candle Open).
        - Checks slippage every 10 mins.
        - If slippage > 0.1%, buys 10% and waits.
        - If slippage <= 0.1%, buys remaining.
        - Max 50 min loop (to avoid Action timeout).
        """
        # 1. Determine Target Price (Current Candle Open)
        try:
            df = data_cache.get_ohlcv_local_first(
                ticker,
                interval=interval,
                count=1,
                allow_api_fallback=True,
            )
            target_price = df['open'].iloc[0]
        except Exception as e:
            logger.error(f"Failed to get target price (OHLCV): {e}")
            target_price = self.get_current_price(ticker)

        if not target_price:
            logger.error(f"[{ticker}] Failed to get any price. Fallback to Smart Buy.")
            return self.smart_buy(ticker, total_amount_krw, interval)

        logger.info(f"[{ticker}] Adaptive Buy Start. Target(Open)={target_price:,.0f} KRW, Budget={total_amount_krw:,.0f} KRW")
        
        remaining_krw = total_amount_krw
        start_time = time.time()
        
        iteration = 0
        logs = []
        total_filled_volume = 0
        total_filled_krw = 0

        # Loop: 0, 10, 20, 30, 40 (Final check) -> Max 50 min
        # GitHub Action timeout is usually 60 min, so 50 min is safe.
        while (time.time() - start_time) < 3300: # 55 min limit
            iteration += 1
            current_price = self.get_current_price(ticker)
            
            if not current_price:
                logger.warning(f"[{ticker}] Price check failed. Waiting...")
                time.sleep(10)
                continue

            slippage_pct = (current_price - target_price) / target_price * 100
            logger.info(f"[{ticker}] Iter {iteration}: Price={current_price:,.0f}, Slip={slippage_pct:+.3f}%")

            buy_now = False
            amount_to_exec = 0

            # Condition 1: Favorable or Small Slippage (<= 0.1%) -> Buy ALL
            if slippage_pct <= 0.1:
                logger.info(f"[{ticker}] Slippage OK ({slippage_pct:.3f}%). Buying ALL remaining.")
                buy_now = True
                amount_to_exec = remaining_krw
            
            # Condition 2: High Slippage (> 0.1%) -> Buy 10% Only
            else:
                logger.info(f"[{ticker}] Slippage High ({slippage_pct:.3f}%). Buying 10% only.")
                # If remaining is small, just buy all (avoid dust)
                if remaining_krw < 10000:
                    buy_now = True
                    amount_to_exec = remaining_krw
                else:
                    buy_now = True
                    amount_to_exec = total_amount_krw * 0.1 # 10% of INITIAL total
                    if amount_to_exec > remaining_krw:
                        amount_to_exec = remaining_krw

            if buy_now and amount_to_exec >= 5000:
                # Use Smart Buy for the chunk (Limit Order logic)
                # Note: Smart Buy handles splits internally. 
                # For small 10% chunk, it might not split much.
                # using interval='minute1' to force quick execution for the chunk?
                # No, stick to original interval config or use tight setting.
                # Let's use current interval config but maybe reduced timeout?
                # Actually, smart_buy is robust.
                
                res = self.smart_buy(ticker, amount_to_exec, interval=interval)
                
                filled = res.get('filled_volume', 0)
                spent = res.get('total_krw', 0)
                avg = res.get('avg_price', 0)
                
                remaining_krw -= spent
                total_filled_volume += filled
                total_filled_krw += spent
                
                logs.append({
                    "iter": iteration, "slippage": slippage_pct, 
                    "action": "buy_all" if amount_to_exec == remaining_krw else "buy_10%",
                    "spent": spent, "avg": avg
                })

                if remaining_krw < 5000:
                    logger.info(f"[{ticker}] Buy Complete.")
                    break

            # If we bought ALL, break
            if remaining_krw < 5000:
                break
            
            # If we bought 10% (High Slippage), Wait 10 min
            logger.info(f"[{ticker}] Waiting 10 minutes... (Elapsed: {(time.time()-start_time)/60:.1f}m)")
            time.sleep(600) 

        # Timeout / Loop End
        if remaining_krw >= 5000:
            logger.info(f"[{ticker}] Timeout. Buying remaining {remaining_krw:,.0f} KRW at Market/Smart")
            res = self.smart_buy(ticker, remaining_krw, interval=interval)
            filled = res.get('filled_volume', 0)
            spent = res.get('total_krw', 0)
            remaining_krw -= spent
            total_filled_volume += filled
            total_filled_krw += spent

        avg_price = total_filled_krw / total_filled_volume if total_filled_volume > 0 else 0
        result = {
            "type": "buy_adaptive",
            "ticker": ticker,
            "filled_volume": total_filled_volume,
            "avg_price": avg_price,
            "total_krw": total_filled_krw,
            "logs": logs
        }
        self.execution_log.append(result)
        return result

    def get_execution_log(self):
        """체결 로그 반환"""
        return self.execution_log

    def get_done_orders(self, ticker=None):
        """체결 완료 주문 조회 (슬리피지 분석용)"""
        try:
            if ticker:
                return self.upbit.get_order(ticker, state='done')
            return self.upbit.get_orders(state='done')
        except Exception as e:
            logger.error(f"Error fetching done orders: {e}")
            return []

    def check_and_trade(self, ticker, interval="day", sma_period=20):
        """
        Check signal and execute trade.
        Returns trade result or status message.
        """
        count = max(200, sma_period * 3)

        df = data_cache.get_ohlcv_local_first(
            ticker,
            interval=interval,
            count=count,
            allow_api_fallback=True,
        )
        if df is None:
            return "Failed to fetch data"

        calc_periods = [sma_period]
        df = self.strategy.create_features(df, periods=calc_periods)

        previous_row = df.iloc[-2]
        current_signal = self.strategy.get_signal(previous_row, strategy_type='SMA_CROSS', ma_period=sma_period)

        krw_balance = self.get_balance("KRW")
        coin_currency = ticker.split('-')[1]
        coin_balance = self.get_balance(coin_currency)
        current_price = self.get_current_price(ticker)

        if current_price is None:
            return "Failed to get current price"

        min_order_amount = 5000

        if current_signal == 'BUY':
            if krw_balance > min_order_amount:
                amount_to_buy = krw_balance * 0.99
                result = self.smart_buy(ticker, amount_to_buy, interval=interval)
                return f"BUY EXECUTED (Smart): avg={result['avg_price']:,.0f}, vol={result['filled_volume']:.6f}"
            else:
                return "BUY SIGNAL but Insufficient KRW"

        elif current_signal == 'SELL':
            coin_value = coin_balance * current_price
            if coin_value > min_order_amount:
                result = self.smart_sell(ticker, coin_balance, interval=interval)
                return f"SELL EXECUTED (Smart): avg={result['avg_price']:,.0f}, krw={result['total_krw']:,.0f}"
            else:
                return "SELL SIGNAL but Insufficient Coin"

        return f"HOLD (Signal: {current_signal})"
