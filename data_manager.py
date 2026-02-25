import threading
import time
import datetime
import data_cache

class MarketDataWorker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(MarketDataWorker, cls).__new__(cls)
        return cls._instance

    def __init__(self, portfolio_list=None, interval='day'):
        if hasattr(self, 'initialized') and self.initialized:
            return

        self.portfolio_list = portfolio_list if portfolio_list else []
        self.interval = interval
        self.data_map = {} # {ticker|interval: dataframe}
        self.last_update_timestr = "Not Started"
        self.is_running = False
        self.worker_thread = None
        self.initialized = True
        self.status_msg = "Initialized"
        self._first_run = True

    def start_worker(self):
        with self._lock:
            if not self.is_running:
                self.is_running = True
                self.worker_thread = threading.Thread(target=self._job_loop, daemon=True)
                self.worker_thread.start()
                self.status_msg = "Worker Started"

    def update_config(self, portfolio_list):
        """Update the target portfolio dynamically"""
        self.portfolio_list = portfolio_list

    def _job_loop(self):
        while self.is_running:
            self.status_msg = "Fetching Data..."
            if not self.portfolio_list:
                time.sleep(2)
                continue

            # Identify unique (ticker, interval) pairs
            targets = set()
            for item in self.portfolio_list:
                ticker = f"{item['market']}-{item['coin'].upper()}"
                interval = item.get('interval', 'day')
                targets.add((ticker, interval))

            for ticker, interval in targets:
                try:
                    key = f"{ticker}|{interval}"

                    if self._first_run:
                        # 첫 실행: 로컬 캐시 우선 로드 (API 없이 즉시)
                        cached = data_cache.load_cached(ticker, interval)
                        if cached is not None and len(cached) > 100:
                            self.data_map[key] = cached
                            continue

                    # data_cache를 통한 증분 다운로드 (캐시 있으면 갭필만)
                    df = data_cache.get_ohlcv_cached(ticker, interval=interval, count=400)
                    if df is not None and len(df) > 0:
                        self.data_map[key] = df

                    time.sleep(0.1)
                except Exception as e:
                    print(f"Worker Error {ticker} ({interval}): {e}")

            self._first_run = False
            self.last_update_timestr = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_msg = f"Idle (Last: {self.last_update_timestr})"

            # Sleep before next cycle
            time.sleep(15)

    def get_data(self, ticker, interval='day'):
        key = f"{ticker}|{interval}"
        df = self.data_map.get(key)
        if df is not None:
            return df
        # 워커 데이터 없으면 로컬 캐시에서 직접 로드
        cached = data_cache.load_cached(ticker, interval)
        if cached is not None and len(cached) > 0:
            self.data_map[key] = cached
            return cached
        return None

    def get_status(self):
        return self.status_msg, self.last_update_timestr


class CoinTradingWorker:
    """
    코인 트레이딩 패널용 백그라운드 워커.
    선택된 티커의 가격/잔고/호가를 2초마다 병렬 갱신.
    UI는 .get()으로 즉시 캐시 읽기만 → 화면 멈춤 없음.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(CoinTradingWorker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self._data = {}
        self._trader = None       # UpbitTrader
        self._ticker = None       # "KRW-BTC"
        self._coin = None         # "BTC"
        self._running = False
        self._thread = None
        self.status_msg = "Not started"

    def configure(self, trader, ticker: str):
        """트레이더 + 현재 선택 티커 설정 (변경 시 즉시 반영)"""
        self._trader = trader
        if ticker != self._ticker:
            self._ticker = ticker
            self._coin = ticker.split("-")[1] if "-" in ticker else ticker
            # 티커 변경 시 호가/가격 캐시 초기화
            self._data.pop('orderbook', None)
            self._data.pop('price', None)

    def start(self):
        with self._lock:
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
                self.status_msg = "Started"

    def _loop(self):
        from concurrent.futures import ThreadPoolExecutor
        while self._running:
            trader = self._trader
            ticker = self._ticker
            coin = self._coin
            if not trader or not ticker:
                time.sleep(1)
                continue
            try:
                self.status_msg = "Fetching..."
                with ThreadPoolExecutor(max_workers=3) as pool:
                    f_price = pool.submit(
                        data_cache.get_current_price_local_first,
                        ticker,
                        3.0,
                        True,
                    )
                    f_krw = pool.submit(trader.get_balance, "KRW")
                    f_ob = pool.submit(
                        data_cache.get_orderbook_cached,
                        ticker,
                        2.0,
                        True,
                    )

                    try:
                        self._data['price'] = f_price.result(timeout=3) or 0
                    except Exception:
                        pass
                    try:
                        self._data['krw_bal'] = f_krw.result(timeout=3) or 0
                    except Exception:
                        pass
                    try:
                        self._data['orderbook'] = f_ob.result(timeout=3)
                    except Exception:
                        pass

                # 코인 잔고는 별도 (잔고 API 1초 제한 회피)
                try:
                    self._data['coin_bal'] = trader.get_balance(coin) or 0
                except Exception:
                    pass

                self._data['ticker'] = ticker
                self._data['last_update'] = time.time()
                self.status_msg = f"OK ({datetime.datetime.now().strftime('%H:%M:%S')})"
            except Exception as e:
                self.status_msg = f"Error: {e}"

            time.sleep(2)

    def get(self, key, default=None):
        """데이터 즉시 읽기 (블로킹 없음)"""
        return self._data.get(key, default)

    def invalidate(self, *keys):
        """거래 후 특정 키 무효화 (다음 루프에서 즉시 갱신)"""
        for k in keys:
            self._data.pop(k, None)

    def is_ready(self):
        return 'last_update' in self._data

    def get_age(self):
        lu = self._data.get('last_update', 0)
        return time.time() - lu if lu > 0 else float('inf')


class GoldDataWorker:
    """
    Gold 시세/잔고/호가를 백그라운드 스레드에서 주기적으로 갱신.
    UI 렌더링은 .get()으로 즉시 캐시 읽기만 하므로 화면 멈춤 없음.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(GoldDataWorker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        self._data = {}        # 공유 데이터 스토어
        self._trader = None
        self._running = False
        self._thread = None
        self._chart_df = None  # 일봉 캐시
        self._chart_updated = 0.0
        self._first_run = True
        self.status_msg = "Not started"

    def configure(self, trader):
        """KiwoomGoldTrader 인스턴스 설정"""
        self._trader = trader

    def start(self):
        """워커 시작 (이미 실행 중이면 무시)"""
        with self._lock:
            if not self._running and self._trader:
                # 첫 시작 시 parquet 캐시에서 즉시 로드 (API 없이)
                if self._first_run:
                    cached = data_cache.load_cached_gold()
                    if cached is not None and len(cached) > 10:
                        self._chart_df = cached
                        self._chart_updated = time.time()
                self._running = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
                self.status_msg = "Started"

    def stop(self):
        self._running = False

    def _loop(self):
        from concurrent.futures import ThreadPoolExecutor
        while self._running:
            if not self._trader:
                time.sleep(2)
                continue
            try:
                self.status_msg = "Fetching..."
                if not self._trader.auth():
                    self.status_msg = "Auth failed"
                    time.sleep(5)
                    continue

                # 잔고 + 현재가 + 호가를 병렬 조회 (최대 5초 타임아웃)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    f_bal = pool.submit(self._trader.get_balance)
                    f_price = pool.submit(
                        data_cache.get_gold_current_price_local_first,
                        self._trader,
                        "M04020000",
                        True,
                        8.0,
                    )
                    f_ob = pool.submit(self._trader.get_orderbook)

                    try:
                        self._data['balance'] = f_bal.result(timeout=5)
                    except Exception:
                        pass
                    try:
                        self._data['price'] = f_price.result(timeout=5) or 0
                    except Exception:
                        pass
                    try:
                        self._data['orderbook'] = f_ob.result(timeout=5)
                    except Exception:
                        pass

                self._data['last_update'] = time.time()
                self.status_msg = f"OK ({datetime.datetime.now().strftime('%H:%M:%S')})"

                # 일봉 차트: 5분마다 갱신 + parquet 캐시 저장
                now = time.time()
                if (now - self._chart_updated) > 300:
                    try:
                        df = data_cache.fetch_and_cache_gold(self._trader, code="M04020000", count=5000)
                        if df is not None and len(df) > 10:
                            self._chart_df = df
                            self._chart_updated = now
                            data_cache.save_cache_gold("M04020000", "day", df)
                    except Exception:
                        pass

                self._first_run = False

            except Exception as e:
                self.status_msg = f"Error: {e}"

            time.sleep(3)

    def get(self, key, default=None):
        """데이터 즉시 읽기 (블로킹 없음)"""
        return self._data.get(key, default)

    def get_chart(self):
        """일봉 차트 데이터 즉시 반환 (블로킹 없음)"""
        return self._chart_df

    def is_ready(self):
        """최소 1회 이상 데이터 갱신 완료 여부"""
        return 'last_update' in self._data

    def get_age(self):
        """마지막 갱신 후 경과 초"""
        lu = self._data.get('last_update', 0)
        return time.time() - lu if lu > 0 else float('inf')
