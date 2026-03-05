import sqlite3
import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import threading

class DBManager:
    _instance = None
    _lock = threading.Lock()
    _local = threading.local()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DBManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized') and self.initialized:
            return
        
        # 기본 경로 설정
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.db_dir = self.base_dir / "cache"
        self.db_path = self.db_dir / "trading_system.db"
        
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.initialized = True

    def _get_conn(self):
        """스레드별 독립적인 커넥션 반환 (WAL 모드 활성)"""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        with self._get_conn() as conn:
            # 1. 앱 설정 테이블 (JSON 저장)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. 계산 결과 캐시 테이블 (Optimization, Precompute 등)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compute_cache (
                    signature TEXT NOT NULL,
                    combo_key TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(signature, combo_key)
                )
            """)
            
            # 3. OHLCV 가격 데이터 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    ticker TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    date TIMESTAMP NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY(ticker, interval, date)
                )
            """)
            
            # 4. ISA 백테스트 진행상황 관리
            conn.execute("""
                CREATE TABLE IF NOT EXISTS isa_live_runs (
                    signature TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    status TEXT,
                    target_count INTEGER,
                    done_count INTEGER,
                    meta_json TEXT
                )
            """)
            
            conn.commit()

    # ── ISA Live Runs 관련 ──
    def save_isa_run(self, signature, status, target_count, done_count, meta_dict=None):
        with self._get_conn() as conn:
            now = datetime.now().isoformat()
            meta_json = json.dumps(meta_dict) if meta_dict else None
            conn.execute("""
                INSERT INTO isa_live_runs (signature, created_at, updated_at, status, target_count, done_count, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    status = excluded.status,
                    done_count = excluded.done_count,
                    meta_json = COALESCE(excluded.meta_json, isa_live_runs.meta_json)
            """, (signature, now, now, status, target_count, done_count, meta_json))
            conn.commit()

    def get_isa_run(self, signature):
        cursor = self._get_conn().cursor()
        cursor.execute("SELECT * FROM isa_live_runs WHERE signature = ?", (signature,))
        row = cursor.fetchone()
        if row:
            columns = [col[0] for col in cursor.description]
            return dict(zip(columns, row))
        return None
    def save_setting(self, key, value_dict):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value_dict, ensure_ascii=False), datetime.now().isoformat())
            )
            conn.commit()

    def load_setting(self, key, default=None):
        cursor = self._get_conn().cursor()
        cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return default

    # ── 계산 캐시 (Compute Cache) 관련 ──
    def save_compute_result(self, signature, combo_key, payload_bytes):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO compute_cache (signature, combo_key, payload, updated_at) VALUES (?, ?, ?, ?)",
                (signature, combo_key, payload_bytes, datetime.now().isoformat())
            )
            conn.commit()

    def get_compute_result(self, signature, combo_key):
        cursor = self._get_conn().cursor()
        cursor.execute("SELECT payload FROM compute_cache WHERE signature = ? AND combo_key = ?", (signature, combo_key))
        row = cursor.fetchone()
        return row[0] if row else None

    def get_existing_keys(self, signature):
        cursor = self._get_conn().cursor()
        cursor.execute("SELECT combo_key FROM compute_cache WHERE signature = ?", (signature,))
        return {row[0] for row in cursor.fetchall()}

    def clear_compute_cache(self, signature=None):
        with self._get_conn() as conn:
            if signature:
                conn.execute("DELETE FROM compute_cache WHERE signature = ?", (signature,))
            else:
                conn.execute("DELETE FROM compute_cache")
            conn.commit()

    # ── 가격 데이터 (Price Data) 관련 ──
    def save_ohlcv(self, ticker, interval, df):
        if df is None or df.empty:
            return
        
        # DataFrame을 SQL로 변환하기 위해 준비
        temp_df = df.copy()
        if temp_df.index.name != 'date':
            temp_df.index.name = 'date'
        temp_df = temp_df.reset_index()
        temp_df['ticker'] = ticker
        temp_df['interval'] = interval
        
        # 열 매핑 (소문자 통일)
        temp_df.columns = [c.lower() for c in temp_df.columns]
        
        # 필요한 열만 필터링
        cols = ['ticker', 'interval', 'date', 'open', 'high', 'low', 'close', 'volume']
        temp_df = temp_df[[c for c in cols if c in temp_df.columns]].copy()

        if temp_df.empty:
            return

        # date 정규화 + 중복 제거 (PRIMARY KEY: ticker, interval, date)
        temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
        temp_df = temp_df.dropna(subset=["date"])
        if temp_df.empty:
            return
        temp_df["date"] = temp_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        temp_df = temp_df.drop_duplicates(subset=["ticker", "interval", "date"], keep="last")

        # SQLite UPSERT (동일 키는 교체)
        rows = list(
            zip(
                temp_df["ticker"].astype(str),
                temp_df["interval"].astype(str),
                temp_df["date"].astype(str),
                pd.to_numeric(temp_df.get("open"), errors="coerce"),
                pd.to_numeric(temp_df.get("high"), errors="coerce"),
                pd.to_numeric(temp_df.get("low"), errors="coerce"),
                pd.to_numeric(temp_df.get("close"), errors="coerce"),
                pd.to_numeric(temp_df.get("volume"), errors="coerce"),
            )
        )
        if not rows:
            return

        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_data (ticker, interval, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def load_ohlcv(self, ticker, interval):
        query = "SELECT date, open, high, low, close, volume FROM price_data WHERE ticker = ? AND interval = ? ORDER BY date"
        df = pd.read_sql_query(query, self._get_conn(), params=(ticker, interval), parse_dates=['date'])
        if not df.empty:
            df.set_index('date', inplace=True)
        return df

    def get_last_date(self, ticker, interval):
        cursor = self._get_conn().cursor()
        cursor.execute("SELECT MAX(date) FROM price_data WHERE ticker = ? AND interval = ?", (ticker, interval))
        row = cursor.fetchone()
        return pd.to_datetime(row[0]) if row and row[0] else None

    def get_first_date(self, ticker, interval):
        cursor = self._get_conn().cursor()
        cursor.execute("SELECT MIN(date) FROM price_data WHERE ticker = ? AND interval = ?", (ticker, interval))
        row = cursor.fetchone()
        return pd.to_datetime(row[0]) if row and row[0] else None
