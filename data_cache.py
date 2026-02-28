"""
로컬 데이터 캐시: OHLCV 데이터를 parquet 파일로 저장/로드
- API 호출 최소화
- 최적화 시 즉시 데이터 사용 가능
"""
import os
import pandas as pd
import pyupbit
from datetime import datetime, timedelta
import threading

# ── yfinance SSL 인증서 경로 수정 (한글 경로 문제 해결) ──
# certifi 경로에 non-ASCII 문자가 있으면 curl이 인증서를 찾지 못함
try:
    import certifi as _certifi
    _cert_path = _certifi.where()
    _has_nonascii = any(ord(c) > 127 for c in _cert_path)
    if _has_nonascii:
        # TEMP 경로도 한글 포함 가능 → C:\temp 사용
        _safe_cert = "C:\\temp\\cacert.pem"
        if not os.path.isfile(_safe_cert):
            import shutil
            os.makedirs(os.path.dirname(_safe_cert), exist_ok=True)
            shutil.copy2(_cert_path, _safe_cert)
        for _env_key in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
            os.environ[_env_key] = _safe_cert
except Exception:
    pass

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# 시간봉별 하루 캔들 수 (갭 계산용)
CANDLES_PER_DAY = {
    "day": 1, "minute240": 6, "minute60": 24,
    "minute30": 48, "minute15": 96, "minute5": 288, "minute1": 1440
}

# 메모리 TTL 캐시 (현재가/호가)
_MEM_CACHE_LOCK = threading.Lock()
_PRICE_MEM_CACHE = {}      # key=ticker -> {"ts": float, "val": float}
_ORDERBOOK_MEM_CACHE = {}  # key=ticker -> {"ts": float, "val": dict}


def _normalize_ohlcv_df(df):
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    try:
        out.index = pd.to_datetime(out.index)
    except Exception:
        pass
    if "close" not in out.columns and "Close" in out.columns:
        out = out.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _estimate_gap_count(last_cached_time, interval):
    """캐시 마지막 시각부터 현재까지 필요한 캔들 수 추정"""
    now = datetime.now()
    if hasattr(last_cached_time, 'tz') and last_cached_time.tz is not None:
        last_cached_time = last_cached_time.tz_localize(None) if hasattr(last_cached_time, 'tz_localize') else last_cached_time.replace(tzinfo=None)
    gap_days = (now - last_cached_time).total_seconds() / 86400
    cpd = CANDLES_PER_DAY.get(interval, 1)
    return int(gap_days * cpd) + 10  # 여유분 +10


def _cache_path(ticker, interval):
    safe_name = ticker.replace("-", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_{interval}.parquet")


def load_cached(ticker, interval):
    """캐시된 데이터 로드. 없으면 None"""
    path = _cache_path(ticker, interval)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            return df
        except Exception:
            return None
    return None


def save_cache(ticker, interval, df):
    """데이터를 parquet으로 저장"""
    _ensure_cache_dir()
    path = _cache_path(ticker, interval)
    df.to_parquet(path)


def get_cache_info(ticker, interval):
    """캐시 파일 정보 반환"""
    path = _cache_path(ticker, interval)
    if os.path.exists(path):
        size = os.path.getsize(path)
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        try:
            df = pd.read_parquet(path)
            return {
                "exists": True,
                "path": path,
                "size_kb": size / 1024,
                "modified": mtime,
                "rows": len(df),
                "start": df.index[0],
                "end": df.index[-1]
            }
        except Exception:
            return {"exists": True, "path": path, "size_kb": size / 1024, "modified": mtime, "rows": 0}
    return {"exists": False}


def _chunked_download(ticker, interval, to, count, progress_callback=None):
    """200개씩 분할 다운로드 (내부 함수)"""
    import time as _time

    if count <= 200:
        df = pyupbit.get_ohlcv(ticker, interval=interval, to=to, count=count)
        if progress_callback and df is not None:
            progress_callback(len(df), count)
        return df

    all_chunks = []
    remaining = count
    current_to = to
    total_fetched = 0
    retries = 0
    prev_earliest = None

    while remaining > 0:
        fetch = min(remaining, 200)
        try:
            chunk = pyupbit.get_ohlcv(ticker, interval=interval, to=current_to, count=fetch)
        except Exception:
            retries += 1
            if retries > 5:
                break
            _time.sleep(2)
            continue

        if chunk is None or len(chunk) == 0:
            break

        got = len(chunk)
        new_earliest = chunk.index[0]

        if prev_earliest is not None and new_earliest >= prev_earliest:
            break

        all_chunks.append(chunk)
        total_fetched += got
        remaining -= got
        retries = 0
        prev_earliest = new_earliest

        earliest_naive = new_earliest.tz_localize(None) if new_earliest.tzinfo else new_earliest
        current_to = earliest_naive.strftime("%Y-%m-%d %H:%M:%S")

        if progress_callback:
            progress_callback(total_fetched, count)

        if got < fetch:
            break

        _time.sleep(0.1)

    if all_chunks:
        df_new = pd.concat(all_chunks)
        df_new = df_new[~df_new.index.duplicated(keep='last')]
        df_new = df_new.sort_index()
        return df_new
    return None


def fetch_and_cache(ticker, interval="day", to=None, count=10000, progress_callback=None):
    """
    API에서 데이터를 가져와 캐시에 저장.
    - 캐시가 있으면 증분 다운로드 (마지막 날짜 이후만)
    - 캐시가 없으면 전체 다운로드 (count만큼)
    progress_callback(fetched, total): 진행률 콜백
    """
    df_existing = load_cached(ticker, interval)

    if df_existing is not None and len(df_existing) > 0:
        # --- 증분 다운로드: 캐시 마지막 날짜 이후만 ---
        gap_count = _estimate_gap_count(df_existing.index[-1], interval)

        if gap_count <= 2:
            # 갭이 거의 없으면 스킵
            if progress_callback:
                progress_callback(0, 0)
            return df_existing

        df_new = _chunked_download(ticker, interval, to=to, count=gap_count, progress_callback=progress_callback)

        if df_new is not None and len(df_new) > 0:
            df_merged = pd.concat([df_existing, df_new])
            df_merged = df_merged[~df_merged.index.duplicated(keep='last')]
            df_merged = df_merged.sort_index()
            save_cache(ticker, interval, df_merged)
            return df_merged

        return df_existing
    else:
        # --- 캐시 없음: 전체 다운로드 ---
        df_new = _chunked_download(ticker, interval, to=to, count=count, progress_callback=progress_callback)

        if df_new is None or (hasattr(df_new, 'empty') and df_new.empty):
            return load_cached(ticker, interval)

        save_cache(ticker, interval, df_new)
        return df_new


def get_ohlcv_cached(ticker, interval="day", to=None, count=200, progress_callback=None):
    """
    캐시 우선 데이터 로드 (영구 캐시).
    - 캐시가 있으면 증분 다운로드 (추가된 날짜만)
    - 캐시가 없으면 전체 다운로드
    progress_callback(fetched, total): 다운로드 진행률 콜백
    """
    cached = load_cached(ticker, interval)

    if cached is not None and len(cached) > 0:
        # --- 캐시 존재 → 증분 갭필 ---
        gap_count = _estimate_gap_count(cached.index[-1], interval)
        if gap_count > 2:
            try:
                fetch_count = min(gap_count, 10000)
                df_new = _chunked_download(ticker, interval, to=to, count=fetch_count)
                if df_new is not None and len(df_new) > 0:
                    merged = pd.concat([cached, df_new])
                    merged = merged[~merged.index.duplicated(keep='last')]
                    merged = merged.sort_index()
                    if len(merged) > len(cached):
                        save_cache(ticker, interval, merged)
                        cached = merged
            except Exception:
                pass  # 갭필 실패해도 기존 캐시 사용

        # to 필터 적용
        if to:
            to_ts = pd.to_datetime(to)
            if cached.index.tz is not None and to_ts.tz is None:
                to_ts = to_ts.tz_localize(cached.index.tz)
            subset = cached[cached.index <= to_ts]
        else:
            subset = cached

        if len(subset) > 0:
            if len(subset) >= count:
                return subset.tail(count)
            return subset

    # --- 캐시 없음 → 전체 다운로드 ---
    return fetch_and_cache(ticker, interval, to=to, count=count, progress_callback=progress_callback)


def get_ohlcv_local_first(ticker, interval="day", to=None, count=200, allow_api_fallback=True, progress_callback=None):
    """
    로컬(parquet) 우선으로 OHLCV 조회.
    - 로컬에 충분하면 즉시 반환
    - 부족하고 allow_api_fallback=True면 API로 보강 후 반환
    """
    cached = _normalize_ohlcv_df(load_cached(ticker, interval))

    req_count = int(max(1, count))
    if cached is not None and len(cached) > 0:
        subset = cached
        if to:
            try:
                to_ts = pd.to_datetime(to)
                if subset.index.tz is not None and to_ts.tz is None:
                    to_ts = to_ts.tz_localize(subset.index.tz)
                subset = subset[subset.index <= to_ts]
            except Exception:
                pass
        if subset is not None and len(subset) > 0:
            if len(subset) >= req_count:
                return subset.tail(req_count)
            if not allow_api_fallback:
                return subset

    if not allow_api_fallback:
        return cached.tail(req_count) if cached is not None and len(cached) > 0 else None

    # 부족 시에는 count 기준으로 다시 내려받아 과거 구간도 보강
    try:
        fresh = _chunked_download(ticker, interval=interval, to=to, count=req_count, progress_callback=progress_callback)
        fresh = _normalize_ohlcv_df(fresh)
        if fresh is not None and not fresh.empty:
            merged = fresh
            if cached is not None and not cached.empty:
                merged = pd.concat([cached, fresh])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            save_cache(ticker, interval, merged)
            if to:
                try:
                    to_ts = pd.to_datetime(to)
                    if merged.index.tz is not None and to_ts.tz is None:
                        to_ts = to_ts.tz_localize(merged.index.tz)
                    merged = merged[merged.index <= to_ts]
                except Exception:
                    pass
            if merged is not None and not merged.empty:
                return merged.tail(req_count)
    except Exception:
        pass

    return get_ohlcv_cached(
        ticker,
        interval=interval,
        to=to,
        count=req_count,
        progress_callback=progress_callback,
    )


def _get_local_last_close_with_ts(ticker, intervals=None):
    ivs = list(intervals or [])
    if not ivs:
        ivs = ["minute1", "minute5", "minute15", "minute30", "minute60", "minute240", "day"]
    seen = set()
    for iv in ivs:
        if iv in seen:
            continue
        seen.add(iv)
        df = _normalize_ohlcv_df(load_cached(ticker, iv))
        if df is None or df.empty:
            continue
        if "close" in df.columns:
            try:
                p = float(df["close"].iloc[-1] or 0.0)
                if p > 0:
                    return p, pd.to_datetime(df.index[-1])
            except Exception:
                continue
    return 0.0, None


def _get_local_last_close(ticker, intervals=None):
    p, _ = _get_local_last_close_with_ts(ticker, intervals=intervals)
    return p


def _is_local_price_stale(last_ts, max_age_sec=900):
    if last_ts is None:
        return True
    try:
        now_ts = pd.Timestamp.now(tz=last_ts.tz) if getattr(last_ts, "tz", None) is not None else pd.Timestamp.now()
        age = (now_ts - last_ts).total_seconds()
        return age > float(max_age_sec)
    except Exception:
        return True


def get_current_price_local_first(ticker, ttl_sec=5.0, allow_api_fallback=True):
    """
    현재가 조회 - API 전용 (실시간).
    메모리 TTL 캐시만 사용 (동일 요청 중복 호출 방지).
    """
    t = str(ticker or "").strip().upper()
    if not t:
        return 0.0

    now_ts = float(datetime.now().timestamp())
    with _MEM_CACHE_LOCK:
        hit = _PRICE_MEM_CACHE.get(t)
        if isinstance(hit, dict) and (now_ts - float(hit.get("ts", 0.0))) <= float(ttl_sec):
            return float(hit.get("val", 0.0) or 0.0)

    p = 0.0
    try:
        p = float(pyupbit.get_current_price(t) or 0.0)
    except Exception:
        p = 0.0

    with _MEM_CACHE_LOCK:
        _PRICE_MEM_CACHE[t] = {"ts": now_ts, "val": float(p if p > 0 else 0.0)}
    return float(p if p > 0 else 0.0)


def get_current_prices_local_first(tickers, ttl_sec=5.0, allow_api_fallback=True):
    """
    다중 현재가 조회 (로컬 우선 + API 폴백).
    Returns: {ticker: price_float}
    """
    unique = []
    seen = set()
    for t in tickers or []:
        key = str(t or "").strip().upper()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)

    if not unique:
        return {}

    now_ts = float(datetime.now().timestamp())
    out = {}
    pending = []

    with _MEM_CACHE_LOCK:
        for t in unique:
            hit = _PRICE_MEM_CACHE.get(t)
            if isinstance(hit, dict) and (now_ts - float(hit.get("ts", 0.0))) <= float(ttl_sec):
                out[t] = float(hit.get("val", 0.0) or 0.0)
            else:
                pending.append(t)

    # API로 현재가 조회 (실시간)
    if pending:
        try:
            api_res = pyupbit.get_current_price(pending)
            if isinstance(api_res, dict):
                for t, v in api_res.items():
                    out[str(t).upper()] = float(v or 0.0)
            elif isinstance(api_res, (int, float)) and len(pending) == 1:
                out[pending[0]] = float(api_res or 0.0)
        except Exception:
            pass

        for t in pending:
            if out.get(t, 0.0) > 0:
                continue
            try:
                out[t] = float(pyupbit.get_current_price(t) or 0.0)
            except Exception:
                out[t] = 0.0

    # 3) TTL 업데이트
    with _MEM_CACHE_LOCK:
        for t in unique:
            val = float(out.get(t, 0.0) or 0.0)
            _PRICE_MEM_CACHE[t] = {"ts": now_ts, "val": val}

    return {t: float(out.get(t, 0.0) or 0.0) for t in unique}


def get_orderbook_cached(ticker, ttl_sec=3.0, allow_api_fallback=True):
    """
    호가 캐시 조회.
    - 메모리 TTL 캐시 우선
    - allow_api_fallback=True면 API 호출
    """
    t = str(ticker or "").strip().upper()
    if not t:
        return None

    now_ts = float(datetime.now().timestamp())
    with _MEM_CACHE_LOCK:
        hit = _ORDERBOOK_MEM_CACHE.get(t)
        if isinstance(hit, dict) and (now_ts - float(hit.get("ts", 0.0))) <= float(ttl_sec):
            return hit.get("val")

    if not allow_api_fallback:
        return None

    ob = None
    try:
        ob = pyupbit.get_orderbook(t)
        if isinstance(ob, list):
            ob = ob[0] if ob else None
    except Exception:
        ob = None

    with _MEM_CACHE_LOCK:
        _ORDERBOOK_MEM_CACHE[t] = {"ts": now_ts, "val": ob}
    return ob


def clear_cache(ticker=None, interval=None):
    """캐시 삭제"""
    _ensure_cache_dir()
    if ticker and interval:
        path = _cache_path(ticker, interval)
        if os.path.exists(path):
            os.remove(path)
    else:
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".parquet"):
                os.remove(os.path.join(CACHE_DIR, f))
    with _MEM_CACHE_LOCK:
        _PRICE_MEM_CACHE.clear()
        _ORDERBOOK_MEM_CACHE.clear()


def batch_download(tickers, intervals=None, count=10000, progress_callback=None):
    """
    여러 종목의 데이터를 일괄 다운로드.
    progress_callback(current, total, ticker, interval, rows): 진행률 콜백
    """
    if intervals is None:
        intervals = ["day"]

    total = len(tickers) * len(intervals)
    current = 0
    results = []

    import time as _time
    for ticker in tickers:
        for interval in intervals:
            current += 1
            # fetch_and_cache가 캐시 있으면 증분, 없으면 전체 다운로드
            df = fetch_and_cache(ticker, interval=interval, count=count)
            rows = len(df) if df is not None else 0

            results.append({"ticker": ticker, "interval": interval, "rows": rows})

            if progress_callback:
                progress_callback(current, total, ticker, interval, rows)

            _time.sleep(0.2)  # API rate limit

    return results


# ═══════════════════════════════════════════════════════
# Gold 캐시 (KRX 금현물 일봉)
# ═══════════════════════════════════════════════════════

def _gold_cache_path(code, interval="day"):
    safe = code.replace("/", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"GOLD_{safe}_{interval}.parquet")


def load_cached_gold(code="M04020000", interval="day"):
    """Gold 캐시 로드. 없으면 None"""
    path = _gold_cache_path(code, interval)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def save_cache_gold(code, interval, df):
    """Gold 데이터를 parquet으로 저장"""
    _ensure_cache_dir()
    path = _gold_cache_path(code, interval)
    df.to_parquet(path)


def fetch_and_cache_gold(trader, code="M04020000", count=2000, progress_callback=None):
    """
    Gold 일봉 API 다운로드 + 캐시 저장.
    - 캐시 있으면 증분 갭필 (최근 날짜 이후만)
    - 캐시 없으면 전체 다운로드
    progress_callback(fetched, total, status_msg)
    """
    cached = load_cached_gold(code)

    if cached is not None and len(cached) > 0:
        # 증분: 캐시 마지막 날짜 이후 갭필
        gap = _estimate_gap_count(cached.index[-1], "day")
        if gap <= 2:
            if progress_callback:
                progress_callback(0, 0, "캐시 최신 상태")
            return cached

        if progress_callback:
            progress_callback(0, gap, "증분 다운로드 중...")

        df_new = trader.get_daily_chart(code=code, count=gap + 10)
        if df_new is not None and len(df_new) > 0:
            merged = pd.concat([cached, df_new])
            merged = merged[~merged.index.duplicated(keep='last')]
            merged = merged.sort_index()
            if len(merged) > len(cached):
                save_cache_gold(code, "day", merged)
            if progress_callback:
                progress_callback(len(merged), len(merged), f"완료 ({len(merged)}개)")
            return merged

        if progress_callback:
            progress_callback(len(cached), len(cached), "갭필 실패, 기존 캐시 사용")
        return cached
    else:
        # 전체 다운로드
        if progress_callback:
            progress_callback(0, count, "전체 다운로드 중...")

        df_new = trader.get_daily_chart(code=code, count=count)
        if df_new is not None and len(df_new) > 0:
            save_cache_gold(code, "day", df_new)
            if progress_callback:
                progress_callback(len(df_new), len(df_new), f"완료 ({len(df_new)}개)")
            return df_new

        if progress_callback:
            progress_callback(0, 0, "다운로드 실패")
        return None


def gold_cache_info(code="M04020000"):
    """Gold 캐시 정보 반환"""
    path = _gold_cache_path(code, "day")
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            return {
                "exists": True,
                "rows": len(df),
                "start": df.index[0],
                "end": df.index[-1],
            }
        except Exception:
            return {"exists": False}
    return {"exists": False}


def get_gold_daily_local_first(trader=None, code="M04020000", count=2000, allow_api_fallback=True):
    """
    Gold 일봉 로컬 우선 조회.
    - cache/GOLD_*.parquet 우선
    - 부족 시 API 보강(fetch_and_cache_gold)
    """
    cnt = int(max(1, count))
    cached = _normalize_ohlcv_df(load_cached_gold(code=code, interval="day"))
    if cached is not None and not cached.empty:
        if len(cached) >= cnt:
            return cached.tail(cnt)
        if not allow_api_fallback:
            return cached

    if allow_api_fallback and trader is not None:
        try:
            api_df = fetch_and_cache_gold(trader, code=code, count=cnt)
            api_df = _normalize_ohlcv_df(api_df)
            if api_df is not None and not api_df.empty:
                return api_df.tail(cnt)
        except Exception:
            pass

    return cached.tail(cnt) if cached is not None and not cached.empty else None


def get_gold_current_price_local_first(trader=None, code="M04020000", allow_api_fallback=True, ttl_sec=8.0):
    """
    Gold 현재가 조회 - API 전용 (실시간).
    메모리 캐시(ttl_sec)만 사용하여 동일 요청 중복 호출 방지.
    """
    key = f"GOLD::{code}"
    now_ts = float(datetime.now().timestamp())
    with _MEM_CACHE_LOCK:
        hit = _PRICE_MEM_CACHE.get(key)
        if isinstance(hit, dict) and (now_ts - float(hit.get("ts", 0.0))) <= float(ttl_sec):
            return float(hit.get("val", 0.0) or 0.0)

    p = 0.0
    if trader is not None and allow_api_fallback:
        try:
            p = float(trader.get_current_price(code) or 0.0)
        except Exception:
            p = 0.0

    # API 실패 시 로컬 일봉(캐시/CSV) 종가를 현재가 대체값으로 사용
    if p <= 0:
        try:
            daily_df = get_gold_daily_local_first(
                trader=trader if allow_api_fallback else None,
                code=code,
                count=3,
                allow_api_fallback=allow_api_fallback,
            )
            if daily_df is not None and not daily_df.empty:
                p = float(daily_df["close"].iloc[-1])
        except Exception:
            p = 0.0

    # 루트 번들 CSV 최종 fallback
    if p <= 0:
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "krx_gold_daily.csv")
            if os.path.exists(csv_path):
                _df = pd.read_csv(csv_path)
                if not _df.empty:
                    cols = {str(c).lower(): c for c in _df.columns}
                    close_col = cols.get("close")
                    if close_col:
                        p = float(_df[close_col].iloc[-1])
        except Exception:
            p = 0.0

    with _MEM_CACHE_LOCK:
        _PRICE_MEM_CACHE[key] = {"ts": now_ts, "val": float(p if p > 0 else 0.0)}
    return float(p if p > 0 else 0.0)


# ═══════════════════════════════════════════════════════
# 번들 CSV 데이터 (오프라인 fallback용)
# ═══════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


WDR_TICKER_START_RATIO = {
    # ISA 위대리 매매 ETF별 시작일 초기 비중(고정값)
    # source: 위대리 V1.0 CSV 기준 (US record 매칭)
    # stock_ratio + cash_ratio = 1.0
    "409820": {"ref_date": "2022-01-07", "stock_ratio": 0.1496, "cash_ratio": 0.8504},
    "418660": {"ref_date": "2022-02-25", "stock_ratio": 0.1531, "cash_ratio": 0.8469},
    "423920": {"ref_date": "2022-04-22", "stock_ratio": 0.1985, "cash_ratio": 0.8015},
    "426030": {"ref_date": "2022-05-13", "stock_ratio": 0.2204, "cash_ratio": 0.7796},
    "465610": {"ref_date": "2023-09-15", "stock_ratio": 0.7253, "cash_ratio": 0.2747},
    "461910": {"ref_date": "2023-11-10", "stock_ratio": 0.7914, "cash_ratio": 0.2086},
}

WDR_TICKER_LISTING_DATE = {
    # KR 매매 ETF 상장일
    "409820": "2022-01-03",
    "418660": "2022-02-22",
    "423920": "2022-04-19",
    "426030": "2022-05-11",
    "465610": "2023-09-12",
    "461910": "2023-11-07",
}


def get_wdr_v10_stock_ratio(trade_etf_code: str, target_date):
    """
    ISA 위대리 초기 비중 고정값 조회.
    CSV를 매번 읽지 않고 티커별 시작일/비중 사전값을 반환한다.
    """
    code = str(trade_etf_code or "").strip()
    if not code:
        return None

    item = WDR_TICKER_START_RATIO.get(code)
    if not item:
        return None

    return {
        "source": "ticker_preset",
        "trade_etf_code": code,
        "ref_date": str(item["ref_date"]),
        "stock_ratio": float(item["stock_ratio"]),
        "cash_ratio": float(item["cash_ratio"]),
    }


def get_wdr_trade_listing_date(trade_etf_code: str):
    """ISA 위대리 매매 ETF 상장일(YYYY-MM-DD) 반환."""
    code = str(trade_etf_code or "").strip()
    if not code:
        return None
    return WDR_TICKER_LISTING_DATE.get(code)


def load_bundled_csv(ticker):
    """data/ 디렉토리의 번들 CSV 파일 로드 (API 실패 시 fallback).
    예: ticker='133690' → data/133690_daily.csv
    """
    csv_path = os.path.join(DATA_DIR, f"{ticker}_daily.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def _save_bundled_csv(ticker, df):
    """data/ 디렉토리에 CSV 자동 저장 (캐시 갱신 시 호출)."""
    if df is None or df.empty:
        return
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, f"{ticker}_daily.csv")
        _df = df.copy()
        _df.index.name = "date"
        _df.columns = [c.lower() for c in _df.columns]
        _df.to_csv(csv_path)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════
# KIS 캐시 (한국투자증권 국내/해외 OHLCV)
# ═══════════════════════════════════════════════════════

def _kis_cache_path(ticker, is_overseas=False):
    prefix = "KIS_OS" if is_overseas else "KIS_DOM"
    safe = ticker.replace("/", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"{prefix}_{safe}.parquet")


def load_cached_kis(ticker, is_overseas=False):
    path = _kis_cache_path(ticker, is_overseas)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def save_cache_kis(ticker, is_overseas=False, df=None):
    if df is None or df.empty: return
    _ensure_cache_dir()
    path = _kis_cache_path(ticker, is_overseas)
    df.to_parquet(path)


def fetch_and_cache_kis_domestic(trader, ticker, count=1500, progress_callback=None):
    """KIS 국내주식 일봉 캐시 업데이트.
    - 증분 로드: 최근 데이터(Forward) 및 과거 데이터(Backward) 갭필 지원
    - API 실패 시 번들 CSV fallback
    """
    cached = load_cached_kis(ticker, is_overseas=False)

    if cached is not None and len(cached) > 0:
        # 1. Forward Gap 필 (최근 데이터)
        gap_forward = _estimate_gap_count(cached.index[-1], "day")
        if gap_forward > 1:
            if progress_callback: progress_callback(0, gap_forward, "최근 데이터 로드...")
            df_new = trader.get_daily_chart(ticker, count=gap_forward + 5) if trader else None
            if df_new is not None and not df_new.empty:
                cached = pd.concat([cached, df_new])
                cached = cached[~cached.index.duplicated(keep='last')].sort_index()

        # 2. Backward Gap 필 (과거 데이터)
        # 만약 요청한 count보다 캐시된 데이터가 적고, 상장일이 더 과거라면 과거 데이터 추가 로드
        if len(cached) < count:
            needed_backward = count - len(cached)
            # 가장 오래된 캐시 날짜 전날부터 backward로 가져옴
            oldest_date = cached.index[0].strftime("%Y%m%d")
            # get_daily_chart는 end_date 기준 count만큼 가져오므로 oldest_date를 end_date로 설정
            if progress_callback: progress_callback(0, needed_backward, "과거 데이터 로드...")
            df_old = trader.get_daily_chart(ticker, end_date=oldest_date, count=needed_backward + 20) if trader else None
            
            if df_old is not None and not df_old.empty:
                cached = pd.concat([df_old, cached])
                cached = cached[~cached.index.duplicated(keep='first')].sort_index()
        
        save_cache_kis(ticker, is_overseas=False, df=cached)
        _save_bundled_csv(ticker, cached)  # CSV도 갱신
        return cached
    else:
        # 전체 로드
        if progress_callback: progress_callback(0, count, "전체 로드...")
        df_new = trader.get_daily_chart(ticker, count=count) if trader else None
        if df_new is not None and not df_new.empty:
            save_cache_kis(ticker, is_overseas=False, df=df_new)
            _save_bundled_csv(ticker, df_new)  # CSV도 갱신
            return df_new

        # API 실패 → 번들 CSV fallback
        bundled = load_bundled_csv(ticker)
        if bundled is not None:
            save_cache_kis(ticker, is_overseas=False, df=bundled)
        return bundled


def get_kis_domestic_local_first(
    trader,
    ticker,
    count=1500,
    end_date=None,
    allow_api_fallback=True,
):
    """로컬(cache/data) 우선 조회 후, 부족할 때만 API로 보강."""
    t = str(ticker).strip()
    if not t:
        return None

    cnt = int(max(1, count))
    local_df = load_cached_kis(t, is_overseas=False)
    if local_df is None or local_df.empty:
        local_df = load_bundled_csv(t)

    if local_df is not None and not local_df.empty:
        try:
            local_df = local_df.copy().sort_index()
            local_df.index = pd.to_datetime(local_df.index)
            local_df = local_df[~local_df.index.duplicated(keep="last")]
        except Exception:
            pass

        if end_date:
            try:
                end_ts = pd.to_datetime(end_date)
                local_df = local_df[local_df.index <= end_ts]
            except Exception:
                pass

        if len(local_df) >= cnt:
            return local_df.tail(cnt)

    if allow_api_fallback and trader is not None:
        try:
            if end_date:
                api_df = trader.get_daily_chart(t, end_date=str(end_date), count=cnt)
            else:
                api_df = fetch_and_cache_kis_domestic(trader, t, count=cnt)
            if api_df is not None and not api_df.empty:
                try:
                    api_df = api_df.copy().sort_index()
                    api_df.index = pd.to_datetime(api_df.index)
                    api_df = api_df[~api_df.index.duplicated(keep="last")]
                except Exception:
                    pass
                if len(api_df) > cnt:
                    api_df = api_df.tail(cnt)
                if end_date:
                    try:
                        save_cache_kis(t, is_overseas=False, df=api_df)
                        _save_bundled_csv(t, api_df)
                    except Exception:
                        pass
                return api_df
        except Exception:
            pass

    if local_df is not None and not local_df.empty:
        if len(local_df) > cnt:
            local_df = local_df.tail(cnt)
        return local_df
    return None


def fetch_and_cache_kis_overseas(trader, symbol, exchange="NAS", count=1500, progress_callback=None):
    """KIS 해외주식 일봉 캐시 업데이트"""
    cached = load_cached_kis(symbol, is_overseas=True)
    
    if cached is not None and len(cached) > 0:
        gap = _estimate_gap_count(cached.index[-1], "day")
        if gap <= 1:
            if progress_callback: progress_callback(0, 0, "최신")
            return cached
        
        if progress_callback: progress_callback(0, gap, "증분 로드...")
        df_new = trader.get_overseas_daily_chart(symbol, exchange=exchange, count=gap + 10)
        if df_new is not None and len(df_new) > 0:
            merged = pd.concat([cached, df_new])
            merged = merged[~merged.index.duplicated(keep='last')].sort_index()
            save_cache_kis(symbol, is_overseas=True, df=merged)
            return merged
        return cached
    else:
        if progress_callback: progress_callback(0, count, "전체 로드...")
        df_new = trader.get_overseas_daily_chart(symbol, exchange=exchange, count=count)
        if df_new is not None and len(df_new) > 0:
            save_cache_kis(symbol, is_overseas=True, df=df_new)
            return df_new
        return None


# ═══════════════════════════════════════════════════════
# yfinance 캐시 (미국 주식 일봉 — QQQ, TQQQ 등)
# ═══════════════════════════════════════════════════════

def _yf_csv_path(ticker):
    """yfinance 데이터 CSV 경로: data/{TICKER}_daily.csv"""
    return os.path.join(DATA_DIR, f"{ticker}_daily.csv")


def load_cached_yf(ticker):
    """yfinance 로컬 CSV 로드. 없으면 None."""
    path = _yf_csv_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


def save_cache_yf(ticker, df):
    """yfinance 데이터를 CSV로 저장."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _yf_csv_path(ticker)
    _df = df.copy()
    _df.index.name = "date"
    _df.columns = [c.lower() for c in _df.columns]
    _df.to_csv(path)


def fetch_and_cache_yf(ticker, start="2010-01-01", force_refresh: bool = False):
    """
    yfinance 데이터 캐시 우선 로드.
    - CSV 있으면 로드 → 마지막 날짜 이후 증분 다운로드 → CSV 갱신
    - CSV 없으면 전체 다운로드 → CSV 저장
    - force_refresh=True면 캐시가 최신처럼 보여도 증분 조회를 시도
    Returns: DataFrame (date index, ohlcv columns) or None
    """
    import yfinance as yf

    cached = load_cached_yf(ticker)

    if cached is not None and len(cached) > 0:
        last_date = cached.index[-1]
        # tz 제거
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)

        gap_days = (datetime.now() - last_date).days
        if (not force_refresh) and gap_days <= 1:
            return cached  # 최신 상태

        # 증분 다운로드 (마지막 날짜 다음날부터)
        next_day = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            df_new = yf.download(ticker, start=next_day, auto_adjust=True, progress=False)
            if df_new is not None and not df_new.empty:
                # MultiIndex 처리 (yfinance 최신 버전)
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = df_new.columns.get_level_values(0)
                df_new = df_new[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
                df_new.index = pd.to_datetime(df_new.index)
                if df_new.index.tz is not None:
                    df_new.index = df_new.index.tz_localize(None)
                df_new.index.name = "date"

                merged = pd.concat([cached, df_new])
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                save_cache_yf(ticker, merged)
                return merged
        except Exception:
            pass  # 갭필 실패 → 기존 캐시 반환

        return cached
    else:
        # 전체 다운로드
        try:
            df_new = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if df_new is None or df_new.empty:
                return cached  # None 반환
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            df_new = df_new[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
            df_new.index = pd.to_datetime(df_new.index)
            if df_new.index.tz is not None:
                df_new.index = df_new.index.tz_localize(None)
            df_new.index.name = "date"
            save_cache_yf(ticker, df_new)
            return df_new
        except Exception:
            return cached  # 실패 시 기존 캐시(None 포함) 반환


def list_cache():
    """캐시 목록 반환"""
    _ensure_cache_dir()
    result = []
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".parquet"):
            path = os.path.join(CACHE_DIR, f)
            size = os.path.getsize(path)
            try:
                df = pd.read_parquet(path)
                result.append({
                    "file": f,
                    "size_kb": round(size / 1024, 1),
                    "rows": len(df),
                    "start": str(df.index[0])[:10] if not df.empty else "-",
                    "end": str(df.index[-1])[:10] if not df.empty else "-",
                })
            except Exception:
                result.append({"file": f, "size_kb": round(size / 1024, 1), "rows": 0})
    return result
