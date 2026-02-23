"""
로컬 데이터 캐시: OHLCV 데이터를 parquet 파일로 저장/로드
- API 호출 최소화
- 최적화 시 즉시 데이터 사용 가능
"""
import os
import pandas as pd
import pyupbit
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# 시간봉별 하루 캔들 수 (갭 계산용)
CANDLES_PER_DAY = {
    "day": 1, "minute240": 6, "minute60": 24,
    "minute30": 48, "minute15": 96, "minute5": 288, "minute1": 1440
}


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


# ═══════════════════════════════════════════════════════
# 번들 CSV 데이터 (오프라인 fallback용)
# ═══════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


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
        return cached
    else:
        # 전체 로드
        if progress_callback: progress_callback(0, count, "전체 로드...")
        df_new = trader.get_daily_chart(ticker, count=count) if trader else None
        if df_new is not None and not df_new.empty:
            save_cache_kis(ticker, is_overseas=False, df=df_new)
            return df_new
        
        # API 실패 → 번들 CSV fallback
        bundled = load_bundled_csv(ticker)
        if bundled is not None:
            save_cache_kis(ticker, is_overseas=False, df=bundled)
        return bundled


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
