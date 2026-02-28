import argparse
import json
import os
from pathlib import Path

import data_cache
from kiwoom_gold import GOLD_CODE_1KG, KiwoomGoldTrader
from kis_trader import KISTrader


TOP_20_TICKERS = [
    "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
    "KRW-ADA", "KRW-SHIB", "KRW-TRX", "KRW-AVAX", "KRW-LINK",
    "KRW-BCH", "KRW-DOT", "KRW-NEAR", "KRW-POL", "KRW-ETC",
    "KRW-XLM", "KRW-STX", "KRW-HBAR", "KRW-EOS", "KRW-SAND",
]

DEFAULT_UPBIT_INTERVALS = ["day", "minute240", "minute60", "minute30"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_config() -> dict:
    return _load_json(Path("user_config.json"))


def _load_portfolio_file() -> dict:
    return _load_json(Path("portfolio.json"))


def _code_only(v) -> str:
    return str(v or "").strip().split()[0] if str(v or "").strip() else ""


GOLD_KRX_ETF_CODE = "411060"
GOLD_LEGACY_ETF_CODES = {"132030"}


def _normalize_gold_kr_etf(v, default: str = GOLD_KRX_ETF_CODE) -> str:
    c = _code_only(v)
    if c in GOLD_LEGACY_ETF_CODES:
        return str(default)
    return c or str(default)


def _env_or(cfg: dict, cfg_key: str, env_keys: tuple[str, ...], default: str = "") -> str:
    v = str(cfg.get(cfg_key, "")).strip()
    if v:
        return v
    for k in env_keys:
        ev = str(os.getenv(k, "")).strip()
        if ev:
            return ev
    return default


def _bool_cfg(cfg: dict, key: str, default: bool = True) -> bool:
    raw = str(cfg.get(key, "1" if default else "0")).strip().lower()
    return raw not in ("0", "false", "no", "off", "")


def _collect_coin_targets(cfg: dict, include_top20: bool = False) -> list[tuple[str, str]]:
    portfolio_file = _load_portfolio_file()
    rows = cfg.get("portfolio")
    if not isinstance(rows, list) or not rows:
        rows = portfolio_file.get("portfolio", [])

    out: list[tuple[str, str]] = []
    seen = set()
    for r in rows:
        try:
            market = str(r.get("market", "KRW")).upper().strip()
            coin = str(r.get("coin", "BTC")).upper().strip()
            interval = str(r.get("interval", "day")).strip()
            ticker = f"{market}-{coin}"
            key = (ticker, interval)
            if ticker and interval and key not in seen:
                seen.add(key)
                out.append(key)
        except Exception:
            continue

    if include_top20:
        for t in TOP_20_TICKERS:
            for iv in DEFAULT_UPBIT_INTERVALS:
                key = (t, iv)
                if key not in seen:
                    seen.add(key)
                    out.append(key)

    return out


def _collect_kis_tickers(cfg: dict) -> list[str]:
    tickers = [
        _code_only(cfg.get("kis_isa_etf_code", "418660")),
        _code_only(cfg.get("kis_isa_trend_etf_code", "133690")),
        _code_only(cfg.get("kr_etf_laa_spy", "360750")),
        _code_only(cfg.get("kr_etf_laa_iwd", "360750")),
        _normalize_gold_kr_etf(cfg.get("kr_etf_laa_gld", GOLD_KRX_ETF_CODE)),
        _code_only(cfg.get("kr_etf_laa_ief", "453540")),
        _code_only(cfg.get("kr_etf_laa_qqq", "133690")),
        _code_only(cfg.get("kr_etf_laa_shy", "114470")),
        _code_only(cfg.get("pen_dm_kr_spy", "360750")),
        _code_only(cfg.get("pen_dm_kr_efa", "453850")),
        _code_only(cfg.get("pen_dm_kr_agg", "453540")),
        _code_only(cfg.get("pen_dm_kr_bil", "114470")),
        _code_only(cfg.get("pen_sa_etf1", "")),
        _code_only(cfg.get("pen_sa_etf2", "")),
    ]
    out: list[str] = []
    for t in tickers:
        if t and t not in out:
            out.append(t)
    return out


def _collect_yf_symbols(cfg: dict) -> list[str]:
    syms = ["QQQ", "TQQQ"]
    extra = cfg.get("yf_symbols", [])
    if isinstance(extra, list):
        for s in extra:
            ss = str(s).strip().upper()
            if ss and ss not in syms:
                syms.append(ss)
    return syms


def _normalize_acct(acct: str, prdt: str):
    raw = str(acct or "").replace("-", "").strip()
    p = str(prdt or "01").strip() or "01"
    if len(raw) == 10 and p in ("", "01"):
        return raw[:8], raw[8:]
    return raw, p


def _build_kis_trader(cfg: dict) -> KISTrader | None:
    app_key = _env_or(cfg, "kis_pension_app_key", ("KIS_APP_KEY", "KIS_PENSION_APP_KEY", "KIS_ISA_APP_KEY"))
    app_secret = _env_or(cfg, "kis_pension_app_secret", ("KIS_APP_SECRET", "KIS_PENSION_APP_SECRET", "KIS_ISA_APP_SECRET"))
    account_no = _env_or(cfg, "kis_pension_account_no", ("KIS_ACCOUNT_NO", "KIS_PENSION_ACCOUNT_NO", "KIS_ISA_ACCOUNT_NO"))
    prdt_cd = _env_or(cfg, "kis_pension_prdt_cd", ("KIS_ACNT_PRDT_CD", "KIS_PENSION_ACNT_PRDT_CD", "KIS_ISA_ACNT_PRDT_CD"), "01")
    account_no, prdt_cd = _normalize_acct(account_no, prdt_cd)
    if not (app_key and app_secret and account_no):
        return None
    t = KISTrader(is_mock=False)
    t.app_key = app_key
    t.app_secret = app_secret
    t.account_no = account_no
    t.acnt_prdt_cd = prdt_cd
    if not t.auth():
        return None
    return t


def _build_gold_trader() -> KiwoomGoldTrader | None:
    if not (os.getenv("Kiwoom_App_Key", "").strip() and os.getenv("Kiwoom_Secret_Key", "").strip()):
        return None
    t = KiwoomGoldTrader(is_mock=False)
    if not t.auth():
        return None
    return t


def preload_coin(cfg: dict, count: int, include_top20: bool):
    targets = _collect_coin_targets(cfg, include_top20=include_top20)
    if not targets:
        print("[COIN] 대상 없음")
        return
    print(f"[COIN] {len(targets)}개 캐시 갱신 시작")
    for i, (ticker, interval) in enumerate(targets, 1):
        try:
            df = data_cache.fetch_and_cache(ticker, interval=interval, count=count)
            n = len(df) if df is not None else 0
            print(f"[COIN {i}/{len(targets)}] {ticker} {interval}: {n}행")
        except Exception as e:
            print(f"[COIN {i}/{len(targets)}] {ticker} {interval}: 실패({e})")


def preload_kis(cfg: dict, count: int):
    tickers = _collect_kis_tickers(cfg)
    if not tickers:
        print("[KIS] 대상 없음")
        return
    trader = _build_kis_trader(cfg)
    if trader is None:
        print("[KIS] 인증 정보 없음 또는 인증 실패 - 스킵")
        return
    print(f"[KIS] {len(tickers)}개 캐시 갱신 시작")
    for i, t in enumerate(tickers, 1):
        try:
            df = data_cache.fetch_and_cache_kis_domestic(trader, t, count=count)
            n = len(df) if df is not None else 0
            if df is not None and not df.empty:
                print(f"[KIS {i}/{len(tickers)}] {t}: {n}행 ({df.index.min().date()} ~ {df.index.max().date()})")
            else:
                print(f"[KIS {i}/{len(tickers)}] {t}: 실패(데이터 없음)")
        except Exception as e:
            print(f"[KIS {i}/{len(tickers)}] {t}: 실패({e})")


def preload_gold(count: int):
    trader = _build_gold_trader()
    if trader is None:
        print("[GOLD] 인증 정보 없음 또는 인증 실패 - 스킵")
        return
    try:
        df = data_cache.fetch_and_cache_gold(trader, code=GOLD_CODE_1KG, count=count)
        n = len(df) if df is not None else 0
        print(f"[GOLD] {GOLD_CODE_1KG}: {n}행")
    except Exception as e:
        print(f"[GOLD] 실패({e})")


def preload_yf(cfg: dict):
    syms = _collect_yf_symbols(cfg)
    if not syms:
        print("[YF] 대상 없음")
        return
    print(f"[YF] {len(syms)}개 캐시 갱신 시작")
    for i, s in enumerate(syms, 1):
        try:
            df = data_cache.fetch_and_cache_yf(s, start="1990-01-01")
            n = len(df) if df is not None else 0
            print(f"[YF {i}/{len(syms)}] {s}: {n}행")
        except Exception as e:
            print(f"[YF {i}/{len(syms)}] {s}: 실패({e})")


def main():
    parser = argparse.ArgumentParser(description="프로젝트 전체 시세 캐시 선적재")
    parser.add_argument("--count", type=int, default=5000, help="종목별 최대 로딩 캔들 수")
    parser.add_argument("--include-top20", action="store_true", help="코인 TOP20 티커도 함께 캐시")
    parser.add_argument("--skip-coin", action="store_true")
    parser.add_argument("--skip-kis", action="store_true")
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--skip-yf", action="store_true")
    args = parser.parse_args()

    cfg = _load_config()
    local_first = _bool_cfg(cfg, "isa_local_first", True) and _bool_cfg(cfg, "pen_local_first", True)
    print(f"[INFO] local-first 정책: {'ON' if local_first else 'OFF'}")
    print(f"[INFO] count={args.count}")

    if not args.skip_coin:
        preload_coin(cfg, count=int(args.count), include_top20=bool(args.include_top20))
    if not args.skip_kis:
        preload_kis(cfg, count=int(args.count))
    if not args.skip_gold:
        preload_gold(count=int(args.count))
    if not args.skip_yf:
        preload_yf(cfg)

    print("완료: cache/*.parquet + data/*_daily.csv 갱신")


if __name__ == "__main__":
    main()
