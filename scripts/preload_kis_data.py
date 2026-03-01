import argparse
import json
import os
from pathlib import Path

import data_cache
from kis_trader import KISTrader


def _load_config() -> dict:
    p = Path("user_config.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _code_only(v) -> str:
    return str(v or "").strip().split()[0] if str(v or "").strip() else ""


GOLD_KRX_ETF_CODE = "411060"
GOLD_LEGACY_ETF_CODES = {"132030"}


def _normalize_gold_kr_etf(v, default: str = GOLD_KRX_ETF_CODE) -> str:
    c = _code_only(v)
    if c in GOLD_LEGACY_ETF_CODES:
        return str(default)
    return c or str(default)


def _collect_tickers(cfg: dict) -> list[str]:
    tickers: list[str] = []

    # ISA
    tickers.extend(
        [
            _code_only(cfg.get("kis_isa_etf_code", "418660")),
            _code_only(cfg.get("kis_isa_trend_etf_code", "133690")),
        ]
    )

    # 연금저축 LAA
    tickers.extend(
        [
            _code_only(cfg.get("kr_etf_laa_spy", "360750")),
            _code_only(cfg.get("kr_etf_laa_iwd", "360750")),
            _normalize_gold_kr_etf(cfg.get("kr_etf_laa_gld", GOLD_KRX_ETF_CODE)),
            _code_only(cfg.get("kr_etf_laa_ief", "453540")),
            _code_only(cfg.get("kr_etf_laa_qqq", "133690")),
            _code_only(cfg.get("kr_etf_laa_shy", "114470")),
        ]
    )

    # 연금저축 듀얼모멘텀
    tickers.extend(
        [
            _code_only(cfg.get("pen_dm_kr_spy", "360750")),
            _code_only(cfg.get("pen_dm_kr_efa", "453850")),
            _code_only(cfg.get("pen_dm_kr_agg", "453540")),
            _code_only(cfg.get("pen_dm_kr_bil", "114470")),
        ]
    )

    # 연금저축 정적배분
    tickers.extend(
        [
            _code_only(cfg.get("pen_sa_etf1", "")),
            _code_only(cfg.get("pen_sa_etf2", "")),
        ]
    )

    out: list[str] = []
    for t in tickers:
        if t and t not in out:
            out.append(t)
    return out


def _env_or(cfg: dict, cfg_key: str, env_keys: tuple[str, ...], default: str = "") -> str:
    v = str(cfg.get(cfg_key, "")).strip()
    if v:
        return v
    for k in env_keys:
        ev = str(os.getenv(k, "")).strip()
        if ev:
            return ev
    return default


def main():
    parser = argparse.ArgumentParser(description="KIS 국내 ETF 로컬 캐시 선다운로드")
    parser.add_argument("--count", type=int, default=5000, help="종목별 최대 일봉 개수")
    args = parser.parse_args()

    cfg = _load_config()
    app_key = _env_or(cfg, "kis_pension_app_key", ("KIS_APP_KEY", "KIS_PENSION_APP_KEY", "KIS_ISA_APP_KEY"))
    app_secret = _env_or(cfg, "kis_pension_app_secret", ("KIS_APP_SECRET", "KIS_PENSION_APP_SECRET", "KIS_ISA_APP_SECRET"))
    account_no = _env_or(cfg, "kis_pension_account_no", ("KIS_ACCOUNT_NO", "KIS_PENSION_ACCOUNT_NO", "KIS_ISA_ACCOUNT_NO"))
    prdt_cd = _env_or(cfg, "kis_pension_prdt_cd", ("KIS_ACNT_PRDT_CD", "KIS_PENSION_ACNT_PRDT_CD", "KIS_ISA_ACNT_PRDT_CD"), "01")

    raw = account_no.replace("-", "")
    if len(raw) == 10 and prdt_cd in ("", "01"):
        account_no, prdt_cd = raw[:8], raw[8:]

    if not (app_key and app_secret and account_no):
        print("KIS 인증정보가 없습니다. KIS_APP_KEY/KIS_APP_SECRET/KIS_ACCOUNT_NO를 설정하세요.")
        return

    trader = KISTrader(is_mock=False)
    trader.app_key = app_key
    trader.app_secret = app_secret
    trader.account_no = account_no
    trader.acnt_prdt_cd = prdt_cd

    if not trader.auth():
        print("KIS 인증 실패")
        return

    tickers = _collect_tickers(cfg)
    if not tickers:
        print("사전 다운로드할 종목이 없습니다.")
        return

    print(f"사전 다운로드 시작: {len(tickers)}종목, count={int(args.count)}")
    for i, t in enumerate(tickers, 1):
        try:
            df = data_cache.fetch_and_cache_kis_domestic(trader, t, count=int(args.count))
            if df is None or df.empty:
                print(f"[{i}/{len(tickers)}] {t}: 실패(데이터 없음)")
                continue
            print(f"[{i}/{len(tickers)}] {t}: {len(df)}개 ({df.index.min().date()} ~ {df.index.max().date()})")
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: 실패({e})")

    print("완료: cache/KIS_DOM_*.parquet 및 data/*_daily.csv 갱신")


if __name__ == "__main__":
    main()
