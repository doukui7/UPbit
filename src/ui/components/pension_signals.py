"""연금저축 전략 시그널 계산 모듈.

LAA, 듀얼모멘텀, VAA 전략의 시그널 계산 함수를 제공한다.
"""

import numpy as np
import pandas as pd

import src.engine.data_cache as data_cache
from src.engine.backtest_core import _normalize_numeric_series
from src.constants import ETF_LEGACY_MERGE
from src.utils.formatting import _fmt_etf_code_name, _code_only


def _merge_legacy_holdings(current_vals: dict, current_qtys: dict) -> dict:
    """레거시 ETF 보유분을 현행 ETF에 합산 (평가금액만, 수량은 별도 유지).

    Returns:
        legacy_qty_map: {legacy_code: qty} — 매도 우선순위용
    """
    legacy_qty_map = {}
    for legacy_code, primary_code in ETF_LEGACY_MERGE.items():
        leg_val = current_vals.pop(legacy_code, 0.0)
        leg_qty = current_qtys.pop(legacy_code, 0)
        if leg_val > 0 or leg_qty > 0:
            current_vals[primary_code] = current_vals.get(primary_code, 0.0) + leg_val
            # 수량은 합산하지 않음 — 현행 ETF 수량만 사용 (매매 대상)
            # 대신 legacy_qty_map에 기록해서 매도 시 우선 처리
            legacy_qty_map[legacy_code] = leg_qty
    return legacy_qty_map


# ---------------------------------------------------------------------------
# LAA 시그널
# ---------------------------------------------------------------------------

def compute_laa_signal(
    *,
    source_codes: dict,
    kr_etf_map: dict,
    get_daily_chart,
    get_current_price,
    bal: dict,
    pen_bt_start_raw: str,
    pen_bt_cap: float,
    pen_bt_start_ts,
    pen_live_auto_backtest: bool,
    pen_api_fallback: bool,
) -> dict:
    """LAA 전략 시그널 계산 + 리밸런싱 배분 테이블.

    Parameters
    ----------
    source_codes : {"SPY": "360750", "IWD": "...", ...}
    kr_etf_map : 전체 ETF 코드 매핑
    get_daily_chart : callable(code, count=420, ...) -> DataFrame
    get_current_price : callable(code) -> float
    bal : 잔고 dict (cash, holdings, total_eval 등)
    pen_bt_start_raw : 백테스트 시작일 문자열
    pen_bt_cap : 백테스트 초기 자본
    pen_bt_start_ts : pd.Timestamp (백테스트 시작)
    pen_live_auto_backtest : 자동 백테스트 활성 여부
    pen_api_fallback : API 폴백 허용 여부
    """
    from src.strategy.laa import LAAStrategy

    tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
    price_data = {}
    source_map = dict(source_codes)
    _today = pd.Timestamp.now().date()

    # SPY는 미국 원본 데이터로 시그널 판단 (200일선), 나머지는 국내 ETF
    _spy_df = data_cache.fetch_and_cache_yf("SPY", start="2022-01-01")
    if _spy_df is not None and not _spy_df.empty:
        _spy_df = _spy_df.copy().sort_index()
        if "close" not in _spy_df.columns and "Close" in _spy_df.columns:
            _spy_df["close"] = _spy_df["Close"]
        if len(_spy_df) >= 2:
            _last_dt = pd.to_datetime(_spy_df.index[-1]).date()
            if _last_dt >= _today:
                _spy_df = _spy_df.iloc[:-1]
        price_data["SPY"] = _spy_df
        source_map["SPY"] = "SPY (US)"
    else:
        return {"error": "SPY 미국 원본 데이터 조회 실패 (yfinance)"}

    for ticker in tickers:
        if ticker == "SPY":
            continue  # 이미 US 데이터 로드 완료
        _code = _code_only(source_map.get(ticker, ""))
        if not _code:
            return {"error": f"{ticker} 국내 ETF 코드가 비어 있습니다."}
        df_t = get_daily_chart(_code, count=420)
        if df_t is None or df_t.empty:
            return {"error": f"{ticker} ({_code}) 국내 데이터 조회에 실패했습니다."}
        df_t = df_t.copy().sort_index()
        if "close" not in df_t.columns and "Close" in df_t.columns:
            df_t["close"] = df_t["Close"]
        if "close" not in df_t.columns:
            return {"error": f"{ticker} ({_code}) 종가 컬럼이 없습니다."}
        if len(df_t) >= 2:
            _last_dt = pd.to_datetime(df_t.index[-1]).date()
            if _last_dt >= _today:
                df_t = df_t.iloc[:-1]
        if df_t is None or df_t.empty:
            return {"error": f"{ticker} ({_code}) 전일 종가 데이터가 없습니다."}
        price_data[ticker] = df_t

    strategy = LAAStrategy(settings={"kr_etf_map": kr_etf_map})
    signal = strategy.analyze(price_data)
    if not signal:
        return {"error": "LAA 분석에 실패했습니다."}

    # 리스크 판단 차트: 미국 SPY 원본 + 200일선 + 이격도
    _risk_chart_code = "SPY (US)"
    _risk_df = _spy_df.copy() if _spy_df is not None else None
    if _risk_df is not None and not _risk_df.empty:
        if "close" in _risk_df.columns:
            _risk_df["ma200"] = _risk_df["close"].rolling(200).mean()
            _risk_df["divergence"] = np.where(
                _risk_df["ma200"] > 0,
                (_risk_df["close"] / _risk_df["ma200"] - 1.0) * 100.0,
                np.nan,
            )
        else:
            _risk_df = None

    bal_local = bal or {"cash": 0.0, "holdings": []}
    cash_local = float(bal_local.get("cash", 0.0))
    holdings_local = bal_local.get("holdings", []) or []

    current_vals = {}
    current_qtys = {}
    current_prices = {}
    for h in holdings_local:
        code = str(h.get("code", "")).strip()
        if not code:
            continue
        current_vals[code] = current_vals.get(code, 0.0) + float(h.get("eval_amt", 0.0) or 0.0)
        _qty = float(h.get("qty", 0.0) or 0.0)
        current_qtys[code] = current_qtys.get(code, 0) + int(np.floor(max(_qty, 0.0)))
        _cur_p = float(h.get("cur_price", 0.0) or 0.0)
        if _cur_p > 0:
            current_prices[code] = _cur_p

    # 레거시 Gold ETF(132030) 평가금액을 현행(411060)에 합산
    _laa_legacy_qty = _merge_legacy_holdings(current_vals, current_qtys)

    price_cache = {}

    def _resolve_etf_price(_code: str, _cur_v: float, _cur_q: int) -> float:
        _key = str(_code).strip()
        if not _key:
            return 0.0
        if _key in price_cache:
            return float(price_cache[_key])
        _p = 0.0
        try:
            _p = float(get_current_price(_key) or 0.0)
        except Exception:
            _p = 0.0
        if _p <= 0:
            _p = float(current_prices.get(_key, 0.0) or 0.0)
        if _p <= 0 and _cur_q > 0:
            _p = float(_cur_v) / float(_cur_q)
        if _p <= 0:
            try:
                _ch = get_daily_chart(_key, count=5)
                if _ch is not None and not _ch.empty:
                    _p = float(_ch["close"].iloc[-1])
            except Exception:
                _p = 0.0
        price_cache[_key] = float(_p if _p > 0 else 0.0)
        return float(price_cache[_key])

    total_eval = float(bal_local.get("total_eval", 0.0)) or (cash_local + sum(current_vals.values()))
    total_eval = max(total_eval, 1.0)

    rows = []
    max_gap = 0.0
    for code, target_w in signal["target_weights_kr"].items():
        cur_v = float(current_vals.get(str(code), 0.0))
        cur_w = cur_v / total_eval
        gap = float(target_w) - float(cur_w)
        max_gap = max(max_gap, abs(gap))
        cur_qty = int(current_qtys.get(str(code), 0))
        px = float(_resolve_etf_price(str(code), cur_v, cur_qty))
        target_v = total_eval * float(target_w)
        target_qty = int(np.floor(target_v / px)) if px > 0 else 0
        trade_qty = int(target_qty - cur_qty)
        trade_side = "매수" if trade_qty > 0 else ("매도" if trade_qty < 0 else "유지")
        trade_notional = abs(trade_qty) * px if px > 0 else 0.0
        rows.append({
            "ETF": _fmt_etf_code_name(code),
            "ETF 코드": str(code),
            "목표 비중(%)": round(target_w * 100.0, 2),
            "현재 비중(%)": round(cur_w * 100.0, 2),
            "비중 차이(%p)": round(gap * 100.0, 2),
            "현재가(KRW)": f"{px:,.0f}" if px > 0 else "-",
            "현재수량(주)": int(cur_qty),
            "목표수량(주)": int(target_qty),
            "주문": trade_side,
            "매매수량(주)": int(abs(trade_qty)),
            "예상주문금액(KRW)": f"{trade_notional:,.0f}" if px > 0 and abs(trade_qty) > 0 else "0",
            "현재 평가(KRW)": f"{cur_v:,.0f}",
            "목표 평가(KRW)": f"{target_v:,.0f}",
        })

    action = "HOLD" if max_gap <= 0.03 else "REBALANCE"

    bt_result = None
    bt_benchmark_series = None
    bt_benchmark_label = "SPY Buy & Hold"
    if pen_live_auto_backtest:
        bt_price_data = {}
        for ticker in tickers:
            _df_bt = data_cache.fetch_and_cache_yf(ticker, start="2000-01-01")
            if _df_bt is None or _df_bt.empty:
                bt_price_data = {}
                break
            _df_bt = _df_bt.copy().sort_index()
            if len(_df_bt) >= 2:
                _last_dt = pd.to_datetime(_df_bt.index[-1]).date()
                if _last_dt >= _today:
                    _df_bt = _df_bt.iloc[:-1]
            _df_bt = _df_bt[_df_bt.index >= pen_bt_start_ts]
            if _df_bt.empty:
                bt_price_data = {}
                break
            bt_price_data[ticker] = _df_bt

        if bt_price_data:
            bt_result = strategy.run_backtest(
                bt_price_data,
                initial_balance=float(pen_bt_cap),
                fee=0.0002,
            )
            _bm_series = _normalize_numeric_series(
                bt_price_data.get("SPY"),
                preferred_cols=("close", "Close"),
            )
            if not _bm_series.empty:
                bt_benchmark_series = _bm_series

    return {
        "signal": signal,
        "action": action,
        "source_map": source_map,
        "alloc_df": pd.DataFrame(rows),
        "legacy_sell_priority": _laa_legacy_qty,
        "price_data": price_data,
        "risk_chart_df": _risk_df,
        "risk_chart_code": _risk_chart_code,
        "balance": bal_local,
        "bt_start_date": str(pen_bt_start_raw),
        "bt_initial_cap": float(pen_bt_cap),
        "bt_result": bt_result,
        "bt_benchmark_series": bt_benchmark_series,
        "bt_benchmark_label": bt_benchmark_label,
    }


# ---------------------------------------------------------------------------
# 듀얼모멘텀 시그널
# ---------------------------------------------------------------------------

def compute_dm_signal(
    *,
    dm_settings: dict,
    get_daily_chart,
    get_current_price,
    bal: dict,
    pen_port_edited,
    pen_bt_start_raw: str,
    pen_bt_cap: float,
    pen_bt_start_ts,
    pen_live_auto_backtest: bool,
    data_source: str = "US",
) -> dict:
    """듀얼모멘텀 전략 시그널 계산 + 예상 리밸런싱 테이블.

    data_source: "US" = 미국 원본(yfinance), "KR" = 국내 ETF
    """
    from src.strategy.dual_momentum import DualMomentumStrategy

    _dm_tickers = []
    for _tk in (dm_settings.get("offensive", []) + dm_settings.get("defensive", []) + dm_settings.get("canary", [])):
        _u = str(_tk).strip().upper()
        if _u and _u not in _dm_tickers:
            _dm_tickers.append(_u)

    if not _dm_tickers:
        return {"error": "듀얼모멘텀 티커 설정이 비어 있습니다."}

    _use_us = data_source.upper() == "US"
    _dm_price_data = {}
    _dm_source_map = {}
    _dm_kr_map = dm_settings.get("kr_etf_map", {}) or {}
    _today = pd.Timestamp.now().date()

    for _ticker in _dm_tickers:
        if _use_us:
            # 미국 원본 데이터 (yfinance)
            _df_t = data_cache.fetch_and_cache_yf(_ticker, start="2022-01-01")
            if _df_t is None or _df_t.empty:
                return {"error": f"{_ticker} 미국 원본 데이터 조회 실패 (yfinance)"}
            _src_label = f"{_ticker} (US)"
        else:
            # 국내 ETF 데이터
            _kr_code = str(_dm_kr_map.get(_ticker, "")).strip()
            if not _kr_code:
                return {"error": f"{_ticker} 국내 ETF 매핑이 없습니다."}
            _df_t = get_daily_chart(_kr_code, count=420)
            if _df_t is None or _df_t.empty:
                return {"error": f"{_ticker} ({_kr_code}) 국내 데이터 조회 실패"}
            _src_label = _kr_code

        _df_t = _df_t.copy().sort_index()
        if "close" not in _df_t.columns and "Close" in _df_t.columns:
            _df_t["close"] = _df_t["Close"]
        if "close" not in _df_t.columns:
            return {"error": f"{_ticker} 종가 컬럼이 없습니다."}

        if len(_df_t) >= 2:
            _last_dt = pd.to_datetime(_df_t.index[-1]).date()
            if _last_dt >= _today:
                _df_t = _df_t.iloc[:-1]
        if _df_t is None or _df_t.empty:
            return {"error": f"{_ticker} 전일 종가 데이터가 없습니다."}

        _dm_price_data[_ticker] = _df_t
        _dm_source_map[_ticker] = _src_label

    _dm_strategy = DualMomentumStrategy(settings=dm_settings)
    _dm_sig = _dm_strategy.analyze(_dm_price_data)
    if not _dm_sig:
        return {"error": "듀얼모멘텀 분석에 실패했습니다."}

    _dm_td_local = int(dm_settings.get("trading_days_per_month", 22))
    _dm_lb_local = int(dm_settings.get("lookback", 12))
    _dm_w_local = dm_settings.get("momentum_weights", {}) or {}
    _w1 = float(_dm_w_local.get("m1", 12.0))
    _w3 = float(_dm_w_local.get("m3", 4.0))
    _w6 = float(_dm_w_local.get("m6", 2.0))
    _w12 = float(_dm_w_local.get("m12", 1.0))

    _mom_rows = []
    _ref_dates = []
    _score_rows = []
    _lookback_col = f"룩백{_dm_lb_local}개월(%)"
    for _tk in _dm_tickers:
        _dfm = _dm_price_data[_tk]
        _prices = _dfm["close"].astype(float).values
        _m1 = DualMomentumStrategy.calc_monthly_return(_prices, 1, _dm_td_local)
        _m3 = DualMomentumStrategy.calc_monthly_return(_prices, 3, _dm_td_local)
        _m6 = DualMomentumStrategy.calc_monthly_return(_prices, 6, _dm_td_local)
        _m12 = DualMomentumStrategy.calc_monthly_return(_prices, 12, _dm_td_local)
        _lb_ret = DualMomentumStrategy.calc_monthly_return(_prices, _dm_lb_local, _dm_td_local)
        _score = (_m1 * _w1 + _m3 * _w3 + _m6 * _w6 + _m12 * _w12) / 4.0
        _role = "공격" if _tk in dm_settings.get("offensive", []) else ("방어" if _tk in dm_settings.get("defensive", []) else "카나리아")
        _last_close = float(_dfm["close"].iloc[-1])
        _last_date = pd.to_datetime(_dfm.index[-1]).date()
        _ref_dates.append(_last_date)
        _score_rows.append({"티커": _tk, "모멘텀 점수": float(_score)})
        _price_label = "기준 종가(USD)" if _use_us else "기준 종가(KRW)"
        _price_fmt = f"{_last_close:,.2f}" if _use_us else f"{_last_close:,.0f}"
        _src_col_name = "데이터 출처" if _use_us else "국내 ETF"
        _src_col_val = _dm_source_map.get(_tk, "")
        if not _use_us:
            _src_col_val = _fmt_etf_code_name(_src_col_val)
        _mom_rows.append({
            "역할": _role,
            "티커": _tk,
            _src_col_name: _src_col_val,
            "기준 종가일": str(_last_date),
            _price_label: _price_fmt,
            "1개월(%)": round(_m1 * 100.0, 2),
            "3개월(%)": round(_m3 * 100.0, 2),
            "6개월(%)": round(_m6 * 100.0, 2),
            "12개월(%)": round(_m12 * 100.0, 2),
            _lookback_col: round(_lb_ret * 100.0, 2),
            "가중 모멘텀 점수": round(_score, 6),
        })

    _ref_date = min(_ref_dates) if _ref_dates else None
    _lag_days = int((_today - _ref_date).days) if _ref_date else None

    _bal_local = bal or {"cash": 0.0, "holdings": []}
    _holdings_local = _bal_local.get("holdings", []) or []
    _target_code = str(_dm_sig.get("target_kr_code", ""))
    _current_qty_map = {}
    for _h in _holdings_local:
        _code = str(_h.get("code", "")).strip()
        if not _code:
            continue
        _q = float(_h.get("qty", 0.0) or 0.0)
        _current_qty_map[_code] = _current_qty_map.get(_code, 0) + int(np.floor(max(_q, 0.0)))

    _all_dm_codes = []
    for _k in ("SPY", "EFA", "AGG"):
        _v = str((_dm_kr_map or {}).get(_k, "")).strip()
        if _v and _v not in _all_dm_codes:
            _all_dm_codes.append(_v)

    _target_holding = next((h for h in _holdings_local if str(h.get("code", "")) == _target_code), None)
    _other_holdings = [h for h in _holdings_local if str(h.get("code", "")) in _all_dm_codes and str(h.get("code", "")) != _target_code]

    if _target_holding and not _other_holdings:
        _action = "HOLD"
    elif _other_holdings:
        _action = "REBALANCE"
    else:
        _action = "BUY"

    _total_eval_local = float(_bal_local.get("total_eval", 0.0)) or (
        float(_bal_local.get("cash", 0.0))
        + sum(float(_h.get("eval_amt", 0.0) or 0.0) for _h in _holdings_local)
    )
    _total_eval_local = max(_total_eval_local, 0.0)
    _dm_weight_pct = float(
        pd.to_numeric(
            pen_port_edited.loc[pen_port_edited["strategy"] == "듀얼모멘텀", "weight"],
            errors="coerce",
        ).fillna(0).sum()
    )
    _dm_weight_pct = max(0.0, min(100.0, _dm_weight_pct))
    _sleeve_eval = _total_eval_local * (_dm_weight_pct / 100.0)
    _target_ticker = str(_dm_sig.get("target_ticker", "")).strip().upper()

    _expected_rows = []
    _alloc_tickers = []
    for _k in (dm_settings.get("offensive", []) + dm_settings.get("defensive", [])):
        _ku = str(_k).strip().upper()
        if _ku and _ku not in _alloc_tickers:
            _alloc_tickers.append(_ku)
    for _tk in _alloc_tickers:
        _code = str(_dm_kr_map.get(_tk, "")).strip()
        # 실제 매매는 국내 ETF → 국내 가격으로 수량 계산
        _px = float(get_current_price(_code) or 0) if _code else 0.0
        _target_w_sleeve = 100.0 if _tk == _target_ticker else 0.0
        _target_w_total = _dm_weight_pct if _tk == _target_ticker else 0.0
        _target_amt = _sleeve_eval if _tk == _target_ticker else 0.0
        _target_qty = int(np.floor(_target_amt / _px)) if _px > 0 else 0
        _cur_qty = int(_current_qty_map.get(_code, 0))
        _delta_qty = int(_target_qty - _cur_qty)
        _side = "매수" if _delta_qty > 0 else ("매도" if _delta_qty < 0 else "유지")
        _expected_rows.append({
            "티커": _tk,
            "국내 ETF": _fmt_etf_code_name(_code),
            "ETF 코드": str(_code),
            "기준 종가일": str(_ref_date) if _ref_date else "-",
            "기준 종가": f"{_px:,.0f}" if _px > 0 else "-",
            "듀얼모멘텀 비중(%)": round(_target_w_sleeve, 2),
            "전체 포트폴리오 환산 비중(%)": round(_target_w_total, 2),
            "목표 평가금액(KRW)": f"{_target_amt:,.0f}",
            "목표수량(주,버림)": int(_target_qty),
            "현재수량(주)": int(_cur_qty),
            "예상 주문": _side,
            "예상 주문수량(주)": int(abs(_delta_qty)),
        })

    _sleeve_alloc_amt = 0.0
    for _row in _expected_rows:
        try:
            _pxf = float(str(_row.get("기준 종가", "0")).replace(",", ""))
        except Exception:
            _pxf = 0.0
        _sleeve_alloc_amt += float(_row.get("목표수량(주,버림)", 0)) * _pxf
    _sleeve_cash_est = max(_sleeve_eval - _sleeve_alloc_amt, 0.0)

    _dm_bt_result = None
    _dm_bm_series = None
    _dm_bm_label = "SPY Buy & Hold"
    if pen_live_auto_backtest:
        _dm_bt_price_data = {}
        for _tk in _dm_tickers:
            if _use_us:
                _df_bt = data_cache.fetch_and_cache_yf(_tk, start="2018-01-01")
            else:
                _bt_kr_code = str(_dm_kr_map.get(_tk, "")).strip()
                _df_bt = get_daily_chart(_bt_kr_code, count=3000, use_disk_cache=True) if _bt_kr_code else None
            if _df_bt is None or _df_bt.empty:
                _dm_bt_price_data = {}
                break
            _df_bt = _df_bt.copy().sort_index()
            if "close" not in _df_bt.columns and "Close" in _df_bt.columns:
                _df_bt["close"] = _df_bt["Close"]
            if "close" not in _df_bt.columns:
                _dm_bt_price_data = {}
                break
            if len(_df_bt) >= 2:
                _last_dt = pd.to_datetime(_df_bt.index[-1]).date()
                if _last_dt >= _today:
                    _df_bt = _df_bt.iloc[:-1]
            _df_bt = _df_bt[_df_bt.index >= pen_bt_start_ts]
            if _df_bt.empty:
                _dm_bt_price_data = {}
                break
            _dm_bt_price_data[_tk] = _df_bt

        if _dm_bt_price_data:
            _dm_bt_result = _dm_strategy.run_backtest(
                _dm_bt_price_data,
                initial_balance=float(pen_bt_cap),
                fee=0.0002,
            )
            _dm_bm_ticker = ""
            for _t in (dm_settings.get("offensive", []) or []):
                if _t in _dm_bt_price_data:
                    _dm_bm_ticker = str(_t)
                    break
            if not _dm_bm_ticker:
                for _t in _dm_tickers:
                    if _t in _dm_bt_price_data:
                        _dm_bm_ticker = str(_t)
                        break
            _dm_bm_series_norm = _normalize_numeric_series(
                _dm_bt_price_data.get(_dm_bm_ticker),
                preferred_cols=("close", "Close"),
            )
            if not _dm_bm_series_norm.empty:
                _dm_bm_series = _dm_bm_series_norm
                if _use_us:
                    _dm_bm_label = f"{_dm_bm_ticker} (US) Buy & Hold"
                else:
                    _bm_kr_code = str(_dm_kr_map.get(_dm_bm_ticker, "")).strip()
                    _dm_bm_label = f"{_fmt_etf_code_name(_bm_kr_code)} Buy & Hold" if _bm_kr_code else f"{_dm_bm_ticker} Buy & Hold"

    return {
        "signal": _dm_sig,
        "action": _action,
        "data_source": data_source,
        "source_map": _dm_source_map,
        "score_df": pd.DataFrame(_score_rows),
        "momentum_detail_df": pd.DataFrame(_mom_rows),
        "expected_rebalance_df": pd.DataFrame(_expected_rows),
        "expected_meta": {
            "ref_date": str(_ref_date) if _ref_date else "",
            "today": str(_today),
            "lag_days": _lag_days,
            "dm_weight_pct": _dm_weight_pct,
            "sleeve_eval": _sleeve_eval,
            "sleeve_cash_est": _sleeve_cash_est,
        },
        "balance": _bal_local,
        "bt_start_date": str(pen_bt_start_raw),
        "bt_initial_cap": float(pen_bt_cap),
        "bt_result": _dm_bt_result,
        "bt_benchmark_series": _dm_bm_series,
        "bt_benchmark_label": _dm_bm_label,
    }


# ---------------------------------------------------------------------------
# VAA 시그널
# ---------------------------------------------------------------------------

def compute_vaa_signal(
    *,
    vaa_settings: dict,
    get_current_price,
    bal: dict,
    pen_port_edited,
) -> dict:
    """VAA 전략 시그널 계산."""
    from src.strategy.vaa import VAAStrategy

    try:
        _vaa_strat = VAAStrategy(settings=vaa_settings)
        _vaa_all_tickers = list(set(vaa_settings["offensive"] + vaa_settings["defensive"]))
        _vaa_price = {}
        for _t in _vaa_all_tickers:
            _df = data_cache.fetch_and_cache_yf(_t, start="2020-01-01")
            if _df is not None and not _df.empty:
                _vaa_price[_t] = _df
        _vaa_sig = _vaa_strat.analyze(_vaa_price)
        if not _vaa_sig:
            return {"error": "VAA 시그널 계산 실패"}

        _tw_kr = _vaa_sig.get("target_weights_kr", {})
        _hold_v = (bal or {}).get("holdings", []) or []
        _cash_v = float((bal or {}).get("cash", 0.0) or 0.0)
        _tot_v = float((bal or {}).get("total_eval", 0.0) or 0.0)
        if _tot_v <= 0:
            _tot_v = _cash_v + sum(float(h.get("eval_amt", 0) or 0) for h in _hold_v)
        _vaa_port_w = 0.0
        for _, _r in pen_port_edited.iterrows():
            if _r["strategy"] == "VAA":
                _vaa_port_w += float(_r.get("weight", 0)) / 100.0
        _vaa_sleeve = _tot_v * _vaa_port_w
        _hold_qty_v = {}
        for _h in _hold_v:
            _c = str(_h.get("code", "")).strip()
            _hold_qty_v[_c] = _hold_qty_v.get(_c, 0) + int(_h.get("qty", 0) or 0)
        _vaa_rebal_rows = []
        _max_gap = 0.0
        for code, w in _tw_kr.items():
            _tgt_eval = _vaa_sleeve * w
            _px = float(get_current_price(code) or 0)
            _tgt_qty = int(np.floor(_tgt_eval / _px)) if _px > 0 else 0
            _cur_qty = _hold_qty_v.get(code, 0)
            _cur_eval = _cur_qty * _px if _px > 0 else 0
            _cur_w = (_cur_eval / max(_tot_v, 1)) * 100
            _tgt_w = (_tgt_eval / max(_tot_v, 1)) * 100
            _max_gap = max(_max_gap, abs(_tgt_w - _cur_w))
            _vaa_rebal_rows.append({
                "ETF 코드": code, "ETF": _fmt_etf_code_name(code),
                "현재수량(주)": _cur_qty, "목표수량(주)": _tgt_qty,
                "목표 비중(%)": round(w * 100, 2),
                "현재 비중(%)": round(_cur_w, 2),
            })
        _vaa_action = "REBALANCE" if _max_gap > 3.0 else "HOLD"
        return {
            "signal": _vaa_sig, "action": _vaa_action,
            "alloc_df": pd.DataFrame(_vaa_rebal_rows) if _vaa_rebal_rows else pd.DataFrame(),
            "kr_etf_map": vaa_settings.get("kr_etf_map", {}),
        }
    except Exception as _e:
        return {"error": str(_e)}
