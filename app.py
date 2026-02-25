import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import requests
from dotenv import load_dotenv
import json
import data_cache

# Import modules
from backtest.engine import BacktestEngine
from trading.upbit_trader import UpbitTrader
from strategy.sma import SMAStrategy
from strategy.donchian import DonchianStrategy

# Load environment variables
load_dotenv(override=True)

# --- Configuration Persistence ---
CONFIG_FILE = "user_config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
        
config = load_config()

# 국내 ETF 코드 -> 한글 종목명 매핑 (ISA/연금저축 표기용)
ETF_NAME_KR = {
    "418660": "TIGER 미국나스닥100레버리지(합성)",
    "409820": "KODEX 미국나스닥100레버리지(합성 H)",
    "423920": "TIGER 미국필라델피아반도체레버리지(합성)",
    "465610": "ACE 미국빅테크TOP7 Plus레버리지(합성)",
    "461910": "PLUS 미국테크TOP10레버리지(합성)",
    "133690": "TIGER 미국나스닥100",
    "360750": "TIGER 미국S&P500",
    "132030": "KODEX Gold선물(H)",
    "453540": "TIGER 미국채10년선물",
    "114470": "KODEX 국고채3년",
    "453850": "TIGER 선진국MSCI World",
    "251350": "KODEX 선진국MSCI World",
    "308620": "KODEX 미국채10년선물",
    "471460": "ACE 미국30년국채액티브",
}


def _etf_name_kr(code: str) -> str:
    return ETF_NAME_KR.get(str(code).strip(), "종목명 미확인")


def _fmt_etf_code_name(code: str) -> str:
    c = str(code).strip()
    if not c:
        return "-"
    return f"{c} {_etf_name_kr(c)}"


def _code_only(v: str) -> str:
    return str(v or "").strip().split()[0] if str(v or "").strip() else ""


def _sidebar_etf_code_input(title: str, code_value: str, key: str, disabled: bool = False) -> str:
    code = _code_only(code_value)
    if not st.session_state.get("_etf_code_input_css_loaded", False):
        st.sidebar.markdown(
            """
            <style>
            section[data-testid="stSidebar"] input[aria-label="종목번호"]{
                font-size:0.90rem !important;
                font-weight:500 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_etf_code_input_css_loaded"] = True

    st.sidebar.markdown(
        f"<div style='font-size:1.02rem; font-weight:700; margin:0.05rem 0 0.28rem 0;'>{_etf_name_kr(code)}"
        f" <span style='font-size:0.92rem; font-weight:600; color:#6b7280;'>({title})</span></div>",
        unsafe_allow_html=True,
    )
    code_col, _spacer_col = st.sidebar.columns([0.9, 2.1])
    with code_col:
        typed_code = st.text_input(
            "종목번호",
            value=code,
            key=key,
            max_chars=6,
            disabled=disabled,
            label_visibility="collapsed",
        )
    return _code_only(typed_code)


def _get_runtime_value(keys, default=""):
    """Read value from env first, then Streamlit secrets."""
    if isinstance(keys, str):
        keys = (keys,)

    for key in keys:
        v = os.getenv(key, "")
        if str(v).strip():
            return v

    try:
        for key in keys:
            v = st.secrets.get(key, "")
            if str(v).strip():
                return v
    except Exception:
        pass

    return default


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def _interval_label(api_key: str) -> str:
    m = {
        "day": "1D",
        "minute240": "4H",
        "minute60": "1H",
        "minute30": "30m",
        "minute15": "15m",
        "minute5": "5m",
        "minute1": "1m",
    }
    return m.get(str(api_key), str(api_key))


def _normalize_kis_account(acct: str, prdt: str):
    acct = str(acct or "").replace("-", "").strip()
    prdt = str(prdt or "").strip() or "01"
    if len(acct) == 10 and prdt in ("", "01"):
        return acct[:8], acct[8:]
    return acct, prdt


def _planned_action_text(signal: str, is_holding: bool) -> str:
    s = str(signal or "").upper()
    if s == "BUY" and not is_holding:
        return "매수 예정"
    if s == "SELL" and is_holding:
        return "매도 예정"
    if s == "BUY" and is_holding:
        return "보유 유지"
    if s == "SELL" and not is_holding:
        return "현금 유지"
    return "유지"


def _send_telegram_message(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        for i in range(0, len(text), 3900):
            chunk = text[i:i + 3900]
            resp = requests.post(
                url,
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"},
                timeout=15,
            )
            ok = False
            desc = ""
            try:
                body = resp.json()
                ok = bool(resp.ok and body.get("ok", False))
                desc = str(body.get("description", ""))
            except Exception:
                ok = bool(resp.ok)
                desc = (resp.text or "")[:200]
            if not ok:
                return False, f"HTTP {resp.status_code} {desc}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _fetch_overseas_chart_any_exchange(trader, symbol: str, count: int = 320):
    for ex in ("NAS", "NYS", "AMS"):
        try:
            df = trader.get_overseas_daily_chart(symbol, exchange=ex, count=count)
        except Exception:
            df = None
        if df is not None and len(df) > 0:
            return df, ex
    return None, None


def _load_coin_portfolios_for_report(latest_cfg: dict):
    pjson = {}
    try:
        p_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")
        if os.path.exists(p_json_path):
            with open(p_json_path, "r", encoding="utf-8") as f:
                pjson = json.load(f) or {}
    except Exception:
        pjson = {}

    coin_portfolio = latest_cfg.get("portfolio")
    if not coin_portfolio:
        coin_portfolio = pjson.get("portfolio", [])
    aux_portfolio = latest_cfg.get("aux_portfolio")
    if aux_portfolio is None:
        aux_portfolio = pjson.get("aux_portfolio", [])
    return coin_portfolio or [], aux_portfolio or []


def _build_telegram_test_report() -> str:
    """
    테스트 전송용 리포트:
    - 전체 자산(계좌별 + 합계)
    - 각 전략별 현재 포지션/시그널/매매 예정
    """
    from trading.upbit_trader import UpbitTrader
    from kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG
    from kis_trader import KISTrader
    from strategy.widaeri import WDRStrategy
    from strategy.laa import LAAStrategy
    from strategy.dual_momentum import DualMomentumStrategy

    latest_cfg = load_config()
    coin_portfolio, _aux_portfolio = _load_coin_portfolios_for_report(latest_cfg)

    lines = [
        "<b>전략 상태 테스트 리포트</b>",
        f"기준시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    asset_rows = []

    # ─────────────────────────────────────────────────────────────
    # 1) 코인(업비트)
    # ─────────────────────────────────────────────────────────────
    upbit_ak = str(_get_runtime_value("UPBIT_ACCESS_KEY", "")).strip()
    upbit_sk = str(_get_runtime_value("UPBIT_SECRET_KEY", "")).strip()
    if upbit_ak and upbit_sk:
        try:
            upbit_trader = UpbitTrader(upbit_ak, upbit_sk)
            balances = upbit_trader.get_all_balances() or {}
            krw = _safe_float(balances.get("KRW", 0.0), 0.0)
            coin_eval = 0.0
            for cc, qty in balances.items():
                if str(cc).upper() == "KRW":
                    continue
                q = _safe_float(qty, 0.0)
                if q <= 0:
                    continue
                p = data_cache.get_current_price_local_first(f"KRW-{str(cc).upper()}", ttl_sec=5.0, allow_api_fallback=True)
                if p:
                    coin_eval += q * float(p)
            upbit_total = krw + coin_eval
            asset_rows.append(("업비트", upbit_total))

            lines.append(f"<b>[코인-업비트]</b> 총자산 {upbit_total:,.0f}원 (현금 {krw:,.0f}원)")
            if coin_portfolio:
                for idx, item in enumerate(coin_portfolio, 1):
                    coin = str(item.get("coin", "BTC")).upper()
                    market = str(item.get("market", "KRW")).upper()
                    ticker = f"{market}-{coin}"
                    strategy_name = str(item.get("strategy", "SMA"))
                    param = _safe_int(item.get("parameter", 20), 20)
                    sell_p = _safe_int(item.get("sell_parameter", 0), 0)
                    interval = str(item.get("interval", "day"))
                    weight = _safe_float(item.get("weight", 0.0), 0.0)

                    row_signal = "N/A"
                    row_detail = ""
                    try:
                        count = max(200, param * 3)
                        df = data_cache.get_ohlcv_local_first(
                            ticker,
                            interval=interval,
                            count=count,
                            allow_api_fallback=True,
                        )
                        if df is not None and len(df) >= max(30, param + 5):
                            if strategy_name == "Donchian":
                                sp = sell_p if sell_p > 0 else max(5, param // 2)
                                dstrat = DonchianStrategy()
                                df_feat = dstrat.create_features(df.copy(), buy_period=param, sell_period=sp)
                                row_signal = dstrat.get_signal(df_feat.iloc[-2], buy_period=param, sell_period=sp)
                                row_detail = f"Donchian({param}/{sp})"
                            else:
                                sstrat = SMAStrategy()
                                df_feat = sstrat.create_features(df.copy(), periods=[param])
                                row_signal = sstrat.get_signal(df_feat.iloc[-2], strategy_type="SMA_CROSS", ma_period=param)
                                row_detail = f"SMA({param})"
                        else:
                            row_detail = f"{strategy_name}({param}) 데이터부족"
                    except Exception as e:
                        row_detail = f"{strategy_name}({param}) 계산오류:{e}"

                    qty = _safe_float(balances.get(coin, 0.0), 0.0)
                    cur_price = _safe_float(
                        data_cache.get_current_price_local_first(ticker, ttl_sec=5.0, allow_api_fallback=True),
                        0.0,
                    )
                    holding_value = qty * cur_price
                    is_holding = qty > 0
                    position = f"보유 {qty:.8f}" if is_holding else "현금"
                    action = _planned_action_text(row_signal, is_holding)

                    target_value = upbit_total * (weight / 100.0)
                    delta_value = target_value - holding_value
                    if action == "매수 예정":
                        action += f" 약 {max(0.0, delta_value):,.0f}원"
                    elif action == "매도 예정":
                        action += f" 약 {max(0.0, -delta_value):,.0f}원"

                    lines.append(
                        f"- 전략{idx}: {ticker} {row_detail} {_interval_label(interval)} | "
                        f"포지션 {position} | 시그널 {row_signal} | 예정 {action}"
                    )
            else:
                lines.append("- 전략 설정이 없습니다.")
            lines.append("")
        except Exception as e:
            lines.append(f"<b>[코인-업비트]</b> 조회 실패: {e}")
            lines.append("")
    else:
        lines.append("<b>[코인-업비트]</b> API 키 미설정")
        lines.append("")

    # ─────────────────────────────────────────────────────────────
    # 2) 골드(키움)
    # ─────────────────────────────────────────────────────────────
    ki_ak = str(_get_runtime_value("Kiwoom_App_Key", "")).strip()
    ki_sk = str(_get_runtime_value("Kiwoom_Secret_Key", "")).strip()
    ki_acct = str(_get_runtime_value("KIWOOM_ACCOUNT", "")).strip()
    if ki_ak and ki_sk and ki_acct:
        try:
            gtr = KiwoomGoldTrader(is_mock=False)
            gtr.app_key = ki_ak
            gtr.app_secret = ki_sk
            gtr.account_no = ki_acct
            if gtr.auth():
                gbal = gtr.get_balance() or {}
                gprice = _safe_float(
                    data_cache.get_gold_current_price_local_first(
                        trader=gtr,
                        code=GOLD_CODE_1KG,
                        allow_api_fallback=True,
                        ttl_sec=8.0,
                    ),
                    0.0,
                )
                gcash = _safe_float(gbal.get("cash_krw", 0.0), 0.0)
                gqty = _safe_float(gbal.get("gold_qty", 0.0), 0.0)
                gval = gqty * gprice
                gtot = gcash + gval
                asset_rows.append(("골드(키움)", gtot))

                lines.append(f"<b>[골드-키움]</b> 총자산 {gtot:,.0f}원 (현금 {gcash:,.0f}원, 금 {gqty:.2f}g)")
                gcfg = latest_cfg.get("gold_strategy", [{"strategy": "Donchian", "buy": 90, "sell": 55, "weight": 100}])
                max_bp = max((_safe_int(x.get("buy", 90), 90) for x in gcfg), default=90)
                gdf = data_cache.get_gold_daily_local_first(
                    trader=gtr,
                    code=GOLD_CODE_1KG,
                    count=max(220, max_bp + 20),
                    allow_api_fallback=True,
                )
                if gdf is not None and len(gdf) >= max_bp + 5:
                    close_now = float(gdf["close"].iloc[-1])
                    total_w = sum(_safe_float(x.get("weight", 0), 0.0) for x in gcfg) or 1.0
                    buy_w = 0.0
                    for i, gs in enumerate(gcfg, 1):
                        sname = str(gs.get("strategy", "Donchian"))
                        bp = _safe_int(gs.get("buy", 90), 90)
                        sp = _safe_int(gs.get("sell", 0), 0) or max(5, bp // 2)
                        w = _safe_float(gs.get("weight", 0.0), 0.0)
                        if sname == "Donchian":
                            upper = gdf["high"].rolling(window=bp).max().shift(1)
                            lower = gdf["low"].rolling(window=sp).min().shift(1)
                            upv = _safe_float(upper.iloc[-1], 0.0)
                            lowv = _safe_float(lower.iloc[-1], 0.0)
                            sig = "BUY" if close_now > upv else "SELL" if close_now < lowv else "HOLD"
                        else:
                            sma = gdf["close"].rolling(window=bp).mean()
                            smav = _safe_float(sma.iloc[-1], 0.0)
                            sig = "BUY" if close_now > smav else "SELL"
                        if sig == "BUY":
                            buy_w += w
                        lines.append(f"- 전략{i}: {sname}({bp}/{sp}) | 시그널 {sig} | 비중 {w:.1f}%")

                    target_ratio = buy_w / total_w
                    cur_ratio = (gval / gtot) if gtot > 0 else 0.0
                    ratio_gap = target_ratio - cur_ratio
                    if abs(ratio_gap) <= 0.05:
                        g_action = "유지"
                    elif ratio_gap > 0:
                        g_action = f"매수 예정 약 {gtot * ratio_gap:,.0f}원"
                    else:
                        g_action = f"매도 예정 약 {gtot * abs(ratio_gap):,.0f}원"
                    g_pos = "금 보유" if gqty > 0 else "현금"
                    lines.append(f"- 현재 포지션 {g_pos} | 예정 {g_action}")
                else:
                    lines.append("- 전략 계산용 차트 데이터 부족")
                lines.append("")
            else:
                lines.append("<b>[골드-키움]</b> 인증 실패")
                lines.append("")
        except Exception as e:
            lines.append(f"<b>[골드-키움]</b> 조회 실패: {e}")
            lines.append("")
    else:
        lines.append("<b>[골드-키움]</b> API 키 미설정")
        lines.append("")

    # ─────────────────────────────────────────────────────────────
    # 3) ISA(WDR)
    # ─────────────────────────────────────────────────────────────
    isa_ak = str(_get_runtime_value(("KIS_ISA_APP_KEY", "KIS_APP_KEY"), "")).strip()
    isa_sk = str(_get_runtime_value(("KIS_ISA_APP_SECRET", "KIS_APP_SECRET"), "")).strip()
    isa_acct = str(latest_cfg.get("kis_isa_account_no", "") or _get_runtime_value(("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO"), "")).strip()
    isa_prdt = str(latest_cfg.get("kis_isa_prdt_cd", "") or _get_runtime_value(("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01")).strip()
    isa_acct, isa_prdt = _normalize_kis_account(isa_acct, isa_prdt)
    if isa_ak and isa_sk and isa_acct:
        try:
            itr = KISTrader(is_mock=False)
            itr.app_key = isa_ak
            itr.app_secret = isa_sk
            itr.account_no = isa_acct
            itr.acnt_prdt_cd = isa_prdt
            if itr.auth():
                ibal = itr.get_balance() or {}
                icash = _safe_float(ibal.get("cash", 0.0), 0.0)
                ihold = ibal.get("holdings", []) or []
                itotal = _safe_float(ibal.get("total_eval", 0.0), 0.0) or (icash + sum(_safe_float(h.get("eval_amt", 0.0), 0.0) for h in ihold))
                asset_rows.append(("KIS ISA", itotal))

                isa_etf = str(latest_cfg.get("kis_isa_etf_code", _get_runtime_value("KIS_ISA_ETF_CODE", "418660")))
                isa_sig_etf = str(latest_cfg.get("kis_isa_trend_etf_code", _get_runtime_value("KIS_ISA_TREND_ETF_CODE", "133690")))
                lines.append(f"<b>[KIS ISA]</b> 총자산 {itotal:,.0f}원 (현금 {icash:,.0f}원)")

                wdr = WDRStrategy(settings={
                    "overvalue_threshold": float(latest_cfg.get("kis_isa_wdr_ov", 5.0)),
                    "undervalue_threshold": float(latest_cfg.get("kis_isa_wdr_un", -6.0)),
                }, evaluation_mode=int(latest_cfg.get("kis_isa_wdr_mode", 3)))
                sig_df = data_cache.get_kis_domestic_local_first(
                    itr,
                    _code_only(isa_sig_etf),
                    count=1500,
                    allow_api_fallback=True,
                )
                sig = wdr.analyze(sig_df) if sig_df is not None and len(sig_df) >= 260 * 5 else None
                _isa_px_df = data_cache.get_kis_domestic_local_first(
                    itr,
                    _code_only(isa_etf),
                    count=3,
                    allow_api_fallback=True,
                )
                cur_price = 0.0
                if _isa_px_df is not None and not _isa_px_df.empty and "close" in _isa_px_df.columns:
                    cur_price = _safe_float(_isa_px_df["close"].iloc[-1], 0.0)
                if cur_price <= 0:
                    cur_price = _safe_float(itr.get_current_price(_code_only(isa_etf)) or 0.0, 0.0)
                cur_shares = 0
                for h in ihold:
                    if str(h.get("code", "")) == isa_etf:
                        cur_shares = int(_safe_float(h.get("qty", 0), 0))
                        break
                pos = f"{isa_etf} {cur_shares}주" if cur_shares > 0 else "현금"

                if sig and cur_price > 0:
                    etf_chart = data_cache.get_kis_domestic_local_first(
                        itr,
                        _code_only(isa_etf),
                        count=10,
                        allow_api_fallback=True,
                    )
                    weekly_pnl = 0.0
                    if etf_chart is not None and len(etf_chart) >= 5 and cur_shares > 0:
                        weekly_pnl = (cur_price - float(etf_chart["close"].iloc[-5])) * cur_shares
                    action = wdr.get_rebalance_action(
                        weekly_pnl=weekly_pnl,
                        divergence=float(sig["divergence"]),
                        current_shares=cur_shares,
                        current_price=cur_price,
                        cash=icash,
                    )
                    act = action.get("action")
                    qty = int(action.get("quantity", 0) or 0)
                    planned = "유지" if not act or qty <= 0 else f"{'매수' if act == 'BUY' else '매도'} 예정 {qty}주"
                    lines.append(
                        f"- 전략: WDR | 포지션 {pos} | 이격도 {float(sig.get('divergence', 0.0)):.2f}% | 예정 {planned}"
                    )
                else:
                    lines.append(f"- 전략: WDR | 포지션 {pos} | 예정 계산 불가(데이터 부족)")
                lines.append("")
            else:
                lines.append("<b>[KIS ISA]</b> 인증 실패")
                lines.append("")
        except Exception as e:
            lines.append(f"<b>[KIS ISA]</b> 조회 실패: {e}")
            lines.append("")
    else:
        lines.append("<b>[KIS ISA]</b> API 키 미설정")
        lines.append("")

    # ─────────────────────────────────────────────────────────────
    # 4) 연금저축(LAA/듀얼모멘텀/정적배분)
    # ─────────────────────────────────────────────────────────────
    pen_ak = str(_get_runtime_value(("KIS_PENSION_APP_KEY", "KIS_APP_KEY"), "")).strip()
    pen_sk = str(_get_runtime_value(("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET"), "")).strip()
    pen_acct = str(latest_cfg.get("kis_pension_account_no", "") or _get_runtime_value(("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO"), "")).strip()
    pen_prdt = str(latest_cfg.get("kis_pension_prdt_cd", "") or _get_runtime_value(("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01")).strip()
    pen_acct, pen_prdt = _normalize_kis_account(pen_acct, pen_prdt)
    if pen_ak and pen_sk and pen_acct:
        try:
            ptr = KISTrader(is_mock=False)
            ptr.app_key = pen_ak
            ptr.app_secret = pen_sk
            ptr.account_no = pen_acct
            ptr.acnt_prdt_cd = pen_prdt
            if ptr.auth():
                pbal = ptr.get_balance() or {}
                pcash = _safe_float(pbal.get("cash", 0.0), 0.0)
                phold = pbal.get("holdings", []) or []
                ptotal = _safe_float(pbal.get("total_eval", 0.0), 0.0) or (pcash + sum(_safe_float(h.get("eval_amt", 0.0), 0.0) for h in phold))
                asset_rows.append(("KIS 연금저축", ptotal))
                lines.append(f"<b>[KIS 연금저축]</b> 총자산 {ptotal:,.0f}원 (현금 {pcash:,.0f}원)")

                pen_port = latest_cfg.get("pension_portfolio", [{"strategy": "LAA", "weight": 100}]) or []
                active_strategies = [
                    (str(r.get("strategy", "")).strip(), _safe_float(r.get("weight", 0), 0.0))
                    for r in pen_port
                    if _safe_float(r.get("weight", 0), 0.0) > 0
                ]

                for s_name, s_weight in active_strategies:
                    try:
                        if s_name == "LAA":
                            kr_map = {
                                "IWD": str(latest_cfg.get("kr_etf_laa_iwd", _get_runtime_value("KR_ETF_LAA_IWD", _get_runtime_value("KR_ETF_SPY", "360750")))),
                                "GLD": str(latest_cfg.get("kr_etf_laa_gld", _get_runtime_value("KR_ETF_LAA_GLD", "132030"))),
                                "IEF": str(latest_cfg.get("kr_etf_laa_ief", _get_runtime_value("KR_ETF_LAA_IEF", _get_runtime_value("KR_ETF_AGG", "453540")))),
                                "QQQ": str(latest_cfg.get("kr_etf_laa_qqq", _get_runtime_value("KR_ETF_LAA_QQQ", "133690"))),
                                "SHY": str(latest_cfg.get("kr_etf_laa_shy", _get_runtime_value("KR_ETF_LAA_SHY", "114470"))),
                            }
                            spy_signal_code = str(
                                latest_cfg.get(
                                    "kr_etf_laa_spy",
                                    _get_runtime_value("KR_ETF_LAA_SPY", _get_runtime_value("KR_ETF_SPY", kr_map.get("IWD", "360750"))),
                                )
                            )
                            tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
                            source_map = {
                                "SPY": _code_only(spy_signal_code or kr_map.get("IWD", "360750")),
                                "IWD": _code_only(kr_map.get("IWD", "")),
                                "GLD": _code_only(kr_map.get("GLD", "")),
                                "IEF": _code_only(kr_map.get("IEF", "")),
                                "QQQ": _code_only(kr_map.get("QQQ", "")),
                                "SHY": _code_only(kr_map.get("SHY", "")),
                            }
                            price_data = {}
                            for t in tickers:
                                _code = _code_only(source_map.get(t, ""))
                                if not _code:
                                    price_data = {}
                                    break
                                df = data_cache.get_kis_domestic_local_first(
                                    ptr,
                                    _code,
                                    count=320,
                                    allow_api_fallback=True,
                                )
                                if df is None or df.empty:
                                    price_data = {}
                                    break
                                df = df.copy().sort_index()
                                if "close" not in df.columns and "Close" in df.columns:
                                    df["close"] = df["Close"]
                                if "close" not in df.columns:
                                    price_data = {}
                                    break
                                price_data[t] = df
                            if price_data:
                                laa = LAAStrategy(settings={"kr_etf_map": kr_map})
                                sig = laa.analyze(price_data)
                                if sig:
                                    tw = sig.get("target_weights_kr", {})
                                    tracked = set(str(c) for c in tw.keys())
                                    cur_vals = {c: 0.0 for c in tracked}
                                    cur_qtys = {c: 0 for c in tracked}
                                    for h in phold:
                                        code = str(h.get("code", ""))
                                        if code in tracked:
                                            cur_vals[code] += _safe_float(h.get("eval_amt", 0.0), 0.0)
                                            cur_qtys[code] += _safe_int(h.get("qty", 0), 0)
                                    total_safe = max(ptotal, 1.0)
                                    planned = []
                                    max_gap = 0.0
                                    for code, w in tw.items():
                                        c = str(code)
                                        target_v = total_safe * _safe_float(w, 0.0)
                                        cur_v = cur_vals.get(c, 0.0)
                                        gap = abs((cur_v / total_safe) - _safe_float(w, 0.0))
                                        max_gap = max(max_gap, gap)
                                        _p_df = data_cache.get_kis_domestic_local_first(
                                            ptr,
                                            c,
                                            count=2,
                                            allow_api_fallback=True,
                                        )
                                        price = 0.0
                                        if _p_df is not None and not _p_df.empty and "close" in _p_df.columns:
                                            price = _safe_float(_p_df["close"].iloc[-1], 0.0)
                                        if price <= 0:
                                            price = _safe_float(ptr.get_current_price(c) or 0.0, 0.0)
                                        if price <= 0:
                                            continue
                                        tgt_q = int(target_v / price)
                                        delta = tgt_q - cur_qtys.get(c, 0)
                                        if delta > 0:
                                            planned.append(f"매수 {c} {delta}주")
                                        elif delta < 0:
                                            planned.append(f"매도 {c} {abs(delta)}주")
                                    pos = "보유" if any(_safe_float(h.get("qty", 0), 0) > 0 for h in phold) else "현금"
                                    action = "유지" if max_gap <= 0.03 else ("; ".join(planned[:3]) if planned else "리밸런싱")
                                    lines.append(
                                        f"- 전략: LAA({s_weight:.0f}%) | 포지션 {pos} | 리스크 {'공격' if sig.get('risk_on') else '방어'} | 예정 {action}"
                                    )
                                else:
                                    lines.append(f"- 전략: LAA({s_weight:.0f}%) | 분석 실패")
                            else:
                                lines.append(f"- 전략: LAA({s_weight:.0f}%) | 데이터 부족")

                        elif s_name == "듀얼모멘텀":
                            dm_settings = {
                                "offensive": ["SPY", "EFA"],
                                "defensive": ["AGG"],
                                "canary": ["BIL"],
                                "lookback": int(latest_cfg.get("pen_dm_lookback", 12)),
                                "trading_days_per_month": int(latest_cfg.get("pen_dm_trading_days", 22)),
                                "momentum_weights": {
                                    "m1": float(latest_cfg.get("pen_dm_w1", 12.0)),
                                    "m3": float(latest_cfg.get("pen_dm_w3", 4.0)),
                                    "m6": float(latest_cfg.get("pen_dm_w6", 2.0)),
                                    "m12": float(latest_cfg.get("pen_dm_w12", 1.0)),
                                },
                                "kr_etf_map": {
                                    "SPY": str(latest_cfg.get("pen_dm_kr_spy", _get_runtime_value("KR_ETF_SPY", "360750"))),
                                    "EFA": str(latest_cfg.get("pen_dm_kr_efa", _get_runtime_value("KR_ETF_EFA", "453850"))),
                                    "AGG": str(latest_cfg.get("pen_dm_kr_agg", _get_runtime_value("KR_ETF_AGG", "453540"))),
                                    "BIL": str(latest_cfg.get("pen_dm_kr_bil", _get_runtime_value("KR_ETF_BIL", _get_runtime_value("KR_ETF_SHY", "114470")))),
                                },
                            }
                            price_data = {}
                            for tk, code in dm_settings["kr_etf_map"].items():
                                df = data_cache.get_kis_domestic_local_first(
                                    ptr,
                                    _code_only(str(code)),
                                    count=320,
                                    allow_api_fallback=True,
                                )
                                if df is None or len(df) < 260:
                                    price_data = {}
                                    break
                                price_data[tk] = df
                            if price_data:
                                dms = DualMomentumStrategy(settings=dm_settings)
                                sig = dms.analyze(price_data)
                                if sig:
                                    target_code = str(sig.get("target_kr_code", ""))
                                    dm_codes = set(str(v) for v in dm_settings["kr_etf_map"].values())
                                    hold_codes = [str(h.get("code", "")) for h in phold if str(h.get("code", "")) in dm_codes and _safe_int(h.get("qty", 0), 0) > 0]
                                    hold_txt = ", ".join(hold_codes) if hold_codes else "현금"
                                    if target_code and hold_codes == [target_code]:
                                        d_action = "유지"
                                    elif target_code:
                                        d_action = f"리밸런싱 → {target_code}"
                                    else:
                                        d_action = "유지"
                                    lines.append(
                                        f"- 전략: 듀얼모멘텀({s_weight:.0f}%) | 포지션 {hold_txt} | 대상 {sig.get('target_ticker')}→{target_code} | 예정 {d_action}"
                                    )
                                else:
                                    lines.append(f"- 전략: 듀얼모멘텀({s_weight:.0f}%) | 분석 실패")
                            else:
                                lines.append(f"- 전략: 듀얼모멘텀({s_weight:.0f}%) | 데이터 부족")

                        elif s_name == "정적배분":
                            etf1 = str(latest_cfg.get("pen_sa_etf1", "360750"))
                            etf2 = str(latest_cfg.get("pen_sa_etf2", "453540"))
                            w1 = _safe_float(latest_cfg.get("pen_sa_w1", 60), 60.0)
                            w2 = _safe_float(latest_cfg.get("pen_sa_w2", 40), 40.0)
                            cur1 = sum(_safe_float(h.get("eval_amt", 0.0), 0.0) for h in phold if str(h.get("code", "")) == etf1)
                            cur2 = sum(_safe_float(h.get("eval_amt", 0.0), 0.0) for h in phold if str(h.get("code", "")) == etf2)
                            total_safe = max(ptotal, 1.0)
                            gap1 = (w1 / 100.0) - (cur1 / total_safe)
                            gap2 = (w2 / 100.0) - (cur2 / total_safe)
                            if max(abs(gap1), abs(gap2)) <= 0.03:
                                sa_action = "유지"
                            else:
                                parts = []
                                if gap1 > 0:
                                    parts.append(f"매수 {etf1} 약 {total_safe * gap1:,.0f}원")
                                elif gap1 < 0:
                                    parts.append(f"매도 {etf1} 약 {total_safe * abs(gap1):,.0f}원")
                                if gap2 > 0:
                                    parts.append(f"매수 {etf2} 약 {total_safe * gap2:,.0f}원")
                                elif gap2 < 0:
                                    parts.append(f"매도 {etf2} 약 {total_safe * abs(gap2):,.0f}원")
                                sa_action = "; ".join(parts) if parts else "리밸런싱"
                            sa_pos = "보유" if (cur1 + cur2) > 0 else "현금"
                            lines.append(
                                f"- 전략: 정적배분({s_weight:.0f}%) | 포지션 {sa_pos} | 목표 {etf1}:{w1:.0f}%/{etf2}:{w2:.0f}% | 예정 {sa_action}"
                            )
                    except Exception as se:
                        lines.append(f"- 전략: {s_name}({s_weight:.0f}%) | 계산 실패: {se}")
                lines.append("")
            else:
                lines.append("<b>[KIS 연금저축]</b> 인증 실패")
                lines.append("")
        except Exception as e:
            lines.append(f"<b>[KIS 연금저축]</b> 조회 실패: {e}")
            lines.append("")
    else:
        lines.append("<b>[KIS 연금저축]</b> API 키 미설정")
        lines.append("")

    if asset_rows:
        total_all = sum(v for _, v in asset_rows)
        head = [f"<b>전체 자산 합계</b>: {total_all:,.0f}원"]
        for nm, v in asset_rows:
            head.append(f"- {nm}: {v:,.0f}원")
        head.append("")
        lines = lines[:2] + head + lines[2:]
    else:
        lines.insert(2, "조회 가능한 자산이 없습니다. API 키/계좌 설정을 확인해 주세요.")
        lines.insert(3, "")

    return "\n".join(lines)


def _normalize_numeric_series(series_obj, preferred_cols=("equity", "close", "Close")) -> pd.Series:
    """Series/DataFrame/array를 숫자 Series로 정규화한다."""
    if series_obj is None:
        return pd.Series(dtype=float)

    if isinstance(series_obj, pd.DataFrame):
        if series_obj.empty:
            return pd.Series(dtype=float)
        pick_col = None
        for col in preferred_cols:
            if col in series_obj.columns:
                pick_col = col
                break
        s = series_obj[pick_col] if pick_col else series_obj.iloc[:, 0]
    elif isinstance(series_obj, pd.Series):
        s = series_obj.copy()
    else:
        try:
            s = pd.Series(series_obj)
        except Exception:
            return pd.Series(dtype=float)

    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float)

    if isinstance(s.index, pd.DatetimeIndex):
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def _infer_periods_per_year(index_like) -> int:
    """DatetimeIndex 간격으로 연환산 주기를 추정한다."""
    if not isinstance(index_like, pd.DatetimeIndex) or len(index_like) < 2:
        return 252
    try:
        deltas = index_like.to_series().diff().dropna().dt.total_seconds()
        if deltas.empty:
            return 252
        med = float(deltas.median())
        if med <= 0:
            return 252
        annual = int(round((365.25 * 24 * 3600) / med))
        return max(1, min(annual, 365))
    except Exception:
        return 252


def _calc_equity_metrics(equity_series: pd.Series, periods_per_year: int = 252) -> dict:
    """equity Series로 기본 성과 지표를 계산한다."""
    eq = _normalize_numeric_series(equity_series, preferred_cols=("equity", "close", "Close"))
    if len(eq) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "avg_yearly_mdd": 0.0,
            "final_equity": float(eq.iloc[-1]) if len(eq) else 0.0,
        }

    init_val = float(eq.iloc[0])
    final_val = float(eq.iloc[-1])
    total_return = ((final_val / init_val) - 1.0) * 100.0 if init_val > 0 else 0.0

    if isinstance(eq.index, pd.DatetimeIndex):
        days = max((eq.index[-1] - eq.index[0]).days, 1)
        cagr = ((final_val / init_val) ** (365.0 / days) - 1.0) * 100.0 if init_val > 0 and final_val > 0 else 0.0
    else:
        years = max((len(eq) - 1) / max(periods_per_year, 1), 1 / max(periods_per_year, 1))
        cagr = ((final_val / init_val) ** (1.0 / years) - 1.0) * 100.0 if init_val > 0 and final_val > 0 else 0.0

    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    mdd = float(dd.min()) if len(dd) else 0.0

    rets = eq.pct_change().dropna()
    if len(rets) > 1 and float(rets.std()) > 0:
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(max(periods_per_year, 1)))
    else:
        sharpe = 0.0

    if isinstance(dd.index, pd.DatetimeIndex) and len(dd) > 0:
        yearly_mdd = dd.groupby(dd.index.year).min()
        avg_yearly_mdd = float(yearly_mdd.mean()) if len(yearly_mdd) > 0 else mdd
    else:
        avg_yearly_mdd = mdd

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "mdd": float(mdd),
        "sharpe": float(sharpe),
        "avg_yearly_mdd": float(avg_yearly_mdd),
        "final_equity": float(final_val),
    }


def _render_analysis_title(title: str):
    st.markdown(
        f"<div style='font-size:1.45rem; font-weight:800; line-height:1.3; margin:1.3rem 0 0.8rem 0;'>{title}</div>",
        unsafe_allow_html=True,
    )


def _apply_dd_hover_format(fig):
    """DD/낙폭 계열 trace hover를 소수점 2자리 + % 형식으로 통일."""
    try:
        for tr in getattr(fig, "data", []):
            _name = str(getattr(tr, "name", "") or "")
            _name_u = _name.upper()
            if ("DD" in _name_u) or ("DRAWDOWN" in _name_u) or ("낙폭" in _name):
                tr.hovertemplate = "%{y:.2f}%<extra>%{fullData.name}</extra>"
    except Exception:
        pass
    return fig


def _render_performance_analysis(
    equity_series,
    benchmark_series=None,
    strategy_metrics: dict | None = None,
    strategy_label: str = "전략",
    benchmark_label: str = "벤치마크",
    monte_carlo_sims: int = 400,
):
    """
    공통 분석 블록 렌더:
    - 벤치마크 대비 성과
    - 월별 성과
    - 몬테카를로
    - 켈리 공식
    """
    eq = _normalize_numeric_series(equity_series, preferred_cols=("equity",))
    if len(eq) < 2:
        st.info("분석에 필요한 전략 자산 데이터가 부족합니다.")
        return

    annual_n = _infer_periods_per_year(eq.index)
    calc_metrics = _calc_equity_metrics(eq, periods_per_year=annual_n)

    strategy_metrics = strategy_metrics or {}
    strat_total = _safe_float(strategy_metrics.get("total_return", calc_metrics["total_return"]), calc_metrics["total_return"])
    strat_cagr = _safe_float(strategy_metrics.get("cagr", calc_metrics["cagr"]), calc_metrics["cagr"])
    strat_mdd = _safe_float(strategy_metrics.get("mdd", calc_metrics["mdd"]), calc_metrics["mdd"])
    strat_sharpe = _safe_float(strategy_metrics.get("sharpe", calc_metrics["sharpe"]), calc_metrics["sharpe"])
    strat_avg_yearly_mdd = _safe_float(strategy_metrics.get("avg_yearly_mdd", calc_metrics["avg_yearly_mdd"]), calc_metrics["avg_yearly_mdd"])

    bench_metrics = None
    bench = _normalize_numeric_series(benchmark_series, preferred_cols=("close", "Close", "equity"))
    if len(bench) > 1:
        try:
            if isinstance(eq.index, pd.DatetimeIndex) and isinstance(bench.index, pd.DatetimeIndex):
                bench = bench[(bench.index >= eq.index.min()) & (bench.index <= eq.index.max())]
                if len(bench) > 1:
                    bench = bench.reindex(eq.index, method="ffill").dropna()
            if len(bench) > 1:
                bench_metrics = _calc_equity_metrics(bench, periods_per_year=annual_n)
        except Exception:
            bench_metrics = None

    _render_analysis_title("벤치마크 vs 전략 성과")
    perf_table = {
        "지표": ["총 수익률 (%)", "CAGR (%)", "MDD (%)", "샤프비율", "연평균MDD (%)"],
        strategy_label: [
            f"{strat_total:.2f}",
            f"{strat_cagr:.2f}",
            f"{strat_mdd:.2f}",
            f"{strat_sharpe:.2f}",
            f"{strat_avg_yearly_mdd:.2f}",
        ],
        benchmark_label: ["-", "-", "-", "-", "-"],
    }
    if bench_metrics is not None:
        perf_table[benchmark_label] = [
            f"{bench_metrics['total_return']:.2f}",
            f"{bench_metrics['cagr']:.2f}",
            f"{bench_metrics['mdd']:.2f}",
            f"{bench_metrics['sharpe']:.2f}",
            f"{bench_metrics['avg_yearly_mdd']:.2f}",
        ]
    st.table(pd.DataFrame(perf_table))

    _render_analysis_title("월별 수익률 (%)")
    if isinstance(eq.index, pd.DatetimeIndex):
        monthly_eq = eq.resample("ME").last().dropna()
        monthly_ret = monthly_eq.pct_change().dropna() * 100.0
        if len(monthly_ret) > 0:
            mr_df = pd.DataFrame(
                {
                    "연도": monthly_ret.index.year,
                    "월": monthly_ret.index.month,
                    "수익률": monthly_ret.values,
                }
            )
            pivot = mr_df.pivot_table(index="연도", columns="월", values="수익률", aggfunc="mean")
            if not pivot.empty:
                pivot = pivot.reindex(sorted(pivot.columns), axis=1)
                pivot.columns = [f"{int(m)}월" for m in pivot.columns]
                yearly_comp = mr_df.groupby("연도")["수익률"].apply(lambda x: (np.prod(1.0 + x / 100.0) - 1.0) * 100.0)
                pivot["연간"] = yearly_comp

                def _color_ret(v):
                    if pd.isna(v):
                        return ""
                    fg = "#D32F2F" if v >= 0 else "#1565C0"
                    bg = "#FFF3F3" if v >= 0 else "#E3F2FD"
                    return f"color: {fg}; background-color: {bg}; font-weight: 600"

                st.dataframe(pivot.style.format("{:.1f}").map(_color_ret), use_container_width=True)
            else:
                st.info("월별 수익률을 표시할 데이터가 부족합니다.")
        else:
            st.info("월별 수익률 데이터가 부족합니다.")
    else:
        st.info("월별 수익률은 날짜 인덱스 데이터에서만 계산할 수 있습니다.")

    _render_analysis_title("몬테카를로 시뮬레이션")
    rets = eq.pct_change().dropna()
    if len(rets) > 15:
        n_sims = int(max(100, min(int(monte_carlo_sims), 2000)))
        n_steps = int(min(len(rets), 750))
        init_cap = float(eq.iloc[-(n_steps + 1)])
        rng = np.random.default_rng(42)
        sampled = rng.choice(rets.values, size=(n_sims, n_steps), replace=True)
        paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        paths[:, 0] = init_cap
        paths[:, 1:] = init_cap * np.cumprod(1.0 + sampled, axis=1)

        final_values = paths[:, -1]
        p5, p25, p50, p75, p95 = [float(np.percentile(final_values, p)) for p in (5, 25, 50, 75, 95)]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("5%ile", f"{p5:,.0f}원")
        mc2.metric("25%ile", f"{p25:,.0f}원")
        mc3.metric("중앙값", f"{p50:,.0f}원")
        mc4.metric("75%ile", f"{p75:,.0f}원")
        mc5.metric("95%ile", f"{p95:,.0f}원")

        p5_path = np.percentile(paths, 5, axis=0)
        p50_path = np.percentile(paths, 50, axis=0)
        p95_path = np.percentile(paths, 95, axis=0)
        x_vals = list(range(n_steps + 1))
        actual_path = eq.iloc[-(n_steps + 1):].values

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(x=x_vals, y=p95_path, mode="lines", line=dict(width=0), name="95%ile", showlegend=False))
        fig_mc.add_trace(
            go.Scatter(
                x=x_vals,
                y=p5_path,
                mode="lines",
                line=dict(width=0),
                name="5%ile",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(255,193,7,0.15)",
            )
        )
        fig_mc.add_trace(go.Scatter(x=x_vals, y=p50_path, mode="lines", name="중앙 경로", line=dict(color="goldenrod", width=2)))
        fig_mc.add_trace(go.Scatter(x=x_vals, y=actual_path, mode="lines", name="실제 경로", line=dict(color="royalblue", width=2)))
        fig_mc.update_layout(
            yaxis_title="자산 (원)",
            xaxis_title="스텝",
            height=340,
            margin=dict(l=0, r=0, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        st.caption(f"부트스트랩 조건: {n_sims}회, 스텝 {n_steps}개, 평균 수익률 {rets.mean()*100:.3f}%, 표준편차 {rets.std()*100:.3f}%")
    else:
        st.info("몬테카를로 시뮬레이션에 필요한 수익률 데이터가 부족합니다.")

    _render_analysis_title("켈리 공식 (Kelly Criterion)")
    if len(rets) > 15:
        win_rets = rets[rets > 0]
        loss_rets = rets[rets < 0]
        if len(win_rets) == 0 or len(loss_rets) == 0:
            st.info("승/패 데이터가 부족해 켈리 공식을 계산할 수 없습니다.")
            return

        win_prob = float(len(win_rets) / len(rets))
        loss_prob = 1.0 - win_prob
        avg_win = float(win_rets.mean())
        avg_loss = float(abs(loss_rets.mean()))
        payoff = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
        kelly_full = max(0.0, min(win_prob - (loss_prob / payoff if payoff > 0 else 0.0), 1.0))
        kelly_half = kelly_full / 2.0
        kelly_quarter = kelly_full / 4.0

        kc1, kc2, kc3, kc4, kc5 = st.columns(5)
        kc1.metric("승률", f"{win_prob*100:.1f}%")
        kc2.metric("손익비 (W/L)", f"{payoff:.2f}")
        kc3.metric("Full Kelly", f"{kelly_full*100:.1f}%")
        kc4.metric("Half Kelly", f"{kelly_half*100:.1f}%")
        kc5.metric("Quarter Kelly", f"{kelly_quarter*100:.1f}%")

        st.markdown(
            f"<div style='background:#F3F8FF;border:1px solid #BBDEFB;border-radius:8px;padding:12px 16px;margin:8px 0;font-size:0.92rem'>"
            f"<b>해석:</b> Full Kelly 기준 권장 비중은 <b>{kelly_full*100:.1f}%</b>입니다. "
            f"실전 운용은 변동성 대응을 위해 <b>Half Kelly ({kelly_half*100:.1f}%)</b> 이하를 권장합니다.<br>"
            f"<span style='color:#666'>평균 수익: +{avg_win*100:.3f}% | 평균 손실: -{avg_loss*100:.3f}% | "
            f"승률: {win_prob*100:.1f}% ({len(win_rets)}/{len(rets)}회)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("켈리 공식 계산에 필요한 수익률 데이터가 부족합니다.")


def render_telegram_sidebar(prefix: str = "coin"):
    """Sidebar Telegram settings (load from .env/secrets, save to user config)."""
    tg_token_default = str(config.get("telegram_bot_token", "") or _get_runtime_value("TELEGRAM_BOT_TOKEN", ""))
    tg_chat_default = str(config.get("telegram_chat_id", "") or _get_runtime_value("TELEGRAM_CHAT_ID", ""))

    with st.sidebar.expander("📨 텔레그램 알림", expanded=False):
        tg_token = st.text_input(
            "봇 토큰",
            value=tg_token_default,
            type="password",
            key=f"{prefix}_telegram_bot_token",
            disabled=IS_CLOUD,
        )
        tg_chat = st.text_input(
            "채팅 ID",
            value=tg_chat_default,
            key=f"{prefix}_telegram_chat_id",
            disabled=IS_CLOUD,
        )

        if IS_CLOUD:
            st.caption("Cloud 환경에서는 편집/저장이 비활성화됩니다.")
            return

        c1, c2 = st.columns(2)
        with c1:
            if st.button("저장", key=f"{prefix}_telegram_save"):
                new_cfg = config.copy()
                new_cfg["telegram_bot_token"] = str(tg_token).strip()
                new_cfg["telegram_chat_id"] = str(tg_chat).strip()
                save_config(new_cfg)
                st.success("텔레그램 설정을 저장했습니다.")
                st.rerun()
        with c2:
            if st.button("테스트 전송", key=f"{prefix}_telegram_test"):
                if not tg_token or not tg_chat:
                    st.warning("봇 토큰과 채팅 ID를 입력해 주세요.")
                else:
                    try:
                        with st.spinner("전체 자산/전략 상태를 조회하고 메시지를 구성하는 중..."):
                            text = _build_telegram_test_report()
                        ok, detail = _send_telegram_message(str(tg_token).strip(), str(tg_chat).strip(), text)
                        if ok:
                            st.success("테스트 리포트 전송 성공")
                        else:
                            st.error(f"전송 실패: {detail}")
                    except Exception as e:
                        st.error(f"전송 오류: {e}")

        st.divider()
        st.markdown("##### 자동 알림 발송 시점/내용")
        st.markdown(
            """
| 발송 시점 (KST) | 발송 내용 | 실행 주체 |
|---|---|---|
| 평일 09:00 | 일일 자산 현황 (업비트/키움/KIS ISA/연금저축 잔고, 보유) | `daily_status` |
| 평일 15:20 | 헬스체크 리포트 (인증/잔고/시세/시그널/가상주문 점검 결과) | `health_check` |
| 전략 실행 시 | 전략별 주문 요약 (BUY/SELL/HOLD, 대상 종목/금액/수량) | `upbit`, `kiwoom_gold`, `kis_isa`, `kis_pension` |
            """
        )
        st.caption("자동 알림은 GitHub Actions에서 실행될 때 발송됩니다. 로컬 Streamlit은 `테스트 전송`만 즉시 발송합니다.")
        st.caption("알림 미수신 시: 1) GitHub Secrets의 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID 설정 2) Actions 스케줄 활성화 3) 워크플로우 실패 로그를 확인하세요.")

# Cloud 환경 감지 (Streamlit Cloud에서는 HOSTNAME이 *.streamlit.app 또는 /mount/src 경로)
IS_CLOUD = os.path.exists("/mount/src") or "streamlit.app" in os.getenv("HOSTNAME", "")

st.set_page_config(page_title="업비트 자동매매", layout="wide")

# --- Custom CSS for Better Readability ---
st.markdown("""
    <style>
    /* Global Font Adjustments */
    html, body, [class*="css"] {
        font-size: 18px;
    }
    .stMarkdown p {
        font-size: 18px !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
        color: #666;
    }
    [data-testid="stMetricDelta"] {
        font-size: 16px !important;
    }

    /* Expander Headers */
    .streamlit-expanderHeader {
        font-size: 22px !important;
        font-weight: 600 !important;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    /* Sidebar Input Labels */
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Width Override (PC) */
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 520px !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    }

    /* === 글자 겹침 방지 === */
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        overflow: hidden !important;
    }
    [data-testid="column"] {
        overflow: hidden !important;
    }
    /* 탭 버튼 겹침 방지 */
    [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    /* 셀렉트박스/인풋 라벨 겹침 방지 */
    .stSelectbox label, .stNumberInput label, .stDateInput label, .stTextInput label {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        max-width: 100% !important;
    }
    /* 캡션/텍스트 겹침 방지 */
    .stCaption, .stMarkdown {
        word-break: break-word !important;
        overflow-wrap: break-word !important;
    }

    /* ===== Mobile Responsive ===== */
    @media (max-width: 768px) {
        html, body, [class*="css"] {
            font-size: 14px;
        }
        .stMarkdown p {
            font-size: 14px !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 12px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        .streamlit-expanderHeader {
            font-size: 16px !important;
        }
        button[data-baseweb="tab"] {
            font-size: 11px !important;
            padding: 4px 8px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-baseweb="tab-list"] {
            gap: 2px !important;
        }
        /* 모바일 메트릭 겹침 방지 */
        [data-testid="stMetricValue"] {
            font-size: 18px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 11px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stSidebar"] {
            min-width: 280px !important;
            max-width: 320px !important;
        }
        /* 모바일에서 컬럼 세로 스택 */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > div {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        /* 차트 높이 조정 */
        .js-plotly-plot {
            max-height: 250px !important;
        }
        /* 데이터프레임 가로 스크롤 */
        [data-testid="stDataFrame"] {
            overflow-x: auto !important;
        }
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    }

    </style>
""", unsafe_allow_html=True)

def render_gold_mode():
    """금(Gold) 현물 거래 모드 - 키움증권 KRX 금시장 (코인 탭과 동일한 구조)"""
    from kiwoom_gold import KiwoomGoldTrader, GOLD_CODE_1KG
    from backtest.engine import BacktestEngine

    st.title("🥇 Gold Trading System (키움증권 KRX)")

    # ── 사이드바: 설정 ─────────────────────────────────────
    st.sidebar.header("Gold 설정")

    # API Keys
    try:
        kiwoom_ak      = st.secrets.get("Kiwoom_App_Key", "")
        kiwoom_sk      = st.secrets.get("Kiwoom_Secret_Key", "")
        kiwoom_account = st.secrets.get("KIWOOM_ACCOUNT", "")
    except Exception:
        kiwoom_ak      = os.getenv("Kiwoom_App_Key", "")
        kiwoom_sk      = os.getenv("Kiwoom_Secret_Key", "")
        kiwoom_account = os.getenv("KIWOOM_ACCOUNT", "")

    if IS_CLOUD:
        st.sidebar.info("📱 조회 전용 모드 (Cloud)")
    else:
        with st.sidebar.expander("키움 API Keys", expanded=False):
            kiwoom_ak      = st.text_input("App Key",    value=kiwoom_ak,      type="password", key="kiwoom_ak")
            kiwoom_sk      = st.text_input("Secret Key", value=kiwoom_sk,      type="password", key="kiwoom_sk")
            kiwoom_account = st.text_input("계좌번호",    value=kiwoom_account, key="kiwoom_acc")

    # 전략 설정 (코인 포트폴리오와 동일하게 다중 전략 지원)
    st.sidebar.subheader("전략 설정")
    st.sidebar.caption("여러 전략을 추가하여 포트폴리오를 구성할 수 있습니다.")

    _gold_cfg_default = [{"strategy": "Donchian", "buy": 90, "sell": 55, "weight": 100}]
    _gold_cfg = config.get("gold_strategy", _gold_cfg_default)

    df_gold_strat = pd.DataFrame(_gold_cfg)
    if IS_CLOUD:
        st.sidebar.dataframe(df_gold_strat, use_container_width=True, hide_index=True)
        edited_gold_strat = df_gold_strat
    else:
        edited_gold_strat = st.sidebar.data_editor(
            df_gold_strat, num_rows="dynamic", use_container_width=True, hide_index=True,
            key="gold_strat_editor",
            column_config={
                "strategy": st.column_config.SelectboxColumn("전략", options=["Donchian", "SMA"], required=True),
                "buy":      st.column_config.NumberColumn("매수", min_value=5, max_value=300, step=1, required=True),
                "sell":     st.column_config.NumberColumn("매도", min_value=0, max_value=300, step=1, required=True, help="Donchian 매도 채널 (SMA는 무시됨, 0=매수의 절반)"),
                "weight":   st.column_config.NumberColumn("비중 %", min_value=1, max_value=100, step=1, required=True),
            },
        )

    # 비중 검증
    gold_total_weight = int(edited_gold_strat["weight"].sum())
    if gold_total_weight > 100:
        st.sidebar.error(f"총 비중이 {gold_total_weight}% 입니다. (100% 이하로 설정해주세요)")
    else:
        gold_cash_weight = 100 - gold_total_weight
        st.sidebar.info(f"투자 비중: {gold_total_weight}% | 현금: {gold_cash_weight}%")

    # 골드 포트폴리오 리스트 생성
    gold_portfolio_list = []
    for _, row in edited_gold_strat.iterrows():
        bp = int(row.get("buy", 90))
        sp = int(row.get("sell", 0) or 0)
        if sp == 0:
            sp = max(5, bp // 2)
        gold_portfolio_list.append({
            "strategy": str(row.get("strategy", "Donchian")),
            "buy_period": bp,
            "sell_period": sp,
            "weight": int(row.get("weight", 100)),
        })

    # 첫 번째 전략 (기본값)
    if gold_portfolio_list:
        _g_first = gold_portfolio_list[0]
        buy_period = _g_first["buy_period"]
        sell_period = _g_first["sell_period"]
    else:
        buy_period = 90
        sell_period = 55

    # 공통 설정
    st.sidebar.subheader("공통 설정")
    _gold_start_default = config.get("gold_start_date", "2022-06-01")
    gold_start_date = st.sidebar.date_input(
        "기준 시작일", value=pd.to_datetime(_gold_start_default).date(),
        help="백테스트 평가 시작일", disabled=IS_CLOUD, key="gold_start_date"
    )
    _gold_cap_default = config.get("gold_initial_cap", 10_000_000)
    gold_initial_cap = st.sidebar.number_input(
        "초기 자본금 (KRW)", value=_gold_cap_default, step=100_000, format="%d",
        disabled=IS_CLOUD, key="gold_initial_cap"
    )
    st.sidebar.caption(f"설정: **{gold_initial_cap:,.0f} KRW**")

    if not IS_CLOUD:
        if st.sidebar.button("💾 Gold 설정 저장", key="gold_save_btn"):
            new_gold_cfg = config.copy()
            new_gold_cfg["gold_strategy"]    = edited_gold_strat.to_dict("records")
            new_gold_cfg["gold_start_date"]  = str(gold_start_date)
            new_gold_cfg["gold_initial_cap"] = gold_initial_cap
            save_config(new_gold_cfg)
            st.sidebar.success("저장 완료!")

    # ── 트레이더 + 백그라운드 워커 초기화 ────────────────────
    from data_manager import GoldDataWorker

    @st.cache_resource
    def _get_gold_trader(ak, sk, acct):
        t = KiwoomGoldTrader(is_mock=False)
        t.app_key = ak
        t.app_secret = sk
        t.account_no = acct
        return t

    @st.cache_resource
    def _get_gold_worker(_trader):
        """백그라운드 워커: 잔고/시세/호가를 3초마다 병렬 갱신"""
        w = GoldDataWorker()
        w.configure(_trader)
        w.start()
        return w

    gold_trader = None
    gold_worker = None
    if kiwoom_ak and kiwoom_sk:
        gold_trader = _get_gold_trader(kiwoom_ak, kiwoom_sk, kiwoom_account)
        gold_worker = _get_gold_worker(gold_trader)

    # ── 데이터 로드 헬퍼 (parquet 캐시 → 워커 → CSV 폴백) ──
    def load_gold_data(buy_p: int) -> pd.DataFrame | None:
        """일봉 데이터: parquet 캐시 → 워커 차트 → CSV 폴백."""
        import data_cache
        # 1순위: parquet 캐시 (사전 다운로드된 대량 데이터)
        cached = data_cache.load_cached_gold()
        if cached is not None and len(cached) >= buy_p + 5:
            return cached
        # 2순위: 백그라운드 워커 차트 데이터
        if gold_worker:
            df_w = gold_worker.get_chart()
            if df_w is not None and len(df_w) >= buy_p + 5:
                return df_w
        # 3순위: CSV 파일 (오프라인 폴백)
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "krx_gold_daily.csv")
        if os.path.exists(csv_path):
            df_csv = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            df_csv.columns = [c.lower() for c in df_csv.columns]
            if "open" not in df_csv.columns: df_csv["open"] = df_csv["close"]
            if "high" not in df_csv.columns: df_csv["high"] = df_csv["close"]
            if "low"  not in df_csv.columns: df_csv["low"]  = df_csv["close"]
            return df_csv
        return None

    # ── 탭 구성 ───────────────────────────────────────────
    tab_g1, tab_g2, tab_g3, tab_g4 = st.tabs(
        ["🚀 실시간 모니터링", "🛒 수동 주문", "📊 백테스트", "💳 수수료/세금"]
    )

    # ══════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링 (코인 탭1과 동일 구조)
    # ══════════════════════════════════════════════════════
    with tab_g1:
        st.header("실시간 금 모니터링")
        _strat_labels = [f"{g['strategy']}({g['buy_period']}/{g['sell_period']}) {g['weight']}%" for g in gold_portfolio_list]
        st.caption(f"전략: {', '.join(_strat_labels)} | 초기자본: {gold_initial_cap:,.0f}원")

        # 새로고침
        col_r1, col_r2 = st.columns([1, 5])
        with col_r1:
            if st.button("🔄 새로고침", key="gold_refresh"):
                for k in list(st.session_state.keys()):
                    if k.startswith("__gc_") or k.startswith("__gt_"):
                        del st.session_state[k]
                st.cache_data.clear()
                st.rerun()

        # 계좌 잔고 (워커에서 읽기 — 블로킹 없음)
        with st.expander("💰 계좌 현황", expanded=True):
            if not gold_worker:
                st.warning("사이드바에서 키움 API Key를 입력해주세요.")
            else:
                bal = gold_worker.get('balance')
                if bal:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("예수금", f"{bal['cash_krw']:,.0f}원")
                    b2.metric("금 보유량", f"{bal['gold_qty']:.4f}g")
                    b3.metric("금 평가금액", f"{bal['gold_eval']:,.0f}원")
                    total_asset = bal['cash_krw'] + bal['gold_eval']
                    pnl = total_asset - gold_initial_cap
                    b4.metric("총 평가", f"{total_asset:,.0f}원", delta=f"{pnl:+,.0f}원")
                elif not gold_worker.is_ready():
                    st.info("데이터 로딩 중... (백그라운드 갱신)")
                    bal = None
                else:
                    st.warning("잔고 조회 실패 (API 인증 확인)")
                    bal = None

        # ── 백테스트 자산현황 vs 실제 자산 비교 ──
        with st.expander("📊 백테스트 vs 실제 자산 비교", expanded=True):
            _bt_max_p = max((g['buy_period'] for g in gold_portfolio_list), default=90)
            _bt_df_gold = load_gold_data(_bt_max_p)

            if _bt_df_gold is None or len(_bt_df_gold) < _bt_max_p + 5:
                st.warning("백테스트용 데이터가 부족합니다. 백테스트 탭에서 사전 다운로드를 실행하세요.")
            else:
                # 캐시 키 생성
                _bt_ck = f"__gt_bt_cmp_{gold_initial_cap}_{gold_start_date}_{len(_bt_df_gold)}_{_bt_df_gold.index[-1]}"
                for _gp in gold_portfolio_list:
                    _bt_ck += f"_{_gp['strategy']}_{_gp['buy_period']}_{_gp['sell_period']}_{_gp['weight']}"

                if _bt_ck not in st.session_state:
                    _total_theo = 0.0
                    _strat_res = []
                    _total_w = sum(g['weight'] for g in gold_portfolio_list)
                    _cash_w = max(0, 100 - _total_w)
                    _reserved = gold_initial_cap * _cash_w / 100

                    for _gp in gold_portfolio_list:
                        _per_cap = gold_initial_cap * _gp['weight'] / 100
                        _eng = BacktestEngine()
                        _sr = _gp['sell_period'] / _gp['buy_period'] if _gp['buy_period'] > 0 else 0.5
                        _bt_r = _eng.run_backtest(
                            ticker=None, df=_bt_df_gold,
                            period=_gp['buy_period'], interval="day",
                            fee=0.003, start_date=str(gold_start_date),
                            initial_balance=_per_cap,
                            strategy_mode=_gp['strategy'],
                            sell_period_ratio=_sr, slippage=0.0,
                        )
                        if "error" not in _bt_r:
                            _perf = _bt_r['performance']
                            _total_theo += _perf['final_equity']
                            _strat_res.append({
                                "label": f"{_gp['strategy']}({_gp['buy_period']}/{_gp['sell_period']})",
                                "weight": _gp['weight'], "initial": _per_cap,
                                "equity": _perf['final_equity'],
                                "return_pct": _perf['total_return'],
                                "status": _perf.get('final_status', 'UNKNOWN'),
                            })
                        else:
                            _total_theo += _per_cap
                            _strat_res.append({
                                "label": f"{_gp['strategy']}({_gp['buy_period']}/{_gp['sell_period']})",
                                "weight": _gp['weight'], "initial": _per_cap,
                                "equity": _per_cap, "return_pct": 0, "status": "ERROR",
                            })

                    _total_theo += _reserved
                    st.session_state[_bt_ck] = {
                        "theo": _total_theo, "strats": _strat_res, "cash": _reserved,
                    }

                _bt_cmp = st.session_state[_bt_ck]
                _theo_total = _bt_cmp["theo"]
                _theo_return = (_theo_total - gold_initial_cap) / gold_initial_cap * 100 if gold_initial_cap > 0 else 0

                # 실제 자산
                _g_bal = gold_worker.get('balance') if gold_worker else None
                _actual_total = (_g_bal['cash_krw'] + _g_bal['gold_eval']) if _g_bal else 0.0
                _actual_return = (_actual_total - gold_initial_cap) / gold_initial_cap * 100 if gold_initial_cap > 0 else 0
                _diff_val = _actual_total - _theo_total

                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("초기 자본", f"{gold_initial_cap:,.0f}원")
                ac2.metric("이론 총자산", f"{_theo_total:,.0f}원", delta=f"{_theo_return:+.2f}%")
                ac3.metric("실제 총자산",
                           f"{_actual_total:,.0f}원" if _g_bal else "조회 불가",
                           delta=f"{_actual_return:+.2f}%" if _g_bal else None)
                ac4.metric("차이 (실제-이론)",
                           f"{_diff_val:+,.0f}원" if _g_bal else "-",
                           delta_color="off" if abs(_diff_val) < 1000 else "inverse")

                # 전략별 상세
                _detail = []
                for _sr in _bt_cmp["strats"]:
                    _detail.append({
                        "전략": _sr["label"], "비중": f"{_sr['weight']}%",
                        "배분 자본": f"{_sr['initial']:,.0f}원",
                        "이론 자산": f"{_sr['equity']:,.0f}원",
                        "수익률": f"{_sr['return_pct']:+.2f}%",
                        "포지션": _sr["status"],
                    })
                if _bt_cmp["cash"] > 0:
                    _cw = max(0, 100 - sum(g['weight'] for g in gold_portfolio_list))
                    _detail.append({
                        "전략": "현금 예비", "비중": f"{_cw}%",
                        "배분 자본": f"{_bt_cmp['cash']:,.0f}원",
                        "이론 자산": f"{_bt_cmp['cash']:,.0f}원",
                        "수익률": "0.00%", "포지션": "CASH",
                    })
                st.dataframe(pd.DataFrame(_detail), use_container_width=True, hide_index=True)

        # 시그널 차트 (다중 전략 지원)
        with st.expander("📊 시그널 모니터링", expanded=True):
            # 가장 큰 buy_period로 데이터 로드
            max_buy_p = max((g['buy_period'] for g in gold_portfolio_list), default=90)
            df_gold = load_gold_data(max_buy_p)

            if df_gold is None or len(df_gold) < max_buy_p + 5:
                st.warning("일봉 데이터 부족. API 연결 또는 krx_gold_daily.csv를 확인하세요.")
            else:
                close_now = float(df_gold['close'].iloc[-1])
                gold_signal_rows = []

                # 전략별 차트 렌더링
                n_strats = len(gold_portfolio_list)
                if n_strats > 0:
                    chart_cols = st.columns(n_strats)

                for gi, gp in enumerate(gold_portfolio_list):
                    g_strat = gp['strategy']
                    g_bp = gp['buy_period']
                    g_sp = gp['sell_period']
                    g_wt = gp['weight']

                    if g_strat == "Donchian":
                        g_upper = df_gold['high'].rolling(window=g_bp).max().shift(1)
                        g_lower = df_gold['low'].rolling(window=g_sp).min().shift(1)
                        g_buy_target = float(g_upper.iloc[-1])
                        g_sell_target = float(g_lower.iloc[-1])
                        g_buy_dist = (close_now - g_buy_target) / g_buy_target * 100 if g_buy_target else 0
                        g_sell_dist = (close_now - g_sell_target) / g_sell_target * 100 if g_sell_target else 0
                        in_pos = False
                        for i in range(len(df_gold)):
                            u = g_upper.iloc[i]; l = g_lower.iloc[i]; c = float(df_gold['close'].iloc[i])
                            if not pd.isna(u) and c > u: in_pos = True
                            elif not pd.isna(l) and c < l: in_pos = False
                        g_signal = ("SELL" if close_now < g_sell_target else "HOLD") if in_pos else \
                                   ("BUY" if close_now > g_buy_target else "WAIT")
                        g_pos_label = "보유" if in_pos else "현금"
                    else:
                        g_sma = df_gold['close'].rolling(window=g_bp).mean()
                        g_buy_target = float(g_sma.iloc[-1])
                        g_sell_target = g_buy_target
                        g_buy_dist = (close_now - g_buy_target) / g_buy_target * 100 if g_buy_target else 0
                        g_sell_dist = g_buy_dist
                        g_signal = "BUY" if close_now > g_buy_target else "SELL"
                        g_pos_label = "보유" if close_now > g_buy_target else "현금"

                    gold_signal_rows.append({
                        "전략": f"{g_strat} {g_bp}/{g_sp}",
                        "비중": f"{g_wt}%",
                        "현재가": f"{close_now:,.0f}",
                        "매수목표": f"{g_buy_target:,.0f}",
                        "매도목표": f"{g_sell_target:,.0f}",
                        "매수이격도": f"{g_buy_dist:+.2f}%",
                        "매도이격도": f"{g_sell_dist:+.2f}%",
                        "포지션": g_pos_label,
                        "시그널": g_signal,
                    })

                    # 차트 렌더링
                    g_sig_color = "green" if g_signal == "BUY" else ("red" if g_signal == "SELL" else ("blue" if g_signal == "WAIT" else "gray"))
                    df_chart = df_gold.iloc[-120:]
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Candlestick(
                        x=df_chart.index, open=df_chart['open'],
                        high=df_chart['high'], low=df_chart['low'],
                        close=df_chart['close'], name='금 일봉',
                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                    ))
                    if g_strat == "Donchian":
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_upper.loc[df_chart.index],
                            name=f'상단({g_bp})', line=dict(color='green', width=1.5, dash='dot')))
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_lower.loc[df_chart.index],
                            name=f'하단({g_sp})', line=dict(color='red', width=1.5, dash='dot')))
                    else:
                        fig_g.add_trace(go.Scatter(
                            x=df_chart.index, y=g_sma.loc[df_chart.index],
                            name=f'SMA({g_bp})', line=dict(color='orange', width=2)))
                    fig_g.update_layout(
                        title=f"KRX 금현물 {g_strat}({g_bp}/{g_sp}) [{g_pos_label}] [{g_buy_dist:+.1f}%]",
                        title_font_color=g_sig_color,
                        height=400, margin=dict(l=0, r=0, t=70, b=30),
                        xaxis_rangeslider_visible=False, showlegend=True,
                        xaxis=dict(showticklabels=True, tickformat='%Y/%m/%d', tickangle=-45, nticks=10),
                        yaxis_title="가격 (원/g)",
                    )
                    with chart_cols[gi]:
                        st.plotly_chart(fig_g, use_container_width=True)

                # 시그널 요약 테이블
                if gold_signal_rows:
                    st.dataframe(pd.DataFrame(gold_signal_rows), use_container_width=True, hide_index=True)

        # 자동매매 규칙
        with st.expander("⚖️ 자동매매 규칙", expanded=False):
            rules_lines = ["**실행 시점**: GitHub Actions - 매 평일 KST 09:05\n"]
            for gp in gold_portfolio_list:
                if gp['strategy'] == "Donchian":
                    rules_lines.append(f"**{gp['strategy']}({gp['buy_period']}/{gp['sell_period']})** 비중 {gp['weight']}%")
                    rules_lines.append(f"- 매수: 종가 > {gp['buy_period']}일 최고가 → 시장가 매수")
                    rules_lines.append(f"- 매도: 종가 < {gp['sell_period']}일 최저가 → 시장가 매도\n")
                else:
                    rules_lines.append(f"**{gp['strategy']}({gp['buy_period']})** 비중 {gp['weight']}%")
                    rules_lines.append(f"- 매수: 종가 > SMA({gp['buy_period']}) → 시장가 매수")
                    rules_lines.append(f"- 매도: 종가 < SMA({gp['buy_period']}) → 시장가 매도\n")
            rules_lines.append("**수수료**: 키움증권 0.165% (왕복 ~0.34%)")
            st.markdown("\n".join(rules_lines))

    # ══════════════════════════════════════════════════════
    # Tab 2: 수동 주문 — HTS 스타일 (코인 트레이딩 패널과 동일 구조)
    # ══════════════════════════════════════════════════════
    with tab_g2:
        st.header("수동 주문")

        if not gold_trader:
            st.warning("API Key를 사이드바에서 입력해주세요.")
        else:
            GOLD_TICK = 10  # KRX 금현물 호가단위: 10원
            GOLD_MIN_QTY = 1.0  # 1KG 종목 최소 수량: 1g

            def _gold_align_price(price, tick=GOLD_TICK):
                return round(price / tick) * tick

            # ── 호가 선택 콜백 ──
            def _on_gold_ob_select():
                sel = st.session_state.get('_gold_ob_sel', '')
                try:
                    price_str = sel.split(' ', 1)[1].replace(',', '')
                    chosen = int(float(price_str))
                    st.session_state['gold_buy_price'] = chosen
                    st.session_state['gold_sell_price'] = chosen
                except (IndexError, ValueError):
                    pass

            # ═══ 트레이딩 패널 (3초 자동갱신, 워커에서 읽기만 → 블로킹 없음) ═══
            @st.fragment
            def gold_trading_panel():
                # ── 워커에서 즉시 읽기 (API 호출 없음) ──
                g_bal = gold_worker.get('balance') if gold_worker else None
                g_price = gold_worker.get('price', 0) if gold_worker else 0

                g_cash = g_bal['cash_krw'] if g_bal else 0.0
                g_qty  = g_bal['gold_qty'] if g_bal else 0.0
                g_eval = g_bal['gold_eval'] if g_bal else 0.0
                g_hold_val = g_qty * g_price if g_price > 0 else g_eval

                # ── 상단 정보 바 ──
                gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                gc1.metric("현재가 (원/g)", f"{g_price:,.0f}")
                gc2.metric("금 보유", f"{g_qty:.2f}g")
                gc3.metric("평가금액", f"{g_hold_val:,.0f}원")
                gc4.metric("예수금", f"{g_cash:,.0f}원")
                gc5.metric("호가단위", f"{GOLD_TICK}원")

                # ── 최근 거래 알림 바 ──
                g_last_trade = st.session_state.get('_gold_last_trade')
                if g_last_trade:
                    gt_type = g_last_trade.get('type', '')
                    gt_time = g_last_trade.get('time', '')
                    gt_detail = g_last_trade.get('detail', '')
                    is_buy = '매수' in gt_type
                    g_color = '#D32F2F' if is_buy else '#1976D2'
                    gnc1, gnc2 = st.columns([6, 1])
                    gnc1.markdown(
                        f'<div style="padding:6px 12px;border-radius:6px;background:{g_color}22;'
                        f'border-left:4px solid {g_color};font-size:14px;">'
                        f'<b style="color:{g_color}">{gt_type}</b> | {gt_detail} | {gt_time}</div>',
                        unsafe_allow_html=True
                    )
                    if gnc2.button("✕", key="_gold_dismiss"):
                        del st.session_state['_gold_last_trade']
                        st.rerun()

                st.divider()

                # ═══ 메인 레이아웃: 호가창(좌) + 주문(우) ═══
                ob_col, order_col = st.columns([2, 3])

                # ── 좌: 호가창 (HTML 테이블) ──
                with ob_col:
                    price_labels = []

                    ob = gold_worker.get('orderbook') if gold_worker else None

                    if ob and ob.get('asks') and ob.get('bids'):
                        asks = ob['asks']  # 매도호가 (낮→높)
                        bids = ob['bids']  # 매수호가 (높→낮)

                        all_qtys = [a['qty'] for a in asks] + [b['qty'] for b in bids]
                        max_qty = max(all_qtys) if all_qtys else 1

                        html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                        html.append(
                            '<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666">'
                            '<td>구분</td><td style="text-align:right">잔량(g)</td>'
                            '<td style="text-align:right">가격(원)</td>'
                            '<td style="text-align:right">등락</td><td>비율</td></tr>'
                        )

                        ask_prices = []
                        bid_prices = []

                        # 매도호가 (높→낮 순서로 표시)
                        for a in reversed(asks):
                            ap = a['price']
                            aq = a['qty']
                            diff = ((ap / g_price) - 1) * 100 if g_price > 0 else 0
                            bar_w = int(aq / max_qty * 100) if max_qty > 0 else 0
                            ask_prices.append(ap)
                            html.append(
                                f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                                f'<td>매도</td>'
                                f'<td style="text-align:right">{aq:.2f}</td>'
                                f'<td style="text-align:right;font-weight:bold">{ap:,.0f}</td>'
                                f'<td style="text-align:right">{diff:+.2f}%</td>'
                                f'<td><div style="background:#1976D233;height:14px;width:{bar_w}%"></div></td></tr>'
                            )

                        # 현재가 구분선
                        html.append(
                            f'<tr style="background:#FFF3E0;border:2px solid #FF9800;height:32px;font-weight:bold">'
                            f'<td colspan="2" style="color:#E65100">현재가</td>'
                            f'<td style="text-align:right;color:#E65100;font-size:15px">{g_price:,.0f}</td>'
                            f'<td colspan="2"></td></tr>'
                        )

                        # 매수호가 (높→낮 순서)
                        for b in bids:
                            bp = b['price']
                            bq = b['qty']
                            diff = ((bp / g_price) - 1) * 100 if g_price > 0 else 0
                            bar_w = int(bq / max_qty * 100) if max_qty > 0 else 0
                            bid_prices.append(bp)
                            html.append(
                                f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                                f'<td>매수</td>'
                                f'<td style="text-align:right">{bq:.2f}</td>'
                                f'<td style="text-align:right;font-weight:bold">{bp:,.0f}</td>'
                                f'<td style="text-align:right">{diff:+.2f}%</td>'
                                f'<td><div style="background:#D32F2F33;height:14px;width:{bar_w}%"></div></td></tr>'
                            )

                        html.append('</table>')
                        st.markdown(''.join(html), unsafe_allow_html=True)

                        # 호가 선택 selectbox
                        ask_prices.reverse()  # 낮→높 → 높→낮으로 복원
                        for ap in ask_prices:
                            price_labels.append(f"매도 {ap:,.0f}")
                        price_labels.append(f"── {g_price:,.0f} ──")
                        for bp in bid_prices:
                            price_labels.append(f"매수 {bp:,.0f}")

                        st.selectbox(
                            "호가 선택 → 주문가 반영", price_labels,
                            index=len(ask_prices),
                            key="_gold_ob_sel", on_change=_on_gold_ob_select
                        )

                        # 스프레드 정보
                        if asks and bids:
                            spread = asks[0]['price'] - bids[0]['price']
                            spread_pct = (spread / g_price * 100) if g_price > 0 else 0
                            total_ask_qty = sum(a['qty'] for a in asks)
                            total_bid_qty = sum(b['qty'] for b in bids)
                            st.caption(
                                f"스프레드: {spread:,.0f}원 ({spread_pct:.3f}%) | "
                                f"매도잔량: {total_ask_qty:.2f}g | 매수잔량: {total_bid_qty:.2f}g"
                            )
                    else:
                        st.info("호가 데이터를 불러오는 중...")

                # ── 우: 주문 패널 ──
                with order_col:
                    buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                    with buy_tab:
                        gb_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="g_buy_type")

                        if gb_type == "시장가":
                            gb_amount = st.number_input(
                                "매수 금액 (원)", min_value=10_000,
                                value=min(int(g_cash * 0.99), 1_000_000) if g_cash > 10_000 else 10_000,
                                step=10_000, key="g_buy_amt"
                            )
                            # % 버튼
                            gqb1, gqb2, gqb3, gqb4 = st.columns(4)
                            if gqb1.button("10%", key="gb10"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.1)
                                st.rerun()
                            if gqb2.button("25%", key="gb25"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.25)
                                st.rerun()
                            if gqb3.button("50%", key="gb50"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.5)
                                st.rerun()
                            if gqb4.button("100%", key="gb100"):
                                st.session_state['g_buy_amt'] = int(g_cash * 0.999)
                                st.rerun()

                            if g_price > 0:
                                st.caption(f"예상 수량: ~{gb_amount / g_price:.2f}g")

                            if st.button("시장가 매수", type="primary", key="g_buy_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gb_amount < 10_000:
                                    st.toast("최소 주문금액: 10,000원", icon="⚠️")
                                elif gb_amount > g_cash:
                                    st.toast(f"예수금 부족 ({g_cash:,.0f}원)", icon="⚠️")
                                else:
                                    with st.spinner("매수 주문 중..."):
                                        if gold_trader.auth():
                                            cur_p = data_cache.get_gold_current_price_local_first(
                                                trader=gold_trader,
                                                code=GOLD_CODE_1KG,
                                                allow_api_fallback=True,
                                                ttl_sec=8.0,
                                            ) or g_price or 1
                                            buy_qty = round(gb_amount / cur_p, 2)
                                            result = gold_trader.send_order("BUY", GOLD_CODE_1KG, qty=buy_qty, ord_tp="3")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 시장가 매수! {gb_amount:,.0f}원 ≈ {buy_qty:.2f}g", icon="🟢")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "시장가 매수", "detail": f"{gb_amount:,.0f}원 ≈ {buy_qty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"매수 실패: {result}", icon="🔴")

                        else:  # 지정가
                            gbc1, gbc2 = st.columns(2)
                            gb_price = gbc1.number_input(
                                "매수 가격 (원/g)", min_value=1,
                                value=_gold_align_price(g_price * 0.99) if g_price > 0 else 100_000,
                                step=GOLD_TICK, key="gold_buy_price"
                            )
                            gb_qty = gbc2.number_input(
                                "매수 수량 (g)", min_value=0.01, value=max(GOLD_MIN_QTY, 1.0),
                                step=0.01, format="%.2f", key="g_buy_qty"
                            )
                            gb_total = gb_price * gb_qty
                            st.caption(f"총액: **{gb_total:,.0f}원** | 호가: {GOLD_TICK}원 | 최소: {GOLD_MIN_QTY}g")

                            gqbc1, gqbc2, gqbc3, gqbc4 = st.columns(4)
                            if gb_price > 0:
                                if gqbc1.button("10%", key="glb10"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.1 / gb_price, 2)
                                    st.rerun()
                                if gqbc2.button("25%", key="glb25"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.25 / gb_price, 2)
                                    st.rerun()
                                if gqbc3.button("50%", key="glb50"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.5 / gb_price, 2)
                                    st.rerun()
                                if gqbc4.button("100%", key="glb100"):
                                    st.session_state['g_buy_qty'] = round(g_cash * 0.999 / gb_price, 2)
                                    st.rerun()

                            if st.button("지정가 매수", type="primary", key="g_lbuy_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gb_total < 10_000:
                                    st.toast("최소 주문금액: 10,000원", icon="⚠️")
                                elif gb_total > g_cash:
                                    st.toast(f"예수금 부족 ({g_cash:,.0f}원)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매수 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("BUY", GOLD_CODE_1KG, qty=gb_qty, price=gb_price, ord_tp="1")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 지정가 매수 등록! {gb_price:,.0f}원 × {gb_qty:.2f}g", icon="🟢")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "지정가 매수", "detail": f"{gb_price:,.0f}원 × {gb_qty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

                    with sell_tab:
                        gs_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="g_sell_type")

                        if gs_type == "시장가":
                            gs_qty = st.number_input(
                                f"매도 수량 (g)", min_value=0.01,
                                value=g_qty if g_qty > 0 else 1.0,
                                step=0.01, format="%.2f", key="g_sell_qty"
                            )
                            gqs1, gqs2, gqs3, gqs4 = st.columns(4)
                            if gqs1.button("25%", key="gs25"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.25, 2)
                                st.rerun()
                            if gqs2.button("50%", key="gs50"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.5, 2)
                                st.rerun()
                            if gqs3.button("75%", key="gs75"):
                                st.session_state['g_sell_qty'] = round(g_qty * 0.75, 2)
                                st.rerun()
                            if gqs4.button("100%", key="gs100"):
                                st.session_state['g_sell_qty'] = round(g_qty, 2)
                                st.rerun()

                            if g_price > 0:
                                st.caption(f"예상 금액: ~{gs_qty * g_price:,.0f}원")

                            if st.button("시장가 매도", type="primary", key="g_sell_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gs_qty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif gs_qty > g_qty:
                                    st.toast(f"보유량 초과 ({g_qty:.2f}g)", icon="⚠️")
                                else:
                                    with st.spinner("매도 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("SELL", GOLD_CODE_1KG, qty=gs_qty, ord_tp="3")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 시장가 매도! {gs_qty:.2f}g", icon="🔴")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "시장가 매도", "detail": f"{gs_qty:.2f}g ≈ {gs_qty * g_price:,.0f}원",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"매도 실패: {result}", icon="🔴")

                        else:  # 지정가
                            gsc1, gsc2 = st.columns(2)
                            gs_price = gsc1.number_input(
                                "매도 가격 (원/g)", min_value=1,
                                value=_gold_align_price(g_price * 1.01) if g_price > 0 else 100_000,
                                step=GOLD_TICK, key="gold_sell_price"
                            )
                            gs_lqty = gsc2.number_input(
                                "매도 수량 (g)", min_value=0.01,
                                value=g_qty if g_qty > 0.01 else 1.0,
                                step=0.01, format="%.2f", key="g_sell_lqty"
                            )
                            gs_total = gs_price * gs_lqty
                            st.caption(f"총액: **{gs_total:,.0f}원** | 호가: {GOLD_TICK}원")

                            gqsc1, gqsc2, gqsc3, gqsc4 = st.columns(4)
                            if g_qty > 0:
                                if gqsc1.button("25%", key="gls25"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.25, 2)
                                    st.rerun()
                                if gqsc2.button("50%", key="gls50"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.5, 2)
                                    st.rerun()
                                if gqsc3.button("75%", key="gls75"):
                                    st.session_state['g_sell_lqty'] = round(g_qty * 0.75, 2)
                                    st.rerun()
                                if gqsc4.button("100%", key="gls100"):
                                    st.session_state['g_sell_lqty'] = round(g_qty, 2)
                                    st.rerun()

                            if st.button("지정가 매도", type="primary", key="g_lsell_exec", use_container_width=True, disabled=IS_CLOUD):
                                if gs_lqty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif gs_lqty > g_qty:
                                    st.toast(f"보유량 초과 ({g_qty:.2f}g)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매도 주문 중..."):
                                        if gold_trader.auth():
                                            result = gold_trader.send_order("SELL", GOLD_CODE_1KG, qty=gs_lqty, price=gs_price, ord_tp="1")
                                        else:
                                            result = None
                                    pass  # 워커가 자동 갱신
                                    if result and result.get("success"):
                                        st.toast(f"✅ 지정가 매도 등록! {gs_price:,.0f}원 × {gs_lqty:.2f}g", icon="🔴")
                                        st.session_state['_gold_last_trade'] = {
                                            "type": "지정가 매도", "detail": f"{gs_price:,.0f}원 × {gs_lqty:.2f}g",
                                            "time": time.strftime('%H:%M:%S')
                                        }
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

            gold_trading_panel()

    # ══════════════════════════════════════════════════════
    # Tab 3: 백테스트 (코인 탭4와 동일 구조 - 3개 서브탭)
    # ══════════════════════════════════════════════════════
    with tab_g3:
        gbt1, gbt2 = st.tabs(["📈 단일 백테스트", "🛠️ 파라미터 최적화"])

        # ── 서브탭1: 단일 백테스트 ──────────────────────────
        with gbt1:
            st.header("금현물 단일 백테스트")

            # ── 데이터 가용 범위 + 사전 다운로드 ──
            import data_cache as _dc_gold
            _gold_info = _dc_gold.gold_cache_info()
            if _gold_info["exists"]:
                _gi_start = _gold_info["start"]
                _gi_end = _gold_info["end"]
                _gi_start_str = _gi_start.strftime('%Y-%m-%d') if hasattr(_gi_start, 'strftime') else str(_gi_start)[:10]
                _gi_end_str = _gi_end.strftime('%Y-%m-%d') if hasattr(_gi_end, 'strftime') else str(_gi_end)[:10]
                st.info(f"사용 가능 데이터: **{_gold_info['rows']:,}**개 캔들 ({_gi_start_str} ~ {_gi_end_str})")
            else:
                st.warning("캐시된 Gold 데이터가 없습니다. 아래 버튼으로 사전 다운로드하세요.")

            if gold_trader and st.button("Gold 일봉 전체 다운로드 (2014~ 전체)", key="gold_predownload"):
                with st.status("Gold 일봉 다운로드 중...", expanded=True) as dl_status:
                    prog_dl = st.progress(0)
                    log_dl = st.empty()
                    def _dl_progress(fetched, total, msg):
                        pct = min(fetched / total, 1.0) if total > 0 else 1.0
                        prog_dl.progress(pct)
                        log_dl.text(msg)
                    df_dl = _dc_gold.fetch_and_cache_gold(gold_trader, count=5000, progress_callback=_dl_progress)
                    if df_dl is not None and len(df_dl) > 0:
                        dl_status.update(label=f"완료! {len(df_dl):,}개 캔들 다운로드됨", state="complete")
                        st.rerun()
                    else:
                        dl_status.update(label="다운로드 실패", state="error")

            st.divider()

            bt_col1, bt_col2, bt_col3 = st.columns(3)
            with bt_col1:
                bt_strategy = st.selectbox("전략", ["Donchian", "SMA"], key="gold_bt_strat")
            with bt_col2:
                bt_buy_p  = st.number_input("매수 기간", min_value=5, max_value=300, value=buy_period,  step=1, key="gold_bt_buy")
            with bt_col3:
                bt_sell_p = st.number_input("매도 기간", min_value=5, max_value=300, value=sell_period, step=1, key="gold_bt_sell",
                                            help="Donchian 매도 채널 (SMA는 무시됨)")

            bt_start = st.date_input("백테스트 시작일", value=gold_start_date, key="gold_bt_start")
            bt_cap   = st.number_input("초기 자본 (원)", value=gold_initial_cap, step=100_000, format="%d", key="gold_bt_cap")

            if st.button("🚀 백테스트 실행", key="gold_bt_run", type="primary"):
                df_bt = load_gold_data(bt_buy_p)
                if df_bt is None or len(df_bt) < bt_buy_p + 5:
                    st.error("데이터 부족. 사전 다운로드를 실행하세요.")
                else:
                    st.caption(f"조회된 캔들: {len(df_bt):,}개 ({df_bt.index[0].strftime('%Y-%m-%d')} ~ {df_bt.index[-1].strftime('%Y-%m-%d')})")
                    with st.spinner("백테스트 실행 중..."):
                        engine = BacktestEngine()
                        result = engine.run_backtest(
                            ticker=None, df=df_bt,
                            period=bt_buy_p,
                            interval="day",
                            fee=0.003,
                            start_date=str(bt_start),
                            initial_balance=bt_cap,
                            strategy_mode=bt_strategy,
                            sell_period_ratio=(bt_sell_p / bt_buy_p) if bt_strategy == "Donchian" else 1.0,
                            slippage=0.0,
                        )

                    if "error" in result:
                        st.error(f"백테스트 오류: {result['error']}")
                    else:
                        p = result["performance"]
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("총 수익률",  f"{p['total_return']:+.2f}%")
                        r2.metric("CAGR",      f"{p['cagr']:+.2f}%")
                        r3.metric("MDD",       f"{p['mdd']:.2f}%")
                        r4.metric("샤프 비율",  f"{p['sharpe']:.2f}")
                        r5, r6, r7, r8 = st.columns(4)
                        r5.metric("매매 횟수",  f"{p['trade_count']}회")
                        r6.metric("승률",      f"{p['win_rate']:.1f}%")
                        r7.metric("최종 자산",  f"{p['final_equity']:,.0f}원")
                        calmar = abs(p['cagr'] / p['mdd']) if p['mdd'] != 0 else 0
                        r8.metric("Calmar",    f"{calmar:.2f}")

                        # ── B&H 성과 비교 ─────────────────────────────
                        df_result_full = result.get("df")
                        equity_curve   = result.get("equity_curve")
                        df_bt_chart_bh = df_bt.loc[df_bt.index >= pd.Timestamp(str(bt_start))]

                        if not df_bt_chart_bh.empty:
                            bh_close = df_bt_chart_bh["close"]
                            bh_start_val = bh_close.iloc[0]
                            bh_end_val = bh_close.iloc[-1]
                            bh_total_ret = (bh_end_val / bh_start_val - 1) * 100
                            bh_days = (bh_close.index[-1] - bh_close.index[0]).days
                            bh_cagr = ((bh_end_val / bh_start_val) ** (365 / bh_days) - 1) * 100 if bh_days > 0 else 0
                            bh_peak = bh_close.cummax()
                            bh_dd_all = (bh_close - bh_peak) / bh_peak * 100
                            bh_mdd = bh_dd_all.min()
                            bh_daily_ret = bh_close.pct_change().dropna()
                            bh_sharpe = (bh_daily_ret.mean() / bh_daily_ret.std() * np.sqrt(365)) if bh_daily_ret.std() > 0 else 0
                            bh_calmar = abs(bh_cagr / bh_mdd) if bh_mdd != 0 else 0

                            st.subheader("전략 vs Buy & Hold")
                            _cmp_df = pd.DataFrame({
                                "": ["전략", "Buy & Hold"],
                                "총수익률(%)": [f"{p['total_return']:+.2f}", f"{bh_total_ret:+.2f}"],
                                "CAGR(%)": [f"{p['cagr']:+.2f}", f"{bh_cagr:+.2f}"],
                                "MDD(%)": [f"{p['mdd']:.2f}", f"{bh_mdd:.2f}"],
                                "Sharpe": [f"{p['sharpe']:.2f}", f"{bh_sharpe:.2f}"],
                                "Calmar": [f"{calmar:.2f}", f"{bh_calmar:.2f}"],
                            })
                            st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

                        # ── 연도별 성과 (상세 컬럼) ──────────────────
                        if df_result_full is not None and equity_curve is not None:
                            eq_series = pd.Series(equity_curve, index=df_result_full.index[-len(equity_curve):])
                            yearly_rows_g = []
                            years = eq_series.index.year.unique()
                            for yr in sorted(years):
                                yr_eq = eq_series[eq_series.index.year == yr]
                                if yr_eq.empty:
                                    continue
                                yr_start_eq = yr_eq.iloc[0]
                                yr_end_eq   = yr_eq.iloc[-1]
                                yr_ret      = (yr_end_eq / yr_start_eq - 1) * 100
                                peak_yr     = yr_eq.cummax()
                                yr_mdd      = ((yr_eq - peak_yr) / peak_yr * 100).min()
                                # B&H 연도별 비교
                                bh_yr_ret = 0.0
                                bh_yr_mdd = 0.0
                                if not df_bt_chart_bh.empty:
                                    bh_yr = bh_close[bh_close.index.year == yr]
                                    if not bh_yr.empty:
                                        bh_yr_ret = (bh_yr.iloc[-1] / bh_yr.iloc[0] - 1) * 100
                                        bh_yr_pk = bh_yr.cummax()
                                        bh_yr_mdd = ((bh_yr - bh_yr_pk) / bh_yr_pk * 100).min()
                                # 상태: 연도 마지막 일의 포지션
                                yr_trades = [t for t in p.get("trades", []) if pd.Timestamp(t["date"]).year <= yr]
                                last_type = yr_trades[-1]["type"] if yr_trades else "—"
                                yr_state = "보유" if last_type == "buy" else "현금"
                                yearly_rows_g.append({
                                    "연도": yr,
                                    "전략(%)": f"{yr_ret:+.2f}",
                                    "B&H(%)": f"{bh_yr_ret:+.2f}",
                                    "전략MDD": f"{yr_mdd:.2f}",
                                    "B&H MDD": f"{bh_yr_mdd:.2f}",
                                    "상태": yr_state,
                                })
                            if yearly_rows_g:
                                st.subheader("연도별 성과")
                                st.dataframe(pd.DataFrame(yearly_rows_g), use_container_width=True, hide_index=True)

                        # ── 에쿼티 커브 차트 ──────────────────────────
                        if equity_curve is not None and len(equity_curve) > 0:
                            df_eq = pd.DataFrame({"equity": equity_curve})
                            df_eq.index = df_result_full.index[-len(equity_curve):]
                            df_eq["return_pct"] = (df_eq["equity"] / bt_cap - 1) * 100
                            # Buy & Hold 비교
                            df_bt_chart = df_bt.loc[df_bt.index >= pd.Timestamp(str(bt_start))]
                            if not df_bt_chart.empty:
                                bh_base = df_bt_chart["close"].iloc[0]
                                bh_pct  = (df_bt_chart["close"] / bh_base - 1) * 100

                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                x=df_eq.index, y=df_eq["return_pct"], mode="lines",
                                name="전략", line=dict(color="gold", width=2)
                            ))
                            if not df_bt_chart.empty:
                                fig_eq.add_trace(go.Scatter(
                                    x=bh_pct.index, y=bh_pct.values, mode="lines",
                                    name="Buy & Hold", line=dict(color="gray", width=1, dash="dot")
                                ))
                            fig_eq.update_layout(
                                title=f"누적 수익률 ({bt_strategy} {bt_buy_p}/{bt_sell_p})",
                                yaxis_title="수익률 (%)", height=350,
                                margin=dict(l=0, r=0, t=70, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            st.plotly_chart(fig_eq, use_container_width=True)

                            # ── DD 차트 ──────────────────────────────
                            dd_idx = df_result_full.index[-len(equity_curve):]
                            strat_dd = df_result_full.loc[dd_idx, "drawdown"] if "drawdown" in df_result_full.columns else None
                            if strat_dd is not None:
                                fig_dd = go.Figure()
                                fig_dd.add_trace(go.Scatter(
                                    x=dd_idx, y=strat_dd.values, mode="lines",
                                    name="전략 DD", line=dict(color="crimson", width=2),
                                    fill="tozeroy", fillcolor="rgba(220,20,60,0.15)"
                                ))
                                if not df_bt_chart.empty:
                                    bh_peak = df_bt_chart["close"].cummax()
                                    bh_dd = (df_bt_chart["close"] - bh_peak) / bh_peak * 100
                                    fig_dd.add_trace(go.Scatter(
                                        x=bh_dd.index, y=bh_dd.values, mode="lines",
                                        name="Buy & Hold DD", line=dict(color="gray", width=1, dash="dot")
                                    ))
                                fig_dd.update_layout(
                                    title="Drawdown",
                                    yaxis_title="DD (%)", height=280,
                                    margin=dict(l=0, r=0, t=70, b=30),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                                )
                                fig_dd = _apply_dd_hover_format(fig_dd)
                                st.plotly_chart(fig_dd, use_container_width=True)

                        # ── 거래 내역 ─────────────────────────────────
                        with st.expander("거래 내역"):
                            trades_list = p.get("trades", [])
                            if trades_list:
                                st.dataframe(pd.DataFrame(trades_list), use_container_width=True, hide_index=True)
                            else:
                                st.info("실행된 거래가 없습니다.")

        # ── 서브탭2: 파라미터 최적화 ────────────────────────
        with gbt2:
            st.header("파라미터 최적화")

            # 데이터 가용 범위 표시
            _gold_info_opt = _dc_gold.gold_cache_info()
            if _gold_info_opt["exists"]:
                _gio_s = _gold_info_opt["start"]
                _gio_e = _gold_info_opt["end"]
                _gio_s_str = _gio_s.strftime('%Y-%m-%d') if hasattr(_gio_s, 'strftime') else str(_gio_s)[:10]
                _gio_e_str = _gio_e.strftime('%Y-%m-%d') if hasattr(_gio_e, 'strftime') else str(_gio_e)[:10]
                st.info(f"사용 가능 데이터: **{_gold_info_opt['rows']:,}**개 캔들 ({_gio_s_str} ~ {_gio_e_str})")
            else:
                st.warning("캐시된 Gold 데이터가 없습니다. 백테스트 탭에서 사전 다운로드를 실행하세요.")

            opt_strat_g = st.selectbox("전략", ["Donchian", "SMA"], key="gold_opt_strat")

            with st.form("gold_optimization_form"):
                gopt_method = st.radio("최적화 방법", ["Grid Search (전수 탐색)", "Optuna (베이지안 최적화)"], horizontal=True, key="gold_opt_method")
                use_optuna_g = "Optuna" in gopt_method

                if opt_strat_g == "Donchian":
                    st.markdown("##### 매수 채널 기간")
                    goc1, goc2, goc3 = st.columns(3)
                    g_buy_start = goc1.number_input("시작", 5, 300, 30,  key="gold_opt_buy_start")
                    g_buy_end   = goc2.number_input("끝",   5, 300, 150, key="gold_opt_buy_end")
                    g_buy_step  = goc3.number_input("간격", 1, 50,  5,   key="gold_opt_buy_step")
                    st.markdown("##### 매도 채널 기간")
                    goc4, goc5, goc6 = st.columns(3)
                    g_sell_start = goc4.number_input("시작", 5, 300, 10, key="gold_opt_sell_start")
                    g_sell_end   = goc5.number_input("끝",   5, 300, 80, key="gold_opt_sell_end")
                    g_sell_step  = goc6.number_input("간격", 1, 50,  5,  key="gold_opt_sell_step")
                    g_sell_mode_label = st.radio("매도 방식", ["하단선 (Lower)", "중심선 (Midline)", "두 방법 비교"], horizontal=True, key="gold_opt_sell_mode")
                else:
                    st.markdown("##### SMA 기간")
                    goc1, goc2, goc3 = st.columns(3)
                    g_sma_start = goc1.number_input("시작", 5, 300, 10, key="gold_opt_sma_start")
                    g_sma_end   = goc2.number_input("끝",   5, 300, 100, key="gold_opt_sma_end")
                    g_sma_step  = goc3.number_input("간격", 1, 50,  5,   key="gold_opt_sma_step")

                if use_optuna_g:
                    st.divider()
                    st.markdown("##### Optuna 설정")
                    uoc1, uoc2 = st.columns(2)
                    g_n_trials  = uoc1.number_input("탐색 횟수", 50, 2000, 200, step=50, key="gold_optuna_trials")
                    g_obj_label = uoc2.selectbox("목적함수", ["Calmar (CAGR/|MDD|)", "Sharpe", "수익률 (Return)", "MDD 최소"], key="gold_optuna_obj")

                st.divider()
                g_opt_start = st.date_input("시작일", value=gold_start_date, key="gold_opt_start")
                _gfee_col, _gslip_col = st.columns(2)
                g_opt_fee   = _gfee_col.number_input("수수료 (%)", value=0.3, format="%.2f", key="gold_opt_fee") / 100
                g_opt_slip  = _gslip_col.number_input("슬리피지 (%)", value=0.0, min_value=0.0, max_value=2.0, step=0.05, format="%.2f", key="gold_opt_slip")
                g_opt_cap   = st.number_input("초기 자본 (원)", value=gold_initial_cap, step=100_000, format="%d", key="gold_opt_cap")
                _gsort_col, _gtrade_col = st.columns(2)
                g_sort_label = _gsort_col.selectbox("정렬 기준", ["Calmar", "Sharpe", "CAGR", "MDD 최소", "수익률"], key="gold_opt_sort")
                g_min_trades = _gtrade_col.number_input("최소 매매 횟수", value=0, min_value=0, max_value=100, step=1, key="gold_opt_min_trades")

                gopt_submitted = st.form_submit_button("최적화 시작", type="primary")

            if gopt_submitted:
                import plotly.express as px
                df_opt_src = load_gold_data(max(
                    g_buy_end if opt_strat_g == "Donchian" else g_sma_end, 300
                ))
                if df_opt_src is None or df_opt_src.empty:
                    st.error("데이터 로드 실패. 백테스트 탭에서 사전 다운로드를 실행하세요.")
                else:
                    st.caption(f"조회된 캔들: {len(df_opt_src):,}개 ({df_opt_src.index[0].strftime('%Y-%m-%d')} ~ {df_opt_src.index[-1].strftime('%Y-%m-%d')})")
                    with st.status("최적화 진행 중...", expanded=True) as gopt_status:
                        prog_bar_g  = st.progress(0)
                        log_area_g  = st.empty()

                        def g_opt_progress(idx, total, msg):
                            pct = min(idx / total, 1.0) if total > 0 else 0
                            prog_bar_g.progress(pct)
                            log_area_g.text(f"{msg} ({idx}/{total})")

                        # sell_mode 결정
                        _g_sell_modes = ["lower"]
                        if opt_strat_g == "Donchian":
                            if g_sell_mode_label == "중심선 (Midline)":
                                _g_sell_modes = ["midline"]
                            elif g_sell_mode_label == "두 방법 비교":
                                _g_sell_modes = ["lower", "midline"]

                        engine_g = BacktestEngine()
                        gopt_results = []
                        try:
                            for _gsm in _g_sell_modes:
                                if use_optuna_g:
                                    obj_map_g = {"Calmar (CAGR/|MDD|)": "calmar", "Sharpe": "sharpe", "수익률 (Return)": "return", "MDD 최소": "mdd"}
                                    obj_key_g = obj_map_g.get(g_obj_label, "calmar")
                                    if opt_strat_g == "Donchian":
                                        opt_res_g = engine_g.optuna_optimize(
                                            df_opt_src, strategy_mode="Donchian",
                                            buy_range=(g_buy_start, g_buy_end),
                                            sell_range=(g_sell_start, g_sell_end),
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            n_trials=g_n_trials,
                                            objective_metric=obj_key_g,
                                            progress_callback=g_opt_progress,
                                            sell_mode=_gsm,
                                        )
                                    else:
                                        opt_res_g = engine_g.optuna_optimize(
                                            df_opt_src, strategy_mode="SMA",
                                            buy_range=(g_sma_start, g_sma_end),
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            n_trials=g_n_trials,
                                            objective_metric=obj_key_g,
                                            progress_callback=g_opt_progress,
                                        )
                                    _trials = opt_res_g.get("trials", [])
                                    for _t in _trials:
                                        _t["sell_mode"] = _gsm
                                    gopt_results.extend(_trials)
                                    best_params  = opt_res_g.get("best_params", {})
                                    gopt_status.update(label=f"Optuna 완료! ({_gsm})", state="complete")
                                    st.success(f"[{_gsm}] 최적 파라미터: {best_params} | 목적함수 값: {opt_res_g['best_value']:.2f}")
                                else:
                                    if opt_strat_g == "Donchian":
                                        buy_r  = range(g_buy_start,  g_buy_end  + 1, g_buy_step)
                                        sell_r = range(g_sell_start, g_sell_end + 1, g_sell_step)
                                        _grid_res = engine_g.optimize_donchian(
                                            df_opt_src, buy_range=buy_r, sell_range=sell_r,
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            progress_callback=g_opt_progress,
                                            sell_mode=_gsm,
                                        )
                                    else:
                                        sma_r = range(g_sma_start, g_sma_end + 1, g_sma_step)
                                        _grid_res = engine_g.optimize_sma(
                                            df_opt_src, sma_range=sma_r,
                                            fee=g_opt_fee, slippage=g_opt_slip,
                                            start_date=str(g_opt_start),
                                            initial_balance=g_opt_cap,
                                            progress_callback=g_opt_progress,
                                        )
                                    for _t in _grid_res:
                                        _t["sell_mode"] = _gsm
                                    gopt_results.extend(_grid_res)
                                    gopt_status.update(label=f"Grid Search 완료! ({_gsm})", state="complete")

                        except Exception as e:
                            gopt_status.update(label=f"오류: {e}", state="error")

                    if gopt_results:
                        df_opt_res = pd.DataFrame(gopt_results)
                        df_opt_res["calmar"] = df_opt_res.apply(
                            lambda r: abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0, axis=1
                        )

                        # 최소 매매 횟수 필터
                        if g_min_trades > 0:
                            df_opt_res = df_opt_res[df_opt_res["trade_count"] >= g_min_trades]

                        # 정렬 기준 적용
                        _sort_map = {"Calmar": "calmar", "Sharpe": "sharpe", "CAGR": "cagr", "MDD 최소": "mdd", "수익률": "total_return"}
                        _sort_col = _sort_map.get(g_sort_label, "calmar")
                        _sort_asc = True if _sort_col == "mdd" else False
                        df_opt_res_sorted = df_opt_res.sort_values(_sort_col, ascending=_sort_asc)

                        st.subheader(f"상위 20개 파라미터 (정렬: {g_sort_label})")
                        _base_cols = ["Buy Period", "Sell Period"] if opt_strat_g == "Donchian" else ["sma_period"]
                        if len(_g_sell_modes) > 1:
                            _base_cols.append("sell_mode")
                        disp_cols = _base_cols + ["total_return", "cagr", "mdd", "sharpe", "calmar", "win_rate", "trade_count"]
                        disp_cols = [c for c in disp_cols if c in df_opt_res_sorted.columns]
                        st.dataframe(
                            df_opt_res_sorted[disp_cols].head(20).style.format({
                                "total_return": "{:.2f}%", "cagr": "{:.2f}%",
                                "mdd": "{:.2f}%", "sharpe": "{:.2f}",
                                "calmar": "{:.2f}", "win_rate": "{:.1f}%",
                            }),
                            use_container_width=True, hide_index=True
                        )

                        # 히트맵 (Donchian Grid Search일 때)
                        if opt_strat_g == "Donchian" and not use_optuna_g and \
                                "Buy Period" in df_opt_res.columns and "Sell Period" in df_opt_res.columns:
                            st.subheader("Calmar 히트맵 (Buy × Sell)")
                            try:
                                df_heat = df_opt_res.pivot_table(index="Buy Period", columns="Sell Period", values="calmar")
                                fig_heat = px.imshow(
                                    df_heat, color_continuous_scale="RdYlGn",
                                    labels=dict(x="Sell Period", y="Buy Period", color="Calmar"),
                                    aspect="auto"
                                )
                                fig_heat.update_layout(height=500, margin=dict(l=0, r=0, t=70, b=30))
                                st.plotly_chart(fig_heat, use_container_width=True)
                            except Exception:
                                pass

                        # 전체 결과 다운로드
                        csv_data_g = df_opt_res_sorted.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 전체 결과 다운로드 (CSV)", data=csv_data_g,
                                           file_name="gold_optimization_results.csv", mime="text/csv")

                        # ── 파라미터 선택 백테스트 ─────────────────────
                        st.divider()
                        st.subheader("파라미터 백테스트")

                        # 상위 결과 목록 생성 (최대 20개)
                        _top_n = min(20, len(df_opt_res_sorted))
                        _sel_options = []
                        for _ri in range(_top_n):
                            _r = df_opt_res_sorted.iloc[_ri]
                            if opt_strat_g == "Donchian":
                                _lbl = f"#{_ri+1}  Donchian {int(_r['Buy Period'])}/{int(_r['Sell Period'])}"
                            else:
                                _sp = int(_r.get("sma_period", _r.get("SMA Period", 20)))
                                _lbl = f"#{_ri+1}  SMA {_sp}"
                            _lbl += f"  |  Calmar {_r['calmar']:.2f}  |  CAGR {_r['cagr']:.2f}%  |  MDD {_r['mdd']:.2f}%"
                            _sel_options.append(_lbl)

                        _sel_idx = st.selectbox("백테스트할 파라미터 선택", range(_top_n),
                                                format_func=lambda x: _sel_options[x], index=0,
                                                key="gold_opt_bt_select")

                        best_row = df_opt_res_sorted.iloc[_sel_idx]
                        if opt_strat_g == "Donchian":
                            best_buy_p = int(best_row["Buy Period"])
                            best_sell_p = int(best_row["Sell Period"])
                            st.info(f"선택 파라미터: **Donchian {best_buy_p}/{best_sell_p}**  |  Calmar {best_row['calmar']:.2f}  |  CAGR {best_row['cagr']:.2f}%  |  MDD {best_row['mdd']:.2f}%")
                        else:
                            best_buy_p = int(best_row.get("sma_period", best_row.get("SMA Period", 20)))
                            best_sell_p = 0
                            st.info(f"선택 파라미터: **SMA {best_buy_p}**  |  Calmar {best_row['calmar']:.2f}  |  CAGR {best_row['cagr']:.2f}%  |  MDD {best_row['mdd']:.2f}%")

                        try:
                            _sell_ratio = (best_sell_p / best_buy_p) if opt_strat_g == "Donchian" and best_buy_p > 0 else 1.0
                            _strat_mode = "Donchian" if opt_strat_g == "Donchian" else "SMA Strategy"
                            best_result = engine_g.run_backtest(
                                ticker=None, df=df_opt_src,
                                period=best_buy_p,
                                interval="day",
                                fee=g_opt_fee,
                                start_date=str(g_opt_start),
                                initial_balance=g_opt_cap,
                                strategy_mode=_strat_mode,
                                sell_period_ratio=_sell_ratio,
                                slippage=0.0,
                            )
                            if "error" not in best_result:
                                bp = best_result["performance"]
                                bc1, bc2, bc3, bc4 = st.columns(4)
                                bc1.metric("총 수익률", f"{bp['total_return']:+.2f}%")
                                bc2.metric("CAGR", f"{bp['cagr']:+.2f}%")
                                bc3.metric("MDD", f"{bp['mdd']:.2f}%")
                                bc4.metric("샤프 비율", f"{bp['sharpe']:.2f}")
                                bc5, bc6, bc7, bc8 = st.columns(4)
                                bc5.metric("매매 횟수", f"{bp['trade_count']}회")
                                bc6.metric("승률", f"{bp['win_rate']:.1f}%")
                                bc7.metric("최종 자산", f"{bp['final_equity']:,.0f}원")
                                _bcalmar = abs(bp['cagr'] / bp['mdd']) if bp['mdd'] != 0 else 0
                                bc8.metric("Calmar", f"{_bcalmar:.2f}")

                                # 에쿼티 + DD 차트
                                _best_df = best_result.get("df")
                                _best_eq = best_result.get("equity_curve")
                                if _best_df is not None and _best_eq is not None and len(_best_eq) > 0:
                                    _beq_idx = _best_df.index[-len(_best_eq):]
                                    _beq_ret = (np.array(_best_eq) / g_opt_cap - 1) * 100

                                    # Buy & Hold 비교
                                    _bh_chart = df_opt_src.loc[df_opt_src.index >= pd.Timestamp(str(g_opt_start))]
                                    _bh_base = _bh_chart["close"].iloc[0] if not _bh_chart.empty else 1
                                    _bh_ret = ((_bh_chart["close"] / _bh_base - 1) * 100) if not _bh_chart.empty else pd.Series()

                                    # 에쿼티(수익률) 차트
                                    _title_lbl = f"Donchian {best_buy_p}/{best_sell_p}" if opt_strat_g == "Donchian" else f"SMA {best_buy_p}"
                                    fig_best_eq = go.Figure()
                                    fig_best_eq.add_trace(go.Scatter(
                                        x=_beq_idx, y=_beq_ret, mode="lines",
                                        name="전략", line=dict(color="gold", width=2)
                                    ))
                                    if not _bh_chart.empty:
                                        fig_best_eq.add_trace(go.Scatter(
                                            x=_bh_ret.index, y=_bh_ret.values, mode="lines",
                                            name="Buy & Hold", line=dict(color="gray", width=1, dash="dot")
                                        ))
                                    fig_best_eq.update_layout(
                                        title=f"누적 수익률 ({_title_lbl})",
                                        yaxis_title="수익률 (%)", height=350,
                                        margin=dict(l=0, r=0, t=70, b=30),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                                    )
                                    st.plotly_chart(fig_best_eq, use_container_width=True)

                                    # DD 차트
                                    _best_dd = _best_df.loc[_beq_idx, "drawdown"] if "drawdown" in _best_df.columns else None
                                    if _best_dd is not None:
                                        fig_best_dd = go.Figure()
                                        fig_best_dd.add_trace(go.Scatter(
                                            x=_beq_idx, y=_best_dd.values, mode="lines",
                                            name="전략 DD", line=dict(color="crimson", width=2),
                                            fill="tozeroy", fillcolor="rgba(220,20,60,0.15)"
                                        ))
                                        if not _bh_chart.empty:
                                            _bh_peak = _bh_chart["close"].cummax()
                                            _bh_dd = (_bh_chart["close"] - _bh_peak) / _bh_peak * 100
                                            fig_best_dd.add_trace(go.Scatter(
                                                x=_bh_dd.index, y=_bh_dd.values, mode="lines",
                                                name="Buy & Hold DD", line=dict(color="gray", width=1, dash="dot")
                                            ))
                                        fig_best_dd.update_layout(
                                            title="Drawdown",
                                            yaxis_title="DD (%)", height=280,
                                            margin=dict(l=0, r=0, t=70, b=30),
                                            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                                        )
                                        fig_best_dd = _apply_dd_hover_format(fig_best_dd)
                                        st.plotly_chart(fig_best_dd, use_container_width=True)
                        except Exception as e:
                            st.warning(f"최적 파라미터 백테스트 오류: {e}")

    # ══════════════════════════════════════════════════════
    # Tab 4: 수수료/세금 (기존 내용 유지)
    # ══════════════════════════════════════════════════════
    with tab_g4:
        st.header("KRX 금현물 수수료 및 세금 안내")
        st.caption("키움증권 기준 | 수수료는 변경될 수 있으니 공식 홈페이지에서 최신 정보를 확인하세요.")

        st.subheader("1. 거래 수수료")
        fee_data = pd.DataFrame([
            {"구분": "매매 수수료", "요율/금액": "약 0.165% (온라인)", "비고": "매수/매도 각각 부과, 부가세 포함"},
            {"구분": "유관기관 수수료", "요율/금액": "0.0046396%", "비고": "거래소/예탁원 등 (매매금액 기준)"},
            {"구분": "합계 (편도)", "요율/금액": "약 0.17%", "비고": "매수 또는 매도 1회당"},
            {"구분": "합계 (왕복)", "요율/금액": "약 0.34%", "비고": "매수 + 매도 1세트"},
        ])
        st.dataframe(fee_data, use_container_width=True, hide_index=True)

        st.subheader("2. 세금")
        tax_data = pd.DataFrame([
            {"구분": "양도소득세", "세율": "비과세", "비고": "KRX 금시장 매매차익은 양도세 면제"},
            {"구분": "배당소득세", "세율": "15.4%", "비고": "보관료 환급금(이자) 발생 시"},
            {"구분": "부가가치세 (매매)", "세율": "면세", "비고": "KRX 장내 거래 시 부가세 없음"},
            {"구분": "부가가치세 (실물 인출)", "세율": "10%", "비고": "평균 매수단가 x 인출 수량 기준"},
        ])
        st.dataframe(tax_data, use_container_width=True, hide_index=True)
        st.info("KRX 금시장의 최대 장점: **매매차익 비과세 + 부가세 면세** (실물 인출 시에만 부가세 10% 부과)")

        st.subheader("3. 보관료")
        st.markdown(
            "| 항목 | 내용 |\n"
            "|------|------|\n"
            "| **보관료율** | 매일 잔량의 시가 환산 금액 x **0.02% (연율)** |\n"
            "| **일할 계산** | 시가 x 보유수량 x 0.0002% / 365일 |\n"
            "| **부가세** | 별도 (보관료의 10%) |\n"
            "| **부과 주기** | 매월 말 정산 |\n"
            "\n"
            "> 예시: 금 100g 보유, 시가 13만원/g → 연간 보관료 약 **2,600원** (부가세 별도)\n"
        )

        st.subheader("4. 금 투자 방법별 비교")
        compare_data = pd.DataFrame([
            {"투자 방법": "KRX 금현물", "매매차익 세금": "비과세", "부가세": "면세 (인출시 10%)", "거래 수수료": "~0.17%", "실물 인출": "가능 (100g/1kg)"},
            {"투자 방법": "금 ETF", "매매차익 세금": "15.4%", "부가세": "해당없음", "거래 수수료": "~0.015%+보수", "실물 인출": "불가"},
            {"투자 방법": "골드뱅킹 (은행)", "매매차익 세금": "15.4%", "부가세": "매입시 면세, 인출시 10%", "거래 수수료": "~1%", "실물 인출": "가능"},
            {"투자 방법": "금 실물 (귀금속점)", "매매차익 세금": "비과세", "부가세": "10% (매입 시)", "거래 수수료": "5~15% (스프레드)", "실물 인출": "즉시"},
        ])
        st.dataframe(compare_data, use_container_width=True, hide_index=True)
        st.caption("출처: 키움증권 금현물 수수료 안내, KRX 금시장 안내서 | 수수료는 변경될 수 있습니다.")





def render_kis_isa_mode():
    """KIS ISA 위대리(WDR) 전략 모드 - 4탭 구성."""
    from kis_trader import KISTrader
    from strategy.widaeri import WDRStrategy
    import data_cache as _dc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd

    st.title("ISA 위대리(WDR) 전략")

    # ── 사이드바 설정 ──
    st.sidebar.header("ISA 설정")
    kis_ak = _get_runtime_value(("KIS_ISA_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_ISA_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(config.get("kis_isa_account_no", "") or _get_runtime_value(("KIS_ISA_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(config.get("kis_isa_prdt_cd", "") or _get_runtime_value(("KIS_ISA_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))

    # 계좌번호 10자리 → 앞8(CANO) + 뒤2(상품코드) 자동 분리
    _raw_acct = kis_acct.replace("-", "").strip()
    if len(_raw_acct) == 10 and kis_prdt in ("01", ""):
        kis_acct = _raw_acct[:8]
        kis_prdt = _raw_acct[8:]

    if not IS_CLOUD:
        with st.sidebar.expander("KIS API 키", expanded=False):
            kis_ak = st.text_input("앱 키", value=kis_ak, type="password", key="isa_app_key")
            kis_sk = st.text_input("시크릿 키", value=kis_sk, type="password", key="isa_app_secret")
            kis_acct = st.text_input("계좌번호 (앞 8자리)", value=kis_acct, key="isa_account_no", help="10자리 입력 시 자동으로 앞8+뒤2 분리")
            kis_prdt = st.text_input("상품코드 (뒤 2자리)", value=kis_prdt, key="isa_prdt_cd")
            _raw2 = kis_acct.replace("-", "").strip()
            if len(_raw2) == 10:
                kis_acct = _raw2[:8]
                kis_prdt = _raw2[8:]

    # ETF 선택: 매매 ETF / TREND ETF(시그널용)
    def _build_etf_options(code_list):
        out = {}
        for _code in code_list:
            _c = str(_code).strip()
            if _c:
                out[f"{_c} {_etf_name_kr(_c)}"] = _c
        return out

    _isa_trade_options = _build_etf_options(["418660", "409820", "423920", "465610", "461910", "133690"])
    _isa_trend_options = _build_etf_options(["133690", "360750", "453850", "251350", "418660", "409820", "423920", "465610", "461910"])

    _saved_trade_etf = str(config.get("kis_isa_etf_code", _get_runtime_value("KIS_ISA_ETF_CODE", "418660")))
    _saved_trend_etf = str(config.get("kis_isa_trend_etf_code", _get_runtime_value("KIS_ISA_TREND_ETF_CODE", "133690")))

    if _saved_trade_etf and _saved_trade_etf not in _isa_trade_options.values():
        _isa_trade_options[f"{_saved_trade_etf} {_etf_name_kr(_saved_trade_etf)}"] = _saved_trade_etf
    if _saved_trend_etf and _saved_trend_etf not in _isa_trend_options.values():
        _isa_trend_options[f"{_saved_trend_etf} {_etf_name_kr(_saved_trend_etf)}"] = _saved_trend_etf

    _trade_default_label = next((k for k, v in _isa_trade_options.items() if v == _saved_trade_etf), list(_isa_trade_options.keys())[0])
    _trend_default_label = next((k for k, v in _isa_trend_options.items() if v == _saved_trend_etf), list(_isa_trend_options.keys())[0])

    selected_trend_etf_label = st.sidebar.selectbox(
        "TREND ETF (시그널)",
        list(_isa_trend_options.keys()),
        index=list(_isa_trend_options.keys()).index(_trend_default_label),
        key="isa_trend_etf_select",
        disabled=IS_CLOUD,
    )
    selected_etf_label = st.sidebar.selectbox(
        "매매 ETF",
        list(_isa_trade_options.keys()),
        index=list(_isa_trade_options.keys()).index(_trade_default_label),
        key="isa_etf_select",
        disabled=IS_CLOUD,
    )
    isa_etf_code = _isa_trade_options[selected_etf_label]
    isa_trend_etf_code = _isa_trend_options[selected_trend_etf_label]

    wdr_eval_mode = st.sidebar.selectbox(
        "평가 시스템", [3, 5], index=[3, 5].index(int(config.get("kis_isa_wdr_mode", 5))),
        format_func=lambda x: f"{x}단계", key="isa_wdr_mode", disabled=IS_CLOUD,
    )
    wdr_ov = st.sidebar.number_input(
        "고평가 임계값 (%)", min_value=0.0, max_value=30.0,
        value=float(config.get("kis_isa_wdr_ov", 5.0)), step=0.5,
        key="isa_wdr_ov", disabled=IS_CLOUD,
    )
    wdr_un = st.sidebar.number_input(
        "저평가 임계값 (%)", min_value=-30.0, max_value=0.0,
        value=float(config.get("kis_isa_wdr_un", -6.0)), step=0.5,
        key="isa_wdr_un", disabled=IS_CLOUD,
    )
    _isa_start_default = config.get("kis_isa_start_date", "2020-01-01")
    isa_start_date = st.sidebar.date_input(
        "시작일",
        value=pd.to_datetime(_isa_start_default).date(),
        key="isa_start_date",
        disabled=IS_CLOUD,
    )

    if not IS_CLOUD and st.sidebar.button("ISA 설정 저장", key="isa_save_cfg"):
        new_cfg = config.copy()
        new_cfg["kis_isa_account_no"] = str(kis_acct).strip()
        new_cfg["kis_isa_prdt_cd"] = str(kis_prdt).strip() or "01"
        new_cfg["kis_isa_etf_code"] = isa_etf_code
        new_cfg["kis_isa_trend_etf_code"] = isa_trend_etf_code
        new_cfg["kis_isa_wdr_mode"] = int(wdr_eval_mode)
        new_cfg["kis_isa_wdr_ov"] = float(wdr_ov)
        new_cfg["kis_isa_wdr_un"] = float(wdr_un)
        new_cfg["kis_isa_start_date"] = str(isa_start_date)
        save_config(new_cfg)
        st.sidebar.success("ISA 설정을 저장했습니다.")

    if not (kis_ak and kis_sk and kis_acct):
        st.warning("KIS ISA API 키와 계좌번호를 설정해 주세요.")
        return

    trader = KISTrader(is_mock=False)
    trader.app_key = kis_ak
    trader.app_secret = kis_sk
    trader.account_no = kis_acct
    trader.acnt_prdt_cd = kis_prdt

    # 토큰 캐싱 — ISA/연금저축 공용 토큰 재사용 (KIS 1분 1회 발급 제한 대응)
    _isa_token_key = f"isa_token_{kis_acct}"
    _kis_shared_token_key = f"kis_token_shared_{str(kis_ak or '')[-8:]}"
    _cached_isa = st.session_state.get(_isa_token_key)
    _cached_shared = st.session_state.get(_kis_shared_token_key)
    _tok = None
    if _cached_isa and (_cached_isa.get("expiry", 0) - time.time()) > 300:
        _tok = _cached_isa
    elif _cached_shared and (_cached_shared.get("expiry", 0) - time.time()) > 300:
        _tok = _cached_shared

    if _tok:
        trader.access_token = _tok.get("token")
        trader.token_expiry = float(_tok.get("expiry", 0))
        st.session_state[_isa_token_key] = {"token": trader.access_token, "expiry": trader.token_expiry}
    else:
        if not trader.auth():
            st.error("KIS 인증에 실패했습니다. API 키/계좌를 확인해 주세요.")
            return
        _new_tok = {"token": trader.access_token, "expiry": trader.token_expiry}
        st.session_state[_isa_token_key] = _new_tok
        st.session_state[_kis_shared_token_key] = _new_tok

    isa_bal_key = f"isa_balance_cache_{kis_acct}_{kis_prdt}"

    # ISA 데이터 로딩 정책: 로컬 파일 우선 + 부족 시 API 보강
    _isa_local_first_raw = str(config.get("isa_local_first", "1")).strip().lower()
    _isa_local_first = _isa_local_first_raw not in ("0", "false", "no", "off")
    _isa_api_fallback_raw = str(config.get("isa_api_fallback", "1")).strip().lower()
    _isa_api_fallback = _isa_api_fallback_raw not in ("0", "false", "no", "off")

    _isa_px_cache_key = f"isa_price_cache_{kis_acct}_{kis_prdt}"
    _isa_px_cache = st.session_state.get(_isa_px_cache_key)
    if not isinstance(_isa_px_cache, dict):
        _isa_px_cache = {}
        st.session_state[_isa_px_cache_key] = _isa_px_cache

    # ISA 진입 시(QQQ/TQQQ 사용 화면) 미국 일봉 CSV를 세션당 1회 최신 거래일까지 동기화
    _isa_us_sync_key = f"isa_us_wtr_yf_synced_{pd.Timestamp.now().strftime('%Y%m%d')}"
    if not st.session_state.get(_isa_us_sync_key):
        try:
            _dc.fetch_and_cache_yf("QQQ", start="1999-03-10", force_refresh=True)
            _dc.fetch_and_cache_yf("TQQQ", start="2010-02-12", force_refresh=True)
        except Exception as _e:
            logging.warning(f"ISA 미국 데이터 자동 동기화 실패: {_e}")
        finally:
            st.session_state[_isa_us_sync_key] = True

    def _get_isa_daily_chart(_code: str, count: int = 120, end_date: str | None = None):
        _c = _code_only(_code)
        if not _c:
            return None
        if _isa_local_first:
            _df = _dc.get_kis_domestic_local_first(
                trader,
                _c,
                count=int(max(1, count)),
                end_date=end_date,
                allow_api_fallback=bool(_isa_api_fallback),
            )
        else:
            try:
                _df = trader.get_daily_chart(_c, count=int(max(1, count)), end_date=end_date or None)
            except Exception:
                _df = None
        if _df is None or _df.empty:
            return _df
        _df = _df.copy().sort_index()
        if "close" not in _df.columns and "Close" in _df.columns:
            _df = _df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        return _df

    def _get_isa_current_price(_code: str, ttl_sec: float = 10.0) -> float:
        _c = _code_only(_code)
        if not _c:
            return 0.0
        _now = float(time.time())
        _hit = _isa_px_cache.get(_c)
        if isinstance(_hit, dict) and (_now - float(_hit.get("ts", 0.0))) <= float(ttl_sec):
            return float(_hit.get("val", 0.0) or 0.0)

        _p = 0.0
        _df = _get_isa_daily_chart(_c, count=3)
        if _df is not None and not _df.empty:
            if "close" in _df.columns:
                _p = float(_df["close"].iloc[-1] or 0.0)
            elif "Close" in _df.columns:
                _p = float(_df["Close"].iloc[-1] or 0.0)
        if _p <= 0 and not _isa_local_first:
            try:
                _p = float(trader.get_current_price(_c) or 0.0)
            except Exception:
                _p = 0.0
        if _p <= 0 and _isa_api_fallback:
            try:
                _p = float(trader.get_current_price(_c) or 0.0)
            except Exception:
                _p = 0.0
        _isa_px_cache[_c] = {"ts": _now, "val": float(_p if _p > 0 else 0.0)}
        return float(_p if _p > 0 else 0.0)

    tab_i1, tab_i2, tab_i3, tab_i4, tab_i5, tab_i6 = st.tabs([
        "🚀 실시간 모니터링", "🛒 수동 주문", "📋 주문방식", "💳 수수료/세금",
        "📊 미국 위대리 백테스트", "🔧 위대리 최적화"
    ])

    # ══════════════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링 (잔고 + WDR 시그널)
    # ══════════════════════════════════════════════════════════════
    with tab_i1:
        st.header("WDR 시그널 모니터링")
        st.caption(f"매매 ETF: {_fmt_etf_code_name(isa_etf_code)} | TREND ETF: {_fmt_etf_code_name(isa_trend_etf_code)}")

        # 잔고 표시 — F5 새로고침 시 자동 조회, 이후 캐시 사용
        if isa_bal_key not in st.session_state:
            with st.spinner("ISA 잔고를 조회하는 중..."):
                st.session_state[isa_bal_key] = trader.get_balance()

        rcol1, rcol2 = st.columns([1, 5])
        with rcol1:
            if st.button("잔고 새로고침", key="isa_refresh_balance"):
                with st.spinner("ISA 잔고를 다시 조회하는 중..."):
                    st.session_state[isa_bal_key] = trader.get_balance()
                st.session_state.pop("isa_signal_result", None)
                st.session_state.pop("isa_signal_params", None)
                st.rerun()

        bal = st.session_state.get(isa_bal_key)
        if not bal:
            st.warning("잔고 조회에 실패했습니다. (응답 None — 네트워크 또는 인증 오류)")
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt}")
        elif bal.get("error"):
            st.error(f"잔고 조회 API 오류: [{bal.get('msg_cd', '')}] {bal.get('msg1', '알 수 없는 오류')}")
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt} / rt_cd: {bal.get('rt_cd', '')}")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []
            total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
            stock_eval = total_eval - cash

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("예수금", f"{cash:,.0f} KRW")
            m2.metric("주식 평가", f"{stock_eval:,.0f} KRW")
            m3.metric("총 평가", f"{total_eval:,.0f} KRW")
            m4.metric("보유 종목 수", f"{len(holdings)}")

            if holdings:
                df_h = pd.DataFrame(holdings)
                cols = [c for c in ["code", "name", "qty", "avg_price", "cur_price", "eval_amt", "pnl_rate"] if c in df_h.columns]
                st.dataframe(df_h[cols], use_container_width=True, hide_index=True)

        st.divider()

        # WDR 시그널 자동 계산 (버튼 없이 항상 표시)
        isa_sig_params = {
            "trade_etf": str(isa_etf_code),
            "trend_etf": str(isa_trend_etf_code),
            "ov": float(wdr_ov),
            "un": float(wdr_un),
            "acct": str(kis_acct),
            "prdt": str(kis_prdt),
        }

        def _compute_isa_signal_result():
            # 추세선 정확도를 위해 상장일부터 전체 데이터 로드 (최대 5000일)
            sig_df = _get_isa_daily_chart(str(isa_trend_etf_code), count=5000)
            if sig_df is None or len(sig_df) < 260:
                return {"error": f"시그널 데이터가 부족합니다. ({isa_trend_etf_code})"}

            strategy = WDRStrategy(settings={
                "overvalue_threshold": float(wdr_ov),
                "undervalue_threshold": float(wdr_un),
            }, evaluation_mode=int(wdr_eval_mode))
            signal = strategy.analyze(sig_df)
            if not signal:
                return {"error": "WDR 분석에 실패했습니다."}

            bal_local = trader.get_balance() or {"cash": 0.0, "holdings": []}
            cash_local = float(bal_local.get("cash", 0.0))
            holdings_local = bal_local.get("holdings", []) or []
            etf_holding = next((h for h in holdings_local if str(h.get("code", "")) == str(isa_etf_code)), None)
            qty = int(etf_holding.get("qty", 0)) if etf_holding else 0
            cur = _get_isa_current_price(str(isa_etf_code)) or 0.0

            weekly_pnl = 0.0
            ch = _get_isa_daily_chart(str(isa_etf_code), count=10)
            if ch is not None and len(ch) >= 5 and qty > 0 and cur > 0:
                p5 = float(ch["close"].iloc[-5])
                weekly_pnl = (cur - p5) * qty

            action = strategy.get_rebalance_action(
                weekly_pnl=weekly_pnl,
                divergence=float(signal["divergence"]),
                current_shares=qty,
                current_price=float(cur) if cur else 1.0,
                cash=cash_local,
            )

            # 백테스트 실행 (시작일 ~ 현재, 전체 데이터)
            bt_trade_df = _get_isa_daily_chart(str(isa_etf_code), count=5000)
            bt_res = None
            if sig_df is not None and bt_trade_df is not None:
                bt_res = strategy.run_backtest(
                    signal_daily_df=sig_df,
                    trade_daily_df=bt_trade_df,
                    initial_balance=10_000_000, # 기준 1000만원
                    start_date=str(isa_start_date)
                )

            weekly = strategy.daily_to_weekly(sig_df)
            trend = strategy.calc_growth_trend(weekly)

            return {
                "signal": signal,
                "action": action,
                "weekly_df": weekly,
                "trend": trend,
                "balance": bal_local,
                "bt_res": bt_res,
                "cur_price": cur,
            }

        if st.session_state.get("isa_signal_result") is None or st.session_state.get("isa_signal_params") != isa_sig_params:
            with st.spinner("WDR 시그널 및 백테스트 계산 중..."):
                st.session_state["isa_signal_result"] = _compute_isa_signal_result()
                st.session_state["isa_signal_params"] = isa_sig_params
                if isinstance(st.session_state["isa_signal_result"], dict) and st.session_state["isa_signal_result"].get("balance"):
                    st.session_state[isa_bal_key] = st.session_state["isa_signal_result"]["balance"]

        res = st.session_state.get("isa_signal_result")
        if res:
            if res.get("error"):
                st.error(res["error"])
            else:
                sig = res["signal"]
                act = res["action"]
                bt = res.get("bt_res")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("이격도", f"{float(sig['divergence']):+.2f}%")
                c2.metric("시장 상태", str(sig["state"]))
                c3.metric("매도 비율", f"{float(sig['sell_ratio']):.1f}%")
                c4.metric("매수 비율", f"{float(sig['buy_ratio']):.1f}%")

                # ── 백테스트 자산현황 vs 실제 계좌 비교 ──
                _bal_valid = bal and not bal.get("error")
                actual_cash_v = float(bal.get("cash", 0.0)) if _bal_valid else 0.0
                actual_hlds = (bal.get("holdings", []) or []) if _bal_valid else []
                actual_etf = next((h for h in actual_hlds if str(h.get("code", "")) == str(isa_etf_code)), None)
                actual_shares = int(actual_etf.get("qty", 0)) if actual_etf else 0
                actual_price = res.get("cur_price", 0)
                actual_eval = actual_shares * actual_price
                actual_total = actual_cash_v + actual_eval
                actual_cash_ratio = actual_cash_v / actual_total * 100 if actual_total > 0 else 0

                bt_shares = 0
                bt_cash_ratio = 0.0
                bt_equity = 0.0
                bt_return_pct = 0.0
                if bt:
                    eq_df = bt["equity_df"]
                    bt_last = eq_df.iloc[-1]
                    bt_shares = int(bt_last["shares"])
                    bt_cash = float(bt_last["cash"])
                    bt_equity = float(bt_last["equity"])
                    bt_return_pct = bt["metrics"]["total_return"]
                    bt_cash_ratio = bt_cash / bt_equity * 100 if bt_equity > 0 else 0

                # 동기화 주문 계산: 실제 자본 기준으로 백테스트 비율 스케일링
                _sync_action = None
                _sync_qty = 0
                _sync_reason = ""
                _bt_init = 10_000_000  # 백테스트 기준 자본

                if bt and _bal_valid and actual_price > 0:
                    # 백테스트의 주식 비율을 실제 자본에 적용
                    _bt_stock_ratio = (bt_shares * float(bt_last["price"])) / bt_equity if bt_equity > 0 else 0
                    _target_stock_val = actual_total * _bt_stock_ratio
                    _target_shares = int(_target_stock_val / actual_price) if actual_price > 0 else 0

                    _diff = _target_shares - actual_shares
                    if _diff > 0 and (_diff * actual_price) <= actual_cash_v:
                        _sync_action = "BUY"
                        _sync_qty = _diff
                        _sync_reason = f"백테스트 주식비율 {_bt_stock_ratio*100:.0f}% 맞추기 위해 {_diff}주 매수 필요"
                    elif _diff > 0 and actual_cash_v > 0:
                        _affordable = int(actual_cash_v * 0.9 / actual_price)  # 현금의 90%
                        if _affordable > 0:
                            _sync_action = "BUY"
                            _sync_qty = _affordable
                            _sync_reason = f"목표 {_target_shares}주 중 현금으로 {_affordable}주 매수 가능"
                        else:
                            _sync_reason = f"목표 {_target_shares}주, 현금 부족 (예수금 {actual_cash_v:,.0f}원)"
                    elif _diff < 0:
                        _sync_action = "SELL"
                        _sync_qty = abs(_diff)
                        _sync_reason = f"백테스트 주식비율 {_bt_stock_ratio*100:.0f}% 맞추기 위해 {abs(_diff)}주 매도 필요"
                    elif actual_total <= 0:
                        _sync_reason = "계좌에 자금이 없습니다"
                    else:
                        _sync_reason = "백테스트와 동기화 완료"

                # ── 백테스트 기준 주문 정보 ──
                _bt_action_str = "HOLD"
                _bt_action_qty = 0
                _bt_action_price = 0.0
                if bt:
                    _bt_action_str = str(bt_last.get("action", "HOLD")) or "HOLD"
                    _bt_action_qty = int(bt_last.get("quantity", 0))
                    _bt_action_price = float(bt_last["price"])

                act_str = act["action"] or "HOLD"
                _weekly_pnl = act['weekly_pnl']
                _weekly_qty = act['quantity']

                st.info(f"**백테스트 최근 동작**: {_bt_action_str} {_bt_action_qty}주 | **종가**: {_bt_action_price:,.0f}원")

                sc1, sc2 = st.columns(2)
                with sc1:
                    if _bt_action_qty > 0:
                        _color = "error" if _bt_action_str == "SELL" else "success"
                        getattr(st, _color)(f"### 📅 다음 거래일 주문: **{_bt_action_str} {_bt_action_qty}주**")
                    else:
                        st.success(f"### 📅 다음 거래일 주문: **HOLD (0주)**")
                    st.caption(f"백테스트 종가 {_bt_action_price:,.0f}원 기준 (1천만원 초기자본)")

                # 백테스트 요약 (시작일 ~ 현재)
                if bt:
                    m = bt["metrics"]
                    with sc2:
                        st.write(f"**전략 성과 ({isa_start_date} ~ 현재)**")
                        st.write(f"수익률: **{m['total_return']:+.2f}%** | MDD: **{m['mdd']:.2f}%** | CAGR: **{m['cagr']:.2f}%**")
                        st.write(f"최종자산: {m['final_equity']:,.0f}원 (1천만원 기준)")

                # 비교 메트릭
                if bt:
                    st.divider()
                    ac1, ac2, ac3, ac4 = st.columns(4)
                    ac1.metric(
                        "백테스트 자산 (1천만원 기준)",
                        f"{bt_equity:,.0f}원",
                        delta=f"{bt_return_pct:+.2f}%",
                    )
                    ac2.metric(
                        "실제 총자산",
                        f"{actual_total:,.0f}원" if _bal_valid else "조회 불가",
                    )
                    shares_diff = actual_shares - bt_shares
                    ac3.metric(
                        "보유주식 (백테/실제)",
                        f"{bt_shares}주 / {actual_shares}주",
                        delta=f"차이 {shares_diff:+d}주" if shares_diff != 0 else "일치",
                    )

                    bt_hold = bt_shares > 0
                    actual_hold = actual_shares > 0
                    sync = bt_hold == actual_hold
                    if sync:
                        ac4.metric("포지션 동기화", "일치")
                        ac4.caption(f"현금비율 — 백테: {bt_cash_ratio:.1f}% / 실제: {actual_cash_ratio:.1f}%")
                    else:
                        bt_state = "HOLD" if bt_hold else "CASH"
                        actual_state = "HOLD" if actual_hold else "CASH"
                        ac4.metric("포지션 동기화", "불일치")
                        ac4.caption(f"백테: {bt_state} / 실제: {actual_state}")

                # ── 실제 계좌 상태 (별도 영역) ──
                if _sync_action and _sync_qty > 0:
                    st.warning(f"**포지션 동기화 필요**: {_sync_action} {_sync_qty}주 — {_sync_reason}")
                if not _bal_valid:
                    st.warning("계좌 잔고 조회에 실패했습니다. KIS API 키를 확인해주세요.")
                elif actual_total <= 0:
                    st.warning("계좌에 자금이 없습니다. 입금 후 포지션 동기화를 진행해주세요.")

                # 추세선 차트
                # 차트 공통 시작일: equity_df가 있으면 실제 거래 시작일 사용
                _chart_start_ts = pd.Timestamp(isa_start_date)
                if bt and bt.get("equity_df") is not None and len(bt["equity_df"]) > 0:
                    _eq_first = bt["equity_df"].index[0]
                    if _eq_first > _chart_start_ts:
                        _chart_start_ts = _eq_first

                weekly_df = res.get("weekly_df")
                trend = res.get("trend")
                if weekly_df is not None and trend is not None:
                    start_ts = _chart_start_ts
                    mask = weekly_df.index >= start_ts
                    weekly_plot = weekly_df.loc[mask]
                    trend_plot = np.asarray(trend)[mask]
                    if len(weekly_plot) == 0:
                        st.warning(f"시작일({start_ts.date()}) 이후 표시할 데이터가 없습니다.")
                    else:
                        _trend_label = _fmt_etf_code_name(isa_trend_etf_code)
                        _trade_label = _fmt_etf_code_name(isa_etf_code)
                        st.markdown(
                            f"<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:0.7rem 0 1.1rem 0;'>"
                            f"WDR 모니터링: {_trend_label} 추세 분석"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=weekly_plot.index, y=weekly_plot["close"],
                            name=f"{_trend_label} (시그널)", line=dict(color="royalblue"),
                        ))
                        fig.add_trace(go.Scatter(
                            x=weekly_plot.index, y=trend_plot,
                            name="성장 추세선", line=dict(color="orange", dash="dash"),
                        ))
                        # 이격도 영역 표시
                        ov_th = float(wdr_ov) / 100.0
                        un_th = float(wdr_un) / 100.0
                        valid = ~np.isnan(trend_plot)
                        if valid.any():
                            ov_line = trend_plot.copy()
                            ov_line[valid] = trend_plot[valid] * (1 + ov_th)
                            un_line = trend_plot.copy()
                            un_line[valid] = trend_plot[valid] * (1 + un_th)
                            fig.add_trace(go.Scatter(
                                x=weekly_plot.index, y=ov_line,
                                name=f"고평가 +{wdr_ov}%", line=dict(color="red", dash="dot", width=1),
                            ))
                            fig.add_trace(go.Scatter(
                                x=weekly_plot.index, y=un_line,
                                name=f"저평가 {wdr_un}%", line=dict(color="green", dash="dot", width=1),
                            ))
                            # 5단계: 초고평가/초저평가 라인 추가
                            if int(wdr_eval_mode) == 5:
                                sov_line = trend_plot.copy()
                                sov_line[valid] = trend_plot[valid] * 1.10
                                sun_line = trend_plot.copy()
                                sun_line[valid] = trend_plot[valid] * 0.90
                                fig.add_trace(go.Scatter(
                                    x=weekly_plot.index, y=sov_line,
                                    name="초고평가 +10%", line=dict(color="darkred", dash="dot", width=1),
                                ))
                                fig.add_trace(go.Scatter(
                                    x=weekly_plot.index, y=sun_line,
                                    name="초저평가 -10%", line=dict(color="darkgreen", dash="dot", width=1),
                                ))
                        # 공통 x축 범위 (트렌드/수익률/DD 모두 동일)
                        _xrange = [start_ts, weekly_plot.index[-1]]
                        fig.update_layout(
                            xaxis_title="날짜", yaxis_title="가격",
                            xaxis=dict(range=_xrange),
                            height=450,
                            margin=dict(l=0, r=0, t=56, b=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.08),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)

                # ── 에쿼티(자산변화) + DD 차트 ──
                if bt:
                    eq_df = bt.get("equity_df")
                    if eq_df is not None and "equity" in eq_df.columns and len(eq_df) > 0:
                        _init_bal = 10_000_000
                        _eq_ret = (eq_df["equity"] / _init_bal - 1) * 100

                        # Buy & Hold (트렌드 ETF 기준)
                        _trend_bm_label = _fmt_etf_code_name(isa_trend_etf_code)
                        _bm_df = bt.get("benchmark_df")

                        # 공통 x축 범위 (트렌드 차트와 동일)
                        _eq_xrange = [_chart_start_ts, eq_df.index[-1]]

                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:0.7rem 0 1.1rem 0;'>누적 수익률 (%)</div>",
                            unsafe_allow_html=True,
                        )
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            x=eq_df.index, y=_eq_ret.values, mode="lines",
                            name="위대리 전략", line=dict(color="gold", width=2)
                        ))
                        if _bm_df is not None and "benchmark_return_pct" in _bm_df.columns:
                            fig_eq.add_trace(go.Scatter(
                                x=_bm_df.index, y=_bm_df["benchmark_return_pct"].values, mode="lines",
                                name=f"{_trend_bm_label} Buy & Hold", line=dict(color="gray", width=1, dash="dot")
                            ))
                        fig_eq.update_layout(
                            yaxis_title="수익률 (%)", height=370,
                            xaxis=dict(range=_eq_xrange),
                            margin=dict(l=0, r=0, t=56, b=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0)
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)
                        st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)

                        # DD 차트
                        _eq_peak = eq_df["equity"].cummax()
                        _eq_dd = (eq_df["equity"] - _eq_peak) / _eq_peak * 100

                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:0.7rem 0 1.1rem 0;'>Drawdown</div>",
                            unsafe_allow_html=True,
                        )
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(
                            x=eq_df.index, y=_eq_dd.values, mode="lines",
                            name="전략 DD", line=dict(color="crimson", width=2),
                            fill="tozeroy", fillcolor="rgba(220,20,60,0.15)"
                        ))
                        if _bm_df is not None and "benchmark_return_pct" in _bm_df.columns:
                            _bm_eq = _bm_df["benchmark_return_pct"] / 100 + 1
                            _bm_peak = _bm_eq.cummax()
                            _bm_dd = (_bm_eq - _bm_peak) / _bm_peak * 100
                            fig_dd.add_trace(go.Scatter(
                                x=_bm_dd.index, y=_bm_dd.values, mode="lines",
                                name=f"{_trend_bm_label} Buy & Hold DD", line=dict(color="gray", width=1, dash="dot")
                            ))
                        fig_dd.update_layout(
                            yaxis_title="DD (%)", height=300,
                            xaxis=dict(range=_eq_xrange),
                            margin=dict(l=0, r=0, t=56, b=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0)
                        )
                        fig_dd = _apply_dd_hover_format(fig_dd)
                        st.plotly_chart(fig_dd, use_container_width=True)

                        # ── 상세 성과 분석 (벤치마크, 월별, 몬테카를로, 켈리) ──
                        _render_performance_analysis(
                            equity_series=eq_df["equity"],
                            benchmark_series=_bm_df["benchmark_return_pct"] / 100 + 1 if _bm_df is not None else None,
                            strategy_metrics=bt.get("metrics"),
                            strategy_label="위대리 전략",
                            benchmark_label=f"{_trend_bm_label} Buy & Hold",
                        )

    # ══════════════════════════════════════════════════════════════
    # Tab 2: 수동 주문
    # ══════════════════════════════════════════════════════════════
    with tab_i2:
        st.header("수동 주문")
        st.caption(f"매매 대상: {selected_etf_label}")

        bal = st.session_state.get(isa_bal_key)
        if not bal:
            st.warning("잔고를 먼저 조회해 주세요. (모니터링 탭에서 잔고 새로고침)")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []
            etf_holding = next((h for h in holdings if str(h.get("code", "")) == str(isa_etf_code)), None)
            holding_qty = int(etf_holding.get("qty", 0)) if etf_holding else 0

            # ── 상단 정보 바 (골드 패널과 동일 구조) ──
            cur_price = _get_isa_current_price(str(isa_etf_code))
            _isa_cur = float(cur_price) if cur_price and cur_price > 0 else 0
            _isa_eval = _isa_cur * holding_qty if _isa_cur > 0 else 0

            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("현재가", f"{_isa_cur:,.0f}원" if _isa_cur > 0 else "–")
            ic2.metric(f"{_fmt_etf_code_name(isa_etf_code)} 보유", f"{holding_qty}주")
            ic3.metric("평가금액", f"{_isa_eval:,.0f}원")
            ic4.metric("예수금", f"{cash:,.0f}원")

            # ═══ 일봉 차트 (상단 전체폭) ═══
            _isa_chart_df = _get_isa_daily_chart(str(isa_etf_code), count=120)
            if _isa_chart_df is not None and len(_isa_chart_df) > 0:
                _isa_chart_df = _isa_chart_df.copy().sort_index()
                if "close" not in _isa_chart_df.columns and "Close" in _isa_chart_df.columns:
                    _isa_chart_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
                st.markdown(f"**{_fmt_etf_code_name(isa_etf_code)} 일봉 차트**")
                _fig_isa = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
                _fig_isa.add_trace(go.Candlestick(
                    x=_isa_chart_df.index, open=_isa_chart_df['open'], high=_isa_chart_df['high'],
                    low=_isa_chart_df['low'], close=_isa_chart_df['close'], name='일봉',
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                ), row=1, col=1)
                _isa_ma5 = _isa_chart_df['close'].rolling(5).mean()
                _isa_ma20 = _isa_chart_df['close'].rolling(20).mean()
                _fig_isa.add_trace(go.Scatter(x=_isa_chart_df.index, y=_isa_ma5, name='MA5', line=dict(color='#FF9800', width=1)), row=1, col=1)
                _fig_isa.add_trace(go.Scatter(x=_isa_chart_df.index, y=_isa_ma20, name='MA20', line=dict(color='#2196F3', width=1)), row=1, col=1)
                if 'volume' in _isa_chart_df.columns:
                    _isa_vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(_isa_chart_df['close'], _isa_chart_df['open'])]
                    _fig_isa.add_trace(go.Bar(x=_isa_chart_df.index, y=_isa_chart_df['volume'], marker_color=_isa_vol_colors, name='거래량', showlegend=False), row=2, col=1)
                _fig_isa.update_layout(
                    height=450, margin=dict(l=0, r=0, t=10, b=30),
                    xaxis_rangeslider_visible=False, showlegend=True,
                    legend=dict(orientation="h", y=1.06, x=0),
                    xaxis2=dict(showticklabels=True, tickformat='%m/%d', tickangle=-45),
                    yaxis=dict(title="", side="right"),
                    yaxis2=dict(title="", side="right"),
                )
                st.plotly_chart(_fig_isa, use_container_width=True, key=f"isa_manual_chart_{isa_etf_code}")
            else:
                st.info("차트 데이터 로딩 중...")


            st.divider()

            ob_col, order_col = st.columns([2, 3])

            # ── 좌: 호가창 ──
            with ob_col:
                ob = trader.get_orderbook(str(isa_etf_code))
                if ob and ob.get("asks") and ob.get("bids"):
                    asks = ob["asks"]
                    bids = ob["bids"]
                    all_qtys = [a["qty"] for a in asks] + [b["qty"] for b in bids]
                    max_qty = max(all_qtys) if all_qtys else 1
                    html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                    html.append(
                        '<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666">'
                        '<td>구분</td><td style="text-align:right">잔량</td>'
                        '<td style="text-align:right">가격(원)</td>'
                        '<td style="text-align:right">등락</td><td>비율</td></tr>'
                    )
                    for a in reversed(asks):
                        ap, aq = a["price"], a["qty"]
                        diff = ((ap / _isa_cur) - 1) * 100 if _isa_cur > 0 else 0
                        bar_w = int(aq / max_qty * 100) if max_qty > 0 else 0
                        html.append(
                            f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                            f'<td>매도</td>'
                            f'<td style="text-align:right">{aq:,}</td>'
                            f'<td style="text-align:right;font-weight:bold">{ap:,.0f}</td>'
                            f'<td style="text-align:right">{diff:+.2f}%</td>'
                            f'<td><div style="background:#1976D233;height:14px;width:{bar_w}%"></div></td></tr>'
                        )
                    html.append(
                        f'<tr style="background:#FFF3E0;border:2px solid #FF9800;height:32px;font-weight:bold">'
                        f'<td colspan="2" style="color:#E65100">현재가</td>'
                        f'<td style="text-align:right;color:#E65100;font-size:15px">{_isa_cur:,.0f}</td>'
                        f'<td colspan="2"></td></tr>'
                    )
                    for b in bids:
                        bp, bq = b["price"], b["qty"]
                        diff = ((bp / _isa_cur) - 1) * 100 if _isa_cur > 0 else 0
                        bar_w = int(bq / max_qty * 100) if max_qty > 0 else 0
                        html.append(
                            f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                            f'<td>매수</td>'
                            f'<td style="text-align:right">{bq:,}</td>'
                            f'<td style="text-align:right;font-weight:bold">{bp:,.0f}</td>'
                            f'<td style="text-align:right">{diff:+.2f}%</td>'
                            f'<td><div style="background:#D32F2F33;height:14px;width:{bar_w}%"></div></td></tr>'
                        )
                    html.append("</table>")
                    st.markdown("".join(html), unsafe_allow_html=True)
                    if asks and bids:
                        spread = asks[0]["price"] - bids[0]["price"]
                        spread_pct = (spread / _isa_cur * 100) if _isa_cur > 0 else 0
                        total_ask_q = sum(a["qty"] for a in asks)
                        total_bid_q = sum(b["qty"] for b in bids)
                        st.caption(
                            f"스프레드: {spread:,.0f}원 ({spread_pct:.3f}%) | "
                            f"매도잔량: {total_ask_q:,} | 매수잔량: {total_bid_q:,}"
                        )
                else:
                    st.info("호가 데이터를 불러오는 중...")

            # ── 우: 주문 패널 ──
            with order_col:
                buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                with buy_tab:
                    buy_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"], key="isa_buy_method", horizontal=True)

                    buy_price = 0
                    if buy_method == "지정가":
                        buy_price = st.number_input("매수 지정가 (원)", min_value=0, value=int(_isa_cur) if _isa_cur > 0 else 0, step=50, key="isa_buy_price")
                    else:
                        buy_price = _isa_cur

                    buy_qty = st.number_input("매수 수량 (주)", min_value=0, value=0, step=1, key="isa_buy_qty")

                    _isa_unit = buy_price if buy_price > 0 else _isa_cur
                    _isa_total = int(buy_qty * _isa_unit) if buy_qty > 0 and _isa_unit > 0 else 0
                    st.markdown(
                        f"<div style='background:#fff3f3;border:1px solid #ffcdd2;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                        f"<b>단가:</b> {_isa_unit:,.0f}원 &nbsp;|&nbsp; "
                        f"<b>수량:</b> {buy_qty:,}주 &nbsp;|&nbsp; "
                        f"<b>총 금액:</b> <span style='color:#D32F2F;font-weight:bold'>{_isa_total:,}원</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    buy_amount = _isa_total

                    if st.button("매수 실행", key="isa_exec_buy", type="primary", disabled=IS_CLOUD):
                        if buy_qty <= 0:
                            st.error("매수 수량을 입력해 주세요.")
                        elif _isa_total > cash:
                            st.error(f"예수금 부족 (필요: {_isa_total:,}원 / 예수금: {cash:,.0f}원)")
                        else:
                            with st.spinner("매수 주문 실행 중..."):
                                if buy_method == "동시호가 (장마감)":
                                    result = trader.execute_closing_auction_buy(str(isa_etf_code), buy_qty) if buy_qty > 0 else None
                                elif buy_method == "지정가" and buy_price > 0:
                                    result = trader.send_order("BUY", str(isa_etf_code), buy_qty, price=buy_price, ord_dvsn="00") if buy_qty > 0 else None
                                elif buy_method == "시간외 종가":
                                    result = trader.send_order("BUY", str(isa_etf_code), buy_qty, price=0, ord_dvsn="06") if buy_qty > 0 else None
                                else:
                                    result = trader.send_order("BUY", str(isa_etf_code), buy_qty, ord_dvsn="01") if buy_qty > 0 else None
                                if result and (isinstance(result, dict) and result.get("success")):
                                    st.success(f"매수 주문 완료: {result}")
                                    st.session_state[isa_bal_key] = trader.get_balance()
                                else:
                                    st.error(f"매수 실패: {result}")

                with sell_tab:
                    sell_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"], key="isa_sell_method", horizontal=True)

                    sell_price = 0
                    if sell_method == "지정가":
                        sell_price = st.number_input("매도 지정가 (원)", min_value=0, value=int(_isa_cur) if _isa_cur > 0 else 0, step=50, key="isa_sell_price")
                    else:
                        sell_price = _isa_cur

                    sell_qty = st.number_input("매도 수량 (주)", min_value=0, max_value=max(holding_qty, 1), value=holding_qty, step=1, key="isa_sell_qty")
                    sell_all = st.checkbox("전량 매도", value=True, key="isa_sell_all")

                    _isa_sell_unit = sell_price if sell_price > 0 else _isa_cur
                    _isa_sell_qty_final = holding_qty if sell_all else sell_qty
                    _isa_sell_total = int(_isa_sell_qty_final * _isa_sell_unit) if _isa_sell_qty_final > 0 and _isa_sell_unit > 0 else 0
                    st.markdown(
                        f"<div style='background:#f3f8ff;border:1px solid #bbdefb;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                        f"<b>단가:</b> {_isa_sell_unit:,.0f}원 &nbsp;|&nbsp; "
                        f"<b>수량:</b> {_isa_sell_qty_final:,}주 &nbsp;|&nbsp; "
                        f"<b>총 금액:</b> <span style='color:#1976D2;font-weight:bold'>{_isa_sell_total:,}원</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if st.button("매도 실행", key="isa_exec_sell", type="primary", disabled=IS_CLOUD):
                        _sq = holding_qty if sell_all else sell_qty
                        if _sq <= 0:
                            st.error("매도할 수량이 없습니다.")
                        else:
                            with st.spinner("매도 주문 실행 중..."):
                                if sell_method == "동시호가 (장마감)":
                                    if sell_all:
                                        result = trader.smart_sell_all_closing(str(isa_etf_code))
                                    else:
                                        result = trader.smart_sell_qty_closing(str(isa_etf_code), _sq)
                                elif sell_method == "지정가" and sell_price > 0:
                                    result = trader.send_order("SELL", str(isa_etf_code), _sq, price=sell_price, ord_dvsn="00")
                                elif sell_method == "시간외 종가":
                                    result = trader.send_order("SELL", str(isa_etf_code), _sq, price=0, ord_dvsn="06")
                                else:
                                    if sell_all:
                                        result = trader.smart_sell_all(str(isa_etf_code))
                                    else:
                                        result = trader.smart_sell_qty(str(isa_etf_code), _sq)
                                if result and (isinstance(result, dict) and result.get("success")):
                                    st.success(f"매도 주문 완료: {result}")
                                    st.session_state[isa_bal_key] = trader.get_balance()
                                else:
                                    st.error(f"매도 실패: {result}")

    # ══════════════════════════════════════════════════════════════
    # Tab 3: 주문방식
    # ══════════════════════════════════════════════════════════════
    with tab_i3:
        st.header("KIS 국내 ETF 주문방식 안내")
        st.dataframe(pd.DataFrame([
            {"구분": "시장가", "API": 'ord_dvsn="01"', "설명": "즉시 체결 (최우선 호가)"},
            {"구분": "지정가", "API": 'ord_dvsn="00"', "설명": "원하는 가격에 주문"},
            {"구분": "동시호가 매수", "API": '상한가(+30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
            {"구분": "동시호가 매도", "API": '하한가(-30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
            {"구분": "시간외 종가", "API": 'ord_dvsn="06"', "설명": "15:40~16:00 당일 종가로 체결"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("호가단위")
        st.dataframe(pd.DataFrame([
            {"가격대": "~5,000원", "호가단위": "5원"},
            {"가격대": "5,000~10,000원", "호가단위": "10원"},
            {"가격대": "10,000~50,000원", "호가단위": "50원"},
            {"가격대": "50,000원~", "호가단위": "100원"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("자동매매 흐름 (GitHub Actions)")
        st.markdown(f"""
1. 매주 금요일 KST 15:20 실행 (`TRADING_MODE=kis_isa`)
2. TREND ETF({_fmt_etf_code_name(isa_trend_etf_code)}) 일봉 → 주봉 변환 → 성장 추세선 계산
3. 이격도 기반 시장 상태 판단 (고평가/중립/저평가)
4. 주간 손익 × 매도/매수 비율 → 주문 수량 산출
5. `execute_closing_auction_buy/sell` → 동시호가 주문 + 미체결 시 시간외 재주문
""")

    # ══════════════════════════════════════════════════════════════
    # Tab 4: 수수료/세금
    # ══════════════════════════════════════════════════════════════
    with tab_i4:
        st.header("ISA 계좌 수수료 및 세금 안내")
        st.subheader("1. 매매 수수료")
        st.dataframe(pd.DataFrame([
            {"증권사": "한국투자증권", "매매 수수료": "0.0140396%", "비고": "나무 온라인 (현재 사용)"},
            {"증권사": "키움증권", "매매 수수료": "0.015%", "비고": "영웅문 온라인"},
            {"증권사": "미래에셋", "매매 수수료": "0.014%", "비고": "m.Stock 온라인"},
            {"증권사": "토스증권", "매매 수수료": "무료~0.015%", "비고": "이벤트 시 무료"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("2. 매매 대상 ETF 보수")
        _isa_fee_rows = [
            {"ETF": "TIGER 미국나스닥100레버리지(합성)", "코드": "418660", "총보수": "0.25%", "구분": "나스닥 2x"},
            {"ETF": "KODEX 미국나스닥100레버리지(합성 H)", "코드": "409820", "총보수": "0.30%", "구분": "나스닥 2x(H)"},
            {"ETF": "TIGER 미국필라델피아반도체레버리지(합성)", "코드": "423920", "총보수": "0.58%", "구분": "반도체 2x"},
            {"ETF": "ACE 미국빅테크TOP7 Plus레버리지(합성)", "코드": "465610", "총보수": "0.60%", "구분": "빅테크 2x"},
            {"ETF": _etf_name_kr(isa_trend_etf_code), "코드": str(isa_trend_etf_code), "총보수": "-", "구분": "TREND ETF(시그널)"},
        ]
        st.dataframe(pd.DataFrame(_isa_fee_rows), use_container_width=True, hide_index=True)

        st.subheader("3. ISA 세제혜택")
        st.markdown("""
| 항목 | 내용 |
|------|------|
| 비과세 한도 | 일반형 200만원 / 서민·청년형 400만원 |
| 한도 초과 | 9.9% 분리과세 (일반 15.4% 대비 유리) |
| 의무가입기간 | 3년 |
| 납입한도 | 연 2,000만원 (총 1억원) |
""")
        st.caption("위대리 전략의 주간 리밸런싱 매매차익이 비과세 한도 내에서 면세 처리되어 절세 효과 극대화")

    # ══════════════════════════════════════════════════════════════
    # Tab 5: 미국 위대리 (QQQ 추세 → TQQQ 매매) 백테스트
    # ══════════════════════════════════════════════════════════════
    with tab_i5:
        st.header("위대리 (WTR) 백테스트")
        st.caption("QQQ 성장 추세선 이격도 기반 TQQQ 주간 리밸런싱 전략")

        from strategy.widaeri import WDRStrategy
        import data_cache as _yf_dc
        from datetime import date as _dt_date

        # ── 평가 시스템 ──
        _us_mode_col, _ = st.columns(2)
        with _us_mode_col:
            us_eval_mode = st.selectbox("평가 시스템", [3, 5], index=1,
                                        format_func=lambda x: f"{x}단계", key="us_wdr_eval_mode")

        # ── 파라미터 입력 ──
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            _def_start = _dt_date(2012, 1, 1)
            bt_start_date = st.date_input("시작일", value=_def_start,
                min_value=_dt_date(1999, 3, 10), max_value=_dt_date.today(), key="us_wdr_start_date")
        with pcol2:
            bt_end_date = st.date_input("종료일", value=_dt_date.today(),
                min_value=_dt_date(2000, 1, 1), max_value=_dt_date.today(), key="us_wdr_end_date")
        with pcol3:
            bt_cap = st.number_input("초기 자본 ($)", min_value=1000,
                value=10000, step=1000, key="bt_tqqq_cap")
        with pcol4:
            bt_fee = st.number_input("매매 수수료 (%)", min_value=0.0, max_value=1.0,
                value=0.01, step=0.005, format="%.3f", key="bt_tqqq_fee_pct")

        pcol5, pcol6, _, _ = st.columns(4)
        with pcol5:
            bt_ov = st.number_input("고평가 임계값 (%)", min_value=0.0, max_value=30.0,
                value=5.0, step=0.5, key="bt_tqqq_ov")
        with pcol6:
            bt_un = st.number_input("저평가 임계값 (%)", min_value=-30.0, max_value=0.0,
                value=-6.0, step=0.5, key="bt_tqqq_un")

        if st.button("백테스트 실행", key="tqqq_wdr_run_bt", type="primary", use_container_width=True):
            with st.spinner("QQQ/TQQQ 데이터 로드 중..."):
                start_str = str(bt_start_date)
                end_str = str(bt_end_date)
                try:
                    df_sig_full = _yf_dc.fetch_and_cache_yf("QQQ", start="1999-03-10", force_refresh=True)
                    df_trade_raw = _yf_dc.fetch_and_cache_yf("TQQQ", start="2010-02-12", force_refresh=True)

                    if df_sig_full is None or df_sig_full.empty or df_trade_raw is None or df_trade_raw.empty:
                        st.error("데이터 로드 실패. data/ 폴더에 QQQ_daily.csv, TQQQ_daily.csv를 확인하세요.")
                    else:
                        _end_ts = pd.Timestamp(end_str)
                        df_sig_full = df_sig_full[df_sig_full.index <= _end_ts]
                        df_trade_raw = df_trade_raw[df_trade_raw.index <= _end_ts]

                        sig_df_full = pd.DataFrame({"close": df_sig_full["close"]}).dropna()
                        trade_df = pd.DataFrame({"close": df_trade_raw["close"]}).dropna()

                        _bt_strategy = WDRStrategy(settings={
                            "overvalue_threshold": bt_ov,
                            "undervalue_threshold": bt_un,
                            "commission_rate": bt_fee / 100.0,
                        }, evaluation_mode=us_eval_mode)

                        result = _bt_strategy.run_backtest(
                            signal_daily_df=sig_df_full,
                            trade_daily_df=trade_df,
                            initial_balance=bt_cap,
                            start_date=start_str
                        )

                        if result:
                            m = result["metrics"]
                            eq_df = result["equity_df"]
                            bench = result["benchmark_df"]

                            # ── 현재 이격도 표시 ──
                            _weekly_full = _bt_strategy.daily_to_weekly(sig_df_full)
                            _trend_full = _bt_strategy.calc_growth_trend(_weekly_full)
                            _trend_full_arr = np.asarray(_trend_full)
                            _last_div = None
                            if len(_weekly_full) > 0 and not np.isnan(_trend_full_arr[-1]):
                                _last_div = (_weekly_full["close"].iloc[-1] - _trend_full_arr[-1]) / _trend_full_arr[-1] * 100
                                _last_state, _, _ = _bt_strategy.get_market_state(_last_div)
                                _state_kr = {"SUPER_OVERVALUE": "초고평가", "OVERVALUE": "고평가", "NEUTRAL": "중립",
                                             "UNDERVALUE": "저평가", "SUPER_UNDERVALUE": "초저평가"}.get(_last_state, _last_state)
                                st.info(f"현재 이격도: **{_last_div:+.2f}%** | 평가: **{_state_kr}** ({us_eval_mode}단계)")

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("총 수익률", f"{m['total_return']:+.2f}%")
                            c2.metric("CAGR", f"{m['cagr']:+.2f}%")
                            c3.metric("MDD", f"{m['mdd']:+.2f}%")
                            c4.metric("Calmar", f"{m['calmar']:.2f}")

                            c5, c6, c7, c8 = st.columns(4)
                            c5.metric("매매 횟수", f"{m['trade_count']}회")
                            c6.metric("승률", f"{m['win_rate']:.1f}%")
                            c7.metric("최종 자산", f"${m['final_equity']:,.0f}")
                            c8.metric("샤프", f"{m['sharpe']:.2f}")

                            # ── QQQ Growth Trend 차트 ──
                            if _weekly_full is not None and _trend_full is not None and len(_weekly_full) > 0:
                                _start_ts = pd.Timestamp(start_str)
                                _chart_mask = _weekly_full.index >= _start_ts
                                _wk_chart = _weekly_full[_chart_mask]
                                _tr_chart = _trend_full_arr[_chart_mask]

                                fig_trend = go.Figure()
                                fig_trend.add_trace(go.Scatter(
                                    x=_wk_chart.index, y=_wk_chart["close"],
                                    name="QQQ", line=dict(color="royalblue"),
                                ))
                                fig_trend.add_trace(go.Scatter(
                                    x=_wk_chart.index, y=_tr_chart,
                                    name="Trend", line=dict(color="orange", dash="dash"),
                                ))
                                _valid = ~np.isnan(_tr_chart)
                                if _valid.any():
                                    _ov_line = _tr_chart.copy()
                                    _ov_line[_valid] = _tr_chart[_valid] * (1 + bt_ov / 100.0)
                                    _un_line = _tr_chart.copy()
                                    _un_line[_valid] = _tr_chart[_valid] * (1 + bt_un / 100.0)
                                    fig_trend.add_trace(go.Scatter(
                                        x=_wk_chart.index, y=_ov_line,
                                        name=f"고평가 +{bt_ov}%", line=dict(color="red", dash="dot", width=1),
                                    ))
                                    fig_trend.add_trace(go.Scatter(
                                        x=_wk_chart.index, y=_un_line,
                                        name=f"저평가 {bt_un}%", line=dict(color="green", dash="dot", width=1),
                                    ))
                                    if us_eval_mode == 5:
                                        _sov_line = _tr_chart.copy()
                                        _sov_line[_valid] = _tr_chart[_valid] * 1.10
                                        _sun_line = _tr_chart.copy()
                                        _sun_line[_valid] = _tr_chart[_valid] * 0.90
                                        fig_trend.add_trace(go.Scatter(
                                            x=_wk_chart.index, y=_sov_line,
                                            name="초고평가 +10%", line=dict(color="darkred", dash="dot", width=1),
                                        ))
                                        fig_trend.add_trace(go.Scatter(
                                            x=_wk_chart.index, y=_sun_line,
                                            name="초저평가 -10%", line=dict(color="darkgreen", dash="dot", width=1),
                                        ))
                                fig_trend.update_layout(
                                    title=f"QQQ Growth Trend ({us_eval_mode}단계)",
                                    xaxis_title="날짜", yaxis_title="가격 ($)",
                                    height=400, margin=dict(l=0, r=0, t=70, b=30),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.06),
                                )
                                st.plotly_chart(fig_trend, use_container_width=True)

                            # ── 에쿼티(수익률) 차트 ──
                            _eq_ret = (eq_df["equity"] / bt_cap - 1) * 100
                            fig_eq = go.Figure()
                            fig_eq.add_trace(go.Scatter(
                                x=eq_df.index, y=_eq_ret.values,
                                name="WTR (QQQ→TQQQ)", line=dict(color="gold", width=2)
                            ))
                            if bench is not None and "benchmark_return_pct" in bench.columns:
                                fig_eq.add_trace(go.Scatter(
                                    x=bench.index, y=bench["benchmark_return_pct"],
                                    name="TQQQ Buy & Hold", line=dict(color="gray", width=1, dash="dot")
                                ))
                            fig_eq.update_layout(
                                title="누적 수익률 비교 (%)",
                                yaxis_title="수익률 (%)", height=400,
                                margin=dict(l=0, r=0, t=70, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.06)
                            )
                            st.plotly_chart(fig_eq, use_container_width=True)

                            # ── DD 차트 ──
                            _dd_peak = eq_df["equity"].cummax()
                            _dd_val = (eq_df["equity"] - _dd_peak) / _dd_peak * 100
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=eq_df.index, y=_dd_val.values, mode="lines",
                                name="전략 DD", line=dict(color="crimson", width=2),
                                fill="tozeroy", fillcolor="rgba(220,20,60,0.15)"
                            ))
                            if bench is not None and "benchmark_return_pct" in bench.columns:
                                _bm_eq = bench["benchmark_return_pct"] / 100 + 1
                                _bm_peak = _bm_eq.cummax()
                                _bm_dd = (_bm_eq - _bm_peak) / _bm_peak * 100
                                fig_dd.add_trace(go.Scatter(
                                    x=_bm_dd.index, y=_bm_dd.values, mode="lines",
                                    name="TQQQ B&H DD", line=dict(color="gray", width=1, dash="dot")
                                ))
                            fig_dd.update_layout(
                                title="Drawdown", yaxis_title="DD (%)", height=280,
                                margin=dict(l=0, r=0, t=70, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.06)
                            )
                            fig_dd = _apply_dd_hover_format(fig_dd)
                            st.plotly_chart(fig_dd, use_container_width=True)

                            # ── 연도별 성과 ──
                            if len(eq_df) > 1:
                                _yearly = []
                                _eq_s = eq_df["equity"]
                                for _yr, _grp in _eq_s.groupby(_eq_s.index.year):
                                    if len(_grp) < 2: continue
                                    _yr_ret = (_grp.iloc[-1] / _grp.iloc[0] - 1) * 100
                                    _yr_peak = _grp.cummax()
                                    _yr_dd = ((_grp - _yr_peak) / _yr_peak * 100).min()
                                    _yearly.append({"연도": _yr, "수익률(%)": round(_yr_ret, 2), "MDD(%)": round(_yr_dd, 2)})
                                if _yearly:
                                    with st.expander("연도별 성과"):
                                        st.dataframe(pd.DataFrame(_yearly), use_container_width=True)

                            with st.expander("상세 거래 내역"):
                                st.dataframe(pd.DataFrame(result["trades"]), use_container_width=True)
                        else:
                            st.error("백테스트 결과 생성 실패 (데이터 부족)")
                except Exception as e:
                    st.error(f"백테스트 중 오류 발생: {e}")



def render_kis_pension_mode():
    """KIS 연금저축 포트폴리오 모드 - 다중 전략 지원."""
    from kis_trader import KISTrader
    from strategy.laa import LAAStrategy
    from strategy.dual_momentum import DualMomentumStrategy
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── 사용 가능한 전략 목록 ──
    PEN_STRATEGIES = ["LAA", "듀얼모멘텀", "정적배분"]

    st.title("연금저축 포트폴리오")
    st.sidebar.header("연금저축 설정")
    _pen_bt_start_raw = str(config.get("start_date", "2020-01-01") or "2020-01-01")
    try:
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()
    except Exception:
        _pen_bt_start_raw = "2020-01-01"
        _pen_bt_start_ts = pd.Timestamp(_pen_bt_start_raw).normalize()

    kis_ak = _get_runtime_value(("KIS_PENSION_APP_KEY", "KIS_APP_KEY"), "")
    kis_sk = _get_runtime_value(("KIS_PENSION_APP_SECRET", "KIS_APP_SECRET"), "")
    kis_acct = str(config.get("kis_pension_account_no", "") or _get_runtime_value(("KIS_PENSION_ACCOUNT_NO", "KIS_ACCOUNT_NO"), ""))
    kis_prdt = str(config.get("kis_pension_prdt_cd", "") or _get_runtime_value(("KIS_PENSION_ACNT_PRDT_CD", "KIS_ACNT_PRDT_CD"), "01"))

    # 계좌번호 10자리 → 앞8(CANO) + 뒤2(상품코드) 자동 분리
    _raw_acct = kis_acct.replace("-", "").strip()
    if len(_raw_acct) == 10 and kis_prdt in ("01", ""):
        kis_acct = _raw_acct[:8]
        kis_prdt = _raw_acct[8:]

    if not IS_CLOUD:
        with st.sidebar.expander("KIS API 키", expanded=False):
            kis_ak = st.text_input("앱 키", value=kis_ak, type="password", key="pen_app_key")
            kis_sk = st.text_input("시크릿 키", value=kis_sk, type="password", key="pen_app_secret")
            kis_acct = st.text_input("계좌번호 (앞 8자리)", value=kis_acct, key="pen_account_no", help="10자리 입력 시 자동으로 앞8+뒤2 분리")
            kis_prdt = st.text_input("상품코드 (뒤 2자리)", value=kis_prdt, key="pen_prdt_cd")
            _raw2 = kis_acct.replace("-", "").strip()
            if len(_raw2) == 10:
                kis_acct = _raw2[:8]
                kis_prdt = _raw2[8:]

    # ── 포트폴리오 편집기 ──
    st.sidebar.subheader("포트폴리오")
    _saved_portfolio = config.get("pension_portfolio", [{"strategy": "LAA", "weight": 100}])
    if not _saved_portfolio:
        _saved_portfolio = [{"strategy": "LAA", "weight": 100}]
    _pen_port_df = pd.DataFrame(_saved_portfolio)

    def _normalize_pen_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            df = pd.DataFrame([{"strategy": "LAA", "weight": 100}])
        if "strategy" not in df.columns:
            df["strategy"] = "LAA"
        if "weight" not in df.columns:
            df["weight"] = 0
        out = df[["strategy", "weight"]].copy()
        out["strategy"] = out["strategy"].astype(str).str.strip()
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0)

        # 연금저축 화면에서는 LAA/듀얼모멘텀 행이 항상 보이도록 유지
        if not (out["strategy"] == "LAA").any():
            out = pd.concat([pd.DataFrame([{"strategy": "LAA", "weight": 100}]), out], ignore_index=True)
        if not (out["strategy"] == "듀얼모멘텀").any():
            out = pd.concat([out, pd.DataFrame([{"strategy": "듀얼모멘텀", "weight": 0}])], ignore_index=True)
        return out

    _pen_port_df = _normalize_pen_portfolio_df(_pen_port_df)

    _pen_port_state_key = "pen_portfolio_editor_df"
    if _pen_port_state_key not in st.session_state or not isinstance(st.session_state.get(_pen_port_state_key), pd.DataFrame):
        st.session_state[_pen_port_state_key] = _normalize_pen_portfolio_df(_pen_port_df)
    else:
        _state_df = st.session_state[_pen_port_state_key].copy()
        st.session_state[_pen_port_state_key] = _normalize_pen_portfolio_df(_state_df)

    _pen_port_edited = st.sidebar.data_editor(
        st.session_state[_pen_port_state_key],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="pen_portfolio_editor",
        column_config={
            "strategy": st.column_config.SelectboxColumn("전략", options=PEN_STRATEGIES, required=True),
            "weight": st.column_config.NumberColumn("비중(%)", min_value=0, max_value=100, step=5, required=True),
        },
    )
    if "strategy" not in _pen_port_edited.columns:
        _pen_port_edited["strategy"] = "LAA"
    if "weight" not in _pen_port_edited.columns:
        _pen_port_edited["weight"] = 0
    _pen_port_edited["weight"] = pd.to_numeric(_pen_port_edited["weight"], errors="coerce").fillna(0)

    # data_editor 내부 상태가 이전 값(예: LAA 단일행)으로 고정되는 경우를 보정
    _normalized_edited = _normalize_pen_portfolio_df(_pen_port_edited)
    _need_editor_resync = (
        len(_normalized_edited) != len(_pen_port_edited)
        or not (_normalized_edited["strategy"] == "LAA").any()
        or not (_normalized_edited["strategy"] == "듀얼모멘텀").any()
    )
    st.session_state[_pen_port_state_key] = _normalized_edited[["strategy", "weight"]].copy()
    _pen_port_edited = _normalized_edited.copy()
    if _need_editor_resync:
        if "pen_portfolio_editor" in st.session_state:
            del st.session_state["pen_portfolio_editor"]
        st.rerun()

    _total_w = float(pd.to_numeric(_pen_port_edited["weight"], errors="coerce").fillna(0).sum()) if not _pen_port_edited.empty else 0.0
    if _total_w > 100:
        st.sidebar.error(f"비중 합계: {_total_w:.0f}% (100% 이하여야 합니다)")
    else:
        _cash_w = max(0.0, 100.0 - _total_w)
        st.sidebar.caption(f"투자 비중: {_total_w:.0f}% | 현금: {_cash_w:.0f}%")

    # 현재 포트폴리오에 포함된 전략 목록
    _active_strategies = list(_pen_port_edited["strategy"].unique()) if not _pen_port_edited.empty else []

    # 전략별 상세 설정 패널 (한 번에 하나만 펼치기)
    # 포트폴리오에 존재하는 전략은 비중과 무관하게 설정 패널 노출
    _panel_strategies = []
    if not _pen_port_edited.empty:
        _panel_df = _pen_port_edited.copy()
        _panel_strategies = list(_panel_df["strategy"].astype(str).unique())
    _panel_options = ["접기"]
    if "LAA" in _panel_strategies:
        _panel_options.append("LAA 전략 설정")
    if "듀얼모멘텀" in _panel_strategies:
        _panel_options.append("듀얼모멘텀 설정")
    if "정적배분" in _panel_strategies:
        _panel_options.append("정적배분 설정")

    _panel_key = "pen_strategy_settings_panel"
    if st.session_state.get(_panel_key) not in _panel_options:
        st.session_state[_panel_key] = _panel_options[0]
    st.sidebar.caption("전략 상세 설정 (하나만 펼치기)")
    _selected_panel = st.sidebar.radio(
        "전략 상세 설정",
        _panel_options,
        key=_panel_key,
        label_visibility="collapsed",
    )

    # ── LAA 전략 설정 ──
    _kr_iwd_default = _code_only(config.get("kr_etf_laa_iwd", _get_runtime_value("KR_ETF_LAA_IWD", _get_runtime_value("KR_ETF_SPY", "360750"))))
    _kr_spy_default = _code_only(config.get("kr_etf_laa_spy", _get_runtime_value("KR_ETF_LAA_SPY", _get_runtime_value("KR_ETF_SPY", _kr_iwd_default or "360750"))))
    _kr_gld_default = _code_only(config.get("kr_etf_laa_gld", _get_runtime_value("KR_ETF_LAA_GLD", "132030")))
    _kr_ief_default = _code_only(config.get("kr_etf_laa_ief", _get_runtime_value("KR_ETF_LAA_IEF", _get_runtime_value("KR_ETF_AGG", "453540"))))
    _kr_qqq_default = _code_only(config.get("kr_etf_laa_qqq", _get_runtime_value("KR_ETF_LAA_QQQ", "133690")))
    _kr_shy_default = _code_only(config.get("kr_etf_laa_shy", _get_runtime_value("KR_ETF_LAA_SHY", "114470")))
    kr_spy = _code_only(st.session_state.get("pen_laa_spy", _kr_spy_default))
    kr_iwd = _code_only(st.session_state.get("pen_laa_iwd", _kr_iwd_default))
    kr_gld = _code_only(st.session_state.get("pen_laa_gld", _kr_gld_default))
    kr_ief = _code_only(st.session_state.get("pen_laa_ief", _kr_ief_default))
    kr_qqq = _code_only(st.session_state.get("pen_laa_qqq", _kr_qqq_default))
    kr_shy = _code_only(st.session_state.get("pen_laa_shy", _kr_shy_default))

    _show_laa_panel = ("LAA" in _active_strategies) and (_selected_panel == "LAA 전략 설정")
    if _show_laa_panel:
        with st.sidebar.expander("LAA 전략 설정", expanded=True):
            st.caption("LAA 전략 전용 설정")
            kr_spy = _sidebar_etf_code_input("SPY 신호 ETF", kr_spy, key="pen_laa_spy", disabled=IS_CLOUD)
            kr_iwd = _sidebar_etf_code_input("IWD 대체 ETF", kr_iwd, key="pen_laa_iwd", disabled=IS_CLOUD)
            kr_gld = _sidebar_etf_code_input("GLD 대체 ETF", kr_gld, key="pen_laa_gld", disabled=IS_CLOUD)
            kr_ief = _sidebar_etf_code_input("IEF 대체 ETF", kr_ief, key="pen_laa_ief", disabled=IS_CLOUD)
            kr_qqq = _sidebar_etf_code_input("QQQ 대체 ETF", kr_qqq, key="pen_laa_qqq", disabled=IS_CLOUD)
            kr_shy = _sidebar_etf_code_input("SHY 대체 ETF", kr_shy, key="pen_laa_shy", disabled=IS_CLOUD)

    _kr_etf_map = {"SPY": str(kr_spy), "IWD": str(kr_iwd), "GLD": str(kr_gld), "IEF": str(kr_ief), "QQQ": str(kr_qqq), "SHY": str(kr_shy)}

    # ── 듀얼모멘텀 전략 설정 ──
    # 백테스트에서 항상 사용 가능하도록 기본 설정은 전략 활성 여부와 무관하게 준비한다.
    _dm_offensive = ["SPY", "EFA"]
    _dm_defensive = ["AGG"]
    _dm_canary = ["BIL"]
    _dm_lookback = int(st.session_state.get("pen_dm_lookback", config.get("pen_dm_lookback", 12)))
    _dm_td = int(st.session_state.get("pen_dm_trading_days", config.get("pen_dm_trading_days", 22)))
    _dm_w1 = float(st.session_state.get("pen_dm_w1", config.get("pen_dm_w1", 12.0)))
    _dm_w3 = float(st.session_state.get("pen_dm_w3", config.get("pen_dm_w3", 4.0)))
    _dm_w6 = float(st.session_state.get("pen_dm_w6", config.get("pen_dm_w6", 2.0)))
    _dm_w12 = float(st.session_state.get("pen_dm_w12", config.get("pen_dm_w12", 1.0)))

    _dm_kr_spy_default = _code_only(config.get("pen_dm_kr_spy", config.get("pen_dm_agg_etf", _get_runtime_value("KR_ETF_SPY", "360750"))))
    _dm_kr_efa_default = _code_only(config.get("pen_dm_kr_efa", _get_runtime_value("KR_ETF_EFA", "453850")))
    _dm_kr_agg_default = _code_only(config.get("pen_dm_kr_agg", config.get("pen_dm_def_etf", _get_runtime_value("KR_ETF_AGG", "453540"))))
    _dm_kr_bil_default = _code_only(config.get("pen_dm_kr_bil", _get_runtime_value("KR_ETF_BIL", _get_runtime_value("KR_ETF_SHY", "114470"))))
    _dm_kr_spy = _code_only(st.session_state.get("pen_dm_kr_spy", _dm_kr_spy_default))
    _dm_kr_efa = _code_only(st.session_state.get("pen_dm_kr_efa", _dm_kr_efa_default))
    _dm_kr_agg = _code_only(st.session_state.get("pen_dm_kr_agg", _dm_kr_agg_default))
    _dm_kr_bil = _code_only(st.session_state.get("pen_dm_kr_bil", _dm_kr_bil_default))

    _show_dm_panel = ("듀얼모멘텀" in _active_strategies) and (_selected_panel == "듀얼모멘텀 설정")
    if _show_dm_panel:
        with st.sidebar.expander("듀얼모멘텀 설정", expanded=True):
            st.caption("듀얼모멘텀 전략 전용 설정")
            dmc1, dmc2 = st.columns(2)
            _dm_lookback = dmc1.number_input(
                "카나리아 룩백(개월)",
                min_value=1,
                max_value=24,
                value=int(_dm_lookback),
                step=1,
                key="pen_dm_lookback",
                help="기본 12개월 (절대 모멘텀 기준)",
            )
            _dm_td = dmc2.number_input(
                "월 환산 거래일",
                min_value=18,
                max_value=24,
                value=int(_dm_td),
                step=1,
                key="pen_dm_trading_days",
                help="기본 22일",
            )

            st.markdown("**모멘텀 가중치 (1M/3M/6M/12M)**")
            w1c, w3c, w6c, w12c = st.columns(4)
            _dm_w1 = w1c.number_input("1M", min_value=0.0, max_value=30.0, value=float(_dm_w1), step=0.5, key="pen_dm_w1")
            _dm_w3 = w3c.number_input("3M", min_value=0.0, max_value=30.0, value=float(_dm_w3), step=0.5, key="pen_dm_w3")
            _dm_w6 = w6c.number_input("6M", min_value=0.0, max_value=30.0, value=float(_dm_w6), step=0.5, key="pen_dm_w6")
            _dm_w12 = w12c.number_input("12M", min_value=0.0, max_value=30.0, value=float(_dm_w12), step=0.5, key="pen_dm_w12")

            st.markdown("**국내 ETF 설정 (시그널/실매매 공용)**")
            _dm_kr_spy = _sidebar_etf_code_input("공격 ETF 1", _dm_kr_spy, key="pen_dm_kr_spy", disabled=IS_CLOUD)
            _dm_kr_efa = _sidebar_etf_code_input("공격 ETF 2", _dm_kr_efa, key="pen_dm_kr_efa", disabled=IS_CLOUD)
            _dm_kr_agg = _sidebar_etf_code_input("방어 ETF", _dm_kr_agg, key="pen_dm_kr_agg", disabled=IS_CLOUD)
            _dm_kr_bil = _sidebar_etf_code_input("카나리아 ETF", _dm_kr_bil, key="pen_dm_kr_bil", disabled=IS_CLOUD)

    _dm_settings = {
        "offensive": _dm_offensive,
        "defensive": _dm_defensive,
        "canary": _dm_canary,
        "lookback": int(_dm_lookback),
        "trading_days_per_month": int(_dm_td),
        "momentum_weights": {
            "m1": float(_dm_w1),
            "m3": float(_dm_w3),
            "m6": float(_dm_w6),
            "m12": float(_dm_w12),
        },
        "kr_etf_map": {
            "SPY": str(_dm_kr_spy),
            "EFA": str(_dm_kr_efa),
            "AGG": str(_dm_kr_agg),
            "BIL": str(_dm_kr_bil),
        },
    }

    # ── 정적배분 전략 설정 ──
    _static_settings = {}
    if "정적배분" in _active_strategies:
        _sa_etf1 = str(st.session_state.get("pen_sa_etf1", config.get("pen_sa_etf1", "360750")))
        _sa_w1 = int(st.session_state.get("pen_sa_w1", config.get("pen_sa_w1", 60)))
        _sa_etf2 = str(st.session_state.get("pen_sa_etf2", config.get("pen_sa_etf2", "453540")))
        _sa_w2 = int(st.session_state.get("pen_sa_w2", config.get("pen_sa_w2", 40)))
        if _selected_panel == "정적배분 설정":
            with st.sidebar.expander("정적배분 설정", expanded=True):
                st.caption("ETF별 고정 비중으로 월간 리밸런싱")
                _sa_etf1 = st.text_input("ETF 1 코드", value=_sa_etf1, key="pen_sa_etf1")
                _sa_w1 = st.number_input("ETF 1 비중(%)", value=int(_sa_w1), min_value=0, max_value=100, step=5, key="pen_sa_w1")
                _sa_etf2 = st.text_input("ETF 2 코드", value=_sa_etf2, key="pen_sa_etf2")
                _sa_w2 = st.number_input("ETF 2 비중(%)", value=int(_sa_w2), min_value=0, max_value=100, step=5, key="pen_sa_w2")
        _static_settings = {"etfs": [{"code": _sa_etf1, "weight": _sa_w1}, {"code": _sa_etf2, "weight": _sa_w2}]}

    # ── 설정 저장 ──
    if not IS_CLOUD and st.sidebar.button("연금저축 설정 저장", key="pen_save_cfg"):
        new_cfg = config.copy()
        new_cfg["kis_pension_account_no"] = str(kis_acct).strip()
        new_cfg["kis_pension_prdt_cd"] = str(kis_prdt).strip() or "01"
        # 포트폴리오 저장
        new_cfg["pension_portfolio"] = _pen_port_edited.to_dict("records")
        # LAA 설정
        if kr_spy: new_cfg["kr_etf_laa_spy"] = kr_spy
        if kr_iwd: new_cfg["kr_etf_laa_iwd"] = kr_iwd
        if kr_gld: new_cfg["kr_etf_laa_gld"] = kr_gld
        if kr_ief: new_cfg["kr_etf_laa_ief"] = kr_ief
        if kr_qqq: new_cfg["kr_etf_laa_qqq"] = kr_qqq
        if kr_shy: new_cfg["kr_etf_laa_shy"] = kr_shy
        # 듀얼모멘텀 설정
        if _dm_settings:
            new_cfg["pen_dm_lookback"] = int(_dm_settings.get("lookback", 12))
            new_cfg["pen_dm_trading_days"] = int(_dm_settings.get("trading_days_per_month", 22))
            new_cfg["pen_dm_offensive"] = ",".join(_dm_settings.get("offensive", ["SPY", "EFA"]))
            new_cfg["pen_dm_defensive"] = ",".join(_dm_settings.get("defensive", ["AGG"]))
            new_cfg["pen_dm_canary"] = ",".join(_dm_settings.get("canary", ["BIL"]))
            _dmw = _dm_settings.get("momentum_weights", {})
            new_cfg["pen_dm_w1"] = float(_dmw.get("m1", 12.0))
            new_cfg["pen_dm_w3"] = float(_dmw.get("m3", 4.0))
            new_cfg["pen_dm_w6"] = float(_dmw.get("m6", 2.0))
            new_cfg["pen_dm_w12"] = float(_dmw.get("m12", 1.0))
            _dm_kr_map = _dm_settings.get("kr_etf_map", {})
            new_cfg["pen_dm_kr_spy"] = str(_dm_kr_map.get("SPY", "360750"))
            new_cfg["pen_dm_kr_efa"] = str(_dm_kr_map.get("EFA", "453850"))
            new_cfg["pen_dm_kr_agg"] = str(_dm_kr_map.get("AGG", "453540"))
            new_cfg["pen_dm_kr_bil"] = str(_dm_kr_map.get("BIL", "114470"))
            # 구버전 키 호환
            new_cfg["pen_dm_agg_etf"] = str(_dm_kr_map.get("SPY", "360750"))
            new_cfg["pen_dm_def_etf"] = str(_dm_kr_map.get("AGG", "453540"))
        # 정적배분 설정
        if _static_settings:
            new_cfg["pen_sa_etf1"] = _static_settings["etfs"][0]["code"] if _static_settings.get("etfs") else ""
            new_cfg["pen_sa_w1"] = _static_settings["etfs"][0]["weight"] if _static_settings.get("etfs") else 60
            new_cfg["pen_sa_etf2"] = _static_settings["etfs"][1]["code"] if len(_static_settings.get("etfs", [])) > 1 else ""
            new_cfg["pen_sa_w2"] = _static_settings["etfs"][1]["weight"] if len(_static_settings.get("etfs", [])) > 1 else 40
        save_config(new_cfg)
        st.sidebar.success("연금저축 설정을 저장했습니다.")

    if not (kis_ak and kis_sk and kis_acct):
        st.warning("KIS 연금저축 API 키와 계좌번호를 설정해 주세요.")
        return

    trader = KISTrader(is_mock=False)
    trader.app_key = kis_ak
    trader.app_secret = kis_sk
    trader.account_no = kis_acct
    trader.acnt_prdt_cd = kis_prdt

    # 토큰 캐싱 — ISA/연금저축 공용 토큰 재사용 (KIS 1분 1회 발급 제한 대응)
    _pen_token_key = f"pen_token_{kis_acct}"
    _kis_shared_token_key = f"kis_token_shared_{str(kis_ak or '')[-8:]}"
    _cached_pen = st.session_state.get(_pen_token_key)
    _cached_shared = st.session_state.get(_kis_shared_token_key)
    _tok = None
    if _cached_pen and (_cached_pen.get("expiry", 0) - time.time()) > 300:
        _tok = _cached_pen
    elif _cached_shared and (_cached_shared.get("expiry", 0) - time.time()) > 300:
        _tok = _cached_shared

    if _tok:
        trader.access_token = _tok.get("token")
        trader.token_expiry = float(_tok.get("expiry", 0))
        st.session_state[_pen_token_key] = {"token": trader.access_token, "expiry": trader.token_expiry}
    else:
        if not trader.auth():
            st.error("KIS 인증에 실패했습니다. API 키/계좌를 확인해 주세요.")
            return
        _new_tok = {"token": trader.access_token, "expiry": trader.token_expiry}
        st.session_state[_pen_token_key] = _new_tok
        st.session_state[_kis_shared_token_key] = _new_tok

    pen_bal_key = f"pension_balance_cache_{kis_acct}_{kis_prdt}"

    # 연금저축 데이터 로딩 정책: 로컬 파일 우선 + 부족 시 API 보강
    _pen_local_first_raw = str(config.get("pen_local_first", config.get("pen_local_data_only", "1"))).strip().lower()
    _pen_local_first = _pen_local_first_raw not in ("0", "false", "no", "off")
    _pen_api_fallback_raw = str(config.get("pen_api_fallback", "1")).strip().lower()
    _pen_api_fallback = _pen_api_fallback_raw not in ("0", "false", "no", "off")

    # 연금저축 모드 내 KIS 호출 캐시 (재렌더링 시 중복 호출 최소화)
    _pen_cache_key = f"pension_api_cache_{kis_acct}_{kis_prdt}"
    _pen_cache = st.session_state.get(_pen_cache_key)
    if not isinstance(_pen_cache, dict):
        _pen_cache = {"daily": {}, "price": {}, "orderbook": {}}
        st.session_state[_pen_cache_key] = _pen_cache

    def _cache_put(_bucket: str, _key: str, _value):
        _store = _pen_cache.setdefault(_bucket, {})
        _store[_key] = {"ts": float(time.time()), "value": _value}
        if len(_store) > 120:
            _oldest = sorted(_store.items(), key=lambda kv: kv[1].get("ts", 0.0))[:20]
            for _k, _ in _oldest:
                _store.pop(_k, None)

    def _cache_get(_bucket: str, _key: str, _ttl_sec: float):
        _store = _pen_cache.get(_bucket, {})
        _hit = _store.get(_key)
        if not _hit:
            return None
        _age = float(time.time()) - float(_hit.get("ts", 0.0))
        if _age > float(_ttl_sec):
            return None
        return _hit.get("value")

    def _normalize_ohlcv_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return df
        out = df.copy().sort_index()
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        has_upper = any(k in out.columns for k in rename_map)
        if has_upper:
            out = out.rename(columns=rename_map)
        return out

    def _get_pen_daily_chart(_code: str, count: int = 420, end_date: str | None = None, use_disk_cache: bool = False):
        _c = _code_only(_code)
        if not _c:
            return None
        _cnt = int(max(1, count))
        _end = str(end_date or "")
        _cache_k = f"{_c}|{_cnt}|{_end}|{int(bool(use_disk_cache))}"
        _ttl = 1800.0 if _cnt >= 3000 else 300.0
        _cached = _cache_get("daily", _cache_k, _ttl)
        if isinstance(_cached, pd.DataFrame) and not _cached.empty:
            return _cached.copy()

        _df = None
        if use_disk_cache or _pen_local_first:
            _df = data_cache.get_kis_domestic_local_first(
                trader if _pen_api_fallback else None,
                _c,
                count=_cnt,
                end_date=_end or None,
                allow_api_fallback=bool(_pen_api_fallback),
            )
        else:
            try:
                _df = trader.get_daily_chart(_c, count=_cnt, end_date=_end or None)
            except Exception:
                _df = None

        _df = _normalize_ohlcv_df(_df)
        if isinstance(_df, pd.DataFrame) and not _df.empty:
            try:
                _df.index = pd.to_datetime(_df.index)
            except Exception:
                pass
            if _end:
                try:
                    _end_ts = pd.to_datetime(_end)
                    _df = _df[_df.index <= _end_ts]
                except Exception:
                    pass
            if len(_df) > _cnt:
                _df = _df.tail(_cnt)

        if isinstance(_df, pd.DataFrame) and not _df.empty:
            _cache_put("daily", _cache_k, _df)
            return _df.copy()
        return _df

    def _get_pen_current_price(_code: str, ttl_sec: float = 8.0) -> float:
        _c = _code_only(_code)
        if not _c:
            return 0.0
        _cached = _cache_get("price", _c, ttl_sec)
        if _cached is not None:
            return float(_cached or 0.0)
        _p = 0.0
        _ref_df = _get_pen_daily_chart(_c, count=3, use_disk_cache=True if _pen_local_first else False)
        if _ref_df is not None and not _ref_df.empty:
            if "close" in _ref_df.columns:
                _p = float(_ref_df["close"].iloc[-1] or 0.0)
            elif "Close" in _ref_df.columns:
                _p = float(_ref_df["Close"].iloc[-1] or 0.0)
        if _p <= 0 and _pen_api_fallback:
            try:
                _p = float(trader.get_current_price(_c) or 0.0)
            except Exception:
                _p = 0.0
        _cache_put("price", _c, float(_p if _p > 0 else 0.0))
        return float(_p if _p > 0 else 0.0)

    def _get_pen_orderbook(_code: str, ttl_sec: float = 8.0):
        _c = _code_only(_code)
        if not _c:
            return None
        _cached = _cache_get("orderbook", _c, ttl_sec)
        if _cached is not None:
            return _cached
        _ob = None
        try:
            _ob = trader.get_orderbook(_c)
        except Exception:
            _ob = None
        _cache_put("orderbook", _c, _ob)
        return _ob

    tab_p1, tab_p2, tab_p3, tab_p4, tab_p5, tab_p6 = st.tabs([
        "🚀 실시간 모니터링", "🧪 백테스트", "🛒 수동 주문",
        "📖 전략 가이드", "📋 주문방식", "💳 수수료/세금"
    ])

    # ══════════════════════════════════════════════════════════════
    # Tab 1: 실시간 모니터링
    # ══════════════════════════════════════════════════════════════
    with tab_p1:
        _strat_summary = " + ".join([f"{r['strategy']} {r['weight']}%" for _, r in _pen_port_edited.iterrows()]) if not _pen_port_edited.empty else "LAA 100%"
        st.header("포트폴리오 모니터링")
        st.caption(f"구성: {_strat_summary}")

        # 잔고 표시 — F5 새로고침 시 자동 조회, 이후 캐시 사용
        if pen_bal_key not in st.session_state:
            with st.spinner("연금저축 잔고를 조회하는 중..."):
                st.session_state[pen_bal_key] = trader.get_balance()

        rcol1, rcol2 = st.columns([1, 5])
        with rcol1:
            if st.button("잔고 새로고침", key="pen_refresh_balance"):
                with st.spinner("연금저축 잔고를 다시 조회하는 중..."):
                    st.session_state[pen_bal_key] = trader.get_balance()
                st.session_state.pop("pen_signal_result", None)
                st.session_state.pop("pen_signal_params", None)
                st.session_state.pop("pen_dm_signal_result", None)
                st.session_state.pop("pen_dm_signal_params", None)
                st.rerun()

        bal = st.session_state.get(pen_bal_key)
        if not bal:
            st.warning("잔고 조회에 실패했습니다. (응답 None — 네트워크 또는 인증 오류)")
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt}")
        elif bal.get("error"):
            st.error(f"잔고 조회 API 오류: [{bal.get('msg_cd', '')}] {bal.get('msg1', '알 수 없는 오류')}")
            st.info(f"계좌: {kis_acct} / 상품코드: {kis_prdt} / rt_cd: {bal.get('rt_cd', '')}")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []
            total_eval = float(bal.get("total_eval", 0.0)) or (cash + sum(float(h.get("eval_amt", 0.0)) for h in holdings))
            stock_eval = total_eval - cash

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("예수금", f"{cash:,.0f} KRW")
            m2.metric("주식 평가", f"{stock_eval:,.0f} KRW")
            m3.metric("총 평가", f"{total_eval:,.0f} KRW")
            m4.metric("보유 종목 수", f"{len(holdings)}")

            if holdings:
                df_h = pd.DataFrame(holdings)
                cols = [c for c in ["code", "name", "qty", "avg_price", "cur_price", "eval_amt", "pnl_rate"] if c in df_h.columns]
                st.dataframe(df_h[cols], use_container_width=True, hide_index=True)

        st.divider()

        # LAA 시그널 자동 계산 (버튼 없이 항상 표시)
        pen_sig_params = {
            "acct": str(kis_acct),
            "prdt": str(kis_prdt),
            "kr_spy": str(kr_spy),
            "kr_iwd": str(kr_iwd),
            "kr_gld": str(kr_gld),
            "kr_ief": str(kr_ief),
            "kr_qqq": str(kr_qqq),
            "kr_shy": str(kr_shy),
            "bt_start": str(_pen_bt_start_raw),
        }

        def _compute_pen_signal_result():
            tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
            price_data = {}
            source_map = {
                "SPY": _code_only(kr_spy or kr_iwd or "360750"),
                "IWD": _code_only(kr_iwd),
                "GLD": _code_only(kr_gld),
                "IEF": _code_only(kr_ief),
                "QQQ": _code_only(kr_qqq),
                "SHY": _code_only(kr_shy),
            }
            _today = pd.Timestamp.now().date()
            for ticker in tickers:
                _code = _code_only(source_map.get(ticker, ""))
                if not _code:
                    return {"error": f"{ticker} 국내 ETF 코드가 비어 있습니다."}
                df_t = _get_pen_daily_chart(_code, count=420)
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

            strategy = LAAStrategy(settings={"kr_etf_map": _kr_etf_map})
            signal = strategy.analyze(price_data)
            if not signal:
                return {"error": "LAA 분석에 실패했습니다."}

            # 리스크 판단 보조 차트: TIGER 미국S&P500(기본 360750) + 200일선 + 이격도
            _risk_chart_code = _code_only((_kr_etf_map or {}).get("IWD", "")) or "360750"
            _risk_df = _get_pen_daily_chart(str(_risk_chart_code), count=800)
            if _risk_df is not None and not _risk_df.empty:
                _risk_df = _risk_df.copy().sort_index()
                if "close" not in _risk_df.columns and "Close" in _risk_df.columns:
                    _risk_df["close"] = _risk_df["Close"]
                if "close" in _risk_df.columns:
                    _risk_df["ma200"] = _risk_df["close"].rolling(200).mean()
                    _risk_df["divergence"] = np.where(
                        _risk_df["ma200"] > 0,
                        (_risk_df["close"] / _risk_df["ma200"] - 1.0) * 100.0,
                        np.nan,
                    )
                else:
                    _risk_df = None

            bal_local = st.session_state.get(pen_bal_key) or trader.get_balance() or {"cash": 0.0, "holdings": []}
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

            price_cache = {}

            def _resolve_etf_price(_code: str, _cur_v: float, _cur_q: int) -> float:
                _key = str(_code).strip()
                if not _key:
                    return 0.0
                if _key in price_cache:
                    return float(price_cache[_key])

                _p = 0.0
                try:
                    _p = float(_get_pen_current_price(_key) or 0.0)
                except Exception:
                    _p = 0.0

                if _p <= 0:
                    _p = float(current_prices.get(_key, 0.0) or 0.0)

                if _p <= 0 and _cur_q > 0:
                    _p = float(_cur_v) / float(_cur_q)

                if _p <= 0:
                    try:
                        _ch = _get_pen_daily_chart(_key, count=5)
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

            bt_price_data = {}
            for ticker in tickers:
                _code = _code_only(source_map.get(ticker, ""))
                if not _code:
                    continue
                _df_bt = _get_pen_daily_chart(_code, count=3000, use_disk_cache=True)
                if _df_bt is None or _df_bt.empty:
                    bt_price_data = {}
                    break
                _df_bt = _df_bt.copy().sort_index()
                if "close" not in _df_bt.columns and "Close" in _df_bt.columns:
                    _df_bt["close"] = _df_bt["Close"]
                if "close" not in _df_bt.columns:
                    bt_price_data = {}
                    break
                if len(_df_bt) >= 2:
                    _last_dt = pd.to_datetime(_df_bt.index[-1]).date()
                    if _last_dt >= _today:
                        _df_bt = _df_bt.iloc[:-1]
                _df_bt = _df_bt[_df_bt.index >= _pen_bt_start_ts]
                if _df_bt.empty:
                    bt_price_data = {}
                    break
                bt_price_data[ticker] = _df_bt

            bt_result = None
            bt_benchmark_series = None
            bt_benchmark_label = "SPY Buy & Hold"
            if bt_price_data:
                bt_result = strategy.run_backtest(
                    bt_price_data,
                    initial_balance=10_000_000.0,
                    fee=0.0002,
                )
                _bm_ticker = "SPY"
                _bm_series = _normalize_numeric_series(
                    bt_price_data.get(_bm_ticker),
                    preferred_cols=("close", "Close"),
                )
                if not _bm_series.empty:
                    bt_benchmark_series = _bm_series
                    _bm_code = _code_only(source_map.get(_bm_ticker, ""))
                    if _bm_code:
                        bt_benchmark_label = f"{_fmt_etf_code_name(_bm_code)} Buy & Hold"

            return {
                "signal": signal,
                "action": action,
                "source_map": source_map,
                "alloc_df": pd.DataFrame(rows),
                "price_data": price_data,
                "risk_chart_df": _risk_df,
                "risk_chart_code": _risk_chart_code,
                "balance": bal_local,
                "bt_start_date": str(_pen_bt_start_raw),
                "bt_result": bt_result,
                "bt_benchmark_series": bt_benchmark_series,
                "bt_benchmark_label": bt_benchmark_label,
            }

        if st.session_state.get("pen_signal_result") is None or st.session_state.get("pen_signal_params") != pen_sig_params:
            with st.spinner("LAA 시그널을 자동 계산하는 중입니다..."):
                st.session_state["pen_signal_result"] = _compute_pen_signal_result()
                st.session_state["pen_signal_params"] = pen_sig_params
                _pen_res = st.session_state["pen_signal_result"]
                if isinstance(_pen_res, dict) and _pen_res.get("balance"):
                    st.session_state[pen_bal_key] = _pen_res["balance"]

        res = st.session_state.get("pen_signal_result")

        # 듀얼모멘텀 시그널 자동 계산 (디스플레이 전에 먼저 계산)
        _dm_res = None
        if "듀얼모멘텀" in _active_strategies and _dm_settings:
            _dm_weights = _dm_settings.get("momentum_weights", {})
            _dm_params = {
                "acct": str(kis_acct),
                "prdt": str(kis_prdt),
                "off": tuple(_dm_settings.get("offensive", [])),
                "def": tuple(_dm_settings.get("defensive", [])),
                "canary": tuple(_dm_settings.get("canary", [])),
                "lookback": int(_dm_settings.get("lookback", 12)),
                "td": int(_dm_settings.get("trading_days_per_month", 22)),
                "w1": float(_dm_weights.get("m1", 12.0)),
                "w3": float(_dm_weights.get("m3", 4.0)),
                "w6": float(_dm_weights.get("m6", 2.0)),
                "w12": float(_dm_weights.get("m12", 1.0)),
                "kr_spy": str((_dm_settings.get("kr_etf_map", {}) or {}).get("SPY", "")),
                "kr_efa": str((_dm_settings.get("kr_etf_map", {}) or {}).get("EFA", "")),
                "kr_agg": str((_dm_settings.get("kr_etf_map", {}) or {}).get("AGG", "")),
                "kr_bil": str((_dm_settings.get("kr_etf_map", {}) or {}).get("BIL", "")),
                "bt_start": str(_pen_bt_start_raw),
            }

            def _compute_dm_signal_result():
                _dm_tickers = []
                for _tk in (_dm_settings.get("offensive", []) + _dm_settings.get("defensive", []) + _dm_settings.get("canary", [])):
                    _u = str(_tk).strip().upper()
                    if _u and _u not in _dm_tickers:
                        _dm_tickers.append(_u)

                if not _dm_tickers:
                    return {"error": "듀얼모멘텀 티커 설정이 비어 있습니다."}

                _dm_price_data = {}
                _dm_source_map = {}
                _dm_kr_map = _dm_settings.get("kr_etf_map", {}) or {}
                _today = pd.Timestamp.now().date()
                for _ticker in _dm_tickers:
                    _kr_code = str(_dm_kr_map.get(_ticker, "")).strip()
                    if not _kr_code:
                        return {"error": f"{_ticker} 국내 ETF 매핑이 없습니다."}
                    _df_t = _get_pen_daily_chart(_kr_code, count=420)
                    if _df_t is None or _df_t.empty:
                        return {"error": f"{_ticker} ({_kr_code}) 국내 데이터 조회 실패"}
                    _df_t = _df_t.copy().sort_index()
                    if "close" not in _df_t.columns and "Close" in _df_t.columns:
                        _df_t["close"] = _df_t["Close"]
                    if "close" not in _df_t.columns:
                        return {"error": f"{_ticker} ({_kr_code}) 종가 컬럼이 없습니다."}

                    if len(_df_t) >= 2:
                        _last_dt = pd.to_datetime(_df_t.index[-1]).date()
                        if _last_dt >= _today:
                            _df_t = _df_t.iloc[:-1]
                    if _df_t is None or _df_t.empty:
                        return {"error": f"{_ticker} ({_kr_code}) 전일 종가 데이터가 없습니다."}

                    _dm_price_data[_ticker] = _df_t
                    _dm_source_map[_ticker] = _kr_code

                _dm_strategy = DualMomentumStrategy(settings=_dm_settings)
                _dm_sig = _dm_strategy.analyze(_dm_price_data)
                if not _dm_sig:
                    return {"error": "듀얼모멘텀 분석에 실패했습니다."}

                _dm_td_local = int(_dm_settings.get("trading_days_per_month", 22))
                _dm_lb_local = int(_dm_settings.get("lookback", 12))
                _dm_w_local = _dm_settings.get("momentum_weights", {}) or {}
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
                    _role = "공격" if _tk in _dm_settings.get("offensive", []) else ("방어" if _tk in _dm_settings.get("defensive", []) else "카나리아")
                    _last_close = float(_dfm["close"].iloc[-1])
                    _last_date = pd.to_datetime(_dfm.index[-1]).date()
                    _ref_dates.append(_last_date)
                    _score_rows.append({"티커": _tk, "모멘텀 점수": float(_score)})
                    _mom_rows.append({
                        "역할": _role,
                        "티커": _tk,
                        "국내 ETF": _fmt_etf_code_name(_dm_source_map.get(_tk, "")),
                        "기준 종가일": str(_last_date),
                        "기준 종가": f"{_last_close:,.0f}",
                        "1개월(%)": round(_m1 * 100.0, 2),
                        "3개월(%)": round(_m3 * 100.0, 2),
                        "6개월(%)": round(_m6 * 100.0, 2),
                        "12개월(%)": round(_m12 * 100.0, 2),
                        _lookback_col: round(_lb_ret * 100.0, 2),
                        "가중 모멘텀 점수": round(_score, 6),
                    })

                _ref_date = min(_ref_dates) if _ref_dates else None
                _lag_days = int((_today - _ref_date).days) if _ref_date else None

                _bal_local = st.session_state.get(pen_bal_key) or trader.get_balance() or {"cash": 0.0, "holdings": []}
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
                    _v = str((_dm_settings.get("kr_etf_map", {}) or {}).get(_k, "")).strip()
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
                        _pen_port_edited.loc[_pen_port_edited["strategy"] == "듀얼모멘텀", "weight"],
                        errors="coerce",
                    ).fillna(0).sum()
                )
                _dm_weight_pct = max(0.0, min(100.0, _dm_weight_pct))
                _sleeve_eval = _total_eval_local * (_dm_weight_pct / 100.0)
                _target_ticker = str(_dm_sig.get("target_ticker", "")).strip().upper()

                _expected_rows = []
                _alloc_tickers = []
                for _k in (_dm_settings.get("offensive", []) + _dm_settings.get("defensive", [])):
                    _ku = str(_k).strip().upper()
                    if _ku and _ku not in _alloc_tickers:
                        _alloc_tickers.append(_ku)
                for _tk in _alloc_tickers:
                    _code = str(_dm_kr_map.get(_tk, "")).strip()
                    _px = 0.0
                    if _tk in _dm_price_data and len(_dm_price_data[_tk]) > 0:
                        _px = float(_dm_price_data[_tk]["close"].iloc[-1])
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

                _dm_bt_price_data = {}
                for _tk in _dm_tickers:
                    _code = str(_dm_kr_map.get(_tk, "")).strip()
                    if not _code:
                        continue
                    _df_bt = _get_pen_daily_chart(_code, count=3000, use_disk_cache=True)
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
                    _df_bt = _df_bt[_df_bt.index >= _pen_bt_start_ts]
                    if _df_bt.empty:
                        _dm_bt_price_data = {}
                        break
                    _dm_bt_price_data[_tk] = _df_bt

                _dm_bt_result = None
                _dm_bm_series = None
                _dm_bm_label = "SPY Buy & Hold"
                if _dm_bt_price_data:
                    _dm_bt_result = _dm_strategy.run_backtest(
                        _dm_bt_price_data,
                        initial_balance=10_000_000.0,
                        fee=0.0002,
                    )
                    _dm_bm_ticker = ""
                    for _t in (_dm_settings.get("offensive", []) or []):
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
                        _dm_bm_code = _code_only(str(_dm_kr_map.get(_dm_bm_ticker, "")))
                        if _dm_bm_code:
                            _dm_bm_label = f"{_fmt_etf_code_name(_dm_bm_code)} Buy & Hold"

                return {
                    "signal": _dm_sig,
                    "action": _action,
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
                    "bt_start_date": str(_pen_bt_start_raw),
                    "bt_result": _dm_bt_result,
                    "bt_benchmark_series": _dm_bm_series,
                    "bt_benchmark_label": _dm_bm_label,
                }

            if st.session_state.get("pen_dm_signal_result") is None or st.session_state.get("pen_dm_signal_params") != _dm_params:
                with st.spinner("듀얼모멘텀 시그널을 자동 계산하는 중입니다..."):
                    st.session_state["pen_dm_signal_result"] = _compute_dm_signal_result()
                    st.session_state["pen_dm_signal_params"] = _dm_params
                    _dm_res_cache = st.session_state["pen_dm_signal_result"]
                    if isinstance(_dm_res_cache, dict) and _dm_res_cache.get("balance"):
                        st.session_state[pen_bal_key] = _dm_res_cache["balance"]

            _dm_res = st.session_state.get("pen_dm_signal_result")

        # ──────────────────────────────────────────────────────────
        # 섹션 2: 전체 포트폴리오 합산
        # ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("전체 포트폴리오 합산")
        _combined_rows = {}

        _laa_action = None
        if res and not res.get("error"):
            _laa_action = res.get("action")
            _laa_df = res.get("alloc_df")
            if isinstance(_laa_df, pd.DataFrame) and not _laa_df.empty:
                for _, row in _laa_df.iterrows():
                    code = str(row.get("ETF 코드", "")).strip()
                    if not code:
                        continue
                    if code not in _combined_rows:
                        _combined_rows[code] = {
                            "ETF": row.get("ETF", _fmt_etf_code_name(code)),
                            "현재수량(주)": int(row.get("현재수량(주)", 0)),
                            "LAA 목표(주)": 0, "DM 목표(주)": 0,
                        }
                    _combined_rows[code]["LAA 목표(주)"] = int(row.get("목표수량(주)", 0))
                    _combined_rows[code]["현재수량(주)"] = max(
                        _combined_rows[code]["현재수량(주)"], int(row.get("현재수량(주)", 0)))

        _dm_action_val = None
        if _dm_res and not _dm_res.get("error"):
            _dm_action_val = _dm_res.get("action")
            _dm_df = _dm_res.get("expected_rebalance_df")
            if isinstance(_dm_df, pd.DataFrame) and not _dm_df.empty:
                for _, row in _dm_df.iterrows():
                    code = str(row.get("ETF 코드", "")).strip()
                    if not code:
                        continue
                    if code not in _combined_rows:
                        _combined_rows[code] = {
                            "ETF": row.get("국내 ETF", _fmt_etf_code_name(code)),
                            "현재수량(주)": int(row.get("현재수량(주)", 0)),
                            "LAA 목표(주)": 0, "DM 목표(주)": 0,
                        }
                    _combined_rows[code]["DM 목표(주)"] = int(row.get("목표수량(주,버림)", 0))
                    _combined_rows[code]["현재수량(주)"] = max(
                        _combined_rows[code]["현재수량(주)"], int(row.get("현재수량(주)", 0)))

        if _combined_rows:
            _combo_list = []
            _total_buy_count = 0
            _total_sell_count = 0
            _rebal_strategies = 0
            if _laa_action == "REBALANCE":
                _rebal_strategies += 1
            if _dm_action_val in ("REBALANCE", "BUY"):
                _rebal_strategies += 1

            _bal_combo = st.session_state.get(pen_bal_key) or {}
            _holdings_combo = _bal_combo.get("holdings", []) or []
            _cash_combo = float(_bal_combo.get("cash", 0.0) or 0.0)
            _total_eval_combo = float(_bal_combo.get("total_eval", 0.0) or 0.0)
            if _total_eval_combo <= 0:
                _total_eval_combo = _cash_combo + sum(float(_h.get("eval_amt", 0.0) or 0.0) for _h in _holdings_combo)
            _total_eval_combo = max(_total_eval_combo, 1.0)

            _hold_eval_map = {}
            _hold_px_map = {}
            for _h in _holdings_combo:
                _code = str(_h.get("code", "")).strip()
                if not _code:
                    continue
                _hold_eval_map[_code] = _hold_eval_map.get(_code, 0.0) + float(_h.get("eval_amt", 0.0) or 0.0)
                _cur_px = float(_h.get("cur_price", 0.0) or 0.0)
                if _cur_px > 0:
                    _hold_px_map[_code] = _cur_px

            _combo_price_cache = {}

            def _resolve_combo_price(_code: str, _cur_qty: int, _cur_eval: float) -> float:
                _k = str(_code).strip()
                if not _k:
                    return 0.0
                if _k in _combo_price_cache:
                    return float(_combo_price_cache[_k])

                _p = float(_hold_px_map.get(_k, 0.0) or 0.0)
                if _p <= 0:
                    try:
                        _p = float(_get_pen_current_price(_k) or 0.0)
                    except Exception:
                        _p = 0.0

                if _p <= 0 and _cur_qty > 0:
                    _p = float(_cur_eval) / float(_cur_qty)

                if _p <= 0:
                    try:
                        _ch = _get_pen_daily_chart(_k, count=5)
                        if _ch is not None and not _ch.empty:
                            if "close" in _ch.columns:
                                _p = float(_ch["close"].iloc[-1])
                            elif "Close" in _ch.columns:
                                _p = float(_ch["Close"].iloc[-1])
                    except Exception:
                        _p = 0.0

                _combo_price_cache[_k] = float(_p if _p > 0 else 0.0)
                return float(_combo_price_cache[_k])

            for code, info in _combined_rows.items():
                _total_target = info["LAA 목표(주)"] + info["DM 목표(주)"]
                _cur = info["현재수량(주)"]
                _buy = max(_total_target - _cur, 0)
                _sell = max(_cur - _total_target, 0)
                _cur_eval = float(_hold_eval_map.get(code, 0.0) or 0.0)
                _px = float(_resolve_combo_price(code, _cur, _cur_eval))
                if _cur_eval <= 0 and _px > 0 and _cur > 0:
                    _cur_eval = _px * _cur
                _target_eval = float(_total_target) * _px if _px > 0 else 0.0
                _cur_weight = (_cur_eval / _total_eval_combo) * 100.0
                _target_weight = (_target_eval / _total_eval_combo) * 100.0
                if _buy > 0:
                    _order_status = "매수"
                    _total_buy_count += 1
                elif _sell > 0:
                    _order_status = "매도"
                    _total_sell_count += 1
                else:
                    _order_status = "유지"
                _combo_list.append({
                    "ETF": info["ETF"],
                    "현재수량(주)": _cur,
                    "현재 비중(%)": round(_cur_weight, 2),
                    "LAA 목표(주)": info["LAA 목표(주)"],
                    "DM 목표(주)": info["DM 목표(주)"],
                    "합산 목표(주)": _total_target,
                    "목표 비중(%)": round(_target_weight, 2),
                    "매수 예정(주)": _buy if _buy > 0 else 0,
                    "매도 예정(주)": _sell if _sell > 0 else 0,
                    "주문 상태": _order_status,
                })

            # 다음 리밸런싱 예정일 계산 (매월 마지막 영업일)
            from datetime import date as _d_date, timedelta as _d_td
            import calendar as _cal
            _today = _d_date.today()
            def _next_rebal_date(_ref):
                """매월 마지막 영업일(월~금) 계산. _ref 이후 가장 가까운 날짜 반환."""
                y, m = _ref.year, _ref.month
                last_day = _cal.monthrange(y, m)[1]
                dt = _d_date(y, m, last_day)
                while dt.weekday() >= 5:  # 토(5)/일(6) → 금요일로
                    dt -= _d_td(days=1)
                if dt >= _ref:
                    return dt
                # 이미 지남 → 다음 달
                if m == 12:
                    y, m = y + 1, 1
                else:
                    m += 1
                last_day = _cal.monthrange(y, m)[1]
                dt = _d_date(y, m, last_day)
                while dt.weekday() >= 5:
                    dt -= _d_td(days=1)
                return dt

            _next_rebal = _next_rebal_date(_today)
            _days_left = (_next_rebal - _today).days

            _sm1, _sm2, _sm3, _sm4 = st.columns(4)
            _sm1.metric("다음 리밸런싱", f"{_next_rebal.strftime('%Y-%m-%d')}", delta=f"D-{_days_left}" if _days_left > 0 else "오늘")
            _sm2.metric("리밸런싱 필요 전략", f"{_rebal_strategies}개")
            _sm3.metric("매수 예정 종목", f"{_total_buy_count}개")
            _sm4.metric("매도 예정 종목", f"{_total_sell_count}개")
            st.dataframe(pd.DataFrame(_combo_list), use_container_width=True, hide_index=True)
        else:
            st.info("시그널 계산 결과가 없습니다. 잔고를 새로고침해주세요.")

        # ──────────────────────────────────────────────────────────
        # 섹션 3: LAA 전략 포트폴리오
        # ──────────────────────────────────────────────────────────
        st.divider()
        import streamlit.components.v1 as components

        _laa_hdr_col1, _laa_hdr_col2 = st.columns([4, 1])
        with _laa_hdr_col1:
            st.subheader("LAA 전략 포트폴리오")
        with _laa_hdr_col2:
            if st.button("📖 전략 가이드", key="pen_laa_guide_btn"):
                components.html("""
                <script>
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                for (let t of tabs) {
                    if (t.textContent.includes('전략 가이드')) { t.click(); break; }
                }
                </script>
                """, height=0)

        if res:
            if res.get("error"):
                st.error(res["error"])
            else:
                sig = res["signal"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("리스크 상태", "공격 (Risk-On)" if sig.get("risk_on") else "방어 (Risk-Off)")
                c2.metric("리스크 자산", sig["selected_risk_asset"])
                c3.metric("국내 ETF", _fmt_etf_code_name(sig["selected_risk_kr_code"]))
                c4.metric("권장 동작", res["action"])
                st.info(sig.get("reason", ""))

                if "source_map" in res:
                    st.caption("시그널 데이터(국내): " + ", ".join([f"{k}→{v}" for k, v in res["source_map"].items()]))

                _risk_chart_df = res.get("risk_chart_df")
                _risk_chart_code = str(res.get("risk_chart_code", "")).strip() or "360750"
                if isinstance(_risk_chart_df, pd.DataFrame) and not _risk_chart_df.empty and "ma200" in _risk_chart_df.columns:
                    _plot_df = _risk_chart_df.dropna(subset=["ma200"]).copy()
                    if not _plot_df.empty:
                        _last = _plot_df.iloc[-1]
                        _last_close = float(_last.get("close", 0.0))
                        _last_ma200 = float(_last.get("ma200", 0.0))
                        _last_div = float(_last.get("divergence", 0.0))
                        _mode_label = "공격 (Risk-On)" if sig.get("risk_on") else "방어 (Risk-Off)"
                        _mode_color = "#16a34a" if sig.get("risk_on") else "#dc2626"

                        st.subheader("TIGER 미국S&P500 + 200일선")
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("현재가", f"{_last_close:,.0f} KRW")
                        cc2.metric("200일선", f"{_last_ma200:,.0f} KRW")
                        cc3.metric("이격도(200일)", f"{_last_div:+.2f}%")
                        st.caption(f"현재 모드: {_mode_label} | 기준 ETF: {_fmt_etf_code_name(_risk_chart_code)}")

                        fig_risk = go.Figure()
                        fig_risk.add_trace(go.Scatter(
                            x=_plot_df.index, y=_plot_df["close"],
                            name=f"{_fmt_etf_code_name(_risk_chart_code)} 종가",
                            line=dict(color="royalblue", width=2),
                        ))
                        fig_risk.add_trace(go.Scatter(
                            x=_plot_df.index, y=_plot_df["ma200"],
                            name="200일 이동평균",
                            line=dict(color="orange", width=2, dash="dash"),
                        ))
                        fig_risk.add_annotation(
                            xref="paper", yref="paper", x=0.01, y=0.98,
                            text=f"현재 모드: {_mode_label}",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor=_mode_color,
                            font=dict(color=_mode_color, size=13),
                        )
                        fig_risk.update_layout(
                            height=430,
                            xaxis_title="날짜",
                            yaxis_title="가격 (KRW)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.06),
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)

                        st.subheader("이격도 차트")
                        fig_div = go.Figure()
                        fig_div.add_trace(go.Scatter(
                            x=_plot_df.index, y=_plot_df["divergence"],
                            name="이격도(%)",
                            line=dict(color="#7c3aed", width=2),
                        ))
                        fig_div.add_hline(y=0, line_dash="dash", line_color="#374151")
                        _max_div = float(np.nanmax(_plot_df["divergence"].values)) if len(_plot_df) else 0.0
                        _min_div = float(np.nanmin(_plot_df["divergence"].values)) if len(_plot_df) else 0.0
                        if _max_div > 0:
                            fig_div.add_hrect(y0=0, y1=_max_div, fillcolor="rgba(22,163,74,0.10)", line_width=0)
                        if _min_div < 0:
                            fig_div.add_hrect(y0=_min_div, y1=0, fillcolor="rgba(220,38,38,0.10)", line_width=0)
                        fig_div.add_annotation(
                            xref="paper", yref="paper", x=0.01, y=0.98,
                            text=f"현재 모드: {_mode_label} ({_last_div:+.2f}%)",
                            showarrow=False,
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor=_mode_color,
                            font=dict(color=_mode_color, size=13),
                        )
                        fig_div.update_layout(
                            height=320,
                            xaxis_title="날짜",
                            yaxis_title="이격도 (%)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.06),
                        )
                        st.plotly_chart(fig_div, use_container_width=True)

                if "alloc_df" in res and isinstance(res["alloc_df"], pd.DataFrame) and not res["alloc_df"].empty:
                    st.subheader("목표 배분 vs 현재 보유")
                    st.caption("매매수량은 비중 차이 기준 자동 계산되며, 소수점 주식은 모두 버림 처리합니다.")
                    st.dataframe(res["alloc_df"], use_container_width=True, hide_index=True)

                _laa_bt = res.get("bt_result")
                if isinstance(_laa_bt, dict):
                    _laa_eq = _laa_bt.get("equity_df")
                    if isinstance(_laa_eq, pd.DataFrame) and not _laa_eq.empty and "equity" in _laa_eq.columns:
                        _laa_m = _laa_bt.get("metrics", {}) or {}
                        _laa_start = str(res.get("bt_start_date", _pen_bt_start_raw))
                        _laa_final_eq = float(_laa_m.get("final_equity", _laa_eq["equity"].iloc[-1]))
                        _laa_total_ret = float(_laa_m.get("total_return", 0.0))
                        _laa_mdd = float(_laa_m.get("mdd", 0.0))
                        _laa_cagr = float(_laa_m.get("cagr", 0.0))

                        _bal_valid = isinstance(bal, dict) and not bal.get("error")
                        _actual_cash_v = float(bal.get("cash", 0.0)) if _bal_valid else 0.0
                        _actual_hlds = (bal.get("holdings", []) or []) if _bal_valid else []
                        _actual_eval_v = sum(float(h.get("eval_amt", 0.0) or 0.0) for h in _actual_hlds)
                        _actual_total_v = _actual_cash_v + _actual_eval_v

                        _alloc_df_sync = res.get("alloc_df")
                        _max_gap_pct = 0.0
                        _sync_ok = True
                        if isinstance(_alloc_df_sync, pd.DataFrame) and not _alloc_df_sync.empty:
                            if "비중 차이(%p)" in _alloc_df_sync.columns:
                                _max_gap_pct = float(
                                    pd.to_numeric(_alloc_df_sync["비중 차이(%p)"], errors="coerce").abs().max()
                                )
                            if "주문" in _alloc_df_sync.columns:
                                _orders = _alloc_df_sync["주문"].astype(str)
                                _sync_ok = not _orders.isin(["매수", "매도"]).any()

                        st.write(f"**전략 성과 ({_laa_start} ~ 현재)**")
                        st.write(
                            f"수익률: **{_laa_total_ret:+.2f}%** | MDD: **{_laa_mdd:.2f}%** | CAGR: **{_laa_cagr:.2f}%**"
                        )
                        st.write(f"최종자산: {_laa_final_eq:,.0f}원 (1천만원 기준)")

                        st.divider()
                        ac1, ac2, ac3, ac4 = st.columns(4)
                        ac1.metric(
                            "백테스트 자산 (1천만원 기준)",
                            f"{_laa_final_eq:,.0f}원",
                            delta=f"{_laa_total_ret:+.2f}%",
                        )
                        ac2.metric(
                            "실제 총자산",
                            f"{_actual_total_v:,.0f}원" if _bal_valid else "조회 불가",
                        )
                        ac3.metric("최대 비중 차이", f"{_max_gap_pct:.2f}%p")
                        ac4.metric("포지션 동기화", "일치" if _sync_ok else "불일치")
                        ac4.caption("주문 상태가 모두 '유지'이면 일치로 판단합니다.")

                        _laa_bm_series = res.get("bt_benchmark_series")
                        _laa_bm_label = str(res.get("bt_benchmark_label", "SPY Buy & Hold"))
                        _laa_bm_ret = None
                        if isinstance(_laa_bm_series, pd.Series):
                            _laa_bm_series = _laa_bm_series.dropna()
                            _laa_bm_series = _laa_bm_series[_laa_bm_series.index >= pd.Timestamp(_laa_start)]
                            if len(_laa_bm_series) > 1 and float(_laa_bm_series.iloc[0]) > 0:
                                _laa_bm_ret = (_laa_bm_series / float(_laa_bm_series.iloc[0]) - 1.0) * 100.0

                        _eq_ret = (_laa_eq["equity"] / 10_000_000.0 - 1.0) * 100.0
                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:0.7rem 0 1.1rem 0;'>누적 수익률 (%)</div>",
                            unsafe_allow_html=True,
                        )
                        fig_eq = go.Figure()
                        fig_eq.add_trace(
                            go.Scatter(
                                x=_laa_eq.index,
                                y=_eq_ret.values,
                                mode="lines",
                                name="LAA 전략",
                                line=dict(color="gold", width=2),
                            )
                        )
                        if isinstance(_laa_bm_ret, pd.Series) and not _laa_bm_ret.empty:
                            fig_eq.add_trace(
                                go.Scatter(
                                    x=_laa_bm_ret.index,
                                    y=_laa_bm_ret.values,
                                    mode="lines",
                                    name=_laa_bm_label,
                                    line=dict(color="gray", width=1, dash="dot"),
                                )
                            )
                        fig_eq.update_layout(
                            yaxis_title="수익률(%)",
                            height=370,
                            margin=dict(l=0, r=0, t=56, b=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)
                        st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)

                        _eq_peak = _laa_eq["equity"].cummax()
                        _eq_dd = (_laa_eq["equity"] - _eq_peak) / _eq_peak * 100.0
                        st.markdown(
                            "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:0.7rem 0 1.1rem 0;'>Drawdown</div>",
                            unsafe_allow_html=True,
                        )
                        fig_dd = go.Figure()
                        fig_dd.add_trace(
                            go.Scatter(
                                x=_laa_eq.index,
                                y=_eq_dd.values,
                                mode="lines",
                                name="LAA 전략 DD(%)",
                                line=dict(color="crimson", width=2),
                                fill="tozeroy",
                                fillcolor="rgba(220,20,60,0.15)",
                            )
                        )
                        if isinstance(_laa_bm_ret, pd.Series) and not _laa_bm_ret.empty:
                            _bm_eq = _laa_bm_ret / 100.0 + 1.0
                            _bm_peak = _bm_eq.cummax()
                            _bm_dd = (_bm_eq - _bm_peak) / _bm_peak * 100.0
                            fig_dd.add_trace(
                                go.Scatter(
                                    x=_bm_dd.index,
                                    y=_bm_dd.values,
                                    mode="lines",
                                    name=f"{_laa_bm_label} DD(%)",
                                    line=dict(color="gray", width=1, dash="dot"),
                                )
                            )
                        fig_dd.update_layout(
                            yaxis_title="DD(%)",
                            height=300,
                            margin=dict(l=0, r=0, t=56, b=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
                        )
                        fig_dd = _apply_dd_hover_format(fig_dd)
                        st.plotly_chart(fig_dd, use_container_width=True)

                        _render_performance_analysis(
                            equity_series=_laa_eq["equity"],
                            benchmark_series=_laa_bm_series if isinstance(_laa_bm_series, pd.Series) else None,
                            strategy_metrics=_laa_m,
                            strategy_label="LAA 전략",
                            benchmark_label=_laa_bm_label,
                            monte_carlo_sims=400,
                        )

        # ──────────────────────────────────────────────────────────
        # 섹션 4: 듀얼모멘텀 전략 포트폴리오
        # ──────────────────────────────────────────────────────────
        if "듀얼모멘텀" in _active_strategies and _dm_settings:
            st.divider()
            _dm_hdr_col1, _dm_hdr_col2 = st.columns([4, 1])
            with _dm_hdr_col1:
                st.subheader("듀얼모멘텀 전략 포트폴리오")
            with _dm_hdr_col2:
                if st.button("📖 전략 가이드", key="pen_dm_guide_btn"):
                    components.html("""
                    <script>
                    const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                    for (let t of tabs) {
                        if (t.textContent.includes('전략 가이드')) { t.click(); break; }
                    }
                    </script>
                    """, height=0)

            st.caption("공격 ETF 2종 상대모멘텀 + 카나리아 절대모멘텀 기반으로 공격/방어 1종을 선택합니다.")

            if _dm_res:
                if _dm_res.get("error"):
                    st.error(_dm_res["error"])
                else:
                    _dm_sig = _dm_res["signal"]
                    _d1, _d2, _d3, _d4 = st.columns(4)
                    _d1.metric("선택 자산", str(_dm_sig.get("target_ticker", "-")))
                    _d2.metric("국내 ETF", _fmt_etf_code_name(_dm_sig.get("target_kr_code", "")))
                    _d3.metric("카나리아 수익률", f"{float(_dm_sig.get('canary_return', 0.0)) * 100:+.2f}%")
                    _d4.metric("권장 동작", str(_dm_res.get("action", "HOLD")))
                    st.info(str(_dm_sig.get("reason", "")))

                    if "source_map" in _dm_res:
                        st.caption("시그널 데이터(국내): " + ", ".join([f"{k}→{v}" for k, v in _dm_res["source_map"].items()]))

                    _dm_w = _dm_settings.get("momentum_weights", {}) or {}
                    _w1v = float(_dm_w.get("m1", 12.0))
                    _w3v = float(_dm_w.get("m3", 4.0))
                    _w6v = float(_dm_w.get("m6", 2.0))
                    _w12v = float(_dm_w.get("m12", 1.0))
                    _lbv = int(_dm_settings.get("lookback", 12))
                    _tdv = int(_dm_settings.get("trading_days_per_month", 22))
                    st.subheader("시그널/전략 선택 로직 설명")
                    st.markdown(
                        f"""
1. 공격 자산(공격 ETF 1, 공격 ETF 2)의 상대 모멘텀 점수를 계산합니다.
   점수식: `((1개월수익률 × {_w1v:g}) + (3개월수익률 × {_w3v:g}) + (6개월수익률 × {_w6v:g}) + (12개월수익률 × {_w12v:g})) ÷ 4`
2. 카나리아 ETF의 절대 모멘텀을 계산합니다.
   절대모멘텀: `룩백 {_lbv}개월 수익률` (월 환산 거래일 `{_tdv}`일 기준)
3. 전략 선택 규칙
   공격 자산 점수 1위가 카나리아 룩백 수익률보다 크면 공격 자산 1위를 선택하고, 아니면 방어 ETF를 선택합니다.
4. 오늘이 리밸런싱일이라고 가정하고, 최근 종가(전일/휴일은 마지막 거래일 종가) 기준으로 목표 비중과 목표 수량(버림)을 계산합니다.
"""
                    )

                    _score_df = _dm_res.get("score_df")
                    if isinstance(_score_df, pd.DataFrame) and not _score_df.empty:
                        st.subheader("요약 점수")
                        st.dataframe(_score_df, use_container_width=True, hide_index=True)

                    _mom_df = _dm_res.get("momentum_detail_df")
                    if isinstance(_mom_df, pd.DataFrame) and not _mom_df.empty:
                        st.subheader("모멘텀 계산 과정")
                        st.dataframe(_mom_df, use_container_width=True, hide_index=True)

                    _exp_df = _dm_res.get("expected_rebalance_df")
                    _meta = _dm_res.get("expected_meta", {}) or {}
                    if isinstance(_exp_df, pd.DataFrame) and not _exp_df.empty:
                        st.subheader("오늘 리밸런싱 가정 예상 포트폴리오")
                        _ref_date = str(_meta.get("ref_date", "") or "-")
                        _lag_days = _meta.get("lag_days", None)
                        _lag_text = "기준일 정보 없음"
                        try:
                            _lag_i = int(_lag_days)
                            if _lag_i <= 0:
                                _lag_text = "당일 종가 기준"
                            elif _lag_i == 1:
                                _lag_text = "전일 종가 기준"
                            else:
                                _lag_text = f"휴일 반영 최근 거래일 기준 (오늘 대비 {_lag_i}일 전)"
                        except Exception:
                            pass
                        st.caption(f"기준 종가일: {_ref_date} | {_lag_text}")
                        em1, em2, em3 = st.columns(3)
                        em1.metric("듀얼모멘텀 전략 비중", f"{float(_meta.get('dm_weight_pct', 0.0)):.1f}%")
                        em2.metric("전략 배정 평가금액", f"{float(_meta.get('sleeve_eval', 0.0)):,.0f} KRW")
                        em3.metric("버림 후 잔여 현금(예상)", f"{float(_meta.get('sleeve_cash_est', 0.0)):,.0f} KRW")
                        st.dataframe(_exp_df, use_container_width=True, hide_index=True)

                    _dm_bt = _dm_res.get("bt_result")
                    if isinstance(_dm_bt, dict):
                        _dm_eq = _dm_bt.get("equity_df")
                        if isinstance(_dm_eq, pd.DataFrame) and not _dm_eq.empty and "equity" in _dm_eq.columns:
                            _dm_m = _dm_bt.get("metrics", {}) or {}
                            _dm_start = str(_dm_res.get("bt_start_date", _pen_bt_start_raw))
                            _dm_final_eq = float(_dm_m.get("final_equity", _dm_eq["equity"].iloc[-1]))
                            _dm_total_ret = float(_dm_m.get("total_return", 0.0))
                            _dm_mdd = float(_dm_m.get("mdd", 0.0))
                            _dm_cagr = float(_dm_m.get("cagr", 0.0))

                            _bal_valid = isinstance(bal, dict) and not bal.get("error")
                            _actual_cash_v = float(bal.get("cash", 0.0)) if _bal_valid else 0.0
                            _actual_hlds = (bal.get("holdings", []) or []) if _bal_valid else []
                            _actual_eval_v = sum(float(h.get("eval_amt", 0.0) or 0.0) for h in _actual_hlds)
                            _actual_total_v = _actual_cash_v + _actual_eval_v

                            _dm_kr_map_local = _dm_settings.get("kr_etf_map", {}) or {}
                            _dm_codes = [str(v).strip() for v in _dm_kr_map_local.values() if str(v).strip()]
                            _actual_dm_codes = []
                            for _h in _actual_hlds:
                                _code = str(_h.get("code", "")).strip()
                                _qty = float(_h.get("qty", 0.0) or 0.0)
                                if _code in _dm_codes and _qty > 0:
                                    _actual_dm_codes.append(_code)
                            _actual_dm_codes = sorted(set(_actual_dm_codes))

                            _bt_last_ticker = ""
                            if "ticker" in _dm_eq.columns and len(_dm_eq) > 0:
                                _bt_last_ticker = str(_dm_eq["ticker"].iloc[-1]).strip().upper()
                            if not _bt_last_ticker:
                                _bt_last_ticker = str(_dm_sig.get("target_ticker", "")).strip().upper()
                            _bt_last_code = str(_dm_kr_map_local.get(_bt_last_ticker, "")).strip()
                            _sync_dm = (len(_actual_dm_codes) == 1 and _actual_dm_codes[0] == _bt_last_code)
                            _bt_pos_label = _fmt_etf_code_name(_bt_last_code) if _bt_last_code else _bt_last_ticker or "-"
                            if _actual_dm_codes:
                                _actual_pos_label = ", ".join([_fmt_etf_code_name(c) for c in _actual_dm_codes])
                            else:
                                _actual_pos_label = "CASH"

                            st.write(f"**전략 성과 ({_dm_start} ~ 현재)**")
                            st.write(
                                f"수익률: **{_dm_total_ret:+.2f}%** | MDD: **{_dm_mdd:.2f}%** | CAGR: **{_dm_cagr:.2f}%**"
                            )
                            st.write(f"최종자산: {_dm_final_eq:,.0f}원 (1천만원 기준)")

                            st.divider()
                            dc1, dc2, dc3, dc4 = st.columns(4)
                            dc1.metric(
                                "백테스트 자산 (1천만원 기준)",
                                f"{_dm_final_eq:,.0f}원",
                                delta=f"{_dm_total_ret:+.2f}%",
                            )
                            dc2.metric(
                                "실제 총자산",
                                f"{_actual_total_v:,.0f}원" if _bal_valid else "조회 불가",
                            )
                            dc3.metric("백테/실제 포지션", f"{_bt_pos_label} / {_actual_pos_label}")
                            dc4.metric("포지션 동기화", "일치" if _sync_dm else "불일치")

                            _dm_bm_series = _dm_res.get("bt_benchmark_series")
                            _dm_bm_label = str(_dm_res.get("bt_benchmark_label", "SPY Buy & Hold"))
                            _dm_bm_ret = None
                            if isinstance(_dm_bm_series, pd.Series):
                                _dm_bm_series = _dm_bm_series.dropna()
                                _dm_bm_series = _dm_bm_series[_dm_bm_series.index >= pd.Timestamp(_dm_start)]
                                if len(_dm_bm_series) > 1 and float(_dm_bm_series.iloc[0]) > 0:
                                    _dm_bm_ret = (_dm_bm_series / float(_dm_bm_series.iloc[0]) - 1.0) * 100.0

                            _dm_eq_ret = (_dm_eq["equity"] / 10_000_000.0 - 1.0) * 100.0
                            st.markdown(
                                "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:0.7rem 0 1.1rem 0;'>누적 수익률 (%)</div>",
                                unsafe_allow_html=True,
                            )
                            fig_dm_eq = go.Figure()
                            fig_dm_eq.add_trace(
                                go.Scatter(
                                    x=_dm_eq.index,
                                    y=_dm_eq_ret.values,
                                    mode="lines",
                                    name="듀얼모멘텀 전략",
                                    line=dict(color="seagreen", width=2),
                                )
                            )
                            if isinstance(_dm_bm_ret, pd.Series) and not _dm_bm_ret.empty:
                                fig_dm_eq.add_trace(
                                    go.Scatter(
                                        x=_dm_bm_ret.index,
                                        y=_dm_bm_ret.values,
                                        mode="lines",
                                        name=_dm_bm_label,
                                        line=dict(color="gray", width=1, dash="dot"),
                                    )
                                )
                            fig_dm_eq.update_layout(
                                yaxis_title="수익률(%)",
                                height=370,
                                margin=dict(l=0, r=0, t=56, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
                            )
                            st.plotly_chart(fig_dm_eq, use_container_width=True)
                            st.markdown("<div style='height:2.2rem;'></div>", unsafe_allow_html=True)

                            _dm_eq_peak = _dm_eq["equity"].cummax()
                            _dm_eq_dd = (_dm_eq["equity"] - _dm_eq_peak) / _dm_eq_peak * 100.0
                            st.markdown(
                                "<div style='font-size:2.05rem; font-weight:800; line-height:1.25; margin:0.7rem 0 1.1rem 0;'>Drawdown</div>",
                                unsafe_allow_html=True,
                            )
                            fig_dm_dd = go.Figure()
                            fig_dm_dd.add_trace(
                                go.Scatter(
                                    x=_dm_eq.index,
                                    y=_dm_eq_dd.values,
                                    mode="lines",
                                    name="듀얼모멘텀 전략 DD(%)",
                                    line=dict(color="crimson", width=2),
                                    fill="tozeroy",
                                    fillcolor="rgba(220,20,60,0.15)",
                                )
                            )
                            if isinstance(_dm_bm_ret, pd.Series) and not _dm_bm_ret.empty:
                                _dm_bm_eq = _dm_bm_ret / 100.0 + 1.0
                                _dm_bm_peak = _dm_bm_eq.cummax()
                                _dm_bm_dd = (_dm_bm_eq - _dm_bm_peak) / _dm_bm_peak * 100.0
                                fig_dm_dd.add_trace(
                                    go.Scatter(
                                        x=_dm_bm_dd.index,
                                        y=_dm_bm_dd.values,
                                        mode="lines",
                                        name=f"{_dm_bm_label} DD(%)",
                                        line=dict(color="gray", width=1, dash="dot"),
                                    )
                                )
                            fig_dm_dd.update_layout(
                                yaxis_title="DD(%)",
                                height=300,
                                margin=dict(l=0, r=0, t=56, b=30),
                                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
                            )
                            fig_dm_dd = _apply_dd_hover_format(fig_dm_dd)
                            st.plotly_chart(fig_dm_dd, use_container_width=True)

                            _render_performance_analysis(
                                equity_series=_dm_eq["equity"],
                                benchmark_series=_dm_bm_series if isinstance(_dm_bm_series, pd.Series) else None,
                                strategy_metrics=_dm_m,
                                strategy_label="듀얼모멘텀 전략",
                                benchmark_label=_dm_bm_label,
                                monte_carlo_sims=400,
                            )

    # ══════════════════════════════════════════════════════════════
    # Tab 2: 백테스트
    # ══════════════════════════════════════════════════════════════
    with tab_p2:
        if _pen_local_first:
            st.info("백테스트 가격 데이터: 로컬 파일(cache/data) 우선, 부족 시 API 보강 모드입니다.")
        _bt_candidates = [s for s in ["LAA", "정적배분"] if s in _active_strategies]
        # 듀얼모멘텀 백테스트는 포트폴리오 표기 상태와 무관하게 항상 선택 가능하게 제공
        if "듀얼모멘텀" not in _bt_candidates:
            _bt_candidates.insert(1 if "LAA" in _bt_candidates else 0, "듀얼모멘텀")
        if not _bt_candidates:
            st.warning("포트폴리오에 활성화된 전략이 없어 백테스트를 실행할 수 없습니다.")
        else:
            _bt_strategy = st.selectbox("백테스트 전략", _bt_candidates, key="pen_bt_strategy_select")

            pen_bt_cap = st.number_input("초기 자본 (KRW)", value=10_000_000, step=1_000_000, key="pen_bt_cap")
            pen_bt_fee = st.number_input("수수료 (%)", value=0.02, format="%.2f", key="pen_bt_fee") / 100.0

            if _bt_strategy == "LAA":
                st.header("LAA 백테스트")
                st.caption("국내 ETF 매핑(SPY/IWD/GLD/IEF/QQQ/SHY) 기반 월간 리밸런싱 시뮬레이션")

                if st.button("LAA 백테스트 실행", key="pen_bt_run_laa", type="primary"):
                    with st.spinner("LAA 백테스트 실행 중... (국내 데이터 조회)"):
                        tickers = ["SPY", "IWD", "GLD", "IEF", "QQQ", "SHY"]
                        source_map = {
                            "SPY": _code_only(kr_spy or kr_iwd or "360750"),
                            "IWD": _code_only(kr_iwd),
                            "GLD": _code_only(kr_gld),
                            "IEF": _code_only(kr_ief),
                            "QQQ": _code_only(kr_qqq),
                            "SHY": _code_only(kr_shy),
                        }
                        price_data = {}
                        for ticker in tickers:
                            _code = _code_only(source_map.get(ticker, ""))
                            if not _code:
                                st.error(f"{ticker} 국내 ETF 코드가 비어 있습니다.")
                                price_data = None
                                break
                            df_t = _get_pen_daily_chart(_code, count=3000, use_disk_cache=True)
                            if df_t is None or df_t.empty:
                                st.error(f"{ticker} ({_code}) 로컬 데이터가 없습니다. cache 또는 data 폴더를 확인하세요.")
                                price_data = None
                                break
                            df_t = df_t.copy().sort_index()
                            if "close" not in df_t.columns and "Close" in df_t.columns:
                                df_t["close"] = df_t["Close"]
                            if "close" not in df_t.columns:
                                st.error(f"{ticker} ({_code}) 종가 컬럼이 없습니다.")
                                price_data = None
                                break
                            price_data[ticker] = df_t

                        if price_data:
                            strategy = LAAStrategy(settings={"kr_etf_map": _kr_etf_map})
                            bt_result = strategy.run_backtest(price_data, initial_balance=float(pen_bt_cap), fee=float(pen_bt_fee))
                            if bt_result:
                                st.session_state["pen_bt_laa_result"] = bt_result
                                st.session_state["pen_bt_result"] = bt_result
                                _laa_bm_ticker = "SPY"
                                _laa_bm_series = _normalize_numeric_series(
                                    price_data.get(_laa_bm_ticker),
                                    preferred_cols=("close", "Close"),
                                )
                                if not _laa_bm_series.empty:
                                    st.session_state["pen_bt_laa_benchmark_series"] = _laa_bm_series
                                    _laa_bm_code = _code_only(source_map.get(_laa_bm_ticker, ""))
                                    if _laa_bm_code:
                                        st.session_state["pen_bt_laa_benchmark_label"] = f"{_fmt_etf_code_name(_laa_bm_code)} Buy & Hold"
                                    else:
                                        st.session_state["pen_bt_laa_benchmark_label"] = f"{_laa_bm_ticker} Buy & Hold"
                            else:
                                st.error("백테스트 실행 실패 (데이터 부족)")

                bt_res = st.session_state.get("pen_bt_laa_result") or st.session_state.get("pen_bt_result")
                if bt_res:
                    metrics = bt_res["metrics"]
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric("총 수익률", f"{metrics['total_return']:.2f}%")
                    mc2.metric("CAGR", f"{metrics['cagr']:.2f}%")
                    mc3.metric("MDD", f"{metrics['mdd']:.2f}%")
                    mc4.metric("샤프", f"{metrics['sharpe']:.2f}")
                    mc5.metric("최종 자산", f"{metrics['final_equity']:,.0f}")

                    eq_df = bt_res["equity_df"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq_df.index, y=eq_df["equity"],
                        name="포트폴리오", line=dict(color="royalblue"),
                    ))
                    fig.update_layout(title="LAA 백테스트 자산 곡선", xaxis_title="날짜", yaxis_title="자산 (KRW)", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    if "equity" in eq_df.columns:
                        yearly = eq_df["equity"].resample("YE").last()
                        if len(yearly) > 1:
                            yr_ret = yearly.pct_change().dropna() * 100
                            yr_data = [{"연도": str(d.year), "수익률(%)": f"{r:.2f}"} for d, r in yr_ret.items()]
                            st.subheader("연도별 수익률")
                            st.dataframe(pd.DataFrame(yr_data), use_container_width=True, hide_index=True)

                    if "allocations" in bt_res:
                        alloc_df = bt_res["allocations"]
                        if not alloc_df.empty:
                            st.subheader("월별 자산 배분 이력")
                            st.dataframe(alloc_df.tail(24), use_container_width=True, hide_index=True)

                    _laa_bm_series = st.session_state.get("pen_bt_laa_benchmark_series")
                    _laa_bm_label = str(st.session_state.get("pen_bt_laa_benchmark_label", "SPY Buy & Hold"))
                    _render_performance_analysis(
                        equity_series=eq_df.get("equity"),
                        benchmark_series=_laa_bm_series,
                        strategy_metrics=metrics,
                        strategy_label="LAA 전략",
                        benchmark_label=_laa_bm_label,
                        monte_carlo_sims=400,
                    )

            elif _bt_strategy == "듀얼모멘텀":
                st.header("듀얼모멘텀 백테스트")
                st.caption("사이드바의 국내 ETF 설정(공격 2종/방어/카나리아) 기반 월간 리밸런싱 시뮬레이션")

                if not _dm_settings:
                    st.warning("사이드바에서 듀얼모멘텀 설정을 먼저 입력해 주세요.")
                else:
                    if st.button("듀얼모멘텀 백테스트 실행", key="pen_bt_run_dm", type="primary"):
                        with st.spinner("듀얼모멘텀 백테스트 실행 중... (국내 데이터 조회)"):
                            dm_tickers = []
                            for tk in (_dm_settings.get("offensive", []) + _dm_settings.get("defensive", []) + _dm_settings.get("canary", [])):
                                tku = str(tk).strip().upper()
                                if tku and tku not in dm_tickers:
                                    dm_tickers.append(tku)

                            dm_price_data = {}
                            dm_kr_map = _dm_settings.get("kr_etf_map", {}) or {}
                            for ticker in dm_tickers:
                                kr_code = str(dm_kr_map.get(ticker, "")).strip()
                                if not kr_code:
                                    st.error(f"{ticker} 국내 ETF 매핑이 없습니다.")
                                    dm_price_data = None
                                    break
                                df_t = _get_pen_daily_chart(kr_code, count=3000, use_disk_cache=True)
                                if df_t is None or df_t.empty:
                                    st.error(f"{ticker} ({kr_code}) 로컬 데이터가 없습니다. cache 또는 data 폴더를 확인하세요.")
                                    dm_price_data = None
                                    break
                                dm_price_data[ticker] = df_t

                            if dm_price_data:
                                dm_strategy = DualMomentumStrategy(settings=_dm_settings)
                                dm_bt_result = dm_strategy.run_backtest(
                                    dm_price_data,
                                    initial_balance=float(pen_bt_cap),
                                    fee=float(pen_bt_fee),
                                )
                                if dm_bt_result:
                                    st.session_state["pen_bt_dm_result"] = dm_bt_result
                                    _dm_bm_ticker = ""
                                    for _t in (_dm_settings.get("offensive", []) or []):
                                        if _t in dm_price_data:
                                            _dm_bm_ticker = str(_t)
                                            break
                                    if not _dm_bm_ticker:
                                        for _t in dm_tickers:
                                            if _t in dm_price_data:
                                                _dm_bm_ticker = str(_t)
                                                break

                                    _dm_bm_series = _normalize_numeric_series(
                                        dm_price_data.get(_dm_bm_ticker),
                                        preferred_cols=("close", "Close"),
                                    )
                                    if not _dm_bm_series.empty:
                                        st.session_state["pen_bt_dm_benchmark_series"] = _dm_bm_series
                                        _dm_bm_code = _code_only(str(dm_kr_map.get(_dm_bm_ticker, "")))
                                        if _dm_bm_code:
                                            st.session_state["pen_bt_dm_benchmark_label"] = f"{_fmt_etf_code_name(_dm_bm_code)} Buy & Hold"
                                        else:
                                            st.session_state["pen_bt_dm_benchmark_label"] = f"{_dm_bm_ticker} Buy & Hold"
                                else:
                                    st.error("백테스트 실행 실패 (데이터 부족)")

                    dm_res = st.session_state.get("pen_bt_dm_result")
                    if dm_res:
                        metrics = dm_res["metrics"]
                        dc1, dc2, dc3, dc4, dc5 = st.columns(5)
                        dc1.metric("총 수익률", f"{metrics['total_return']:.2f}%")
                        dc2.metric("CAGR", f"{metrics['cagr']:.2f}%")
                        dc3.metric("MDD", f"{metrics['mdd']:.2f}%")
                        dc4.metric("샤프", f"{metrics['sharpe']:.2f}")
                        dc5.metric("최종 자산", f"{metrics['final_equity']:,.0f}")

                        eq_df = dm_res["equity_df"]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=eq_df.index, y=eq_df["equity"],
                            name="듀얼모멘텀", line=dict(color="seagreen"),
                        ))
                        fig.update_layout(title="듀얼모멘텀 백테스트 자산 곡선", xaxis_title="날짜", yaxis_title="자산 (KRW)", height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        if "equity" in eq_df.columns:
                            yearly = eq_df["equity"].resample("YE").last()
                            if len(yearly) > 1:
                                yr_ret = yearly.pct_change().dropna() * 100
                                yr_data = [{"연도": str(d.year), "수익률(%)": f"{r:.2f}"} for d, r in yr_ret.items()]
                                st.subheader("연도별 수익률")
                                st.dataframe(pd.DataFrame(yr_data), use_container_width=True, hide_index=True)

                        pos_df = dm_res.get("positions")
                        if isinstance(pos_df, pd.DataFrame) and not pos_df.empty:
                            st.subheader("월별 포지션 이력")
                            st.dataframe(pos_df.tail(36), use_container_width=True, hide_index=True)

                        _dm_bm_series = st.session_state.get("pen_bt_dm_benchmark_series")
                        _dm_bm_label = str(st.session_state.get("pen_bt_dm_benchmark_label", "SPY Buy & Hold"))
                        _render_performance_analysis(
                            equity_series=eq_df.get("equity"),
                            benchmark_series=_dm_bm_series,
                            strategy_metrics=metrics,
                            strategy_label="듀얼모멘텀 전략",
                            benchmark_label=_dm_bm_label,
                            monte_carlo_sims=400,
                        )

            else:
                st.header("정적배분 백테스트")
                st.info("정적배분 백테스트는 다음 단계에서 연결 예정입니다.")

    # ══════════════════════════════════════════════════════════════
    # Tab 3: 수동 주문
    # ══════════════════════════════════════════════════════════════
    with tab_p3:
        st.header("수동 주문")

        bal = st.session_state.get(pen_bal_key)
        if not bal:
            st.warning("잔고를 먼저 조회해 주세요.")
        else:
            cash = float(bal.get("cash", 0.0))
            holdings = bal.get("holdings", []) or []

            # 매매 대상 ETF 선택 (활성 전략 전체 반영)
            all_etf_codes = []
            all_etf_codes.extend([kr_iwd, kr_gld, kr_ief, kr_qqq, kr_shy])
            if _dm_settings:
                _dm_map_local = _dm_settings.get("kr_etf_map", {}) or {}
                all_etf_codes.extend([_dm_map_local.get("SPY", ""), _dm_map_local.get("EFA", ""), _dm_map_local.get("AGG", "")])
            if _static_settings:
                all_etf_codes.extend([str(x.get("code", "")) for x in _static_settings.get("etfs", [])])

            all_etf_codes = [str(c).strip() for c in all_etf_codes if str(c).strip()]
            all_etf_codes = list(dict.fromkeys(all_etf_codes))
            if not all_etf_codes:
                all_etf_codes = ["360750"]

            etf_options = {_fmt_etf_code_name(c): c for c in all_etf_codes}
            selected_pen_label = st.selectbox("매매 ETF 선택", list(etf_options.keys()), key="pen_trade_etf_label")
            selected_pen_etf = etf_options[selected_pen_label]

            pen_holding = next((h for h in holdings if str(h.get("code", "")) == str(selected_pen_etf)), None)
            pen_qty = int(pen_holding.get("qty", 0)) if pen_holding else 0

            # ── 상단 정보 바 (골드 패널과 동일 구조) ──
            cur_price = _get_pen_current_price(str(selected_pen_etf))
            _pen_cur = float(cur_price) if cur_price and cur_price > 0 else 0
            _pen_eval = _pen_cur * pen_qty if _pen_cur > 0 else 0

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("현재가", f"{_pen_cur:,.0f}원" if _pen_cur > 0 else "–")
            pc2.metric(f"{_fmt_etf_code_name(selected_pen_etf)} 보유", f"{pen_qty}주")
            pc3.metric("평가금액", f"{_pen_eval:,.0f}원")
            pc4.metric("예수금", f"{cash:,.0f}원")

            # ═══ 일봉 차트 (상단 전체폭) ═══
            _pen_chart_df = _get_pen_daily_chart(str(selected_pen_etf), count=120)
            if _pen_chart_df is not None and len(_pen_chart_df) > 0:
                _pen_chart_df = _pen_chart_df.copy().sort_index()
                if "close" not in _pen_chart_df.columns and "Close" in _pen_chart_df.columns:
                    _pen_chart_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
                st.markdown(f"**{_fmt_etf_code_name(selected_pen_etf)} 일봉 차트**")
                _fig_pen = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
                _fig_pen.add_trace(go.Candlestick(
                    x=_pen_chart_df.index, open=_pen_chart_df['open'], high=_pen_chart_df['high'],
                    low=_pen_chart_df['low'], close=_pen_chart_df['close'], name='일봉',
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                ), row=1, col=1)
                _pen_ma5 = _pen_chart_df['close'].rolling(5).mean()
                _pen_ma20 = _pen_chart_df['close'].rolling(20).mean()
                _fig_pen.add_trace(go.Scatter(x=_pen_chart_df.index, y=_pen_ma5, name='MA5', line=dict(color='#FF9800', width=1)), row=1, col=1)
                _fig_pen.add_trace(go.Scatter(x=_pen_chart_df.index, y=_pen_ma20, name='MA20', line=dict(color='#2196F3', width=1)), row=1, col=1)
                if 'volume' in _pen_chart_df.columns:
                    _pen_vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(_pen_chart_df['close'], _pen_chart_df['open'])]
                    _fig_pen.add_trace(go.Bar(x=_pen_chart_df.index, y=_pen_chart_df['volume'], marker_color=_pen_vol_colors, name='거래량', showlegend=False), row=2, col=1)
                _fig_pen.update_layout(
                    height=450, margin=dict(l=0, r=0, t=10, b=30),
                    xaxis_rangeslider_visible=False, showlegend=True,
                    legend=dict(orientation="h", y=1.06, x=0),
                    xaxis2=dict(showticklabels=True, tickformat='%m/%d', tickangle=-45),
                    yaxis=dict(title="", side="right"),
                    yaxis2=dict(title="", side="right"),
                )
                st.plotly_chart(_fig_pen, use_container_width=True, key=f"pen_manual_chart_{selected_pen_etf}")
            else:
                st.info("차트 데이터 로딩 중...")


            st.divider()

            ob_col, order_col = st.columns([2, 3])

            # ── 좌: 호가창 ──
            with ob_col:
                ob = _get_pen_orderbook(str(selected_pen_etf))
                if ob and ob.get("asks") and ob.get("bids"):
                    asks = ob["asks"]
                    bids = ob["bids"]
                    all_qtys = [a["qty"] for a in asks] + [b["qty"] for b in bids]
                    max_qty = max(all_qtys) if all_qtys else 1
                    html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                    html.append(
                        '<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666">'
                        '<td>구분</td><td style="text-align:right">잔량</td>'
                        '<td style="text-align:right">가격(원)</td>'
                        '<td style="text-align:right">등락</td><td>비율</td></tr>'
                    )
                    for a in reversed(asks):
                        ap, aq = a["price"], a["qty"]
                        diff = ((ap / _pen_cur) - 1) * 100 if _pen_cur > 0 else 0
                        bar_w = int(aq / max_qty * 100) if max_qty > 0 else 0
                        html.append(
                            f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                            f'<td>매도</td>'
                            f'<td style="text-align:right">{aq:,}</td>'
                            f'<td style="text-align:right;font-weight:bold">{ap:,.0f}</td>'
                            f'<td style="text-align:right">{diff:+.2f}%</td>'
                            f'<td><div style="background:#1976D233;height:14px;width:{bar_w}%"></div></td></tr>'
                        )
                    html.append(
                        f'<tr style="background:#FFF3E0;border:2px solid #FF9800;height:32px;font-weight:bold">'
                        f'<td colspan="2" style="color:#E65100">현재가</td>'
                        f'<td style="text-align:right;color:#E65100;font-size:15px">{_pen_cur:,.0f}</td>'
                        f'<td colspan="2"></td></tr>'
                    )
                    for b in bids:
                        bp, bq = b["price"], b["qty"]
                        diff = ((bp / _pen_cur) - 1) * 100 if _pen_cur > 0 else 0
                        bar_w = int(bq / max_qty * 100) if max_qty > 0 else 0
                        html.append(
                            f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                            f'<td>매수</td>'
                            f'<td style="text-align:right">{bq:,}</td>'
                            f'<td style="text-align:right;font-weight:bold">{bp:,.0f}</td>'
                            f'<td style="text-align:right">{diff:+.2f}%</td>'
                            f'<td><div style="background:#D32F2F33;height:14px;width:{bar_w}%"></div></td></tr>'
                        )
                    html.append("</table>")
                    st.markdown("".join(html), unsafe_allow_html=True)
                    if asks and bids:
                        spread = asks[0]["price"] - bids[0]["price"]
                        spread_pct = (spread / _pen_cur * 100) if _pen_cur > 0 else 0
                        total_ask_q = sum(a["qty"] for a in asks)
                        total_bid_q = sum(b["qty"] for b in bids)
                        st.caption(
                            f"스프레드: {spread:,.0f}원 ({spread_pct:.3f}%) | "
                            f"매도잔량: {total_ask_q:,} | 매수잔량: {total_bid_q:,}"
                        )
                else:
                    st.info("호가 데이터를 불러오는 중...")

            # ── 우: 주문 패널 ──
            with order_col:
                buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                with buy_tab:
                    pen_buy_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"], key="pen_buy_method", horizontal=True)

                    pen_buy_price = 0
                    if pen_buy_method == "지정가":
                        pen_buy_price = st.number_input("매수 지정가 (원)", min_value=0, value=int(_pen_cur) if _pen_cur > 0 else 0, step=50, key="pen_buy_price")
                    else:
                        pen_buy_price = _pen_cur

                    pen_buy_qty = st.number_input("매수 수량 (주)", min_value=0, value=0, step=1, key="pen_buy_qty")

                    _pen_buy_unit = pen_buy_price if pen_buy_price > 0 else _pen_cur
                    _pen_total = int(pen_buy_qty * _pen_buy_unit) if pen_buy_qty > 0 and _pen_buy_unit > 0 else 0
                    st.markdown(
                        f"<div style='background:#fff3f3;border:1px solid #ffcdd2;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                        f"<b>단가:</b> {_pen_buy_unit:,.0f}원 &nbsp;|&nbsp; "
                        f"<b>수량:</b> {pen_buy_qty:,}주 &nbsp;|&nbsp; "
                        f"<b>총 금액:</b> <span style='color:#D32F2F;font-weight:bold'>{_pen_total:,}원</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    pen_buy_amt = _pen_total

                    if st.button("매수 실행", key="pen_exec_buy", type="primary", disabled=IS_CLOUD):
                        if pen_buy_amt <= 0:
                            st.error("매수 금액을 입력해 주세요.")
                        else:
                            with st.spinner("매수 주문 실행 중..."):
                                if pen_buy_method == "동시호가 (장마감)":
                                    result = trader.execute_closing_auction_buy(str(selected_pen_etf), pen_buy_qty) if pen_buy_qty > 0 else None
                                elif pen_buy_method == "지정가" and pen_buy_price > 0:
                                    result = trader.send_order("BUY", str(selected_pen_etf), pen_buy_qty, price=pen_buy_price, ord_dvsn="00") if pen_buy_qty > 0 else None
                                elif pen_buy_method == "시간외 종가":
                                    result = trader.send_order("BUY", str(selected_pen_etf), pen_buy_qty, price=0, ord_dvsn="06") if pen_buy_qty > 0 else None
                                else:
                                    result = trader.send_order("BUY", str(selected_pen_etf), pen_buy_qty, ord_dvsn="01") if pen_buy_qty > 0 else None
                                if result and (isinstance(result, dict) and result.get("success")):
                                    st.success(f"매수 완료: {result}")
                                    st.session_state[pen_bal_key] = trader.get_balance()
                                else:
                                    st.error(f"매수 실패: {result}")

                with sell_tab:
                    pen_sell_method = st.radio("주문 방식", ["시장가", "지정가", "동시호가 (장마감)", "시간외 종가"], key="pen_sell_method", horizontal=True)

                    pen_sell_price = 0
                    if pen_sell_method == "지정가":
                        pen_sell_price = st.number_input("매도 지정가 (원)", min_value=0, value=int(_pen_cur) if _pen_cur > 0 else 0, step=50, key="pen_sell_price")
                    else:
                        pen_sell_price = _pen_cur

                    pen_sell_qty = st.number_input("매도 수량 (주)", min_value=0, max_value=max(pen_qty, 1), value=pen_qty, step=1, key="pen_sell_qty")
                    pen_sell_all = st.checkbox("전량 매도", value=True, key="pen_sell_all")

                    _pen_sell_unit = pen_sell_price if pen_sell_price > 0 else _pen_cur
                    _pen_sell_qty_final = pen_qty if pen_sell_all else pen_sell_qty
                    _pen_sell_total = int(_pen_sell_qty_final * _pen_sell_unit) if _pen_sell_qty_final > 0 and _pen_sell_unit > 0 else 0
                    st.markdown(
                        f"<div style='background:#f3f8ff;border:1px solid #bbdefb;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                        f"<b>단가:</b> {_pen_sell_unit:,.0f}원 &nbsp;|&nbsp; "
                        f"<b>수량:</b> {_pen_sell_qty_final:,}주 &nbsp;|&nbsp; "
                        f"<b>총 금액:</b> <span style='color:#1976D2;font-weight:bold'>{_pen_sell_total:,}원</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if st.button("매도 실행", key="pen_exec_sell", type="primary", disabled=IS_CLOUD):
                        _sq = pen_qty if pen_sell_all else pen_sell_qty
                        if _sq <= 0:
                            st.error("매도할 수량이 없습니다.")
                        else:
                            with st.spinner("매도 주문 실행 중..."):
                                if pen_sell_method == "동시호가 (장마감)":
                                    result = trader.smart_sell_all_closing(str(selected_pen_etf)) if pen_sell_all else trader.smart_sell_qty_closing(str(selected_pen_etf), _sq)
                                elif pen_sell_method == "지정가" and pen_sell_price > 0:
                                    result = trader.send_order("SELL", str(selected_pen_etf), _sq, price=pen_sell_price, ord_dvsn="00")
                                elif pen_sell_method == "시간외 종가":
                                    result = trader.send_order("SELL", str(selected_pen_etf), _sq, price=0, ord_dvsn="06")
                                else:
                                    result = trader.smart_sell_all(str(selected_pen_etf)) if pen_sell_all else trader.smart_sell_qty(str(selected_pen_etf), _sq)
                                if result and (isinstance(result, dict) and result.get("success")):
                                    st.success(f"매도 완료: {result}")
                                    st.session_state[pen_bal_key] = trader.get_balance()
                                else:
                                    st.error(f"매도 실패: {result}")

    # ══════════════════════════════════════════════════════════════
    # Tab 4: 전략 가이드
    # ══════════════════════════════════════════════════════════════
    with tab_p4:
        st.header("연금저축 전략 가이드")

        st.subheader("1. LAA (Lethargic Asset Allocation) 전략")
        st.markdown("""
**LAA**는 Keller & Keuning이 제안한 게으른 자산배분 전략입니다.

- **코어 자산 (75%)**: IWD(미국 가치주), GLD(금), IEF(미국 중기채) 각 25%
- **리스크 자산 (25%)**: SPY가 200일 이동평균선 위 → QQQ, 아래 → SHY(단기채)
- **리밸런싱**: 월 1회 (월말 기준)
""")

        st.subheader("2. LAA 의사결정 흐름")
        st.markdown("""
```
매월 말 기준:
  1. SPY 종가 vs SPY 200일 이동평균
  2. SPY > 200일선 → 리스크 자산 = QQQ (공격)
     SPY < 200일선 → 리스크 자산 = SHY (방어)
  3. 코어 3종목 25%씩 + 리스크 자산 25% 배분
  4. 목표 비중 대비 괴리 > 3%p이면 리밸런싱 실행
```
""")

        st.subheader("3. LAA 국내 ETF 매핑")
        st.dataframe(pd.DataFrame([
            {"미국 티커": "IWD", "국내 ETF": "TIGER 미국S&P500 (360750)", "역할": "코어 - 미국 가치주"},
            {"미국 티커": "GLD", "국내 ETF": "KODEX Gold선물(H) (132030)", "역할": "코어 - 금"},
            {"미국 티커": "IEF", "국내 ETF": "TIGER 미국채10년선물 (453540)", "역할": "코어 - 중기채"},
            {"미국 티커": "QQQ", "국내 ETF": "TIGER 미국나스닥100 (133690)", "역할": "리스크 공격"},
            {"미국 티커": "SHY", "국내 ETF": "KODEX 국고채3년 (114470)", "역할": "리스크 방어"},
        ]), use_container_width=True, hide_index=True)
        st.caption("연금저축 계좌에서 해외 ETF 직접 매매 불가 → 국내 ETF로 대체 실행")

        st.subheader("4. 듀얼모멘텀 (GEM) 전략")
        _guide_dm_map = (_dm_settings.get("kr_etf_map", {}) if isinstance(_dm_settings, dict) else {}) or {}
        _guide_dm_spy = str(_guide_dm_map.get("SPY", _code_only(config.get("pen_dm_kr_spy", "360750"))))
        _guide_dm_efa = str(_guide_dm_map.get("EFA", _code_only(config.get("pen_dm_kr_efa", "453850"))))
        _guide_dm_agg = str(_guide_dm_map.get("AGG", _code_only(config.get("pen_dm_kr_agg", "453540"))))
        _guide_dm_bil = str(_guide_dm_map.get("BIL", _code_only(config.get("pen_dm_kr_bil", "114470"))))
        _guide_dm_w = (_dm_settings.get("momentum_weights", {}) if isinstance(_dm_settings, dict) else {}) or {}
        _guide_dm_w1 = float(_guide_dm_w.get("m1", config.get("pen_dm_w1", 12.0)))
        _guide_dm_w3 = float(_guide_dm_w.get("m3", config.get("pen_dm_w3", 4.0)))
        _guide_dm_w6 = float(_guide_dm_w.get("m6", config.get("pen_dm_w6", 2.0)))
        _guide_dm_w12 = float(_guide_dm_w.get("m12", config.get("pen_dm_w12", 1.0)))
        _guide_dm_lb = int((_dm_settings.get("lookback", config.get("pen_dm_lookback", 12)) if isinstance(_dm_settings, dict) else config.get("pen_dm_lookback", 12)))
        _guide_dm_td = int((_dm_settings.get("trading_days_per_month", config.get("pen_dm_trading_days", 22)) if isinstance(_dm_settings, dict) else config.get("pen_dm_trading_days", 22)))

        st.markdown(f"""
**듀얼모멘텀(GEM)**은 상대모멘텀 + 절대모멘텀을 결합한 월간 리밸런싱 전략입니다.

- **공격 자산(2개)**: SPY, EFA 중 모멘텀 점수 상위 1개 선택
- **방어 자산(1개)**: AGG
- **카나리아(1개)**: BIL
- **리밸런싱**: 월 1회 (월말 기준)

모멘텀 점수식:
`((1개월수익률 × {_guide_dm_w1:g}) + (3개월수익률 × {_guide_dm_w3:g}) + (6개월수익률 × {_guide_dm_w6:g}) + (12개월수익률 × {_guide_dm_w12:g})) ÷ 4`

절대모멘텀 기준:
`카나리아 룩백 {_guide_dm_lb}개월 수익률` (월 환산 거래일 `{_guide_dm_td}`일 기준)
""")

        st.subheader("5. 듀얼모멘텀 의사결정 흐름")
        st.markdown(f"""
```
매월 말 기준:
  1. 공격 자산(SPY, EFA)의 가중 모멘텀 점수 계산
  2. 카나리아(BIL) 룩백 {_guide_dm_lb}개월 수익률 계산
  3. 공격 1위 점수 > 카나리아 수익률  → 공격 1위 100%
     공격 1위 점수 <= 카나리아 수익률 → 방어(AGG) 100%
  4. 목표 비중 대비 괴리 발생 시 리밸런싱 실행
```
""")

        st.subheader("6. 듀얼모멘텀 국내 ETF 매핑")
        st.dataframe(pd.DataFrame([
            {"전략 키": "SPY", "국내 ETF": _fmt_etf_code_name(_guide_dm_spy), "역할": "공격 자산 1"},
            {"전략 키": "EFA", "국내 ETF": _fmt_etf_code_name(_guide_dm_efa), "역할": "공격 자산 2"},
            {"전략 키": "AGG", "국내 ETF": _fmt_etf_code_name(_guide_dm_agg), "역할": "방어 자산"},
            {"전략 키": "BIL", "국내 ETF": _fmt_etf_code_name(_guide_dm_bil), "역할": "카나리아"},
        ]), use_container_width=True, hide_index=True)
        st.caption("듀얼모멘텀도 연금저축 계좌에서 국내 ETF로 시그널/실매매를 수행합니다.")

        st.subheader("7. 기대 성과/운용 특성")
        st.markdown("""
- **시장 대응**: 상승장에서는 공격 자산, 하락장/둔화장에서는 방어 자산으로 자동 전환
- **리스크 관리**: 절대모멘텀(카나리아) 기준으로 하락 추세 회피
- **운용 빈도**: 월 1회 리밸런싱으로 과도한 매매 방지
- **과세이연**: 연금저축 계좌 내 매매차익 과세이연 효과로 복리 운용에 유리
""")

    # ══════════════════════════════════════════════════════════════
    # Tab 5: 주문방식
    # ══════════════════════════════════════════════════════════════
    with tab_p5:
        st.header("KIS 국내 ETF 주문방식 안내")
        st.dataframe(pd.DataFrame([
            {"구분": "시장가", "API": 'ord_dvsn="01"', "설명": "즉시 체결 (최우선 호가)"},
            {"구분": "지정가", "API": 'ord_dvsn="00"', "설명": "원하는 가격에 주문"},
            {"구분": "동시호가 매수", "API": '상한가(+30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
            {"구분": "동시호가 매도", "API": '하한가(-30%) 지정가', "설명": "15:20~15:30 동시호가 참여 → 60초 대기 → 미체결 시 시간외 재주문"},
            {"구분": "시간외 종가", "API": 'ord_dvsn="06"', "설명": "15:40~16:00 당일 종가로 체결"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("호가단위")
        st.dataframe(pd.DataFrame([
            {"가격대": "~5,000원", "호가단위": "5원"},
            {"가격대": "5,000~10,000원", "호가단위": "10원"},
            {"가격대": "10,000~50,000원", "호가단위": "50원"},
            {"가격대": "50,000원~", "호가단위": "100원"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("자동매매 흐름 (GitHub Actions)")
        st.markdown("""
1. 매월 25~31일 평일 KST 15:20 실행 (`TRADING_MODE=kis_pension`)
2. 국내 ETF(SPY/IWD/GLD/IEF/QQQ/SHY 매핑) 일봉 조회
3. SPY vs 200일선 → 리스크 자산 결정 (QQQ or SHY)
4. 목표 배분 vs 현재 보유 비교 → 리밸런싱 필요 여부 판단
5. 매도 → `smart_sell_all_closing()` (동시호가+시간외)
6. 매수 → `smart_buy_krw_closing()` (동시호가+시간외)
""")

    # ══════════════════════════════════════════════════════════════
    # Tab 6: 수수료/세금
    # ══════════════════════════════════════════════════════════════
    with tab_p6:
        st.header("연금저축 수수료 및 세금 안내")

        st.subheader("1. 매매 수수료")
        st.dataframe(pd.DataFrame([
            {"증권사": "한국투자증권", "매매 수수료": "0.0140396%", "비고": "나무 온라인 (현재 사용)"},
            {"증권사": "키움증권", "매매 수수료": "0.015%", "비고": "영웅문 온라인"},
            {"증권사": "미래에셋", "매매 수수료": "0.014%", "비고": "m.Stock 온라인"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("2. 매매 대상 ETF 보수")
        st.dataframe(pd.DataFrame([
            {"ETF": "TIGER 미국S&P500", "코드": "360750", "총보수": "0.07%", "추종": "SPY 대용"},
            {"ETF": "KODEX Gold선물(H)", "코드": "132030", "총보수": "0.09%", "추종": "GLD 대용"},
            {"ETF": "TIGER 미국채10년선물", "코드": "453540", "총보수": "0.10%", "추종": "IEF 대용"},
            {"ETF": "TIGER 미국나스닥100", "코드": "133690", "총보수": "0.07%", "추종": "QQQ 대용"},
            {"ETF": "KODEX 국고채3년", "코드": "114470", "총보수": "0.05%", "추종": "SHY 대용"},
        ]), use_container_width=True, hide_index=True)

        st.subheader("3. 연금저축 세제혜택")
        st.markdown("""
| 항목 | 내용 |
|------|------|
| 세액공제 | 연 최대 600만원 (IRP 합산 900만원) |
| 공제율 | 총급여 5,500만원 이하: 16.5% / 초과: 13.2% |
| 과세이연 | 매매차익·배당 세금 인출 시까지 이연 |
| 연금 수령 시 | 3.3~5.5% 연금소득세 (일반 15.4% 대비 유리) |
| 중도 인출 시 | 16.5% 기타소득세 (불이익) |
""")
        st.caption("LAA 월간 리밸런싱 매매차익이 모두 과세이연되어 복리 효과 극대화 (일반 계좌 대비 연 1~2% 추가 수익)")



def main():
    # --- Mode Select (Sidebar Top) ---
    _mode_map = {
        "코인": "COIN",
        "골드": "GOLD",
        "ISA": "ISA",
        "연금저축": "PENSION",
    }
    _mode_keys = list(_mode_map.keys())
    _mode_reverse = {v: k for k, v in _mode_map.items()}

    # query_params에서 저장된 모드 복원
    _qp = st.query_params
    _saved_mode = _qp.get("mode", "")
    _default_idx = 0
    if _saved_mode in _mode_reverse:
        _restored_label = _mode_reverse[_saved_mode]
        if _restored_label in _mode_keys:
            _default_idx = _mode_keys.index(_restored_label)

    _mode_label = st.sidebar.selectbox(
        "거래 모드",
        _mode_keys,
        index=_default_idx,
        key="trading_mode_label",
        label_visibility="collapsed",
    )
    trading_mode = _mode_map[_mode_label]

    # 사용자가 모드를 변경했을 때만 query_params 갱신 (최초 로드 시 rerun 방지)
    _prev_mode = st.session_state.get("_last_trading_mode", "")
    if _prev_mode and _prev_mode != trading_mode:
        st.query_params["mode"] = trading_mode
    elif not _saved_mode:
        # 최초 진입 시 URL에 모드 기록 (rerun 없이)
        st.query_params["mode"] = trading_mode
    st.session_state["_last_trading_mode"] = trading_mode

    # 텔레그램 알림 설정 (전 모드 공통)
    render_telegram_sidebar(prefix="global")

    if trading_mode == "GOLD":
        render_gold_mode()
        return
    if trading_mode == "ISA":
        render_kis_isa_mode()
        return
    if trading_mode == "PENSION":
        render_kis_pension_mode()
        return
    # === 코인 모드 (기존 코드) ===
    st.title("🪙 업비트 자동매매 시스템")

    # Sticky Header (JS로 Streamlit DOM 직접 조작)
    import streamlit.components.v1 as components
    components.html("""
    <script>
        const doc = window.parent.document;
        if (!doc.getElementById('sticky-title-style')) {
            const style = doc.createElement('style');
            style.id = 'sticky-title-style';
            style.textContent = `
                section[data-testid="stMain"] > div.block-container {
                    overflow: visible !important;
                }
                #sticky-title-wrap {
                    position: sticky;
                    top: 0;
                    background: white;
                    z-index: 999;
                    padding-bottom: 6px;
                    border-bottom: 2px solid #e6e6e6;
                }
            `;
            doc.head.appendChild(style);
        }

        function applySticky() {
            if (doc.getElementById('sticky-title-wrap')) return;
            const titles = doc.querySelectorAll('h1');
            for (const h1 of titles) {
                if (h1.textContent.includes('Upbit SMA')) {
                    const wrapper = h1.closest('[data-testid="stVerticalBlockBorderWrapper"]')
                                  || h1.parentElement?.parentElement;
                    if (wrapper) {
                        wrapper.id = 'sticky-title-wrap';
                    }
                    break;
                }
            }
        }
        applySticky();
        setTimeout(applySticky, 500);
        setTimeout(applySticky, 1500);
    </script>
    """, height=0)

    # --- Sidebar: Configuration ---
    st.sidebar.header("설정")
    
    # API Keys (Streamlit Cloud secrets 또는 .env 지원)
    try:
        env_access = st.secrets["UPBIT_ACCESS_KEY"]
        env_secret = st.secrets["UPBIT_SECRET_KEY"]
    except Exception:
        env_access = os.getenv("UPBIT_ACCESS_KEY")
        env_secret = os.getenv("UPBIT_SECRET_KEY")
    
    if IS_CLOUD:
        # Cloud: secrets에서 자동 로드, 편집 불가
        current_ak = env_access
        current_sk = env_secret
        st.sidebar.info("📱 조회 전용 모드 (Cloud)")
    else:
        with st.sidebar.expander("API 키", expanded=False):
            ak_input = st.text_input("Access Key", value=env_access if env_access else "", type="password")
            sk_input = st.text_input("Secret Key", value=env_secret if env_secret else "", type="password")
            current_ak = ak_input if ak_input else env_access
            current_sk = sk_input if sk_input else env_secret

    # 포트폴리오 관리
    st.sidebar.subheader("포트폴리오")
    st.sidebar.caption("메인 행의 [+]를 체크하면 바로 아래에 보조 행이 추가됩니다.")

    # Interval mapping (UI label -> API key)
    INTERVAL_MAP = {
        "1D": "day",
        "4H": "minute240",
        "1H": "minute60",
        "30m": "minute30",
        "15m": "minute15",
        "5m": "minute5",
        "1m": "minute1",
    }
    INTERVAL_REV_MAP = {v: k for k, v in INTERVAL_MAP.items()}
    CANDLES_PER_DAY = {
        "day": 1, "minute240": 6, "minute60": 24,
        "minute30": 48, "minute15": 96, "minute5": 288, "minute1": 1440,
    }
    ROW_TYPE_MAIN = "메인"
    ROW_TYPE_AUX = "보조"
    STRATEGY_AUX = "보조"

    def _is_aux_row(row_type_val="", strategy_val="", force_aux=False):
        if force_aux:
            return True
        rt = str(row_type_val or "").strip().lower()
        stg = str(strategy_val or "").strip().lower()
        return (rt in {"aux", "보조"}) or (stg in {"aux", "보조"})

    def _normalize_aux_ma_count_label(val) -> str:
        _v = str(val).strip().lower()
        if _v in {"1", "1개", "single", "one"}:
            return "1개"
        return "2개"

    def _apply_strategy_no(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        out = df.copy()
        main_no_map = {}
        next_no = 1

        # 1차: 메인 행 번호 부여
        for _, rr in out.iterrows():
            is_aux = _is_aux_row(rr.get("row_type", ""), rr.get("strategy", ""))
            if is_aux:
                continue
            rid = str(rr.get("row_id", "") or "").strip()
            pid = str(rr.get("parent_id", "") or "").strip()

            no = None
            if rid and rid in main_no_map:
                no = main_no_map[rid]
            elif pid and pid in main_no_map:
                no = main_no_map[pid]
            else:
                no = next_no
                next_no += 1

            if rid:
                main_no_map[rid] = no
            if pid:
                main_no_map[pid] = no

        # 2차: 전체 행(보조 포함) 번호 계산
        no_vals = []
        for _, rr in out.iterrows():
            is_aux = _is_aux_row(rr.get("row_type", ""), rr.get("strategy", ""))
            rid = str(rr.get("row_id", "") or "").strip()
            pid = str(rr.get("parent_id", "") or "").strip()

            if is_aux:
                no = main_no_map.get(pid) or main_no_map.get(rid)
            else:
                no = main_no_map.get(rid) or main_no_map.get(pid)

            no_vals.append("" if no is None else int(no))

        out["strategy_no"] = no_vals
        return out

    # Load portfolio from user config, then portfolio.json
    PORTFOLIO_JSON_LOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")
    _pjson_config = {}
    if os.path.exists(PORTFOLIO_JSON_LOAD):
        try:
            with open(PORTFOLIO_JSON_LOAD, "r", encoding="utf-8") as f:
                _pjson_raw = json.load(f)
            if isinstance(_pjson_raw, dict):
                _pjson_config = _pjson_raw
            elif isinstance(_pjson_raw, list):
                _pjson_config = {"portfolio": _pjson_raw}
        except Exception:
            pass

    default_portfolio = config.get("portfolio", None) or _pjson_config.get("portfolio", None)
    default_aux_portfolio = config.get("aux_portfolio", None)
    if default_aux_portfolio is None:
        default_aux_portfolio = _pjson_config.get("aux_portfolio", [])

    if not default_portfolio:
        st.error("portfolio.json에 포트폴리오 데이터가 없습니다.")
        st.stop()

    merged_default = list(default_portfolio)
    for ax in (default_aux_portfolio or []):
        ax_row = dict(ax)
        ax_row["is_aux"] = True
        merged_default.append(ax_row)

    # Normalize rows for sidebar editor
    sanitized_portfolio = []
    main_count = 0
    for p in merged_default:
        is_aux = _is_aux_row(
            p.get("row_type", ""),
            p.get("strategy", ""),
            force_aux=bool(p.get("is_aux", False)),
        )
        if not is_aux:
            main_count += 1
    if main_count <= 0:
        main_count = 1

    strat_map = {"SMA ??": "SMA", "??? ??": "Donchian", "Donchian Trend": "Donchian"}

    for idx, p in enumerate(merged_default):
        coin_val = str(p.get("coin", "BTC")).upper()
        is_aux = _is_aux_row(
            p.get("row_type", ""),
            p.get("strategy", ""),
            force_aux=bool(p.get("is_aux", False)),
        )
        row_type = ROW_TYPE_AUX if is_aux else ROW_TYPE_MAIN

        api_interval = p.get("interval", "day")
        label_interval = INTERVAL_REV_MAP.get(api_interval, "1D")

        strat_val = p.get("strategy", "SMA")
        strat_val = strat_map.get(strat_val, strat_val)
        if is_aux:
            strat_val = STRATEGY_AUX

        row_id = str(p.get("row_id", "")) or f"{row_type}_{coin_val}_{idx}"
        parent_id = str(p.get("parent_id", ""))
        if (not parent_id) and (not is_aux):
            parent_id = row_id

        parameter = None if is_aux else int(p.get("parameter", p.get("sma", 20)) or 20)
        sell_param = None if is_aux else int(p.get("sell_parameter", 0) or 0)
        if is_aux:
            weight_val = None
        else:
            weight_val = float(p.get("weight", (100 // main_count)) or 0)

        sanitized_portfolio.append({
            "add_aux": False,
            "row_type": row_type,
            "coin": coin_val,
            "strategy": strat_val,
            "parameter": parameter,
            "sell_parameter": sell_param,
            "weight": weight_val,
            "interval": label_interval,
            "aux_ma_count": _normalize_aux_ma_count_label(p.get("aux_ma_count", 2)),
            "aux_ma_short": int(p.get("aux_ma_short", 5) or 5),
            "aux_ma_long": int(p.get("aux_ma_long", 20) or 20),
            "aux_threshold": float(p.get("aux_threshold", -5.0) or -5.0),
            "aux_tp1": float(p.get("aux_tp1", 3.0) or 3.0),
            "aux_tp2": float(p.get("aux_tp2", 10.0) or 10.0),
            "aux_split": int(p.get("aux_split", 3) or 3),
            "aux_seed_mode": {"equal": "균등", "pyramiding": "피라미딩"}.get(str(p.get("aux_seed_mode", "equal") or "equal"), "균등"),
            "aux_pyramid_ratio": float(p.get("aux_pyramid_ratio", 1.3) or 1.3),
            "row_id": row_id,
            "parent_id": parent_id,
        })

    df_portfolio = pd.DataFrame(sanitized_portfolio)
    editor_columns = [
        "add_aux", "strategy_no", "row_type", "coin", "strategy", "parameter", "sell_parameter", "weight", "interval",
        "aux_ma_count", "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio",
        "row_id", "parent_id",
    ]
    for c in editor_columns:
        if c not in df_portfolio.columns:
            df_portfolio[c] = "" if c in {"strategy_no", "row_type", "coin", "strategy", "interval", "aux_ma_count", "aux_seed_mode", "row_id", "parent_id"} else 0
    if "aux_ma_count" in df_portfolio.columns:
        df_portfolio["aux_ma_count"] = df_portfolio["aux_ma_count"].apply(_normalize_aux_ma_count_label)
    df_portfolio = df_portfolio[editor_columns]
    df_portfolio = _apply_strategy_no(df_portfolio)

    interval_options = list(INTERVAL_MAP.keys())
    strategy_options = ["SMA", "Donchian", STRATEGY_AUX]

    editor_state_key = "portfolio_editor_df"
    source_df = st.session_state.get(editor_state_key)
    if not isinstance(source_df, pd.DataFrame) or list(source_df.columns) != editor_columns:
        source_df = df_portfolio.copy()
    if "aux_ma_count" in source_df.columns:
        source_df["aux_ma_count"] = source_df["aux_ma_count"].apply(_normalize_aux_ma_count_label)
    source_df = _apply_strategy_no(source_df)

    _aux_cols = ["aux_ma_count", "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio"]
    _main_display_cols = [c for c in source_df.columns if c not in _aux_cols and c not in ("row_id", "parent_id")]

    if IS_CLOUD:
        st.sidebar.dataframe(source_df[_main_display_cols], use_container_width=True, hide_index=True)
        edited_portfolio = source_df.copy()
    else:
        _editor_df = source_df.drop(columns=["row_id", "parent_id"])
        edited_portfolio = st.sidebar.data_editor(
            _editor_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="portfolio_editor_widget",
            column_order=_main_display_cols,
            column_config={
                "add_aux": st.column_config.CheckboxColumn("+", help="메인 행 아래에 보조 행 추가", default=False),
                "strategy_no": st.column_config.TextColumn("번호", disabled=True),
                "row_type": st.column_config.TextColumn("유형", disabled=True),
                "coin": st.column_config.TextColumn("코인", required=True),
                "strategy": st.column_config.SelectboxColumn("전략", options=strategy_options, required=True, default="SMA"),
                "parameter": st.column_config.NumberColumn("매수", min_value=0, max_value=300, step=1, required=False),
                "sell_parameter": st.column_config.NumberColumn("매도", min_value=0, max_value=300, step=1, required=False),
                "weight": st.column_config.NumberColumn("비중", min_value=0, max_value=100, step=1, required=False, format="%d%%"),
                "interval": st.column_config.SelectboxColumn("주기", options=interval_options, required=True, default="1D"),
            },
        )

        # Re-attach internal columns from source_df by position (best effort)
        src_internal = source_df[["row_id", "parent_id"]].reset_index(drop=True)
        edited_portfolio = edited_portfolio.reset_index(drop=True)
        if len(src_internal) >= len(edited_portfolio):
            edited_portfolio["row_id"] = src_internal.loc[:len(edited_portfolio)-1, "row_id"].values
            edited_portfolio["parent_id"] = src_internal.loc[:len(edited_portfolio)-1, "parent_id"].values
        else:
            edited_portfolio["row_id"] = [""] * len(edited_portfolio)
            edited_portfolio["parent_id"] = [""] * len(edited_portfolio)

        # Add AUX row right below main row when [+] checked
        rows = edited_portfolio.to_dict("records")
        existing_aux_parent = set()
        for rr in rows:
            if _is_aux_row(rr.get("row_type", ""), rr.get("strategy", "")):
                existing_aux_parent.add(str(rr.get("parent_id", "")))

        new_rows = []
        added_any = False
        for ridx, r in enumerate(rows):
            rr = dict(r)
            rr["coin"] = str(rr.get("coin", "BTC")).upper()
            rr["row_type"] = ROW_TYPE_AUX if _is_aux_row(rr.get("row_type", ""), rr.get("strategy", "")) else ROW_TYPE_MAIN
            rr["strategy"] = STRATEGY_AUX if rr["row_type"] == ROW_TYPE_AUX else str(rr.get("strategy", "SMA"))
            rr["interval"] = str(rr.get("interval", "1D"))
            rr["row_id"] = str(rr.get("row_id", "")) or f"{rr['row_type']}_{rr['coin']}_{ridx}"
            if rr["row_type"] == ROW_TYPE_MAIN:
                rr["parent_id"] = rr["row_id"]
                if pd.isna(rr.get("parameter", None)):
                    rr["parameter"] = 20
                if pd.isna(rr.get("sell_parameter", None)):
                    rr["sell_parameter"] = 0
            else:
                rr["parent_id"] = str(rr.get("parent_id", ""))
                rr["weight"] = None
                rr["parameter"] = None
                rr["sell_parameter"] = None

            trigger_add = bool(rr.get("add_aux", False))
            rr["add_aux"] = False
            new_rows.append(rr)

            if rr["row_type"] == ROW_TYPE_MAIN and trigger_add:
                parent_id = rr["row_id"]
                has_aux_already = (parent_id in existing_aux_parent) or any(
                    _is_aux_row(x.get("row_type", ""), x.get("strategy", "")) and str(x.get("parent_id", "")) == parent_id
                    for x in new_rows
                )
                if has_aux_already:
                    st.sidebar.info(f"{rr['coin']} / {rr.get('strategy', '')} 아래에 이미 보조 행이 있습니다.")
                else:
                    aux_row = {
                        "add_aux": False,
                        "row_type": ROW_TYPE_AUX,
                        "coin": rr["coin"],
                        "strategy": STRATEGY_AUX,
                        "parameter": None,
                        "sell_parameter": None,
                        "weight": None,
                        "interval": "1H",
                        "aux_ma_count": "2개",
                        "aux_ma_short": int(rr.get("aux_ma_short", 5) or 5),
                        "aux_ma_long": max(int(rr.get("aux_ma_long", 20) or 20), int(rr.get("aux_ma_short", 5) or 5) + 1),
                        "aux_threshold": float(rr.get("aux_threshold", -5.0) or -5.0),
                        "aux_tp1": float(rr.get("aux_tp1", 3.0) or 3.0),
                        "aux_tp2": max(float(rr.get("aux_tp2", 10.0) or 10.0), float(rr.get("aux_tp1", 3.0) or 3.0)),
                        "aux_split": int(rr.get("aux_split", 3) or 3),
                        "aux_seed_mode": "균등",
                        "aux_pyramid_ratio": float(rr.get("aux_pyramid_ratio", 1.3) or 1.3),
                        "row_id": f"aux_{rr['coin']}_{ridx}",
                        "parent_id": parent_id,
                    }
                    new_rows.append(aux_row)
                    added_any = True

        edited_portfolio = pd.DataFrame(new_rows)
        for c in editor_columns:
            if c not in edited_portfolio.columns:
                edited_portfolio[c] = "" if c in {"strategy_no", "row_type", "coin", "strategy", "interval", "aux_ma_count", "aux_seed_mode", "row_id", "parent_id"} else 0
        if "aux_ma_count" in edited_portfolio.columns:
            edited_portfolio["aux_ma_count"] = edited_portfolio["aux_ma_count"].apply(_normalize_aux_ma_count_label)
        edited_portfolio = edited_portfolio[editor_columns]
        edited_portfolio = _apply_strategy_no(edited_portfolio)

        st.session_state[editor_state_key] = edited_portfolio.copy()
        if added_any:
            st.rerun()

    edited_portfolio = _apply_strategy_no(edited_portfolio)

    # ── 보조 전략 파라미터 편집 (별도 expander + data_editor) ──
    _aux_mask = edited_portfolio.apply(
        lambda _r: _is_aux_row(_r.get("row_type", ""), _r.get("strategy", "")),
        axis=1,
    )
    if _aux_mask.any() and not IS_CLOUD:
        with st.sidebar.expander("⚙️ 보조 전략 설정", expanded=True):
            st.caption("이평선 수가 1개일 때는 장기MA 값이 무시되고 단기MA 이격도만 사용됩니다.")
            _aux_display_cols = ["strategy_no", "coin", "aux_ma_count", "aux_ma_short", "aux_ma_long", "aux_threshold", "aux_tp1", "aux_tp2", "aux_split", "aux_seed_mode", "aux_pyramid_ratio"]
            _aux_df = edited_portfolio.loc[_aux_mask, _aux_display_cols].copy()
            _aux_df = _aux_df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
            _aux_edit = st.data_editor(
                _aux_df.drop(columns=["_orig_idx"]),
                use_container_width=True,
                hide_index=True,
                key="aux_param_editor",
                column_config={
                    "strategy_no": st.column_config.TextColumn("번호", disabled=True),
                    "coin": st.column_config.TextColumn("코인", disabled=True),
                    "aux_ma_count": st.column_config.SelectboxColumn("이평선 수", options=["1개", "2개"], required=True),
                    "aux_ma_short": st.column_config.NumberColumn("단기MA", min_value=2, max_value=500, step=1),
                    "aux_ma_long": st.column_config.NumberColumn("장기MA", min_value=3, max_value=500, step=1),
                    "aux_threshold": st.column_config.NumberColumn("임계(%)", min_value=-50.0, max_value=0.0, step=0.5, format="%.1f"),
                    "aux_tp1": st.column_config.NumberColumn("TP1(%)", min_value=0.1, max_value=50.0, step=0.5, format="%.1f"),
                    "aux_tp2": st.column_config.NumberColumn("TP2(%)", min_value=0.1, max_value=100.0, step=0.5, format="%.1f"),
                    "aux_split": st.column_config.NumberColumn("매수분할", min_value=1, max_value=10, step=1),
                    "aux_seed_mode": st.column_config.SelectboxColumn("시드모드", options=["균등", "피라미딩"]),
                    "aux_pyramid_ratio": st.column_config.NumberColumn("피라미딩비율", min_value=1.0, max_value=5.0, step=0.1, format="%.1f"),
                },
            )
            # 변경사항을 edited_portfolio에 반영
            _aux_orig_indices = []
            for _ri in range(len(_aux_edit)):
                _orig = int(_aux_df.iloc[_ri]["_orig_idx"])
                _aux_orig_indices.append(_orig)
                for _col in _aux_display_cols:
                    if _col in {"strategy_no", "coin"}:
                        continue
                    edited_portfolio.at[_orig, _col] = _aux_edit.iloc[_ri][_col]
            for _orig in _aux_orig_indices:
                _ma_count_label = _normalize_aux_ma_count_label(edited_portfolio.at[_orig, "aux_ma_count"])
                edited_portfolio.at[_orig, "aux_ma_count"] = _ma_count_label

                _ms_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_ma_short"], errors="coerce")
                _ms = int(_ms_raw) if pd.notna(_ms_raw) else 5
                _ms = max(2, _ms)

                _ml_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_ma_long"], errors="coerce")
                _ml = int(_ml_raw) if pd.notna(_ml_raw) else 20
                _ml = max(3, _ml)
                if _ma_count_label == "2개" and _ml <= _ms:
                    _ml = _ms + 1
                if _ma_count_label == "1개":
                    _ml = _ms

                _tp1_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_tp1"], errors="coerce")
                _tp2_raw = pd.to_numeric(edited_portfolio.at[_orig, "aux_tp2"], errors="coerce")
                _tp1 = float(_tp1_raw) if pd.notna(_tp1_raw) else 3.0
                _tp2 = float(_tp2_raw) if pd.notna(_tp2_raw) else 10.0
                if _tp2 < _tp1:
                    _tp2 = _tp1

                edited_portfolio.at[_orig, "aux_ma_short"] = _ms
                edited_portfolio.at[_orig, "aux_ma_long"] = _ml
                edited_portfolio.at[_orig, "aux_tp1"] = _tp1
                edited_portfolio.at[_orig, "aux_tp2"] = _tp2
            edited_portfolio = _apply_strategy_no(edited_portfolio)
            st.session_state[editor_state_key] = edited_portfolio.copy()

    # Calculate total weight from main rows only
    _main_rows_df = edited_portfolio[~_aux_mask]
    total_weight = pd.to_numeric(_main_rows_df["weight"], errors="coerce").fillna(0).sum()
    if total_weight > 100:
        st.sidebar.error(f"총 비중이 {total_weight}%입니다 (100% 이하여야 합니다)")
    else:
        cash_weight = 100 - total_weight
        st.sidebar.info(f"투자: {total_weight}% | 현금: {cash_weight}%")

    # Convert back to dict lists
    portfolio_list = []
    aux_portfolio_list = []
    for r in edited_portfolio.to_dict('records'):
        label_key = r.get('interval', '1D')
        api_key = INTERVAL_MAP.get(label_key, 'day')
        row_is_aux = _is_aux_row(r.get('row_type', ''), r.get('strategy', ''))

        coin_val = str(r.get('coin', 'BTC')).upper().strip()
        if not coin_val:
            continue

        if row_is_aux:
            _ma_count_label = _normalize_aux_ma_count_label(r.get('aux_ma_count', 2))
            _ma_short_raw = pd.to_numeric(r.get('aux_ma_short', 5), errors='coerce')
            _ma_long_raw = pd.to_numeric(r.get('aux_ma_long', 20), errors='coerce')
            _ma_short = int(_ma_short_raw) if pd.notna(_ma_short_raw) else 5
            _ma_long = int(_ma_long_raw) if pd.notna(_ma_long_raw) else 20
            _ma_short = max(2, _ma_short)
            _ma_long = max(3, _ma_long)
            if _ma_count_label == "2개" and _ma_long <= _ma_short:
                _ma_long = _ma_short + 1
            if _ma_count_label == "1개":
                _ma_long = _ma_short
            aux_portfolio_list.append({
                'coin': coin_val,
                'interval': api_key,
                'parent_id': str(r.get('parent_id', '')),
                'aux_ma_count': 1 if _ma_count_label == "1개" else 2,
                'aux_ma_short': _ma_short,
                'aux_ma_long': _ma_long,
                'aux_threshold': float(r.get('aux_threshold', -5.0) or -5.0),
                'aux_tp1': float(r.get('aux_tp1', 3.0) or 3.0),
                'aux_tp2': float(r.get('aux_tp2', 10.0) or 10.0),
                'aux_split': int(r.get('aux_split', 3) or 3),
                'aux_seed_mode': {"균등": "equal", "피라미딩": "pyramiding"}.get(str(r.get('aux_seed_mode', '균등') or '균등'), 'equal'),
                'aux_pyramid_ratio': float(r.get('aux_pyramid_ratio', 1.3) or 1.3),
            })
            continue

        param_raw = r.get('parameter', 20)
        if pd.isna(param_raw) or str(param_raw).strip() == "":
            param_val = 20
        else:
            param_val = int(float(param_raw))

        sell_raw = r.get('sell_parameter', 0)
        if pd.isna(sell_raw) or str(sell_raw).strip() == "":
            sell_p = 0
        else:
            sell_p = int(float(sell_raw))

        weight_raw = r.get('weight', 0)
        if pd.isna(weight_raw) or str(weight_raw).strip() == "":
            weight_val = 0.0
        else:
            weight_val = float(weight_raw)
        portfolio_list.append({
            'market': 'KRW',
            'coin': coin_val,
            'strategy': str(r.get('strategy', 'SMA')),
            'parameter': param_val,
            'sell_parameter': sell_p,
            'weight': weight_val,
            'interval': api_key,
        })

    # Global Settings
    st.sidebar.subheader("공통 설정")
    # Interval Removed (Per-Coin Setting)
    
    default_start_str = config.get("start_date", None) or _pjson_config.get("start_date", None)
    if not default_start_str:
        st.error("start_date 설정이 없습니다. 로컬에서 portfolio.json에 start_date를 설정 후 push 해주세요.")
        st.stop()
    try:
        default_start = pd.to_datetime(default_start_str).date()
    except:
        st.error(f"start_date 형식 오류: {default_start_str}")
        st.stop()
    start_date = st.sidebar.date_input(
        "기준 시작일",
        value=default_start,
        help="수익률 계산 및 이론적 자산 비교를 위한 기준일입니다. 실제 매매 신호와는 무관합니다.",
        disabled=IS_CLOUD
    )

    # Capital Input Customization
    default_cap = config.get("initial_cap", None) or _pjson_config.get("initial_cap", None)
    if not default_cap:
        st.error("initial_cap 설정이 없습니다. 로컬에서 portfolio.json에 initial_cap을 설정 후 push 해주세요.")
        st.stop()
    initial_cap = st.sidebar.number_input(
        "초기 자본금 (KRW - 원 단위)",
        value=default_cap, step=100000, format="%d",
        help="시뮬레이션을 위한 초기 투자금 설정입니다. 실제 계좌 잔고와는 무관하며, 수익률 계산의 기준이 됩니다.",
        disabled=IS_CLOUD
    )
    st.sidebar.caption(f"설정: **{initial_cap:,.0f} KRW**")
    
    # Strategy Selection REMOVED (Moved to Per-Coin)

    PORTFOLIO_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")

    if not IS_CLOUD:
        if st.sidebar.button("Save"):
            new_config = {
                "portfolio": portfolio_list,
                "aux_portfolio": aux_portfolio_list,
                "start_date": str(start_date),
                "initial_cap": initial_cap,
            }
            save_config(new_config)
            portfolio_json_data = {
                "portfolio": portfolio_list,
                "aux_portfolio": aux_portfolio_list,
                "start_date": str(start_date),
                "initial_cap": initial_cap,
            }
            with open(PORTFOLIO_JSON, "w", encoding="utf-8") as f:
                json.dump(portfolio_json_data, f, indent=2, ensure_ascii=False)
            st.sidebar.success("Saved")

    # --- data_manager Import ---
    from data_manager import MarketDataWorker

    # ... (Keep existing history cache if useful, or move to worker too. Let's keep separate for now)
    @st.cache_data(ttl=60)
    # Function to fetch history (Caching disabled for now due to obj hashing)
    def fetch_history_cached(_trader, kind, currency="KRW"):
         try:
            return _trader.get_history(kind, currency)
         except TypeError:
            # Fallback if get_history signature issues
            return _trader.get_history(kind)
    
    # Initialize Objects
    backtest_engine = BacktestEngine()

    @st.cache_data(ttl=300)
    def _cached_backtest(ticker, period, interval, count, start_date_str, initial_balance, strategy_mode, sell_period_ratio, _df_hash):
        """백테스트 결과를 5분간 캐싱 (동일 파라미터 재계산 방지)"""
        df_bt_local = data_cache.load_cached(ticker, interval)
        if df_bt_local is None or len(df_bt_local) < period:
            return None
        return backtest_engine.run_backtest(
            ticker, period=period, interval=interval, count=count,
            start_date=start_date_str, initial_balance=initial_balance,
            df=df_bt_local, strategy_mode=strategy_mode,
            sell_period_ratio=sell_period_ratio
        )
    
    trader = None
    if current_ak and current_sk:
        @st.cache_resource
        def get_trader(ak, sk):
            return UpbitTrader(ak, sk)
        trader = get_trader(current_ak, current_sk)

    # --- Background Worker Setup ---
    from data_manager import CoinTradingWorker

    @st.cache_resource
    def get_worker():
        return MarketDataWorker()

    @st.cache_resource
    def get_coin_trading_worker():
        w = CoinTradingWorker()
        w.start()
        return w

    worker = get_worker()
    _ = get_coin_trading_worker()

    # 업비트 KRW 마켓 호가 단위 (Tick Size)
    def get_tick_size(price):
        """가격에 따른 업비트 호가 단위 반환"""
        if price >= 2_000_000: return 1000
        elif price >= 1_000_000: return 1000
        elif price >= 500_000: return 500
        elif price >= 100_000: return 100
        elif price >= 50_000: return 50
        elif price >= 10_000: return 10
        elif price >= 5_000: return 5
        elif price >= 1_000: return 1
        elif price >= 100: return 1
        elif price >= 10: return 0.1
        elif price >= 1: return 0.01
        else: return 0.001

    def align_price(price, tick_size):
        """가격을 호가 단위에 맞게 정렬"""
        if tick_size >= 1:
            return int(price // tick_size * tick_size)
        else:
            import math
            decimals = max(0, -int(math.floor(math.log10(tick_size))))
            return round(price // tick_size * tick_size, decimals)

    # ── TTL 캐시: API 호출 최소화 ──
    def _ttl_cache(key, fn, ttl=5):
        """세션 기반 TTL 캐시. ttl초 이내 재호출시 캐시 반환."""
        now = time.time()
        ck, tk = f"__c_{key}", f"__t_{key}"
        if ck in st.session_state and (now - st.session_state.get(tk, 0)) < ttl:
            return st.session_state[ck]
        val = fn()
        st.session_state[ck] = val
        st.session_state[tk] = now
        return val

    def _clear_cache(*keys):
        """거래 후 캐시 무효화"""
        for k in keys:
            st.session_state.pop(f"__c_{k}", None)
            st.session_state.pop(f"__t_{k}", None)

    # 시가총액 상위 20 티커 (글로벌 Market Cap 기준)
    TOP_20_TICKERS = [
        "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE",
        "KRW-ADA", "KRW-SHIB", "KRW-TRX", "KRW-AVAX", "KRW-LINK",
        "KRW-BCH", "KRW-DOT", "KRW-NEAR", "KRW-POL", "KRW-ETC",
        "KRW-XLM", "KRW-STX", "KRW-HBAR", "KRW-EOS", "KRW-SAND"
    ]

    # --- Tabs ---
    tab1, tab5, tab3, tab4 = st.tabs(["🚀 실시간 포트폴리오", "🛒 수동 주문", "📜 거래 내역", "📊 백테스트"])

    # --- Tab 1: Live Portfolio (Default) ---
    with tab1:
        st.header("실시간 포트폴리오 대시보드")
        st.caption("설정된 모든 자산을 모니터링합니다.")
        
        if not trader:
            st.warning("사이드바에서 API 키를 설정해주세요.")
        else:
            # Configure and Start Worker
            worker.update_config(portfolio_list)
            worker.start_worker()

            w_msg, w_time = worker.get_status()

            # Control Bar
            col_ctrl1, col_ctrl2 = st.columns([1,3])
            with col_ctrl1:
                if st.button("🔄 새로고침"):
                    _clear_cache("krw_bal_t1", "prices_t1", "balances_t1")
                    st.rerun()
            with col_ctrl2:
                st.info(f"워커 상태: **{w_msg}**")

            if not portfolio_list:
                st.warning("사이드바에서 포트폴리오에 코인을 추가해주세요.")
            else:
                count = len(portfolio_list)
                per_coin_cap = initial_cap / count

                # ── 일괄 API 호출 (TTL 캐시): 가격·잔고를 1회씩만 가져옴 ──
                unique_coins = list(dict.fromkeys(item['coin'].upper() for item in portfolio_list))
                unique_tickers = list(dict.fromkeys(f"{item['market']}-{item['coin'].upper()}" for item in portfolio_list))

                krw_bal = _ttl_cache("krw_bal_t1", lambda: trader.get_balance("KRW") or 0, ttl=10)

                def _fetch_all_prices():
                    """모든 코인 가격을 한번에 가져옴"""
                    return data_cache.get_current_prices_local_first(
                        unique_tickers,
                        ttl_sec=5.0,
                        allow_api_fallback=True,
                    )

                all_prices = _ttl_cache("prices_t1", _fetch_all_prices, ttl=5)

                def _fetch_all_balances():
                    """모든 코인 잔고를 1회 API 호출로 가져옴"""
                    if hasattr(trader, 'get_all_balances'):
                        raw = trader.get_all_balances()
                        return {c: raw.get(c, 0) for c in unique_coins}
                    # 폴백: 개별 호출
                    return {c: (trader.get_balance(c) or 0) for c in unique_coins}

                all_balances = _ttl_cache("balances_t1", _fetch_all_balances, ttl=10)

                # --- Total Summary Container ---
                st.subheader("🏁 포트폴리오 요약")
                st.caption(f"초기자본: {initial_cap:,.0f} KRW | 자산수: {count} | 자산당: {per_coin_cap:,.0f} KRW")

                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

                total_real_val = krw_bal
                total_init_val = initial_cap

                # Cash Logic
                total_weight_alloc = sum([item.get('weight', 0) for item in portfolio_list])
                cash_ratio = max(0, 100 - total_weight_alloc) / 100.0
                reserved_cash = initial_cap * cash_ratio

                # Add reserved cash to Theo Value (as it stays as cash)
                total_theo_val = reserved_cash

                # --- 전체 자산 현황 테이블 (캐시된 데이터 사용) ---
                asset_summary_rows = [{"자산": "KRW (현금)", "보유량": f"{krw_bal:,.0f}", "현재가": "-", "평가금액(KRW)": f"{krw_bal:,.0f}", "상태": "-"}]
                seen_coins_summary = set()
                for s_item in portfolio_list:
                    s_coin = s_item['coin'].upper()
                    if s_coin in seen_coins_summary:
                        continue
                    seen_coins_summary.add(s_coin)
                    s_ticker = f"{s_item['market']}-{s_coin}"
                    s_bal = all_balances.get(s_coin, 0)
                    s_price = all_prices.get(s_ticker, 0) or 0
                    s_val = s_bal * s_price
                    is_holding = s_val >= 5000
                    asset_summary_rows.append({
                        "자산": s_coin,
                        "보유량": (f"{s_bal:.8f}" if s_bal < 1 else f"{s_bal:,.4f}") if s_bal > 0 else "0",
                        "현재가": f"{s_price:,.0f}",
                        "평가금액(KRW)": f"{s_val:,.0f}",
                        "상태": "보유중" if is_holding else "미보유",
                    })
                total_real_summary = krw_bal + sum(
                    all_balances.get(c, 0) * (all_prices.get(f"KRW-{c}", 0) or 0)
                    for c in seen_coins_summary
                )
                asset_summary_rows.append({
                    "자산": "합계",
                    "보유량": "",
                    "현재가": "",
                    "평가금액(KRW)": f"{total_real_summary:,.0f}",
                    "상태": "",
                })
                with st.expander(f"💰 전체 자산 현황 (Total: {total_real_summary:,.0f} KRW)", expanded=True):
                    st.dataframe(pd.DataFrame(asset_summary_rows), use_container_width=True, hide_index=True)

                    # ── 포트폴리오 리밸런싱 (자산현황 내 통합) ──
                    st.divider()
                    st.markdown("**⚖️ 포트폴리오 리밸런싱**")
                    krw_balance = krw_bal

                    asset_states = []
                    for rb_idx, rb_item in enumerate(portfolio_list):
                        rb_ticker = f"{rb_item['market']}-{rb_item['coin'].upper()}"
                        rb_coin = rb_item['coin'].upper()
                        rb_weight = rb_item.get('weight', 0)
                        rb_interval = rb_item.get('interval', 'day')
                        rb_strategy = rb_item.get('strategy', 'SMA Strategy')
                        rb_param = rb_item.get('parameter', 20)
                        rb_sell_param = rb_item.get('sell_parameter', 0)

                        rb_coin_bal = all_balances.get(rb_coin, 0)
                        rb_price = all_prices.get(rb_ticker, 0) or 0
                        rb_coin_val = rb_coin_bal * rb_price
                        rb_status = "HOLD" if rb_coin_val > 5000 else "CASH"

                        rb_signal = "N/A"
                        try:
                            rb_df = worker.get_data(rb_ticker, rb_interval)
                            if rb_df is not None and len(rb_df) >= rb_param:
                                if rb_strategy == "Donchian":
                                    rb_eng = DonchianStrategy()
                                    rb_sp = rb_sell_param or max(5, rb_param // 2)
                                    rb_df = rb_eng.create_features(rb_df, buy_period=rb_param, sell_period=rb_sp)
                                    rb_signal = rb_eng.get_signal(rb_df.iloc[-2], buy_period=rb_param, sell_period=rb_sp)
                                else:
                                    rb_eng = SMAStrategy()
                                    rb_df = rb_eng.create_features(rb_df, periods=[rb_param])
                                    rb_signal = rb_eng.get_signal(rb_df.iloc[-2], strategy_type='SMA_CROSS', ma_period=rb_param)
                        except Exception:
                            pass

                        rb_target_krw = total_real_summary * (rb_weight / 100.0)

                        asset_states.append({
                            "ticker": rb_ticker, "coin": rb_coin, "weight": rb_weight,
                            "interval": rb_interval, "strategy": rb_strategy,
                            "param": rb_param, "sell_param": rb_sell_param,
                            "status": rb_status, "signal": rb_signal,
                            "coin_bal": rb_coin_bal, "coin_val": rb_coin_val,
                            "price": rb_price, "target_krw": rb_target_krw,
                        })

                    cash_assets = [a for a in asset_states if a['status'] == 'CASH']
                    buy_signal_assets = [a for a in asset_states if a['signal'] == 'BUY']

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("보유 현금 (KRW)", f"{krw_balance:,.0f}")
                    rc2.metric("CASH 자산", f"{len(cash_assets)} / {len(asset_states)}")
                    rc3.metric("BUY 시그널", f"{len(buy_signal_assets)} / {len(asset_states)}")

                    rebal_data = []
                    for a in asset_states:
                        action = ""
                        if a['status'] == 'CASH' and a['signal'] == 'BUY':
                            action = "BUY"
                        elif a['status'] == 'CASH' and a['signal'] != 'BUY':
                            action = "대기 (시그널 없음)"
                        elif a['status'] == 'HOLD':
                            action = "보유 중"
                        rebal_data.append({
                            "종목": a['ticker'],
                            "전략": f"{a['strategy']} {a['param']}",
                            "비중": f"{a['weight']}%",
                            "시간봉": a['interval'],
                            "상태": a['status'],
                            "시그널": a['signal'],
                            "현재가치(KRW)": f"{a['coin_val']:,.0f}",
                            "목표(KRW)": f"{a['target_krw']:,.0f}",
                            "액션": action,
                        })
                    st.dataframe(pd.DataFrame(rebal_data), use_container_width=True, hide_index=True)

                    buyable = [a for a in asset_states if a['status'] == 'CASH' and a['signal'] == 'BUY']
                    if not buyable:
                        if len(cash_assets) == 0:
                            st.success("모든 자산이 이미 보유 중입니다.")
                        else:
                            st.info(f"현금 자산 {len(cash_assets)}개가 있지만 BUY 시그널이 없습니다. 시그널 발생 시 매수 가능합니다.")
                    else:
                        st.warning(f"**{len(buyable)}개 자산**에 BUY 시그널이 있습니다. 리밸런싱 매수를 실행할 수 있습니다.")
                        total_buy_weight = sum(a['weight'] for a in buyable)
                        available_krw = krw_balance * 0.999

                        buy_plan = []
                        for a in buyable:
                            alloc_krw = available_krw * (a['weight'] / total_buy_weight) if total_buy_weight > 0 else 0
                            alloc_krw = min(alloc_krw, available_krw)
                            buy_plan.append({
                                "종목": a['ticker'], "비중": f"{a['weight']}%",
                                "배분 금액(KRW)": f"{alloc_krw:,.0f}",
                                "시간봉": a['interval'], "현재가": f"{a['price']:,.0f}",
                                "_ticker": a['ticker'], "_krw": alloc_krw, "_interval": a['interval'],
                            })
                        plan_df = pd.DataFrame(buy_plan)
                        st.dataframe(plan_df[["종목", "비중", "배분 금액(KRW)", "시간봉", "현재가"]], use_container_width=True, hide_index=True)
                        st.caption(f"총 배분 금액: {sum(p['_krw'] for p in buy_plan):,.0f} KRW / 보유 현금: {krw_balance:,.0f} KRW")

                        if st.button("🚀 리밸런싱 매수 실행", key="btn_rebalance_exec", type="primary"):
                            rebal_results = []
                            rebal_progress = st.progress(0)
                            rebal_log = st.empty()
                            for pi, plan in enumerate(buy_plan):
                                p_ticker = plan['_ticker']
                                p_krw = plan['_krw']
                                p_interval = plan['_interval']
                                if p_krw < 5000:
                                    rebal_results.append({"종목": p_ticker, "결과": "금액 부족 (5,000원 미만)"})
                                    continue
                                rebal_log.text(f"매수 중: {p_ticker} ({p_krw:,.0f} KRW)...")
                                try:
                                    exec_res = trader.smart_buy(p_ticker, p_krw, interval=p_interval)
                                    avg_p = exec_res.get('avg_price', 0)
                                    vol = exec_res.get('filled_volume', 0)
                                    rebal_results.append({
                                        "종목": p_ticker,
                                        "결과": f"체결 완료: {vol:.6f} @ {avg_p:,.0f}",
                                        "금액": f"{exec_res.get('total_krw', 0):,.0f} KRW"
                                    })
                                except Exception as e:
                                    rebal_results.append({"종목": p_ticker, "결과": f"오류: {e}"})
                                rebal_progress.progress((pi + 1) / len(buy_plan))
                                time.sleep(0.5)
                            rebal_progress.progress(1.0)
                            rebal_log.empty()
                            st.success("리밸런싱 완료!")
                            st.dataframe(pd.DataFrame(rebal_results), use_container_width=True, hide_index=True)

                # --- 단기 모니터링 차트 (60봉) ---
                with st.expander("📊 단기 시그널 모니터링 (60봉)", expanded=True):
                    signal_rows = []

                    # BTC / 비BTC 분리 (BTC: 일봉→4시간봉 순)
                    interval_order = {'day': 0, 'minute240': 1, 'minute60': 2, 'minute30': 3, 'minute15': 4, 'minute10': 5}
                    btc_items = sorted(
                        [x for x in portfolio_list if x.get('coin', '').upper() == 'BTC'],
                        key=lambda x: interval_order.get(x.get('interval', 'day'), 99)
                    )
                    other_items = sorted(
                        [x for x in portfolio_list if x.get('coin', '').upper() != 'BTC'],
                        key=lambda x: interval_order.get(x.get('interval', 'day'), 99)
                    )

                    # 차트 데이터 수집 + 렌더링 함수
                    def render_chart_row(items):
                        if not items:
                            return
                        cols = st.columns(len(items))
                        for ci, item in enumerate(items):
                            p_ticker = f"{item['market']}-{item['coin'].upper()}"
                            p_strategy = item.get('strategy', 'SMA')
                            p_param = item.get('parameter', 20)
                            p_sell_param = item.get('sell_parameter', 0) or max(5, p_param // 2)
                            p_interval = item.get('interval', 'day')
                            iv_label = INTERVAL_REV_MAP.get(p_interval, p_interval)

                            try:
                                # Worker 캐시 데이터 우선 사용 (API 호출 제거)
                                df_60 = worker.get_data(p_ticker, p_interval)
                                if df_60 is None or len(df_60) < p_param + 5:
                                    # Worker 데이터 없으면 TTL 캐시로 API 호출
                                    df_60 = _ttl_cache(
                                        f"ohlcv_{p_ticker}_{p_interval}",
                                        lambda t=p_ticker, iv=p_interval, pp=p_param: data_cache.get_ohlcv_local_first(
                                            t,
                                            interval=iv,
                                            count=max(60 + pp, 200),
                                            allow_api_fallback=True,
                                        ),
                                        ttl=30
                                    )
                                if df_60 is None or len(df_60) < p_param + 5:
                                    continue

                                close_now = df_60['close'].iloc[-1]

                                if p_strategy == "Donchian":
                                    upper_vals = df_60['high'].rolling(window=p_param).max().shift(1)
                                    lower_vals = df_60['low'].rolling(window=p_sell_param).min().shift(1)
                                    buy_target = upper_vals.iloc[-1]
                                    sell_target = lower_vals.iloc[-1]
                                    buy_dist = (close_now - buy_target) / buy_target * 100 if buy_target else 0
                                    sell_dist = (close_now - sell_target) / sell_target * 100 if sell_target else 0

                                    # 포지션 상태 시뮬레이션 (돈치안은 상태 기반)
                                    in_position = False
                                    for i in range(len(df_60)):
                                        u = upper_vals.iloc[i]
                                        l = lower_vals.iloc[i]
                                        c = df_60['close'].iloc[i]
                                        if not pd.isna(u) and c > u:
                                            in_position = True
                                        elif not pd.isna(l) and c < l:
                                            in_position = False

                                    if in_position:
                                        position_label = "보유"
                                        signal = "SELL" if close_now < sell_target else "HOLD"
                                    else:
                                        position_label = "현금"
                                        signal = "BUY" if close_now > buy_target else "WAIT"
                                else:
                                    sma_vals = df_60['close'].rolling(window=p_param).mean()
                                    buy_target = sma_vals.iloc[-1]
                                    sell_target = buy_target
                                    buy_dist = (close_now - buy_target) / buy_target * 100 if buy_target else 0
                                    sell_dist = buy_dist
                                    if close_now > buy_target:
                                        signal = "BUY"
                                        position_label = "보유"
                                    else:
                                        signal = "SELL"
                                        position_label = "현금"

                                signal_rows.append({
                                    "종목": p_ticker.replace("KRW-", ""),
                                    "전략": f"{p_strategy} {p_param}",
                                    "시간봉": iv_label,
                                    "포지션": position_label,
                                    "현재가": f"{close_now:,.0f}",
                                    "매수목표": f"{buy_target:,.0f}",
                                    "매도목표": f"{sell_target:,.0f}",
                                    "매수이격도": f"{buy_dist:+.2f}%",
                                    "매도이격도": f"{sell_dist:+.2f}%",
                                })

                                df_chart = df_60.iloc[-60:]
                                with cols[ci]:
                                    fig_m = go.Figure()
                                    fig_m.add_trace(go.Candlestick(
                                        x=df_chart.index, open=df_chart['open'],
                                        high=df_chart['high'], low=df_chart['low'],
                                        close=df_chart['close'], name='가격',
                                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                                    ))

                                    if p_strategy == "Donchian":
                                        upper_chart = upper_vals.loc[df_chart.index]
                                        lower_chart = lower_vals.loc[df_chart.index]
                                        fig_m.add_trace(go.Scatter(
                                            x=df_chart.index, y=upper_chart,
                                            name=f'상단({p_param})', line=dict(color='green', width=1, dash='dot')
                                        ))
                                        fig_m.add_trace(go.Scatter(
                                            x=df_chart.index, y=lower_chart,
                                            name=f'하단({p_sell_param})', line=dict(color='red', width=1, dash='dot')
                                        ))
                                    else:
                                        sma_chart = sma_vals.loc[df_chart.index]
                                        fig_m.add_trace(go.Scatter(
                                            x=df_chart.index, y=sma_chart,
                                            name=f'SMA({p_param})', line=dict(color='orange', width=2)
                                        ))

                                    sig_color = "green" if signal == "BUY" else ("red" if signal == "SELL" else ("blue" if signal == "WAIT" else "gray"))
                                    title_pos = f" [{position_label}]" if p_strategy == "Donchian" else ""
                                    fig_m.update_layout(
                                        title=f"{p_ticker.replace('KRW-','')} {p_strategy}{p_param} ({iv_label}){title_pos} [{buy_dist:+.1f}%]",
                                        title_font_color=sig_color,
                                        height=300, margin=dict(l=0, r=0, t=35, b=30),
                                        xaxis_rangeslider_visible=False,
                                        showlegend=False,
                                        xaxis=dict(showticklabels=True, tickformat='%m/%d %H:%M', tickangle=-45, nticks=6),
                                    )
                                    st.plotly_chart(fig_m, use_container_width=True)

                            except Exception as chart_err:
                                with cols[ci]:
                                    st.warning(f"{p_ticker} 데이터 로드 실패: {chart_err}")
                                continue

                    # 1행: BTC 전략 (일봉 → 4시간봉)
                    render_chart_row(btc_items)
                    # 2행: ETH, SOL 등
                    render_chart_row(other_items)

                    # 시그널 요약 테이블
                    if signal_rows:
                        df_sig = pd.DataFrame(signal_rows)
                        st.dataframe(df_sig, use_container_width=True, hide_index=True)

                # 리밸런싱 규칙 (항상 표시)
                with st.expander("⚖️ 리밸런싱 규칙", expanded=False):
                    st.markdown("""
**실행 시점**: GitHub Action 실행 시마다 (자동: 매일 09:05 KST / 수동 실행 가능)

**실행 순서**: 전체 시그널 분석 → 매도 먼저 실행 (현금 확보) → 현금 비례 배분 매수

**매매 판단** (전일 종가 기준)

| 현재 상태 | 시그널 | 실행 내용 |
|-----------|--------|-----------|
| 코인 미보유 | 매수 시그널 | **매수** — 현금에서 비중 비례 배분 |
| 코인 미보유 | 매도/중립 | **대기** — 현금 보존 (비중만큼 예비) |
| 코인 보유 중 | 매도 시그널 | **매도** — 전량 시장가 매도 |
| 코인 보유 중 | 매수/중립 | **유지** — 계속 보유 (추가 매수 없음) |

**매수 금액 계산**: 보유 중인 자산은 무시, 현금을 미보유 자산 비중끼리 비례 배분

> 예) BTC 40%(보유중), ETH 30%(미보유), SOL 30%(미보유)
> → 미보유 비중 합계 = 60%
> → ETH 매수액 = 현금 × 30/60, SOL 매수액 = 현금 × 30/60

**시그널 발생 조건**

| | 매수 시그널 | 매도 시그널 |
|---|---------|---------|
| **SMA** | 종가 > 이동평균선 | 종가 < 이동평균선 |
| **Donchian** | 종가 > N일 최고가 돌파 | 종가 < M일 최저가 이탈 |
""")

                # 합산 포트폴리오 자리 미리 확보 (데이터 수집 후 렌더링)
                combined_portfolio_container = st.container()

                st.write(f"### 📋 자산 상세 (현금 예비: {reserved_cash:,.0f} KRW)")

                # 포트폴리오 합산용 에쿼티 수집
                portfolio_equity_data = []  # [(label, equity_series, close_series, per_coin_cap, perf)]

                for asset_idx, item in enumerate(portfolio_list):
                    ticker = f"{item['market']}-{item['coin'].upper()}"
                    
                    # Per-Coin Strategy Settings
                    strategy_mode = item.get('strategy', 'SMA Strategy')
                    param_val = item.get('parameter', item.get('sma', 20)) # Backwards compat
                    
                    weight = item.get('weight', 0)
                    interval = item.get('interval', 'day')
                    
                    # Calculate Allocated Capital
                    per_coin_cap = initial_cap * (weight / 100.0)
                    
                    # Collapse by default to save rendering time
                    with st.expander(f"**{ticker}** ({strategy_mode} {param_val}, {weight}%, {interval})", expanded=False):
                        try:
                            # 1. Get Data from Worker
                            df_curr = worker.get_data(ticker, interval)
                            
                            if df_curr is None or len(df_curr) < param_val:
                                st.warning(f"데이터 대기 중... ({ticker}, {interval})")
                                total_theo_val += per_coin_cap 
                                continue
                                
                            # Dynamic Strategy Selection
                            if strategy_mode == "Donchian":
                                strategy_eng = DonchianStrategy()
                                buy_p = param_val
                                sell_p = item.get('sell_parameter', 0) or max(5, buy_p // 2)
                                
                                df_curr = strategy_eng.create_features(df_curr, buy_period=buy_p, sell_period=sell_p)
                                last_candle = df_curr.iloc[-2]
                                
                                # Visuals for Donchian
                                curr_upper = last_candle.get(f'Donchian_Upper_{buy_p}', 0)
                                curr_lower = last_candle.get(f'Donchian_Lower_{sell_p}', 0)
                                curr_sma = (curr_upper + curr_lower) / 2 # Mid for display
                                
                                
                            else: # SMA Strategy (Default)
                                strategy_eng = SMAStrategy()
                                calc_periods = [param_val]
                                    
                                df_curr = strategy_eng.create_features(df_curr, periods=calc_periods)
                                last_candle = df_curr.iloc[-2]
                                
                                curr_sma = last_candle[f'SMA_{param_val}']
                            # 캐시된 가격·잔고 사용 (일괄 조회 결과)
                            curr_price = all_prices.get(ticker, 0) or 0
                            coin_sym = item['coin'].upper()
                            coin_bal = all_balances.get(coin_sym, 0)

                            # 3. Theo Backtest (Sync Check) - 캐시된 백테스트 사용
                            sell_ratio = (item.get('sell_parameter', 0) or max(5, param_val // 2)) / param_val if param_val > 0 else 0.5
                            df_bt = data_cache.load_cached(ticker, interval)
                            if df_bt is not None and len(df_bt) >= param_val:
                                req_count = len(df_bt)
                                df_hash = f"{len(df_bt)}_{df_bt.index[-1]}"
                            else:
                                df_bt = df_curr
                                req_count = len(df_bt)
                                df_hash = f"{len(df_bt)}_{df_bt.index[-1]}"
                            bt_res = _cached_backtest(
                                ticker, param_val, interval, req_count,
                                str(start_date), per_coin_cap, strategy_mode,
                                sell_ratio, df_hash
                            )
                            if bt_res is None:
                                bt_res = backtest_engine.run_backtest(
                                    ticker, period=param_val, interval=interval,
                                    count=req_count, start_date=start_date,
                                    initial_balance=per_coin_cap, df=df_bt,
                                    strategy_mode=strategy_mode,
                                    sell_period_ratio=sell_ratio
                                )
                            
                            expected_eq = 0
                            theo_status = "UNKNOWN"
                            
                            if "error" not in bt_res:
                                perf = bt_res['performance']
                                theo_status = perf['final_status']
                                expected_eq = perf['final_equity']
                                total_theo_val += expected_eq
                                # 합산 포트폴리오용 에쿼티 수집
                                hist_df_tmp = bt_res['df']
                                label = f"{ticker} ({strategy_mode} {param_val}, {interval})"
                                portfolio_equity_data.append({
                                    "label": label,
                                    "equity": hist_df_tmp['equity'],
                                    "close": hist_df_tmp['close'],
                                    "cap": per_coin_cap,
                                    "perf": perf,
                                })
                            else:
                                total_theo_val += per_coin_cap # Fallback if error
                                
                            # 4. Real Status
                            coin_val = coin_bal * curr_price
                            total_real_val += coin_val # Add coin value to total
                            real_status = "HOLD" if coin_val > 5000 else "CASH"
                            
                            # --- Display Metrics ---
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("가격 / SMA", f"{curr_price:,.0f}", delta=f"{curr_price - curr_sma:,.0f}")
                            
                            
                            # Signal Metric Removed as requested
                            # c2.markdown(f"**Signal**: :{sig_color}[{curr_signal}]")
                            if strategy_mode == "Donchian":
                                c2.metric("채널", f"{buy_p}/{sell_p}")
                            else:
                                c2.metric("SMA 기간", f"{param_val}")
                            
                            # Asset Performance
                            roi_theo = (expected_eq - per_coin_cap) / per_coin_cap * 100
                            c3.metric(f"이론 자산", f"{expected_eq:,.0f}", delta=f"{roi_theo:.2f}%")
                            
                            match = (real_status == theo_status)
                            match_color = "green" if match else "red"
                            c4.markdown(f"**동기화**: :{match_color}[{'일치' if match else '불일치'}]")
                            c4.caption(f"실제: {coin_bal:,.4f} {coin_sym} ({real_status})")
                            
                            st.divider()
                            
                            # --- Tabs for Charts & Orders ---
                            p_tab1, p_tab2 = st.tabs(["📈 분석 & 벤치마크", "📋 체결 내역"])

                            with p_tab1:
                                if "error" not in bt_res:
                                    hist_df = bt_res['df']
                                    start_equity = hist_df['equity'].iloc[0]
                                    start_price = hist_df['close'].iloc[0]

                                    # Normalized Comparison
                                    hist_df['Norm_Strat'] = hist_df['equity'] / start_equity * 100
                                    hist_df['Norm_Bench'] = hist_df['close'] / start_price * 100

                                    fig_comp = go.Figure()
                                    fig_comp.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Norm_Strat'], name='전략', line=dict(color='blue')))
                                    fig_comp.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Norm_Bench'], name='벤치마크', line=dict(color='gray', dash='dot')))

                                    # 매수/매도 마커 추가
                                    if perf.get('trades'):
                                        buy_trades = [t for t in perf['trades'] if t['type'] == 'buy']
                                        sell_trades = [t for t in perf['trades'] if t['type'] == 'sell']
                                        if buy_trades:
                                            buy_dates = [t['date'] for t in buy_trades]
                                            buy_vals = [hist_df.loc[d, 'Norm_Strat'] if d in hist_df.index else None for d in buy_dates]
                                            fig_comp.add_trace(go.Scatter(
                                                x=buy_dates, y=buy_vals, mode='markers', name='매수',
                                                marker=dict(symbol='triangle-up', size=10, color='green')
                                            ))
                                        if sell_trades:
                                            sell_dates = [t['date'] for t in sell_trades]
                                            sell_vals = [hist_df.loc[d, 'Norm_Strat'] if d in hist_df.index else None for d in sell_dates]
                                            fig_comp.add_trace(go.Scatter(
                                                x=sell_dates, y=sell_vals, mode='markers', name='매도',
                                                marker=dict(symbol='triangle-down', size=10, color='red')
                                            ))

                                    fig_comp.update_layout(height=300, title="전략 vs 단순보유 (정규화)", margin=dict(l=0,r=0,t=80,b=0),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                                    st.plotly_chart(fig_comp, use_container_width=True)

                                    # 연도별 성과 테이블
                                    if 'yearly_stats' in perf:
                                        st.caption("📅 연도별 성과")
                                        ys = perf['yearly_stats'].copy()
                                        ys.index.name = "연도"
                                        st.dataframe(ys.style.format("{:.2f}"), use_container_width=True)

                                    _render_performance_analysis(
                                        equity_series=hist_df.get("equity"),
                                        benchmark_series=hist_df.get("close"),
                                        strategy_metrics=perf,
                                        strategy_label=f"{strategy_mode} 전략",
                                        benchmark_label=f"{ticker} 단순보유",
                                        monte_carlo_sims=220,
                                    )
                            
                            with p_tab2:
                                # ── 가상(백테스트) 체결 내역 ──
                                st.markdown("**가상 체결 (백테스트)**")
                                if "error" not in bt_res and perf.get('trades'):
                                    vt_rows = []
                                    for t in perf['trades']:
                                        t_date = t.get('date', '')
                                        if hasattr(t_date, 'strftime'):
                                            t_date = t_date.strftime('%Y-%m-%d')
                                        t_type = t.get('type', '')
                                        t_side = "매수" if t_type == 'buy' else ("매도" if t_type == 'sell' else t_type)
                                        t_price = t.get('price', 0)
                                        t_amount = t.get('amount', 0)
                                        t_equity = t.get('equity', 0)
                                        vt_rows.append({
                                            "일시": t_date,
                                            "구분": f"🔴 {t_side}" if t_type == 'buy' else f"🔵 {t_side}",
                                            "체결가": f"{t_price:,.0f}",
                                            "수량": f"{t_amount:.6f}" if t_amount < 1 else f"{t_amount:,.4f}",
                                            "자산(KRW)": f"{t_equity:,.0f}",
                                        })
                                    if vt_rows:
                                        st.dataframe(pd.DataFrame(vt_rows[-20:]), use_container_width=True, hide_index=True)
                                        st.caption(f"최근 {min(20, len(vt_rows))}건 / 총 {len(vt_rows)}건")
                                    else:
                                        st.info("백테스트 체결 기록 없음")
                                else:
                                    st.info("백테스트 데이터 없음")

                                # ── 실제 체결 내역 (거래소) ──
                                st.markdown("**실제 체결 (거래소)**")
                                try:
                                    done = _ttl_cache(
                                        f"done_{ticker}",
                                        lambda t=ticker: trader.get_done_orders(t),
                                        ttl=30
                                    )
                                    if done:
                                        rt_rows = []
                                        for r in done[:20]:
                                            side = r.get('side', '')
                                            side_kr = "매수" if side == 'bid' else ("매도" if side == 'ask' else side)
                                            price_r = float(r.get('price', 0) or 0)
                                            exec_vol = float(r.get('executed_volume', 0) or 0)
                                            if price_r > 0 and exec_vol > 0:
                                                total_k = price_r * exec_vol
                                            elif 'trades' in r and r['trades']:
                                                total_k = sum(float(tr.get('funds', 0)) for tr in r['trades'])
                                            else:
                                                total_k = price_r
                                            created = r.get('created_at', '')
                                            if pd.notna(created):
                                                try: created = pd.to_datetime(created).strftime('%Y-%m-%d %H:%M')
                                                except: pass
                                            rt_rows.append({
                                                "일시": created,
                                                "구분": f"🔴 {side_kr}" if side == 'bid' else f"🔵 {side_kr}",
                                                "체결가": f"{price_r:,.0f}" if price_r > 0 else "-",
                                                "수량": f"{exec_vol:.6f}" if exec_vol < 1 else f"{exec_vol:,.4f}",
                                                "금액(KRW)": f"{total_k:,.0f}",
                                            })
                                        if rt_rows:
                                            st.dataframe(pd.DataFrame(rt_rows), use_container_width=True, hide_index=True)
                                        else:
                                            st.info("체결 완료 주문 없음")
                                    else:
                                        st.info("체결 완료 주문 없음")
                                except Exception:
                                    st.info("체결 내역 조회 불가 (API 권한 확인)")

                        except Exception as e:
                            st.error(f"{ticker} 처리 오류: {e}")
                
                # --- Populate Total Summary ---
                total_roi = (total_theo_val - total_init_val) / total_init_val * 100 if total_init_val else 0
                real_roi = (total_real_val - total_init_val) / total_init_val * 100 if total_init_val else 0
                diff_val = total_real_val - total_theo_val

                sum_col1.metric("초기 자본", f"{total_init_val:,.0f} KRW")
                sum_col2.metric("이론 총자산", f"{total_theo_val:,.0f} KRW", delta=f"{total_roi:.2f}%")
                sum_col3.metric("실제 총자산", f"{total_real_val:,.0f} KRW", delta=f"{real_roi:.2f}%")
                sum_col4.metric("차이 (실제-이론)", f"{diff_val:,.0f} KRW", delta_color="off" if abs(diff_val)<1000 else "inverse")

                # --- 합산 포트폴리오 성과 (Combined Portfolio) → 위에 예약한 container에 렌더링 ---
                if portfolio_equity_data:
                    with combined_portfolio_container:
                        with st.expander("📊 합산 포트폴리오 성과", expanded=True):
                            import numpy as np

                            # 각 자산의 에쿼티를 일자 기준으로 합산
                            equity_dfs = []
                            bench_dfs = []
                            for ed in portfolio_equity_data:
                                eq = ed['equity'].copy()
                                cl = ed['close'].copy()
                                cap = ed['cap']

                                if hasattr(eq.index, 'tz') and eq.index.tz is not None:
                                    eq.index = eq.index.tz_localize(None)
                                    cl.index = cl.index.tz_localize(None)
                                eq_daily = eq.resample('D').last().dropna()
                                cl_daily = cl.resample('D').last().dropna()

                                bench_daily = (cl_daily / cl_daily.iloc[0]) * cap

                                eq_daily.name = ed['label']
                                bench_daily.name = ed['label']
                                equity_dfs.append(eq_daily)
                                bench_dfs.append(bench_daily)

                            combined_eq = pd.concat(equity_dfs, axis=1).sort_index()
                            combined_bench = pd.concat(bench_dfs, axis=1).sort_index()

                            combined_eq = combined_eq.ffill().bfill()
                            combined_bench = combined_bench.ffill().bfill()

                            combined_eq['cash_reserve'] = reserved_cash
                            combined_bench['cash_reserve'] = reserved_cash

                            total_eq = combined_eq.sum(axis=1)
                            total_bench = combined_bench.sum(axis=1)

                            norm_eq = total_eq / total_eq.iloc[0] * 100
                            norm_bench = total_bench / total_bench.iloc[0] * 100

                            # 성과 지표 계산
                            port_final = total_eq.iloc[-1]
                            port_init = total_eq.iloc[0]
                            port_return = (port_final - port_init) / port_init * 100

                            port_days = (total_eq.index[-1] - total_eq.index[0]).days
                            port_cagr = 0
                            if port_days > 0 and port_final > 0:
                                port_cagr = ((port_final / port_init) ** (365 / port_days) - 1) * 100

                            port_peak = total_eq.cummax()
                            port_dd = (total_eq - port_peak) / port_peak * 100
                            port_mdd = port_dd.min()

                            port_returns = total_eq.pct_change().dropna()
                            port_sharpe = 0
                            if port_returns.std() > 0:
                                port_sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(365)

                            bench_final = total_bench.iloc[-1]
                            bench_init = total_bench.iloc[0]
                            bench_return = (bench_final - bench_init) / bench_init * 100

                            # 메트릭 표시
                            pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                            pm1.metric("총 수익률", f"{port_return:.2f}%")
                            pm2.metric("CAGR", f"{port_cagr:.2f}%")
                            pm3.metric("MDD", f"{port_mdd:.2f}%")
                            pm4.metric("Sharpe", f"{port_sharpe:.2f}")
                            pm5.metric("vs 단순보유", f"{port_return - bench_return:+.2f}%p")

                            st.caption(f"기간: {total_eq.index[0].strftime('%Y-%m-%d')} ~ {total_eq.index[-1].strftime('%Y-%m-%d')} ({port_days}일) | 초기자금: {port_init:,.0f} → 최종: {port_final:,.0f} KRW")

                            # 합산 차트
                            fig_port = go.Figure()
                            fig_port.add_trace(go.Scatter(
                                x=norm_eq.index, y=norm_eq.values,
                                name='포트폴리오 (전략)', line=dict(color='blue', width=2)
                            ))
                            fig_port.add_trace(go.Scatter(
                                x=norm_bench.index, y=norm_bench.values,
                                name='포트폴리오 (단순보유)', line=dict(color='gray', dash='dot')
                            ))

                            # 합산 차트에 매수/매도 마커 표시
                            all_buy_dates = []
                            all_sell_dates = []
                            for ed in portfolio_equity_data:
                                for t in ed['perf'].get('trades', []):
                                    if t['type'] == 'buy':
                                        all_buy_dates.append(t['date'])
                                    elif t['type'] == 'sell':
                                        all_sell_dates.append(t['date'])

                            if all_buy_dates:
                                # 날짜를 norm_eq 인덱스와 매칭 (일봉 리샘플링 됐으므로 가장 가까운 날짜 사용)
                                buy_vals = []
                                buy_dates_valid = []
                                for d in all_buy_dates:
                                    d_ts = pd.Timestamp(d)
                                    if hasattr(d_ts, 'tz') and d_ts.tz is not None:
                                        d_ts = d_ts.tz_localize(None)
                                    idx = norm_eq.index.get_indexer([d_ts], method='nearest')
                                    if idx[0] >= 0:
                                        buy_dates_valid.append(norm_eq.index[idx[0]])
                                        buy_vals.append(norm_eq.iloc[idx[0]])
                                if buy_dates_valid:
                                    fig_port.add_trace(go.Scatter(
                                        x=buy_dates_valid, y=buy_vals, mode='markers', name='매수',
                                        marker=dict(symbol='triangle-up', size=8, color='green', opacity=0.7)
                                    ))

                            if all_sell_dates:
                                sell_vals = []
                                sell_dates_valid = []
                                for d in all_sell_dates:
                                    d_ts = pd.Timestamp(d)
                                    if hasattr(d_ts, 'tz') and d_ts.tz is not None:
                                        d_ts = d_ts.tz_localize(None)
                                    idx = norm_eq.index.get_indexer([d_ts], method='nearest')
                                    if idx[0] >= 0:
                                        sell_dates_valid.append(norm_eq.index[idx[0]])
                                        sell_vals.append(norm_eq.iloc[idx[0]])
                                if sell_dates_valid:
                                    fig_port.add_trace(go.Scatter(
                                        x=sell_dates_valid, y=sell_vals, mode='markers', name='매도',
                                        marker=dict(symbol='triangle-down', size=8, color='red', opacity=0.7)
                                    ))

                            fig_port.update_layout(
                                height=350,
                                title="합산 포트폴리오: 전략 vs 단순보유 (정규화)",
                                yaxis_title="정규화 (%)",
                                margin=dict(l=0, r=0, t=80, b=0),
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            st.plotly_chart(fig_port, use_container_width=True)

                            # 포트폴리오 MDD(Drawdown) 차트 추가
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=port_dd.index, y=port_dd.values,
                                name='낙폭', fill='tozeroy',
                                line=dict(color='red', width=1)
                            ))
                            fig_dd.update_layout(
                                height=200,
                                title="포트폴리오 낙폭 (%)",
                                yaxis_title="낙폭 (%)",
                                margin=dict(l=0, r=0, t=80, b=0),
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            fig_dd = _apply_dd_hover_format(fig_dd)
                            st.plotly_chart(fig_dd, use_container_width=True)

                            # 개별 자산 에쿼티 기여도 차트
                            fig_stack = go.Figure()
                            for ed in portfolio_equity_data:
                                eq = ed['equity'].copy()
                                if hasattr(eq.index, 'tz') and eq.index.tz is not None:
                                    eq.index = eq.index.tz_localize(None)
                                eq_d = eq.resample('D').last().dropna()
                                fig_stack.add_trace(go.Scatter(
                                    x=eq_d.index, y=eq_d.values,
                                    name=ed['label'], stackgroup='one'
                                ))
                            if reserved_cash > 0:
                                fig_stack.add_trace(go.Scatter(
                                    x=total_eq.index, y=[reserved_cash] * len(total_eq),
                                    name='현금 예비', stackgroup='one',
                                    line=dict(color='lightgray')
                                ))
                            fig_stack.update_layout(
                                height=350,
                                title="자산별 기여도 (적층)",
                                yaxis_title="KRW",
                                margin=dict(l=0, r=0, t=80, b=0),
                                hovermode='x unified',
                                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0)
                            )
                            st.plotly_chart(fig_stack, use_container_width=True)

                            # 개별 자산 성과 테이블
                            asset_perf_rows = []
                            for ed in portfolio_equity_data:
                                p = ed['perf']
                                asset_perf_rows.append({
                                    "자산": ed['label'],
                                    "배분자본": f"{ed['cap']:,.0f}",
                                    "최종자산": f"{p['final_equity']:,.0f}",
                                    "수익률(%)": f"{p['total_return']:.2f}",
                                    "CAGR(%)": f"{p['cagr']:.2f}",
                                    "MDD(%)": f"{p['mdd']:.2f}",
                                    "승률(%)": f"{p['win_rate']:.1f}",
                                    "거래수": p['trade_count'],
                                    "Sharpe": f"{p['sharpe']:.2f}",
                                    "상태": p['final_status'],
                                })
                            st.dataframe(pd.DataFrame(asset_perf_rows), use_container_width=True, hide_index=True)

                            # 📅 합산 포트폴리오 연도별 성과 테이블
                            st.caption("📅 합산 포트폴리오 연도별 성과")
                            port_daily_ret = total_eq.pct_change().fillna(0)
                            port_year = total_eq.index.year
                            port_dd_series = port_dd

                            yearly_rows = []
                            for yr in sorted(port_year.unique()):
                                yr_mask = port_year == yr
                                yr_ret = (1 + port_daily_ret[yr_mask]).prod() - 1
                                yr_mdd = port_dd_series[yr_mask].min()
                                yr_eq_start = total_eq[yr_mask].iloc[0]
                                yr_eq_end = total_eq[yr_mask].iloc[-1]

                                # 벤치마크 연도별
                                yr_bench_start = total_bench[yr_mask].iloc[0]
                                yr_bench_end = total_bench[yr_mask].iloc[-1]
                                yr_bench_ret = (yr_bench_end - yr_bench_start) / yr_bench_start * 100

                                yearly_rows.append({
                                    "연도": yr,
                                    "수익률(%)": f"{yr_ret * 100:.2f}",
                                    "MDD(%)": f"{yr_mdd:.2f}",
                                    "시작자산": f"{yr_eq_start:,.0f}",
                                    "최종자산": f"{yr_eq_end:,.0f}",
                                    "Buy&Hold(%)": f"{yr_bench_ret:.2f}",
                                    "초과수익(%p)": f"{yr_ret * 100 - yr_bench_ret:.2f}",
                                })
                            st.dataframe(pd.DataFrame(yearly_rows), use_container_width=True, hide_index=True)

    # --- Tab 5: Manual Trade (거래소 스타일) ---
    with tab5:
        st.header("수동 주문")

        if not trader:
            st.warning("사이드바에서 API 키를 설정해주세요.")
        else:
            # 코인 선택 (변경시만 full rerun)
            port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
            manual_options = list(dict.fromkeys(port_tickers + TOP_20_TICKERS))
            mt_col1, mt_col2 = st.columns(2)
            mt_selected = mt_col1.selectbox("코인 선택", manual_options + ["직접입력"], key="mt_ticker")
            if mt_selected == "직접입력":
                mt_custom = mt_col2.text_input("코인 심볼", "BTC", key="mt_custom")
                mt_ticker = f"KRW-{mt_custom.upper()}"
            else:
                mt_ticker = mt_selected
                mt_col2.empty()

            mt_coin = mt_ticker.split("-")[1] if "-" in mt_ticker else mt_ticker

            # ── 코인 트레이딩 워커 시작 (백그라운드 갱신) ──
            from data_manager import CoinTradingWorker

            @st.cache_resource
            def _get_coin_worker(_trader):
                w = CoinTradingWorker()
                return w

            coin_worker = _get_coin_worker(trader)
            coin_worker.configure(trader, mt_ticker)
            coin_worker.start()

            # ═══ 트레이딩 패널 (fragment → 3초마다 자동갱신, 워커에서 읽기만) ═══
            @st.fragment
            def trading_panel():
                # ── 워커에서 즉시 읽기 (API 호출 없음 → 블로킹 없음) ──
                mt_price = coin_worker.get('price', 0)
                krw_avail = coin_worker.get('krw_bal', 0)
                mt_coin_bal = coin_worker.get('coin_bal', 0)
                mt_coin_val = mt_coin_bal * mt_price
                mt_tick = get_tick_size(mt_price) if mt_price > 0 else 1
                mt_min_qty = round(5000 / mt_price, 8) if mt_price > 0 else 0.00000001

                # 상단 정보 바
                ic1, ic2, ic3, ic4, ic5 = st.columns(5)
                ic1.metric("현재가", f"{mt_price:,.0f}")
                ic2.metric(f"{mt_coin} 보유", f"{mt_coin_bal:.8f}" if mt_coin_bal < 1 else f"{mt_coin_bal:,.4f}")
                ic3.metric("평가금액", f"{mt_coin_val:,.0f} KRW")
                ic4.metric("보유 KRW", f"{krw_avail:,.0f}")
                ic5.metric("호가단위", f"{mt_tick:,g}원" if mt_tick >= 1 else f"{mt_tick}원")

                # ── 최근 거래 결과 알림 바 (세션 유지) ──
                last_trade = st.session_state.get('_last_trade')
                if last_trade:
                    t_type = last_trade.get('type', '')
                    t_time = last_trade.get('time', '')
                    t_ticker = last_trade.get('ticker', '')
                    t_amt = last_trade.get('amount', '')
                    t_price = last_trade.get('price', '')
                    t_qty = last_trade.get('qty', '')
                    is_buy = '매수' in t_type
                    color = '#D32F2F' if is_buy else '#1976D2'
                    detail = t_amt if t_amt else f"{t_price} x {t_qty}"
                    nc1, nc2 = st.columns([6, 1])
                    nc1.markdown(
                        f'<div style="padding:6px 12px;border-radius:6px;background:{color}22;border-left:4px solid {color};font-size:14px;">'
                        f'<b style="color:{color}">{t_type}</b> {t_ticker} | {detail} | {t_time}</div>',
                        unsafe_allow_html=True
                    )
                    if nc2.button("✕", key="_dismiss_trade"):
                        del st.session_state['_last_trade']
                        st.rerun()

                st.divider()

                # ═══ 레이아웃: 30분봉 차트(상단 전체폭) + 주문가격/주문 패널(하단) ═══
                st.markdown("**30분봉 차트**")
                df_30m = _ttl_cache(
                    f"m30_{mt_ticker}",
                    lambda: data_cache.get_ohlcv_local_first(
                        mt_ticker,
                        interval="minute30",
                        count=48,
                        allow_api_fallback=True,
                    ),
                    ttl=30,
                )
                if df_30m is not None and len(df_30m) > 0:
                    fig_30m = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
                    fig_30m.add_trace(go.Candlestick(
                        x=df_30m.index, open=df_30m['open'], high=df_30m['high'],
                        low=df_30m['low'], close=df_30m['close'], name='30분봉',
                        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                    ), row=1, col=1)
                    ma5 = df_30m['close'].rolling(5).mean()
                    ma20 = df_30m['close'].rolling(20).mean()
                    fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma5, name='MA5', line=dict(color='#FF9800', width=1)), row=1, col=1)
                    fig_30m.add_trace(go.Scatter(x=df_30m.index, y=ma20, name='MA20', line=dict(color='#2196F3', width=1)), row=1, col=1)
                    colors_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_30m['close'], df_30m['open'])]
                    fig_30m.add_trace(go.Bar(
                        x=df_30m.index, y=df_30m['volume'], marker_color=colors_vol, name='거래량', showlegend=False
                    ), row=2, col=1)
                    fig_30m.update_layout(
                        height=520, margin=dict(l=0, r=0, t=10, b=30),
                        xaxis_rangeslider_visible=False, showlegend=True,
                        legend=dict(orientation="h", y=1.06, x=0),
                        xaxis2=dict(showticklabels=True, tickformat='%H:%M', tickangle=-45),
                        yaxis=dict(title="", side="right"),
                        yaxis2=dict(title="", side="right"),
                    )
                    st.plotly_chart(fig_30m, use_container_width=True, key=f"chart30m_{mt_ticker}")
                else:
                    st.info("차트 데이터 로딩 중...")

                st.divider()

                # ═══ 좌: 호가창 | 우: 주문 패널 (가로 배치 — 골드와 동일 구조) ═══
                ob_col, order_col = st.columns([2, 3])

                # ── 좌: 호가창 (HTML 렌더링) ──
                with ob_col:
                    st.markdown("**주문가격 표**")
                    raw_prices = []
                    try:
                        ob_data = coin_worker.get('orderbook')
                        if ob_data and len(ob_data) > 0:
                            ob = ob_data[0] if isinstance(ob_data, list) else ob_data
                            units = ob.get('orderbook_units', [])[:10]

                            if units:
                                max_size = max(
                                    max(u.get('ask_size', 0) for u in units),
                                    max(u.get('bid_size', 0) for u in units)
                                )

                                # ── HTML 호가창 테이블 생성 ──
                                html = ['<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:monospace;">']
                                html.append('<tr style="border-bottom:2px solid #ddd;font-weight:bold;color:#666"><td>구분</td><td style="text-align:right">잔량</td><td style="text-align:right">가격</td><td style="text-align:right">등락</td><td>비율</td></tr>')

                                ask_prices = []
                                bid_prices = []

                                for u in reversed(units):
                                    ask_p = u.get('ask_price', 0)
                                    ask_s = u.get('ask_size', 0)
                                    diff = ((ask_p / mt_price) - 1) * 100 if mt_price > 0 else 0
                                    bar_w = int(ask_s / max_size * 100) if max_size > 0 else 0
                                    raw_prices.append(ask_p)
                                    ask_prices.append(ask_p)
                                    html.append(
                                        f'<tr style="color:#1976D2;border-bottom:1px solid #f0f0f0;height:28px">'
                                        f'<td>매도</td>'
                                        f'<td style="text-align:right">{ask_s:.4f}</td>'
                                        f'<td style="text-align:right;font-weight:bold">{ask_p:,.0f}</td>'
                                        f'<td style="text-align:right">{diff:+.2f}%</td>'
                                        f'<td><div style="background:#1976D2;height:12px;width:{bar_w}%;opacity:0.3"></div></td>'
                                        f'</tr>'
                                    )

                                # 중간 구분선
                                html.append('<tr style="border-top:3px solid #333;border-bottom:3px solid #333;height:4px"><td colspan="5"></td></tr>')

                                for u in units:
                                    bid_p = u.get('bid_price', 0)
                                    bid_s = u.get('bid_size', 0)
                                    diff = ((bid_p / mt_price) - 1) * 100 if mt_price > 0 else 0
                                    bar_w = int(bid_s / max_size * 100) if max_size > 0 else 0
                                    raw_prices.append(bid_p)
                                    bid_prices.append(bid_p)
                                    html.append(
                                        f'<tr style="color:#D32F2F;border-bottom:1px solid #f0f0f0;height:28px">'
                                        f'<td>매수</td>'
                                        f'<td style="text-align:right">{bid_s:.4f}</td>'
                                        f'<td style="text-align:right;font-weight:bold">{bid_p:,.0f}</td>'
                                        f'<td style="text-align:right">{diff:+.2f}%</td>'
                                        f'<td><div style="background:#D32F2F;height:12px;width:{bar_w}%;opacity:0.3"></div></td>'
                                        f'</tr>'
                                    )

                                html.append('</table>')
                                st.markdown(''.join(html), unsafe_allow_html=True)

                                # ── 호가 선택 → 주문가 반영 ──
                                def _on_ob_select():
                                    """사용자가 직접 선택했을 때만 주문가에 반영"""
                                    sel_label = st.session_state.get('_ob_sel_label', '')
                                    try:
                                        price_str = sel_label.split(' ', 1)[1].replace(',', '')
                                        chosen = int(float(price_str))
                                        tick = get_tick_size(chosen)
                                        if tick >= 1:
                                            st.session_state['mt_buy_price'] = int(chosen)
                                            st.session_state['mt_sell_price'] = int(chosen)
                                        else:
                                            st.session_state['mt_buy_price'] = float(chosen)
                                            st.session_state['mt_sell_price'] = float(chosen)
                                    except (IndexError, ValueError):
                                        pass

                                price_labels = (
                                    [f"매도 {p:,.0f}" for p in ask_prices] +
                                    [f"매수 {p:,.0f}" for p in bid_prices]
                                )

                                st.selectbox(
                                    "호가 선택 → 주문가 반영",
                                    price_labels,
                                    index=len(ask_prices),  # 기본: 최우선 매수호가
                                    key="_ob_sel_label",
                                    on_change=_on_ob_select,
                                )

                                best_ask = units[0].get('ask_price', 0)
                                best_bid = units[0].get('bid_price', 0)
                                spread = best_ask - best_bid
                                spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
                                total_ask = ob.get('total_ask_size', 0)
                                total_bid = ob.get('total_bid_size', 0)
                                ob_ratio = total_bid / (total_ask + total_bid) * 100 if (total_ask + total_bid) > 0 else 50
                                st.caption(f"스프레드 **{spread:,.0f}** ({spread_pct:.3f}%) | 매도 {total_ask:.2f} | 매수 {total_bid:.2f} | 매수비율 {ob_ratio:.0f}%")
                            else:
                                st.info("호가 데이터가 없습니다.")
                        else:
                            st.info("호가 데이터를 불러올 수 없습니다.")
                    except Exception as e:
                        st.warning(f"호가 조회 실패: {e}")

                # ── 우: 주문 패널 ──
                with order_col:
                    st.markdown("**주문 실행**")
                    buy_tab, sell_tab = st.tabs(["🔴 매수", "🔵 매도"])

                    with buy_tab:
                        buy_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_buy_type")

                        if buy_type == "시장가":
                            buy_amount = st.number_input("매수 금액 (KRW)", min_value=5000, value=10000, step=1000, key="mt_buy_amt")
                            qb1, qb2, qb3, qb4 = st.columns(4)
                            if qb1.button("10%", key="mt_b10"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.1)
                                st.rerun()
                            if qb2.button("25%", key="mt_b25"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.25)
                                st.rerun()
                            if qb3.button("50%", key="mt_b50"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.5)
                                st.rerun()
                            if qb4.button("100%", key="mt_b100"):
                                st.session_state['mt_buy_amt'] = int(krw_avail * 0.999)
                                st.rerun()

                            if mt_price > 0:
                                st.caption(f"예상 수량: ~{buy_amount / mt_price:.8f} {mt_coin}")

                            if st.button("시장가 매수", type="primary", key="mt_buy_exec", use_container_width=True):
                                if buy_amount < 5000:
                                    st.toast("최소 주문금액: 5,000 KRW", icon="⚠️")
                                elif buy_amount > krw_avail:
                                    st.toast(f"잔고 부족 ({krw_avail:,.0f} KRW)", icon="⚠️")
                                else:
                                    with st.spinner("매수 주문 중..."):
                                        result = trader.buy_market(mt_ticker, buy_amount)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 시장가 매수 체결! {buy_amount:,.0f} KRW", icon="🟢")
                                        st.session_state['_last_trade'] = {"type": "시장가 매수", "ticker": mt_ticker, "amount": f"{buy_amount:,.0f} KRW", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"매수 실패: {result}", icon="🔴")

                        else:  # 지정가
                            buy_default = align_price(mt_price * 0.99, mt_tick) if mt_price > 0 else 1
                            bc1, bc2 = st.columns(2)
                            if mt_tick >= 1:
                                buy_price = bc1.number_input("매수 가격", min_value=1, value=int(buy_default), step=int(mt_tick), key="mt_buy_price")
                            else:
                                buy_price = bc1.number_input("매수 가격", min_value=0.0001, value=float(buy_default), step=float(mt_tick), format="%.4f", key="mt_buy_price")
                            buy_qty = bc2.number_input("매수 수량", min_value=mt_min_qty, value=max(mt_min_qty, 0.001), format="%.8f", key="mt_buy_qty")

                            buy_total = buy_price * buy_qty
                            st.caption(f"총액: **{buy_total:,.0f} KRW** | 호가: {mt_tick:,g}원 | 최소: {mt_min_qty:.8f}")

                            qbc1, qbc2, qbc3, qbc4 = st.columns(4)
                            if buy_price > 0:
                                if qbc1.button("10%", key="mt_lb10"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.1 / buy_price, 8)
                                    st.rerun()
                                if qbc2.button("25%", key="mt_lb25"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.25 / buy_price, 8)
                                    st.rerun()
                                if qbc3.button("50%", key="mt_lb50"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.5 / buy_price, 8)
                                    st.rerun()
                                if qbc4.button("100%", key="mt_lb100"):
                                    st.session_state['mt_buy_qty'] = round(krw_avail * 0.999 / buy_price, 8)
                                    st.rerun()

                            if st.button("지정가 매수", type="primary", key="mt_lbuy_exec", use_container_width=True):
                                if buy_total < 5000:
                                    st.toast("최소 주문금액: 5,000 KRW", icon="⚠️")
                                elif buy_total > krw_avail:
                                    st.toast(f"잔고 부족 ({krw_avail:,.0f} KRW)", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매수 주문 중..."):
                                        result = trader.buy_limit(mt_ticker, buy_price, buy_qty)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 지정가 매수 등록! {buy_price:,.0f} × {buy_qty:.8f}", icon="🟢")
                                        st.session_state['_last_trade'] = {"type": "지정가 매수", "ticker": mt_ticker, "price": f"{buy_price:,.0f}", "qty": f"{buy_qty:.8f}", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

                    with sell_tab:
                        sell_type = st.radio("주문 유형", ["시장가", "지정가"], horizontal=True, key="mt_sell_type")

                        if sell_type == "시장가":
                            sell_qty = st.number_input(
                                f"매도 수량 ({mt_coin})", min_value=0.00000001,
                                value=mt_coin_bal if mt_coin_bal > 0 else 0.00000001,
                                format="%.8f", key="mt_sell_qty"
                            )
                            qs1, qs2, qs3, qs4 = st.columns(4)
                            if qs1.button("25%", key="mt_s25"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal * 0.25, 8)
                                st.rerun()
                            if qs2.button("50%", key="mt_s50"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal * 0.5, 8)
                                st.rerun()
                            if qs3.button("75%", key="mt_s75"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal * 0.75, 8)
                                st.rerun()
                            if qs4.button("100%", key="mt_s100"):
                                st.session_state['mt_sell_qty'] = round(mt_coin_bal, 8)
                                st.rerun()

                            if mt_price > 0:
                                st.caption(f"예상 금액: ~{sell_qty * mt_price:,.0f} KRW")

                            if st.button("시장가 매도", type="primary", key="mt_sell_exec", use_container_width=True):
                                if sell_qty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif mt_price > 0 and sell_qty * mt_price < 5000:
                                    st.toast(f"최소 주문금액 미달 ({sell_qty * mt_price:,.0f} KRW < 5,000)", icon="⚠️")
                                elif sell_qty > mt_coin_bal:
                                    st.toast(f"보유량 초과 ({mt_coin_bal:.8f})", icon="⚠️")
                                else:
                                    with st.spinner("매도 주문 중..."):
                                        result = trader.sell_market(mt_ticker, sell_qty)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 시장가 매도 체결! {sell_qty:.8f} {mt_coin}", icon="🔴")
                                        st.session_state['_last_trade'] = {"type": "시장가 매도", "ticker": mt_ticker, "qty": f"{sell_qty:.8f}", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"매도 실패: {result}", icon="🔴")

                        else:  # 지정가
                            sell_default = align_price(mt_price * 1.01, mt_tick) if mt_price > 0 else 1
                            sc1, sc2 = st.columns(2)
                            if mt_tick >= 1:
                                sell_price = sc1.number_input("매도 가격", min_value=1, value=int(sell_default), step=int(mt_tick), key="mt_sell_price")
                            else:
                                sell_price = sc1.number_input("매도 가격", min_value=0.0001, value=float(sell_default), step=float(mt_tick), format="%.4f", key="mt_sell_price")
                            sell_default_qty = mt_coin_bal if mt_coin_bal > mt_min_qty else mt_min_qty
                            sell_limit_qty = sc2.number_input("매도 수량", min_value=mt_min_qty, value=sell_default_qty, format="%.8f", key="mt_sell_lqty")

                            sell_total = sell_price * sell_limit_qty
                            st.caption(f"총액: **{sell_total:,.0f} KRW** | 호가: {mt_tick:,g}원 | 최소: {mt_min_qty:.8f}")

                            qsc1, qsc2, qsc3, qsc4 = st.columns(4)
                            if mt_coin_bal > 0:
                                if qsc1.button("25%", key="mt_ls25"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal * 0.25, 8)
                                    st.rerun()
                                if qsc2.button("50%", key="mt_ls50"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal * 0.5, 8)
                                    st.rerun()
                                if qsc3.button("75%", key="mt_ls75"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal * 0.75, 8)
                                    st.rerun()
                                if qsc4.button("100%", key="mt_ls100"):
                                    st.session_state['mt_sell_lqty'] = round(mt_coin_bal, 8)
                                    st.rerun()

                            if st.button("지정가 매도", type="primary", key="mt_lsell_exec", use_container_width=True):
                                if sell_limit_qty <= 0:
                                    st.toast("매도 수량을 입력해주세요.", icon="⚠️")
                                elif sell_limit_qty > mt_coin_bal:
                                    st.toast(f"보유량 초과 ({mt_coin_bal:.8f})", icon="⚠️")
                                else:
                                    with st.spinner("지정가 매도 주문 중..."):
                                        result = trader.sell_limit(mt_ticker, sell_price, sell_limit_qty)
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    if result and "error" not in result:
                                        st.toast(f"✅ 지정가 매도 등록! {sell_price:,.0f} × {sell_limit_qty:.8f}", icon="🔴")
                                        st.session_state['_last_trade'] = {"type": "지정가 매도", "ticker": mt_ticker, "price": f"{sell_price:,.0f}", "qty": f"{sell_limit_qty:.8f}", "result": result, "time": time.strftime('%H:%M:%S')}
                                    else:
                                        st.toast(f"주문 실패: {result}", icon="🔴")

                # ── 미체결 주문 ──
                st.divider()
                st.subheader("미체결 주문")
                if st.button("미체결 주문 조회", key="mt_pending"):
                    with st.spinner("조회 중..."):
                        pending = trader.get_orders(state="wait")
                    if pending and len(pending) > 0:
                        for i, order in enumerate(pending):
                            side_kr = "매수" if order.get('side') == 'bid' else "매도"
                            side_color = "red" if order.get('side') == 'bid' else "blue"
                            market = order.get('market', '')
                            price = float(order.get('price', 0) or 0)
                            remaining = float(order.get('remaining_volume', 0) or 0)
                            created = order.get('created_at', '')
                            if pd.notna(created):
                                try:
                                    created = pd.to_datetime(created).strftime('%m/%d %H:%M')
                                except:
                                    pass
                            oc1, oc2 = st.columns([4, 1])
                            oc1.markdown(f"**:{side_color}[{side_kr}]** {market} | {price:,.0f} × {remaining:.8f} | {created}")
                            if oc2.button("취소", key=f"mt_cancel_{i}"):
                                cancel_result = trader.cancel_order(order.get('uuid'))
                                if cancel_result and "error" not in cancel_result:
                                    st.toast("주문 취소 완료", icon="✅")
                                    coin_worker.invalidate('krw_bal', 'coin_bal')
                                    st.rerun()
                                else:
                                    st.toast(f"취소 실패: {cancel_result}", icon="🔴")
                    else:
                        st.info("미체결 주문이 없습니다.")

            trading_panel()

    # --- Tab 3: History ---
    with tab3:
        st.header("거래 내역")

        hist_tab1, hist_tab2, hist_tab3 = st.tabs(["💸 실제 거래 내역 (거래소)", "🧪 가상 로그 (페이퍼)", "📊 슬리피지 분석"])

        with hist_tab1:
            st.subheader("실제 거래 내역")

            if not trader:
                st.warning("사이드바에서 API 키를 설정해주세요.")
            else:
                c_h1, c_h2 = st.columns(2)
                h_type = c_h1.selectbox("조회 유형", ["전체", "입금", "출금", "체결 주문"])
                h_curr = c_h2.selectbox("화폐", ["전체", "KRW", "BTC", "ETH", "XRP", "SOL", "USDT", "DOGE", "ADA", "AVAX", "LINK"])

                d_h1, d_h2 = st.columns(2)
                h_date_start = d_h1.date_input("조회 시작일", value=datetime.now().date() - timedelta(days=90), key="hist_start")
                h_date_end = d_h2.date_input("조회 종료일", value=datetime.now().date(), key="hist_end")

                if st.button("조회"):
                    with st.spinner("Upbit API 조회 중..."):
                        api_curr = None if h_curr == "전체" else h_curr

                        # ── 조회 유형별 데이터 수집 ──
                        def _parse_deposit_withdraw(raw, type_label):
                            """입금/출금 데이터를 통합 포맷으로 변환"""
                            rows = []
                            for r in raw:
                                done = r.get('done_at', r.get('created_at', ''))
                                if pd.notna(done):
                                    try: done = pd.to_datetime(done).strftime('%Y-%m-%d %H:%M')
                                    except: pass
                                amount = float(r.get('amount', 0))
                                fee_val = float(r.get('fee', 0))
                                state = r.get('state', '')
                                state_kr = {"ACCEPTED": "완료", "REJECTED": "거부", "CANCELLED": "취소", "PROCESSING": "처리중", "WAITING": "대기중"}.get(state, state)
                                rows.append({
                                    "거래일시": done, "유형": type_label,
                                    "화폐/코인": r.get('currency', ''),
                                    "구분": type_label,
                                    "금액/수량": f"{amount:,.4f}" if amount < 100 else f"{amount:,.0f}",
                                    "체결금액(KRW)": "-",
                                    "수수료": f"{fee_val:,.4f}" if fee_val > 0 else "-",
                                    "상태": state_kr,
                                    "_sort_dt": done,
                                })
                            return rows

                        def _parse_orders(raw):
                            """체결 주문 데이터를 통합 포맷으로 변환"""
                            rows = []
                            for r in raw:
                                market = r.get('market', '')
                                coin = market.split('-')[1] if '-' in str(market) else market
                                side = r.get('side', '')
                                side_kr = "매수" if side == 'bid' else ("매도" if side == 'ask' else side)
                                state = r.get('state', '')
                                state_kr = {"done": "체결완료", "cancel": "취소", "wait": "대기"}.get(state, state)
                                price = float(r.get('price', 0) or 0)
                                executed_vol = float(r.get('executed_volume', 0) or 0)
                                paid_fee = float(r.get('paid_fee', 0) or 0)
                                if price > 0 and executed_vol > 0:
                                    total_krw = price * executed_vol
                                elif 'trades' in r and r['trades']:
                                    total_krw = sum(float(t.get('funds', 0)) for t in r['trades'])
                                else:
                                    total_krw = price
                                ord_type = r.get('ord_type', '')
                                type_kr = {"limit": "지정가", "price": "시장가(매수)", "market": "시장가(매도)"}.get(ord_type, ord_type)
                                created = r.get('created_at', '')
                                if pd.notna(created):
                                    try: created = pd.to_datetime(created).strftime('%Y-%m-%d %H:%M')
                                    except: pass
                                rows.append({
                                    "거래일시": created, "유형": f"체결({type_kr})",
                                    "화폐/코인": coin,
                                    "구분": side_kr,
                                    "금액/수량": f"{executed_vol:,.8f}" if executed_vol < 1 else f"{executed_vol:,.4f}",
                                    "체결금액(KRW)": f"{total_krw:,.0f}",
                                    "수수료": f"{paid_fee:,.2f}",
                                    "상태": state_kr,
                                    "_sort_dt": created,
                                })
                            return rows

                        api_curr = None if h_curr == "전체" else h_curr
                        all_rows = []
                        error_msgs = []

                        # 조회 대상 결정
                        query_types = []
                        if h_type == "전체":
                            query_types = [("deposit", "입금"), ("withdraw", "출금"), ("order", "체결")]
                        elif "입금" in h_type:
                            query_types = [("deposit", "입금")]
                        elif "출금" in h_type:
                            query_types = [("withdraw", "출금")]
                        elif "체결" in h_type:
                            query_types = [("order", "체결")]

                        for api_type, label in query_types:
                            try:
                                data, err = trader.get_history(api_type, api_curr)
                                if err:
                                    error_msgs.append(f"{label}: {err}")
                                if data:
                                    if api_type in ("deposit", "withdraw"):
                                        all_rows.extend(_parse_deposit_withdraw(data, label))
                                    else:
                                        all_rows.extend(_parse_orders(data))
                            except Exception as e:
                                error_msgs.append(f"{label}: {e}")

                        # 에러 표시
                        for em in error_msgs:
                            if "out_of_scope" in em or "권한" in em:
                                st.error(f"API 권한 부족 ({em.split(':')[0]})")
                            else:
                                st.error(f"API 오류: {em}")
                        if error_msgs and not all_rows:
                            st.info("[업비트 > 마이페이지 > Open API 관리]에서 **자산조회**, **입출금 조회** 권한을 활성화해주세요.")

                        # 날짜 필터 + 표시
                        if all_rows:
                            result_df = pd.DataFrame(all_rows)
                            # 날짜 필터링
                            try:
                                result_df['_dt'] = pd.to_datetime(result_df['_sort_dt'], errors='coerce')
                                mask = (result_df['_dt'].dt.date >= h_date_start) & (result_df['_dt'].dt.date <= h_date_end)
                                result_df = result_df[mask].sort_values('_dt', ascending=False)
                            except Exception:
                                pass
                            result_df = result_df.drop(columns=['_sort_dt', '_dt'], errors='ignore')

                            if len(result_df) > 0:
                                st.success(f"{len(result_df)}건 조회됨")
                                def _color_side(val):
                                    if val == "매수": return "color: #e74c3c"
                                    elif val == "매도": return "color: #2980b9"
                                    elif val == "입금": return "color: #27ae60"
                                    elif val == "출금": return "color: #8e44ad"
                                    return ""
                                st.dataframe(
                                    result_df.style.map(_color_side, subset=["구분"]),
                                    use_container_width=True, hide_index=True
                                )
                            else:
                                st.warning("해당 기간에 내역이 없습니다.")
                        elif not error_msgs:
                            st.warning(f"조회 결과 없음. (유형: {h_type}, 화폐: {h_curr})")
                            st.caption("Upbit API는 최근 내역만 반환합니다. 조회 유형을 변경해보세요.")

            st.caption("Upbit API 제한: 최근 100건까지 조회 가능")

        with hist_tab2:
            st.subheader("가상 계좌 관리")

            if 'virtual_adjustment' not in st.session_state:
                st.session_state.virtual_adjustment = 0

            c1, c2 = st.columns(2)
            amount = c1.number_input("금액 (KRW)", step=100000)
            if c2.button("입출금 (가상)"):
                st.session_state.virtual_adjustment += amount
                st.success(f"가상 잔고 조정: {amount:,.0f} KRW")

            st.info(f"누적 가상 조정액: {st.session_state.virtual_adjustment:,.0f} KRW")

        with hist_tab3:
            st.subheader("슬리피지 분석 (실제 체결 vs 백테스트)")

            if not trader:
                st.warning("API Key가 필요합니다.")
            else:
                sa_col1, sa_col2 = st.columns(2)
                sa_ticker_list = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
                sa_ticker = sa_col1.selectbox("코인 선택", sa_ticker_list, key="sa_ticker")
                sa_interval = sa_col2.selectbox("시간봉", list(INTERVAL_MAP.keys()), key="sa_interval")

                if st.button("슬리피지 분석", key="sa_run"):
                    with st.spinner("체결 데이터 조회 중..."):
                        # 1. 실제 체결 주문 조회
                        done_orders = trader.get_done_orders(sa_ticker)

                        if not done_orders:
                            st.info("체결 완료된 주문이 없습니다.")
                        else:
                            df_orders = pd.DataFrame(done_orders)

                            # 필요한 컬럼 처리
                            if 'created_at' in df_orders.columns:
                                df_orders['date'] = pd.to_datetime(df_orders['created_at'])
                            if 'price' in df_orders.columns:
                                df_orders['exec_price'] = pd.to_numeric(df_orders['price'], errors='coerce')
                            if 'executed_volume' in df_orders.columns:
                                df_orders['exec_volume'] = pd.to_numeric(df_orders['executed_volume'], errors='coerce')

                            # 2. 해당 기간 OHLCV 조회 → Open 가격과 비교
                            api_interval = INTERVAL_MAP.get(sa_interval, "day")
                            df_ohlcv = data_cache.get_ohlcv_local_first(
                                sa_ticker,
                                interval=api_interval,
                                count=200,
                                allow_api_fallback=True,
                            )

                            if df_ohlcv is not None and 'date' in df_orders.columns and 'exec_price' in df_orders.columns:
                                # 날짜별 Open 가격 매핑
                                df_ohlcv['open_price'] = df_ohlcv['open']

                                slip_data = []
                                for _, order in df_orders.iterrows():
                                    order_date = order.get('date')
                                    exec_price = order.get('exec_price', 0)
                                    side = order.get('side', '')

                                    if pd.isna(order_date) or exec_price == 0:
                                        continue

                                    # 가장 가까운 캔들의 Open 가격 찾기
                                    if df_ohlcv.index.tz is not None and order_date.tzinfo is None:
                                        order_date = order_date.tz_localize(df_ohlcv.index.tz)

                                    idx = df_ohlcv.index.searchsorted(order_date)
                                    if idx < len(df_ohlcv):
                                        candle_open = df_ohlcv.iloc[idx]['open']
                                        slippage_pct = (exec_price - candle_open) / candle_open * 100
                                        if side == 'ask':  # 매도
                                            slippage_pct = -slippage_pct

                                        slip_data.append({
                                            'date': order_date,
                                            'side': 'BUY' if side == 'bid' else 'SELL',
                                            'exec_price': exec_price,
                                            'candle_open': candle_open,
                                            'slippage_pct': slippage_pct,
                                            'volume': order.get('exec_volume', 0)
                                        })

                                if slip_data:
                                    df_slip = pd.DataFrame(slip_data)

                                    # 요약 통계
                                    avg_slip = df_slip['slippage_pct'].mean()
                                    max_slip = df_slip['slippage_pct'].max()
                                    min_slip = df_slip['slippage_pct'].min()

                                    sc1, sc2, sc3, sc4 = st.columns(4)
                                    sc1.metric("평균 슬리피지", f"{avg_slip:.3f}%")
                                    sc2.metric("최대 (불리)", f"{max_slip:.3f}%")
                                    sc3.metric("최소 (유리)", f"{min_slip:.3f}%")
                                    sc4.metric("거래 수", f"{len(df_slip)}건")

                                    # 매수/매도 분리 통계
                                    buy_slip = df_slip[df_slip['side'] == 'BUY']
                                    sell_slip = df_slip[df_slip['side'] == 'SELL']

                                    if not buy_slip.empty:
                                        st.caption(f"매수 평균 슬리피지: {buy_slip['slippage_pct'].mean():.3f}% ({len(buy_slip)}건)")
                                    if not sell_slip.empty:
                                        st.caption(f"매도 평균 슬리피지: {sell_slip['slippage_pct'].mean():.3f}% ({len(sell_slip)}건)")

                                    # 차트
                                    fig_slip = go.Figure()
                                    fig_slip.add_trace(go.Bar(
                                        x=df_slip['date'], y=df_slip['slippage_pct'],
                                        marker_color=['red' if s > 0 else 'green' for s in df_slip['slippage_pct']],
                                        name='슬리피지 %'
                                    ))
                                    fig_slip.add_hline(y=avg_slip, line_dash="dash", line_color="blue",
                                                       annotation_text=f"Avg: {avg_slip:.3f}%")
                                    fig_slip.update_layout(title="거래 슬리피지 (+ = 불리)", height=350, margin=dict(t=80),
                                        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                                    st.plotly_chart(fig_slip, use_container_width=True)

                                    # 상세 테이블
                                    st.dataframe(
                                        df_slip.style.format({
                                            'exec_price': '{:,.0f}',
                                            'candle_open': '{:,.0f}',
                                            'slippage_pct': '{:.3f}%',
                                            'volume': '{:.6f}'
                                        }).background_gradient(cmap='RdYlGn_r', subset=['slippage_pct']),
                                        use_container_width=True
                                    )

                                    st.info(
                                        f"권장 백테스트 슬리피지: **{abs(avg_slip):.2f}%** "
                                        f"(실제 평균 기반, 백테스트 탭에서 설정)"
                                    )
                                else:
                                    st.info("매칭 가능한 체결-캔들 데이터가 없습니다.")
                            else:
                                st.dataframe(df_orders)
                                st.caption("OHLCV 매칭 불가 - 원본 주문 데이터 표시")

    # --- Tab 4: 백테스트 ---
    with tab4:
        bt_sub1, bt_sub2, bt_sub4, bt_sub3 = st.tabs(
            ["📈 개별 백테스트", "🛠️ 파라미터 최적화", "🧩 보조 전략(역추세)", "📡 전체 종목 스캔"]
        )

        # === 서브탭1: 개별 백테스트 ===
        with bt_sub1:
            st.header("개별 자산 백테스트")

            port_tickers_bt = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
            base_options_bt = list(dict.fromkeys(port_tickers_bt + TOP_20_TICKERS))

            selected_ticker_bt = st.selectbox("백테스트 대상", base_options_bt + ["직접입력"], key="bt_target")

            bt_ticker = ""
            bt_sma = 0
            bt_buy_period = 20
            bt_sell_period = 10

            if selected_ticker_bt == "직접입력":
                bt_custom = st.text_input("코인", "BTC", key="bt_c")
                bt_ticker = f"KRW-{bt_custom.upper()}"
            else:
                bt_ticker = selected_ticker_bt

            port_match = next((item for item in portfolio_list if f"{item['market']}-{item['coin'].upper()}" == bt_ticker), None)

            default_strat_idx = 0
            if port_match and port_match.get('strategy') == 'Donchian':
                default_strat_idx = 1

            bt_strategy = st.selectbox("전략 선택", ["SMA 전략", "돈키안 전략"], index=default_strat_idx, key="bt_strategy_sel")

            # 기본값 초기화 (전략에 따라 덮어씀)
            bt_sma = 60
            bt_buy_period = 20
            bt_sell_period = 10
            bt_sell_mode = "하단선 (Lower)"  # SMA일 때 사용 안 되지만 undefined 방지

            if bt_strategy == "SMA 전략":
                default_sma = port_match.get('parameter', 60) if port_match else 60
                bt_sma = st.number_input("단기 SMA (추세)", value=default_sma, key="bt_sma_select", min_value=5, step=1)
            else:
                default_buy = int(port_match.get('parameter', 20)) if port_match and port_match.get('strategy') == 'Donchian' else 20
                default_sell = int(port_match.get('sell_parameter', 10)) if port_match and port_match.get('strategy') == 'Donchian' else 10
                if default_sell == 0:
                    default_sell = max(5, default_buy // 2)
                dc_col1, dc_col2 = st.columns(2)
                with dc_col1:
                    bt_buy_period = st.number_input("매수 채널 기간", value=default_buy, min_value=5, max_value=300, step=1, key="bt_dc_buy")
                with dc_col2:
                    bt_sell_period = st.number_input("매도 채널 기간", value=default_sell, min_value=5, max_value=300, step=1, key="bt_dc_sell")

                st.divider()
                st.caption("📌 매도 기준 선택")
                bt_sell_mode = st.radio(
                    "매도 라인",
                    ["하단선 (Lower)", "중심선 (Midline)", "두 방법 비교"],
                    horizontal=True,
                    key="bt_sell_mode",
                    help="하단선: 저가 채널 이탈 시 매도 / 중심선: (상단+하단)/2 이탈 시 매도"
                )

            default_interval_idx = 0
            if port_match:
                port_iv_label = INTERVAL_REV_MAP.get(port_match.get('interval', 'day'), '일봉')
                interval_keys = list(INTERVAL_MAP.keys())
                if port_iv_label in interval_keys:
                    default_interval_idx = interval_keys.index(port_iv_label)

            bt_interval_label = st.selectbox("시간봉 선택", options=list(INTERVAL_MAP.keys()), index=default_interval_idx, key="bt_interval_sel")
            bt_interval = INTERVAL_MAP[bt_interval_label]

            DEFAULT_SLIPPAGE = {
                "major": {"day": 0.03, "minute240": 0.05, "minute60": 0.08, "minute30": 0.08, "minute15": 0.10, "minute5": 0.15, "minute1": 0.20},
                "mid":   {"day": 0.05, "minute240": 0.08, "minute60": 0.10, "minute30": 0.10, "minute15": 0.15, "minute5": 0.20, "minute1": 0.30},
                "alt":   {"day": 0.10, "minute240": 0.15, "minute60": 0.20, "minute30": 0.20, "minute15": 0.25, "minute5": 0.35, "minute1": 0.50},
            }
            MAJOR_COINS = {"BTC", "ETH"}
            MID_COINS = {"XRP", "SOL", "DOGE", "ADA", "TRX", "AVAX", "LINK", "BCH", "DOT", "ETC"}

            def get_default_slippage(ticker, interval):
                coin = ticker.split("-")[-1].upper() if "-" in ticker else ticker.upper()
                if coin in MAJOR_COINS:
                    tier = "major"
                elif coin in MID_COINS:
                    tier = "mid"
                else:
                    tier = "alt"
                return DEFAULT_SLIPPAGE[tier].get(interval, 0.10)

            default_slip = get_default_slippage(bt_ticker, bt_interval)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.caption("백테스트 기간")
                d_col1, d_col2 = st.columns(2)
                try:
                    default_start_bt = datetime(2020, 1, 1).date()
                except:
                    default_start_bt = datetime.now().date() - timedelta(days=365)
                default_end_bt = datetime.now().date()

                bt_start = d_col1.date_input("시작일", value=default_start_bt, max_value=datetime.now().date(), format="YYYY.MM.DD", key="bt_start")
                bt_end = d_col2.date_input("종료일", value=default_end_bt, max_value=datetime.now().date(), format="YYYY.MM.DD", key="bt_end")

                if bt_start > bt_end:
                    st.error("시작일은 종료일보다 빨라야 합니다.")
                    bt_end = bt_start

                days_diff = (bt_end - bt_start).days
                st.caption(f"기간: {days_diff}일")

                fee = st.number_input("매매 수수료 (%)", value=0.05, format="%.2f", key="bt_fee") / 100
                bt_slippage = st.number_input("슬리피지 (%)", value=default_slip, min_value=0.0, max_value=2.0, step=0.01, format="%.2f", key="bt_slip")

                fee_pct = fee * 100
                cost_per_trade = fee_pct + bt_slippage
                cost_round_trip = (fee_pct * 2) + (bt_slippage * 2)
                st.caption(f"편도: {cost_per_trade:.2f}% | 왕복: {cost_round_trip:.2f}%")

                run_btn = st.button("백테스트 실행", type="primary", key="bt_run")

            if run_btn:
                if bt_strategy == "돈키안 전략":
                    req_period = max(bt_buy_period, bt_sell_period)
                    bt_strategy_mode = "Donchian"
                    bt_sell_ratio = bt_sell_period / bt_buy_period if bt_buy_period > 0 else 0.5
                    # 매도방식 파싱
                    _smode_raw = bt_sell_mode if bt_strategy == "돈키안 전략" else "하단선 (Lower)"
                    _compare_mode = _smode_raw == "두 방법 비교"
                    _sell_mode_api = "midline" if _smode_raw == "중심선 (Midline)" else "lower"
                else:
                    req_period = bt_sma
                    bt_strategy_mode = "SMA 전략"
                    bt_sell_ratio = 0.5
                    _compare_mode = False
                    _sell_mode_api = "lower"

                to_date = bt_end + timedelta(days=1)
                to_str = to_date.strftime("%Y-%m-%d 09:00:00")
                cpd = CANDLES_PER_DAY.get(bt_interval, 1)
                req_count = days_diff * cpd + req_period + 300
                fetch_count = max(req_count, req_period + 300)

                with st.spinner(f"백테스트 실행 중 ({bt_start} ~ {bt_end}, {bt_interval_label}, {bt_strategy})..."):
                    df_bt = data_cache.get_ohlcv_local_first(
                        bt_ticker,
                        interval=bt_interval,
                        to=to_str,
                        count=fetch_count,
                        allow_api_fallback=True,
                    )
                    if df_bt is None or df_bt.empty:
                        st.error("데이터를 가져올 수 없습니다.")
                        st.stop()

                    st.caption(f"조회된 캔들: {len(df_bt)}개 ({df_bt.index[0].strftime('%Y-%m-%d')} ~ {df_bt.index[-1].strftime('%Y-%m-%d')})")

                    result = backtest_engine.run_backtest(
                        bt_ticker, period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                        interval=bt_interval, count=fetch_count, fee=fee,
                        start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                        strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=bt_slippage,
                        sell_mode="lower" if _compare_mode else _sell_mode_api
                    )
                    # 비교 모드: 중심선 결과도 실행
                    if _compare_mode and bt_strategy_mode == "Donchian":
                        result_mid = backtest_engine.run_backtest(
                            bt_ticker, period=bt_buy_period,
                            interval=bt_interval, count=fetch_count, fee=fee,
                            start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                            strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=bt_slippage,
                            sell_mode="midline"
                        )
                    else:
                        result_mid = None

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        df = result["df"]
                        res = result["performance"]

                        # ── 비교 요약 테이블 (비교모드) ──
                        if _compare_mode and result_mid and "error" not in result_mid:
                            res_mid = result_mid["performance"]
                            st.subheader("📊 하단선 vs 중심선 비교")
                            cmp_data = {
                                "항목": ["총 수익률", "CAGR", "MDD", "샤프비율", "승률", "거래 횟수", "최종 자산"],
                                f"하단선 Lower({bt_sell_period})": [
                                    f"{res['total_return']:,.2f}%",
                                    f"{res.get('cagr', 0):,.2f}%",
                                    f"{res['mdd']:,.2f}%",
                                    f"{res.get('sharpe', 0):.2f}",
                                    f"{res['win_rate']:,.2f}%",
                                    f"{res['trade_count']}회",
                                    f"{res['final_equity']:,.0f} KRW",
                                ],
                                f"중심선 Midline": [
                                    f"{res_mid['total_return']:,.2f}%",
                                    f"{res_mid.get('cagr', 0):,.2f}%",
                                    f"{res_mid['mdd']:,.2f}%",
                                    f"{res_mid.get('sharpe', 0):.2f}",
                                    f"{res_mid['win_rate']:,.2f}%",
                                    f"{res_mid['trade_count']}회",
                                    f"{res_mid['final_equity']:,.0f} KRW",
                                ],
                            }
                            st.dataframe(pd.DataFrame(cmp_data).set_index("항목"), use_container_width=True)

                            # 승자 표시
                            if res['total_return'] > res_mid['total_return']:
                                st.success(f"✅ 하단선(Lower) 방식이 수익률 {res['total_return']:.2f}% vs {res_mid['total_return']:.2f}% 로 우수")
                            elif res_mid['total_return'] > res['total_return']:
                                st.success(f"✅ 중심선(Midline) 방식이 수익률 {res_mid['total_return']:.2f}% vs {res['total_return']:.2f}% 로 우수")
                            else:
                                st.info("두 방식의 수익률이 동일합니다.")
                            st.divider()

                        sell_mode_label = "중심선(Midline)" if _sell_mode_api == "midline" and not _compare_mode else ("하단선(Lower)" if not _compare_mode else "하단선(Lower) [기준]")
                        if not _compare_mode:
                            st.caption(f"매도방식: **{sell_mode_label}**")

                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("총 수익률", f"{res['total_return']:,.2f}%")
                        m2.metric("연평균(CAGR)", f"{res.get('cagr', 0):,.2f}%")
                        m3.metric("승률", f"{res['win_rate']:,.2f}%")
                        m4.metric("최대낙폭(MDD)", f"{res['mdd']:,.2f}%")
                        m5.metric("샤프비율", f"{res['sharpe']:.2f}")

                        trade_count = res['trade_count']
                        total_cost_pct = cost_round_trip * trade_count
                        st.success(
                            f"최종 잔고: **{res['final_equity']:,.0f} KRW** (초기 {initial_cap:,.0f} KRW) | "
                            f"거래 {trade_count}회 | 왕복비용 {cost_round_trip:.2f}% | 누적 약 {total_cost_pct:.1f}%"
                        )

                        if bt_slippage > 0:
                            result_no_slip = backtest_engine.run_backtest(
                                bt_ticker, period=bt_buy_period if bt_strategy_mode == "Donchian" else bt_sma,
                                interval=bt_interval, count=fetch_count, fee=fee,
                                start_date=bt_start, initial_balance=initial_cap, df=df_bt,
                                strategy_mode=bt_strategy_mode, sell_period_ratio=bt_sell_ratio, slippage=0.0
                            )
                            if "error" not in result_no_slip:
                                res_ns = result_no_slip['performance']
                                slip_ret_diff = res_ns['total_return'] - res['total_return']
                                slip_cost = res_ns['final_equity'] - res['final_equity']
                                st.info(f"슬리피지 영향: 수익률 차이 **{slip_ret_diff:,.2f}%p**, 금액 차이 **{slip_cost:,.0f} KRW**")

                        st.subheader("가격 & 전략 성과")
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])

                        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name='가격'), row=1, col=1, secondary_y=False)

                        if bt_strategy_mode == "Donchian":
                            upper_col = f'Donchian_Upper_{bt_buy_period}'
                            lower_col = f'Donchian_Lower_{bt_sell_period}'
                            if upper_col in df.columns:
                                fig.add_trace(go.Scatter(x=df.index, y=df[upper_col], line=dict(color='green', width=1.5, dash='dash'), name=f'상단 ({bt_buy_period})'), row=1, col=1, secondary_y=False)
                            if lower_col in df.columns:
                                fig.add_trace(go.Scatter(x=df.index, y=df[lower_col], line=dict(color='red', width=1.5, dash='dash'), name=f'하단 ({bt_sell_period})'), row=1, col=1, secondary_y=False)
                        else:
                            fig.add_trace(go.Scatter(x=df.index, y=df[f'SMA_{bt_sma}'], line=dict(color='orange', width=2), name=f'SMA {bt_sma}'), row=1, col=1, secondary_y=False)

                        fig.add_trace(go.Scatter(x=df.index, y=df['equity'], line=dict(color='blue', width=2), name='전략 자산'), row=1, col=1, secondary_y=True)

                        buy_dates = [t['date'] for t in res['trades'] if t['type'] == 'buy']
                        buy_prices = [t['price'] for t in res['trades'] if t['type'] == 'buy']
                        sell_dates = [t['date'] for t in res['trades'] if t['type'] == 'sell']
                        sell_prices = [t['price'] for t in res['trades'] if t['type'] == 'sell']
                        if buy_dates:
                            fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='매수'), row=1, col=1, secondary_y=False)
                        if sell_dates:
                            fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='매도'), row=1, col=1, secondary_y=False)

                        fig.add_trace(go.Scatter(x=df.index, y=df['drawdown'], name='낙폭 (%)', fill='tozeroy', line=dict(color='red', width=1)), row=2, col=1)
                        fig.update_layout(height=800, title_text="백테스트 결과", xaxis_rangeslider_visible=False, margin=dict(t=80),
                            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0))
                        fig.update_yaxes(title_text="가격 (KRW)", row=1, col=1, secondary_y=False)
                        fig.update_yaxes(title_text="자산 (KRW)", row=1, col=1, secondary_y=True)
                        fig.update_yaxes(title_text="낙폭 (%)", row=2, col=1)
                        fig = _apply_dd_hover_format(fig)
                        st.plotly_chart(fig, use_container_width=True)

                        if 'yearly_stats' in res:
                            st.subheader("연도별 성과")
                            st.dataframe(res['yearly_stats'].style.format("{:.2f}%"))

                        _render_performance_analysis(
                            equity_series=df.get("equity"),
                            benchmark_series=df.get("close"),
                            strategy_metrics=res,
                            strategy_label="백테스트 전략",
                            benchmark_label=f"{bt_ticker} 단순보유",
                            monte_carlo_sims=400,
                        )

                        st.info(f"전략 상태: **{res['final_status']}** | 다음 행동: **{res['next_action'] if res['next_action'] else '없음'}**")

                        with st.expander("거래 내역"):
                            if res['trades']:
                                trades_df = pd.DataFrame(res['trades'])
                                st.dataframe(trades_df.style.format({"price": "{:,.2f}", "amount": "{:,.6f}", "balance": "{:,.2f}", "profit": "{:,.2f}%"}))
                            else:
                                st.info("실행된 거래가 없습니다.")

                        csv_data = df.to_csv(index=True).encode('utf-8')
                        st.download_button(label="일별 로그 다운로드", data=csv_data, file_name=f"{bt_ticker}_{bt_start}_daily_log.csv", mime="text/csv")

        # === 서브탭2: 파라미터 최적화 ===
        with bt_sub2:
            st.header("파라미터 최적화")

            with st.expander("데이터 캐시 관리", expanded=False):
                cache_list = data_cache.list_cache()
                if cache_list:
                    st.dataframe(pd.DataFrame(cache_list), use_container_width=True, hide_index=True)
                else:
                    st.info("캐시된 데이터가 없습니다.")

                if st.button("캐시 전체 삭제", key="opt_clear_cache"):
                    data_cache.clear_cache()
                    st.success("캐시가 삭제되었습니다.")
                    st.rerun()

            # 최적화 대상 설정
            opt_port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
            opt_base_options = list(dict.fromkeys(opt_port_tickers + TOP_20_TICKERS))
            opt_ticker_sel = st.selectbox("최적화 대상", opt_base_options, key="opt_ticker")
            opt_strategy_sel = st.selectbox("전략", ["SMA 전략", "돈키안 전략"], key="opt_strat_sel")

            with st.form("optimization_form"):
                opt_method = st.radio("최적화 방법", ["Grid Search (전수 탐색)", "Optuna (베이지안 최적화)"], horizontal=True, key="opt_method")
                use_optuna = "Optuna" in opt_method

                opt_interval_label = st.selectbox("시간봉", options=list(INTERVAL_MAP.keys()), index=0, key="opt_interval_sel")
                opt_interval = INTERVAL_MAP[opt_interval_label]

                if opt_strategy_sel == "돈키안 전략":
                    st.caption("돈치안 채널의 매수/매도 기간을 최적화합니다.")
                    st.markdown("##### 매수 채널 기간")
                    oc1, oc2, oc3 = st.columns(3)
                    opt_buy_start = oc1.number_input("시작", 5, 200, 10, key="opt_dc_buy_start")
                    opt_buy_end = oc2.number_input("끝", 5, 200, 60, key="opt_dc_buy_end")
                    opt_buy_step = oc3.number_input("간격", 1, 50, 5, key="opt_dc_buy_step")
                    st.markdown("##### 매도 채널 기간")
                    oc4, oc5, oc6 = st.columns(3)
                    opt_sell_start = oc4.number_input("시작", 5, 200, 5, key="opt_dc_sell_start")
                    opt_sell_end = oc5.number_input("끝", 5, 200, 30, key="opt_dc_sell_end")
                    opt_sell_step = oc6.number_input("간격", 1, 50, 5, key="opt_dc_sell_step")
                    st.markdown("##### 매도 방식")
                    st.caption("하단선: 저가 채널 이탈 시 매도 | 중심선: (상단+하단)/2 이탈 시 매도")
                    opt_dc_sell_mode = st.radio(
                        "매도 라인",
                        ["하단선 (Lower)", "중심선 (Midline)", "두 방법 비교"],
                        horizontal=True,
                        key="opt_dc_sell_mode",
                    )
                else:
                    st.caption("SMA 이동평균 기간을 최적화합니다.")
                    st.markdown("##### SMA 기간")
                    oc1, oc2, oc3 = st.columns(3)
                    opt_s_start = oc1.number_input("시작", 5, 200, 20, key="opt_s_start")
                    opt_s_end = oc2.number_input("끝", 5, 200, 60, key="opt_s_end")
                    opt_s_step = oc3.number_input("간격", 1, 50, 5, key="opt_s_step")

                if use_optuna:
                    st.divider()
                    st.markdown("##### Optuna 설정")
                    opc1, opc2 = st.columns(2)
                    optuna_n_trials = opc1.number_input("탐색 횟수", 50, 2000, 200, step=50, key="optuna_trials")
                    optuna_objective = opc2.selectbox("목적함수", ["Calmar (CAGR/|MDD|)", "Sharpe", "수익률 (Return)", "MDD 최소"], key="optuna_obj")

                # 기간 설정
                st.divider()
                opt_d1, opt_d2 = st.columns(2)
                opt_start = opt_d1.date_input("시작일", value=datetime(2020, 1, 1).date(), key="opt_start_date")
                opt_end = opt_d2.date_input("종료일", value=datetime.now().date(), key="opt_end_date")
                opt_fee = st.number_input("수수료 (%)", value=0.05, format="%.2f", key="opt_fee") / 100
                opt_slippage = st.number_input("슬리피지 (%)", value=0.05, min_value=0.0, max_value=2.0, step=0.01, format="%.2f", key="opt_slippage")

                opt_submitted = st.form_submit_button("최적화 시작", type="primary")

            # opt_dc_sell_mode가 form 외부에서도 참조되지 않도록 기본값 세팅
            if opt_strategy_sel != "돈키안 전략":
                opt_dc_sell_mode = "하단선 (Lower)"

            if opt_submitted:
                import plotly.express as px
                opt_results = []
                opt_days_diff = (opt_end - opt_start).days

                with st.status("최적화 진행 중...", expanded=True) as status:
                    progress_bar = st.progress(0)
                    log_area = st.empty()

                    try:
                        import time as _time
                        opt_cpd = CANDLES_PER_DAY.get(opt_interval, 1)
                        to_str_opt = (opt_end + timedelta(days=1)).strftime("%Y-%m-%d 09:00:00")

                        if opt_strategy_sel == "돈키안 전략":
                            buy_range = range(opt_buy_start, opt_buy_end + 1, opt_buy_step)
                            sell_range = range(opt_sell_start, opt_sell_end + 1, opt_sell_step)
                            total_iter = len(buy_range) * len(sell_range)
                            max_req_p = max(opt_buy_end, opt_sell_end)
                            fetch_count_opt = opt_days_diff * opt_cpd + max_req_p + 300
                        else:
                            sma_range = range(opt_s_start, opt_s_end + 1, opt_s_step)
                            total_iter = len(sma_range)
                            fetch_count_opt = opt_days_diff * opt_cpd + opt_s_end + 300

                        def dl_progress(fetched, total):
                            pct = min(fetched / total, 1.0) if total > 0 else 0
                            progress_bar.progress(pct * 0.3)
                            log_area.text(f"다운로드: {fetched:,}/{total:,} candles ({pct*100:.0f}%)")

                        t0 = _time.time()
                        full_df = data_cache.get_ohlcv_cached(opt_ticker_sel, interval=opt_interval, to=to_str_opt, count=fetch_count_opt, progress_callback=dl_progress)
                        dl_elapsed = _time.time() - t0

                        if full_df is None or full_df.empty:
                            status.update(label="데이터 로드 실패", state="error")
                        else:
                            st.write(f"데이터 준비: {len(full_df):,} candles ({dl_elapsed:.1f}초)")

                            def opt_progress(idx, total, msg):
                                pct = 0.3 + (idx / total) * 0.7
                                progress_bar.progress(min(pct, 1.0))
                                log_area.text(f"{msg} ({idx}/{total})")

                            t1 = _time.time()
                            optuna_result = None

                            if use_optuna:
                                obj_map = {"Calmar (CAGR/|MDD|)": "calmar", "Sharpe": "sharpe", "수익률 (Return)": "return", "MDD 최소": "mdd"}
                                obj_key = obj_map.get(optuna_objective, "calmar")

                                if opt_strategy_sel == "돈키안 전략":
                                    optuna_result = backtest_engine.optuna_optimize(
                                        full_df, strategy_mode="Donchian", buy_range=(opt_buy_start, opt_buy_end),
                                        sell_range=(opt_sell_start, opt_sell_end), fee=opt_fee, slippage=opt_slippage,
                                        start_date=opt_start, initial_balance=initial_cap, n_trials=optuna_n_trials,
                                        objective_metric=obj_key, progress_callback=opt_progress)
                                else:
                                    optuna_result = backtest_engine.optuna_optimize(
                                        full_df, strategy_mode="SMA 전략", buy_range=(opt_s_start, opt_s_end),
                                        fee=opt_fee, slippage=opt_slippage, start_date=opt_start,
                                        initial_balance=initial_cap, n_trials=optuna_n_trials,
                                        objective_metric=obj_key, progress_callback=opt_progress)

                                for r in optuna_result['trials']:
                                    row = {"Total Return (%)": r["total_return"], "CAGR (%)": r["cagr"], "MDD (%)": r["mdd"],
                                           "Calmar": r["calmar"], "Win Rate (%)": r["win_rate"], "Sharpe": r["sharpe"], "Trades": r["trade_count"]}
                                    if opt_strategy_sel == "돈키안 전략":
                                        row["Buy Period"] = r["buy_period"]
                                        row["Sell Period"] = r["sell_period"]
                                    else:
                                        row["SMA Period"] = r["sma_period"]
                                    opt_results.append(row)
                                total_iter = optuna_n_trials
                            else:
                                if opt_strategy_sel == "돈키안 전략":
                                    buy_range  = range(opt_buy_start,  opt_buy_end  + 1, opt_buy_step)
                                    sell_range = range(opt_sell_start, opt_sell_end + 1, opt_sell_step)

                                    _modes_to_run = []
                                    if opt_dc_sell_mode == "두 방법 비교":
                                        _modes_to_run = [("lower", "하단선"), ("midline", "중심선")]
                                    elif opt_dc_sell_mode == "중심선 (Midline)":
                                        _modes_to_run = [("midline", "중심선")]
                                    else:
                                        _modes_to_run = [("lower", "하단선")]

                                    _all_mode_results = {}
                                    for _sm, _sm_label in _modes_to_run:
                                        _raw = backtest_engine.optimize_donchian(
                                            full_df, buy_range, sell_range, fee=opt_fee, slippage=opt_slippage,
                                            start_date=opt_start, initial_balance=initial_cap, progress_callback=opt_progress,
                                            sell_mode=_sm)
                                        _mode_rows = []
                                        for r in _raw:
                                            _mode_rows.append({"Buy Period": r["Buy Period"], "Sell Period": r["Sell Period"],
                                                "Total Return (%)": r["total_return"], "CAGR (%)": r["cagr"], "MDD (%)": r["mdd"],
                                                "Calmar": abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0,
                                                "Win Rate (%)": r["win_rate"], "Sharpe": r["sharpe"], "Trades": r["trade_count"]})
                                        _all_mode_results[_sm_label] = _mode_rows
                                        # 단일 모드일 때는 opt_results에도 담기
                                        if len(_modes_to_run) == 1:
                                            opt_results = _mode_rows
                                    # 비교 모드 결과를 세션 상태에 저장
                                    if len(_modes_to_run) == 2:
                                        st.session_state["opt_compare_results"] = _all_mode_results
                                        st.session_state.pop("opt_single_results", None)
                                    else:
                                        st.session_state["opt_single_results"] = {
                                            "rows": opt_results,
                                            "strategy": opt_strategy_sel,
                                            "use_optuna": use_optuna,
                                            "ticker": opt_ticker_sel,
                                            "interval": opt_interval,
                                            "start_date": str(opt_start),
                                            "end_date": str(opt_end),
                                            "fee": opt_fee,
                                            "slippage": opt_slippage,
                                            "initial_balance": initial_cap,
                                        }
                                        st.session_state.pop("opt_compare_results", None)
                                else:
                                    raw_results = backtest_engine.optimize_sma(
                                        full_df, sma_range, fee=opt_fee, slippage=opt_slippage,
                                        start_date=opt_start, initial_balance=initial_cap, progress_callback=opt_progress)
                                    for r in raw_results:
                                        opt_results.append({"SMA Period": r["SMA Period"],
                                            "Total Return (%)": r["total_return"], "CAGR (%)": r["cagr"], "MDD (%)": r["mdd"],
                                            "Calmar": abs(r["cagr"] / r["mdd"]) if r["mdd"] != 0 else 0,
                                            "Win Rate (%)": r["win_rate"], "Sharpe": r["sharpe"], "Trades": r["trade_count"]})

                            opt_elapsed = _time.time() - t1
                            status.update(label=f"완료! ({total_iter}건, {dl_elapsed:.1f}초 + {opt_elapsed:.1f}초)", state="complete")

                            # Optuna/SMA Grid 결과도 세션에 저장 (아직 저장 안 된 경우)
                            if opt_results and "opt_single_results" not in st.session_state:
                                st.session_state["opt_single_results"] = {
                                    "rows": opt_results,
                                    "strategy": opt_strategy_sel,
                                    "use_optuna": use_optuna,
                                    "ticker": opt_ticker_sel,
                                    "interval": opt_interval,
                                    "start_date": str(opt_start),
                                    "end_date": str(opt_end),
                                    "fee": opt_fee,
                                    "slippage": opt_slippage,
                                    "initial_balance": initial_cap,
                                }

                    except Exception as e:
                        status.update(label=f"오류: {e}", state="error")
                        import traceback
                        st.code(traceback.format_exc())

            # ── 결과 표시 (모든 지점에서 세션 상태로 유지됨) ───────────────────────

            def _add_robustness(df_in, neighbor_steps=2):
                """각 (Buy Period, Sell Period) 조합에 대해 인접 ±neighbor_steps 단계 Calmar 평균 = Robustness
                절대값이 아닌 정렬된 고유값 인덱스 기준으로 이웃을 찾아 step 크기에 무관하게 동작."""
                if "Robustness" in df_in.columns: return df_in
                df_out = df_in.copy()
                
                # 1. Donchian (2D)
                if "Buy Period" in df_out.columns and "Sell Period" in df_out.columns:
                    # 고유 Buy/Sell 값을 정렬해 인덱스 매핑
                    buy_vals  = sorted(df_out["Buy Period"].unique())
                    sell_vals = sorted(df_out["Sell Period"].unique())
                    buy_idx   = {v: i for i, v in enumerate(buy_vals)}
                    sell_idx  = {v: i for i, v in enumerate(sell_vals)}
                    max_bi, max_si = len(buy_vals) - 1, len(sell_vals) - 1

                    calmar_lookup = {}
                    for _, row in df_out.iterrows():
                        calmar_lookup[(row["Buy Period"], row["Sell Period"])] = row["Calmar"]

                    def _rob(bp, sp):
                        bi, si = buy_idx.get(bp, -1), sell_idx.get(sp, -1)
                        if bi == -1 or si == -1: return 0.0
                        vals = []
                        for b_i in range(max(0, bi - neighbor_steps), min(max_bi, bi + neighbor_steps) + 1):
                            for s_i in range(max(0, si - neighbor_steps), min(max_si, si + neighbor_steps) + 1):
                                k = (buy_vals[b_i], sell_vals[s_i])
                                if k in calmar_lookup:
                                    vals.append(calmar_lookup[k])
                        return round(sum(vals) / len(vals), 2) if vals else 0.0
                    df_out["Robustness"] = df_out.apply(lambda r: _rob(r["Buy Period"], r["Sell Period"]), axis=1)
                
                # 2. SMA (1D)
                elif "SMA Period" in df_out.columns:
                    vals = sorted(df_out["SMA Period"].unique())
                    v_idx = {v: i for i, v in enumerate(vals)}
                    max_i = len(vals) - 1
                    lookup = {}
                    for _, row in df_out.iterrows():
                        lookup[row["SMA Period"]] = row["Calmar"]
                    
                    def _rob_sma(val):
                        idx = v_idx.get(val, -1)
                        if idx == -1: return 0.0
                        n_vals = []
                        for i in range(max(0, idx - neighbor_steps), min(max_i, idx + neighbor_steps) + 1):
                            nv = vals[i]
                            if nv in lookup:
                                n_vals.append(lookup[nv])
                        return round(sum(n_vals) / len(n_vals), 2) if n_vals else 0.0
                    df_out["Robustness"] = df_out["SMA Period"].apply(_rob_sma)

                return df_out

            # ── 히트맵 헬퍼 (★ 표시) ──
            def _render_go_heatmap(df_in, x_col, y_col, z_col, title=""):
                import plotly.graph_objects as _go
                pivot = df_in.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean")
                pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
                if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
                    return None
                colorscale = "RdYlGn_r" if "MDD" in z_col else "RdYlGn"
                vals = pivot.values.copy()
                text_strs = np.where(np.isnan(vals), "", np.vectorize(lambda x: f"{x:.1f}")(vals))
                mask_valid = ~np.isnan(vals)
                if mask_valid.any():
                    best_flat = int(np.nanargmax(vals))
                    best_r, best_c = np.unravel_index(best_flat, vals.shape)
                    text_strs[best_r][best_c] = f"\u2605{text_strs[best_r][best_c]}"
                fig = _go.Figure(data=_go.Heatmap(
                    z=vals, x=[str(int(c)) for c in pivot.columns], y=[str(int(r)) for r in pivot.index],
                    colorscale=colorscale, text=text_strs, texttemplate="%{text}", textfont=dict(size=11),
                    hovertemplate=f"{y_col}: %{{y}}<br>{x_col}: %{{x}}<br>{z_col}: %{{z:.2f}}<extra></extra>",
                    colorbar=dict(title=z_col),
                ))
                fig.update_layout(title=title or f"{y_col} vs {x_col} ({z_col})",
                    xaxis_title=x_col, yaxis_title=y_col, xaxis_type="category", yaxis_type="category", height=400)
                return fig

            def _evaluate_cv(cv):
                if cv <= 10: return "매우 안정", "\U0001f7e2"
                elif cv <= 20: return "안정", "\U0001f535"
                elif cv <= 35: return "보통", "\U0001f7e1"
                elif cv <= 50: return "불안정", "\U0001f7e0"
                else: return "매우 불안정", "\U0001f534"

            def _compute_stability(df_in, metrics=None):
                if metrics is None:
                    metrics = ["Calmar", "Total Return (%)", "MDD (%)"]
                stability = {}
                for col in metrics:
                    if col in df_in.columns and len(df_in) > 1:
                        mean_v = df_in[col].mean()
                        std_v = df_in[col].std()
                        cv = abs(std_v / mean_v) * 100 if mean_v != 0 else 0
                        stability[col] = {"평균": round(mean_v, 2), "표준편차": round(std_v, 2),
                            "CV(%)": round(cv, 1), "최소": round(df_in[col].min(), 2), "최대": round(df_in[col].max(), 2)}
                return stability

            def _monte_carlo_sim(daily_rets, n_sims=1000, n_days=756, init_cap=10000):
                """부트스트랩 몬테카를로 시뮬레이션."""
                paths = np.zeros((n_sims, n_days + 1))
                paths[:, 0] = init_cap
                for i in range(n_sims):
                    sampled = np.random.choice(daily_rets, size=n_days, replace=True)
                    paths[i, 1:] = init_cap * np.cumprod(1 + sampled)
                final_values = paths[:, -1]
                years = n_days / 252
                cagr_dist = ((final_values / init_cap) ** (1 / max(years, 0.01)) - 1) * 100
                mdd_dist = np.zeros(n_sims)
                for i in range(n_sims):
                    peak = np.maximum.accumulate(paths[i])
                    dd = (paths[i] - peak) / peak * 100
                    mdd_dist[i] = dd.min()
                pcts = {p: float(np.percentile(final_values, p)) for p in [5, 25, 50, 75, 95]}
                return {"paths": paths, "final_values": final_values, "cagr_dist": cagr_dist,
                        "mdd_dist": mdd_dist, "percentiles": pcts}

            def _kelly_criterion(daily_rets):
                """켈리 기준 투자비중 계산."""
                wins = daily_rets[daily_rets > 0]
                losses = daily_rets[daily_rets < 0]
                if len(wins) == 0 or len(losses) == 0:
                    return {"kelly_full": 0, "kelly_half": 0, "kelly_quarter": 0, "win_rate": 0,
                            "avg_win": 0, "avg_loss": 0, "payoff_ratio": 0,
                            "grade": "분석 불가", "recommendation": "승/패 데이터가 부족합니다."}
                win_rate = len(wins) / len(daily_rets)
                avg_win = wins.mean() * 100
                avg_loss = abs(losses.mean()) * 100
                payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
                kelly_full = max(0, min(((payoff_ratio * win_rate - (1 - win_rate)) / payoff_ratio * 100) if payoff_ratio > 0 else 0, 100))
                kelly_half = kelly_full / 2
                kelly_quarter = kelly_full / 4
                if kelly_full <= 0:
                    grade, rec = "투자 불가", "Kelly 기준으로 이 전략에 투자하지 않는 것이 좋습니다."
                elif kelly_full <= 15:
                    grade, rec = "보수적", "소규모 투자만 권장합니다. 하프 켈리 이하로 운용하세요."
                elif kelly_full <= 30:
                    grade, rec = "적정", "적정 수준의 투자가 가능합니다. 하프 켈리를 기준으로 운용하세요."
                elif kelly_full <= 50:
                    grade, rec = "공격적", "높은 비중이 가능하지만, 하프 켈리로 보수적 접근을 권장합니다."
                else:
                    grade, rec = "매우 공격적", "풀 켈리는 과도한 위험을 수반합니다. 쿼터~하프 켈리를 권장합니다."
                return {"kelly_full": kelly_full, "kelly_half": kelly_half, "kelly_quarter": kelly_quarter,
                        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "payoff_ratio": payoff_ratio,
                        "grade": grade, "recommendation": rec}

            def _show_robustness_evaluation(stab):
                """CV 기반 종합 평가 한글 텍스트 표시."""
                if not stab:
                    return
                st.markdown("#### 종합 평가")
                lines = []
                for metric, info in stab.items():
                    cv = info.get("CV(%)", 0)
                    grade, icon = _evaluate_cv(cv)
                    mn, mx = info.get("최소", 0), info.get("최대", 0)
                    avg = info.get("평균", 0)
                    if metric == "MDD (%)":
                        lines.append(f"- {icon} **{metric}**: 평균 {avg:.2f}%, 범위 {mn:.2f}% ~ {mx:.2f}%, CV {cv:.1f}% → **{grade}**")
                    else:
                        lines.append(f"- {icon} **{metric}**: 평균 {avg:.2f}, 범위 {mn:.2f} ~ {mx:.2f}, CV {cv:.1f}% → **{grade}**")
                st.markdown("\n".join(lines))
                # 종합 등급
                key_metrics = ["Calmar", "Total Return (%)", "MDD (%)"]
                cvs = [stab[m]["CV(%)"] for m in key_metrics if m in stab]
                avg_cv = np.mean(cvs) if cvs else 0
                overall_grade, overall_icon = _evaluate_cv(avg_cv)
                st.markdown(f"### {overall_icon} 종합 안정성: **{overall_grade}** (평균 CV {avg_cv:.1f}%)")
                # 실전 권고
                calmar_cv = stab.get("Calmar", {}).get("CV(%)", 0)
                if calmar_cv <= 15:
                    st.markdown("> **Calmar Ratio**가 파라미터 변화에 둔감하여 안정적입니다.")
                elif calmar_cv <= 30:
                    st.markdown("> **Calmar Ratio**가 어느 정도 변동하지만 수용 가능한 수준입니다.")
                else:
                    st.markdown("> **Calmar Ratio**가 파라미터 변화에 민감합니다. 과적합 가능성을 주의하세요.")
                if avg_cv <= 15:
                    st.success("파라미터 안정성이 높아 **실전 적용에 적합**합니다.")
                elif avg_cv <= 30:
                    st.info(f"파라미터 안정성이 보통 수준입니다. 주변 파라미터 평균 성과를 기대값으로 잡는 것이 현실적입니다.")
                else:
                    st.warning("파라미터 민감도가 높아 **과적합 위험**이 있습니다. 다른 기간으로 교차 검증을 권장합니다.")

            _saved_compare = st.session_state.get("opt_compare_results", {})
            _saved_single  = st.session_state.get("opt_single_results", {})

            if _saved_compare:
                st.subheader("🔀 매도방식 비교 결과")
                tab_labels = list(_saved_compare.keys())
                cmp_tabs = st.tabs([f"📊 {lbl}" for lbl in tab_labels])
                for _tab, _lbl in zip(cmp_tabs, tab_labels):
                    with _tab:
                        _rows = _saved_compare[_lbl]
                        if _rows:
                            _df = pd.DataFrame(_rows).sort_values("Total Return (%)", ascending=False).reset_index(drop=True)
                            _df = _add_robustness(_df)
                            _df.index = _df.index + 1
                            _df.index.name = "순위"
                            _best = _df.iloc[0]
                            st.success(f"【{_lbl}】 최적: 매수 **{int(_best['Buy Period'])}**, 매도 **{int(_best['Sell Period'])}** → 수익률 {_best['Total Return (%)']:.1f}%, Calmar {_best['Calmar']:.2f}, Robustness {_best['Robustness']:.2f}")
                            st.dataframe(
                                _df.style.background_gradient(cmap='RdYlGn', subset=['Total Return (%)', 'Calmar', 'Sharpe', 'Robustness'])
                                    .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)']).format("{:,.2f}"),
                                use_container_width=True, height=400)
                            import plotly.express as _px
                            _fig = _px.density_heatmap(_df.reset_index(), x="Buy Period", y="Sell Period", z="Total Return (%)",
                                                       histfunc="avg", title=f"[{_lbl}] 히트맵", text_auto=".0f", color_continuous_scale="RdYlGn")
                            st.plotly_chart(_fig, use_container_width=True)
                        else:
                            st.info(f"{_lbl} 결과 없음")

                # 핵심 지표 비교 테이블
                if len(tab_labels) == 2:
                    st.subheader("📋 핵심 지표 비교")
                    _compare_rows = []
                    for _lbl in tab_labels:
                        _rows = _saved_compare[_lbl]
                        if _rows:
                            _df2 = pd.DataFrame(_rows)
                            # 비교 데이터프레임에도 Robustness 추가
                            _df2 = _add_robustness(_df2)
                            _best2 = _df2.sort_values("Total Return (%)", ascending=False).iloc[0]
                            _compare_rows.append({
                                "매도방식": _lbl,
                                "최적 매수": int(_best2["Buy Period"]),
                                "최적 매도": int(_best2["Sell Period"]),
                                "수익률 (%)": round(_best2["Total Return (%)"], 2),
                                "CAGR (%)": round(_best2["CAGR (%)"], 2),
                                "MDD (%)": round(_best2["MDD (%)"], 2),
                                "Calmar": round(_best2["Calmar"], 2),
                                "Robustness": round(_best2["Robustness"], 2),
                                "Sharpe": round(_best2["Sharpe"], 2),
                                "거래횟수": int(_best2["Trades"]),
                            })
                    if _compare_rows:
                        _cmp_df = pd.DataFrame(_compare_rows).set_index("매도방식")
                        st.dataframe(_cmp_df.style.highlight_max(axis=0, color="#d4edda", subset=["수익률 (%)", "Calmar", "Robustness", "Sharpe"]).highlight_min(axis=0, color="#f8d7da", subset=["MDD (%)"]), use_container_width=True)

            elif _saved_single:
                import plotly.express as px
                opt_results  = _saved_single["rows"]
                _s_strategy  = _saved_single["strategy"]
                _s_optuna    = _saved_single["use_optuna"]
                if opt_results:
                    opt_df = pd.DataFrame(opt_results)
                    opt_df = _add_robustness(opt_df)
                    _total_combos = len(opt_df)

                    # ── 결과 필터 & 정렬 ──
                    _fc1, _fc2, _fc3 = st.columns(3)
                    _SORT_OPTIONS = ["Calmar (CAGR/MDD)", "수익률 (높은순)", "CAGR (높은순)", "MDD (낮은순)", "Sharpe (높은순)", "Robustness (높은순)"]
                    _opt_sort = _fc1.selectbox("정렬 기준", _SORT_OPTIONS, key="opt_sort_by")
                    _opt_mdd_filter = _fc2.number_input("최대 MDD (%)", -100.0, 0.0, -50.0, 5.0, format="%.1f", key="opt_max_mdd", help="이 값보다 MDD가 나쁜 조합은 제외")
                    _opt_top_n = int(_fc3.number_input("상위 N개", 5, 200, 30, 5, key="opt_top_n"))

                    # 정렬
                    _sort_map = {"Calmar (CAGR/MDD)": ("Calmar", False), "수익률 (높은순)": ("Total Return (%)", False),
                                 "CAGR (높은순)": ("CAGR (%)", False), "MDD (낮은순)": ("MDD (%)", True),
                                 "Sharpe (높은순)": ("Sharpe", False), "Robustness (높은순)": ("Robustness", False)}
                    _scol, _sasc = _sort_map.get(_opt_sort, ("Calmar", False))
                    if _scol in opt_df.columns:
                        opt_df = opt_df.sort_values(_scol, ascending=_sasc).reset_index(drop=True)
                    best_row = opt_df.iloc[0]

                    # MDD 필터 (표시용)
                    _filtered_df = opt_df[opt_df["MDD (%)"] >= _opt_mdd_filter].reset_index(drop=True)
                    _n_filtered = len(_filtered_df)

                    # 최적 결과 요약
                    if _s_strategy == "돈키안 전략":
                        st.success(f"최적: 매수 **{int(best_row['Buy Period'])}**, 매도 **{int(best_row['Sell Period'])}** → 수익률 {best_row['Total Return (%)']:.1f}%, CAGR {best_row['CAGR (%)']:.1f}%, MDD {best_row['MDD (%)']:.1f}%, Calmar {best_row['Calmar']:.2f}, Robustness {best_row.get('Robustness', 0):.2f}")
                    else:
                        st.success(f"최적: SMA **{int(best_row['SMA Period'])}** → 수익률 {best_row['Total Return (%)']:.1f}%, CAGR {best_row['CAGR (%)']:.1f}%, MDD {best_row['MDD (%)']:.1f}%, Calmar {best_row['Calmar']:.2f}, Robustness {best_row.get('Robustness', 0):.2f}")
                    if _n_filtered < _total_combos:
                        st.caption(f"총 {_total_combos}개 중 {_n_filtered}개 통과 (MDD ≥ {_opt_mdd_filter:.1f}%) | {_opt_sort} 기준 상위 {min(_opt_top_n, _n_filtered)}개 표시")
                    else:
                        st.caption(f"총 {_total_combos}개 | {_opt_sort} 기준 상위 {min(_opt_top_n, _total_combos)}개 표시")

                    # 결과 테이블 (Top-N, 필터 적용)
                    _display_src = _filtered_df if _n_filtered > 0 else opt_df
                    _display_df = _display_src.head(_opt_top_n).copy()
                    _display_df.index = _display_df.index + 1
                    _display_df.index.name = "순위"
                    _grad_cols = [c for c in ['Total Return (%)', 'Calmar', 'Sharpe', 'Robustness'] if c in _display_df.columns]
                    st.dataframe(
                        _display_df.style.background_gradient(cmap='RdYlGn', subset=_grad_cols)
                            .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)']).format("{:,.2f}"),
                        use_container_width=True, height=500)

                    # 🔍 상세 분석 Expander (Robustness Check)
                    with st.expander("🔍 최적 파라미터 주변 상세 분석 (Robustness)", expanded=False):
                        try:
                            if _s_strategy == "돈키안 전략" and "Buy Period" in opt_df.columns:
                                st.caption("최적 (Buy, Sell) 파라미터 기준 ±2단계 이웃들의 성과를 분석합니다.")
                                b_val, s_val = int(best_row["Buy Period"]), int(best_row["Sell Period"])
                                b_uniq = sorted(opt_df["Buy Period"].unique())
                                s_uniq = sorted(opt_df["Sell Period"].unique())
                                
                                if b_val in b_uniq and s_val in s_uniq:
                                    b_idx, s_idx = b_uniq.index(b_val), s_uniq.index(s_val)
                                    nb_vals = b_uniq[max(0, b_idx-2) : min(len(b_uniq), b_idx+3)]
                                    ns_vals = s_uniq[max(0, s_idx-2) : min(len(s_uniq), s_idx+3)]
                                    
                                    sub_df = opt_df[
                                        (opt_df["Buy Period"].isin(nb_vals)) & 
                                        (opt_df["Sell Period"].isin(ns_vals))
                                    ].copy()
                                    
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric("이웃 평균 수익률", f"{sub_df['Total Return (%)'].mean():.1f}%")
                                    c2.metric("이웃 평균 Calmar", f"{sub_df['Calmar'].mean():.2f}")
                                    c3.metric("이웃 최소 MDD", f"{sub_df['MDD (%)'].min():.2f}%")
                                    
                                    st.dataframe(sub_df.style.background_gradient(cmap='RdYlGn', subset=['Calmar']), use_container_width=True)
                                else:
                                    st.warning("파라미터 인덱스 조회 실패")
                            
                            elif _s_strategy != "돈키안 전략" and "SMA Period" in opt_df.columns:
                                st.caption("최적 SMA Period 기준 ±2단계 이웃들의 성과를 분석합니다.")
                                p_val = int(best_row["SMA Period"])
                                p_uniq = sorted(opt_df["SMA Period"].unique())
                                
                                if p_val in p_uniq:
                                    p_idx = p_uniq.index(p_val)
                                    np_vals = p_uniq[max(0, p_idx-2) : min(len(p_uniq), p_idx+3)]
                                    
                                    sub_df = opt_df[opt_df["SMA Period"].isin(np_vals)].copy()
                                    
                                    c1, c2 = st.columns(2)
                                    c1.metric("이웃 평균 수익률", f"{sub_df['Total Return (%)'].mean():.1f}%")
                                    c2.metric("이웃 평균 Calmar", f"{sub_df['Calmar'].mean():.2f}")
                                    
                                    st.bar_chart(sub_df.set_index("SMA Period")[["Calmar", "Total Return (%)"]])
                                else:
                                    st.warning("파라미터 인덱스 조회 실패")
                        except Exception as e:
                            st.error(f"상세 분석 중 오류 발생: {e}")

                    if _s_strategy == "돈키안 전략" and not _s_optuna:
                        fig_opt = px.density_heatmap(opt_df.reset_index(), x="Buy Period", y="Sell Period", z="Total Return (%)", histfunc="avg", title="돈키안 최적화 히트맵", text_auto=".0f", color_continuous_scale="RdYlGn")
                        st.plotly_chart(fig_opt, use_container_width=True)
                    elif _s_strategy != "돈키안 전략" and not _s_optuna:
                        st.line_chart(opt_df.reset_index().set_index("SMA Period")[['Total Return (%)', 'MDD (%)']])

        # === 서브탭3: 전체 종목 스캔 ===
        with bt_sub4:
            st.header("보조 전략 백테스트")
            st.caption("메인 전략이 CASH일 때만 보조 분할매수 전략을 실행합니다.")

            # 위젯 생성 전 pending 적용 (Streamlit widget key 직접 수정 오류 방지)
            _aux_pending_apply = st.session_state.pop("aux_opt_apply_pending", None)
            if isinstance(_aux_pending_apply, dict) and _aux_pending_apply:
                for _k, _v in _aux_pending_apply.items():
                    st.session_state[_k] = _v

            aux_col1, aux_col2 = st.columns(2)

            with aux_col1:
                st.subheader("메인 전략 설정")
                _aux_port_tickers = [f"{r['market']}-{r['coin'].upper()}" for r in portfolio_list]
                _aux_base = list(dict.fromkeys(_aux_port_tickers + TOP_20_TICKERS))
                aux_ticker = st.selectbox("대상 티커", _aux_base + ["직접 입력"], key="aux_bt_ticker")
                if aux_ticker == "직접 입력":
                    aux_ticker = st.text_input("티커 입력 (예: KRW-BTC)", "KRW-BTC", key="aux_bt_ticker_custom")

                aux_main_strat = st.selectbox("메인 전략", ["Donchian", "SMA"], key="aux_bt_main_strat")
                amc1, amc2 = st.columns(2)
                aux_main_buy = amc1.number_input("메인 매수 기간", 5, 300, 115, key="aux_bt_main_buy")
                aux_main_sell = amc2.number_input(
                    "메인 매도 기간", 0, 300, 55, key="aux_bt_main_sell", help="SMA 선택 시 0이면 자동으로 buy_period/2를 사용합니다."
                )
                if aux_main_sell == 0:
                    aux_main_sell = max(5, int(aux_main_buy) // 2)

            with aux_col2:
                st.subheader("보조 전략 설정")
                aux_ma_count_label = st.radio("이평선 수", ["2개", "1개"], horizontal=True, key="aux_bt_ma_count")
                apc1, apc2 = st.columns(2)
                aux_ma_short = apc1.number_input("단기 MA", 3, 500, 5, key="aux_bt_ma_short")
                if aux_ma_count_label == "2개":
                    aux_ma_long = apc2.number_input("장기 MA", 5, 300, 20, key="aux_bt_ma_long")
                    if aux_ma_long <= aux_ma_short:
                        aux_ma_long = aux_ma_short + 1
                else:
                    aux_ma_long = int(aux_ma_short)
                    apc2.caption("1개 모드: 장기 MA는 사용하지 않습니다.")

                aux_threshold = st.slider("과매도 임계값(이격도 %)", -30.0, -0.5, -5.0, 0.5, key="aux_bt_threshold")
                aux_use_rsi = st.checkbox("RSI 필터 사용", value=False, key="aux_bt_use_rsi")
                if aux_use_rsi:
                    arc1, arc2 = st.columns(2)
                    aux_rsi_period = int(arc1.number_input("RSI 기간", min_value=2, max_value=50, value=2, step=1, key="aux_bt_rsi_period"))
                    aux_rsi_threshold = float(arc2.number_input("RSI 과매도 기준", min_value=1.0, max_value=30.0, value=8.0, step=0.5, key="aux_bt_rsi_threshold"))
                else:
                    aux_rsi_period = int(st.session_state.get("aux_bt_rsi_period", 2))
                    aux_rsi_threshold = float(st.session_state.get("aux_bt_rsi_threshold", 8.0))

                atc1, atc2 = st.columns(2)
                aux_tp1 = atc1.number_input("TP1 - 1차 매도 (%)", 1.0, 30.0, 3.0, 0.5, key="aux_bt_tp1")
                aux_tp2 = atc2.number_input("TP2 - 2차 매도 (%)", 1.0, 50.0, 10.0, 0.5, key="aux_bt_tp2")
                if aux_tp2 < aux_tp1:
                    aux_tp2 = aux_tp1
                st.caption("매도: TP1 도달 시 50% 매도 → TP2 도달 시 나머지 50% 매도")

                aux_split = st.number_input("분할 매수 횟수", 1, 20, 3, key="aux_bt_split")
                aux_seed_label = st.radio("매수 시드 방식", ["동일", "피라미딩"], horizontal=True, key="aux_bt_seed_mode")
                aux_seed_mode = "pyramiding" if aux_seed_label == "피라미딩" else "equal"

                aux_pyramid_ratio = 1.0
                if aux_seed_mode == "pyramiding":
                    aux_pyramid_ratio = st.number_input("피라미딩 배율", 1.05, 3.00, 1.30, 0.05, key="aux_bt_pyramid_ratio")

                _weights = np.ones(int(aux_split), dtype=float)
                if aux_seed_mode == "pyramiding":
                    _weights = np.array([aux_pyramid_ratio ** i for i in range(int(aux_split))], dtype=float)
                _weights = _weights / _weights.sum()
                st.caption("매수 시드 비중: " + " / ".join([f"{w * 100:.1f}%" for w in _weights]))

            iv_col1, iv_col2, iv_col3, iv_col4, iv_col5 = st.columns(5)
            aux_interval_label = iv_col1.selectbox(
                "보조 실행 주기",
                list(INTERVAL_MAP.keys()),
                index=2 if len(INTERVAL_MAP) > 2 else 0,
                key="aux_bt_interval",
            )
            aux_main_interval_label = iv_col2.selectbox(
                "메인 신호 주기",
                list(INTERVAL_MAP.keys()),
                index=1 if len(INTERVAL_MAP) > 1 else 0,
                key="aux_bt_main_interval",
            )
            _aux_start_default = datetime(2020, 1, 1).date()
            try:
                _aux_start_default = start_date
            except Exception:
                pass
            aux_start = iv_col3.date_input("시작일", value=_aux_start_default, key="aux_bt_start")
            aux_fee = iv_col4.number_input("수수료(%)", 0.0, 1.0, 0.05, 0.01, key="aux_bt_fee") / 100.0
            aux_slippage = iv_col5.number_input("슬리피지(%)", 0.0, 2.0, 0.10, 0.05, key="aux_bt_slip")

            def _prepare_aux_frames(_warmup_max: int):
                api_iv = INTERVAL_MAP.get(aux_interval_label, "day")
                api_main_iv = INTERVAL_MAP.get(aux_main_interval_label, api_iv)

                days = max((datetime.now().date() - aux_start).days, 30)
                cpd = CANDLES_PER_DAY.get(api_iv, 1)
                main_cpd = CANDLES_PER_DAY.get(api_main_iv, 1)

                base_warmup = max(int(_warmup_max), int(aux_main_buy), int(aux_main_sell), 30)
                aux_count = min(max(days * cpd + base_warmup + 300, 500), 12000)
                main_count = min(max(days * main_cpd + base_warmup + 300, 500), 12000)

                df_aux_local = data_cache.get_ohlcv_cached(aux_ticker, interval=api_iv, count=aux_count)
                if df_aux_local is None or len(df_aux_local) < max(50, int(_warmup_max) + 5):
                    return None, None, api_iv, api_main_iv, "보조 실행 캔들 데이터가 부족합니다."

                df_main_local = None
                if api_main_iv != api_iv:
                    df_main_local = data_cache.get_ohlcv_cached(aux_ticker, interval=api_main_iv, count=main_count)
                    if df_main_local is None or len(df_main_local) < max(50, int(aux_main_buy) + 5):
                        return None, None, api_iv, api_main_iv, "메인 신호 캔들 데이터가 부족합니다."

                return df_aux_local, df_main_local, api_iv, api_main_iv, None

            run_aux = st.button("보조 전략 백테스트 실행", type="primary", key="run_aux_bt")

            if run_aux:
                with st.spinner("보조 전략 백테스트 실행 중..."):
                    _warmup_bt = int(aux_ma_long) if aux_ma_count_label == "2개" else int(aux_ma_short)
                    if aux_use_rsi:
                        _warmup_bt = max(_warmup_bt, int(aux_rsi_period))
                    df_aux, df_main_aux, api_iv, api_main_iv, _prep_err = _prepare_aux_frames(_warmup_bt)
                    if _prep_err:
                        st.error(_prep_err)
                    elif api_main_iv == api_iv or df_main_aux is not None:
                        res_aux = backtest_engine.run_aux_backtest(
                            df_aux,
                            main_strategy=aux_main_strat,
                            main_buy_p=int(aux_main_buy),
                            main_sell_p=int(aux_main_sell),
                            ma_count=(1 if aux_ma_count_label == "1개" else 2),
                            ma_short=int(aux_ma_short),
                            ma_long=int(aux_ma_long),
                            oversold_threshold=float(aux_threshold),
                            tp1_pct=float(aux_tp1),
                            tp2_pct=float(aux_tp2),
                            fee=float(aux_fee),
                            slippage=float(aux_slippage),
                            start_date=str(aux_start),
                            initial_balance=initial_cap,
                            split_count=int(aux_split),
                            buy_seed_mode=aux_seed_mode,
                            pyramid_ratio=float(aux_pyramid_ratio),
                            use_rsi_filter=bool(aux_use_rsi),
                            rsi_period=int(aux_rsi_period),
                            rsi_threshold=float(aux_rsi_threshold),
                            main_df=(None if api_main_iv == api_iv else df_main_aux),
                        )
                        st.session_state["aux_bt_result"] = res_aux

            if "aux_bt_result" in st.session_state:
                abr = st.session_state["aux_bt_result"]
                if isinstance(abr, dict) and "error" in abr:
                    st.error(f"백테스트 오류: {abr['error']}")
                elif isinstance(abr, dict):
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("총수익률", f"{abr.get('total_return', 0):.2f}%")
                    m2.metric("CAGR", f"{abr.get('cagr', 0):.2f}%")
                    m3.metric("MDD", f"{abr.get('mdd', 0):.2f}%")
                    _calmar = abs(abr.get('cagr', 0) / abr.get('mdd', 1e-9)) if abr.get('mdd', 0) != 0 else 0
                    m4.metric("Calmar", f"{_calmar:.2f}")
                    m5.metric("승률", f"{abr.get('win_rate', 0):.1f}%")
                    m6.metric("거래 수", f"{abr.get('trade_count', 0)}")

                    _seed_mode_out = abr.get("buy_seed_mode", aux_seed_mode)
                    _seed_note = (
                        f" x{abr.get('pyramid_ratio', aux_pyramid_ratio):.2f}"
                        if _seed_mode_out == "pyramiding"
                        else ""
                    )
                    _use_rsi_out = bool(abr.get("use_rsi_filter", aux_use_rsi))
                    _rsi_note = ""
                    if _use_rsi_out:
                        _rsi_note = (
                            f" | RSI({int(abr.get('rsi_period', aux_rsi_period))})"
                            f"<={float(abr.get('rsi_threshold', aux_rsi_threshold)):.1f}"
                        )
                    st.caption(
                        f"시드={_seed_mode_out}{_seed_note}"
                        + f" | MA={aux_ma_count_label}"
                        + f" | split={int(aux_split)}"
                        + _rsi_note
                        + f" | interval={aux_interval_label}/{aux_main_interval_label}"
                    )
                    st.info(
                        f"상태: {abr.get('final_status', 'N/A')} | "
                        f"다음 액션: {abr.get('next_action') if abr.get('next_action') else '-'}"
                    )

                    _dates = abr.get("dates")
                    _strat_ret = abr.get("strategy_return_curve")
                    _bench_ret = abr.get("benchmark_return_curve")
                    _strat_dd = abr.get("drawdown_curve")
                    _bench_dd = abr.get("benchmark_dd_curve")

                    if _dates is not None and _strat_ret is not None and len(_strat_ret) > 1:
                        _plot_df = pd.DataFrame({"date": pd.to_datetime(_dates)})
                        _plot_df["strategy_ret"] = np.asarray(_strat_ret, dtype=float)
                        if _bench_ret is not None and len(_bench_ret) == len(_plot_df):
                            _plot_df["benchmark_ret"] = np.asarray(_bench_ret, dtype=float)
                        if _strat_dd is not None and len(_strat_dd) == len(_plot_df):
                            _plot_df["strategy_dd"] = np.asarray(_strat_dd, dtype=float)
                        if _bench_dd is not None and len(_bench_dd) == len(_plot_df):
                            _plot_df["benchmark_dd"] = np.asarray(_bench_dd, dtype=float)

                        fig_aux = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                        fig_aux.add_trace(
                            go.Scatter(x=_plot_df["date"], y=_plot_df["strategy_ret"], mode="lines", name="Aux Return (%)"),
                            row=1,
                            col=1,
                        )
                        if "benchmark_ret" in _plot_df.columns:
                            fig_aux.add_trace(
                                go.Scatter(x=_plot_df["date"], y=_plot_df["benchmark_ret"], mode="lines", name="B&H Return (%)"),
                                row=1,
                                col=1,
                            )
                        if "strategy_dd" in _plot_df.columns:
                            fig_aux.add_trace(
                                go.Scatter(x=_plot_df["date"], y=_plot_df["strategy_dd"], mode="lines", name="Aux DD (%)"),
                                row=2,
                                col=1,
                            )
                        if "benchmark_dd" in _plot_df.columns:
                            fig_aux.add_trace(
                                go.Scatter(x=_plot_df["date"], y=_plot_df["benchmark_dd"], mode="lines", name="B&H DD (%)"),
                                row=2,
                                col=1,
                            )
                        fig_aux.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=20))
                        fig_aux.update_yaxes(title_text="Return (%)", row=1, col=1)
                        fig_aux.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
                        fig_aux = _apply_dd_hover_format(fig_aux)
                        st.plotly_chart(fig_aux, use_container_width=True)

                        _rsi_curve = abr.get("rsi_curve")
                        _rsi_enabled = bool(abr.get("use_rsi_filter", False))
                        if _rsi_enabled and _rsi_curve is not None and len(_rsi_curve) == len(_plot_df):
                            _rsi_thr = float(abr.get("rsi_threshold", aux_rsi_threshold))
                            _rsi_p = int(abr.get("rsi_period", aux_rsi_period))
                            _fig_rsi = go.Figure()
                            _fig_rsi.add_trace(
                                go.Scatter(
                                    x=_plot_df["date"],
                                    y=np.asarray(_rsi_curve, dtype=float),
                                    mode="lines",
                                    name=f"RSI({_rsi_p})",
                                )
                            )
                            _fig_rsi.add_hline(y=_rsi_thr, line_dash="dash", line_color="#ef4444", annotation_text=f"과매도 기준 {_rsi_thr:.1f}")
                            _fig_rsi.add_hline(y=30.0, line_dash="dot", line_color="#9ca3af")
                            _fig_rsi.add_hline(y=70.0, line_dash="dot", line_color="#9ca3af")
                            _fig_rsi.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="RSI")
                            st.plotly_chart(_fig_rsi, use_container_width=True)

            st.divider()
            st.subheader("보조 전략 최적화")
            st.caption("보조 전략 백테스트 아래에서 바로 최적화를 실행합니다.")

            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
            aux_opt_method_label = opt_col1.selectbox("최적화 방식", ["Optuna", "그리드"], key="aux_opt_method")
            if aux_opt_method_label == "Optuna":
                aux_opt_trials = int(opt_col2.number_input("시도 횟수", min_value=20, max_value=500, value=80, step=10, key="aux_opt_trials"))
                aux_opt_max_grid_evals = 30000
            else:
                aux_opt_max_grid_evals = int(
                    opt_col2.number_input(
                        "그리드 최대 평가 수",
                        min_value=500,
                        max_value=200000,
                        value=30000,
                        step=500,
                        key="aux_opt_max_grid_evals",
                    )
                )
                aux_opt_trials = int(st.session_state.get("aux_opt_trials", 80))

            aux_opt_obj_label = opt_col3.selectbox("목표 지표", ["Calmar", "Sharpe", "수익률", "낮은 MDD"], key="aux_opt_objective")
            aux_opt_min_trades = int(opt_col4.number_input("최소 거래 수", min_value=0, max_value=200, value=5, step=1, key="aux_opt_min_trades"))
            aux_opt_use_rsi = st.checkbox("최적화에 RSI 필터 포함", value=bool(aux_use_rsi), key="aux_opt_use_rsi")

            st.caption("최적화 범위")
            ms_col1, ms_col2 = st.columns(2)
            aux_opt_ms_min = int(ms_col1.number_input("단기 MA 최소", min_value=2, max_value=500, value=3, step=1, key="aux_opt_ms_min"))
            aux_opt_ms_max = int(ms_col2.number_input("단기 MA 최대", min_value=2, max_value=500, value=30, step=1, key="aux_opt_ms_max"))

            if aux_ma_count_label == "2개":
                ml_col1, ml_col2 = st.columns(2)
                aux_opt_ml_min = int(ml_col1.number_input("장기 MA 최소", min_value=3, max_value=200, value=10, step=1, key="aux_opt_ml_min"))
                aux_opt_ml_max = int(ml_col2.number_input("장기 MA 최대", min_value=3, max_value=240, value=120, step=1, key="aux_opt_ml_max"))
            else:
                aux_opt_ml_min = int(aux_opt_ms_min)
                aux_opt_ml_max = int(aux_opt_ms_max)
                st.caption("1개 모드에서는 장기 MA를 최적화하지 않습니다.")

            thr_col1, thr_col2 = st.columns(2)
            aux_opt_thr_min = float(thr_col1.number_input("임계값 최소(%)", min_value=-30.0, max_value=-0.5, value=-15.0, step=0.5, key="aux_opt_thr_min"))
            aux_opt_thr_max = float(thr_col2.number_input("임계값 최대(%)", min_value=-30.0, max_value=-0.5, value=-1.0, step=0.5, key="aux_opt_thr_max"))

            tp_row1, tp_row2, split_row1, split_row2 = st.columns(4)
            aux_opt_tp1_min = float(tp_row1.number_input("TP1 최소(%)", min_value=0.5, max_value=30.0, value=2.0, step=0.5, key="aux_opt_tp1_min"))
            aux_opt_tp1_max = float(tp_row2.number_input("TP1 최대(%)", min_value=0.5, max_value=30.0, value=10.0, step=0.5, key="aux_opt_tp1_max"))
            aux_opt_split_min = int(split_row1.number_input("분할수 최소", min_value=1, max_value=20, value=1, step=1, key="aux_opt_split_min"))
            aux_opt_split_max = int(split_row2.number_input("분할수 최대", min_value=1, max_value=20, value=5, step=1, key="aux_opt_split_max"))

            tp2_row1, tp2_row2 = st.columns(2)
            aux_opt_tp2_min = float(tp2_row1.number_input("TP2 최소(%)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, key="aux_opt_tp2_min"))
            aux_opt_tp2_max = float(tp2_row2.number_input("TP2 최대(%)", min_value=0.5, max_value=50.0, value=20.0, step=0.5, key="aux_opt_tp2_max"))

            if aux_opt_use_rsi:
                rsi_row1, rsi_row2 = st.columns(2)
                aux_opt_rsi_p_min = int(rsi_row1.number_input("RSI 기간 최소", min_value=2, max_value=50, value=2, step=1, key="aux_opt_rsi_p_min"))
                aux_opt_rsi_p_max = int(rsi_row2.number_input("RSI 기간 최대", min_value=2, max_value=50, value=10, step=1, key="aux_opt_rsi_p_max"))
                rsi_t_row1, rsi_t_row2 = st.columns(2)
                aux_opt_rsi_t_min = float(rsi_t_row1.number_input("RSI 기준 최소", min_value=1.0, max_value=30.0, value=5.0, step=0.5, key="aux_opt_rsi_t_min"))
                aux_opt_rsi_t_max = float(rsi_t_row2.number_input("RSI 기준 최대", min_value=1.0, max_value=30.0, value=10.0, step=0.5, key="aux_opt_rsi_t_max"))
            else:
                aux_opt_rsi_p_min = int(aux_rsi_period)
                aux_opt_rsi_p_max = int(aux_rsi_period)
                aux_opt_rsi_t_min = float(aux_rsi_threshold)
                aux_opt_rsi_t_max = float(aux_rsi_threshold)

            aux_opt_ms_max = max(aux_opt_ms_min, aux_opt_ms_max)
            if aux_ma_count_label == "2개":
                aux_opt_ml_min = max(aux_opt_ms_min + 1, aux_opt_ml_min)
                aux_opt_ml_max = max(aux_opt_ml_min, aux_opt_ml_max)
            else:
                aux_opt_ml_min = aux_opt_ms_min
                aux_opt_ml_max = aux_opt_ms_max
            if aux_opt_thr_min > aux_opt_thr_max:
                aux_opt_thr_min, aux_opt_thr_max = aux_opt_thr_max, aux_opt_thr_min
            if aux_opt_tp1_min > aux_opt_tp1_max:
                aux_opt_tp1_min, aux_opt_tp1_max = aux_opt_tp1_max, aux_opt_tp1_min
            if aux_opt_tp2_min > aux_opt_tp2_max:
                aux_opt_tp2_min, aux_opt_tp2_max = aux_opt_tp2_max, aux_opt_tp2_min
            if aux_opt_tp2_min < aux_opt_tp1_min:
                aux_opt_tp2_min = aux_opt_tp1_min
            if aux_opt_tp2_max < aux_opt_tp2_min:
                aux_opt_tp2_max = aux_opt_tp2_min
            if aux_opt_split_min > aux_opt_split_max:
                aux_opt_split_min, aux_opt_split_max = aux_opt_split_max, aux_opt_split_min
            if aux_opt_rsi_p_min > aux_opt_rsi_p_max:
                aux_opt_rsi_p_min, aux_opt_rsi_p_max = aux_opt_rsi_p_max, aux_opt_rsi_p_min
            if aux_opt_rsi_t_min > aux_opt_rsi_t_max:
                aux_opt_rsi_t_min, aux_opt_rsi_t_max = aux_opt_rsi_t_max, aux_opt_rsi_t_min

            if aux_opt_method_label == "그리드":
                st.caption("그리드 간격")
                gs1, gs2, gs3, gs4 = st.columns(4)
                aux_opt_ms_step = int(gs1.number_input("단기 MA 간격", min_value=1, max_value=100, value=5, step=1, key="aux_opt_ms_step"))
                if aux_ma_count_label == "2개":
                    aux_opt_ml_step = int(gs2.number_input("장기 MA 간격", min_value=1, max_value=100, value=5, step=1, key="aux_opt_ml_step"))
                else:
                    aux_opt_ml_step = int(max(1, aux_opt_ms_step))
                    gs2.caption("1개 모드: 장기 MA 간격 미사용")
                aux_opt_thr_step = float(gs3.number_input("임계값 간격(%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="aux_opt_thr_step"))
                aux_opt_tp_step = float(gs4.number_input("TP 간격(%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="aux_opt_tp_step"))
                aux_opt_split_step = int(st.number_input("분할수 간격", min_value=1, max_value=5, value=1, step=1, key="aux_opt_split_step"))
                if aux_opt_use_rsi:
                    gr1, gr2 = st.columns(2)
                    aux_opt_rsi_p_step = int(gr1.number_input("RSI 기간 간격", min_value=1, max_value=10, value=1, step=1, key="aux_opt_rsi_p_step"))
                    aux_opt_rsi_t_step = float(gr2.number_input("RSI 기준 간격", min_value=0.1, max_value=5.0, value=0.5, step=0.1, key="aux_opt_rsi_t_step"))
                else:
                    aux_opt_rsi_p_step = 1
                    aux_opt_rsi_t_step = 0.5

                def _grid_count(_vmin, _vmax, _step):
                    _step = max(float(_step), 1e-9)
                    _span = max(float(_vmax) - float(_vmin), 0.0)
                    return int(np.floor(_span / _step + 1e-9)) + 1

                _ms_n = _grid_count(aux_opt_ms_min, aux_opt_ms_max, aux_opt_ms_step)
                _ml_n = 1 if aux_ma_count_label == "1개" else _grid_count(aux_opt_ml_min, aux_opt_ml_max, aux_opt_ml_step)
                _thr_n = _grid_count(aux_opt_thr_min, aux_opt_thr_max, aux_opt_thr_step)
                _tp1_n = _grid_count(aux_opt_tp1_min, aux_opt_tp1_max, aux_opt_tp_step)
                _tp2_n = _grid_count(aux_opt_tp2_min, aux_opt_tp2_max, aux_opt_tp_step)
                _sp_n = _grid_count(aux_opt_split_min, aux_opt_split_max, aux_opt_split_step)
                _rsi_p_n = _grid_count(aux_opt_rsi_p_min, aux_opt_rsi_p_max, aux_opt_rsi_p_step) if aux_opt_use_rsi else 1
                _rsi_t_n = _grid_count(aux_opt_rsi_t_min, aux_opt_rsi_t_max, aux_opt_rsi_t_step) if aux_opt_use_rsi else 1
                _est_grid = int(_ms_n * _ml_n * _thr_n * _tp1_n * _tp2_n * _sp_n * _rsi_p_n * _rsi_t_n)
                st.caption(f"예상 조합 수(필터 전): 약 {_est_grid:,}개 | 최대 평가 수: {int(aux_opt_max_grid_evals):,}개")
            else:
                aux_opt_ms_step = 1
                aux_opt_ml_step = 1
                aux_opt_thr_step = 0.5
                aux_opt_tp_step = 0.5
                aux_opt_split_step = 1
                aux_opt_rsi_p_step = 1
                aux_opt_rsi_t_step = 0.5

            run_aux_opt = st.button("보조 전략 최적화 실행", type="secondary", key="run_aux_opt")
            if run_aux_opt:
                with st.spinner("보조 전략 최적화 실행 중..."):
                    _objective_map = {"Calmar": "calmar", "Sharpe": "sharpe", "수익률": "return", "낮은 MDD": "mdd"}
                    _warmup_opt = int(aux_opt_ml_max) if aux_ma_count_label == "2개" else int(aux_opt_ms_max)
                    if aux_opt_use_rsi:
                        _warmup_opt = max(_warmup_opt, int(aux_opt_rsi_p_max))
                    df_aux_opt, df_main_aux_opt, api_iv_opt, api_main_iv_opt, _prep_err = _prepare_aux_frames(_warmup_opt)
                    if _prep_err:
                        st.error(_prep_err)
                    else:
                        _pbar = st.progress(0)
                        _pmsg = st.empty()

                        def _aux_opt_progress(cur, total, msg):
                            _pct = int((float(cur) / max(float(total), 1.0)) * 100.0)
                            _pbar.progress(max(0, min(100, _pct)))
                            _pmsg.caption(f"{cur}/{total} | {msg}")

                        try:
                            _opt_result = backtest_engine.optimize_aux(
                                df_aux_opt,
                                main_strategy=aux_main_strat,
                                main_buy_p=int(aux_main_buy),
                                main_sell_p=int(aux_main_sell),
                                ma_count=(1 if aux_ma_count_label == "1개" else 2),
                                ma_short_range=(int(aux_opt_ms_min), int(aux_opt_ms_max)),
                                ma_long_range=(int(aux_opt_ml_min), int(aux_opt_ml_max)),
                                threshold_range=(float(aux_opt_thr_min), float(aux_opt_thr_max)),
                                tp1_range=(float(aux_opt_tp1_min), float(aux_opt_tp1_max)),
                                tp2_range=(float(aux_opt_tp2_min), float(aux_opt_tp2_max)),
                                split_count_range=(int(aux_opt_split_min), int(aux_opt_split_max)),
                                fee=float(aux_fee),
                                slippage=float(aux_slippage),
                                start_date=str(aux_start),
                                initial_balance=initial_cap,
                                n_trials=int(aux_opt_trials),
                                objective_metric=_objective_map.get(aux_opt_obj_label, "calmar"),
                                progress_callback=_aux_opt_progress,
                                buy_seed_mode=aux_seed_mode,
                                pyramid_ratio=float(aux_pyramid_ratio),
                                main_df=(None if api_main_iv_opt == api_iv_opt else df_main_aux_opt),
                                min_trade_count=int(aux_opt_min_trades),
                                optimization_method=("grid" if aux_opt_method_label == "그리드" else "optuna"),
                                ma_short_step=int(aux_opt_ms_step),
                                ma_long_step=int(aux_opt_ml_step),
                                threshold_step=float(aux_opt_thr_step),
                                tp_step=float(aux_opt_tp_step),
                                split_step=int(aux_opt_split_step),
                                max_grid_evals=int(aux_opt_max_grid_evals),
                                use_rsi_filter=bool(aux_opt_use_rsi),
                                rsi_period_range=(int(aux_opt_rsi_p_min), int(aux_opt_rsi_p_max)),
                                rsi_threshold_range=(float(aux_opt_rsi_t_min), float(aux_opt_rsi_t_max)),
                                rsi_period_step=int(aux_opt_rsi_p_step),
                                rsi_threshold_step=float(aux_opt_rsi_t_step),
                            )
                            _pbar.progress(100)
                            _pmsg.caption("최적화 완료")
                            st.session_state["aux_opt_result"] = {
                                "raw": _opt_result,
                                "method_label": aux_opt_method_label,
                                "objective_label": aux_opt_obj_label,
                                "ma_label": aux_ma_count_label,
                                "ticker": aux_ticker,
                                "interval": f"{aux_interval_label}/{aux_main_interval_label}",
                            }
                        except Exception as e:
                            st.session_state["aux_opt_result"] = {"error": str(e)}
                            st.error(f"보조 전략 최적화 오류: {e}")

            if "aux_opt_result" in st.session_state:
                _aor = st.session_state["aux_opt_result"]
                st.markdown("#### 최적화 결과")
                if isinstance(_aor, dict) and _aor.get("error"):
                    st.error(_aor.get("error", "알 수 없는 오류"))
                else:
                    _opt_rows = ((_aor or {}).get("raw", {}) or {}).get("trials", [])
                    if not _opt_rows:
                        st.info("최적화 결과가 없습니다.")
                    else:
                        _opt_df = pd.DataFrame(_opt_rows)
                        if "score" in _opt_df.columns:
                            _opt_df = _opt_df.sort_values("score", ascending=False).reset_index(drop=True)
                        _best = _opt_df.iloc[0]

                        b1, b2, b3, b4, b5, b6 = st.columns(6)
                        b1.metric("최적 점수", f"{float(_best.get('score', 0.0)):.2f}")
                        b2.metric("총수익률", f"{float(_best.get('total_return', 0.0)):.2f}%")
                        b3.metric("CAGR", f"{float(_best.get('cagr', 0.0)):.2f}%")
                        b4.metric("MDD", f"{float(_best.get('mdd', 0.0)):.2f}%")
                        b5.metric("Calmar", f"{float(_best.get('calmar', 0.0)):.2f}")
                        b6.metric("거래 수", f"{int(_best.get('trade_count', 0))}")

                        _raw_opt = ((_aor or {}).get("raw", {}) or {})
                        _eval_cnt = int(_raw_opt.get("evaluated_count", len(_opt_rows)))
                        _est_cnt = _raw_opt.get("total_estimated", None)
                        _method_lbl = str(_aor.get("method_label", "Optuna"))

                        st.caption(
                            f"방식={_method_lbl} | "
                            f"평가={_eval_cnt:,}건"
                            + (f" / 예상={int(_est_cnt):,}건" if _est_cnt is not None else "")
                            + " | "
                            f"목표={_aor.get('objective_label', 'Calmar')} | "
                            f"MA={_aor.get('ma_label', aux_ma_count_label)} | "
                            f"티커={_aor.get('ticker', aux_ticker)} | "
                            f"주기={_aor.get('interval', f'{aux_interval_label}/{aux_main_interval_label}')}"
                        )

                        _top_n = int(st.number_input("결과 표시 개수", min_value=5, max_value=200, value=30, step=5, key="aux_opt_top_n"))
                        _view_cols = [
                            "MA Count", "MA Short", "MA Long", "Threshold", "TP1 %", "TP2 %", "Split",
                            "Use RSI", "RSI Period", "RSI Threshold",
                            "total_return", "cagr", "mdd", "calmar", "sharpe", "win_rate", "trade_count", "score",
                        ]
                        _view_cols = [c for c in _view_cols if c in _opt_df.columns]
                        _show_df = _opt_df[_view_cols].head(_top_n).copy()
                        _show_df = _show_df.rename(columns={
                            "MA Count": "MA수",
                            "MA Short": "단기MA",
                            "MA Long": "장기MA",
                            "Threshold": "임계(%)",
                            "TP1 %": "TP1(%)",
                            "TP2 %": "TP2(%)",
                            "Split": "분할수",
                            "Use RSI": "RSI사용",
                            "RSI Period": "RSI기간",
                            "RSI Threshold": "RSI기준",
                            "total_return": "총수익률(%)",
                            "cagr": "CAGR(%)",
                            "mdd": "MDD(%)",
                            "calmar": "Calmar",
                            "sharpe": "Sharpe",
                            "win_rate": "승률(%)",
                            "trade_count": "거래수",
                            "score": "점수",
                        })
                        if "RSI사용" in _show_df.columns:
                            _show_df["RSI사용"] = _show_df["RSI사용"].map(lambda v: "예" if bool(v) else "아니오")
                        _grad_cols = [c for c in ["총수익률(%)", "CAGR(%)", "Calmar", "Sharpe", "승률(%)", "점수"] if c in _show_df.columns]
                        _num_cols = [c for c in _show_df.columns if pd.api.types.is_numeric_dtype(_show_df[c])]
                        st.dataframe(
                            _show_df.style.background_gradient(cmap="RdYlGn", subset=_grad_cols)
                            .background_gradient(cmap="RdYlGn_r", subset=[c for c in ["MDD(%)"] if c in _show_df.columns])
                            .format("{:,.2f}", subset=_num_cols),
                            use_container_width=True,
                            hide_index=True,
                        )

                        _best_ma_count = int(_best.get("MA Count", 2))
                        _best_ma_short = int(_best.get("MA Short", aux_ma_short))
                        _best_ma_long = int(_best.get("MA Long", aux_ma_long))
                        _best_thr = float(_best.get("Threshold", aux_threshold))
                        _best_tp1 = float(_best.get("TP1 %", aux_tp1))
                        _best_tp2 = float(_best.get("TP2 %", aux_tp2))
                        _best_split = int(_best.get("Split", aux_split))
                        _best_use_rsi = bool(_best.get("Use RSI", aux_opt_use_rsi))
                        _best_rsi_p = int(_best.get("RSI Period", aux_rsi_period))
                        _best_rsi_t = float(_best.get("RSI Threshold", aux_rsi_threshold))

                        # 최적 파라미터 기준 보조 전략 자체 성과/DD 차트
                        _curve_sig = (
                            str(aux_ticker),
                            str(aux_interval_label),
                            str(aux_main_interval_label),
                            str(aux_start),
                            float(aux_fee),
                            float(aux_slippage),
                            int(aux_main_buy),
                            int(aux_main_sell),
                            int(_best_ma_count),
                            int(_best_ma_short),
                            int(_best_ma_long),
                            float(_best_thr),
                            float(_best_tp1),
                            float(_best_tp2),
                            int(_best_split),
                            bool(_best_use_rsi),
                            int(_best_rsi_p),
                            float(_best_rsi_t),
                            str(aux_seed_mode),
                            float(aux_pyramid_ratio),
                            int(initial_cap),
                        )
                        if st.session_state.get("aux_opt_best_curve_sig") != _curve_sig:
                            _warmup_curve = int(_best_ma_long) if _best_ma_count == 2 else int(_best_ma_short)
                            if _best_use_rsi:
                                _warmup_curve = max(_warmup_curve, int(_best_rsi_p))
                            _df_aux_curve, _df_main_curve, _api_iv_curve, _api_main_iv_curve, _curve_prep_err = _prepare_aux_frames(_warmup_curve)
                            if _curve_prep_err:
                                st.session_state["aux_opt_best_curve"] = {"error": _curve_prep_err}
                            else:
                                with st.spinner("최적 파라미터 성과/DD 차트 계산 중..."):
                                    _curve_res = backtest_engine.run_aux_backtest(
                                        _df_aux_curve,
                                        main_strategy=aux_main_strat,
                                        main_buy_p=int(aux_main_buy),
                                        main_sell_p=int(aux_main_sell),
                                        ma_count=int(_best_ma_count),
                                        ma_short=int(_best_ma_short),
                                        ma_long=int(_best_ma_long),
                                        oversold_threshold=float(_best_thr),
                                        tp1_pct=float(_best_tp1),
                                        tp2_pct=float(_best_tp2),
                                        fee=float(aux_fee),
                                        slippage=float(aux_slippage),
                                        start_date=str(aux_start),
                                        initial_balance=initial_cap,
                                        split_count=int(_best_split),
                                        buy_seed_mode=aux_seed_mode,
                                        pyramid_ratio=float(aux_pyramid_ratio),
                                        use_rsi_filter=bool(_best_use_rsi),
                                        rsi_period=int(_best_rsi_p),
                                        rsi_threshold=float(_best_rsi_t),
                                        main_df=(None if _api_main_iv_curve == _api_iv_curve else _df_main_curve),
                                    )
                                st.session_state["aux_opt_best_curve"] = _curve_res
                            st.session_state["aux_opt_best_curve_sig"] = _curve_sig

                        _curve_show = st.session_state.get("aux_opt_best_curve", {})
                        if isinstance(_curve_show, dict) and _curve_show.get("error"):
                            st.warning(f"최적 파라미터 차트 생성 실패: {_curve_show.get('error')}")
                        elif isinstance(_curve_show, dict):
                            _dates2 = _curve_show.get("dates")
                            _ret2 = _curve_show.get("strategy_return_curve")
                            _dd2 = _curve_show.get("drawdown_curve")
                            _bench_ret2 = _curve_show.get("benchmark_return_curve")
                            _bench_dd2 = _curve_show.get("benchmark_dd_curve")
                            _disp_s2 = _curve_show.get("disparity_short_curve")
                            _disp_l2 = _curve_show.get("disparity_long_curve")
                            _thr2 = float(_curve_show.get("oversold_threshold", _best_thr))
                            _ma_count2 = int(_curve_show.get("ma_count", _best_ma_count))
                            _ma_s2 = int(_curve_show.get("ma_short", _best_ma_short))
                            _ma_l2 = int(_curve_show.get("ma_long", _best_ma_long))
                            _use_rsi2 = bool(_curve_show.get("use_rsi_filter", _best_use_rsi))
                            _rsi_p2 = int(_curve_show.get("rsi_period", _best_rsi_p))
                            _rsi_thr2 = float(_curve_show.get("rsi_threshold", _best_rsi_t))
                            _rsi2 = _curve_show.get("rsi_curve")
                            if _dates2 is not None and _ret2 is not None and len(_ret2) > 1:
                                _cdf = pd.DataFrame({"date": pd.to_datetime(_dates2)})
                                _cdf["strategy_ret"] = np.asarray(_ret2, dtype=float)
                                if _bench_ret2 is not None and len(_bench_ret2) == len(_cdf):
                                    _cdf["benchmark_ret"] = np.asarray(_bench_ret2, dtype=float)
                                if _dd2 is not None and len(_dd2) == len(_cdf):
                                    _cdf["strategy_dd"] = np.asarray(_dd2, dtype=float)
                                if _bench_dd2 is not None and len(_bench_dd2) == len(_cdf):
                                    _cdf["benchmark_dd"] = np.asarray(_bench_dd2, dtype=float)

                                st.markdown("##### 보조 전략 자체 성과 차트")
                                _fig_perf = go.Figure()
                                _fig_perf.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["strategy_ret"], mode="lines", name="보조 전략 수익률(%)"))
                                if "benchmark_ret" in _cdf.columns:
                                    _fig_perf.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["benchmark_ret"], mode="lines", name="단순보유 수익률(%)", line=dict(dash="dot")))
                                _fig_perf.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="수익률(%)")
                                st.plotly_chart(_fig_perf, use_container_width=True)

                                st.markdown("##### 보조 전략 DD 차트")
                                _fig_dd = go.Figure()
                                if "strategy_dd" in _cdf.columns:
                                    _fig_dd.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["strategy_dd"], mode="lines", name="보조 전략 DD(%)"))
                                if "benchmark_dd" in _cdf.columns:
                                    _fig_dd.add_trace(go.Scatter(x=_cdf["date"], y=_cdf["benchmark_dd"], mode="lines", name="단순보유 DD(%)", line=dict(dash="dot")))
                                _fig_dd.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="DD(%)")
                                _fig_dd = _apply_dd_hover_format(_fig_dd)
                                st.plotly_chart(_fig_dd, use_container_width=True)

                                if _disp_s2 is not None and len(_disp_s2) == len(_cdf):
                                    st.markdown("##### 기준 이평선 이격도 차트")
                                    _fig_disp = go.Figure()
                                    _fig_disp.add_trace(
                                        go.Scatter(
                                            x=_cdf["date"],
                                            y=np.asarray(_disp_s2, dtype=float),
                                            mode="lines",
                                            name=f"단기MA({_ma_s2}) 이격도(%)",
                                        )
                                    )
                                    if _ma_count2 == 2 and _disp_l2 is not None and len(_disp_l2) == len(_cdf):
                                        _fig_disp.add_trace(
                                            go.Scatter(
                                                x=_cdf["date"],
                                                y=np.asarray(_disp_l2, dtype=float),
                                                mode="lines",
                                                name=f"장기MA({_ma_l2}) 이격도(%)",
                                                line=dict(dash="dot"),
                                            )
                                        )
                                    _fig_disp.add_hline(y=0.0, line_dash="dot", line_color="#9ca3af")
                                    _fig_disp.add_hline(
                                        y=float(_thr2),
                                        line_dash="dash",
                                        line_color="#ef4444",
                                        annotation_text=f"과매도 임계값 {_thr2:.2f}%",
                                        annotation_position="top right",
                                    )
                                    _fig_disp.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="이격도(%)")
                                    st.plotly_chart(_fig_disp, use_container_width=True)

                                if _use_rsi2 and _rsi2 is not None and len(_rsi2) == len(_cdf):
                                    st.markdown("##### RSI 차트")
                                    _fig_rsi2 = go.Figure()
                                    _fig_rsi2.add_trace(
                                        go.Scatter(
                                            x=_cdf["date"],
                                            y=np.asarray(_rsi2, dtype=float),
                                            mode="lines",
                                            name=f"RSI({_rsi_p2})",
                                        )
                                    )
                                    _fig_rsi2.add_hline(y=float(_rsi_thr2), line_dash="dash", line_color="#ef4444", annotation_text=f"RSI 기준 {_rsi_thr2:.1f}")
                                    _fig_rsi2.add_hline(y=30.0, line_dash="dot", line_color="#9ca3af")
                                    _fig_rsi2.add_hline(y=70.0, line_dash="dot", line_color="#9ca3af")
                                    _fig_rsi2.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=10), yaxis_title="RSI")
                                    st.plotly_chart(_fig_rsi2, use_container_width=True)

                        if st.button("최적 파라미터를 현재 입력값에 반영", key="aux_opt_apply_best"):
                            st.session_state["aux_opt_apply_pending"] = {
                                "aux_bt_ma_count": "1개" if _best_ma_count == 1 else "2개",
                                "aux_bt_ma_short": int(_best_ma_short),
                                "aux_bt_ma_long": int(max(5, _best_ma_long)),
                                "aux_bt_threshold": float(_best_thr),
                                "aux_bt_tp1": float(_best_tp1),
                                "aux_bt_tp2": float(_best_tp2),
                                "aux_bt_split": int(_best_split),
                                "aux_bt_use_rsi": bool(_best_use_rsi),
                                "aux_bt_rsi_period": int(_best_rsi_p),
                                "aux_bt_rsi_threshold": float(_best_rsi_t),
                            }
                            st.rerun()

        with bt_sub3:
            st.header("전체 종목 스캔")
            st.caption("상위 종목을 전 시간대/전략으로 백테스트하여 Calmar 순으로 정렬합니다.")

            # 스캔 설정
            scan_col1, scan_col2, scan_col3 = st.columns(3)
            scan_strategy = scan_col1.selectbox("전략", ["SMA", "Donchian"], key="scan_strat")
            scan_period = scan_col2.number_input("기간 (Period)", 5, 300, 20, key="scan_period")
            scan_count = scan_col3.number_input("백테스트 캔들 수", 200, 10000, 2000, step=200, key="scan_count")

            scan_col4, scan_col5 = st.columns(2)
            _scan_interval_alias = {
                "일봉": "1D",
                "4시간": "4H",
                "1시간": "1H",
                "30분": "30m",
                "15분": "15m",
                "5분": "5m",
                "1분": "1m",
            }
            _scan_default_raw = st.session_state.get("scan_intervals", ["1D", "4H", "1H"])
            if not isinstance(_scan_default_raw, (list, tuple)):
                _scan_default_raw = ["1D", "4H", "1H"]
            _scan_defaults = []
            for _v in _scan_default_raw:
                _k = _scan_interval_alias.get(str(_v), str(_v))
                if _k in INTERVAL_MAP and _k not in _scan_defaults:
                    _scan_defaults.append(_k)
            if not _scan_defaults:
                _scan_defaults = [k for k in ["1D", "4H", "1H"] if k in INTERVAL_MAP]

            scan_intervals = scan_col4.multiselect(
                "시간봉", list(INTERVAL_MAP.keys()),
                default=_scan_defaults,
                key="scan_intervals"
            )
            sell_ratio = 0.5
            if scan_strategy == "Donchian":
                sell_ratio = st.slider("매도 채널 비율", 0.1, 1.0, 0.5, 0.1, key="scan_sell_ratio")

            st.caption(f"대상: 시가총액 상위 {len(TOP_20_TICKERS)}개 — {', '.join(t.replace('KRW-','') for t in TOP_20_TICKERS)}")

            if st.button("🔍 스캔 시작", key="scan_run", type="primary"):
                engine = BacktestEngine()
                top_tickers = TOP_20_TICKERS

                if top_tickers:
                    interval_apis = [INTERVAL_MAP[k] for k in scan_intervals]
                    total_jobs = len(top_tickers) * len(interval_apis)
                    st.write(f"종목 {len(top_tickers)}개 × 시간봉 {len(interval_apis)}개 = 총 **{total_jobs}건** 백테스트")

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []
                    done = 0

                    for ticker in top_tickers:
                        for interval_api in interval_apis:
                            done += 1
                            interval_label = INTERVAL_REV_MAP.get(interval_api, interval_api)
                            status_text.text(f"[{done}/{total_jobs}] {ticker} ({interval_label})")
                            progress_bar.progress(done / total_jobs)

                            try:
                                # 데이터 조회 (캐시 우선)
                                df = data_cache.get_ohlcv_cached(ticker, interval=interval_api, count=scan_count)
                                if df is None or len(df) < scan_period + 10:
                                    continue

                                df = df.copy()

                                # 시그널 생성
                                if scan_strategy == "Donchian":
                                    strat = DonchianStrategy()
                                    sell_p = max(5, int(scan_period * sell_ratio))
                                    df = strat.create_features(df, buy_period=scan_period, sell_period=sell_p)
                                    signal_arr = np.zeros(len(df), dtype=np.int8)
                                    upper_col = f'Donchian_Upper_{scan_period}'
                                    lower_col = f'Donchian_Lower_{sell_p}'
                                    if upper_col in df.columns and lower_col in df.columns:
                                        signal_arr[df['close'].values > df[upper_col].values] = 1
                                        signal_arr[df['close'].values < df[lower_col].values] = -1
                                    else:
                                        continue
                                else:
                                    sma_vals = df['close'].rolling(window=scan_period).mean().values
                                    close_vals = df['close'].values
                                    signal_arr = np.zeros(len(df), dtype=np.int8)
                                    valid = ~np.isnan(sma_vals)
                                    signal_arr[valid & (close_vals > sma_vals)] = 1
                                    signal_arr[valid & (close_vals <= sma_vals)] = -1

                                open_arr = df['open'].values
                                close_arr = df['close'].values

                                # 고속 시뮬레이션
                                res = engine._fast_simulate(open_arr, close_arr, signal_arr, fee=0.0005, slippage=0.0, initial_balance=1000000)

                                # Buy & Hold 수익률
                                bnh_return = (close_arr[-1] / close_arr[0] - 1) * 100

                                # Calmar = CAGR / |MDD| (MDD가 0이면 inf 방지)
                                calmar = abs(res['cagr'] / res['mdd']) if res['mdd'] != 0 else 0

                                results.append({
                                    '종목': ticker,
                                    '시간봉': interval_label,
                                    'CAGR (%)': round(res['cagr'], 2),
                                    'MDD (%)': round(res['mdd'], 2),
                                    'Calmar': round(calmar, 2),
                                    '수익률 (%)': round(res['total_return'], 2),
                                    'B&H (%)': round(bnh_return, 2),
                                    '초과수익 (%)': round(res['total_return'] - bnh_return, 2),
                                    '승률 (%)': round(res['win_rate'], 1),
                                    '거래수': res['trade_count'],
                                    'Sharpe': round(res['sharpe'], 2),
                                    '캔들수': len(df),
                                })
                            except Exception:
                                continue

                    progress_bar.progress(1.0)
                    status_text.text(f"완료! {len(results)}건 결과")

                    if results:
                        df_results = pd.DataFrame(results)
                        df_results = df_results.sort_values('Calmar', ascending=False).reset_index(drop=True)
                        df_results.index = df_results.index + 1  # 1부터 시작
                        df_results.index.name = "순위"

                        # 요약
                        st.success(f"스캔 완료: {len(results)}건 중 수익 {len(df_results[df_results['수익률 (%)'] > 0])}건, 손실 {len(df_results[df_results['수익률 (%)'] <= 0])}건")

                        # Calmar 상위 결과 테이블
                        st.dataframe(
                            df_results.style.format({
                                'CAGR (%)': '{:.2f}',
                                'MDD (%)': '{:.2f}',
                                'Calmar': '{:.2f}',
                                '수익률 (%)': '{:.2f}',
                                'B&H (%)': '{:.2f}',
                                '초과수익 (%)': '{:.2f}',
                                '승률 (%)': '{:.1f}',
                                'Sharpe': '{:.2f}',
                            }).background_gradient(cmap='RdYlGn', subset=['Calmar', '초과수익 (%)'])
                            .background_gradient(cmap='RdYlGn_r', subset=['MDD (%)']),
                            use_container_width=True,
                            height=700,
                        )

                        # 전략별/시간봉별 요약
                        st.divider()
                        sum_col1, sum_col2 = st.columns(2)
                        with sum_col1:
                            st.caption("시간봉별 평균 Calmar")
                            interval_summary = df_results.groupby('시간봉').agg(
                                Calmar_평균=('Calmar', 'mean'),
                                수익률_평균=('수익률 (%)', 'mean'),
                                종목수=('종목', 'count')
                            ).sort_values('Calmar_평균', ascending=False)
                            st.dataframe(interval_summary.style.format({'Calmar_평균': '{:.2f}', '수익률_평균': '{:.2f}'}), use_container_width=True)

                        with sum_col2:
                            st.caption("종목별 최고 Calmar 시간봉")
                            best_per_ticker = df_results.loc[df_results.groupby('종목')['Calmar'].idxmax()][['종목', '시간봉', 'Calmar', '수익률 (%)', 'MDD (%)']].reset_index(drop=True)
                            best_per_ticker.index = best_per_ticker.index + 1
                            st.dataframe(best_per_ticker.style.format({'Calmar': '{:.2f}', '수익률 (%)': '{:.2f}', 'MDD (%)': '{:.2f}'}), use_container_width=True)
                    else:
                        st.warning("결과가 없습니다. 데이터 다운로드가 필요할 수 있습니다.")


if __name__ == "__main__":
    main()
