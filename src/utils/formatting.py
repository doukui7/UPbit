from src.constants import ETF_NAME_KR, ISA_WDR_TRADE_ETF_SET

def _safe_float(v, default=0.0):
    try:
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if s.endswith("%"):
                s = s[:-1]
            if s in ("", "-", "--", "None", "null"):
                return default
            return float(s)
        return float(v)
    except Exception:
        return default

def _safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def _etf_name_kr(code: str) -> str:
    return ETF_NAME_KR.get(str(code).strip(), "종목명 미확인")

def _fmt_etf_code_name(code: str) -> str:
    c = str(code).strip()
    if not c:
        return "-"
    return f"{c} {_etf_name_kr(c)}"

def _code_only(v: str) -> str:
    return str(v or "").strip().split()[0] if str(v or "").strip() else ""

def _sanitize_isa_trade_etf(code: str, default: str = "418660") -> str:
    c = _code_only(code)
    return c if c in ISA_WDR_TRADE_ETF_SET else str(default)

def _format_kis_holdings_df(holdings: list):
    """KIS 보유종목 리스트를 Streamlit dataframe용 DataFrame으로 변환."""
    import pandas as pd
    rows = []
    for h in holdings:
        rows.append({
            "종목코드": h.get("code", ""),
            "종목명": h.get("name", ""),
            "수량": int(h.get("qty", 0)),
            "현재가": f"{h.get('price', 0):,.0f}",
            "평가금액": f"{h.get('eval_amt', 0):,.0f}",
            "수익률(%)": f"{h.get('pnl_pct', 0):+.2f}",
        })
    return pd.DataFrame(rows)
