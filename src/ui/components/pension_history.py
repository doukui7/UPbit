"""연금저축 거래내역 조회 탭."""
import streamlit as st
import pandas as pd
import numpy as np

from src.utils.formatting import (
    _safe_float,
    _safe_int,
    _fmt_etf_code_name,
    _code_only,
)
from src.utils.kis import _compute_kis_balance_summary


def _pick_dep_num(data: dict, keys: list[str]) -> float:
    if not isinstance(data, dict):
        return 0.0
    for k in keys:
        if k in data:
            return float(_safe_float(data.get(k), 0.0))
    return 0.0


def _dep_core_sum(data: dict) -> float:
    if not isinstance(data, dict):
        return 0.0
    return float(
        _pick_dep_num(data, ["dnca_tota", "dnca_tot_amt"])
        + _pick_dep_num(data, ["nxdy_excc_amt"])
        + _pick_dep_num(data, ["nxdy_sttl_amt"])
        + _pick_dep_num(data, ["nx2_day_sttl_amt"])
    )


def _pick_first(row: dict, keys: list[str], default=""):
    for k in keys:
        if k in row:
            v = row.get(k)
            if v not in ("", None):
                return v
    return default


def _fmt_date(v) -> str:
    s = "".join(ch for ch in str(v or "") if ch.isdigit())
    if len(s) == 8:
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return str(v or "")


def _fmt_time(v) -> str:
    s = "".join(ch for ch in str(v or "") if ch.isdigit())
    if len(s) >= 6:
        s = s[:6]
        return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"
    return str(v or "")


def render_pension_history_tab(trader, kis_acct: str, kis_prdt: str,
                                pen_bal_key: str, pen_bal_ts_key: str):
    """거래내역 조회 탭 렌더링."""
    import time

    st.header("거래내역 조회")
    st.caption("연금저축 계좌의 주문/체결 내역을 조회합니다.")

    _hist_cache_key = f"pen_trade_history_cache_{kis_acct}_{kis_prdt}"
    _dep_cache_key = f"pen_deposit_history_cache_{kis_acct}_{kis_prdt}"

    _default_end = pd.Timestamp.now().date()
    _default_start = (pd.Timestamp.now() - pd.Timedelta(days=30)).date()

    h1, h2, h3, h4 = st.columns([1.2, 1.2, 1.0, 1.0])
    with h1:
        hist_start = st.date_input("조회 시작일", value=_default_start, key="pen_hist_start")
    with h2:
        hist_end = st.date_input("조회 종료일", value=_default_end, key="pen_hist_end")
    with h3:
        hist_side_label = st.selectbox("매매구분", ["전체", "매수", "매도"], key="pen_hist_side")
    with h4:
        hist_ccld_label = st.selectbox("체결구분", ["전체", "체결", "미체결"], key="pen_hist_ccld")

    h5, h6, h7 = st.columns([1.1, 1.1, 1.2])
    with h5:
        hist_stock_code = _code_only(st.text_input("종목코드(선택)", value="", key="pen_hist_code"))
    with h6:
        hist_order_no = st.text_input("주문번호(선택)", value="", key="pen_hist_order_no").strip()
    with h7:
        hist_max_rows = int(
            st.selectbox("최대 조회 건수", [50, 100, 200, 500], index=2, key="pen_hist_max_rows")
        )

    # ── 입출금/정산 내역 ──
    st.divider()
    st.subheader("입출금/정산 내역")
    st.caption("KIS 퇴직연금 예수금조회 기준으로 입출금 관련 금액과 정산 금액을 표시합니다.")

    def _fetch_best_pension_deposit() -> dict:
        _codes = ["00", "01", "02", "03", "04", "05", "99"]
        _best_res = None
        _best_sum = -1.0
        _trace = []
        for _cd in _codes:
            _res = trader.get_pension_deposit_info(acca_dvsn_cd=_cd)
            _ok = bool(_res.get("success")) if isinstance(_res, dict) else False
            _msg = str(_res.get("msg", "") or "") if isinstance(_res, dict) else ""
            _data = _res.get("data", {}) if isinstance(_res, dict) else {}
            _sum = _dep_core_sum(_data) if _ok else -1.0
            _trace.append(f"{_cd}:{'ok' if _ok else 'fail'}:{_sum:,.0f}")
            if _ok and _sum > _best_sum:
                _best_sum = _sum
                _best_res = {
                    "success": True,
                    "msg": _msg or "ok",
                    "data": _data if isinstance(_data, dict) else {},
                    "source": str(_res.get("source", "") or ""),
                    "acca_dvsn_cd": _cd,
                }
        if _best_res is None:
            return {
                "success": False,
                "msg": "입출금 조회 실패",
                "data": {},
                "all_zero": True,
                "trace": " / ".join(_trace),
            }
        _best_res["all_zero"] = bool(_best_sum <= 0.0)
        _best_res["trace"] = " / ".join(_trace)
        return _best_res

    if st.button("입출금 내역 조회", key="pen_dep_fetch_btn"):
        with st.spinner("입출금/정산 내역을 조회하는 중..."):
            _dep_result = _fetch_best_pension_deposit()
        st.session_state[_dep_cache_key] = _dep_result

    _dep_res = st.session_state.get(_dep_cache_key)
    if _dep_res:
        _dep_ok = bool(_dep_res.get("success")) if isinstance(_dep_res, dict) else False
        _dep_msg = str(_dep_res.get("msg", "") or "") if isinstance(_dep_res, dict) else ""
        _dep_data = _dep_res.get("data", {}) if isinstance(_dep_res, dict) else {}
        _dep_all_zero = bool(_dep_res.get("all_zero")) if isinstance(_dep_res, dict) else False
        _dep_trace = str(_dep_res.get("trace", "") or "") if isinstance(_dep_res, dict) else ""
        _dep_sel_cd = str(_dep_res.get("acca_dvsn_cd", "") or "") if isinstance(_dep_res, dict) else ""

        if not _dep_ok:
            st.error(f"입출금 내역 조회 실패: {_dep_msg or '응답 없음'}")
        else:
            _dep_rows = [
                {"항목": "예수금총액", "금액(원)": _pick_dep_num(_dep_data, ["dnca_tota", "dnca_tot_amt"])},
                {"항목": "익일정산액", "금액(원)": _pick_dep_num(_dep_data, ["nxdy_excc_amt"])},
                {"항목": "익일결제금액", "금액(원)": _pick_dep_num(_dep_data, ["nxdy_sttl_amt"])},
                {"항목": "2익일결제금액", "금액(원)": _pick_dep_num(_dep_data, ["nx2_day_sttl_amt"])},
            ]

            _in_amt = _pick_dep_num(_dep_data, ["in_amt", "dpst_amt", "depo_amt", "in_acmt_amt"])
            _out_amt = _pick_dep_num(_dep_data, ["out_amt", "wdrw_amt", "drwl_amt", "out_acmt_amt"])
            if _in_amt != 0:
                _dep_rows.append({"항목": "입금금액", "금액(원)": _in_amt})
            if _out_amt != 0:
                _dep_rows.append({"항목": "출금금액", "금액(원)": _out_amt})

            if _dep_all_zero:
                _bal_for_dep = st.session_state.get(pen_bal_key)
                if not isinstance(_bal_for_dep, dict) or bool(_bal_for_dep.get("error")):
                    _bal_for_dep = trader.get_balance()
                    if isinstance(_bal_for_dep, dict) and not bool(_bal_for_dep.get("error")):
                        st.session_state[pen_bal_key] = _bal_for_dep
                        st.session_state[pen_bal_ts_key] = float(time.time())
                _bal_sum_for_dep = _compute_kis_balance_summary(_bal_for_dep if isinstance(_bal_for_dep, dict) else {})
                _fallback_cash = float(
                    _bal_sum_for_dep.get("buyable_cash", 0.0) or _bal_sum_for_dep.get("cash", 0.0) or 0.0
                )
                if _fallback_cash > 0:
                    _dep_rows[0] = {"항목": "예수금총액(잔고기준)", "금액(원)": _fallback_cash}
                    st.info("입출금 API 응답이 0으로 내려와 잔고 조회 기준 예수금으로 보정 표시했습니다.")
                else:
                    st.warning("입출금 API 응답이 0으로 내려왔습니다. 계좌 유형/거래시간에 따라 0으로 반환될 수 있습니다.")

            _dep_df = pd.DataFrame(_dep_rows)
            st.dataframe(_dep_df, use_container_width=True, hide_index=True)
            if _dep_sel_cd:
                st.caption(f"입출금 조회 구분코드: {_dep_sel_cd}")
            if _dep_trace:
                st.caption(f"조회 시도: {_dep_trace}")
            with st.expander("입출금 원본 응답 보기"):
                st.dataframe(pd.DataFrame([_dep_data]), use_container_width=True, hide_index=True)

    # ── 주문/체결 내역 ──
    st.divider()
    st.subheader("주문/체결 내역")

    _side_map = {"전체": "00", "매도": "01", "매수": "02"}
    _ccld_map = {"전체": "00", "체결": "01", "미체결": "02"}

    if st.button("거래내역 조회", key="pen_hist_fetch_btn", type="primary"):
        _start_ymd = pd.Timestamp(hist_start).strftime("%Y%m%d")
        _end_ymd = pd.Timestamp(hist_end).strftime("%Y%m%d")
        with st.spinner("연금저축 거래내역을 조회하는 중..."):
            _hist_result = trader.get_pension_trade_history(
                start_date=_start_ymd,
                end_date=_end_ymd,
                side=_side_map.get(hist_side_label, "00"),
                ccld=_ccld_map.get(hist_ccld_label, "00"),
                stock_code=hist_stock_code,
                order_no=hist_order_no,
                max_rows=hist_max_rows,
            )
        st.session_state[_hist_cache_key] = {
            "start": _start_ymd,
            "end": _end_ymd,
            "result": _hist_result,
        }

    _hist_pack = st.session_state.get(_hist_cache_key)
    if not _hist_pack:
        st.info("조회 조건을 설정한 뒤 `거래내역 조회` 버튼을 눌러주세요.")
    else:
        _hist_res = _hist_pack.get("result", {}) if isinstance(_hist_pack, dict) else {}
        _hist_rows = _hist_res.get("rows", []) if isinstance(_hist_res, dict) else []
        _hist_ok = bool(_hist_res.get("success")) if isinstance(_hist_res, dict) else False
        _hist_msg = str(_hist_res.get("msg", "") or "") if isinstance(_hist_res, dict) else ""
        _hist_source = str(_hist_res.get("source", "") or "") if isinstance(_hist_res, dict) else ""
        _fallback_msg = str(_hist_res.get("fallback_msg", "") or "") if isinstance(_hist_res, dict) else ""

        if _hist_ok:
            st.caption(
                f"조회기간: {_hist_pack.get('start', '')} ~ {_hist_pack.get('end', '')}"
                + (f" | 소스: {_hist_source}" if _hist_source else "")
            )
            if _fallback_msg:
                st.caption(f"보조 조회 사용: {_fallback_msg}")
        else:
            st.error(f"거래내역 조회 실패: {_hist_msg or '응답 없음'}")

        if not _hist_rows:
            st.info("조회된 거래내역이 없습니다.")
        else:
            _view_rows = []
            for _r in _hist_rows:
                if not isinstance(_r, dict):
                    continue

                _code = str(_pick_first(_r, ["pdno", "mksc_shrn_iscd", "shtn_pdno"], "")).strip()
                _name = str(_pick_first(_r, ["prdt_name", "pd_name", "item_name", "prdt_abrv_name"], "")).strip()
                _side_cd = str(_pick_first(_r, ["sll_buy_dvsn_cd", "sll_buy_dvsn", "trde_dvsn_cd"], "")).strip()
                _side_nm = str(_pick_first(_r, ["sll_buy_dvsn_name", "sll_buy_dvsn_cd_name"], "")).strip()
                _status_cd = str(_pick_first(_r, ["ccld_dvsn", "ccld_nccs_dvsn"], "")).strip()

                _order_qty = _safe_int(_pick_first(_r, ["ord_qty", "tot_ord_qty", "order_qty"], 0), 0)
                _filled_qty = _safe_int(_pick_first(_r, ["tot_ccld_qty", "ccld_qty", "exec_qty"], 0), 0)
                _remain_qty = _safe_int(_pick_first(_r, ["rmn_qty", "nccs_qty", "ord_rmn_qty"], 0), 0)
                _order_px = _safe_float(_pick_first(_r, ["ord_unpr", "order_price"], 0.0), 0.0)
                _filled_px = _safe_float(_pick_first(_r, ["avg_prvs", "avg_ccld_unpr", "ccld_unpr"], 0.0), 0.0)
                _filled_amt = _safe_float(_pick_first(_r, ["tot_ccld_amt", "ccld_amt", "exec_amt"], 0.0), 0.0)

                _trade_date = _fmt_date(_pick_first(_r, ["ord_dt", "ord_dd", "trde_dt", "ccld_dt"], ""))
                _trade_time = _fmt_time(_pick_first(_r, ["ord_tmd", "ord_tm", "trde_tmd", "ccld_tmd"], ""))
                _trade_dt = f"{_trade_date} {_trade_time}".strip()

                if _side_cd in ("01", "1"):
                    _side = "매도"
                elif _side_cd in ("02", "2"):
                    _side = "매수"
                else:
                    _side = _side_nm or "-"

                if _filled_qty > 0 and _remain_qty > 0:
                    _status = "부분체결"
                elif _filled_qty > 0:
                    _status = "체결"
                elif _status_cd in ("02",):
                    _status = "미체결"
                elif _status_cd in ("01",):
                    _status = "체결"
                else:
                    _status = "미체결" if _remain_qty > 0 else "-"

                _view_rows.append(
                    {
                        "거래일시": _trade_dt,
                        "종목": _fmt_etf_code_name(_code) if _code else (_name or "-"),
                        "매매": _side,
                        "상태": _status,
                        "주문수량(주)": int(_order_qty),
                        "체결수량(주)": int(_filled_qty),
                        "미체결수량(주)": int(_remain_qty),
                        "주문가격(원)": float(_order_px),
                        "체결평균가(원)": float(_filled_px),
                        "체결금액(원)": float(_filled_amt),
                        "주문번호": str(_pick_first(_r, ["odno", "ord_no"], "")),
                    }
                )

            _hist_df = pd.DataFrame(_view_rows)
            if _hist_df.empty:
                st.info("표시할 거래내역이 없습니다.")
            else:
                _filled_notional = float(_hist_df["체결금액(원)"].sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("조회건수", f"{len(_hist_df):,}건")
                m2.metric("체결금액 합계", f"{_filled_notional:,.0f}원")
                m3.metric("체결건수", f"{int((_hist_df['상태'] != '미체결').sum()):,}건")

                st.dataframe(_hist_df, use_container_width=True, hide_index=True)

            with st.expander("원본 응답 보기"):
                st.dataframe(pd.DataFrame(_hist_rows), use_container_width=True, hide_index=True)
