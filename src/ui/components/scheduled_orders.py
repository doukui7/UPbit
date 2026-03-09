"""예약 주문 탭 — 예정 주문 조건 상세 + 과거 주문 내역 + 보충 설정."""
import json
import os
import subprocess
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.ui.coin_utils import get_signal_entry, load_balance_cache, load_signal_state

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_KST = timezone(timedelta(hours=9))
_SCHED_HOURS = [1, 5, 9, 13, 17, 21]


def _load_json(filename: str):
    fpath = os.path.join(_PROJECT_ROOT, filename)
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _safe_float(val, default=0.0):
    try:
        v = float(str(val).replace(",", ""))
        return default if pd.isna(v) else v
    except Exception:
        return default


def _normalize_interval(iv: str) -> str:
    for alias, real in [("4h", "minute240"), ("1h", "minute60"), ("1d", "day")]:
        if iv.lower() == alias:
            return real
    return iv


def _make_signal_key(item: dict) -> str:
    ticker = f"{item.get('market', 'KRW')}-{str(item.get('coin', '')).upper()}"
    strategy = item.get("strategy", "SMA")
    try:
        param = int(float(item.get("parameter", 20) or 20))
    except Exception:
        param = 20
    interval = _normalize_interval(item.get("interval", "day"))
    return f"{ticker}_{strategy}_{param}_{interval}"


def _iv_label(iv: str) -> str:
    m = {"minute240": "4H", "minute60": "1H", "day": "1D",
         "4h": "4H", "1h": "1H", "1d": "1D"}
    return m.get(iv, iv)


def _fmt_krw(v):
    """숫자를 쉼표 포함 원화 문자열로."""
    if v and v > 0:
        return f"{v:,.0f}"
    return "-"


# ═══════════════════════════════════════════
# 서브탭 1: 예정 주문 (테이블 형식)
# ═══════════════════════════════════════════
def _render_upcoming_orders(portfolio_list, initial_cap, config):
    now_kst = datetime.now(_KST)

    # ── 스케줄 슬롯 생성 ──
    future_slots = []
    for day_off in range(2):
        for h in _SCHED_HOURS:
            cand = (now_kst + timedelta(days=day_off)).replace(
                hour=h, minute=0, second=0, microsecond=0
            )
            if now_kst < cand <= now_kst + timedelta(hours=24):
                future_slots.append(cand)
    future_slots.sort()
    if not future_slots:
        future_slots.append((now_kst + timedelta(days=1)).replace(hour=1, minute=0, second=0, microsecond=0))

    # ── 데이터 로드 (GitHub 자동 동기화 포함) ──
    sig_state = load_signal_state() or {}
    bc = load_balance_cache() or {}
    bals = bc.get("balances", {})
    prices = bc.get("prices", {})
    bc_time = bc.get("updated_at", "N/A")

    tu_on = config.get("topup_enabled", False)
    tu_buy = config.get("topup_buy_amount", 5000)
    tu_sell = config.get("topup_sell_amount", 5000)

    # ── 코인별 weight 합산 ──
    coin_wt_sum = {}
    for pi in portfolio_list:
        sym = pi["coin"].upper()
        coin_wt_sum[sym] = coin_wt_sum.get(sym, 0) + float(pi.get("weight", 0))

    # ── 총 포트폴리오 가치 ──
    krw_bal = _safe_float(bals.get("KRW", 0))
    total_coin_v = 0
    seen = set()
    for pi in portfolio_list:
        sym = pi["coin"].upper()
        if sym not in seen:
            coin_b = _safe_float(bals.get(sym, 0))
            tk = f"{pi.get('market', 'KRW')}-{sym}"
            coin_p = _safe_float(prices.get(tk, 0))
            total_coin_v += coin_b * coin_p
            seen.add(sym)
    total_pv = krw_bal + total_coin_v

    # ── 다음 실행 표시 ──
    next_dt = future_slots[0]
    remain = next_dt - now_kst
    hh = int(remain.total_seconds() // 3600)
    mm = int((remain.total_seconds() % 3600) // 60)
    trading_mode = config.get("trading_mode", "real")
    mode_label = "📡 Signal" if trading_mode == "signal" else "🔴 Real"
    sig_updated = ""
    # signal_state에서 최신 갱신 시각 표시
    for _v in sig_state.values():
        if isinstance(_v, dict) and _v.get("updated_at"):
            sig_updated = _v["updated_at"]
            break
    st.info(
        f"다음 실행: **{next_dt.strftime('%Y-%m-%d %H:%M KST')}** ({hh}시간 {mm}분 후) "
        f"| 모드: **{mode_label}** | 시그널: {sig_updated or 'N/A'} | 캐시: {bc_time}"
    )

    # ── 테이블 데이터 생성 (VM signal_state 기반) ──
    rows = []
    for slot in future_slots:
        sh = slot.hour
        slot_label = slot.strftime("%m/%d %H:%M")

        for pi in portfolio_list:
            sym = pi["coin"].upper()
            tk = f"{pi.get('market', 'KRW')}-{sym}"
            strat_name = pi.get("strategy", "SMA")
            param = int(pi.get("parameter", 20) or 20)
            iv = pi.get("interval", "day")
            iv_norm = _normalize_interval(iv)
            wt = float(pi.get("weight", 0))
            skey = _make_signal_key(pi)

            close_now = _safe_float(prices.get(tk, 0))
            coin_b = _safe_float(bals.get(sym, 0))
            cw_total = coin_wt_sum.get(sym, wt)
            sell_ratio = wt / cw_total if cw_total > 0 else 1
            my_qty = coin_b * sell_ratio
            my_val = my_qty * close_now

            # VM이 계산한 signal_state에서 목표가/상태 읽기
            entry = get_signal_entry(sig_state, skey)
            state = entry.get("state", "").upper()
            buy_target = _safe_float(entry.get("buy_target", 0))
            sell_target = _safe_float(entry.get("sell_target", 0))

            # Donchian HOLD → 보유 여부로 판단
            if state == "HOLD":
                state = "BUY" if my_val >= 5000 else "SELL"
            elif state not in ("BUY", "SELL"):
                state = "미확인"

            is_signal_skip = (iv_norm == "day" and sh != 9)
            strat_label = f"{strat_name}({param}, {_iv_label(iv)})"

            # 조건가 / 이격도 / 예상 주문 결정
            cond_price = "-"
            gap_str = "-"
            expected = ""
            note = ""

            _skip_tag = " [09시]" if is_signal_skip else ""

            if close_now <= 0:
                expected = "가격 조회 실패"
            elif state == "BUY":
                # 보유 중 → 매도 조건 분석
                if sell_target > 0:
                    cond_price = _fmt_krw(sell_target)
                    dist = (close_now - sell_target) / sell_target * 100
                    gap_str = f"{dist:+.1f}%"
                    if close_now < sell_target:
                        expected = f"SELL ({my_qty:.8g}개, {_fmt_krw(my_val)}원){_skip_tag}"
                    else:
                        expected = "HOLD"
                else:
                    expected = "HOLD (미산출)"

                # 보충 매수
                if tu_on and wt > 0:
                    target_v = total_pv * (wt / 100)
                    need = target_v - my_val
                    if need >= 5000:
                        buy_amt = min(tu_buy, need)
                        note = f"보충매수 {_fmt_krw(buy_amt)}원"

            elif state == "SELL":
                # 현금 → 매수 조건 분석
                if buy_target > 0:
                    cond_price = _fmt_krw(buy_target)
                    dist = (close_now - buy_target) / buy_target * 100
                    gap_str = f"{dist:+.1f}%"
                    if close_now > buy_target:
                        target_v = total_pv * (wt / 100)
                        expected = f"BUY (~{_fmt_krw(target_v)}원){_skip_tag}"
                    else:
                        expected = "HOLD"
                else:
                    expected = "HOLD (미산출)"

                # 보충 매도
                if tu_on and my_val >= 5000:
                    sell_amt = min(tu_sell, my_val)
                    note = f"보충매도 {_fmt_krw(sell_amt)}원"
            else:
                expected = "미확인"

            rows.append({
                "실행시간": slot_label,
                "종목": tk,
                "전략": strat_label,
                "상태": state,
                "현재가": _fmt_krw(close_now),
                "조건가": cond_price,
                "이격도": gap_str,
                "예상주문": expected,
                "비고": note or "-",
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("표시할 예정 주문이 없습니다.")

    st.caption(
        f"KRW {krw_bal:,.0f}원 | 코인 {total_coin_v:,.0f}원 | "
        f"총 {total_pv:,.0f}원"
    )
    st.caption("※ VM이 마지막 실행 시 계산한 시그널 기준 예상입니다.")


# ═══════════════════════════════════════════
# 서브탭 2: 과거 주문 내역 (테이블)
# ═══════════════════════════════════════════
def _render_past_orders():
    st.subheader("과거 주문 내역")
    st.caption("매 실행 주기마다 전략별 판단(BUY/SELL/HOLD/SKIP) 및 실행 결과를 기록합니다.")

    tl = _load_json("trade_log.json")
    if not isinstance(tl, list) or not tl:
        st.info("매매 로그 없음 (trade_log.json)")
        return

    # 필터 UI
    side_map = {
        "BUY": "매수", "SELL": "매도", "HOLD": "보유유지",
        "SKIP": "주기스킵", "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도",
    }
    all_labels = list(side_map.values())
    fc1, fc2 = st.columns([3, 1])
    sides = fc1.multiselect("구분 필터", all_labels, default=all_labels, key="past_side_filter")
    max_rows = fc2.number_input("표시 건수", min_value=10, max_value=500, value=50, step=10, key="past_max_rows")

    reverse_map = {v: k for k, v in side_map.items()}
    selected_keys = {reverse_map.get(s, s) for s in sides}

    filtered = [e for e in tl if e.get("side", "") in selected_keys][:max_rows]

    if not filtered:
        st.info("필터 조건에 맞는 기록이 없습니다.")
        return

    # 테이블 생성
    rows = []
    for e in filtered:
        side = e.get("side", "")
        result = e.get("result", "")
        res_icon = "OK" if result == "success" else ("FAIL" if result == "error" else "-")
        cur_price = _safe_float(e.get("current_price", 0))
        sell_tgt = _safe_float(e.get("sell_target", 0))
        buy_tgt = _safe_float(e.get("buy_target", 0))
        mode_raw = e.get("mode", "")
        mode_label = {"signal": "Signal", "auto": "Real", "manual": "수동"}.get(mode_raw, mode_raw)
        rows.append({
            "시간": e.get("time", ""),
            "모드": mode_label,
            "종목": e.get("ticker", ""),
            "판단": side_map.get(side, side),
            "현재가": _fmt_krw(cur_price),
            "매도가": _fmt_krw(sell_tgt),
            "매수가": _fmt_krw(buy_tgt),
            "이격도": e.get("sell_gap", "") or e.get("buy_gap", "") or "-",
            "사유": e.get("reason", "") or "-",
            "금액": e.get("amount", "") or e.get("qty", "") or "-",
            "결과": res_icon,
            "전략": e.get("strategy", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"총 {len(tl)}건 중 {len(filtered)}건 표시")


# ═══════════════════════════════════════════
# 서브탭 3: 보충 매수/매도 설정
# ═══════════════════════════════════════════
def _render_topup_settings(config, save_config):
    st.subheader("보충 매수/매도 설정")
    st.caption("BUY 시그널 유지 중 목표 미달 → 매 실행마다 설정 금액 추가 매수 / SELL 시그널 유지 중 잔량 보유 → 설정 금액 추가 매도")

    topup_enabled = config.get("topup_enabled", False)
    topup_buy_amt = config.get("topup_buy_amount", 5000)
    topup_sell_amt = config.get("topup_sell_amount", 5000)

    tu_on = st.checkbox("보충 매수/매도 활성화", value=topup_enabled, key="sched_topup_toggle")

    if tu_on:
        tc1, tc2 = st.columns(2)
        tu_buy = tc1.number_input(
            "보충 매수 금액 (KRW/회)", min_value=5000, max_value=1000000,
            value=max(5000, topup_buy_amt), step=5000, key="sched_topup_buy",
        )
        tu_sell = tc2.number_input(
            "보충 매도 금액 (KRW/회)", min_value=5000, max_value=1000000,
            value=max(5000, topup_sell_amt), step=5000, key="sched_topup_sell",
        )
    else:
        tu_buy = topup_buy_amt
        tu_sell = topup_sell_amt

    # 변경 감지 후 저장
    if (tu_on != topup_enabled
        or (tu_on and (tu_buy != topup_buy_amt or tu_sell != topup_sell_amt))):
        new_cfg = config.copy()
        new_cfg["topup_enabled"] = tu_on
        new_cfg["topup_buy_amount"] = tu_buy
        new_cfg["topup_sell_amount"] = tu_sell
        save_config(new_cfg)
        try:
            subprocess.run(["git", "add", "user_config.json"],
                           cwd=_PROJECT_ROOT, capture_output=True, timeout=10)
            subprocess.run(["git", "commit", "-m", "auto: 보충 매수/매도 설정 변경"],
                           cwd=_PROJECT_ROOT, capture_output=True, timeout=10)
            subprocess.run(["git", "push"],
                           cwd=_PROJECT_ROOT, capture_output=True, timeout=30)
            st.success("보충 설정 저장 + Git 푸시 완료")
        except Exception as e:
            st.warning(f"설정 저장됨 (Git 푸시 실패: {e})")
        st.rerun()


# ═══════════════════════════════════════════
# 메인 렌더 함수
# ═══════════════════════════════════════════
def render_scheduled_orders_tab(portfolio_list, initial_cap, config, save_config):
    """예약 주문 탭 메인 렌더러."""
    st.header("예약 주문")

    sub1, sub2, sub3 = st.tabs(["📋 예정 주문", "📜 과거 내역", "⚙️ 보충 설정"])

    with sub1:
        if not portfolio_list:
            st.warning("포트폴리오를 먼저 설정해주세요.")
        else:
            _render_upcoming_orders(portfolio_list, initial_cap, config)

    with sub2:
        _render_past_orders()

    with sub3:
        _render_topup_settings(config, save_config)
