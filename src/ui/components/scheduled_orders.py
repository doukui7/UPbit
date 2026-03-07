"""예약 주문 탭 — 예정 주문 조건 상세 + 과거 주문 내역 + 보충 설정."""
import json
import os
import subprocess
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone

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


# ═══════════════════════════════════════════
# 서브탭 1: 예정 주문 (향후 24시간 상세 조건)
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

    # ── 데이터 로드 ──
    sig_state = _load_json("signal_state.json") or {}
    bc = _load_json("balance_cache.json") or {}
    bals = bc.get("balances", {})
    prices = bc.get("prices", {})
    bc_time = bc.get("updated_at", "N/A")

    tu_on = config.get("topup_enabled", False)
    tu_buy = config.get("topup_buy_amount", 5000)
    tu_sell = config.get("topup_sell_amount", 5000)

    # ── 전략별 목표가 계산 ──
    # SMA/Donchian feature 계산
    strategy_targets = {}
    try:
        from src.strategy.sma import SMAStrategy
        from src.strategy.donchian import DonchianStrategy
        import src.engine.data_cache as data_cache

        for pi in portfolio_list:
            tk = f"{pi.get('market', 'KRW')}-{pi['coin'].upper()}"
            strat_name = pi.get("strategy", "SMA").lower()
            param = int(pi.get("parameter", 20) or 20)
            sell_param = int(pi.get("sell_parameter", 0) or 0) or param
            iv = _normalize_interval(pi.get("interval", "day"))
            skey = _make_signal_key(pi)

            try:
                df = data_cache.get_ohlcv_local_first(tk, iv, count=max(200, param + 20))
                if df is None or len(df) < 2:
                    continue

                if strat_name.startswith("donchian"):
                    strat = DonchianStrategy()
                    df_feat = strat.create_features(df, buy_period=param, sell_period=sell_param)
                    if df_feat is None or len(df_feat) < 2:
                        continue
                    eval_c = df_feat.iloc[-2]
                    upper_key = f"Donchian_Upper_{param}"
                    lower_key = f"Donchian_Lower_{sell_param}"
                    strategy_targets[skey] = {
                        "buy_target": _safe_float(eval_c.get(upper_key)),
                        "sell_target": _safe_float(eval_c.get(lower_key)),
                        "type": "donchian",
                        "buy_label": f"상단밴드({param})",
                        "sell_label": f"하단밴드({sell_param})",
                    }
                else:
                    strat = SMAStrategy()
                    df_feat = strat.create_features(df, periods=[param])
                    if df_feat is None or len(df_feat) < 2:
                        continue
                    eval_c = df_feat.iloc[-2]
                    sma_key = f"SMA_{param}"
                    sma_val = _safe_float(eval_c.get(sma_key))
                    strategy_targets[skey] = {
                        "buy_target": sma_val,
                        "sell_target": sma_val,
                        "type": "sma",
                        "buy_label": f"SMA({param})",
                        "sell_label": f"SMA({param})",
                    }
            except Exception:
                pass
    except ImportError:
        pass

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
    st.info(f"다음 실행: **{next_dt.strftime('%Y-%m-%d %H:%M KST')}** ({hh}시간 {mm}분 후)")

    # ── 각 슬롯별 상세 조건 표시 ──
    for slot in future_slots:
        sh = slot.hour
        slot_remain = slot - now_kst
        s_hh = int(slot_remain.total_seconds() // 3600)
        s_mm = int((slot_remain.total_seconds() % 3600) // 60)
        is_next = (slot == next_dt)

        header = f"{'▶ ' if is_next else ''}{slot.strftime('%m월 %d일 %H:%M')} KST"
        if is_next:
            header += f" ({s_hh}시간 {s_mm}분 후)"

        with st.expander(header, expanded=is_next):
            for pi in portfolio_list:
                sym = pi["coin"].upper()
                tk = f"{pi.get('market', 'KRW')}-{sym}"
                strat_name = pi.get("strategy", "SMA")
                param = int(pi.get("parameter", 20) or 20)
                iv = pi.get("interval", "day")
                iv_norm = _normalize_interval(iv)
                wt = float(pi.get("weight", 0))
                skey = _make_signal_key(pi)

                state = sig_state.get(skey, "?")
                if state == "HOLD":
                    state = "BUY"
                elif state == "?":
                    state = "미확인"

                close_now = _safe_float(prices.get(tk, 0))
                coin_b = _safe_float(bals.get(sym, 0))
                coin_v = coin_b * close_now

                targets = strategy_targets.get(skey, {})
                buy_target = targets.get("buy_target", 0)
                sell_target = targets.get("sell_target", 0)
                buy_label = targets.get("buy_label", "목표가")
                sell_label = targets.get("sell_label", "목표가")

                # 1D 전략은 09시에만 시그널 분석, 보충 주문은 매 슬롯 실행
                is_signal_skip = (iv_norm == "day" and sh != 9)
                skip_note = " — 시그널 분석은 09시만" if is_signal_skip else ""

                # ── 비중 / 보유 계산 ──
                cw_total = coin_wt_sum.get(sym, wt)
                sell_ratio = wt / cw_total if cw_total > 0 else 1
                my_qty = coin_b * sell_ratio
                my_val = my_qty * close_now

                st.markdown(
                    f"**[{strat_name} {param} / {_iv_label(iv)}]** "
                    f"현재: {state} ({'보유' if state == 'BUY' else '현금'}) | "
                    f"비중 {wt:.0f}%{skip_note}"
                )

                # ── 현재가 + 조건 + 예상 주문 ──
                if close_now > 0:
                    st.markdown(f"  현재가: **{close_now:,.0f}원**")

                    if state == "BUY":
                        # 매도 조건 분석
                        if sell_target > 0:
                            dist = (close_now - sell_target) / sell_target * 100
                            met = close_now < sell_target
                            st.markdown(f"  매도 조건가: {sell_label} = **{sell_target:,.0f}원** | 이격도 **{dist:+.1f}%**")
                            if is_signal_skip:
                                expected = "HOLD (1D → 09시만 시그널 분석)"
                            elif met:
                                expected = f"**SELL** → {my_qty:.8g}개 ({my_val:,.0f}원) 매도"
                            else:
                                expected = "HOLD (조건 미충족)"
                        else:
                            expected = "HOLD (목표가 미산출)" if not is_signal_skip else "HOLD (1D → 09시만)"
                        st.markdown(f"  예상 주문: {expected}")

                        # 보유 정보
                        if coin_b > 0:
                            st.caption(f"  보유: {sym} {my_qty:.8g}개 ({my_val:,.0f}원) / 전체 {coin_b:.8g}개 ({coin_v:,.0f}원)")

                        # 보충 매수
                        if tu_on and wt > 0:
                            target_v = total_pv * (wt / 100)
                            need = target_v - my_val
                            if need >= 5000:
                                buy_amt = min(tu_buy, need)
                                st.caption(
                                    f"  보충매수: 목표 {target_v:,.0f}원 - 보유 {my_val:,.0f}원 = "
                                    f"부족 {need:,.0f}원 → **{buy_amt:,.0f}원 매수 예정**"
                                )

                    elif state == "SELL":
                        # 매수 조건 분석
                        if buy_target > 0:
                            dist = (close_now - buy_target) / buy_target * 100
                            met = close_now > buy_target
                            st.markdown(f"  매수 조건가: {buy_label} = **{buy_target:,.0f}원** | 이격도 **{dist:+.1f}%**")
                            if is_signal_skip:
                                expected = "HOLD (1D → 09시만 시그널 분석)"
                            elif met:
                                target_v = total_pv * (wt / 100)
                                est_qty = target_v / close_now if close_now > 0 else 0
                                expected = f"**BUY** → ~{est_qty:.8g}개 ({target_v:,.0f}원) 매수"
                            else:
                                expected = "HOLD (조건 미충족)"
                        else:
                            expected = "HOLD (목표가 미산출)" if not is_signal_skip else "HOLD (1D → 09시만)"
                        st.markdown(f"  예상 주문: {expected}")

                        # 보충 매도
                        if tu_on and coin_v >= 5000:
                            sell_val_tu = my_val if my_val >= 5000 else coin_v
                            sell_amt = min(tu_sell, sell_val_tu)
                            st.caption(f"  보충매도: 잔량 {sell_val_tu:,.0f}원 → **{sell_amt:,.0f}원 매도 예정**")

                    else:
                        st.markdown(f"  예상 주문: 시그널 미확인 — 전략 분석 후 결정")
                else:
                    st.markdown(f"  현재가: 조회 실패")

    st.caption(
        f"잔고 KRW {krw_bal:,.0f}원 | 코인 {total_coin_v:,.0f}원 | "
        f"총 {total_pv:,.0f}원 (캐시: {bc_time})"
    )
    st.caption("※ 예상 동작은 현재 시그널 상태 기준이며, 실제 전략 분석 결과에 따라 변경될 수 있습니다.")


# ═══════════════════════════════════════════
# 서브탭 2: 과거 주문 내역
# ═══════════════════════════════════════════
def _render_past_orders():
    st.subheader("과거 주문 내역")
    st.caption("매 실행 주기마다 전략별 판단(BUY/SELL/HOLD/SKIP) 및 실행 결과를 기록합니다.")

    tl = _load_json("trade_log.json")
    if not isinstance(tl, list) or not tl:
        st.info("매매 로그 없음 (trade_log.json)")
        return

    # 필터
    side_map = {
        "BUY": "매수", "SELL": "매도", "HOLD": "보유유지",
        "SKIP": "주기스킵", "BUY_TOPUP": "보충매수", "SELL_TOPUP": "보충매도",
    }
    side_icons = {
        "BUY": "🟢", "SELL": "🔴", "HOLD": "⚪",
        "SKIP": "⏭️", "BUY_TOPUP": "🟡", "SELL_TOPUP": "🟠",
    }
    all_labels = list(side_map.values())
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    sides = fc1.multiselect("구분 필터", all_labels, default=all_labels, key="past_side_filter")
    max_rows = fc2.number_input("표시 건수", min_value=10, max_value=500, value=50, step=10, key="past_max_rows")
    view_mode = fc3.radio("보기", ["테이블", "상세"], key="past_view_mode", horizontal=True)

    reverse_map = {v: k for k, v in side_map.items()}
    selected_keys = {reverse_map.get(s, s) for s in sides}

    filtered = [e for e in tl if e.get("side", "") in selected_keys][:max_rows]

    if not filtered:
        st.info("필터 조건에 맞는 기록이 없습니다.")
        return

    if view_mode == "상세":
        # 상세 보기: 각 항목을 카드 형태로 표시
        for e in filtered:
            side = e.get("side", "")
            icon = side_icons.get(side, "❓")
            label = side_map.get(side, side)
            time_str = e.get("time", "")
            ticker = e.get("ticker", "")
            strategy = e.get("strategy", "")
            reason = e.get("reason", "")
            condition = e.get("condition", "")
            cur_price = e.get("current_price", 0)
            buy_tgt = e.get("buy_target", 0)
            sell_tgt = e.get("sell_target", 0)
            buy_gap = e.get("buy_gap", "")
            sell_gap = e.get("sell_gap", "")
            result = e.get("result", "")
            amount = e.get("amount", "") or e.get("qty", "")

            header = f"{icon} **{label}** | {ticker} | {strategy} | {time_str}"
            if result:
                res_icon = "✅" if result == "success" else ("❌" if result == "error" else "")
                header += f" | {res_icon}"

            with st.container():
                st.markdown(header)
                lines = []
                if cur_price:
                    lines.append(f"현재가: **{_safe_float(cur_price):,.0f}원**")
                if sell_tgt and _safe_float(sell_tgt) > 0:
                    lines.append(f"매도가: {_safe_float(sell_tgt):,.0f}원 (이격 {sell_gap})")
                if buy_tgt and _safe_float(buy_tgt) > 0:
                    lines.append(f"매수가: {_safe_float(buy_tgt):,.0f}원 (이격 {buy_gap})")
                if reason:
                    lines.append(f"사유: {reason}")
                if amount:
                    lines.append(f"금액: {amount}")
                if condition and not reason:
                    lines.append(f"조건: {condition}")
                if lines:
                    st.caption("  |  ".join(lines))
                st.markdown("---")
    else:
        # 테이블 보기
        rows = []
        for e in filtered:
            side = e.get("side", "")
            icon = side_icons.get(side, "")
            result = e.get("result", "")
            res_icon = "✅" if result == "success" else ("❌" if result == "error" else "-")
            cur_price = _safe_float(e.get("current_price", 0))
            sell_tgt = _safe_float(e.get("sell_target", 0))
            buy_tgt = _safe_float(e.get("buy_target", 0))
            rows.append({
                "시간": e.get("time", ""),
                "종목": e.get("ticker", ""),
                "판단": f"{icon} {side_map.get(side, side)}",
                "현재가": f"{cur_price:,.0f}" if cur_price else "-",
                "매도가": f"{sell_tgt:,.0f}" if sell_tgt else "-",
                "매수가": f"{buy_tgt:,.0f}" if buy_tgt else "-",
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
