#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
위대리V1.0 ◈K-2X 레버지리◈ codex CSV vs WDRStrategy 로직 비교 검증
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from strategy.widaeri import WDRStrategy

# ── CSV 파싱 ──
csv_path = r"위대리V1.0 ◈K-2X 레버지리◈ codex.csv"
raw = pd.read_csv(csv_path, header=None, encoding='utf-8-sig')

# 헤더 찾기 (Date 열)
header_row = None
date_col = None
for idx, row in raw.iterrows():
    for col_idx in range(len(raw.columns)):
        val = str(row.iloc[col_idx]) if pd.notna(row.iloc[col_idx]) else ""
        if val.strip() == "Date":
            header_row = idx
            date_col = col_idx
            break
    if header_row is not None:
        break

if header_row is None:
    print("ERROR: 헤더를 찾을 수 없습니다.")
    sys.exit(1)

print(f"헤더 행: {header_row}, Date 열: {date_col}")

def parse_won(val):
    if pd.isna(val):
        return 0.0
    s = str(val).replace("₩", "").replace("\u20a9", "").replace(",", "").replace(" ", "").strip()
    if s == "" or s == "nan":
        return 0.0
    s = s.replace("−", "-").replace("–", "-")
    try:
        return float(s)
    except:
        return 0.0

def parse_pct(val):
    if pd.isna(val):
        return 0.0
    s = str(val).replace("%", "").replace(",", "").strip()
    if s == "" or s == "nan":
        return 0.0
    s = s.replace("−", "-").replace("–", "-")
    try:
        return float(s)
    except:
        return 0.0

# 데이터 파싱
data_rows = []
for idx in range(header_row + 1, len(raw)):
    row = raw.iloc[idx]
    date_val = str(row.iloc[date_col]).strip() if pd.notna(row.iloc[date_col]) else ""
    if not date_val or date_val == "nan":
        continue

    c = date_col
    price = parse_won(row.iloc[c+1])
    change_pct = parse_pct(row.iloc[c+2])
    divergence = parse_pct(row.iloc[c+3])
    target_amount = parse_won(row.iloc[c+5])
    target_qty = parse_won(row.iloc[c+6])
    actual_qty = parse_won(row.iloc[c+7])
    actual_amount = parse_won(row.iloc[c+8])
    total_shares = parse_won(row.iloc[c+9])
    total_eval = parse_won(row.iloc[c+10])
    new_eval = parse_won(row.iloc[c+11])
    cash = parse_won(row.iloc[c+12])
    total_assets = parse_won(row.iloc[c+13])
    cash_ratio = parse_pct(row.iloc[c+14])
    dd = parse_pct(row.iloc[c+15])

    if price <= 0:
        continue

    data_rows.append({
        "date": date_val,
        "price": price,
        "change_pct": change_pct,
        "divergence": divergence,
        "csv_target_amount": target_amount,
        "csv_target_qty": int(target_qty),
        "csv_actual_qty": int(actual_qty),
        "csv_actual_amount": actual_amount,
        "csv_shares": int(total_shares),
        "csv_total_eval": total_eval,
        "csv_new_eval": new_eval,
        "csv_cash": cash,
        "csv_total_assets": total_assets,
        "csv_cash_ratio": cash_ratio,
        "csv_dd": dd,
    })

df = pd.DataFrame(data_rows)
print(f"파싱된 주차: {len(df)}개")
print(f"기간: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")

# ═══════════════════════════════════════════════════════════
# CSV 설정값 확인
# ═══════════════════════════════════════════════════════════
INITIAL_CAPITAL = 20_000_000
INITIAL_CASH_RATIO = 0.16
OV_THRESHOLD = 5.0
UN_THRESHOLD = -6.0
COMMISSION = 0.0  # CSV는 수수료 없음

# 3단계 매매비율
SELL_OV = 100.0
SELL_NEU = 66.7
SELL_UN = 60.0
BUY_OV = 66.7
BUY_NEU = 66.7
BUY_UN = 120.0
MIN_CASH = 0.10
MIN_STOCK = 0.10

print(f"\n{'='*80}")
print(f"CSV 설정값 (3단계)")
print(f"{'='*80}")
print(f"초기자본: ₩{INITIAL_CAPITAL:,}  |  초기현금비율: {INITIAL_CASH_RATIO*100}%")
print(f"고평가 > {OV_THRESHOLD}%: 매도 {SELL_OV}% / 매수 {BUY_OV}%")
print(f"중립: 매도 {SELL_NEU}% / 매수 {BUY_NEU}%")
print(f"저평가 < {UN_THRESHOLD}%: 매도 {SELL_UN}% / 매수 {BUY_UN}%")
print(f"최소 현금/주식 비율: {MIN_CASH*100}% / {MIN_STOCK*100}%")

# ═══════════════════════════════════════════════════════════
# 방법 1: 수동 시뮬레이션 (CSV 이격도 사용)
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"[방법 1] 수동 시뮬레이션 vs CSV 비교")
print(f"{'='*80}")

first_price = df.iloc[0]["price"]
buy_ratio = 1.0 - INITIAL_CASH_RATIO
shares = int((INITIAL_CAPITAL * buy_ratio) / first_price)
cash = INITIAL_CAPITAL - shares * first_price

print(f"초기: 가격={first_price:,.0f}, 주식={shares}주, 현금={cash:,.0f}")
print(f"CSV:  주식={df.iloc[0]['csv_shares']}주, 현금={df.iloc[0]['csv_cash']:,.0f}")

mismatches = []
match_count = 0
peak_assets = cash + shares * first_price

# 첫 주 비교
if shares != df.iloc[0]["csv_shares"]:
    mismatches.append(f"W0 초기주식: SIM={shares} CSV={df.iloc[0]['csv_shares']}")
if abs(cash - df.iloc[0]["csv_cash"]) > 1:
    mismatches.append(f"W0 초기현금: SIM={cash:,.0f} CSV={df.iloc[0]['csv_cash']:,.0f}")

for i in range(1, len(df)):
    row = df.iloc[i]
    cur_price = row["price"]
    prev_price = df.iloc[i-1]["price"]
    divergence = row["divergence"]

    weekly_pnl = (cur_price - prev_price) * shares
    total_value = cash + shares * cur_price

    # 3단계 시장 판단
    if divergence > OV_THRESHOLD:
        sell_r = SELL_OV
        buy_r = BUY_OV
    elif divergence < UN_THRESHOLD:
        sell_r = SELL_UN
        buy_r = BUY_UN
    else:
        sell_r = SELL_NEU
        buy_r = BUY_NEU

    action = None
    qty = 0

    if weekly_pnl > 0 and shares > 0:
        sell_amount = weekly_pnl * (sell_r / 100.0)
        qty = int(sell_amount / cur_price)
        if qty > 0:
            new_shares = shares - qty
            new_stock_value = new_shares * cur_price
            if total_value > 0 and new_stock_value / total_value < MIN_STOCK:
                max_sell = shares - max(1, int(total_value * MIN_STOCK / cur_price))
                qty = max(0, min(qty, max_sell))
        if qty > shares:
            qty = shares
        if qty > 0:
            action = "SELL"

    elif weekly_pnl < 0:
        buy_amount = abs(weekly_pnl) * (buy_r / 100.0)
        max_buy_cash = cash - max(0, total_value * MIN_CASH)
        buy_amount = min(buy_amount, max(0, max_buy_cash))
        qty = int(buy_amount / cur_price)
        if qty > 0 and (qty * cur_price) <= cash:
            action = "BUY"
        else:
            qty = 0

    if action == "SELL" and qty > 0:
        cash += qty * cur_price
        shares -= qty
    elif action == "BUY" and qty > 0:
        if qty * cur_price <= cash:
            cash -= qty * cur_price
            shares += qty

    # CSV 비교
    csv_shares = row["csv_shares"]
    csv_cash = row["csv_cash"]
    csv_total = row["csv_total_assets"]
    sim_total = cash + shares * cur_price

    shares_ok = (shares == csv_shares)
    cash_ok = abs(cash - csv_cash) <= 1
    total_ok = abs(sim_total - csv_total) <= 10

    if shares_ok and cash_ok:
        match_count += 1
    else:
        detail = f"W{i:3d} {row['date']:>15s} | div={divergence:>7.2f}% | pnl={weekly_pnl:>12,.0f}"
        if not shares_ok:
            detail += f" | 주식: SIM={shares} CSV={csv_shares} (차이={shares-csv_shares})"
        if not cash_ok:
            detail += f" | 현금: SIM={cash:>12,.0f} CSV={csv_cash:>12,.0f} (차이={cash-csv_cash:>8,.0f})"
        mismatches.append(detail)

    cur_total = cash + shares * cur_price
    if cur_total > peak_assets:
        peak_assets = cur_total

total_weeks = len(df) - 1
print(f"\n총 {total_weeks}주 중 {match_count}주 일치 ({match_count/max(1,total_weeks)*100:.1f}%)")
print(f"불일치: {len(mismatches)}건")

if mismatches:
    print(f"\n불일치 상세 (처음 20건):")
    print("-"*120)
    for m in mismatches[:20]:
        print(m)
    if len(mismatches) > 20:
        print(f"... 외 {len(mismatches)-20}건 더")

csv_final = df.iloc[-1]
sim_total = cash + shares * csv_final["price"]
csv_total = csv_final["csv_total_assets"]
print(f"\n최종 결과:")
print(f"  SIM: 주식={shares}주, 현금=₩{cash:,.0f}, 총자산=₩{sim_total:,.0f}")
print(f"  CSV: 주식={csv_final['csv_shares']}주, 현금=₩{csv_final['csv_cash']:,.0f}, 총자산=₩{csv_total:,.0f}")
print(f"  차이: 주식={shares-csv_final['csv_shares']}주, 총자산=₩{sim_total-csv_total:,.0f}")

# ═══════════════════════════════════════════════════════════
# 방법 2: WDRStrategy.get_rebalance_action() 사용
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"[방법 2] WDRStrategy.get_rebalance_action() vs CSV 비교")
print(f"{'='*80}")

strategy = WDRStrategy(settings={
    'overvalue_threshold': OV_THRESHOLD,
    'undervalue_threshold': UN_THRESHOLD,
}, evaluation_mode=3)

# 초기
s_shares = int((INITIAL_CAPITAL * (1.0 - INITIAL_CASH_RATIO)) / first_price)
s_cash = INITIAL_CAPITAL - s_shares * first_price

print(f"초기: 가격={first_price:,.0f}, 주식={s_shares}주, 현금={s_cash:,.0f}")

wdr_mismatches = []
wdr_match = 0

for i in range(1, len(df)):
    row = df.iloc[i]
    cur_price = row["price"]
    prev_price = df.iloc[i-1]["price"]
    divergence = row["divergence"]

    weekly_pnl = (cur_price - prev_price) * s_shares

    result = strategy.get_rebalance_action(
        weekly_pnl=weekly_pnl,
        divergence=divergence,
        current_shares=s_shares,
        current_price=cur_price,
        cash=s_cash,
    )

    act = result["action"]
    qty = int(result["quantity"])

    if act == "SELL" and qty > 0:
        s_cash += qty * cur_price  # 수수료 0
        s_shares -= qty
    elif act == "BUY" and qty > 0:
        if qty * cur_price <= s_cash:
            s_cash -= qty * cur_price
            s_shares += qty

    csv_shares = row["csv_shares"]
    csv_cash = row["csv_cash"]
    shares_ok = (s_shares == csv_shares)
    cash_ok = abs(s_cash - csv_cash) <= 1

    if shares_ok and cash_ok:
        wdr_match += 1
    else:
        detail = f"W{i:3d} {row['date']:>15s} | div={divergence:>7.2f}% | {result['state']:>12s}"
        detail += f" | pnl={weekly_pnl:>10,.0f} | act={str(act):>4s} qty={qty:>4d}"
        if not shares_ok:
            detail += f" | 주식: WDR={s_shares} CSV={csv_shares} (Δ={s_shares-csv_shares})"
        if not cash_ok:
            detail += f" | 현금: WDR={s_cash:>12,.0f} CSV={csv_cash:>12,.0f} (Δ={s_cash-csv_cash:>8,.0f})"
        wdr_mismatches.append(detail)

print(f"\n총 {total_weeks}주 중 {wdr_match}주 일치 ({wdr_match/max(1,total_weeks)*100:.1f}%)")
print(f"불일치: {len(wdr_mismatches)}건")

if wdr_mismatches:
    print(f"\n불일치 상세 (처음 20건):")
    print("-"*140)
    for m in wdr_mismatches[:20]:
        print(m)
    if len(wdr_mismatches) > 20:
        print(f"... 외 {len(wdr_mismatches)-20}건 더")

wdr_total = s_cash + s_shares * csv_final["price"]
print(f"\n최종 결과:")
print(f"  WDR: 주식={s_shares}주, 현금=₩{s_cash:,.0f}, 총자산=₩{wdr_total:,.0f}")
print(f"  CSV: 주식={csv_final['csv_shares']}주, 현금=₩{csv_final['csv_cash']:,.0f}, 총자산=₩{csv_total:,.0f}")
print(f"  차이: 주식={s_shares-csv_final['csv_shares']}주, 총자산=₩{wdr_total-csv_total:,.0f}")

# ═══════════════════════════════════════════════════════════
# 방법 1 vs 방법 2 일치 확인
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"[검증 요약]")
print(f"{'='*80}")
print(f"방법1 (수동 시뮬): {match_count}/{total_weeks}주 일치 ({match_count/max(1,total_weeks)*100:.1f}%)")
print(f"방법2 (WDRStrategy): {wdr_match}/{total_weeks}주 일치 ({wdr_match/max(1,total_weeks)*100:.1f}%)")
print(f"방법1 vs 방법2 최종: 주식={shares}=={s_shares}? {'✓' if shares==s_shares else '✗'}, 현금 차이={abs(cash-s_cash):,.0f}")

csv_return = (csv_total / INITIAL_CAPITAL - 1) * 100
sim1_return = (sim_total / INITIAL_CAPITAL - 1) * 100
sim2_return = (wdr_total / INITIAL_CAPITAL - 1) * 100
print(f"\n수익률 비교:")
print(f"  CSV:  {csv_return:+.2f}%  (₩{csv_total:,.0f})")
print(f"  수동: {sim1_return:+.2f}%  (₩{sim_total:,.0f})")
print(f"  WDR:  {sim2_return:+.2f}%  (₩{wdr_total:,.0f})")
