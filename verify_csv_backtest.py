#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
위대리 K-2X 백테스트 CSV vs 앱 WDR 전략 로직 비교 검증
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import re

# ── CSV 파싱 ──
csv_path = r"위대리 K-2X 백테스트.csv"
raw = pd.read_csv(csv_path, header=None, encoding='utf-8-sig')

# 데이터 영역: 열 10~25 (K~Z), 행 1이 헤더
# 헤더 찾기
header_row = None
for idx, row in raw.iterrows():
    for col_idx in range(len(raw.columns)):
        val = str(row.iloc[col_idx]) if pd.notna(row.iloc[col_idx]) else ""
        if "Date" in val:
            header_row = idx
            date_col = col_idx
            break
    if header_row is not None:
        break

if header_row is None:
    print("ERROR: 헤더를 찾을 수 없습니다.")
    sys.exit(1)

print(f"Date 열 오프셋: {date_col}")

print(f"헤더 행: {header_row}")

# 데이터 파싱
data_rows = []
for idx in range(header_row + 1, len(raw)):
    row = raw.iloc[idx]
    date_val = str(row.iloc[date_col]).strip() if pd.notna(row.iloc[date_col]) else ""
    if not date_val or date_val == "nan":
        continue

    def parse_won(val):
        """₩ 기호와 콤마 제거하여 숫자로 변환"""
        if pd.isna(val):
            return 0.0
        s = str(val).replace("₩", "").replace("â©", "").replace(",", "").replace(" ", "").strip()
        if s == "" or s == "nan":
            return 0.0
        # 음수 처리
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

    c = date_col  # 기준 오프셋
    price = parse_won(row.iloc[c+1])        # TIGER나스닥레버리지 가격
    change_pct = parse_pct(row.iloc[c+2])   # 주가변동%
    divergence = parse_pct(row.iloc[c+3])   # TIGER미국나스닥100 시장평가
    target_amount = parse_won(row.iloc[c+5]) # 목표 매매금
    target_qty = parse_won(row.iloc[c+6])    # 목표 매매 수량
    actual_qty = parse_won(row.iloc[c+7])    # 실 매매 수량
    actual_amount = parse_won(row.iloc[c+8]) # 목표 실 매매금
    total_shares = parse_won(row.iloc[c+9])  # 총 보유 주식수
    total_eval = parse_won(row.iloc[c+10])   # 총 평가금
    new_eval = parse_won(row.iloc[c+11])     # New 평가금
    cash = parse_won(row.iloc[c+12])         # 예수금
    total_assets = parse_won(row.iloc[c+13]) # 총자산
    cash_ratio = parse_pct(row.iloc[c+14])   # 현금비율
    dd = parse_pct(row.iloc[c+15])           # DD

    if price <= 0:
        continue

    data_rows.append({
        "date": date_val,
        "price": price,
        "change_pct": change_pct,
        "divergence": divergence,
        "target_amount": target_amount,
        "target_qty": int(target_qty),
        "actual_qty": int(actual_qty),
        "actual_amount": actual_amount,
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

# ── CSV 설정값 ──
print("\n" + "="*70)
print("CSV 설정값 (3단계)")
print("="*70)
INITIAL_CAPITAL = 20_000_000
INITIAL_CASH_RATIO = 0.16  # 16%
OV_THRESHOLD = 5.0
UN_THRESHOLD = -6.0
SELL_OV = 100.0
SELL_NEU = 66.7
SELL_UN = 60.0
BUY_OV = 66.7
BUY_NEU = 66.7
BUY_UN = 120.0
MIN_CASH_RATIO = 0.10
MIN_STOCK_RATIO = 0.10
COMMISSION = 0.0  # 수수료 없음

print(f"초기자본: ₩{INITIAL_CAPITAL:,}")
print(f"초기현금비율: {INITIAL_CASH_RATIO*100}%")
print(f"고평가 > {OV_THRESHOLD}% | 저평가 < {UN_THRESHOLD}%")
print(f"수수료: {COMMISSION*100}%")

# ── 전략 시뮬레이션 ──
print("\n" + "="*70)
print("주별 비교 시작")
print("="*70)

first_price = df.iloc[0]["price"]
buy_ratio = 1.0 - INITIAL_CASH_RATIO  # 0.84
shares = int((INITIAL_CAPITAL * buy_ratio) / first_price)
cash = INITIAL_CAPITAL - shares * first_price

print(f"\n초기매수: 가격={first_price:,.0f}, 주식={shares}주, 현금={cash:,.0f}")
print(f"CSV 초기: 주식={df.iloc[0]['csv_shares']}주, 현금={df.iloc[0]['csv_cash']:,.0f}")

mismatches = []
total_weeks = len(df)
match_count = 0

# 첫 주 확인
if shares != df.iloc[0]["csv_shares"]:
    mismatches.append(f"W0 초기주식: SIM={shares} CSV={df.iloc[0]['csv_shares']}")
if abs(cash - df.iloc[0]["csv_cash"]) > 1:
    mismatches.append(f"W0 초기현금: SIM={cash:,.0f} CSV={df.iloc[0]['csv_cash']:,.0f}")

peak_assets = cash + shares * first_price

for i in range(1, len(df)):
    row = df.iloc[i]
    cur_price = row["price"]
    prev_price = df.iloc[i-1]["price"]
    divergence = row["divergence"]

    weekly_pnl = (cur_price - prev_price) * shares
    total_value = cash + shares * cur_price

    # 시장 상태 판단 (3단계)
    if divergence > OV_THRESHOLD:
        state = "고평가"
        sell_ratio = SELL_OV
        buy_ratio_val = BUY_OV
    elif divergence < UN_THRESHOLD:
        state = "저평가"
        sell_ratio = SELL_UN
        buy_ratio_val = BUY_UN
    else:
        state = "중립"
        sell_ratio = SELL_NEU
        buy_ratio_val = BUY_NEU

    action = None
    qty = 0

    if weekly_pnl > 0 and shares > 0:
        # 매도 (이익 실현)
        sell_amount = weekly_pnl * (sell_ratio / 100.0)
        qty = int(sell_amount / cur_price)

        # 최소 주식 비율 확인
        if qty > 0:
            new_shares = shares - qty
            new_stock_value = new_shares * cur_price
            if total_value > 0 and new_stock_value / total_value < MIN_STOCK_RATIO:
                max_sell = shares - max(1, int(total_value * MIN_STOCK_RATIO / cur_price))
                qty = max(0, min(qty, max_sell))

        if qty > shares:
            qty = shares
        if qty > 0:
            action = "SELL"

    elif weekly_pnl < 0:
        # 매수 (저가 매수)
        buy_amount = abs(weekly_pnl) * (buy_ratio_val / 100.0)
        max_buy_cash = cash - max(0, total_value * MIN_CASH_RATIO)
        buy_amount = min(buy_amount, max(0, max_buy_cash))
        qty = int(buy_amount / cur_price)

        if qty > 0 and (qty * cur_price) <= cash:
            action = "BUY"
        else:
            qty = 0

    # 실행
    if action == "SELL" and qty > 0:
        amount = qty * cur_price
        fee = amount * COMMISSION
        cash += amount - fee
        shares -= qty
    elif action == "BUY" and qty > 0:
        amount = qty * cur_price
        fee = amount * COMMISSION
        total_cost = amount + fee
        if total_cost <= cash:
            cash -= total_cost
            shares += qty

    # CSV와 비교
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
        detail = f"W{i:3d} {row['date']:>15s} | div={divergence:>7.2f}% | pnl={weekly_pnl:>12,.0f} | "
        if not shares_ok:
            detail += f"주식: SIM={shares} CSV={csv_shares} (차이={shares-csv_shares}) | "
        if not cash_ok:
            detail += f"현금: SIM={cash:>12,.0f} CSV={csv_cash:>12,.0f} (차이={cash-csv_cash:>8,.0f}) | "
        if not total_ok:
            detail += f"총자산: SIM={sim_total:>12,.0f} CSV={csv_total:>12,.0f}"
        mismatches.append(detail)

    # DD 계산
    cur_total = cash + shares * cur_price
    if cur_total > peak_assets:
        peak_assets = cur_total

# ── 결과 요약 ──
print(f"\n{'='*70}")
print(f"검증 결과 요약")
print(f"{'='*70}")
print(f"총 {total_weeks}주 중 {match_count}주 일치 ({match_count/max(1,total_weeks-1)*100:.1f}%)")
print(f"불일치: {len(mismatches)}건")

if mismatches:
    print(f"\n불일치 상세 (처음 30건):")
    print("-"*100)
    for m in mismatches[:30]:
        print(m)
    if len(mismatches) > 30:
        print(f"... 외 {len(mismatches)-30}건 더")

# 최종 비교
csv_final = df.iloc[-1]
sim_final_total = cash + shares * csv_final["price"]
csv_final_total = csv_final["csv_total_assets"]

print(f"\n최종 결과:")
print(f"  SIM: 주식={shares}주, 현금=₩{cash:,.0f}, 총자산=₩{sim_final_total:,.0f}")
print(f"  CSV: 주식={csv_final['csv_shares']}주, 현금=₩{csv_final['csv_cash']:,.0f}, 총자산=₩{csv_final_total:,.0f}")
print(f"  차이: 주식={shares-csv_final['csv_shares']}주, 현금=₩{cash-csv_final['csv_cash']:,.0f}, 총자산=₩{sim_final_total-csv_final_total:,.0f}")

# CAGR / MDD 계산
days = (len(df) - 1) * 7  # 대략적 일수
cagr = ((sim_final_total / INITIAL_CAPITAL) ** (365.0/days) - 1) * 100 if days > 0 else 0
print(f"\n  SIM 수익률: {(sim_final_total/INITIAL_CAPITAL-1)*100:+.2f}%")
print(f"  CSV 수익률: {(csv_final_total/INITIAL_CAPITAL-1)*100:+.2f}%")
