"""
보조 전략: MA 이격도 기반 역추세 분할매수 + 익절 전략

메인 전략(SMA/Donchian)이 현금(CASH) 포지션일 때만 작동.
단기·중기 이동평균 이격도가 모두 과매도 영역이면 분할 매수,
**전체 평균 매수가(평단가)** 대비 TP1%~TP2% 균등 분배 지정가 매도.

분할 매수 조건: 과매도 + 직전 매수가보다 가격이 더 하락한 경우에만 추가 매수.

매수 시드 배분:
  - 동일: 각 분할 동일 금액
  - 피라미딩: 분할차수마다 pyramid_ratio배 증가 (예: 1.5배)
"""
import numpy as np
import pandas as pd


def compute_disparity(close_arr: np.ndarray, ma_period: int) -> np.ndarray:
    """이격도(%) 계산: (close - MA) / MA * 100"""
    ma = pd.Series(close_arr).rolling(window=ma_period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        disp = (close_arr - ma) / ma * 100
    return disp


def generate_main_position(close_arr: np.ndarray, high_arr: np.ndarray,
                           low_arr: np.ndarray,
                           strategy: str, buy_p: int, sell_p: int) -> np.ndarray:
    """
    메인 전략의 포지션 상태 배열 생성.
    Returns: int8 array (0=CASH, 1=HOLD) — 실제 포지션 상태(다음봉 체결 반영).
    """
    n = len(close_arr)

    if strategy == "Donchian":
        upper = pd.Series(high_arr).rolling(window=buy_p).max().shift(1).values
        lower = pd.Series(low_arr).rolling(window=sell_p).min().shift(1).values
        signal = np.zeros(n, dtype=np.int8)
        signal[close_arr > upper] = 1   # BUY
        signal[close_arr < lower] = -1  # SELL
    else:  # SMA
        ma = pd.Series(close_arr).rolling(window=buy_p).mean().values
        signal = np.zeros(n, dtype=np.int8)
        valid = ~np.isnan(ma)
        signal[valid & (close_arr > ma)] = 1
        signal[valid & (close_arr <= ma)] = -1

    # signal → position 변환 (다음봉 시가 체결)
    pos = np.zeros(n, dtype=np.int8)
    cur = 0  # 0=CASH
    pending = 0
    for i in range(n):
        if pending == 1:
            cur = 1
            pending = 0
        elif pending == -1:
            cur = 0
            pending = 0

        if cur == 0 and signal[i] == 1:
            pending = 1
        elif cur == 1 and signal[i] == -1:
            pending = -1

        pos[i] = cur

    return pos


def _calc_buy_weights(split_count: int, buy_seed_mode: str = "equal", pyramid_ratio: float = 1.0) -> np.ndarray:
    """
    분할 매수 가중치 배열 계산.

    buy_seed_mode:
      - "equal": 동일 분할 (각 1/N)
      - "pyramiding": 피라미딩 (i차 = ratio^i, 정규화)

    Returns: weights array, sum = 1.0
    """
    if split_count <= 1:
        return np.ones(split_count) / split_count

    if buy_seed_mode != "pyramiding":
        return np.ones(split_count) / split_count

    # 피라미딩 모드에서 ratio가 1 이하면 사실상 동일 분할
    if pyramid_ratio <= 1.0:
        return np.ones(split_count) / split_count

    raw = np.array([pyramid_ratio ** i for i in range(split_count)])
    return raw / raw.sum()


def fast_simulate_aux(open_arr: np.ndarray, close_arr: np.ndarray,
                      high_arr: np.ndarray, low_arr: np.ndarray,
                      main_pos: np.ndarray,
                      disp_short: np.ndarray, disp_long: np.ndarray,
                      oversold_threshold: float,
                      tp1_pct: float, tp2_pct: float,
                      fee: float, slippage: float,
                      initial_balance: float,
                      year_arr: np.ndarray = None,
                      split_count: int = 2,
                      buy_seed_mode: str = "equal",
                      pyramid_ratio: float = 1.0,
                      return_series: bool = False) -> dict:
    """
    보조 전략 고속 시뮬레이션 (N분할 매수 + 평단가 기반 TP).

    핵심 로직:
      1) 과매도 시그널 + 직전 매수가보다 하락 시에만 분할 매수
      2) TP는 전체 평균 매수가(평단가) 기준으로 계산
      3) 매수 때마다 평단가 갱신 → TP 라인 재계산
      4) 메인 전략 BUY 전환 시 보조 전량 청산

    split_count: 분할 매수 횟수 (1~20)
    buy_seed_mode: "equal" | "pyramiding"
    pyramid_ratio: 피라미딩 배율 (예: 1.3이면 다음 티어가 이전 티어의 1.3배)
    tp1_pct ~ tp2_pct: 평단가 대비 익절 % (N등분)
    """
    n = len(open_arr)
    split_count = max(1, min(split_count, 20))

    balance = initial_balance

    # TP 비율 배열 (균등 분배)
    if split_count == 1:
        tp_pcts = np.array([tp2_pct])
    else:
        tp_pcts = np.linspace(tp1_pct, tp2_pct, split_count)

    # 매수 가중치 (균등/피라미딩)
    buy_weights = _calc_buy_weights(split_count, buy_seed_mode=buy_seed_mode, pyramid_ratio=pyramid_ratio)

    # ── 포지션 추적 변수 ──
    total_qty = 0.0        # 보유 코인 총 수량
    weighted_cost = 0.0    # sum(qty_i * entry_price_i) → 평단가 계산용
    avg_price = 0.0        # 평균 매수가 (평단가)
    buys_done = 0          # 완료된 매수 횟수
    sells_done = 0         # 완료된 TP 매도 횟수
    last_buy_price = 0.0   # 직전 매수 체결가 (추가하락 조건용)
    pending_buy = False

    # TP 가격 배열 (평단가 변경 시 재계산)
    tp_prices = np.zeros(split_count)
    tp_sold = np.zeros(split_count, dtype=bool)

    wins = 0
    total_trades = 0
    equity = np.empty(n)

    slip_buy = 1 + slippage / 100
    slip_sell = 1 - slippage / 100
    fee_mult = 1 - fee

    for i in range(n):
        op = open_arr[i]
        hi = high_arr[i]
        cl = close_arr[i]
        m_pos = main_pos[i]

        # ── 1) Pending buy 체결 (전일 시그널 → 금일 시가) ──
        if pending_buy and buys_done < split_count:
            ep = op * slip_buy

            # 매수 금액 결정
            # 피라미딩: 사이클 잔액 기준 가중치로 배분
            # 남은 가중치 합계로 정규화하여 잔액 전체 활용
            remaining_weight_sum = buy_weights[buys_done:].sum()
            buy_amount = balance * (buy_weights[buys_done] / remaining_weight_sum)
            coin_qty = buy_amount * fee_mult / ep

            # 포지션 갱신
            total_qty += coin_qty
            weighted_cost += coin_qty * ep
            avg_price = weighted_cost / total_qty
            balance -= buy_amount
            last_buy_price = ep
            buys_done += 1
            pending_buy = False

            # 평단가 변경 → 미매도 TP 전부 재계산
            for s in range(split_count):
                if not tp_sold[s]:
                    tp_prices[s] = avg_price * (1 + tp_pcts[s] / 100)

        # ── 2) TP 체크 (장중 고가 기준, 낮은 TP부터) ──
        for s in range(split_count):
            if s < buys_done and not tp_sold[s] and total_qty > 0:
                if hi >= tp_prices[s]:
                    # 남은 TP 횟수로 균등 분배
                    remaining_tps = split_count - sells_done
                    sell_qty = total_qty / remaining_tps if remaining_tps > 0 else total_qty

                    balance += sell_qty * tp_prices[s] * fee_mult
                    total_qty -= sell_qty
                    weighted_cost -= sell_qty * avg_price  # 평단가 유지
                    total_trades += 1
                    wins += 1  # TP 매도는 항상 이익
                    tp_sold[s] = True
                    sells_done += 1

        # ── 3) 메인이 BUY 전환 시 보조 포지션 전량 청산 ──
        if total_qty > 0 and m_pos == 1:
            sell_price = op * slip_sell
            balance += total_qty * sell_price * fee_mult
            total_trades += 1
            if sell_price > avg_price:
                wins += 1
            # 완전 리셋
            total_qty = 0.0
            weighted_cost = 0.0
            avg_price = 0.0
            buys_done = 0
            sells_done = 0
            last_buy_price = 0.0
            pending_buy = False
            tp_sold[:] = False

        # ── 4) 사이클 완료 리셋 (모든 매도 완료) ──
        if buys_done > 0 and total_qty <= 1e-12:
            buys_done = 0
            sells_done = 0
            total_qty = 0.0
            weighted_cost = 0.0
            avg_price = 0.0
            last_buy_price = 0.0
            tp_sold[:] = False

        # ── 5) 매수 시그널 체크 ──
        #   조건: 메인 CASH + 남은 분할 있음 + 과매도 + 가격 추가 하락
        if m_pos == 0 and buys_done < split_count and balance > 0 and not pending_buy:
            ds = disp_short[i]
            dl = disp_long[i]
            if not (np.isnan(ds) or np.isnan(dl)):
                if ds < oversold_threshold and dl < oversold_threshold:
                    # 첫 매수: 무조건 / 추가 매수: 직전 매수가보다 하락 시에만
                    if buys_done == 0 or cl < last_buy_price:
                        pending_buy = True

        # ── 6) Equity 기록 ──
        equity[i] = balance + total_qty * cl

    # ── 성과 지표 계산 ──
    final_eq = equity[-1]
    total_ret = (final_eq - initial_balance) / initial_balance * 100

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    mdd = dd.min()

    avg_yearly_mdd = mdd
    if year_arr is not None and len(year_arr) == n:
        unique_years = np.unique(year_arr)
        yearly_mdds = [dd[year_arr == yr].min() for yr in unique_years]
        if yearly_mdds:
            avg_yearly_mdd = float(np.mean(yearly_mdds))

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1)
    returns = np.nan_to_num(returns)
    std = returns.std()
    sharpe = (returns.mean() / std * np.sqrt(365)) if std > 0 else 0

    days_total = max(1, n)
    cagr = 0
    if final_eq > 0 and initial_balance > 0 and days_total > 1:
        cagr = ((final_eq / initial_balance) ** (365 / days_total) - 1) * 100

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "mdd": mdd,
        "avg_yearly_mdd": avg_yearly_mdd,
        "win_rate": win_rate,
        "trade_count": total_trades,
        "sharpe": sharpe,
        "final_equity": final_eq,
        "final_status": "HOLD" if total_qty > 1e-12 else "CASH",
        "next_action": "BUY" if pending_buy else None,
        "buy_seed_mode": buy_seed_mode,
        "pyramid_ratio": pyramid_ratio,
        "equity_curve": equity if return_series else None,
        "drawdown_curve": dd if return_series else None,
    }
