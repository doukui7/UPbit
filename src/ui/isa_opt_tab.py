"""ISA 위대리 최적화 탭 (tab_i6).

render_isa_opt_tab(ctx) 함수 하나를 외부에 노출한다.
ctx keys:
  trader, isa_etf_code, isa_trend_etf_code, wdr_eval_mode, isa_start_date, isa_seed, get_daily_chart, wdr_ov, wdr_un
"""
import streamlit as st
import pandas as pd
import numpy as np
import itertools
import os
import time
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.strategy.widaeri import WDRStrategy, _wdr_opt_task
import src.engine.data_cache as _dc
from src.utils.db_manager import DBManager

def _snap_step(_v: float, _step: float, _vmin: float | None = None, _vmax: float | None = None) -> float:
    _s = float(_step) if float(_step) > 0 else 1.0
    _x = round(round(float(_v) / _s) * _s, 4)
    if _vmin is not None:
        _x = max(float(_vmin), _x)
    if _vmax is not None:
        _x = min(float(_vmax), _x)
    return float(round(_x, 4))

def _grid_values(_vmin: float, _vmax: float, _step: float) -> list[float]:
    _step = max(float(_step), 1e-9)
    if float(_vmax) < float(_vmin):
        _vmin, _vmax = _vmax, _vmin
    _n = int(np.floor((float(_vmax) - float(_vmin)) / _step + 1e-9)) + 1
    vals = [round(float(_vmin) + i * _step, 4) for i in range(max(1, _n))]
    if vals[-1] < float(_vmax) - 1e-9:
        vals.append(round(float(_vmax), 4))
    return vals

def render_isa_opt_tab(ctx: dict):
    trader = ctx["trader"]
    isa_etf_code = ctx["isa_etf_code"]
    isa_trend_etf_code = ctx["isa_trend_etf_code"]
    wdr_eval_mode = ctx["wdr_eval_mode"]
    isa_start_date = ctx["isa_start_date"]
    isa_seed = ctx["isa_seed"]
    _get_isa_daily_chart = ctx["get_daily_chart"]
    try:
        wdr_ov = float(ctx.get("wdr_ov", st.session_state.get("isa_wdr_ov", 5.0)))
    except Exception:
        wdr_ov = 5.0
    try:
        wdr_un = float(ctx.get("wdr_un", st.session_state.get("isa_wdr_un", -6.0)))
    except Exception:
        wdr_un = -6.0

    st.header("위대리 최적화")
    st.caption("그리드/랜덤 탐색으로 임계값과 초기 비중 조합을 빠르게 비교합니다.")

    def _snap_step(_v: float, _step: float, _vmin: float | None = None, _vmax: float | None = None) -> float:
        _s = float(_step) if float(_step) > 0 else 1.0
        _x = round(round(float(_v) / _s) * _s, 4)
        if _vmin is not None:
            _x = max(float(_vmin), _x)
        if _vmax is not None:
            _x = min(float(_vmax), _x)
        return float(round(_x, 4))

    def _grid_values(_vmin: float, _vmax: float, _step: float) -> list[float]:
        _step = max(float(_step), 1e-9)
        if float(_vmax) < float(_vmin):
            _vmin, _vmax = _vmax, _vmin
        _n = int(np.floor((float(_vmax) - float(_vmin)) / _step + 1e-9)) + 1
        vals = [round(float(_vmin) + i * _step, 4) for i in range(max(1, _n))]
        if vals[-1] < float(_vmax) - 1e-9:
            vals.append(round(float(_vmax), 4))
        return vals

    _opt_listing_date = _dc.get_wdr_trade_listing_date(str(isa_etf_code))
    _opt_min_start = pd.to_datetime(_opt_listing_date).date() if _opt_listing_date else pd.to_datetime("2012-01-01").date()

    # ETF 변경 시 opt_wdr_start 자동 갱신
    _opt_etf_sync_key = "isa_opt_etf_sync"
    if st.session_state.get(_opt_etf_sync_key) != str(isa_etf_code):
        st.session_state[_opt_etf_sync_key] = str(isa_etf_code)
        st.session_state["opt_wdr_start"] = _opt_min_start

    # min_value 보다 작은 저장값은 클램프 (ETF 변경 직후 등)
    _opt_cur_start = st.session_state.get("opt_wdr_start")
    if _opt_cur_start is not None:
        try:
            if pd.to_datetime(_opt_cur_start).date() < _opt_min_start:
                st.session_state["opt_wdr_start"] = _opt_min_start
        except Exception:
            st.session_state["opt_wdr_start"] = _opt_min_start

    # 최적화 파라미터 초기화
    _opt_start_default = max(
        pd.to_datetime(isa_start_date).date() if isa_start_date else _opt_min_start,
        _opt_min_start,
    )
    _opt_params_init = {
        "opt_wdr_eval_mode": int(wdr_eval_mode),
        "opt_wdr_start": _opt_start_default,
        "opt_wdr_end": pd.Timestamp.now().date(),
        "opt_wdr_cap": int(isa_seed),
        "opt_wdr_fee": 0.05,
        "opt_wdr_max_evals": 50000,
        "opt_wdr_n_trials": 2000,
        "opt_ov_min": 2.0,
        "opt_ov_max": 10.0,
        "opt_ov_step": 1.0,
        "opt_un_min": -12.0,
        "opt_un_max": -3.0,
        "opt_un_step": 1.0,
        "opt_ratio_min": 50.0,
        "opt_ratio_max": 100.0,
        "opt_ratio_step": 10.0,
    }
    for _k, _v in _opt_params_init.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        opt_eval_mode = st.selectbox(
            "평가 시스템",
            [3, 5],
            format_func=lambda x: f"{x}단계",
            key="opt_wdr_eval_mode",
        )
        opt_start = st.date_input(
            "탐색 시작일",
            min_value=_opt_min_start,
            key="opt_wdr_start",
        )
        if _opt_listing_date:
            st.caption(f"매매 ETF 상장일: {_opt_listing_date}")
    with r1c2:
        opt_search_mode = st.selectbox("탐색 방식", ["그리드 탐색", "랜덤 탐색", "Optuna (베이지안)"], key="opt_wdr_search_mode")
        opt_sort_col = st.selectbox("정렬 기준", ["Calmar", "Sharpe", "CAGR(%)", "수익률(%)"], key="opt_wdr_sort_col")
        opt_end = st.date_input(
            "탐색 종료일",
            key="opt_wdr_end",
        )
    with r1c3:
        opt_cap = int(st.number_input("초기 자본(원)", min_value=1_000_000, max_value=1_000_000_000, step=1_000_000, key="opt_wdr_cap"))
        opt_fee = float(st.number_input("매매 수수료(%)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="opt_wdr_fee"))
        opt_max_evals = int(st.number_input("최대 평가 수", min_value=100, max_value=500000, step=100, key="opt_wdr_max_evals"))
        opt_n_trials = 0
        if opt_search_mode in ["랜덤 탐색", "Optuna (베이지안)"]:
            _label = "탐색 횟수" if opt_search_mode == "Optuna (베이지안)" else "랜덤 탐색 횟수"
            opt_n_trials = int(st.number_input(_label, min_value=100, max_value=500000, step=100, key="opt_wdr_n_trials"))

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        opt_ov_min = float(st.number_input("고평가 최소(%)", step=0.5, key="opt_ov_min"))
        opt_ov_max = float(st.number_input("고평가 최대(%)", step=0.5, key="opt_ov_max"))
        opt_ov_step = float(st.number_input("고평가 스텝", min_value=0.1, step=0.1, key="opt_ov_step"))
    with r2c2:
        opt_un_min = float(st.number_input("저평가 최소(%)", step=0.5, key="opt_un_min"))
        opt_un_max = float(st.number_input("저평가 최대(%)", step=0.5, key="opt_un_max"))
        opt_un_step = float(st.number_input("저평가 스텝", min_value=0.1, step=0.1, key="opt_un_step"))
    with r2c3:
        opt_ratio_min = float(st.number_input("초기 주식 비중 최소(%)", min_value=0.0, max_value=100.0, step=1.0, key="opt_ratio_min"))
        opt_ratio_max = float(st.number_input("초기 주식 비중 최대(%)", min_value=0.0, max_value=100.0, step=1.0, key="opt_ratio_max"))
        opt_ratio_step = float(st.number_input("초기 주식 비중 스텝", min_value=0.1, max_value=100.0, step=0.1, key="opt_ratio_step"))

    _ratio_defaults = {
        "sell_ratio_overvalue": 100.0,
        "sell_ratio_neutral": 66.7,
        "sell_ratio_undervalue": 60.0,
        "buy_ratio_overvalue": 66.7,
        "buy_ratio_neutral": 66.7,
        "buy_ratio_undervalue": 120.0,
        "sell_ratio_super_overvalue": 150.0,
        "sell_ratio_super_undervalue": 33.0,
        "buy_ratio_super_overvalue": 33.0,
        "buy_ratio_super_undervalue": 200.0,
    }
    # 비율 기본값 초기화
    for _k, _v in _ratio_defaults.items():
        _key = f"opt_{_k.replace('ratio_', '')}"
        if _key not in st.session_state:
            st.session_state[_key] = _v

    with st.expander("평가별 매수/매도 비율(최적화 기본값)", expanded=False):
        rr1, rr2 = st.columns(2)
        with rr1:
            st.number_input("매도 고평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_sell_ov")
            st.number_input("매도 중립(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_sell_neu")
            st.number_input("매도 저평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_sell_un")
            if int(opt_eval_mode) == 5:
                st.number_input("매도 초고평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_sell_sov")
                st.number_input("매도 초저평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_sell_sun")
        with rr2:
            st.number_input("매수 고평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_buy_ov")
            st.number_input("매수 중립(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_buy_neu")
            st.number_input("매수 저평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_buy_un")
            if int(opt_eval_mode) == 5:
                st.number_input("매수 초고평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_buy_sov")
                st.number_input("매수 초저평가(%)", min_value=0.0, max_value=300.0, step=0.1, key="opt_buy_sun")

    _ov_vals = _grid_values(opt_ov_min, opt_ov_max, opt_ov_step)
    _un_vals = _grid_values(opt_un_min, opt_un_max, opt_un_step)
    _ratio_vals = _grid_values(opt_ratio_min, opt_ratio_max, opt_ratio_step)
    _grid_total = len(_ov_vals) * len(_un_vals) * len(_ratio_vals)
    if opt_search_mode == "그리드 탐색":
        st.info(f"예상 조합: {_grid_total:,}개 | 최대 평가 수: {int(opt_max_evals):,}개")
    else:
        st.info(f"랜덤 탐색: {int(opt_n_trials):,}회 | 최대 평가 수: {int(opt_max_evals):,}개")

    run_opt = st.button("최적화 시작", type="primary", key="opt_wdr_run")
    if run_opt:
        if opt_start > opt_end:
            st.error("기간을 확인해 주세요. (시작일 <= 종료일)")
        else:
            sig_df = _get_isa_daily_chart(str(isa_trend_etf_code), count=5000, end_date=str(opt_end))
            trade_df = _get_isa_daily_chart(str(isa_etf_code), count=5000, end_date=str(opt_end))
            if sig_df is None or trade_df is None or len(sig_df) < 200 or len(trade_df) < 60:
                st.error(f"데이터를 불러오지 못했습니다. ({isa_trend_etf_code}/{isa_etf_code})")
            else:
                _opt_start_eff = str(opt_start)
                if _opt_listing_date and _opt_start_eff < _opt_listing_date:
                    _opt_start_eff = _opt_listing_date

                ratio_settings = {
                    "sell_ratio_overvalue": float(st.session_state.get("opt_sell_ov", _ratio_defaults["sell_ratio_overvalue"])),
                    "sell_ratio_neutral": float(st.session_state.get("opt_sell_neu", _ratio_defaults["sell_ratio_neutral"])),
                    "sell_ratio_undervalue": float(st.session_state.get("opt_sell_un", _ratio_defaults["sell_ratio_undervalue"])),
                    "buy_ratio_overvalue": float(st.session_state.get("opt_buy_ov", _ratio_defaults["buy_ratio_overvalue"])),
                    "buy_ratio_neutral": float(st.session_state.get("opt_buy_neu", _ratio_defaults["buy_ratio_neutral"])),
                    "buy_ratio_undervalue": float(st.session_state.get("opt_buy_un", _ratio_defaults["buy_ratio_undervalue"])),
                }
                if int(opt_eval_mode) == 5:
                    ratio_settings.update({
                        "sell_ratio_super_overvalue": float(st.session_state.get("opt_sell_sov", _ratio_defaults["sell_ratio_super_overvalue"])),
                        "sell_ratio_super_undervalue": float(st.session_state.get("opt_sell_sun", _ratio_defaults["sell_ratio_super_undervalue"])),
                        "buy_ratio_super_overvalue": float(st.session_state.get("opt_buy_sov", _ratio_defaults["buy_ratio_super_overvalue"])),
                        "buy_ratio_super_undervalue": float(st.session_state.get("opt_buy_sun", _ratio_defaults["buy_ratio_super_undervalue"])),
                    })

                tasks: list[tuple] = []
                results = []
                
                if opt_search_mode == "Optuna (베이지안)":
                    import optuna
                    st.info(f"Optuna 베이지안 탐색 시작: {int(opt_n_trials):,}회")
                    
                    def objective(trial):
                        _ov_v = trial.suggest_float("ov", opt_ov_min, opt_ov_max, step=opt_ov_step)
                        _un_v = trial.suggest_float("un", opt_un_min, opt_un_max, step=opt_un_step)
                        _ir_v = trial.suggest_float("ir", opt_ratio_min, opt_ratio_max, step=opt_ratio_step)
                        
                        _settings = {"overvalue_threshold": float(_ov_v), "undervalue_threshold": float(_un_v), **ratio_settings}
                        res = _wdr_opt_task(sig_df, trade_df, _settings, int(opt_eval_mode), float(opt_cap), _opt_start_eff, float(opt_fee) / 100.0, float(_ir_v))
                        
                        if res:
                            # Optuna에 저장할 결과 객체 추가
                            results.append(res)
                            # 최적화할 스코어 반환
                            return res.get(opt_sort_col, 0.0)
                        return -999.0

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=int(opt_n_trials))
                    
                elif opt_search_mode == "그리드 탐색":
                    all_combos = list(itertools.product(_ov_vals, _un_vals, _ratio_vals))
                    if len(all_combos) > int(opt_max_evals):
                        _stride = max(1, len(all_combos) // int(opt_max_evals))
                        all_combos = all_combos[::_stride][:int(opt_max_evals)]
                    for _ov_v, _un_v, _ir_v in all_combos:
                        _settings = {"overvalue_threshold": float(_ov_v), "undervalue_threshold": float(_un_v), **ratio_settings}
                        tasks.append((sig_df, trade_df, _settings, int(opt_eval_mode), float(opt_cap), _opt_start_eff, float(opt_fee) / 100.0, float(_ir_v)))
                else:
                    _n = min(int(opt_n_trials), int(opt_max_evals))
                    _rng = np.random.default_rng(seed=20260302)
                    for _ in range(_n):
                        _ov_v = _snap_step(_rng.uniform(min(_ov_vals), max(_ov_vals)), opt_ov_step, min(_ov_vals), max(_ov_vals))
                        _un_v = _snap_step(_rng.uniform(min(_un_vals), max(_un_vals)), opt_un_step, min(_un_vals), max(_un_vals))
                        _ir_v = _snap_step(_rng.uniform(min(_ratio_vals), max(_ratio_vals)), opt_ratio_step, min(_ratio_vals), max(_ratio_vals))
                        _settings = {"overvalue_threshold": float(_ov_v), "undervalue_threshold": float(_un_v), **ratio_settings}
                        tasks.append((sig_df, trade_df, _settings, int(opt_eval_mode), float(opt_cap), _opt_start_eff, float(opt_fee) / 100.0, float(_ir_v)))

                if tasks:
                    workers = max(2, min((os.cpu_count() or 4), 16))
                    prog = st.progress(0.0)
                    done = 0
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(_wdr_opt_task, *t) for t in tasks]
                        total_n = max(1, len(futures))
                        for f in as_completed(futures):
                            try:
                                r = f.result()
                                if isinstance(r, dict):
                                    results.append(r)
                            except Exception:
                                pass
                            done += 1
                            if done == total_n or done % max(1, total_n // 100) == 0:
                                prog.progress(done / total_n)
                    prog.empty()
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                st.caption(f"완료: {len(results):,}개 결과 / {len(tasks):,}개 평가, 워커 {workers}개, {elapsed_ms:,.1f} ms")

                if results:
                    df_opt = pd.DataFrame(results)
                    _sort_by = opt_sort_col if opt_sort_col in df_opt.columns else "Calmar"
                    df_opt = df_opt.sort_values(_sort_by, ascending=False).reset_index(drop=True)
                    st.session_state["isa_opt_results_df"] = df_opt
                    # DB에 결과 저장
                    db = DBManager()
                    db.save_compute_result(
                        "isa_opt_latest", 
                        f"{isa_trend_etf_code}_{isa_etf_code}_{opt_eval_mode}", 
                        zlib.compress(pickle.dumps(df_opt, protocol=pickle.HIGHEST_PROTOCOL))
                    )
                    st.session_state["isa_opt_results_meta"] = {
                        "sort_col": _sort_by,
                        "eval_mode": int(opt_eval_mode),
                        "start": str(_opt_start_eff),
                        "end": str(opt_end),
                        "cap": int(opt_cap),
                        "fee": float(opt_fee),
                        "ratio_settings": ratio_settings,
                    }

    _opt_df = st.session_state.get("isa_opt_results_df")
    _opt_meta = st.session_state.get("isa_opt_results_meta") or {}
    if isinstance(_opt_df, pd.DataFrame) and not _opt_df.empty:
        st.subheader(f"최적화 결과 ({_opt_meta.get('sort_col', 'Calmar')} 순)")
        st.dataframe(_opt_df.head(300), use_container_width=True, hide_index=True)
        _best = _opt_df.iloc[0]
        st.success(
            f"1등 조합: 고평가 {_best.get('고평가(%)')}% | 저평가 {_best.get('저평가(%)')}% | 초기비중 {_best.get('초기비중(%)')}% | "
            f"수익률 {_best.get('수익률(%)', 0):+.2f}% | Calmar {_best.get('Calmar', 0):.3f}"
        )
        if st.button("1등 조합을 라이브 백테스트에 적용", key="isa_opt_apply_best"):
            payload = {
                "isa_live_eval_mode": int(_opt_meta.get("eval_mode", int(wdr_eval_mode))),
                "isa_live_start": pd.to_datetime(_opt_meta.get("start", isa_start_date)).date(),
                "isa_live_end": pd.to_datetime(_opt_meta.get("end", pd.Timestamp.now().date())).date(),
                "isa_live_cap": int(_opt_meta.get("cap", int(isa_seed))),
                "isa_live_fee": float(_opt_meta.get("fee", 0.05)),
                "isa_live_ov": float(_best.get("고평가(%)", wdr_ov)),
                "isa_live_un": float(_best.get("저평가(%)", wdr_un)),
                "isa_live_ratio": float(_best.get("초기비중(%)", 60.0)),
                "isa_live_sell_ov": float(_opt_meta.get("ratio_settings", {}).get("sell_ratio_overvalue", 100.0)),
                "isa_live_sell_neu": float(_opt_meta.get("ratio_settings", {}).get("sell_ratio_neutral", 66.7)),
                "isa_live_sell_un": float(_opt_meta.get("ratio_settings", {}).get("sell_ratio_undervalue", 60.0)),
                "isa_live_buy_ov": float(_opt_meta.get("ratio_settings", {}).get("buy_ratio_overvalue", 66.7)),
                "isa_live_buy_neu": float(_opt_meta.get("ratio_settings", {}).get("buy_ratio_neutral", 66.7)),
                "isa_live_buy_un": float(_opt_meta.get("ratio_settings", {}).get("buy_ratio_undervalue", 120.0)),
            }
            if int(payload["isa_live_eval_mode"]) == 5:
                payload.update({
                    "isa_live_sell_sov": float(_opt_meta.get("ratio_settings", {}).get("sell_ratio_super_overvalue", 150.0)),
                    "isa_live_sell_sun": float(_opt_meta.get("ratio_settings", {}).get("sell_ratio_super_undervalue", 33.0)),
                    "isa_live_buy_sov": float(_opt_meta.get("ratio_settings", {}).get("buy_ratio_super_overvalue", 33.0)),
                    "isa_live_buy_sun": float(_opt_meta.get("ratio_settings", {}).get("buy_ratio_super_undervalue", 200.0)),
                })
            st.session_state["isa_live_apply_payload"] = payload
            st.session_state["isa_live_apply_token"] = time.time()
            st.success("라이브 백테스트 탭에 적용할 값이 준비되었습니다.")

