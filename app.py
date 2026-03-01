import streamlit as st
import os
import json
from dotenv import load_dotenv

# --- Import Constants & Config ---
from src.constants import *
from src.utils.helpers import load_config, save_config

# --- UI Modes ---
from src.ui.coin_mode import render_coin_mode
from src.ui.gold_mode import render_gold_mode
from src.ui.isa_mode import render_kis_isa_mode
from src.ui.pension_mode import render_kis_pension_mode

# Load environment variables
load_dotenv(override=True)

def main():
    st.set_page_config(
        page_title="UPbit/KIS 통합 트레이딩 시스템",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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
    )
    trading_mode = _mode_map[_mode_label]

    # 사용자가 모드를 변경했을 때만 query_params 갱신
    _prev_mode = st.session_state.get("_last_trading_mode", "")
    if _prev_mode and _prev_mode != trading_mode:
        st.query_params["mode"] = trading_mode
    elif not _saved_mode:
        st.query_params["mode"] = trading_mode
    st.session_state["_last_trading_mode"] = trading_mode

    # 전역 앱 설정 로드
    config = load_config()

    # 모드별 UI 렌더링
    if trading_mode == "GOLD":
        render_gold_mode(config, save_config)
    elif trading_mode == "ISA":
        render_kis_isa_mode(config, save_config)
    elif trading_mode == "PENSION":
        render_kis_pension_mode(config, save_config)
    else:
        render_coin_mode(config, save_config)

if __name__ == "__main__":
    main()
