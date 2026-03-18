import streamlit as st
import os
import json
from dotenv import load_dotenv

# --- Import Constants & Config ---
from src.constants import *
from src.utils.helpers import load_config, save_config, load_mode_config, save_mode_config

# --- UI Modes ---
from src.ui.coin_mode import render_coin_mode
from src.ui.gold_mode import render_gold_mode
from src.ui.isa_mode import render_kis_isa_mode
from src.ui.pension_mode import render_kis_pension_mode
from src.ui.project_logic import render_project_logic
from src.ui.us_stock_mode import render_us_stock_mode

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
        "미국주식": "US_STOCK",
        "프로젝트 로직": "PROJECT",
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

    # ── 텔레그램 설정 (모든 모드 공통, config/common.json) ──
    from src.utils.helpers import _get_runtime_value
    from src.constants import IS_CLOUD
    _common_cfg = load_mode_config("common")
    _tg_token = str(_common_cfg.get("telegram_bot_token", "") or config.get("telegram_bot_token", "") or _get_runtime_value(("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN"), ""))
    _tg_chat = str(_common_cfg.get("telegram_chat_id", "") or config.get("telegram_chat_id", "") or _get_runtime_value(("TELEGRAM_CHAT_ID",), ""))
    if not IS_CLOUD:
        with st.sidebar.expander("📬 텔레그램 알림", expanded=False):
            _tg_token = st.text_input("봇 토큰", value=_tg_token, type="password", key="global_tg_bot_token")
            _tg_chat = st.text_input("Chat ID", value=_tg_chat, key="global_tg_chat_id")
            if st.button("텔레그램 설정 저장", key="save_tg_cfg"):
                _common_cfg["telegram_bot_token"] = str(_tg_token).strip()
                _common_cfg["telegram_chat_id"] = str(_tg_chat).strip()
                save_mode_config("common", _common_cfg)
                # 전역 config에도 반영 (하위호환)
                config["telegram_bot_token"] = str(_tg_token).strip()
                config["telegram_chat_id"] = str(_tg_chat).strip()
                save_config(config)
                st.success("텔레그램 설정 저장 완료!")
    st.session_state["_tg_bot_token"] = _tg_token
    st.session_state["_tg_chat_id"] = _tg_chat

    # 모드별 UI 렌더링
    if trading_mode == "GOLD":
        render_gold_mode(config, save_config)
    elif trading_mode == "ISA":
        render_kis_isa_mode(config, save_config)
    elif trading_mode == "PENSION":
        render_kis_pension_mode(config, save_config)
    elif trading_mode == "US_STOCK":
        render_us_stock_mode(config, save_config)
    elif trading_mode == "PROJECT":
        render_project_logic()
    else:
        render_coin_mode(config, save_config)

if __name__ == "__main__":
    main()
