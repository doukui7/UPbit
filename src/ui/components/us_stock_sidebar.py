"""미국주식 모드 — 사이드바 (LS증권 API 설정)."""
import streamlit as st
from src.utils.helpers import load_mode_config, save_mode_config
from src.constants import IS_CLOUD


def render_us_stock_sidebar(config, save_config) -> dict | None:
    """사이드바에 LS증권 API 키 / 계좌 설정을 렌더링하고 설정값을 반환."""
    _cfg = load_mode_config("us_stock")

    # 환경변수 / config / mode_config 순서로 폴백
    import os
    _ak = str(_cfg.get("ls_app_key", "") or config.get("ls_app_key", "") or os.getenv("LS_APP_KEY", ""))
    _sk = str(_cfg.get("ls_secret_key", "") or config.get("ls_secret_key", "") or os.getenv("LS_SECRET_KEY", ""))
    _acct = str(_cfg.get("ls_account_no", "") or config.get("ls_account_no", "") or os.getenv("LS_ACCOUNT_NO", ""))
    _pwd = str(_cfg.get("ls_account_pwd", "") or config.get("ls_account_pwd", "") or os.getenv("LS_ACCOUNT_PWD", ""))

    if not IS_CLOUD:
        with st.sidebar.expander("LS증권 API 설정", expanded=not bool(_ak)):
            _ak = st.text_input("App Key", value=_ak, type="password", key="ls_app_key_input")
            _sk = st.text_input("Secret Key", value=_sk, type="password", key="ls_secret_key_input")
            _acct = st.text_input("계좌번호", value=_acct, key="ls_account_no_input")
            _pwd = st.text_input("계좌비밀번호", value=_pwd, type="password", key="ls_account_pwd_input")
            if st.button("LS 설정 저장", key="save_ls_cfg"):
                _cfg["ls_app_key"] = _ak.strip()
                _cfg["ls_secret_key"] = _sk.strip()
                _cfg["ls_account_no"] = _acct.strip()
                _cfg["ls_account_pwd"] = _pwd.strip()
                save_mode_config("us_stock", _cfg)
                config["ls_app_key"] = _ak.strip()
                config["ls_secret_key"] = _sk.strip()
                config["ls_account_no"] = _acct.strip()
                config["ls_account_pwd"] = _pwd.strip()
                save_config(config)
                st.success("LS 설정 저장 완료!")

    if not _ak or not _sk:
        st.sidebar.warning("LS증권 API Key를 입력하세요.")
        return None

    return {
        "ls_app_key": _ak.strip(),
        "ls_secret_key": _sk.strip(),
        "ls_account_no": _acct.strip(),
        "ls_account_pwd": _pwd.strip(),
        "us_cfg": _cfg,
    }
