import streamlit as st
import os

def _get_runtime_value(keys, default=""):
    """환경변수 및 Streamlit Secrets에서 값을 읽어오는 공통 함수."""
    if isinstance(keys, str):
        keys = (keys,)

    for key in keys:
        v = os.getenv(key, "")
        if str(v).strip():
            return v

    try:
        for key in keys:
            v = st.secrets.get(key, "")
            if str(v).strip():
                return v
    except Exception:
        pass

    return default

def _load_config(config_file):
    """설정 파일 로드 공통 함수."""
    import json
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_config(config_file, config):
    """설정 파일 저장 공통 함수."""
    import json
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config():
    """기본 설정 파일(user_config.json) 로드."""
    return _load_config("user_config.json")

def save_config(config):
    """기본 설정 파일(user_config.json)에 저장."""
    _save_config("user_config.json", config)
