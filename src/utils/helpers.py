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
    """DB에서 설정을 로드하고, 없으면 user_config.json에서 로드."""
    from src.utils.db_manager import DBManager
    db = DBManager()
    cfg = db.load_setting("app_config")
    if cfg:
        return cfg
    
    # DB에 없으면 기존 JSON 파일 시도
    cfg = _load_config("user_config.json")
    if cfg:
        # JSON에서 읽은 걸 DB에도 저장 (최초 1회 마이그레이션)
        db.save_setting("app_config", cfg)
    return cfg

def save_config(config):
    """DB에 설정을 저장하고 user_config.json에도 백업."""
    from src.utils.db_manager import DBManager
    db = DBManager()
    db.save_setting("app_config", config)

    # 기존 JSON 파일에도 백업 저장
    _save_config("user_config.json", config)


# ── 모드별 개별 설정 파일 ──

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, "config")

def load_mode_config(mode: str) -> dict:
    """모드별 설정 파일 로드 (config/{mode}.json).
    파일이 없으면 기존 전역 config에서 해당 모드 키를 추출하여 반환."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    path = os.path.join(_CONFIG_DIR, f"{mode}.json")
    cfg = _load_config(path)
    if cfg:
        return cfg
    # fallback: 기존 전역 config에서 추출
    return load_config() or {}

def save_mode_config(mode: str, data: dict):
    """모드별 설정 파일 저장 (config/{mode}.json)."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    path = os.path.join(_CONFIG_DIR, f"{mode}.json")
    _save_config(path, data)
