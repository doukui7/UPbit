from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import os
import json
from dotenv import load_dotenv

load_dotenv()

class UpbitConfig(BaseModel):
    access_key: str = Field(default_factory=lambda: os.getenv("UPBIT_ACCESS_KEY", ""))
    secret_key: str = Field(default_factory=lambda: os.getenv("UPBIT_SECRET_KEY", ""))

class KISConfig(BaseModel):
    app_key: str = Field(default_factory=lambda: os.getenv("KIS_APP_KEY", ""))
    app_secret: str = Field(default_factory=lambda: os.getenv("KIS_APP_SECRET", ""))
    account_no: str = Field(default_factory=lambda: os.getenv("KIS_ACCOUNT_NO", ""))
    acnt_prdt_cd: str = Field(default="01")

class AppSettings(BaseModel):
    upbit: UpbitConfig = Field(default_factory=UpbitConfig)
    kis: KISConfig = Field(default_factory=KISConfig)
    telegram_token: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN"))
    telegram_chat_id: Optional[str] = Field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    
    # 전략 공통 설정
    start_date: str = "2020-01-01"
    initial_cap: float = 10000000.0
    
    # 포트폴리오 데이터 (동적 로드용)
    portfolio: List[Dict] = []
    aux_portfolio: List[Dict] = []

    class Config:
        env_prefix = "APP_"

def get_settings() -> AppSettings:
    """Load settings from env, user_config.json, or streamlit secrets."""
    # 기초 설정 생성
    settings = AppSettings()
    
    # user_config.json 로드 (오버라이드)
    if os.path.exists("user_config.json"):
        try:
            with open("user_config.json", "r", encoding="utf-8") as f:
                disk_config = json.load(f)
                # 간소화된 병합 로직
                for k, v in disk_config.items():
                    if hasattr(settings, k):
                        setattr(settings, k, v)
        except Exception as e:
            print(f"Error loading user_config.json: {e}")
            
    return settings
