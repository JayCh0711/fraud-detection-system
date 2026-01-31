"""
API Configuration for Fraud Detection System
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class APISettings(BaseSettings):
    """API Configuration Settings"""
    
    # Application Info
    APP_NAME: str = "Fraud Detection API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Real-time Fraud Detection API for BFSI"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False
    
    # API Settings
    API_PREFIX: str = "/api/v1"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"
    
    # Security
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = ["fraud-detection-api-key-2024"]  # In production, use env vars
    ENABLE_AUTH: bool = True
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Logging
    LOG_REQUESTS: bool = True
    LOG_RESPONSES: bool = True
    
    # Model Settings
    MODEL_PATH: str = "artifacts/production_model"
    AUTO_RELOAD_MODEL: bool = False
    
    # Prediction Settings
    DEFAULT_THRESHOLD: float = 0.5
    MAX_BATCH_SIZE: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> APISettings:
    """Get cached settings instance."""
    return APISettings()


settings = get_settings()