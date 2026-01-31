"""
Authentication Middleware for Fraud Detection API
"""

from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from src.api.core.config import settings
from src.logger import logger


# API Key Header
api_key_header = APIKeyHeader(
    name=settings.API_KEY_HEADER,
    auto_error=False
)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Verify API key from request header.
    
    Args:
        api_key: API key from request header
    
    Returns:
        Verified API key
    
    Raises:
        HTTPException: If API key is invalid
    """
    # Skip auth if disabled
    if not settings.ENABLE_AUTH:
        return "auth_disabled"
    
    if api_key is None:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Missing API Key. Please provide X-API-Key header."
        )
    
    if api_key not in settings.API_KEYS:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    
    return api_key


def get_api_key_optional(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[str]:
    """
    Get API key without raising exception.
    Used for endpoints that support optional auth.
    
    Args:
        api_key: API key from request header
    
    Returns:
        API key or None
    """
    if not settings.ENABLE_AUTH:
        return "auth_disabled"
    
    if api_key and api_key in settings.API_KEYS:
        return api_key
    
    return None