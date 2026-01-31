"""
Health Check Routes for Fraud Detection API
"""

from fastapi import APIRouter
from datetime import datetime

from src.api.schemas.response import HealthResponse
from src.api.core.config import settings
from src.api.routes.prediction import get_prediction_pipeline
from src.logger import logger


# Create router
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API health and model status"
)
async def health_check() -> HealthResponse:
    """
    Check API health status.
    
    Returns:
        Health status information
    """
    try:
        pipeline = get_prediction_pipeline()
        model_loaded = pipeline._is_initialized
        model_version = pipeline.model_version if model_loaded else None
    except Exception:
        model_loaded = False
        model_version = None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_loaded=model_loaded,
        model_version=model_version,
        timestamp=datetime.now().isoformat()
    )


@router.get(
    "/",
    response_model=dict,
    summary="Root Endpoint",
    description="API welcome message"
)
async def root() -> dict:
    """
    Root endpoint with welcome message.
    
    Returns:
        Welcome message
    """
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": settings.DOCS_URL,
        "health": "/health"
    }


@router.get(
    "/ready",
    response_model=dict,
    summary="Readiness Check",
    description="Check if API is ready to accept requests"
)
async def readiness_check() -> dict:
    """
    Check if API is ready.
    
    Returns:
        Readiness status
    """
    try:
        pipeline = get_prediction_pipeline()
        is_ready = pipeline._is_initialized
    except Exception:
        is_ready = False
    
    return {
        "ready": is_ready,
        "timestamp": datetime.now().isoformat()
    }


@router.get(
    "/live",
    response_model=dict,
    summary="Liveness Check",
    description="Check if API is alive"
)
async def liveness_check() -> dict:
    """
    Check if API is alive.
    
    Returns:
        Liveness status
    """
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }