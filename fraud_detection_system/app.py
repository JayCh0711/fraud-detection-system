"""
Main FastAPI Application for Fraud Detection System

Production-ready API with:
- Single and batch prediction endpoints
- Health checks
- Model information
- Authentication
- Request logging
- CORS support
- Swagger documentation
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.core.config import settings
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.routes import prediction, health, model
from src.logger import logger


# ============== LIFESPAN EVENTS ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Runs on startup and shutdown.
    """
    # Startup
    logger.info(f"{'='*60}")
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"{'='*60}")
    
    # Pre-load model
    try:
        from src.api.routes.prediction import get_prediction_pipeline
        pipeline = get_prediction_pipeline()
        logger.info(f"Model loaded: v{pipeline.model_version}")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {str(e)}")
    
    logger.info(f"API docs available at: http://{settings.HOST}:{settings.PORT}{settings.DOCS_URL}")
    logger.info(f"{'='*60}")
    
    yield
    
    # Shutdown
    logger.info(f"{'='*60}")
    logger.info(f"Shutting down {settings.APP_NAME}")
    logger.info(f"{'='*60}")


# ============== CREATE APPLICATION ==============
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
    lifespan=lifespan
)


# ============== MIDDLEWARE ==============

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Request Logging
app.add_middleware(RequestLoggingMiddleware)


# ============== EXCEPTION HANDLERS ==============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "error_message": exc.detail,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "error_message": "An unexpected error occurred",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


# ============== INCLUDE ROUTERS ==============

# Health routes (no prefix)
app.include_router(health.router)

# API routes with prefix
app.include_router(
    prediction.router,
    prefix=settings.API_PREFIX
)

app.include_router(
    model.router,
    prefix=settings.API_PREFIX
)


# ============== STATIC FILES & TEMPLATES ==============

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


# ============== ADDITIONAL ENDPOINTS ==============

@app.get("/api/v1/status", tags=["Status"])
async def api_status():
    """Get API status and configuration."""
    return {
        "api_name": settings.APP_NAME,
        "api_version": settings.APP_VERSION,
        "api_prefix": settings.API_PREFIX,
        "auth_enabled": settings.ENABLE_AUTH,
        "rate_limit_enabled": settings.RATE_LIMIT_ENABLED,
        "debug_mode": settings.DEBUG,
        "endpoints": {
            "health": "/health",
            "predict_single": f"{settings.API_PREFIX}/predict/single",
            "predict_batch": f"{settings.API_PREFIX}/predict/batch",
            "model_info": f"{settings.API_PREFIX}/model/info",
            "docs": settings.DOCS_URL,
            "redoc": settings.REDOC_URL
        }
    }


# ============== RUN APPLICATION ==============

def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()