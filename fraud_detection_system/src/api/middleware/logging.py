"""
Request Logging Middleware for Fraud Detection API
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.logger import logger
from src.api.core.config import settings


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests and responses.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
        
        Returns:
            Response object
        """
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        if settings.LOG_REQUESTS:
            logger.info(
                f"[{request_id}] → {request.method} {request.url.path} "
                f"| Client: {request.client.host if request.client else 'unknown'}"
            )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] ✗ {request.method} {request.url.path} "
                f"| Error: {str(e)} | Time: {process_time:.2f}ms"
            )
            raise
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
        
        # Log response
        if settings.LOG_RESPONSES:
            status_icon = "✓" if response.status_code < 400 else "✗"
            logger.info(
                f"[{request_id}] {status_icon} {request.method} {request.url.path} "
                f"| Status: {response.status_code} | Time: {process_time:.2f}ms"
            )
        
        return response