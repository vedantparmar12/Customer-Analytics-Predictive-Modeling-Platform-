"""
Advanced Middleware System for FastAPI ML Serving Platform
"""

import time
import json
import uuid
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import prometheus_client

from .cache_manager import AdvancedCacheManager


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging"""
    
    def __init__(self, app: ASGIApp, logger_name: str = "api_requests"):
        super().__init__(app)
        self.request_logger = logging.getLogger(logger_name)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request details
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        self.request_logger.info(
            f"REQUEST_START - ID: {request_id} | "
            f"Method: {request.method} | "
            f"URL: {request.url} | "
            f"Client IP: {client_ip} | "
            f"User Agent: {request.headers.get('user-agent', 'unknown')}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response details
            self.request_logger.info(
                f"REQUEST_COMPLETE - ID: {request_id} | "
                f"Status: {response.status_code} | "
                f"Processing Time: {processing_time:.4f}s | "
                f"Response Size: {response.headers.get('content-length', 'unknown')} bytes"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(processing_time)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            self.request_logger.error(
                f"REQUEST_ERROR - ID: {request_id} | "
                f"Error: {str(e)} | "
                f"Processing Time: {processing_time:.4f}s"
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"X-Request-ID": request_id}
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting detailed API metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
        # Prometheus metrics
        self.request_count = prometheus_client.Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = prometheus_client.Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.request_size = prometheus_client.Histogram(
            'api_request_size_bytes',
            'API request size',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
        
        self.response_size = prometheus_client.Histogram(
            'api_response_size_bytes',
            'API response size',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000]
        )
        
        self.active_requests = prometheus_client.Gauge(
            'api_active_requests',
            'Currently active API requests'
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract endpoint pattern for metrics
        endpoint = self._get_endpoint_pattern(request)
        method = request.method
        
        # Track active requests
        self.active_requests.inc()
        
        start_time = time.time()
        
        try:
            # Get request size
            request_size = 0
            if hasattr(request, 'body'):
                body = await request.body()
                request_size = len(body)
                # Re-wrap body for downstream processing
                request._body = body
            
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration = time.time() - start_time
            status_code = str(response.status_code)
            
            # Get response size
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body)
            elif 'content-length' in response.headers:
                response_size = int(response.headers['content-length'])
            
            # Record metrics
            self.request_count.labels(
                method=method, 
                endpoint=endpoint, 
                status_code=status_code
            ).inc()
            
            self.request_duration.labels(
                method=method, 
                endpoint=endpoint
            ).observe(duration)
            
            if request_size > 0:
                self.request_size.labels(
                    method=method, 
                    endpoint=endpoint
                ).observe(request_size)
            
            if response_size > 0:
                self.response_size.labels(
                    method=method, 
                    endpoint=endpoint
                ).observe(response_size)
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.request_count.labels(
                method=method, 
                endpoint=endpoint, 
                status_code="500"
            ).inc()
            
            raise e
            
        finally:
            # Decrement active requests
            self.active_requests.dec()
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """Extract endpoint pattern from request path"""
        path = request.url.path
        
        # Map common patterns to clean endpoint names
        if path.startswith("/predict/"):
            return f"/predict/{path.split('/')[-1]}"
        elif path.startswith("/model/"):
            return f"/model/{path.split('/')[-1]}"
        elif path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return path
        else:
            return "/other"


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting using Redis"""
    
    def __init__(self, app: ASGIApp, cache_manager: AdvancedCacheManager):
        super().__init__(app)
        self.cache_manager = cache_manager
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_identifier(request)
        
        # Check rate limit
        rate_limit_result = await self.cache_manager.implement_rate_limiting(
            identifier=client_id,
            max_requests=100,  # Default limit
            window_seconds=60  # 1 minute window
        )
        
        if not rate_limit_result["allowed"]:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {rate_limit_result['max_requests']} requests per {rate_limit_result['window_seconds']} seconds",
                    "current_requests": rate_limit_result["current_requests"],
                    "reset_time": rate_limit_result["reset_time"],
                    "request_id": getattr(request.state, 'request_id', 'unknown')
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit_result['max_requests']),
                    "X-RateLimit-Remaining": str(max(0, rate_limit_result['max_requests'] - rate_limit_result['current_requests'])),
                    "X-RateLimit-Reset": str(rate_limit_result['reset_time'])
                }
            )
        
        # Add rate limit headers to successful responses
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit_result['max_requests'])
        response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limit_result['max_requests'] - rate_limit_result['current_requests']))
        response.headers["X-RateLimit-Reset"] = str(rate_limit_result['reset_time'])
        
        return response
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting"""
        # Try API key first (if available)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"


class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for intelligent response caching"""
    
    def __init__(self, app: ASGIApp, cache_manager: AdvancedCacheManager):
        super().__init__(app)
        self.cache_manager = cache_manager
        
        # Define cacheable endpoints and their TTL
        self.cacheable_endpoints = {
            "/predict/recommendations": 1800,  # 30 minutes
            "/predict/segmentation": 3600,     # 1 hour
            "/model/info": 300,                # 5 minutes
            "/health": 60                      # 1 minute
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests for specific endpoints
        if request.method != "GET" or request.url.path not in self.cacheable_endpoints:
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache
        cached_response = await self.cache_manager.get(cache_key)
        if cached_response:
            logger.debug(f"Cache HIT for {request.url.path}")
            return JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers={
                    **cached_response.get("headers", {}),
                    "X-Cache": "HIT",
                    "X-Cache-Timestamp": cached_response["timestamp"]
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if 200 <= response.status_code < 300:
            ttl = self.cacheable_endpoints[request.url.path]
            
            # Read response content
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Parse response content
            try:
                content = json.loads(response_body.decode())
                
                cache_data = {
                    "content": content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self.cache_manager.set(cache_key, cache_data, ttl)
                logger.debug(f"Cache MISS - Cached response for {request.url.path}")
                
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning(f"Could not cache response for {request.url.path} - invalid JSON")
            
            # Recreate response with cache headers
            return JSONResponse(
                content=json.loads(response_body.decode()),
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "X-Cache": "MISS"
                }
            )
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        key_components = [
            "response_cache",
            request.url.path,
            str(sorted(request.query_params.items()))
        ]
        return ":".join(key_components)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors
            request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
            
            logger.error(
                f"Unexpected error in request {request_id}: {str(e)}",
                exc_info=True
            )
            
            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred while processing your request",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"X-Request-ID": request_id}
            )