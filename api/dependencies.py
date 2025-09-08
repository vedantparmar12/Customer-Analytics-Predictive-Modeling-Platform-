"""
Advanced Dependency Injection System for FastAPI ML Platform
"""

import os
import logging
from typing import Optional, Dict, Any, Generator
from functools import lru_cache
from datetime import datetime

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis

from .cache_manager import AdvancedCacheManager, RedisClusterManager
from .auth import JWTAuth, APIKeyAuth


logger = logging.getLogger(__name__)


class DatabaseDependency:
    """Database connection dependency"""
    
    def __init__(self):
        self._connection = None
    
    async def get_connection(self):
        """Get database connection (placeholder for actual DB)"""
        # In a real implementation, this would return a DB connection
        # For now, we'll simulate with a dict
        if not self._connection:
            self._connection = {"status": "connected", "type": "postgresql"}
        return self._connection


class CacheDependency:
    """Cache manager dependency"""
    
    def __init__(self):
        self._cache_manager: Optional[AdvancedCacheManager] = None
        self._redis_manager: Optional[RedisClusterManager] = None
    
    async def get_cache_manager(self) -> AdvancedCacheManager:
        """Get cache manager instance"""
        if not self._cache_manager:
            # Initialize Redis cluster manager
            self._redis_manager = RedisClusterManager(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD'),
                cluster_mode=os.getenv('REDIS_CLUSTER_MODE', 'false').lower() == 'true'
            )
            
            # Initialize cache manager
            self._cache_manager = AdvancedCacheManager(self._redis_manager)
            await self._cache_manager.initialize()
            
            logger.info("Cache manager initialized successfully")
        
        return self._cache_manager
    
    async def close(self):
        """Close cache connections"""
        if self._redis_manager:
            await self._redis_manager.close()


class ModelDependency:
    """ML model loading and management dependency"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def get_model(self, model_name: str) -> Any:
        """Get loaded model by name"""
        if model_name not in self._models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{model_name}' not loaded"
            )
        return self._models[model_name]
    
    async def load_model(self, model_name: str, model_path: str) -> bool:
        """Load model from path"""
        try:
            import joblib
            model = joblib.load(model_path)
            self._models[model_name] = model
            self._model_metadata[model_name] = {
                "path": model_path,
                "loaded_at": datetime.utcnow().isoformat(),
                "type": type(model).__name__
            }
            logger.info(f"Model '{model_name}' loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models"""
        return {
            name: {
                **metadata,
                "loaded": True,
                "model_class": type(model).__name__
            }
            for name, (model, metadata) in zip(
                self._models.keys(),
                self._model_metadata.values()
            )
        }


class AuthDependency:
    """Authentication and authorization dependencies"""
    
    def __init__(self):
        self.jwt_auth = JWTAuth()
        self.api_key_auth = APIKeyAuth()
        self.bearer_scheme = HTTPBearer()
    
    async def get_current_user(self, 
                              credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
        """Get current authenticated user from JWT token"""
        try:
            token = credentials.credentials
            payload = self.jwt_auth.verify_token(token)
            
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return payload
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def get_api_key_user(self, request: Request) -> Dict[str, Any]:
        """Get user information from API key"""
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        user_info = await self.api_key_auth.verify_api_key(api_key)
        
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        return user_info
    
    async def get_optional_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get user information if authentication is provided (optional)"""
        # Try JWT token first
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = self.jwt_auth.verify_token(token)
                if payload:
                    return payload
            except:
                pass
        
        # Try API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            try:
                user_info = await self.api_key_auth.verify_api_key(api_key)
                if user_info:
                    return user_info
            except:
                pass
        
        return None


class ConfigDependency:
    """Configuration dependency"""
    
    @lru_cache()
    def get_settings(self) -> Dict[str, Any]:
        """Get application settings"""
        return {
            "api_title": os.getenv("API_TITLE", "E-commerce ML API"),
            "api_version": os.getenv("API_VERSION", "1.0.0"),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
            "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            "rate_limit_window": int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            "cache_default_ttl": int(os.getenv("CACHE_DEFAULT_TTL", "3600")),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }


class HealthDependency:
    """Health check dependency"""
    
    def __init__(self, 
                 cache_dep: CacheDependency, 
                 model_dep: ModelDependency,
                 db_dep: DatabaseDependency):
        self.cache_dep = cache_dep
        self.model_dep = model_dep
        self.db_dep = db_dep
    
    async def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Check cache
        try:
            cache_manager = await self.cache_dep.get_cache_manager()
            cache_stats = await cache_manager.get_cache_stats()
            health_status["services"]["cache"] = {
                "status": "healthy",
                "stats": cache_stats
            }
        except Exception as e:
            health_status["services"]["cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check models
        try:
            loaded_models = self.model_dep.get_loaded_models()
            health_status["services"]["models"] = {
                "status": "healthy" if loaded_models else "degraded",
                "loaded_models": loaded_models
            }
            
            if not loaded_models:
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["services"]["models"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check database
        try:
            db_connection = await self.db_dep.get_connection()
            health_status["services"]["database"] = {
                "status": "healthy",
                "connection": db_connection["status"]
            }
        except Exception as e:
            health_status["services"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status


# Global dependency instances
database_dep = DatabaseDependency()
cache_dep = CacheDependency()
model_dep = ModelDependency()
auth_dep = AuthDependency()
config_dep = ConfigDependency()
health_dep = HealthDependency(cache_dep, model_dep, database_dep)


# Dependency functions for FastAPI
async def get_cache_manager() -> AdvancedCacheManager:
    """FastAPI dependency for cache manager"""
    from .main import cache_manager
    if not cache_manager:
        raise HTTPException(status_code=503, detail="Cache service not available")
    return cache_manager


async def get_database() -> Any:
    """FastAPI dependency for database connection"""
    return await database_dep.get_connection()


async def get_model(model_name: str = "churn") -> Any:
    """FastAPI dependency for ML models"""
    return await model_dep.get_model(model_name)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> Dict[str, Any]:
    """FastAPI dependency for JWT authenticated user"""
    from .main import jwt_auth
    try:
        token = credentials.credentials
        payload = jwt_auth.verify_token(token)
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_api_key_user(request: Request) -> Dict[str, Any]:
    """FastAPI dependency for API key authenticated user"""
    from .main import api_key_auth
    
    if not api_key_auth:
        raise HTTPException(status_code=503, detail="API key service not available")
    
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    user_info = await api_key_auth.verify_api_key(api_key)
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return user_info


async def get_optional_user(request: Request) -> Optional[Dict[str, Any]]:
    """FastAPI dependency for optional authentication"""
    from .main import jwt_auth, api_key_auth
    
    # Try JWT token first
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt_auth.verify_token(token)
            if payload:
                return payload
        except:
            pass
    
    # Try API key
    if api_key_auth:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            try:
                user_info = await api_key_auth.verify_api_key(api_key)
                if user_info:
                    return user_info
            except:
                pass
    
    return None


def get_settings() -> Dict[str, Any]:
    """FastAPI dependency for application settings"""
    return config_dep.get_settings()


async def get_health_status() -> Dict[str, Any]:
    """FastAPI dependency for health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    from .main import cache_manager, ml_model_manager
    
    # Check cache
    try:
        if cache_manager:
            cache_stats = await cache_manager.get_cache_stats()
            health_status["services"]["cache"] = {
                "status": "healthy",
                "stats": cache_stats
            }
        else:
            health_status["services"]["cache"] = {
                "status": "unavailable",
                "error": "Cache manager not initialized"
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check models
    try:
        if ml_model_manager:
            loaded_models = len(ml_model_manager.models)
            health_status["services"]["models"] = {
                "status": "healthy" if loaded_models > 0 else "degraded",
                "loaded_models_count": loaded_models
            }
            
            if loaded_models == 0:
                health_status["status"] = "degraded"
        else:
            health_status["services"]["models"] = {
                "status": "unavailable",
                "error": "Model manager not initialized"
            }
            health_status["status"] = "degraded"
                
    except Exception as e:
        health_status["services"]["models"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check database (placeholder)
    try:
        health_status["services"]["database"] = {
            "status": "healthy",
            "connection": "simulated"
        }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status