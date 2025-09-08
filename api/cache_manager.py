"""
Advanced Redis Cache Manager for ML Model Predictions and API Optimization
"""

import json
import time
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import logging

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError
import numpy as np


logger = logging.getLogger(__name__)


class CacheConfig:
    """Cache configuration settings"""
    DEFAULT_TTL = 3600  # 1 hour
    PREDICTION_TTL = 1800  # 30 minutes
    MODEL_TTL = 86400  # 24 hours
    DB_QUERY_TTL = 600  # 10 minutes
    
    # Cache key prefixes
    CHURN_PREFIX = "churn_pred"
    RECOMMENDATION_PREFIX = "recommendations"
    SEGMENTATION_PREFIX = "segmentation"
    MODEL_PREFIX = "model_cache"
    QUERY_PREFIX = "db_query"
    BATCH_PREFIX = "batch_pred"
    
    # Rate limiting
    RATE_LIMIT_WINDOW = 60  # 1 minute
    RATE_LIMIT_MAX_REQUESTS = 100


class RedisClusterManager:
    """Advanced Redis cache manager with clustering and optimization"""
    
    def __init__(self, 
                 host: str = "redis", 
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 cluster_mode: bool = False):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.cluster_mode = cluster_mode
        self.redis_client: Optional[Redis] = None
        self.connection_pool = None
        
    async def initialize(self):
        """Initialize Redis connection with connection pooling"""
        try:
            if self.cluster_mode:
                # Redis Cluster configuration
                from redis.asyncio.cluster import RedisCluster
                startup_nodes = [{"host": self.host, "port": self.port}]
                self.redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=True,
                    skip_full_coverage_check=True,
                    password=self.password
                )
            else:
                # Single Redis instance with connection pooling
                self.connection_pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    max_connections=20,
                    retry_on_timeout=True
                )
                self.redis_client = Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()


class AdvancedCacheManager:
    """Advanced caching system with multiple strategies"""
    
    def __init__(self, redis_manager: RedisClusterManager):
        self.redis_manager = redis_manager
        self.redis_client = None
        
    async def initialize(self):
        """Initialize cache manager"""
        await self.redis_manager.initialize()
        self.redis_client = self.redis_manager.redis_client
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate consistent cache key from parameters"""
        key_data = f"{prefix}:{':'.join(map(str, args))}"
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_data += f":{':'.join(f'{k}={v}' for k, v in sorted_kwargs)}"
        
        # Hash long keys to prevent Redis key length issues
        if len(key_data) > 250:
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            key_data = f"{prefix}:hash:{key_hash}"
        
        return key_data
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with error handling"""
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = CacheConfig.DEFAULT_TTL) -> bool:
        """Set value in cache with TTL"""
        try:
            serialized_value = json.dumps(value, default=self._json_serializer)
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
        except (RedisError, TypeError) as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.redis_client.delete(key)
            return bool(result)
        except RedisError as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(await self.redis_client.exists(key))
        except RedisError:
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key"""
        try:
            return await self.redis_client.ttl(key)
        except RedisError:
            return -1
    
    async def cache_prediction(self, 
                              customer_id: str, 
                              prediction_type: str, 
                              prediction_data: Dict[str, Any],
                              ttl: int = CacheConfig.PREDICTION_TTL) -> bool:
        """Cache ML prediction results"""
        cache_key = self._generate_cache_key(
            f"{prediction_type}_pred", 
            customer_id
        )
        
        cache_data = {
            "prediction": prediction_data,
            "timestamp": datetime.utcnow().isoformat(),
            "customer_id": customer_id,
            "type": prediction_type
        }
        
        return await self.set(cache_key, cache_data, ttl)
    
    async def get_cached_prediction(self, 
                                   customer_id: str, 
                                   prediction_type: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction for customer"""
        cache_key = self._generate_cache_key(
            f"{prediction_type}_pred", 
            customer_id
        )
        return await self.get(cache_key)
    
    async def cache_batch_predictions(self, 
                                     batch_id: str, 
                                     predictions: List[Dict[str, Any]],
                                     ttl: int = CacheConfig.PREDICTION_TTL) -> bool:
        """Cache batch prediction results"""
        cache_key = self._generate_cache_key(CacheConfig.BATCH_PREFIX, batch_id)
        
        cache_data = {
            "predictions": predictions,
            "timestamp": datetime.utcnow().isoformat(),
            "batch_id": batch_id,
            "count": len(predictions)
        }
        
        return await self.set(cache_key, cache_data, ttl)
    
    async def get_cached_batch_predictions(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get cached batch predictions"""
        cache_key = self._generate_cache_key(CacheConfig.BATCH_PREFIX, batch_id)
        return await self.get(cache_key)
    
    async def cache_model_artifacts(self, 
                                   model_name: str, 
                                   model_data: Dict[str, Any],
                                   ttl: int = CacheConfig.MODEL_TTL) -> bool:
        """Cache model artifacts and metadata"""
        cache_key = self._generate_cache_key(CacheConfig.MODEL_PREFIX, model_name)
        
        cache_data = {
            "model_metadata": model_data,
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name
        }
        
        return await self.set(cache_key, cache_data, ttl)
    
    async def warm_cache(self, 
                        cache_warming_tasks: List[Callable]) -> Dict[str, bool]:
        """Warm up cache with commonly accessed data"""
        results = {}
        
        for task in cache_warming_tasks:
            try:
                task_name = task.__name__
                await task()
                results[task_name] = True
                logger.info(f"Cache warming completed for {task_name}")
            except Exception as e:
                results[task_name] = False
                logger.error(f"Cache warming failed for {task_name}: {e}")
        
        return results
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except RedisError as e:
            logger.error(f"Pattern invalidation error: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics and health metrics"""
        try:
            info = await self.redis_client.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0), 
                    info.get("keyspace_misses", 0)
                ),
                "timestamp": datetime.utcnow().isoformat()
            }
        except RedisError as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def implement_rate_limiting(self, 
                                     identifier: str, 
                                     max_requests: int = CacheConfig.RATE_LIMIT_MAX_REQUESTS,
                                     window_seconds: int = CacheConfig.RATE_LIMIT_WINDOW) -> Dict[str, Any]:
        """Implement rate limiting using Redis"""
        key = f"rate_limit:{identifier}"
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            is_allowed = current_requests < max_requests
            
            return {
                "allowed": is_allowed,
                "current_requests": current_requests,
                "max_requests": max_requests,
                "window_seconds": window_seconds,
                "reset_time": current_time + window_seconds
            }
            
        except RedisError as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if Redis is unavailable
            return {"allowed": True, "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays and other types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Decorator for caching function results
def cached_result(cache_manager: AdvancedCacheManager, 
                 prefix: str, 
                 ttl: int = CacheConfig.DEFAULT_TTL):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and parameters
            cache_key = cache_manager._generate_cache_key(prefix, func.__name__, *args, **kwargs)
            
            # Try to get from cache first
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator