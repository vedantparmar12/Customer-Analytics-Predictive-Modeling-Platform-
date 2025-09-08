"""
Tests for Redis cache manager
"""

import pytest
import json
import time
from datetime import datetime

from api.cache_manager import AdvancedCacheManager, RedisClusterManager, CacheConfig


class TestAdvancedCacheManager:
    """Test advanced cache manager functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, cache_manager):
        """Test basic get/set/delete operations"""
        # Test set and get
        key = "test_key"
        value = {"test": "data", "number": 123}
        
        success = await cache_manager.set(key, value, ttl=60)
        assert success is True
        
        retrieved = await cache_manager.get(key)
        assert retrieved == value
        
        # Test exists
        exists = await cache_manager.exists(key)
        assert exists is True
        
        # Test delete
        deleted = await cache_manager.delete(key)
        assert deleted is True
        
        # Verify deletion
        retrieved = await cache_manager.get(key)
        assert retrieved is None
        
        exists = await cache_manager.exists(key)
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_ttl_functionality(self, cache_manager):
        """Test TTL (time-to-live) functionality"""
        key = "test_ttl_key"
        value = "test_value"
        
        # Set with short TTL
        await cache_manager.set(key, value, ttl=1)  # 1 second
        
        # Should exist immediately
        assert await cache_manager.exists(key)
        
        # Check TTL
        ttl = await cache_manager.get_ttl(key)
        assert 0 <= ttl <= 1
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should not exist after expiration
        assert not await cache_manager.exists(key)
        retrieved = await cache_manager.get(key)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_cache_prediction(self, cache_manager):
        """Test prediction caching"""
        customer_id = "test_customer_123"
        prediction_type = "churn"
        prediction_data = {
            "prediction": 1,
            "probability": 0.85,
            "confidence": 0.92
        }
        
        # Cache prediction
        success = await cache_manager.cache_prediction(
            customer_id, prediction_type, prediction_data
        )
        assert success is True
        
        # Retrieve cached prediction
        cached = await cache_manager.get_cached_prediction(customer_id, prediction_type)
        assert cached is not None
        assert cached["prediction"] == prediction_data
        assert cached["customer_id"] == customer_id
        assert cached["type"] == prediction_type
        assert "timestamp" in cached
    
    @pytest.mark.asyncio
    async def test_batch_prediction_caching(self, cache_manager):
        """Test batch prediction caching"""
        batch_id = "test_batch_123"
        predictions = [
            {"customer_id": "cust_1", "prediction": 0, "probability": 0.2},
            {"customer_id": "cust_2", "prediction": 1, "probability": 0.8},
            {"customer_id": "cust_3", "prediction": 0, "probability": 0.3}
        ]
        
        # Cache batch predictions
        success = await cache_manager.cache_batch_predictions(batch_id, predictions)
        assert success is True
        
        # Retrieve cached batch
        cached = await cache_manager.get_cached_batch_predictions(batch_id)
        assert cached is not None
        assert cached["predictions"] == predictions
        assert cached["batch_id"] == batch_id
        assert cached["count"] == 3
        assert "timestamp" in cached
    
    @pytest.mark.asyncio
    async def test_model_artifacts_caching(self, cache_manager):
        """Test model artifacts caching"""
        model_name = "test_model"
        model_data = {
            "version": "1.0.0",
            "algorithm": "XGBoost",
            "performance": {"accuracy": 0.92, "precision": 0.89}
        }
        
        # Cache model artifacts
        success = await cache_manager.cache_model_artifacts(model_name, model_data)
        assert success is True
        
        # Verify caching
        key = cache_manager._generate_cache_key(CacheConfig.MODEL_PREFIX, model_name)
        cached = await cache_manager.get(key)
        
        assert cached is not None
        assert cached["model_metadata"] == model_data
        assert cached["model_name"] == model_name
        assert "timestamp" in cached
    
    @pytest.mark.asyncio
    async def test_pattern_invalidation(self, cache_manager):
        """Test pattern-based cache invalidation"""
        # Set multiple keys with similar patterns
        keys = [
            "test_pattern:user_1",
            "test_pattern:user_2", 
            "test_pattern:user_3",
            "other_pattern:data_1"
        ]
        
        for key in keys:
            await cache_manager.set(key, f"data_for_{key}")
        
        # Verify all keys exist
        for key in keys:
            assert await cache_manager.exists(key)
        
        # Invalidate pattern
        deleted_count = await cache_manager.invalidate_pattern("test_pattern:*")
        assert deleted_count == 3
        
        # Verify pattern keys are deleted
        for key in keys[:3]:
            assert not await cache_manager.exists(key)
        
        # Verify other pattern still exists
        assert await cache_manager.exists(keys[3])
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, cache_manager):
        """Test Redis-based rate limiting"""
        identifier = "test_user_123"
        max_requests = 5
        window_seconds = 10
        
        # Make requests within limit
        for i in range(max_requests):
            result = await cache_manager.implement_rate_limiting(
                identifier, max_requests, window_seconds
            )
            assert result["allowed"] is True
            assert result["current_requests"] == i + 1
        
        # Exceed rate limit
        result = await cache_manager.implement_rate_limiting(
            identifier, max_requests, window_seconds
        )
        assert result["allowed"] is False
        assert result["current_requests"] == max_requests
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics retrieval"""
        # Perform some cache operations
        await cache_manager.set("stats_test_1", "value1")
        await cache_manager.set("stats_test_2", "value2")
        await cache_manager.get("stats_test_1")
        await cache_manager.get("stats_test_1")  # Cache hit
        await cache_manager.get("nonexistent")  # Cache miss
        
        # Get stats
        stats = await cache_manager.get_cache_stats()
        
        assert "redis_version" in stats
        assert "connected_clients" in stats
        assert "used_memory" in stats
        assert "timestamp" in stats
        
        # Stats should be valid
        if "keyspace_hits" in stats and "keyspace_misses" in stats:
            assert isinstance(stats["hit_rate"], (int, float))
            assert 0 <= stats["hit_rate"] <= 100
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache_manager):
        """Test cache key generation"""
        # Test basic key generation
        key1 = cache_manager._generate_cache_key("prefix", "arg1", "arg2")
        key2 = cache_manager._generate_cache_key("prefix", "arg1", "arg2")
        assert key1 == key2  # Same inputs should generate same key
        
        # Test with kwargs
        key3 = cache_manager._generate_cache_key("prefix", "arg1", param1="value1", param2="value2")
        key4 = cache_manager._generate_cache_key("prefix", "arg1", param2="value2", param1="value1")
        assert key3 == key4  # Order shouldn't matter for kwargs
        
        # Test different inputs generate different keys
        key5 = cache_manager._generate_cache_key("prefix", "arg1", "arg3")
        assert key1 != key5
        
        # Test long key hashing
        long_args = ["very_long_argument"] * 50
        long_key = cache_manager._generate_cache_key("prefix", *long_args)
        assert len(long_key) < 300  # Should be hashed to shorter length
        assert "hash:" in long_key
    
    @pytest.mark.asyncio
    async def test_json_serialization(self, cache_manager):
        """Test JSON serialization with numpy arrays"""
        import numpy as np
        
        # Test with numpy array
        data = {
            "array": np.array([1, 2, 3, 4, 5]),
            "float": np.float64(3.14),
            "int": np.int32(42),
            "datetime": datetime.utcnow(),
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        key = "serialization_test"
        success = await cache_manager.set(key, data)
        assert success is True
        
        retrieved = await cache_manager.get(key)
        assert retrieved is not None
        
        # Verify data integrity (arrays become lists)
        assert retrieved["array"] == [1, 2, 3, 4, 5]
        assert retrieved["float"] == 3.14
        assert retrieved["int"] == 42
        assert isinstance(retrieved["datetime"], str)  # Datetime becomes ISO string
        assert retrieved["list"] == [1, 2, 3]
        assert retrieved["dict"] == {"nested": "value"}
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cache_manager):
        """Test error handling in cache operations"""
        # Test with invalid JSON data
        key = "error_test"
        
        # This should handle errors gracefully
        success = await cache_manager.set(key, {"valid": "data"})
        assert success is True
        
        # Test getting non-existent key (should return None, not error)
        result = await cache_manager.get("definitely_does_not_exist")
        assert result is None
        
        # Test deleting non-existent key
        deleted = await cache_manager.delete("definitely_does_not_exist")
        # Should return False but not error
        assert isinstance(deleted, bool)