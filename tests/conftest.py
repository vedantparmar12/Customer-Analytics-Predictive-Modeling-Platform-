"""
Test configuration and fixtures
"""

import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set test environment
os.environ["TESTING"] = "True"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"

from api.main import app
from api.cache_manager import AdvancedCacheManager, RedisClusterManager
from api.auth import JWTAuth, APIKeyAuth
from api.ml_serving import MLModelManager


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
async def cache_manager():
    """Create test cache manager"""
    redis_manager = RedisClusterManager(host="localhost", port=6379, db=1)  # Use test DB
    cache_manager = AdvancedCacheManager(redis_manager)
    
    try:
        await cache_manager.initialize()
        yield cache_manager
    finally:
        await cache_manager.redis_manager.close()


@pytest.fixture
def jwt_auth():
    """Create JWT auth instance for testing"""
    return JWTAuth()


@pytest.fixture
async def api_key_auth(cache_manager):
    """Create API key auth instance for testing"""
    return APIKeyAuth(cache_manager)


@pytest.fixture
async def ml_model_manager(cache_manager):
    """Create ML model manager for testing"""
    return MLModelManager(cache_manager)


@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return {
        "customer_id": "test_customer_123",
        "recency_days": 30.0,
        "frequency": 5,
        "monetary_value": 250.0,
        "customer_lifetime_days": 365,
        "total_orders": 8,
        "avg_order_value": 75.0,
        "customer_state": "SP",
        "preferred_payment_type": "credit_card"
    }


@pytest.fixture
def sample_churn_request(sample_customer_data):
    """Sample churn prediction request"""
    return {
        "customers": [sample_customer_data],
        "return_probabilities": True
    }


@pytest.fixture
def sample_recommendation_request():
    """Sample recommendation request"""
    return {
        "customer_id": "test_customer_123",
        "n_recommendations": 5,
        "recommendation_type": "collaborative"
    }


@pytest.fixture
def sample_batch_request(sample_customer_data):
    """Sample batch prediction request"""
    return {
        "customers": [sample_customer_data] * 10,  # 10 customers
        "include_churn": True,
        "include_segmentation": True,
        "include_recommendations": False
    }


@pytest.fixture
def valid_user_credentials():
    """Valid user credentials for testing"""
    return {
        "username": "admin",
        "password": "admin123"
    }


@pytest.fixture
def test_api_key():
    """Test API key"""
    return "ak_test_987654321"


@pytest.fixture
async def authenticated_headers(client, valid_user_credentials):
    """Get authentication headers"""
    # Login to get token
    response = client.post("/auth/token", data=valid_user_credentials)
    assert response.status_code == 200
    
    token_data = response.json()
    access_token = token_data["access_token"]
    
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def api_key_headers(test_api_key):
    """Get API key headers"""
    return {"X-API-Key": test_api_key}


# Mock data fixtures
@pytest.fixture
def mock_model():
    """Mock ML model for testing"""
    class MockModel:
        def predict(self, X):
            return [0, 1, 0, 1, 0][:len(X)]
        
        def predict_proba(self, X):
            return [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]][:len(X)]
    
    return MockModel()


@pytest.fixture
def mock_redis_data():
    """Mock Redis data for testing"""
    return {
        "churn_pred:test_customer_123": {
            "prediction": 0,
            "probability": 0.25,
            "timestamp": "2023-01-01T00:00:00"
        },
        "recommendations:test_customer_123:collaborative": {
            "customer_id": "test_customer_123",
            "recommendations": [
                {"product_id": "PROD_0001", "score": 0.9, "category": "Category_1"}
            ],
            "recommendation_type": "collaborative",
            "timestamp": "2023-01-01T00:00:00"
        }
    }


# Test database cleanup
@pytest.fixture(autouse=True)
async def cleanup_test_data(cache_manager):
    """Clean up test data after each test"""
    yield
    
    # Clean up test patterns
    if cache_manager and cache_manager.redis_client:
        test_patterns = [
            "test_*",
            "churn_pred:test_*",
            "recommendations:test_*",
            "batch_job:*",
            "api_key:ak_test_*"
        ]
        
        for pattern in test_patterns:
            try:
                await cache_manager.invalidate_pattern(pattern)
            except:
                pass  # Ignore cleanup errors


# Performance testing fixtures
@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing"""
    import random
    
    customers = []
    for i in range(1000):  # 1000 customers
        customers.append({
            "customer_id": f"perf_test_customer_{i}",
            "recency_days": random.uniform(1, 365),
            "frequency": random.randint(1, 20),
            "monetary_value": random.uniform(10, 1000),
            "customer_lifetime_days": random.randint(30, 1000),
            "total_orders": random.randint(1, 50),
            "avg_order_value": random.uniform(20, 200)
        })
    
    return customers


# Integration test fixtures
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests"""
    return {
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", 6379)),
        "api_timeout": 30,
        "batch_size": 100,
        "max_workers": 4
    }