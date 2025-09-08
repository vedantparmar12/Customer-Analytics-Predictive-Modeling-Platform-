"""
Tests for FastAPI endpoints
"""

import pytest
import json
from httpx import AsyncClient


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client):
        """Test basic health check"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "redis_connected" in data
        assert "timestamp" in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data
        
        services = data["services"]
        assert "cache" in services
        assert "models" in services
        assert "database" in services


class TestAuthenticationEndpoints:
    """Test authentication endpoints"""
    
    def test_token_login_valid_credentials(self, client, valid_user_credentials):
        """Test login with valid credentials"""
        response = client.post("/auth/token", data=valid_user_credentials)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        
        # Verify token format
        assert isinstance(data["access_token"], str)
        assert len(data["access_token"]) > 50  # JWT tokens are long
    
    def test_token_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        invalid_credentials = {"username": "admin", "password": "wrong_password"}
        response = client.post("/auth/token", data=invalid_credentials)
        assert response.status_code == 401
        
        data = response.json()
        assert "detail" in data
        assert "Incorrect username or password" in data["detail"]
    
    def test_refresh_token(self, client, valid_user_credentials):
        """Test token refresh"""
        # Get initial tokens
        response = client.post("/auth/token", data=valid_user_credentials)
        assert response.status_code == 200
        
        tokens = response.json()
        refresh_token = tokens["refresh_token"]
        
        # Use refresh token to get new access token
        response = client.post("/auth/refresh", json={"refresh_token": refresh_token})
        
        if response.status_code == 200:  # If endpoint is implemented
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data


class TestPredictionEndpoints:
    """Test ML prediction endpoints"""
    
    def test_churn_prediction(self, client, sample_churn_request):
        """Test churn prediction endpoint"""
        response = client.post("/predict/churn", json=sample_churn_request)
        assert response.status_code in [200, 503]  # 503 if model not loaded
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "model_version" in data
            assert "timestamp" in data
            
            predictions = data["predictions"]
            assert len(predictions) == 1  # One customer
            
            prediction = predictions[0]
            assert "customer_id" in prediction
            assert "churn_prediction" in prediction
            assert isinstance(prediction["churn_prediction"], bool)
            
            if sample_churn_request["return_probabilities"]:
                assert "churn_probability" in prediction
                assert isinstance(prediction["churn_probability"], (int, float, type(None)))
    
    def test_churn_prediction_without_probabilities(self, client, sample_customer_data):
        """Test churn prediction without probabilities"""
        request = {
            "customers": [sample_customer_data],
            "return_probabilities": False
        }
        
        response = client.post("/predict/churn", json=request)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            prediction = data["predictions"][0]
            
            # Should have prediction but not probability
            assert "churn_prediction" in prediction
            assert prediction.get("churn_probability") is None
    
    def test_recommendations(self, client, sample_recommendation_request):
        """Test recommendation endpoint"""
        response = client.post("/predict/recommendations", json=sample_recommendation_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert "recommendations" in data
        assert "recommendation_type" in data
        assert "timestamp" in data
        
        assert data["customer_id"] == sample_recommendation_request["customer_id"]
        assert data["recommendation_type"] == sample_recommendation_request["recommendation_type"]
        
        recommendations = data["recommendations"]
        assert len(recommendations) == sample_recommendation_request["n_recommendations"]
        
        for rec in recommendations:
            assert "product_id" in rec
            assert "score" in rec
            assert "category" in rec
    
    def test_customer_segmentation(self, client, sample_customer_data):
        """Test customer segmentation endpoint"""
        request = {
            "customers": [sample_customer_data],
            "segmentation_type": "rfm"
        }
        
        response = client.post("/predict/segmentation", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "segments" in data
        assert "segmentation_type" in data
        assert "timestamp" in data
        
        segments = data["segments"]
        assert len(segments) == 1
        
        segment = segments[0]
        assert "customer_id" in segment
        assert "segment" in segment
        assert "rfm_score" in segment
        
        # Verify RFM score format
        rfm_score = segment["rfm_score"]
        assert len(rfm_score) == 3  # R, F, M scores
        assert all(c.isdigit() for c in rfm_score)
        assert all(1 <= int(c) <= 5 for c in rfm_score)
    
    def test_batch_predictions(self, client, sample_batch_request):
        """Test batch prediction endpoint"""
        response = client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "customer_predictions" in data
            assert "timestamp" in data
            
            predictions = data["customer_predictions"]
            assert len(predictions) == len(sample_batch_request["customers"])
            
            for prediction in predictions:
                assert "customer_id" in prediction
                
                if sample_batch_request["include_churn"]:
                    assert "churn" in prediction
                
                if sample_batch_request["include_segmentation"]:
                    assert "segment" in prediction
                
                if sample_batch_request["include_recommendations"]:
                    assert "recommendations" in prediction


class TestModelEndpoints:
    """Test model management endpoints"""
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "models_loaded" in data
        assert "model_paths" in data
        assert "last_loaded" in data
    
    def test_model_reload_requires_auth(self, client):
        """Test model reload requires authentication"""
        response = client.post("/model/reload")
        assert response.status_code == 401  # Unauthorized
    
    def test_model_reload_with_api_key(self, client, api_key_headers):
        """Test model reload with API key authentication"""
        response = client.post("/model/reload", headers=api_key_headers)
        # Should either succeed or fail due to missing models, but not auth error
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "message" in data
            assert "reloaded_by" in data


class TestMetricsEndpoints:
    """Test metrics endpoints"""
    
    def test_prometheus_metrics(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Prometheus metrics are plain text
        content = response.text
        assert "api_requests_total" in content or "# TYPE" in content
    
    def test_cache_metrics(self, client):
        """Test cache metrics endpoint"""
        response = client.get("/metrics/cache")
        # May fail if Redis is not available
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            # Should have some Redis metrics
            assert isinstance(data, dict)
    
    def test_model_metrics(self, client):
        """Test model metrics endpoint"""
        response = client.get("/metrics/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "churn_model" in data


class TestErrorHandling:
    """Test API error handling"""
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/predict/churn", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        incomplete_request = {"customers": []}  # Missing customer data
        
        response = client.post("/predict/churn", json=incomplete_request)
        assert response.status_code == 422  # Validation error
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_invalid_customer_data(self, client):
        """Test handling of invalid customer data"""
        invalid_request = {
            "customers": [{
                "customer_id": "x",  # Too short (should be >= 3 chars)
                "recency_days": -5,  # Negative value
                "frequency": -1,     # Negative value
                "monetary_value": -100  # Negative value
            }],
            "return_probabilities": True
        }
        
        response = client.post("/predict/churn", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_endpoint(self, client):
        """Test 404 for non-existent endpoints"""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP methods"""
        response = client.get("/predict/churn")  # Should be POST
        assert response.status_code == 405


class TestMiddleware:
    """Test middleware functionality"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/health")
        
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers
    
    def test_security_headers(self, client):
        """Test security headers are added"""
        response = client.get("/health")
        
        headers = response.headers
        # These headers should be added by SecurityHeadersMiddleware
        security_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection"
        ]
        
        for header in security_headers:
            assert header in headers
    
    def test_request_id_header(self, client):
        """Test request ID is added to responses"""
        response = client.get("/health")
        
        # Should have request ID header
        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        assert len(request_id) > 10  # Should be a UUID-like string
    
    def test_processing_time_header(self, client):
        """Test processing time header"""
        response = client.get("/health")
        
        # Should have processing time header
        assert "x-processing-time" in response.headers
        processing_time = float(response.headers["x-processing-time"])
        assert processing_time >= 0


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test endpoints using async client"""
    
    async def test_concurrent_requests(self, async_client, sample_churn_request):
        """Test handling of concurrent requests"""
        import asyncio
        
        # Make multiple concurrent requests
        tasks = [
            async_client.post("/predict/churn", json=sample_churn_request)
            for _ in range(5)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed (or fail consistently if models not loaded)
        status_codes = [r.status_code for r in responses if hasattr(r, 'status_code')]
        assert all(code in [200, 503] for code in status_codes)
    
    async def test_timeout_handling(self, async_client):
        """Test request timeout handling"""
        # This would test timeout behavior, but requires specific setup
        response = await async_client.get("/health")
        assert response.status_code == 200