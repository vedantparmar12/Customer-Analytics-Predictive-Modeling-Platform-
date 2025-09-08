"""
Real-time Prediction API for E-commerce Analytics Platform
"""

import os
import json
import time
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import redis
import joblib
from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.core import CollectorRegistry

# Import our advanced modules
from .cache_manager import AdvancedCacheManager, RedisClusterManager
from .middleware import (
    RequestLoggingMiddleware, MetricsMiddleware, RateLimitingMiddleware,
    CacheMiddleware, SecurityHeadersMiddleware, ErrorHandlingMiddleware
)
from .dependencies import (
    get_cache_manager, get_current_user, get_api_key_user, get_optional_user,
    get_settings, get_health_status
)
from .auth import JWTAuth, APIKeyAuth
from .ml_serving import MLModelManager, BatchPredictionManager, ModelExplainer

# Initialize advanced components (will be initialized in startup)
redis_manager = None
cache_manager = None
ml_model_manager = None
batch_manager = None
model_explainer = None
jwt_auth = JWTAuth()
api_key_auth = None

# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="E-commerce Analytics API",
    description="Advanced ML serving platform with Redis caching, authentication, and monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Authentication", "description": "JWT and API key authentication"},
        {"name": "Predictions", "description": "ML model predictions"},
        {"name": "Batch Processing", "description": "Batch prediction jobs"},
        {"name": "Model Management", "description": "Model versioning and stats"},
        {"name": "Monitoring", "description": "Health checks and metrics"}
    ]
)

# Add advanced middleware stack
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache and rate limiting middleware (will be added after startup)
async def add_cache_middleware():
    """Add cache and rate limiting middleware after dependencies are initialized"""
    app.add_middleware(CacheMiddleware, cache_manager=cache_manager)
    app.add_middleware(RateLimitingMiddleware, cache_manager=cache_manager)

# Legacy Redis connection for backward compatibility
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Prometheus metrics
registry = CollectorRegistry()
prediction_counter = Counter(
    'predictions_total', 
    'Total number of predictions',
    ['model_type', 'status'],
    registry=registry
)
prediction_latency = Histogram(
    'prediction_duration_seconds',
    'Prediction latency',
    ['model_type'],
    registry=registry
)

# Global model cache
MODEL_CACHE = {}

class CustomerFeatures(BaseModel):
    """Customer features for prediction"""
    customer_id: str
    recency_days: float = Field(..., ge=0)
    frequency: int = Field(..., ge=0)
    monetary_value: float = Field(..., ge=0)
    customer_lifetime_days: int = Field(..., ge=0)
    total_orders: int = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    customer_state: Optional[str] = None
    preferred_payment_type: Optional[str] = None
    
    @validator('customer_id')
    def validate_customer_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Customer ID must be at least 3 characters')
        return v

class ChurnPredictionRequest(BaseModel):
    """Request model for churn prediction"""
    customers: List[CustomerFeatures]
    return_probabilities: bool = True

class ChurnPredictionResponse(BaseModel):
    """Response model for churn prediction"""
    predictions: List[Dict[str, Union[str, float, bool]]]
    model_version: str
    timestamp: str

class RecommendationRequest(BaseModel):
    """Request model for recommendations"""
    customer_id: str
    n_recommendations: int = Field(default=5, ge=1, le=20)
    recommendation_type: str = Field(default="collaborative", pattern="^(collaborative|content|hybrid)$")

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    customer_id: str
    recommendations: List[Dict[str, Union[str, float]]]
    recommendation_type: str
    timestamp: str

class SegmentationRequest(BaseModel):
    """Request model for customer segmentation"""
    customers: List[CustomerFeatures]
    segmentation_type: str = Field(default="rfm", pattern="^(rfm|behavioral|clv)$")

class SegmentationResponse(BaseModel):
    """Response model for segmentation"""
    segments: List[Dict[str, Union[str, int]]]
    segmentation_type: str
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    customers: List[CustomerFeatures]
    include_churn: bool = True
    include_segmentation: bool = True
    include_recommendations: bool = False

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]
    redis_connected: bool
    timestamp: str

async def load_advanced_models():
    """Load ML models using advanced model manager"""
    try:
        # Load churn prediction model
        model_path = "artifacts/models/xgboost_model_tuned.pkl"
        if os.path.exists(model_path):
            await ml_model_manager.load_model(
                model_name="churn",
                model_path=model_path,
                version="1.0.0",
                algorithm="XGBoost"
            )
        
        # Load feature scaler
        scaler_path = "artifacts/models/feature_scaler.pkl"
        if os.path.exists(scaler_path):
            await ml_model_manager.load_model(
                model_name="scaler",
                model_path=scaler_path,
                version="1.0.0",
                algorithm="StandardScaler"
            )
        
        # Load RFM segmentation parameters
        rfm_path = "artifacts/models/rfm_parameters.json"
        if os.path.exists(rfm_path):
            with open(rfm_path, 'r') as f:
                rfm_params = json.load(f)
                # Cache RFM parameters
                await cache_manager.set("rfm_parameters", rfm_params, ttl=86400)
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Legacy function for backward compatibility
def load_models():
    """Load ML models into memory (legacy)"""
    global MODEL_CACHE
    
    try:
        # Load churn prediction model
        model_path = "artifacts/models/xgboost_model_tuned.pkl"
        if os.path.exists(model_path):
            MODEL_CACHE['churn'] = joblib.load(model_path)
        
        # Load feature scaler
        scaler_path = "artifacts/models/feature_scaler.pkl"
        if os.path.exists(scaler_path):
            MODEL_CACHE['scaler'] = joblib.load(scaler_path)
        
        # Load RFM segmentation parameters
        rfm_path = "artifacts/models/rfm_parameters.json"
        if os.path.exists(rfm_path):
            with open(rfm_path, 'r') as f:
                MODEL_CACHE['rfm_params'] = json.load(f)
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global redis_manager, cache_manager, ml_model_manager, batch_manager, model_explainer, api_key_auth
    
    try:
        # Initialize Redis and cache manager
        redis_manager = RedisClusterManager(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD'),
            cluster_mode=os.getenv('REDIS_CLUSTER_MODE', 'false').lower() == 'true'
        )
        
        cache_manager = AdvancedCacheManager(redis_manager)
        await cache_manager.initialize()
        print("✅ Cache manager initialized")
        
        # Initialize ML components
        ml_model_manager = MLModelManager(cache_manager)
        batch_manager = BatchPredictionManager(ml_model_manager, cache_manager)
        model_explainer = ModelExplainer(ml_model_manager)
        
        # Initialize API key auth with cache
        api_key_auth = APIKeyAuth(cache_manager)
        
        # Add cache-dependent middleware
        await add_cache_middleware()
        print("✅ Advanced middleware added")
        
        # Load ML models
        await load_advanced_models()
        print("✅ ML models loaded successfully")
        
        print("🚀 E-commerce Analytics API v2.0 started successfully!")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        # Don't raise in development to allow partial functionality
        if os.getenv('ENVIRONMENT') == 'production':
            raise

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = await get_health_status()
    
    return HealthResponse(
        status=health_status["status"],
        models_loaded=health_status["services"]["models"].get("loaded_models", {}),
        redis_connected=health_status["services"]["cache"]["status"] == "healthy",
        timestamp=health_status["timestamp"]
    )

@app.get("/health/detailed", tags=["Monitoring"])
async def detailed_health_check():
    """Detailed health check with all service statuses"""
    return await get_health_status()

@app.post("/predict/churn", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    """Predict customer churn"""
    start_time = time.time()
    
    try:
        if 'churn' not in MODEL_CACHE:
            raise HTTPException(status_code=503, detail="Churn model not loaded")
        
        # Prepare features
        feature_data = []
        for customer in request.customers:
            features = [
                customer.recency_days,
                customer.frequency,
                customer.monetary_value,
                customer.customer_lifetime_days,
                customer.total_orders,
                customer.avg_order_value
            ]
            feature_data.append(features)
        
        # Convert to numpy array
        X = np.array(feature_data)
        
        # Scale features if scaler is available
        if 'scaler' in MODEL_CACHE:
            X = MODEL_CACHE['scaler'].transform(X)
        
        # Make predictions
        model = MODEL_CACHE['churn']
        predictions = model.predict(X)
        
        if request.return_probabilities:
            probabilities = model.predict_proba(X)[:, 1]
        else:
            probabilities = [None] * len(predictions)
        
        # Cache predictions in Redis
        for i, customer in enumerate(request.customers):
            cache_key = f"churn_pred:{customer.customer_id}"
            cache_value = {
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i]) if probabilities[i] is not None else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            redis_client.setex(cache_key, 3600, json.dumps(cache_value))
        
        # Prepare response
        results = []
        for i, customer in enumerate(request.customers):
            results.append({
                "customer_id": customer.customer_id,
                "churn_prediction": bool(predictions[i]),
                "churn_probability": float(probabilities[i]) if probabilities[i] is not None else None
            })
        
        # Record metrics
        prediction_counter.labels(model_type='churn', status='success').inc(len(request.customers))
        prediction_latency.labels(model_type='churn').observe(time.time() - start_time)
        
        return ChurnPredictionResponse(
            predictions=results,
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        prediction_counter.labels(model_type='churn', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get product recommendations for a customer"""
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"recommendations:{request.customer_id}:{request.recommendation_type}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            cached_data = json.loads(cached_result)
            return RecommendationResponse(**cached_data)
        
        # Generate mock recommendations (replace with actual recommendation engine)
        recommendations = []
        for i in range(request.n_recommendations):
            recommendations.append({
                "product_id": f"PROD_{i+1:04d}",
                "score": round(0.9 - i * 0.1, 2),
                "category": f"Category_{i % 3 + 1}"
            })
        
        response = RecommendationResponse(
            customer_id=request.customer_id,
            recommendations=recommendations,
            recommendation_type=request.recommendation_type,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Cache results
        redis_client.setex(cache_key, 3600, response.json())
        
        # Record metrics
        prediction_counter.labels(model_type='recommendation', status='success').inc()
        prediction_latency.labels(model_type='recommendation').observe(time.time() - start_time)
        
        return response
    
    except Exception as e:
        prediction_counter.labels(model_type='recommendation', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/segmentation", response_model=SegmentationResponse)
async def segment_customers(request: SegmentationRequest):
    """Segment customers based on RFM or other methods"""
    start_time = time.time()
    
    try:
        segments = []
        
        for customer in request.customers:
            if request.segmentation_type == "rfm":
                # Simple RFM segmentation logic
                r_score = 5 if customer.recency_days < 30 else 4 if customer.recency_days < 60 else 3 if customer.recency_days < 90 else 2 if customer.recency_days < 180 else 1
                f_score = 5 if customer.frequency > 10 else 4 if customer.frequency > 5 else 3 if customer.frequency > 3 else 2 if customer.frequency > 1 else 1
                m_score = 5 if customer.monetary_value > 1000 else 4 if customer.monetary_value > 500 else 3 if customer.monetary_value > 250 else 2 if customer.monetary_value > 100 else 1
                
                rfm_score = f"{r_score}{f_score}{m_score}"
                
                # Map to segment names
                if rfm_score in ['555', '554', '544', '545', '454', '455', '445']:
                    segment = 'Champions'
                elif rfm_score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                    segment = 'Loyal Customers'
                elif rfm_score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                    segment = 'Potential Loyalists'
                elif rfm_score in ['512', '511', '422', '421', '412', '411', '311']:
                    segment = 'New Customers'
                else:
                    segment = 'At Risk'
                
                segments.append({
                    "customer_id": customer.customer_id,
                    "segment": segment,
                    "rfm_score": rfm_score
                })
        
        # Record metrics
        prediction_counter.labels(model_type='segmentation', status='success').inc(len(request.customers))
        prediction_latency.labels(model_type='segmentation').observe(time.time() - start_time)
        
        return SegmentationResponse(
            segments=segments,
            segmentation_type=request.segmentation_type,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        prediction_counter.labels(model_type='segmentation', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def batch_predictions(request: BatchPredictionRequest):
    """Get multiple predictions in a single request"""
    results = {
        "customer_predictions": [],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for customer in request.customers:
        customer_result = {
            "customer_id": customer.customer_id
        }
        
        # Churn prediction
        if request.include_churn:
            churn_req = ChurnPredictionRequest(customers=[customer])
            churn_resp = await predict_churn(churn_req)
            customer_result["churn"] = churn_resp.predictions[0]
        
        # Segmentation
        if request.include_segmentation:
            seg_req = SegmentationRequest(customers=[customer])
            seg_resp = await segment_customers(seg_req)
            customer_result["segment"] = seg_resp.segments[0]["segment"]
        
        # Recommendations
        if request.include_recommendations:
            rec_req = RecommendationRequest(customer_id=customer.customer_id)
            rec_resp = await get_recommendations(rec_req)
            customer_result["recommendations"] = rec_resp.recommendations
        
        results["customer_predictions"].append(customer_result)
    
    return results

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(registry)

@app.get("/metrics/cache", tags=["Monitoring"])
async def get_cache_metrics():
    """Cache-specific metrics"""
    if not cache_manager:
        return {"error": "Cache manager not available"}
        
    return await cache_manager.get_cache_stats()

@app.get("/metrics/models", tags=["Monitoring"])
async def get_model_metrics():
    """Model performance metrics"""
    if not ml_model_manager:
        return {
            "error": "Model manager not available",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    return {
        "churn_model": ml_model_manager.get_model_stats("churn"),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get information about loaded models"""
    info = {
        "models_loaded": list(MODEL_CACHE.keys()),
        "model_paths": {
            "churn": "artifacts/models/xgboost_model_tuned.pkl",
            "scaler": "artifacts/models/feature_scaler.pkl",
            "rfm": "artifacts/models/rfm_parameters.json"
        },
        "last_loaded": datetime.utcnow().isoformat()
    }
    return info

@app.post("/model/reload", tags=["Model Management"])
async def reload_models(user: dict = Depends(get_api_key_user)):
    """Reload models from disk (requires authentication)"""
    success = await load_advanced_models()
    if success:
        return {
            "status": "success", 
            "message": "Models reloaded successfully",
            "reloaded_by": user.get("user_id"),
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload models")

@app.get("/model/versions", tags=["Model Management"])
async def list_model_versions():
    """List all loaded model versions"""
    return {
        "models": ml_model_manager.model_versions,
        "timestamp": datetime.utcnow().isoformat()
    }

# Authentication endpoints
@app.post("/auth/token", tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login"""
    user = jwt_auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = jwt_auth.create_access_token(data={"sub": user["username"]})
    refresh_token = jwt_auth.create_refresh_token(data={"sub": user["username"]})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.post("/auth/refresh", tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    new_token = jwt_auth.refresh_access_token(refresh_token)
    if not new_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    return {
        "access_token": new_token,
        "token_type": "bearer"
    }

@app.post("/auth/api-key", tags=["Authentication"])
async def create_api_key(request: dict, user: dict = Depends(get_current_user)):
    """Create new API key"""
    if not api_key_auth:
        raise HTTPException(status_code=503, detail="API key service not available")
        
    api_key_data = await api_key_auth.create_api_key(
        user_id=user["username"],
        name=request.get("name", "API Key"),
        permissions=request.get("permissions", ["read"]),
        rate_limit=request.get("rate_limit", 100)
    )
    
    return api_key_data

# Batch processing endpoints
@app.post("/batch/predict", tags=["Batch Processing"])
async def submit_batch_prediction(request: dict, background_tasks: BackgroundTasks):
    """Submit batch prediction job"""
    if not batch_manager:
        raise HTTPException(status_code=503, detail="Batch processing service not available")
        
    job_id = await batch_manager.submit_batch_job(
        model_name=request["model_name"],
        features_batch=request["features"],
        job_config=request.get("config", {})
    )
    
    # Process in background
    await batch_manager.process_batch_job(job_id, background_tasks)
    
    return {
        "job_id": job_id,
        "status": "submitted",
        "message": "Batch job submitted for processing"
    }

@app.get("/batch/jobs/{job_id}", tags=["Batch Processing"])
async def get_batch_job_status(job_id: str):
    """Get batch job status"""
    if not batch_manager:
        raise HTTPException(status_code=503, detail="Batch processing service not available")
        
    job_status = await batch_manager.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status

@app.get("/batch/jobs", tags=["Batch Processing"])
async def list_batch_jobs():
    """List recent batch jobs"""
    if not batch_manager:
        raise HTTPException(status_code=503, detail="Batch processing service not available")
        
    return {
        "jobs": batch_manager.list_jobs(limit=50),
        "timestamp": datetime.utcnow().isoformat()
    }

# Model explanation endpoints
@app.post("/explain/{model_name}", tags=["Model Management"])
async def explain_prediction(model_name: str, request: dict):
    """Get model prediction explanations"""
    if not model_explainer:
        raise HTTPException(status_code=503, detail="Model explanation service not available")
        
    features = np.array(request["features"])
    explanation_type = request.get("explanation_type", "shap")
    
    explanation = await model_explainer.explain_prediction(
        model_name=model_name,
        features=features,
        explanation_type=explanation_type
    )
    
    return explanation

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)