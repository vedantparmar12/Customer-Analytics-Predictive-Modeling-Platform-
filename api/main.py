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
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.core import CollectorRegistry

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Analytics API",
    description="Real-time prediction API for customer churn, recommendations, and segmentation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

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

def load_models():
    """Load ML models into memory"""
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
    """Initialize models on startup"""
    load_models()
    print("Models loaded successfully")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        redis_connected = redis_client.ping()
    except:
        redis_connected = False
    
    return HealthResponse(
        status="healthy" if MODEL_CACHE else "degraded",
        models_loaded={
            "churn": "churn" in MODEL_CACHE,
            "scaler": "scaler" in MODEL_CACHE,
            "rfm": "rfm_params" in MODEL_CACHE
        },
        redis_connected=redis_connected,
        timestamp=datetime.utcnow().isoformat()
    )

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

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(registry)

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

@app.post("/model/reload")
async def reload_models():
    """Reload models from disk"""
    success = load_models()
    if success:
        return {"status": "success", "message": "Models reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload models")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)