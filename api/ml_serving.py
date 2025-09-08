"""
Advanced ML Model Serving with Batch Processing and Optimization
"""

import asyncio
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import traceback

import numpy as np
import pandas as pd
import joblib
from fastapi import HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from .cache_manager import AdvancedCacheManager, cached_result, CacheConfig


logger = logging.getLogger(__name__)


class BatchPredictionJob(BaseModel):
    """Batch prediction job model"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="pending")  # pending, running, completed, failed
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = Field(default=0.0)
    total_samples: int = 0
    processed_samples: int = 0
    results_location: Optional[str] = None


class ModelVersion(BaseModel):
    """Model version information"""
    model_name: str
    version: str
    algorithm: str
    created_at: str
    performance_metrics: Dict[str, float]
    is_active: bool = True


class PredictionRequest(BaseModel):
    """Enhanced prediction request with metadata"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    features: List[Dict[str, Any]]
    return_probabilities: bool = True
    return_explanations: bool = False
    cache_results: bool = True
    timeout_seconds: int = 30


class PredictionResponse(BaseModel):
    """Enhanced prediction response"""
    request_id: str
    model_name: str
    model_version: str
    predictions: List[Dict[str, Any]]
    processing_time: float
    cache_hit: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MLModelManager:
    """Advanced ML model manager with versioning and hot-swapping"""
    
    def __init__(self, cache_manager: AdvancedCacheManager):
        self.cache_manager = cache_manager
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Model performance tracking
        self.prediction_counts = {}
        self.error_counts = {}
        self.latencies = {}
    
    async def load_model(self, 
                        model_name: str, 
                        model_path: str, 
                        version: str = "1.0.0",
                        algorithm: str = "unknown") -> bool:
        """Load ML model with versioning"""
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor, 
                joblib.load, 
                model_path
            )
            
            # Store model
            if model_name not in self.models:
                self.models[model_name] = {}
            
            self.models[model_name][version] = {
                "model": model,
                "path": model_path,
                "loaded_at": datetime.utcnow().isoformat(),
                "algorithm": algorithm,
                "is_active": True
            }
            
            # Update version tracking
            if model_name not in self.model_versions:
                self.model_versions[model_name] = []
            
            model_version = ModelVersion(
                model_name=model_name,
                version=version,
                algorithm=algorithm,
                created_at=datetime.utcnow().isoformat(),
                performance_metrics={}
            )
            
            self.model_versions[model_name].append(model_version)
            
            # Initialize stats
            self.prediction_counts[f"{model_name}:{version}"] = 0
            self.error_counts[f"{model_name}:{version}"] = 0
            self.latencies[f"{model_name}:{version}"] = []
            
            logger.info(f"Model '{model_name}' v{version} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            return False
    
    def get_model(self, model_name: str, version: str = "latest") -> Optional[Any]:
        """Get model by name and version"""
        if model_name not in self.models:
            return None
        
        if version == "latest":
            # Get the latest active version
            active_versions = {
                v: data for v, data in self.models[model_name].items() 
                if data.get("is_active", True)
            }
            if not active_versions:
                return None
            
            # Sort by loaded time and get latest
            latest_version = max(active_versions.keys(), 
                               key=lambda x: active_versions[x]["loaded_at"])
            return self.models[model_name][latest_version]["model"]
        
        return self.models[model_name].get(version, {}).get("model")
    
    async def predict_single(self, 
                           model_name: str, 
                           features: np.ndarray, 
                           return_probabilities: bool = True,
                           version: str = "latest") -> Dict[str, Any]:
        """Make single prediction with caching"""
        start_time = time.time()
        model_key = f"{model_name}:{version}"
        
        try:
            model = self.get_model(model_name, version)
            if model is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_name}' version '{version}' not found"
                )
            
            # Make prediction in thread pool
            loop = asyncio.get_event_loop()
            
            if return_probabilities and hasattr(model, 'predict_proba'):
                prediction = await loop.run_in_executor(
                    self.executor, 
                    model.predict, 
                    features
                )
                probabilities = await loop.run_in_executor(
                    self.executor,
                    model.predict_proba,
                    features
                )
                
                result = {
                    "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                    "probabilities": probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
                }
            else:
                prediction = await loop.run_in_executor(
                    self.executor,
                    model.predict, 
                    features
                )
                result = {
                    "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction
                }
            
            # Update stats
            processing_time = time.time() - start_time
            self.prediction_counts[model_key] += 1
            self.latencies[model_key].append(processing_time)
            
            # Keep only recent latencies for memory efficiency
            if len(self.latencies[model_key]) > 1000:
                self.latencies[model_key] = self.latencies[model_key][-500:]
            
            return result
            
        except Exception as e:
            self.error_counts[model_key] = self.error_counts.get(model_key, 0) + 1
            logger.error(f"Prediction error for {model_key}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )
    
    async def predict_batch(self, 
                          model_name: str, 
                          features_batch: List[np.ndarray],
                          return_probabilities: bool = True,
                          version: str = "latest",
                          chunk_size: int = 100) -> List[Dict[str, Any]]:
        """Make batch predictions with chunking for memory efficiency"""
        model = self.get_model(model_name, version)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' version '{version}' not found"
            )
        
        results = []
        
        # Process in chunks to manage memory
        for i in range(0, len(features_batch), chunk_size):
            chunk = features_batch[i:i + chunk_size]
            
            # Stack features for batch prediction
            try:
                features_array = np.vstack(chunk)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Feature array stacking failed: {str(e)}"
                )
            
            # Make prediction
            loop = asyncio.get_event_loop()
            
            try:
                if return_probabilities and hasattr(model, 'predict_proba'):
                    predictions, probabilities = await asyncio.gather(
                        loop.run_in_executor(self.executor, model.predict, features_array),
                        loop.run_in_executor(self.executor, model.predict_proba, features_array)
                    )
                    
                    for j, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        results.append({
                            "index": i + j,
                            "prediction": pred.tolist() if hasattr(pred, 'tolist') else pred,
                            "probabilities": prob.tolist() if hasattr(prob, 'tolist') else prob
                        })
                else:
                    predictions = await loop.run_in_executor(
                        self.executor, 
                        model.predict, 
                        features_array
                    )
                    
                    for j, pred in enumerate(predictions):
                        results.append({
                            "index": i + j,
                            "prediction": pred.tolist() if hasattr(pred, 'tolist') else pred
                        })
                        
            except Exception as e:
                logger.error(f"Batch prediction error for chunk {i//chunk_size}: {e}")
                # Add error placeholders for this chunk
                for j in range(len(chunk)):
                    results.append({
                        "index": i + j,
                        "error": str(e),
                        "prediction": None
                    })
        
        return results
    
    def get_model_stats(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """Get model performance statistics"""
        model_key = f"{model_name}:{version}"
        
        if version == "latest" and model_name in self.models:
            # Get stats for all versions
            all_stats = {}
            for v in self.models[model_name].keys():
                key = f"{model_name}:{v}"
                if key in self.prediction_counts:
                    all_stats[v] = self._get_stats_for_key(key)
            return all_stats
        
        return self._get_stats_for_key(model_key)
    
    def _get_stats_for_key(self, model_key: str) -> Dict[str, Any]:
        """Get statistics for specific model key"""
        latencies = self.latencies.get(model_key, [])
        
        return {
            "prediction_count": self.prediction_counts.get(model_key, 0),
            "error_count": self.error_counts.get(model_key, 0),
            "avg_latency": np.mean(latencies) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency": np.percentile(latencies, 99) if latencies else 0,
            "error_rate": (
                self.error_counts.get(model_key, 0) / 
                max(1, self.prediction_counts.get(model_key, 0))
            )
        }
    
    async def a_b_test_models(self, 
                            model_a: str, 
                            model_b: str,
                            traffic_split: float,
                            features: np.ndarray) -> Dict[str, Any]:
        """A/B test two model versions"""
        import random
        
        # Determine which model to use based on traffic split
        use_model_a = random.random() < traffic_split
        selected_model = model_a if use_model_a else model_b
        
        # Make prediction
        result = await self.predict_single(selected_model, features)
        
        # Add A/B test metadata
        result["ab_test"] = {
            "model_used": selected_model,
            "traffic_split": traffic_split,
            "model_a": model_a,
            "model_b": model_b
        }
        
        return result


class BatchPredictionManager:
    """Manager for long-running batch prediction jobs"""
    
    def __init__(self, model_manager: MLModelManager, cache_manager: AdvancedCacheManager):
        self.model_manager = model_manager
        self.cache_manager = cache_manager
        self.jobs: Dict[str, BatchPredictionJob] = {}
    
    async def submit_batch_job(self, 
                             model_name: str,
                             features_batch: List[Dict[str, Any]], 
                             job_config: Dict[str, Any]) -> str:
        """Submit a batch prediction job"""
        job = BatchPredictionJob(
            total_samples=len(features_batch)
        )
        
        self.jobs[job.job_id] = job
        
        # Cache job info
        await self.cache_manager.cache_batch_predictions(
            job.job_id,
            job.dict(),
            ttl=86400  # 24 hours
        )
        
        return job.job_id
    
    async def process_batch_job(self, 
                              job_id: str,
                              background_tasks: BackgroundTasks):
        """Process batch job in background"""
        background_tasks.add_task(self._run_batch_job, job_id)
    
    async def _run_batch_job(self, job_id: str):
        """Run batch prediction job"""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        try:
            job.status = "running"
            job.started_at = datetime.utcnow().isoformat()
            
            # Update job status in cache
            await self.cache_manager.set(
                f"batch_job:{job_id}",
                job.dict(),
                ttl=86400
            )
            
            # Simulate batch processing (replace with actual processing)
            for i in range(job.total_samples):
                # Process one sample
                await asyncio.sleep(0.1)  # Simulate processing time
                
                job.processed_samples += 1
                job.progress = (job.processed_samples / job.total_samples) * 100
                
                # Update progress every 10 samples
                if i % 10 == 0:
                    await self.cache_manager.set(
                        f"batch_job:{job_id}",
                        job.dict(),
                        ttl=86400
                    )
            
            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat()
            job.progress = 100.0
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()
            logger.error(f"Batch job {job_id} failed: {e}")
        
        finally:
            # Final update
            await self.cache_manager.set(
                f"batch_job:{job_id}",
                job.dict(),
                ttl=86400
            )
    
    async def get_job_status(self, job_id: str) -> Optional[BatchPredictionJob]:
        """Get batch job status"""
        # Try cache first
        cached_job = await self.cache_manager.get(f"batch_job:{job_id}")
        if cached_job:
            return BatchPredictionJob(**cached_job)
        
        # Fall back to memory
        return self.jobs.get(job_id)
    
    def list_jobs(self, limit: int = 100) -> List[BatchPredictionJob]:
        """List recent batch jobs"""
        jobs = list(self.jobs.values())
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]


class ModelExplainer:
    """Model explanation and interpretability"""
    
    def __init__(self, model_manager: MLModelManager):
        self.model_manager = model_manager
    
    async def explain_prediction(self, 
                               model_name: str,
                               features: np.ndarray,
                               explanation_type: str = "shap") -> Dict[str, Any]:
        """Generate prediction explanations"""
        try:
            if explanation_type == "shap":
                return await self._shap_explanation(model_name, features)
            elif explanation_type == "lime":
                return await self._lime_explanation(model_name, features)
            else:
                return {"error": f"Unsupported explanation type: {explanation_type}"}
                
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {"error": str(e)}
    
    async def _shap_explanation(self, model_name: str, features: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        try:
            import shap
            
            model = self.model_manager.get_model(model_name)
            if model is None:
                return {"error": "Model not found"}
            
            # Create SHAP explainer (simplified)
            explainer = shap.Explainer(model)
            shap_values = explainer(features)
            
            return {
                "explanation_type": "shap",
                "feature_importance": shap_values.values.tolist(),
                "base_values": shap_values.base_values.tolist(),
                "feature_names": getattr(shap_values, 'feature_names', None)
            }
            
        except ImportError:
            return {"error": "SHAP not available"}
        except Exception as e:
            return {"error": f"SHAP explanation failed: {str(e)}"}
    
    async def _lime_explanation(self, model_name: str, features: np.ndarray) -> Dict[str, Any]:
        """Generate LIME explanations"""
        return {"error": "LIME explanation not implemented"}