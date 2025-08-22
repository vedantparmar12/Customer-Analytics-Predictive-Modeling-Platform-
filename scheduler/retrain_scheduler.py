"""
Automated Model Retraining Scheduler
Handles periodic retraining of ML models with new data
"""

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
import schedule
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
from src.data_processing_final import EnhancedDataProcessor
from src.model_training_final import FinalModelTrainer
from src.business_metrics import BusinessMetricsCalculator
from src.cohort_analysis import AdvancedCohortAnalysis

# Configuration
RETRAIN_SCHEDULE = os.getenv('RETRAIN_SCHEDULE', '0 2 * * 0')  # Weekly on Sunday at 2 AM
DATA_DIR = Path('/app/data')
ARTIFACTS_DIR = Path('/app/artifacts')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:///app/mlruns')

# Setup logging
logger = get_logger(__name__)

class ModelRetrainer:
    """Handles model retraining logic"""
    
    def __init__(self):
        self.data_processor = EnhancedDataProcessor()
        self.model_trainer = FinalModelTrainer()
        self.metrics_calculator = BusinessMetricsCalculator()
        self.cohort_analyzer = AdvancedCohortAnalysis()
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Performance thresholds
        self.performance_thresholds = {
            'accuracy': 0.75,
            'precision': 0.70,
            'recall': 0.65,
            'f1': 0.68
        }
        
        # Drift detection parameters
        self.drift_threshold = 0.15
        
    def check_data_availability(self):
        """Check if new data is available for retraining"""
        try:
            # Check for new data files
            data_files = [
                'olist_customers_dataset.csv',
                'olist_orders_dataset.csv',
                'olist_order_items_dataset.csv',
                'olist_order_reviews_dataset.csv'
            ]
            
            for file in data_files:
                file_path = DATA_DIR / file
                if not file_path.exists():
                    logger.error(f"Required data file not found: {file}")
                    return False
            
            # Check last modified time
            latest_modified = max([
                (DATA_DIR / file).stat().st_mtime 
                for file in data_files
            ])
            
            # Check if data is newer than last training
            last_training_file = ARTIFACTS_DIR / 'last_training.json'
            if last_training_file.exists():
                with open(last_training_file, 'r') as f:
                    last_training = json.load(f)
                    last_training_time = datetime.fromisoformat(last_training['timestamp'])
                    
                    if datetime.fromtimestamp(latest_modified) <= last_training_time:
                        logger.info("No new data available since last training")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return False
    
    def detect_data_drift(self, new_data, reference_data):
        """Detect if there's significant data drift"""
        try:
            # Calculate statistical measures for drift detection
            drift_scores = {}
            
            numeric_columns = new_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in reference_data.columns:
                    # KS test for numerical features
                    from scipy import stats
                    ks_statistic, p_value = stats.ks_2samp(
                        reference_data[col].dropna(),
                        new_data[col].dropna()
                    )
                    drift_scores[col] = ks_statistic
            
            # Check if any feature has significant drift
            max_drift = max(drift_scores.values()) if drift_scores else 0
            
            if max_drift > self.drift_threshold:
                logger.warning(f"Data drift detected: max KS statistic = {max_drift}")
                return True, drift_scores
            
            return False, drift_scores
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False, {}
    
    def evaluate_current_model_performance(self):
        """Evaluate performance of the current production model"""
        try:
            # Load current model metrics
            metrics_file = ARTIFACTS_DIR / 'models' / 'model_metrics.json'
            if not metrics_file.exists():
                logger.warning("No current model metrics found")
                return None
            
            with open(metrics_file, 'r') as f:
                current_metrics = json.load(f)
            
            # Check if model meets performance thresholds
            needs_retraining = False
            for metric, threshold in self.performance_thresholds.items():
                if metric in current_metrics and current_metrics[metric] < threshold:
                    logger.warning(f"Model performance below threshold: {metric}={current_metrics[metric]}")
                    needs_retraining = True
            
            return {
                'metrics': current_metrics,
                'needs_retraining': needs_retraining
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return None
    
    def retrain_models(self):
        """Main retraining pipeline"""
        logger.info("Starting automated model retraining...")
        
        try:
            # Start MLflow run
            with mlflow.start_run(run_name="automated_retrain") as run:
                mlflow.set_tag("retrain_type", "scheduled")
                mlflow.set_tag("timestamp", datetime.utcnow().isoformat())
                
                # Step 1: Check data availability
                if not self.check_data_availability():
                    logger.info("Skipping retraining - no new data")
                    mlflow.set_tag("status", "skipped_no_new_data")
                    return
                
                # Step 2: Load and process data
                logger.info("Processing data...")
                processed_data = self.data_processor.run_pipeline()
                
                if processed_data is None:
                    logger.error("Data processing failed")
                    mlflow.set_tag("status", "failed_data_processing")
                    return
                
                # Step 3: Check for data drift
                if (ARTIFACTS_DIR / 'processed' / 'customer_features.csv').exists():
                    reference_data = pd.read_csv(ARTIFACTS_DIR / 'processed' / 'customer_features.csv')
                    has_drift, drift_scores = self.detect_data_drift(
                        processed_data['customer_features'],
                        reference_data
                    )
                    mlflow.log_metric("max_data_drift", max(drift_scores.values()) if drift_scores else 0)
                    mlflow.log_dict(drift_scores, "drift_scores.json")
                
                # Step 4: Evaluate current model
                current_performance = self.evaluate_current_model_performance()
                if current_performance:
                    mlflow.log_metrics({
                        f"current_{k}": v 
                        for k, v in current_performance['metrics'].items()
                        if isinstance(v, (int, float))
                    })
                
                # Step 5: Train new models
                logger.info("Training new models...")
                training_results = self.model_trainer.train_all_models(
                    processed_data['customer_features']
                )
                
                if not training_results['success']:
                    logger.error("Model training failed")
                    mlflow.set_tag("status", "failed_training")
                    return
                
                # Step 6: Compare performance
                new_metrics = training_results['metrics']
                performance_improved = self._compare_model_performance(
                    current_performance['metrics'] if current_performance else {},
                    new_metrics
                )
                
                mlflow.log_metrics({
                    f"new_{k}": v 
                    for k, v in new_metrics.items()
                    if isinstance(v, (int, float))
                })
                
                # Step 7: Deploy if improved
                if performance_improved:
                    logger.info("New model shows improvement - deploying...")
                    self._deploy_new_model(training_results)
                    mlflow.set_tag("status", "deployed")
                    mlflow.set_tag("deployment_timestamp", datetime.utcnow().isoformat())
                else:
                    logger.info("New model does not show significant improvement - keeping current model")
                    mlflow.set_tag("status", "not_deployed")
                
                # Step 8: Update business metrics
                logger.info("Updating business metrics...")
                self.metrics_calculator.calculate_all_metrics()
                
                # Step 9: Update cohort analysis
                logger.info("Updating cohort analysis...")
                self.cohort_analyzer.run_complete_analysis()
                
                # Step 10: Save training metadata
                training_metadata = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'mlflow_run_id': run.info.run_id,
                    'model_deployed': performance_improved,
                    'metrics': new_metrics,
                    'data_drift_detected': has_drift if 'has_drift' in locals() else False
                }
                
                with open(ARTIFACTS_DIR / 'last_training.json', 'w') as f:
                    json.dump(training_metadata, f, indent=2)
                
                logger.info("Retraining pipeline completed successfully")
                
        except Exception as e:
            logger.error(f"Error in retraining pipeline: {e}")
            logger.error(traceback.format_exc())
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
    
    def _compare_model_performance(self, current_metrics, new_metrics):
        """Compare model performance to decide on deployment"""
        if not current_metrics:
            return True  # Deploy if no current model
        
        # Define improvement threshold
        improvement_threshold = 0.02  # 2% improvement required
        
        # Key metrics to compare
        key_metrics = ['accuracy', 'f1', 'auc']
        
        improvements = []
        for metric in key_metrics:
            if metric in current_metrics and metric in new_metrics:
                improvement = (new_metrics[metric] - current_metrics[metric]) / current_metrics[metric]
                improvements.append(improvement)
                logger.info(f"{metric}: {current_metrics[metric]:.4f} -> {new_metrics[metric]:.4f} ({improvement*100:.2f}%)")
        
        # Deploy if average improvement exceeds threshold
        avg_improvement = np.mean(improvements) if improvements else 0
        return avg_improvement >= improvement_threshold
    
    def _deploy_new_model(self, training_results):
        """Deploy new model to production"""
        try:
            # Backup current model
            backup_dir = ARTIFACTS_DIR / 'models' / 'backup' / datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy current models to backup
            import shutil
            current_models_dir = ARTIFACTS_DIR / 'models'
            for model_file in current_models_dir.glob('*.pkl'):
                shutil.copy2(model_file, backup_dir)
            
            # Deploy new model (already saved by trainer)
            logger.info("New model deployed successfully")
            
            # Update model version info
            version_info = {
                'version': datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
                'deployed_at': datetime.utcnow().isoformat(),
                'metrics': training_results['metrics'],
                'mlflow_run_id': mlflow.active_run().info.run_id if mlflow.active_run() else None
            }
            
            with open(ARTIFACTS_DIR / 'models' / 'version_info.json', 'w') as f:
                json.dump(version_info, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise

def run_scheduler():
    """Run the scheduling loop"""
    retrainer = ModelRetrainer()
    
    # Schedule regular retraining
    schedule.every().sunday.at("02:00").do(retrainer.retrain_models)
    
    # Also check every hour for urgent retraining needs
    def check_urgent_retrain():
        performance = retrainer.evaluate_current_model_performance()
        if performance and performance['needs_retraining']:
            logger.warning("Urgent retraining triggered due to performance degradation")
            retrainer.retrain_models()
    
    schedule.every().hour.do(check_urgent_retrain)
    
    logger.info("Model retraining scheduler started")
    logger.info(f"Scheduled retraining: {RETRAIN_SCHEDULE}")
    
    # Run once on startup if requested
    if os.getenv('RUN_ON_STARTUP', 'false').lower() == 'true':
        logger.info("Running initial training on startup...")
        retrainer.retrain_models()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    run_scheduler()