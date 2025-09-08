"""
Centralized configuration management for the e-commerce analytics platform
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Data paths
DATA_PATHS = {
    "customers": DATA_DIR / "olist_customers_dataset.csv",
    "orders": DATA_DIR / "olist_orders_dataset.csv",
    "order_items": DATA_DIR / "olist_order_items_dataset.csv",
    "order_payments": DATA_DIR / "olist_order_payments_dataset.csv",
    "order_reviews": DATA_DIR / "olist_order_reviews_dataset.csv",
    "products": DATA_DIR / "olist_products_dataset.csv",
    "sellers": DATA_DIR / "olist_sellers_dataset.csv",
    "geolocation": DATA_DIR / "olist_geolocation_dataset.csv",
    "category_translation": DATA_DIR / "product_category_name_translation.csv"
}

# Output paths
OUTPUT_PATHS = {
    "processed": ARTIFACTS_DIR / "processed",
    "processed_final": ARTIFACTS_DIR / "processed_final",
    "models": ARTIFACTS_DIR / "models",
    "models_final": ARTIFACTS_DIR / "models_final",
    "eda": ARTIFACTS_DIR / "eda",
    "nlp": ARTIFACTS_DIR / "nlp",
    "recommendations": ARTIFACTS_DIR / "recommendations",
    "business_metrics": ARTIFACTS_DIR / "business_metrics",
    "cohort_analysis": ARTIFACTS_DIR / "cohort_analysis"
}

# Model configurations
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "max_regularization": True,
    "class_balance": True
}

# Data processing configurations
DATA_CONFIG = {
    "churn_days": 90,  # Days without order to be considered churned
    "min_orders": 2,   # Minimum orders to be included in analysis
    "date_columns": ["order_purchase_timestamp", "order_approved_at", 
                    "order_delivered_carrier_date", "order_delivered_customer_date",
                    "order_estimated_delivery_date"],
    "chunk_size": 10000,  # For memory-efficient processing
    "sample_size": None   # Set to integer to sample data for testing
}

# Feature engineering configurations
FEATURE_CONFIG = {
    "rfm_bins": 5,
    "cohort_period": "M",  # Monthly cohorts
    "time_features": ["hour", "day_of_week", "month", "quarter"],
    "exclude_features": ["customer_id", "order_id", "product_id", "seller_id"]
}

# NLP configurations
NLP_CONFIG = {
    "language": "portuguese",
    "min_reviews": 10,
    "sentiment_threshold": 0.1,
    "topic_count": 10,
    "min_word_frequency": 5
}

# Streamlit app configurations
APP_CONFIG = {
    "page_title": "Customer Analytics Platform",
    "page_icon": "🚀",
    "layout": "wide",
    "theme": {
        "primaryColor": "#2196f3",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f8f9fa",
        "textColor": "#1a1a1a"
    }
}

# API configurations
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "reload": bool(os.getenv("API_RELOAD", True)),
    "workers": int(os.getenv("API_WORKERS", 1))
}

# Logging configurations
LOG_CONFIG = {
    "log_dir": BASE_DIR / "logs",
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# MLflow configurations
MLFLOW_CONFIG = {
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
    "experiment_name": "customer_analytics",
    "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", None)
}

# Create necessary directories
def create_directories():
    """Create all necessary directories if they don't exist"""
    for path in OUTPUT_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    LOG_CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)

# Initialize directories on import
create_directories()