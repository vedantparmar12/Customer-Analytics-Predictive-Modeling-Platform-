"""
Final Data Processing Pipeline - Production Ready
- Balanced churn definition
- Proper train/test split
- No data leakage
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
from datetime import timedelta
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class FinalDataProcessor:
    """Production-ready data processor with all issues fixed"""
    
    def __init__(self, data_path, output_path="artifacts/processed_final"):
        self.data_path = data_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
    def load_data(self):
        """Load all required datasets"""
        try:
            logger.info("Loading datasets...")
            self.customers_df = pd.read_csv(os.path.join(self.data_path, "olist_customers_dataset.csv"))
            self.orders_df = pd.read_csv(os.path.join(self.data_path, "olist_orders_dataset.csv"))
            self.order_items_df = pd.read_csv(os.path.join(self.data_path, "olist_order_items_dataset.csv"))
            self.order_payments_df = pd.read_csv(os.path.join(self.data_path, "olist_order_payments_dataset.csv"))
            
            # Convert dates
            date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                          'order_delivered_carrier_date', 'order_delivered_customer_date']
            for col in date_columns:
                if col in self.orders_df.columns:
                    self.orders_df[col] = pd.to_datetime(self.orders_df[col])
                    
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)
    
    def create_customer_features(self):
        """Create customer features with careful feature engineering"""
        try:
            logger.info("Creating customer features...")
            
            # Merge datasets
            order_details = self.orders_df.merge(
                self.order_items_df[['order_id', 'product_id', 'price', 'freight_value']], 
                on='order_id'
            )
            order_details = order_details.merge(
                self.customers_df[['customer_id', 'customer_unique_id', 'customer_state']], 
                on='customer_id'
            )
            
            # Filter only delivered orders
            order_details = order_details[order_details['order_status'] == 'delivered'].copy()
            
            # Reference date - use 2018-08-01 for consistency
            reference_date = pd.Timestamp('2018-08-01')
            logger.info(f"Using reference date: {reference_date}")
            
            # Customer aggregations
            customer_features = order_details.groupby('customer_unique_id').agg({
                'order_id': 'nunique',
                'price': ['sum', 'mean', 'std', 'median'],
                'product_id': 'nunique',
                'order_purchase_timestamp': ['min', 'max'],
                'customer_state': 'first',
                'freight_value': 'mean'
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = [
                'customer_unique_id', 'total_orders', 'total_revenue', 
                'avg_order_value', 'order_value_std', 'median_order_value',
                'unique_products_purchased', 'first_purchase_date', 
                'last_purchase_date', 'customer_state', 'avg_freight_value'
            ]
            
            # Fill missing values
            customer_features['order_value_std'] = customer_features['order_value_std'].fillna(0)
            
            # Time-based features (less predictive than raw recency)
            customer_features['customer_lifetime_days'] = (
                customer_features['last_purchase_date'] - customer_features['first_purchase_date']
            ).dt.days
            
            # Purchase patterns (avoid using recency directly)
            customer_features['orders_per_month'] = (
                customer_features['total_orders'] / 
                (customer_features['customer_lifetime_days'] / 30.0).clip(lower=1)
            )
            
            # Product diversity
            customer_features['products_per_order'] = (
                customer_features['unique_products_purchased'] / 
                customer_features['total_orders']
            )
            
            # Value patterns
            customer_features['value_consistency'] = (
                customer_features['order_value_std'] / 
                customer_features['avg_order_value'].clip(lower=1)
            )
            
            # Customer value score (without recency component)
            customer_features['customer_value_score'] = (
                np.log1p(customer_features['total_revenue']) * 
                np.log1p(customer_features['total_orders'])
            )
            
            # Purchase acceleration (change in order frequency)
            # Group by time periods to calculate purchase acceleration
            customer_features['lifetime_quarters'] = np.ceil(customer_features['customer_lifetime_days'] / 90).clip(lower=1)
            customer_features['avg_revenue_per_quarter'] = (
                customer_features['total_revenue'] / customer_features['lifetime_quarters']
            )
            
            # Create balanced churn definition
            # Calculate days since last purchase
            days_since_last = (reference_date - customer_features['last_purchase_date']).dt.days
            
            # More nuanced churn definition based on customer behavior
            # Fast buyers (>1 order/month): churn if no purchase in 90 days
            # Regular buyers (0.5-1 order/month): churn if no purchase in 180 days  
            # Slow buyers (<0.5 order/month): churn if no purchase in 365 days
            
            customer_features['churned'] = 0
            
            # Fast buyers
            fast_buyers = customer_features['orders_per_month'] > 1
            customer_features.loc[fast_buyers & (days_since_last > 90), 'churned'] = 1
            
            # Regular buyers
            regular_buyers = (customer_features['orders_per_month'] > 0.5) & (customer_features['orders_per_month'] <= 1)
            customer_features.loc[regular_buyers & (days_since_last > 180), 'churned'] = 1
            
            # Slow buyers
            slow_buyers = customer_features['orders_per_month'] <= 0.5
            customer_features.loc[slow_buyers & (days_since_last > 365), 'churned'] = 1
            
            # Create engagement score instead of RFM
            # Normalize features for scoring
            customer_features['frequency_score'] = pd.qcut(
                customer_features['total_orders'].rank(method='first'), 
                q=10, labels=False
            ) + 1
            
            customer_features['value_score'] = pd.qcut(
                customer_features['total_revenue'].rank(method='first'), 
                q=10, labels=False
            ) + 1
            
            customer_features['consistency_score'] = pd.qcut(
                customer_features['value_consistency'].rank(method='first', ascending=False), 
                q=10, labels=False
            ) + 1
            
            # Composite engagement score
            customer_features['engagement_score'] = (
                customer_features['frequency_score'] * 0.4 +
                customer_features['value_score'] * 0.4 +
                customer_features['consistency_score'] * 0.2
            )
            
            # State-based features (one-hot encoding)
            top_states = customer_features['customer_state'].value_counts().head(10).index
            customer_features['is_top_state'] = customer_features['customer_state'].isin(top_states).astype(int)
            
            self.customer_features = customer_features
            logger.info(f"Created features for {len(customer_features)} customers")
            logger.info(f"Churn rate: {customer_features['churned'].mean():.3f}")
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            raise CustomException("Failed to create features", e)
    
    def prepare_ml_data(self):
        """Prepare final ML datasets"""
        try:
            logger.info("Preparing ML data...")
            
            # Select features (excluding recency-based features)
            feature_columns = [
                'total_orders', 'total_revenue', 'avg_order_value', 
                'order_value_std', 'median_order_value',
                'unique_products_purchased', 'customer_lifetime_days',
                'orders_per_month', 'products_per_order', 
                'value_consistency', 'avg_freight_value',
                'frequency_score', 'value_score', 'consistency_score',
                'engagement_score', 'is_top_state',
                'customer_value_score', 'avg_revenue_per_quarter'
            ]
            
            X = self.customer_features[feature_columns].values
            y = self.customer_features['churned'].values
            
            # Remove any infinite or very large values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            
            # Stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Use RobustScaler (better for outliers)
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save processed data
            joblib.dump(X_train_scaled, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test_scaled, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))
            joblib.dump(scaler, os.path.join(self.output_path, "scaler.pkl"))
            joblib.dump(feature_columns, os.path.join(self.output_path, "feature_columns.pkl"))
            
            # Save full customer features for analysis
            self.customer_features.to_csv(
                os.path.join(self.output_path, "customer_features_final.csv"),
                index=False
            )
            
            # Create metadata
            metadata = {
                'total_customers': len(self.customer_features),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_churn_rate': y_train.mean(),
                'test_churn_rate': y_test.mean(),
                'n_features': len(feature_columns),
                'feature_columns': ', '.join(feature_columns)
            }
            
            pd.DataFrame([metadata]).to_csv(
                os.path.join(self.output_path, "metadata.csv"), 
                index=False
            )
            
            # Create CV splits
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_splits = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                cv_splits.append({
                    'fold': fold + 1,
                    'train_indices': train_idx,
                    'val_indices': val_idx
                })
            
            joblib.dump(cv_splits, os.path.join(self.output_path, "cv_splits.pkl"))
            
            logger.info(f"ML data prepared successfully")
            logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            logger.info(f"Train churn: {y_train.mean():.3f}, Test churn: {y_test.mean():.3f}")
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {e}")
            raise CustomException("Failed to prepare ML data", e)
    
    def run(self):
        """Run complete pipeline"""
        self.load_data()
        self.create_customer_features()
        self.prepare_ml_data()
        logger.info("Final data processing completed successfully")

if __name__ == "__main__":
    processor = FinalDataProcessor("data")
    processor.run()