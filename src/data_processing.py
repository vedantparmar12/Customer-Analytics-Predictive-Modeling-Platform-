import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class CustomerDataProcessor:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.customers_df = None
        self.orders_df = None
        self.order_items_df = None
        self.order_payments_df = None
        self.order_reviews_df = None
        self.products_df = None
        self.customer_features_df = None
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Customer Data Processor initialized")
    
    def load_data(self):
        """Load all ecommerce datasets"""
        try:
            logger.info("Loading ecommerce datasets...")
            self.customers_df = pd.read_csv(os.path.join(self.data_path, "olist_customers_dataset.csv"))
            self.orders_df = pd.read_csv(os.path.join(self.data_path, "olist_orders_dataset.csv"))
            self.order_items_df = pd.read_csv(os.path.join(self.data_path, "olist_order_items_dataset.csv"))
            self.order_payments_df = pd.read_csv(os.path.join(self.data_path, "olist_order_payments_dataset.csv"))
            self.order_reviews_df = pd.read_csv(os.path.join(self.data_path, "olist_order_reviews_dataset.csv"))
            self.products_df = pd.read_csv(os.path.join(self.data_path, "olist_products_dataset.csv"))
            
            logger.info(f"Loaded {len(self.customers_df)} customers")
            logger.info(f"Loaded {len(self.orders_df)} orders")
            logger.info("All datasets loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)
    
    def preprocess_dates(self):
        """Convert date columns to datetime"""
        try:
            date_columns = [
                'order_purchase_timestamp', 'order_approved_at', 
                'order_delivered_carrier_date', 'order_delivered_customer_date',
                'order_estimated_delivery_date'
            ]
            
            for col in date_columns:
                if col in self.orders_df.columns:
                    self.orders_df[col] = pd.to_datetime(self.orders_df[col])
            
            logger.info("Date preprocessing completed")
            
        except Exception as e:
            logger.error(f"Error preprocessing dates: {e}")
            raise CustomException("Failed to preprocess dates", e)
    
    def create_customer_features(self):
        """Create comprehensive customer features for analytics"""
        try:
            logger.info("Creating customer features...")
            
            # Merge datasets
            order_details = self.orders_df.merge(
                self.order_items_df[['order_id', 'product_id', 'price', 'freight_value']], 
                on='order_id'
            )
            order_details = order_details.merge(
                self.order_payments_df[['order_id', 'payment_value']], 
                on='order_id'
            )
            order_details = order_details.merge(
                self.customers_df[['customer_id', 'customer_unique_id', 'customer_city', 'customer_state']], 
                on='customer_id'
            )
            
            # Calculate reference date (last date in dataset)
            reference_date = self.orders_df['order_purchase_timestamp'].max()
            
            # Customer-level aggregations
            customer_features = order_details.groupby('customer_unique_id').agg({
                'order_id': 'nunique',  # Number of orders
                'payment_value': ['sum', 'mean', 'std'],  # Total, average, std spending
                'order_purchase_timestamp': ['min', 'max'],  # First and last purchase
                'product_id': 'nunique',  # Number of unique products
                'customer_state': 'first',  # Customer state
                'customer_city': 'first'  # Customer city
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                       for col in customer_features.columns.values]
            
            # Rename columns for clarity
            customer_features.rename(columns={
                'customer_unique_id_': 'customer_unique_id',
                'order_id_nunique': 'total_orders',
                'payment_value_sum': 'total_revenue',
                'payment_value_mean': 'avg_order_value',
                'payment_value_std': 'order_value_std',
                'order_purchase_timestamp_min': 'first_purchase_date',
                'order_purchase_timestamp_max': 'last_purchase_date',
                'product_id_nunique': 'unique_products_purchased',
                'customer_state_first': 'customer_state',
                'customer_city_first': 'customer_city'
            }, inplace=True)
            
            # Calculate RFM metrics
            customer_features['recency_days'] = (reference_date - customer_features['last_purchase_date']).dt.days
            customer_features['frequency'] = customer_features['total_orders']
            customer_features['monetary_value'] = customer_features['total_revenue']
            
            # Calculate customer lifetime (days)
            customer_features['customer_lifetime_days'] = (
                customer_features['last_purchase_date'] - customer_features['first_purchase_date']
            ).dt.days
            
            # Purchase behavior features
            customer_features['avg_days_between_orders'] = (
                customer_features['customer_lifetime_days'] / 
                customer_features['total_orders'].clip(lower=1)
            )
            
            # Binary churn label - more realistic definition
            # A customer is considered churned if:
            # 1. They haven't purchased in over 180 days (6 months) for frequent buyers
            # 2. They haven't purchased in over 365 days (12 months) for occasional buyers
            # 3. They haven't purchased in over 540 days (18 months) for one-time buyers
            # This accounts for different customer purchase patterns
            
            # First, identify customer types based on purchase frequency
            avg_days_between = customer_features['avg_days_between_orders'].fillna(365)
            
            # Frequent buyers: avg days between orders < 90
            frequent_buyers = avg_days_between < 90
            
            # Calculate churn based on customer type
            customer_features['churned'] = 0  # Initialize as not churned
            
            # Frequent buyers churn if no purchase in 180+ days (6 months)
            customer_features.loc[frequent_buyers & (customer_features['recency_days'] > 180), 'churned'] = 1
            
            # Occasional buyers churn if no purchase in 365+ days (12 months)
            customer_features.loc[~frequent_buyers & (customer_features['frequency'] > 1) & (customer_features['recency_days'] > 365), 'churned'] = 1
            
            # One-time buyers (frequency = 1) churn if no purchase in 540+ days (18 months)
            customer_features.loc[(customer_features['frequency'] == 1) & (customer_features['recency_days'] > 540), 'churned'] = 1
            
            customer_features['churned'] = customer_features['churned'].astype(int)
            
            # Product diversity score
            customer_features['product_diversity_score'] = (
                customer_features['unique_products_purchased'] / 
                customer_features['total_orders'].clip(lower=1)
            )
            
            # Fill missing values
            customer_features['order_value_std'].fillna(0, inplace=True)
            customer_features['avg_days_between_orders'].fillna(
                customer_features['avg_days_between_orders'].median(), inplace=True
            )
            
            self.customer_features_df = customer_features
            logger.info(f"Created features for {len(customer_features)} customers")
            
            # Log churn statistics
            churn_rate = customer_features['churned'].mean()
            logger.info(f"Churn rate: {churn_rate:.2%} (Churned: {customer_features['churned'].sum()}, Active: {(~customer_features['churned'].astype(bool)).sum()})")
            
        except Exception as e:
            logger.error(f"Error creating customer features: {e}")
            raise CustomException("Failed to create customer features", e)
    
    def create_rfm_segments(self):
        """Create RFM segments for customer analytics"""
        try:
            logger.info("Creating RFM segments...")
            
            # Create RFM scores using quantiles
            rfm_df = self.customer_features_df[['customer_unique_id', 'recency_days', 'frequency', 'monetary_value']].copy()
            
            # Create bins for RFM scores (1-5)
            rfm_df['R_score'] = pd.qcut(rfm_df['recency_days'], 5, labels=[5, 4, 3, 2, 1])
            rfm_df['F_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
            rfm_df['M_score'] = pd.qcut(rfm_df['monetary_value'], 5, labels=[1, 2, 3, 4, 5])
            
            # Combine RFM scores
            rfm_df['RFM_score'] = rfm_df['R_score'].astype(str) + rfm_df['F_score'].astype(str) + rfm_df['M_score'].astype(str)
            
            # Define customer segments
            def get_customer_segment(row):
                r, f, m = int(row['R_score']), int(row['F_score']), int(row['M_score'])
                
                if r >= 4 and f >= 4 and m >= 4:
                    return 'Champions'
                elif r >= 3 and f >= 3 and m >= 4:
                    return 'Loyal Customers'
                elif r >= 3 and f <= 2 and m >= 3:
                    return 'Potential Loyalists'
                elif r >= 4 and f <= 2:
                    return 'New Customers'
                elif r <= 2 and f >= 3 and m >= 3:
                    return 'At Risk'
                elif r <= 2 and f >= 4 and m >= 4:
                    return 'Cant Lose Them'
                elif r <= 2 and f <= 2 and m <= 2:
                    return 'Lost'
                else:
                    return 'Others'
            
            rfm_df['customer_segment'] = rfm_df.apply(get_customer_segment, axis=1)
            
            # Merge RFM segments back to main features
            self.customer_features_df = self.customer_features_df.merge(
                rfm_df[['customer_unique_id', 'R_score', 'F_score', 'M_score', 'RFM_score', 'customer_segment']],
                on='customer_unique_id'
            )
            
            logger.info("RFM segmentation completed")
            
        except Exception as e:
            logger.error(f"Error creating RFM segments: {e}")
            raise CustomException("Failed to create RFM segments", e)
    
    def prepare_ml_data(self):
        """Prepare data for machine learning models"""
        try:
            logger.info("Preparing data for ML models...")
            
            # Select features for churn prediction
            feature_columns = [
                'recency_days', 'frequency', 'monetary_value',
                'total_orders', 'avg_order_value', 'order_value_std',
                'unique_products_purchased', 'customer_lifetime_days',
                'avg_days_between_orders', 'product_diversity_score',
                'R_score', 'F_score', 'M_score'
            ]
            
            # Convert categorical RFM scores to numeric
            for col in ['R_score', 'F_score', 'M_score']:
                self.customer_features_df[col] = self.customer_features_df[col].astype(int)
            
            # Prepare features and target
            X = self.customer_features_df[feature_columns]
            y = self.customer_features_df['churned']
            
            # Handle any remaining missing values
            X = X.fillna(X.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save processed data
            joblib.dump(X_train_scaled, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test_scaled, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))
            joblib.dump(scaler, os.path.join(self.output_path, "scaler.pkl"))
            joblib.dump(feature_columns, os.path.join(self.output_path, "feature_columns.pkl"))
            
            # Save customer features for later use
            self.customer_features_df.to_csv(
                os.path.join(self.output_path, "customer_features.csv"), 
                index=False
            )
            
            logger.info(f"ML data prepared - Train: {len(X_train)}, Test: {len(X_test)}")
            logger.info(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
            
        except Exception as e:
            logger.error(f"Error preparing ML data: {e}")
            raise CustomException("Failed to prepare ML data", e)
    
    def run(self):
        """Run the complete data processing pipeline"""
        self.load_data()
        self.preprocess_dates()
        self.create_customer_features()
        self.create_rfm_segments()
        self.prepare_ml_data()
        logger.info("Data processing pipeline completed successfully")

if __name__ == "__main__":
    processor = CustomerDataProcessor("data", "artifacts/processed")
    processor.run()