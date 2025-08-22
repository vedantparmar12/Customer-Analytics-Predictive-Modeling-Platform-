"""
Time Series Data Preparation for Sales Forecasting
Prepares e-commerce data for various forecasting models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class TimeSeriesPreprocessor:
    """Prepares e-commerce data for time series forecasting"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        logger.info("TimeSeriesPreprocessor initialized")
    
    def load_data(self):
        """Load necessary datasets for time series analysis"""
        try:
            logger.info("Loading datasets for time series analysis")
            
            # Load orders data
            self.orders = pd.read_csv(f"{self.data_path}/olist_orders_dataset.csv")
            self.order_items = pd.read_csv(f"{self.data_path}/olist_order_items_dataset.csv")
            self.products = pd.read_csv(f"{self.data_path}/olist_products_dataset.csv")
            self.customers = pd.read_csv(f"{self.data_path}/olist_customers_dataset.csv")
            
            # Convert date columns
            date_columns = ['order_purchase_timestamp', 'order_approved_at', 
                          'order_delivered_carrier_date', 'order_delivered_customer_date']
            for col in date_columns:
                if col in self.orders.columns:
                    self.orders[col] = pd.to_datetime(self.orders[col])
            
            logger.info(f"Loaded {len(self.orders)} orders")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)
    
    def prepare_daily_sales(self):
        """Prepare daily sales aggregation"""
        try:
            logger.info("Preparing daily sales data")
            
            # Merge orders with order items
            order_sales = self.orders.merge(
                self.order_items[['order_id', 'price', 'freight_value']],
                on='order_id',
                how='left'
            )
            
            # Calculate total value per order
            order_sales['total_value'] = order_sales['price'] + order_sales['freight_value']
            
            # Filter only delivered orders
            delivered_orders = order_sales[order_sales['order_status'] == 'delivered'].copy()
            
            # Group by date
            delivered_orders['date'] = delivered_orders['order_purchase_timestamp'].dt.date
            daily_sales = delivered_orders.groupby('date').agg({
                'total_value': 'sum',
                'order_id': 'count',
                'customer_id': 'nunique'
            }).reset_index()
            
            daily_sales.columns = ['date', 'revenue', 'order_count', 'unique_customers']
            
            # Ensure continuous date range
            date_range = pd.date_range(
                start=daily_sales['date'].min(),
                end=daily_sales['date'].max(),
                freq='D'
            )
            
            daily_sales['date'] = pd.to_datetime(daily_sales['date'])
            daily_sales = daily_sales.set_index('date').reindex(date_range, fill_value=0).reset_index()
            daily_sales.columns = ['date', 'revenue', 'order_count', 'unique_customers']
            
            logger.info(f"Prepared {len(daily_sales)} days of sales data")
            return daily_sales
            
        except Exception as e:
            logger.error(f"Error preparing daily sales: {e}")
            raise CustomException("Failed to prepare daily sales", e)
    
    def prepare_weekly_sales(self):
        """Prepare weekly sales aggregation"""
        try:
            daily_sales = self.prepare_daily_sales()
            
            # Add week information
            daily_sales['week'] = daily_sales['date'].dt.to_period('W')
            
            weekly_sales = daily_sales.groupby('week').agg({
                'revenue': 'sum',
                'order_count': 'sum',
                'unique_customers': 'sum'
            }).reset_index()
            
            weekly_sales['week_start'] = weekly_sales['week'].dt.start_time
            
            logger.info(f"Prepared {len(weekly_sales)} weeks of sales data")
            return weekly_sales
            
        except Exception as e:
            logger.error(f"Error preparing weekly sales: {e}")
            raise CustomException("Failed to prepare weekly sales", e)
    
    def prepare_monthly_sales(self):
        """Prepare monthly sales aggregation"""
        try:
            daily_sales = self.prepare_daily_sales()
            
            # Add month information
            daily_sales['month'] = daily_sales['date'].dt.to_period('M')
            
            monthly_sales = daily_sales.groupby('month').agg({
                'revenue': 'sum',
                'order_count': 'sum',
                'unique_customers': 'sum'
            }).reset_index()
            
            monthly_sales['month_start'] = monthly_sales['month'].dt.start_time
            
            logger.info(f"Prepared {len(monthly_sales)} months of sales data")
            return monthly_sales
            
        except Exception as e:
            logger.error(f"Error preparing monthly sales: {e}")
            raise CustomException("Failed to prepare monthly sales", e)
    
    def prepare_product_sales(self, product_category=None):
        """Prepare sales data for specific product categories"""
        try:
            logger.info("Preparing product-level sales data")
            
            # Merge all necessary data
            product_sales = self.order_items.merge(
                self.orders[['order_id', 'order_purchase_timestamp', 'order_status']],
                on='order_id',
                how='left'
            ).merge(
                self.products[['product_id', 'product_category_name']],
                on='product_id',
                how='left'
            )
            
            # Filter delivered orders
            product_sales = product_sales[product_sales['order_status'] == 'delivered']
            
            # Filter by category if specified
            if product_category:
                product_sales = product_sales[
                    product_sales['product_category_name'] == product_category
                ]
            
            # Convert timestamp
            product_sales['date'] = pd.to_datetime(
                product_sales['order_purchase_timestamp']
            ).dt.date
            
            # Aggregate by date and category
            daily_product_sales = product_sales.groupby(['date', 'product_category_name']).agg({
                'price': 'sum',
                'order_id': 'count'
            }).reset_index()
            
            daily_product_sales.columns = ['date', 'category', 'revenue', 'order_count']
            
            logger.info(f"Prepared product sales for {daily_product_sales['category'].nunique()} categories")
            return daily_product_sales
            
        except Exception as e:
            logger.error(f"Error preparing product sales: {e}")
            raise CustomException("Failed to prepare product sales", e)
    
    def add_time_features(self, df, date_col='date'):
        """Add time-based features for modeling"""
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Extract time features
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day'] = df[date_col].dt.day
            df['dayofweek'] = df[date_col].dt.dayofweek
            df['quarter'] = df[date_col].dt.quarter
            df['dayofyear'] = df[date_col].dt.dayofyear
            df['weekofyear'] = df[date_col].dt.isocalendar().week
            
            # Add holiday indicators (Brazilian holidays)
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            
            # Add month start/end indicators
            df['is_month_start'] = df[date_col].dt.day <= 5
            df['is_month_end'] = df[date_col].dt.day >= 25
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            raise CustomException("Failed to add time features", e)
    
    def check_stationarity(self, timeseries, title=''):
        """Check if time series is stationary using ADF test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Perform ADF test
            result = adfuller(timeseries.dropna())
            
            logger.info(f'\nADF Test Results for {title}:')
            logger.info(f'ADF Statistic: {result[0]:.4f}')
            logger.info(f'p-value: {result[1]:.4f}')
            logger.info(f'Critical Values:')
            for key, value in result[4].items():
                logger.info(f'\t{key}: {value:.3f}')
            
            # Interpretation
            if result[1] <= 0.05:
                logger.info(f"Result: {title} is stationary (reject H0)")
                return True
            else:
                logger.info(f"Result: {title} is non-stationary (fail to reject H0)")
                return False
                
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return None
    
    def make_stationary(self, timeseries, d=1):
        """Make time series stationary through differencing"""
        try:
            # Apply differencing
            diff_series = timeseries.diff(d).dropna()
            return diff_series
            
        except Exception as e:
            logger.error(f"Error making series stationary: {e}")
            raise CustomException("Failed to make series stationary", e)
    
    def train_test_split_timeseries(self, df, test_size=0.2, date_col='date'):
        """Split time series data maintaining temporal order"""
        try:
            df = df.sort_values(date_col)
            
            split_index = int(len(df) * (1 - test_size))
            
            train = df.iloc[:split_index]
            test = df.iloc[split_index:]
            
            logger.info(f"Train set: {len(train)} samples ({train[date_col].min()} to {train[date_col].max()})")
            logger.info(f"Test set: {len(test)} samples ({test[date_col].min()} to {test[date_col].max()})")
            
            return train, test
            
        except Exception as e:
            logger.error(f"Error splitting time series: {e}")
            raise CustomException("Failed to split time series", e)
    
    def save_prepared_data(self, output_path):
        """Save all prepared time series data"""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Prepare all aggregations
            daily_sales = self.prepare_daily_sales()
            weekly_sales = self.prepare_weekly_sales()
            monthly_sales = self.prepare_monthly_sales()
            
            # Add time features
            daily_sales = self.add_time_features(daily_sales)
            
            # Save to CSV
            daily_sales.to_csv(f"{output_path}/daily_sales.csv", index=False)
            weekly_sales.to_csv(f"{output_path}/weekly_sales.csv", index=False)
            monthly_sales.to_csv(f"{output_path}/monthly_sales.csv", index=False)
            
            # Save summary statistics
            summary = {
                'date_range': {
                    'start': str(daily_sales['date'].min()),
                    'end': str(daily_sales['date'].max())
                },
                'total_days': len(daily_sales),
                'total_revenue': float(daily_sales['revenue'].sum()),
                'avg_daily_revenue': float(daily_sales['revenue'].mean()),
                'total_orders': int(daily_sales['order_count'].sum())
            }
            
            import json
            with open(f"{output_path}/data_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved prepared time series data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prepared data: {e}")
            raise CustomException("Failed to save prepared data", e)

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TimeSeriesPreprocessor("data")
    preprocessor.load_data()
    
    # Prepare and save data
    preprocessor.save_prepared_data("artifacts/forecasting/prepared_data")