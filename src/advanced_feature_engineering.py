import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import LabelEncoder
from .logger import get_logger
from .custom_exception import CustomException
import os
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

class AdvancedFeatureEngineering:
    def __init__(self, data_path="data", output_path="artifacts/processed"):
        self.data_path = data_path
        self.output_path = output_path
        self.brazil_holidays = holidays.Brazil()
        logger.info("Advanced Feature Engineering initialized")
    
    def create_product_category_features(self, df):
        """Create product category preference features"""
        try:
            logger.info("Creating product category features...")
            
            # Load product data
            products_df = pd.read_csv(f"{self.data_path}/olist_products_dataset.csv")
            category_translation = pd.read_csv(f"{self.data_path}/product_category_name_translation.csv")
            
            # Merge to get English category names
            products_df = products_df.merge(category_translation, on='product_category_name', how='left')
            df = df.merge(products_df[['product_id', 'product_category_name_english']], on='product_id', how='left')
            
            # Customer category preferences
            customer_category_stats = df.groupby(['customer_unique_id', 'product_category_name_english']).agg({
                'order_id': 'count',
                'price': ['sum', 'mean']
            }).reset_index()
            
            customer_category_stats.columns = ['customer_unique_id', 'category', 'category_orders', 
                                              'category_revenue', 'category_avg_price']
            
            # Find favorite category
            idx = customer_category_stats.groupby('customer_unique_id')['category_orders'].idxmax()
            customer_fav_category = customer_category_stats.loc[idx, ['customer_unique_id', 'category']]
            customer_fav_category.columns = ['customer_unique_id', 'favorite_category']
            
            # Category diversity score
            category_diversity = customer_category_stats.groupby('customer_unique_id').agg({
                'category': 'nunique',
                'category_orders': lambda x: x.std() / x.mean() if x.mean() > 0 else 0
            }).reset_index()
            
            category_diversity.columns = ['customer_unique_id', 'num_categories_purchased', 'category_concentration']
            
            # Merge features
            category_features = customer_fav_category.merge(category_diversity, on='customer_unique_id')
            
            # Add high-value category indicator
            high_value_categories = ['computers', 'electronics', 'watches_gifts', 'computers_accessories']
            category_features['prefers_high_value_items'] = category_features['favorite_category'].isin(high_value_categories).astype(int)
            
            logger.info(f"Created category features for {len(category_features)} customers")
            return category_features
            
        except Exception as e:
            logger.error(f"Error creating category features: {e}")
            raise CustomException("Failed to create category features", e)
    
    def create_temporal_features(self, df):
        """Create advanced temporal and seasonal features"""
        try:
            logger.info("Creating temporal features...")
            
            # Ensure datetime
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
            
            # Time-based features
            df['order_hour'] = df['order_purchase_timestamp'].dt.hour
            df['order_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
            df['order_day'] = df['order_purchase_timestamp'].dt.day
            df['order_month'] = df['order_purchase_timestamp'].dt.month
            df['order_quarter'] = df['order_purchase_timestamp'].dt.quarter
            df['order_year'] = df['order_purchase_timestamp'].dt.year
            
            # Weekend/weekday
            df['is_weekend'] = df['order_day_of_week'].isin([5, 6]).astype(int)
            
            # Time of day categories
            df['time_of_day'] = pd.cut(df['order_hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            
            # Holiday indicator
            df['is_holiday'] = df['order_purchase_timestamp'].apply(
                lambda x: 1 if x.date() in self.brazil_holidays else 0
            )
            
            # Days to nearest holiday
            df['days_to_holiday'] = df['order_purchase_timestamp'].apply(
                lambda x: min([abs((holiday - x.date()).days) 
                             for holiday in self.brazil_holidays.keys() 
                             if holiday.year == x.year])
            )
            
            # Seasonal features
            df['is_black_friday_period'] = ((df['order_month'] == 11) & 
                                           (df['order_day'] >= 20)).astype(int)
            df['is_christmas_period'] = ((df['order_month'] == 12) & 
                                        (df['order_day'] <= 25)).astype(int)
            
            # Customer temporal patterns
            customer_temporal = df.groupby('customer_unique_id').agg({
                'is_weekend': 'mean',
                'order_hour': ['mean', 'std'],
                'time_of_day': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
                'is_holiday': 'mean',
                'order_day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
            }).reset_index()
            
            customer_temporal.columns = ['customer_unique_id', 'weekend_order_ratio', 
                                        'avg_order_hour', 'order_hour_consistency',
                                        'preferred_time_of_day', 'holiday_order_ratio',
                                        'preferred_day_of_week']
            
            # Order frequency patterns
            customer_orders = df.groupby('customer_unique_id')['order_purchase_timestamp'].apply(list).reset_index()
            
            def calculate_order_patterns(timestamps):
                if len(timestamps) < 2:
                    return pd.Series({
                        'order_frequency_days': 0,
                        'order_frequency_std': 0,
                        'has_regular_pattern': 0,
                        'avg_days_between_orders': 0
                    })
                
                timestamps = sorted(timestamps)
                days_between = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
                
                return pd.Series({
                    'order_frequency_days': np.mean(days_between),
                    'order_frequency_std': np.std(days_between),
                    'has_regular_pattern': 1 if np.std(days_between) < np.mean(days_between) * 0.5 else 0,
                    'avg_days_between_orders': np.mean(days_between)
                })
            
            order_patterns = customer_orders['order_purchase_timestamp'].apply(calculate_order_patterns)
            customer_temporal = pd.concat([customer_temporal, order_patterns], axis=1)
            
            logger.info(f"Created temporal features for {len(customer_temporal)} customers")
            return customer_temporal
            
        except Exception as e:
            logger.error(f"Error creating temporal features: {e}")
            raise CustomException("Failed to create temporal features", e)
    
    def create_delivery_satisfaction_features(self, df):
        """Create delivery and satisfaction features"""
        try:
            logger.info("Creating delivery satisfaction features...")
            
            # Convert dates
            date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 
                        'order_delivered_carrier_date', 'order_estimated_delivery_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Delivery performance
            df['actual_delivery_time'] = (df['order_delivered_customer_date'] - 
                                         df['order_purchase_timestamp']).dt.days
            df['estimated_delivery_time'] = (df['order_estimated_delivery_date'] - 
                                            df['order_purchase_timestamp']).dt.days
            df['delivery_delay'] = (df['order_delivered_customer_date'] - 
                                   df['order_estimated_delivery_date']).dt.days
            df['carrier_delay'] = (df['order_delivered_carrier_date'] - 
                                  df['order_purchase_timestamp']).dt.days
            
            # Delivery satisfaction indicators
            df['on_time_delivery'] = (df['delivery_delay'] <= 0).astype(int)
            df['early_delivery'] = (df['delivery_delay'] < -2).astype(int)
            df['late_delivery'] = (df['delivery_delay'] > 2).astype(int)
            
            # Customer delivery experience
            customer_delivery = df.groupby('customer_unique_id').agg({
                'actual_delivery_time': ['mean', 'std'],
                'delivery_delay': ['mean', 'max', 'min'],
                'on_time_delivery': 'mean',
                'early_delivery': 'mean',
                'late_delivery': 'mean',
                'review_score': 'mean'
            }).reset_index()
            
            customer_delivery.columns = ['customer_unique_id', 'avg_delivery_time', 'delivery_time_variance',
                                        'avg_delivery_delay', 'worst_delay', 'best_early_delivery',
                                        'on_time_rate', 'early_delivery_rate', 'late_delivery_rate',
                                        'avg_review_score']
            
            # Delivery impact on satisfaction
            customer_delivery['delivery_satisfaction_correlation'] = customer_delivery.apply(
                lambda x: -1 if x['avg_delivery_delay'] > 3 and x['avg_review_score'] < 3 else 
                         (1 if x['on_time_rate'] > 0.8 and x['avg_review_score'] > 4 else 0), axis=1
            )
            
            logger.info(f"Created delivery features for {len(customer_delivery)} customers")
            return customer_delivery
            
        except Exception as e:
            logger.error(f"Error creating delivery features: {e}")
            raise CustomException("Failed to create delivery features", e)
    
    def create_customer_engagement_features(self, df):
        """Create customer engagement and interaction features"""
        try:
            logger.info("Creating customer engagement features...")
            
            # Review engagement
            # First ensure review_score is numeric
            df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce')
            
            review_engagement = df.groupby('customer_unique_id').agg({
                'review_score': ['count', 'mean', 'std'],
                'review_comment_message': lambda x: (x.notna()).sum(),
                'review_creation_date': lambda x: (pd.to_datetime(x, errors='coerce').max() - pd.to_datetime(x, errors='coerce').min()).days if len(x) > 1 else 0,
                'review_answer_timestamp': lambda x: (x.notna()).sum()
            }).reset_index()
            
            review_engagement.columns = ['customer_unique_id', 'total_reviews', 'avg_review_score', 
                                        'review_score_variance', 'reviews_with_comments',
                                        'review_span_days', 'reviews_with_seller_response']
            
            # Calculate engagement metrics
            review_engagement['review_rate'] = review_engagement['total_reviews'] / df.groupby('customer_unique_id')['order_id'].nunique().values
            review_engagement['comment_rate'] = review_engagement['reviews_with_comments'] / review_engagement['total_reviews'].clip(lower=1)
            review_engagement['is_vocal_customer'] = (review_engagement['comment_rate'] > 0.5).astype(int)
            
            # Payment behavior
            payment_behavior = df.groupby('customer_unique_id').agg({
                'payment_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
                'payment_installments': ['mean', 'max'],
                'payment_sequential': 'max'
            }).reset_index()
            
            payment_behavior.columns = ['customer_unique_id', 'preferred_payment_type', 
                                       'avg_installments', 'max_installments', 'max_payment_attempts']
            
            # Multi-payment indicator
            payment_behavior['uses_installments'] = (payment_behavior['avg_installments'] > 1).astype(int)
            payment_behavior['payment_complexity'] = payment_behavior['max_payment_attempts']
            
            # Merge engagement features
            engagement_features = review_engagement.merge(payment_behavior, on='customer_unique_id', how='outer')
            
            logger.info(f"Created engagement features for {len(engagement_features)} customers")
            return engagement_features
            
        except Exception as e:
            logger.error(f"Error creating engagement features: {e}")
            raise CustomException("Failed to create engagement features", e)
    
    def create_geographic_features(self, df):
        """Create geographic and demographic features"""
        try:
            logger.info("Creating geographic features...")
            
            # Load geolocation data
            geo_df = pd.read_csv(f"{self.data_path}/olist_geolocation_dataset.csv")
            
            # State-level aggregations
            state_stats = df.groupby('customer_state').agg({
                'price': ['mean', 'std'],
                'customer_unique_id': 'nunique',
                'review_score': 'mean'
            }).reset_index()
            
            state_stats.columns = ['state', 'state_avg_order_value', 'state_order_value_std',
                                  'state_customer_count', 'state_avg_satisfaction']
            
            # City-level features
            city_stats = df.groupby('customer_city').agg({
                'customer_unique_id': 'nunique',
                'price': 'mean'
            }).reset_index()
            
            city_stats.columns = ['city', 'city_customer_count', 'city_avg_order_value']
            
            # Customer geographic features
            customer_geo = df.groupby('customer_unique_id').agg({
                'customer_state': 'first',
                'customer_city': 'first',
                'customer_zip_code_prefix': 'first'
            }).reset_index()
            
            # Merge state and city stats
            customer_geo = customer_geo.merge(state_stats, left_on='customer_state', right_on='state', how='left')
            customer_geo = customer_geo.merge(city_stats, left_on='customer_city', right_on='city', how='left')
            
            # Economic indicators by region
            major_cities = ['sao paulo', 'rio de janeiro', 'brasilia', 'salvador', 'fortaleza', 
                           'belo horizonte', 'curitiba', 'recife', 'porto alegre']
            
            customer_geo['is_major_city'] = customer_geo['customer_city'].str.lower().isin(major_cities).astype(int)
            
            # Regional categories
            south_states = ['RS', 'SC', 'PR']
            southeast_states = ['SP', 'RJ', 'MG', 'ES']
            northeast_states = ['BA', 'PE', 'CE', 'MA', 'PB', 'RN', 'AL', 'PI', 'SE']
            north_states = ['AM', 'PA', 'AC', 'RO', 'RR', 'AP', 'TO']
            centerwest_states = ['MT', 'MS', 'GO', 'DF']
            
            def get_region(state):
                if state in south_states:
                    return 'South'
                elif state in southeast_states:
                    return 'Southeast'
                elif state in northeast_states:
                    return 'Northeast'
                elif state in north_states:
                    return 'North'
                elif state in centerwest_states:
                    return 'Center-West'
                else:
                    return 'Unknown'
            
            customer_geo['region'] = customer_geo['customer_state'].apply(get_region)
            
            # Drop redundant columns
            customer_geo = customer_geo.drop(['state', 'city'], axis=1)
            
            logger.info(f"Created geographic features for {len(customer_geo)} customers")
            return customer_geo
            
        except Exception as e:
            logger.error(f"Error creating geographic features: {e}")
            raise CustomException("Failed to create geographic features", e)
    
    def create_price_sensitivity_features(self, df):
        """Create price sensitivity and discount response features"""
        try:
            logger.info("Creating price sensitivity features...")
            
            # Price statistics per customer
            customer_price = df.groupby('customer_unique_id').agg({
                'price': ['mean', 'std', 'min', 'max', 'median'],
                'freight_value': ['mean', 'sum']
            }).reset_index()
            
            customer_price.columns = ['customer_unique_id', 'avg_item_price', 'price_variance',
                                     'min_price', 'max_price', 'median_price',
                                     'avg_freight_cost', 'total_freight_paid']
            
            # Price sensitivity indicators
            customer_price['price_range'] = customer_price['max_price'] - customer_price['min_price']
            customer_price['relative_price_variance'] = customer_price['price_variance'] / customer_price['avg_item_price'].clip(lower=1)
            customer_price['freight_ratio'] = customer_price['avg_freight_cost'] / customer_price['avg_item_price'].clip(lower=1)
            
            # Categorize price sensitivity
            customer_price['price_sensitivity'] = pd.cut(
                customer_price['relative_price_variance'],
                bins=[0, 0.2, 0.5, 1.0, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High']
            )
            
            # Discount affinity (assuming items below median price are "discounted")
            overall_median_price = df['price'].median()
            discount_purchases = df[df['price'] < overall_median_price].groupby('customer_unique_id').size()
            total_purchases = df.groupby('customer_unique_id').size()
            
            customer_price['discount_purchase_ratio'] = (discount_purchases / total_purchases).fillna(0)
            customer_price['is_bargain_hunter'] = (customer_price['discount_purchase_ratio'] > 0.7).astype(int)
            
            logger.info(f"Created price sensitivity features for {len(customer_price)} customers")
            return customer_price
            
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
            raise CustomException("Failed to create price features", e)
    
    def combine_all_features(self):
        """Combine all advanced features into a single dataset"""
        try:
            logger.info("Combining all advanced features...")
            
            # Load base data
            orders_df = pd.read_csv(f"{self.data_path}/olist_orders_dataset.csv")
            order_items_df = pd.read_csv(f"{self.data_path}/olist_order_items_dataset.csv")
            customers_df = pd.read_csv(f"{self.data_path}/olist_customers_dataset.csv")
            payments_df = pd.read_csv(f"{self.data_path}/olist_order_payments_dataset.csv")
            reviews_df = pd.read_csv(f"{self.data_path}/olist_order_reviews_dataset.csv")
            
            # Merge datasets
            df = orders_df.merge(order_items_df, on='order_id')
            df = df.merge(customers_df, on='customer_id')
            df = df.merge(payments_df, on='order_id')
            df = df.merge(reviews_df, on='order_id', how='left')
            
            # Create all feature sets
            category_features = self.create_product_category_features(df)
            temporal_features = self.create_temporal_features(df)
            delivery_features = self.create_delivery_satisfaction_features(df)
            engagement_features = self.create_customer_engagement_features(df)
            geographic_features = self.create_geographic_features(df)
            price_features = self.create_price_sensitivity_features(df)
            
            # Load basic customer features if available
            basic_features_path = f"{self.output_path}/customer_features.csv"
            if os.path.exists(basic_features_path):
                basic_features = pd.read_csv(basic_features_path)
            else:
                # Create basic features if not available
                logger.warning("Basic features not found, creating minimal features")
                # Create customer-level aggregation
                basic_features = df.groupby('customer_unique_id').agg({
                    'customer_state': 'first',
                    'customer_city': 'first',
                    'order_id': 'nunique'
                }).reset_index()
                basic_features.rename(columns={'order_id': 'total_orders'}, inplace=True)
                basic_features['churned'] = 0  # Default value
            
            # Merge all features
            all_features = basic_features
            for features_df in [category_features, temporal_features, delivery_features, 
                              engagement_features, geographic_features, price_features]:
                all_features = all_features.merge(features_df, on='customer_unique_id', how='left')
            
            # Save combined features
            all_features.to_csv(f"{self.output_path}/customer_features_advanced.csv", index=False)
            
            logger.info(f"Combined features shape: {all_features.shape}")
            logger.info(f"Total features: {len(all_features.columns)}")
            
            # Feature summary
            feature_summary = {
                'total_features': len(all_features.columns),
                'basic_features': len(basic_features.columns),
                'category_features': len(category_features.columns),
                'temporal_features': len(temporal_features.columns),
                'delivery_features': len(delivery_features.columns),
                'engagement_features': len(engagement_features.columns),
                'geographic_features': len(geographic_features.columns),
                'price_features': len(price_features.columns)
            }
            
            logger.info(f"Feature summary: {feature_summary}")
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error combining features: {e}")
            raise CustomException("Failed to combine features", e)
    
    def run(self):
        """Run advanced feature engineering pipeline"""
        all_features = self.combine_all_features()
        logger.info("Advanced feature engineering completed successfully")
        return all_features

if __name__ == "__main__":
    feature_engineer = AdvancedFeatureEngineering()
    feature_engineer.run()