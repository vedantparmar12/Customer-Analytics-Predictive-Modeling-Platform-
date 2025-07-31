import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import os
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class SparkDataProcessor:
    def __init__(self, data_path, output_path="artifacts/processed"):
        self.data_path = data_path
        self.output_path = output_path
        self.spark = None
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Spark Data Processor initialized")
    
    def create_spark_session(self):
        """Create Spark session for large-scale processing"""
        try:
            logger.info("Creating Spark session...")
            self.spark = SparkSession.builder \
                .appName("CustomerAnalytics") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.shuffle.partitions", "200") \
                .getOrCreate()
            
            logger.info("Spark session created successfully")
        except Exception as e:
            logger.error(f"Error creating Spark session: {e}")
            raise CustomException("Failed to create Spark session", e)
    
    def load_data_spark(self):
        """Load data using Spark for better performance"""
        try:
            logger.info("Loading data with Spark...")
            
            # Load all datasets
            self.customers_df = self.spark.read.csv(
                os.path.join(self.data_path, "olist_customers_dataset.csv"), 
                header=True, inferSchema=True
            )
            self.orders_df = self.spark.read.csv(
                os.path.join(self.data_path, "olist_orders_dataset.csv"), 
                header=True, inferSchema=True
            )
            self.order_items_df = self.spark.read.csv(
                os.path.join(self.data_path, "olist_order_items_dataset.csv"), 
                header=True, inferSchema=True
            )
            self.order_payments_df = self.spark.read.csv(
                os.path.join(self.data_path, "olist_order_payments_dataset.csv"), 
                header=True, inferSchema=True
            )
            
            # Cache frequently used dataframes
            self.orders_df.cache()
            self.order_items_df.cache()
            
            logger.info(f"Loaded {self.customers_df.count()} customers")
            logger.info(f"Loaded {self.orders_df.count()} orders")
            
        except Exception as e:
            logger.error(f"Error loading data with Spark: {e}")
            raise CustomException("Failed to load data with Spark", e)
    
    def create_customer_features_spark(self):
        """Create customer features using Spark for scalability"""
        try:
            logger.info("Creating customer features with Spark...")
            
            # Convert timestamp columns
            timestamp_cols = ['order_purchase_timestamp', 'order_delivered_customer_date']
            for col in timestamp_cols:
                self.orders_df = self.orders_df.withColumn(
                    col, F.to_timestamp(F.col(col))
                )
            
            # Join orders with items and payments
            order_details = self.orders_df \
                .join(self.order_items_df, "order_id") \
                .join(self.order_payments_df, "order_id") \
                .join(self.customers_df, "customer_id")
            
            # Get reference date
            reference_date = self.orders_df.agg(
                F.max("order_purchase_timestamp")
            ).collect()[0][0]
            
            # Customer aggregations
            customer_features = order_details.groupBy("customer_unique_id").agg(
                F.countDistinct("order_id").alias("total_orders"),
                F.sum("payment_value").alias("total_revenue"),
                F.avg("payment_value").alias("avg_order_value"),
                F.stddev("payment_value").alias("order_value_std"),
                F.min("order_purchase_timestamp").alias("first_purchase_date"),
                F.max("order_purchase_timestamp").alias("last_purchase_date"),
                F.countDistinct("product_id").alias("unique_products_purchased"),
                F.first("customer_state").alias("customer_state"),
                F.first("customer_city").alias("customer_city"),
                F.avg(F.datediff("order_delivered_customer_date", "order_purchase_timestamp")).alias("avg_delivery_days")
            )
            
            # Calculate recency, frequency, monetary
            customer_features = customer_features.withColumn(
                "recency_days",
                F.datediff(F.lit(reference_date), F.col("last_purchase_date"))
            ).withColumn(
                "frequency", F.col("total_orders")
            ).withColumn(
                "monetary_value", F.col("total_revenue")
            )
            
            # Customer lifetime
            customer_features = customer_features.withColumn(
                "customer_lifetime_days",
                F.datediff(F.col("last_purchase_date"), F.col("first_purchase_date"))
            )
            
            # Average days between orders
            customer_features = customer_features.withColumn(
                "avg_days_between_orders",
                F.when(F.col("total_orders") > 1,
                      F.col("customer_lifetime_days") / (F.col("total_orders") - 1)
                ).otherwise(0)
            )
            
            # Churn label
            customer_features = customer_features.withColumn(
                "churned",
                F.when(F.col("recency_days") > 90, 1).otherwise(0)
            )
            
            # Product diversity score
            customer_features = customer_features.withColumn(
                "product_diversity_score",
                F.col("unique_products_purchased") / F.col("total_orders")
            )
            
            # Calculate RFM scores using percentiles
            r_percentiles = customer_features.select(
                F.expr("percentile_approx(recency_days, array(0.2, 0.4, 0.6, 0.8))").alias("r_percentiles")
            ).collect()[0]["r_percentiles"]
            
            f_percentiles = customer_features.select(
                F.expr("percentile_approx(frequency, array(0.2, 0.4, 0.6, 0.8))").alias("f_percentiles")
            ).collect()[0]["f_percentiles"]
            
            m_percentiles = customer_features.select(
                F.expr("percentile_approx(monetary_value, array(0.2, 0.4, 0.6, 0.8))").alias("m_percentiles")
            ).collect()[0]["m_percentiles"]
            
            # Assign RFM scores
            customer_features = customer_features.withColumn(
                "R_score",
                F.when(F.col("recency_days") <= r_percentiles[0], 5)
                .when(F.col("recency_days") <= r_percentiles[1], 4)
                .when(F.col("recency_days") <= r_percentiles[2], 3)
                .when(F.col("recency_days") <= r_percentiles[3], 2)
                .otherwise(1)
            )
            
            customer_features = customer_features.withColumn(
                "F_score",
                F.when(F.col("frequency") >= f_percentiles[3], 5)
                .when(F.col("frequency") >= f_percentiles[2], 4)
                .when(F.col("frequency") >= f_percentiles[1], 3)
                .when(F.col("frequency") >= f_percentiles[0], 2)
                .otherwise(1)
            )
            
            customer_features = customer_features.withColumn(
                "M_score",
                F.when(F.col("monetary_value") >= m_percentiles[3], 5)
                .when(F.col("monetary_value") >= m_percentiles[2], 4)
                .when(F.col("monetary_value") >= m_percentiles[1], 3)
                .when(F.col("monetary_value") >= m_percentiles[0], 2)
                .otherwise(1)
            )
            
            # Create RFM segments
            customer_features = customer_features.withColumn(
                "customer_segment",
                F.when((F.col("R_score") >= 4) & (F.col("F_score") >= 4) & (F.col("M_score") >= 4), "Champions")
                .when((F.col("R_score") >= 3) & (F.col("F_score") >= 3) & (F.col("M_score") >= 4), "Loyal Customers")
                .when((F.col("R_score") >= 3) & (F.col("F_score") <= 2) & (F.col("M_score") >= 3), "Potential Loyalists")
                .when((F.col("R_score") >= 4) & (F.col("F_score") <= 2), "New Customers")
                .when((F.col("R_score") <= 2) & (F.col("F_score") >= 3) & (F.col("M_score") >= 3), "At Risk")
                .when((F.col("R_score") <= 2) & (F.col("F_score") >= 4) & (F.col("M_score") >= 4), "Cant Lose Them")
                .when((F.col("R_score") <= 2) & (F.col("F_score") <= 2) & (F.col("M_score") <= 2), "Lost")
                .otherwise("Others")
            )
            
            # Save to parquet for better performance
            customer_features.write.mode("overwrite").parquet(
                os.path.join(self.output_path, "customer_features_spark.parquet")
            )
            
            # Also save as CSV for compatibility
            customer_features.toPandas().to_csv(
                os.path.join(self.output_path, "customer_features_spark.csv"),
                index=False
            )
            
            logger.info(f"Created features for {customer_features.count()} customers")
            
            return customer_features
            
        except Exception as e:
            logger.error(f"Error creating features with Spark: {e}")
            raise CustomException("Failed to create features with Spark", e)
    
    def create_time_series_features(self):
        """Create time-series features for better predictions"""
        try:
            logger.info("Creating time series features...")
            
            # Add time-based features
            orders_with_time = self.orders_df.withColumn(
                "order_year", F.year("order_purchase_timestamp")
            ).withColumn(
                "order_month", F.month("order_purchase_timestamp")
            ).withColumn(
                "order_day_of_week", F.dayofweek("order_purchase_timestamp")
            ).withColumn(
                "order_hour", F.hour("order_purchase_timestamp")
            ).withColumn(
                "is_weekend", 
                F.when(F.col("order_day_of_week").isin([1, 7]), 1).otherwise(0)
            )
            
            # Customer ordering patterns
            window_spec = Window.partitionBy("customer_id").orderBy("order_purchase_timestamp")
            
            customer_patterns = orders_with_time.withColumn(
                "prev_order_timestamp",
                F.lag("order_purchase_timestamp").over(window_spec)
            ).withColumn(
                "days_since_last_order",
                F.datediff("order_purchase_timestamp", "prev_order_timestamp")
            )
            
            # Aggregate patterns
            pattern_features = customer_patterns.groupBy("customer_id").agg(
                F.avg("days_since_last_order").alias("avg_days_between_orders_pattern"),
                F.stddev("days_since_last_order").alias("std_days_between_orders"),
                F.avg("is_weekend").alias("weekend_order_ratio"),
                F.mode("order_hour").alias("preferred_order_hour"),
                F.mode("order_day_of_week").alias("preferred_order_day")
            )
            
            # Save pattern features
            pattern_features.write.mode("overwrite").parquet(
                os.path.join(self.output_path, "customer_pattern_features.parquet")
            )
            
            logger.info("Time series features created")
            
        except Exception as e:
            logger.error(f"Error creating time series features: {e}")
            raise CustomException("Failed to create time series features", e)
    
    def run(self):
        """Run Spark data processing pipeline"""
        self.create_spark_session()
        self.load_data_spark()
        customer_features = self.create_customer_features_spark()
        self.create_time_series_features()
        
        # Stop Spark session
        self.spark.stop()
        
        logger.info("Spark data processing completed successfully")
        return customer_features

if __name__ == "__main__":
    processor = SparkDataProcessor("data")
    processor.run()