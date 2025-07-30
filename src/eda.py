import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .logger import get_logger
from .custom_exception import CustomException
import os

logger = get_logger(__name__)

class CustomerAnalyticsEDA:
    def __init__(self, customer_features_path, output_path="artifacts/eda"):
        self.customer_features_path = customer_features_path
        self.output_path = output_path
        self.df = None
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Customer Analytics EDA initialized")
    
    def load_data(self):
        """Load customer features data"""
        try:
            self.df = pd.read_csv(self.customer_features_path)
            logger.info(f"Loaded customer features: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load customer features", e)
    
    def basic_statistics(self):
        """Generate basic statistics"""
        try:
            logger.info("Generating basic statistics...")
            
            # Basic info
            stats = {
                'Total Customers': len(self.df),
                'Churned Customers': self.df['churned'].sum(),
                'Churn Rate': f"{self.df['churned'].mean():.2%}",
                'Avg Customer Lifetime Value': f"${self.df['monetary_value'].mean():.2f}",
                'Total Revenue': f"${self.df['monetary_value'].sum():,.2f}",
                'Avg Orders per Customer': f"{self.df['total_orders'].mean():.2f}"
            }
            
            # Save statistics
            with open(os.path.join(self.output_path, "basic_statistics.txt"), 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                    logger.info(f"{key}: {value}")
            
            # Customer segment distribution
            segment_dist = self.df['customer_segment'].value_counts()
            segment_dist.to_csv(os.path.join(self.output_path, "segment_distribution.csv"))
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            raise CustomException("Failed to generate statistics", e)
    
    def create_visualizations(self):
        """Create key visualizations for customer analytics"""
        try:
            logger.info("Creating visualizations...")
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # 1. Customer Segment Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_counts = self.df['customer_segment'].value_counts()
            segment_counts.plot(kind='bar', ax=ax)
            ax.set_title('Customer Segment Distribution', fontsize=16)
            ax.set_xlabel('Customer Segment')
            ax.set_ylabel('Number of Customers')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'customer_segments.png'), dpi=300)
            plt.close()
            
            # 2. RFM Score Distribution
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            self.df['R_score'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title('Recency Score Distribution')
            axes[0].set_xlabel('Recency Score')
            
            self.df['F_score'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='lightcoral')
            axes[1].set_title('Frequency Score Distribution')
            axes[1].set_xlabel('Frequency Score')
            
            self.df['M_score'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='lightgreen')
            axes[2].set_title('Monetary Score Distribution')
            axes[2].set_xlabel('Monetary Score')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'rfm_distributions.png'), dpi=300)
            plt.close()
            
            # 3. Churn Analysis by Segment
            fig, ax = plt.subplots(figsize=(10, 6))
            churn_by_segment = self.df.groupby('customer_segment')['churned'].mean().sort_values(ascending=False)
            churn_by_segment.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Churn Rate by Customer Segment', fontsize=16)
            ax.set_xlabel('Customer Segment')
            ax.set_ylabel('Churn Rate')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'churn_by_segment.png'), dpi=300)
            plt.close()
            
            # 4. Customer Value Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            self.df['monetary_value'].hist(bins=50, ax=ax, edgecolor='black')
            ax.set_title('Customer Lifetime Value Distribution', fontsize=16)
            ax.set_xlabel('Total Spend ($)')
            ax.set_ylabel('Number of Customers')
            ax.axvline(self.df['monetary_value'].median(), color='red', linestyle='--', 
                      label=f'Median: ${self.df["monetary_value"].median():.2f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'clv_distribution.png'), dpi=300)
            plt.close()
            
            # 5. Correlation Heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            numeric_cols = ['recency_days', 'frequency', 'monetary_value', 'total_orders', 
                          'avg_order_value', 'unique_products_purchased', 'customer_lifetime_days',
                          'avg_days_between_orders', 'product_diversity_score', 'churned']
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'correlation_matrix.png'), dpi=300)
            plt.close()
            
            # 6. Purchase Behavior by State (Top 10)
            fig, ax = plt.subplots(figsize=(12, 6))
            top_states = self.df['customer_state'].value_counts().head(10)
            top_states.plot(kind='bar', ax=ax, color='teal')
            ax.set_title('Top 10 States by Number of Customers', fontsize=16)
            ax.set_xlabel('State')
            ax.set_ylabel('Number of Customers')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'customers_by_state.png'), dpi=300)
            plt.close()
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise CustomException("Failed to create visualizations", e)
    
    def cohort_analysis(self):
        """Perform cohort analysis"""
        try:
            logger.info("Performing cohort analysis...")
            
            # Convert dates if they're strings
            self.df['first_purchase_date'] = pd.to_datetime(self.df['first_purchase_date'])
            
            # Create cohort based on first purchase month
            self.df['cohort_month'] = self.df['first_purchase_date'].dt.to_period('M')
            
            # Calculate retention by cohort
            cohort_data = self.df.groupby(['cohort_month', 'churned']).size().unstack(fill_value=0)
            cohort_retention = cohort_data[0] / (cohort_data[0] + cohort_data[1])
            
            # Save cohort analysis
            cohort_retention.to_csv(os.path.join(self.output_path, 'cohort_retention.csv'))
            
            logger.info("Cohort analysis completed")
            
        except Exception as e:
            logger.error(f"Error in cohort analysis: {e}")
            raise CustomException("Failed to perform cohort analysis", e)
    
    def run(self):
        """Run complete EDA pipeline"""
        self.load_data()
        self.basic_statistics()
        self.create_visualizations()
        self.cohort_analysis()
        logger.info("EDA pipeline completed successfully")

if __name__ == "__main__":
    eda = CustomerAnalyticsEDA("artifacts/processed/customer_features.csv")
    eda.run()