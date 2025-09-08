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
            
            # Map column names based on what's available
            revenue_col = 'total_revenue' if 'total_revenue' in self.df.columns else 'monetary_value'
            churned_col = 'churned' if 'churned' in self.df.columns else None
            
            # Basic info
            stats = {
                'Total Customers': len(self.df),
                'Total Revenue': f"${self.df[revenue_col].sum():,.2f}" if revenue_col in self.df.columns else "N/A",
                'Avg Revenue per Customer': f"${self.df[revenue_col].mean():.2f}" if revenue_col in self.df.columns else "N/A",
                'Avg Orders per Customer': f"{self.df['total_orders'].mean():.2f}" if 'total_orders' in self.df.columns else "N/A"
            }
            
            # Add churn stats if available
            if churned_col and churned_col in self.df.columns:
                stats['Churned Customers'] = self.df[churned_col].sum()
                stats['Churn Rate'] = f"{self.df[churned_col].mean():.2%}"
            
            # Save statistics
            with open(os.path.join(self.output_path, "basic_statistics.txt"), 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
                    logger.info(f"{key}: {value}")
            
            # Customer segment distribution if available
            if 'customer_segment' in self.df.columns:
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
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                plt.style.use('seaborn-darkgrid')
            sns.set_palette("husl")
            
            # 1. Customer Distribution by State (if available)
            if 'customer_state' in self.df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                state_counts = self.df['customer_state'].value_counts().head(15)
                state_counts.plot(kind='bar', ax=ax)
                ax.set_title('Top 15 States by Customer Count', fontsize=16)
                ax.set_xlabel('State')
                ax.set_ylabel('Number of Customers')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, 'customer_by_state.png'), dpi=300)
                plt.close()
            
            # 2. Revenue and Order Distribution
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            if 'total_revenue' in self.df.columns:
                self.df['total_revenue'].hist(bins=50, ax=axes[0], color='skyblue', edgecolor='black')
                axes[0].set_title('Customer Revenue Distribution')
                axes[0].set_xlabel('Total Revenue')
                axes[0].set_ylabel('Number of Customers')
            
            if 'total_orders' in self.df.columns:
                self.df['total_orders'].hist(bins=30, ax=axes[1], color='lightcoral', edgecolor='black')
                axes[1].set_title('Customer Order Count Distribution')
                axes[1].set_xlabel('Total Orders')
                axes[1].set_ylabel('Number of Customers')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'rfm_distributions.png'), dpi=300)
            plt.close()
            
            # 3. Churn Analysis if available
            if 'churned' in self.df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                churn_dist = self.df['churned'].value_counts()
                churn_dist.index = ['Active', 'Churned']
                churn_dist.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['green', 'red'])
                ax.set_title('Customer Churn Distribution', fontsize=16)
                ax.set_ylabel('')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, 'churn_distribution.png'), dpi=300)
                plt.close()
            
            # 4. Customer Value Distribution
            revenue_col = 'total_revenue' if 'total_revenue' in self.df.columns else 'monetary_value'
            if revenue_col in self.df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                self.df[revenue_col].hist(bins=50, ax=ax, edgecolor='black')
                ax.set_title('Customer Revenue Distribution', fontsize=16)
                ax.set_xlabel('Total Revenue ($)')
                ax.set_ylabel('Number of Customers')
                ax.axvline(self.df[revenue_col].median(), color='red', linestyle='--', 
                          label=f'Median: ${self.df[revenue_col].median():.2f}')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, 'revenue_distribution.png'), dpi=300)
                plt.close()
            
            # 5. Correlation Heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            # Get numeric columns that actually exist in the dataframe
            potential_cols = ['recency_days', 'frequency', 'monetary_value', 'total_orders', 
                          'avg_order_value', 'unique_products_purchased', 'customer_lifetime_days',
                          'avg_days_between_orders', 'product_diversity_score', 'churned']
            numeric_cols = [col for col in potential_cols if col in self.df.columns]
            
            # If we don't have enough columns, use all numeric columns
            if len(numeric_cols) < 5:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:15].tolist()
            
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'correlation_matrix.png'), dpi=300)
            plt.close()
            
            # 6. Purchase Behavior by State (Top 10) - if state column exists
            if 'customer_state' in self.df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                top_states = self.df['customer_state'].value_counts().head(10)
                top_states.plot(kind='bar', ax=ax, color='teal')
                ax.set_title('Top 10 States by Number of Customers', fontsize=16)
                ax.set_xlabel('State')
                ax.set_ylabel('Number of Customers')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, 'customers_by_state.png'), dpi=300)
                plt.close()
            else:
                # Alternative visualization if no state data
                fig, ax = plt.subplots(figsize=(12, 6))
                # Show feature importance or distribution of another column
                if 'churned' in self.df.columns:
                    churn_dist = self.df['churned'].value_counts()
                    churn_dist.index = ['Active', 'Churned']
                    churn_dist.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['green', 'red'])
                    ax.set_title('Customer Churn Distribution', fontsize=16)
                    ax.set_ylabel('')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_path, 'churn_distribution.png'), dpi=300)
                    plt.close()
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise CustomException("Failed to create visualizations", e)
    
    def cohort_analysis(self):
        """Perform cohort analysis"""
        try:
            logger.info("Performing cohort analysis...")
            
            # Check if we have the necessary columns
            if 'first_purchase_date' in self.df.columns:
                # Convert dates if they're strings
                self.df['first_purchase_date'] = pd.to_datetime(self.df['first_purchase_date'])
                
                # Create cohort based on first purchase month
                self.df['cohort_month'] = self.df['first_purchase_date'].dt.to_period('M')
                
                # Calculate retention by cohort if churned column exists
                if 'churned' in self.df.columns:
                    cohort_data = self.df.groupby(['cohort_month', 'churned']).size().unstack(fill_value=0)
                    if 0 in cohort_data.columns and 1 in cohort_data.columns:
                        cohort_retention = cohort_data[0] / (cohort_data[0] + cohort_data[1])
                    else:
                        cohort_retention = pd.Series(dtype=float)
                else:
                    # Just count customers by cohort
                    cohort_retention = self.df.groupby('cohort_month').size()
                
                # Save cohort analysis
                cohort_retention.to_csv(os.path.join(self.output_path, 'cohort_analysis.csv'))
                logger.info("Cohort analysis completed")
            else:
                logger.info("Skipping cohort analysis - first_purchase_date not available")
            
        except Exception as e:
            logger.warning(f"Could not perform cohort analysis: {e}")
            # Don't raise exception - allow pipeline to continue
    
    def generate_comprehensive_report(self):
        """Generate comprehensive EDA report - alias for run()"""
        self.run()
    
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