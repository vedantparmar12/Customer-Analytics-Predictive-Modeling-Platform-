import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class BusinessMetricsCalculator:
    def __init__(self, customer_features_path, output_path="artifacts/business_metrics"):
        self.customer_features_path = customer_features_path
        self.output_path = output_path
        self.df = None
        
        logger.info("Business Metrics Calculator initialized")
    
    def load_data(self):
        """Load customer features data"""
        try:
            self.df = pd.read_csv(self.customer_features_path)
            logger.info(f"Loaded customer data: {self.df.shape}")
            
            # Ensure required columns exist or create defaults
            required_columns = {
                'monetary_value': 0,
                'frequency': 1,
                'recency_days': 365,
                'customer_lifetime_days': 0,
                'avg_order_value': 0,
                'churned': 0
            }
            
            for col, default_value in required_columns.items():
                if col not in self.df.columns:
                    logger.warning(f"Column '{col}' not found, creating with default value {default_value}")
                    # Try to find alternative columns
                    if col == 'monetary_value' and 'total_revenue' in self.df.columns:
                        self.df['monetary_value'] = self.df['total_revenue']
                    elif col == 'frequency' and 'total_orders' in self.df.columns:
                        self.df['frequency'] = self.df['total_orders']
                    elif col == 'avg_order_value' and 'monetary_value' in self.df.columns and 'frequency' in self.df.columns:
                        self.df['avg_order_value'] = self.df['monetary_value'] / self.df['frequency'].clip(lower=1)
                    else:
                        self.df[col] = default_value
                        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load customer features", e)
    
    def calculate_clv(self, time_period_months=12, discount_rate=0.1):
        """Calculate Customer Lifetime Value with multiple methods"""
        try:
            logger.info("Calculating Customer Lifetime Value...")
            
            # Method 1: Historical CLV (actual value generated)
            self.df['historical_clv'] = self.df['monetary_value']
            
            # Method 2: Simple CLV projection
            # CLV = (Average Order Value) × (Purchase Frequency) × (Customer Lifespan)
            avg_customer_lifespan_months = self.df['customer_lifetime_days'].mean() / 30
            self.df['simple_clv'] = (
                self.df['avg_order_value'] * 
                (self.df['frequency'] / avg_customer_lifespan_months * time_period_months)
            )
            
            # Method 3: Predictive CLV with retention probability
            # Calculate monthly retention rate from data
            active_months = (self.df['customer_lifetime_days'] / 30).clip(lower=1)
            monthly_orders = self.df['frequency'] / active_months
            
            # Estimate retention probability based on recency
            retention_prob = 1 - (self.df['recency_days'] / 365).clip(upper=1)
            
            # Projected CLV with retention
            monthly_value = self.df['avg_order_value'] * monthly_orders
            
            # Calculate NPV of expected future cash flows
            clv_projected = 0
            for month in range(time_period_months):
                month_discount = (1 + discount_rate/12) ** month
                month_retention = retention_prob ** month
                clv_projected += (monthly_value * month_retention) / month_discount
            
            self.df['predictive_clv'] = clv_projected
            
            # Method 4: Segment-based CLV (if segment exists)
            if 'customer_segment' in self.df.columns:
                segment_clv = self.df.groupby('customer_segment').agg({
                    'monetary_value': 'mean',
                    'frequency': 'mean',
                    'customer_lifetime_days': 'mean'
                })
                
                segment_clv['segment_clv'] = (
                    segment_clv['monetary_value'] * 
                    (segment_clv['frequency'] / (segment_clv['customer_lifetime_days'] / 30) * time_period_months)
                )
                
                # Merge segment CLV back
                self.df = self.df.merge(
                    segment_clv[['segment_clv']], 
                    left_on='customer_segment', 
                    right_index=True, 
                    how='left'
                )
            else:
                # If no segment, use average CLV
                self.df['segment_clv'] = self.df['predictive_clv']
            
            # CLV tiers
            self.df['clv_tier'] = pd.qcut(
                self.df['predictive_clv'], 
                q=5, 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            
            # Save CLV analysis
            clv_summary = pd.DataFrame({
                'metric': ['Total Historical CLV', 'Average Historical CLV', 'Total Predictive CLV', 
                          'Average Predictive CLV', 'CLV Standard Deviation'],
                'value': [
                    self.df['historical_clv'].sum(),
                    self.df['historical_clv'].mean(),
                    self.df['predictive_clv'].sum(),
                    self.df['predictive_clv'].mean(),
                    self.df['predictive_clv'].std()
                ]
            })
            
            clv_summary.to_csv(f"{self.output_path}/clv_summary.csv", index=False)
            
            logger.info("CLV calculation completed")
            return self.df
            
        except Exception as e:
            logger.error(f"Error calculating CLV: {e}")
            raise CustomException("Failed to calculate CLV", e)
    
    def calculate_customer_profitability(self, acquisition_cost=50, service_cost_per_order=5):
        """Calculate customer profitability metrics"""
        try:
            logger.info("Calculating customer profitability...")
            
            # Revenue components
            self.df['gross_revenue'] = self.df['monetary_value']
            
            # Cost components
            self.df['acquisition_cost'] = acquisition_cost
            self.df['total_service_cost'] = self.df['frequency'] * service_cost_per_order
            
            # Estimate return cost (based on review scores if available)
            if 'review_score' in self.df.columns:
                self.df['estimated_return_rate'] = self.df['review_score'].map({
                    1: 0.15, 2: 0.10, 3: 0.05, 4: 0.02, 5: 0.01
                }).fillna(0.05)
            elif 'avg_review_score' in self.df.columns:
                # Use average review score if individual scores not available
                self.df['estimated_return_rate'] = self.df['avg_review_score'].map({
                    1: 0.15, 2: 0.10, 3: 0.05, 4: 0.02, 5: 0.01
                }).fillna(0.05)
            else:
                # Default return rate if no review data
                self.df['estimated_return_rate'] = 0.05
            
            self.df['estimated_return_cost'] = self.df['gross_revenue'] * self.df['estimated_return_rate'] * 0.2
            
            # Total cost
            self.df['total_cost'] = (
                self.df['acquisition_cost'] + 
                self.df['total_service_cost'] + 
                self.df['estimated_return_cost']
            )
            
            # Profitability metrics
            self.df['gross_profit'] = self.df['gross_revenue'] - self.df['total_cost']
            self.df['profit_margin'] = self.df['gross_profit'] / self.df['gross_revenue']
            self.df['roi'] = (self.df['gross_profit'] / self.df['total_cost']) * 100
            
            # Profitability segments
            self.df['profitability_segment'] = pd.cut(
                self.df['profit_margin'],
                bins=[-np.inf, 0, 0.1, 0.2, 0.3, np.inf],
                labels=['Loss Making', 'Low Profit', 'Medium Profit', 'High Profit', 'Very High Profit']
            )
            
            # Segment analysis (if segment exists)
            if 'customer_segment' in self.df.columns:
                profitability_by_segment = self.df.groupby('customer_segment').agg({
                    'gross_revenue': 'sum',
                    'gross_profit': 'sum',
                    'profit_margin': 'mean',
                    'roi': 'mean',
                    'customer_unique_id': 'count'
                }).round(2)
                
                profitability_by_segment.to_csv(f"{self.output_path}/profitability_by_segment.csv")
            
            logger.info("Profitability calculation completed")
            return self.df
            
        except Exception as e:
            logger.error(f"Error calculating profitability: {e}")
            raise CustomException("Failed to calculate profitability", e)
    
    def calculate_retention_metrics(self):
        """Calculate detailed retention and churn metrics"""
        try:
            logger.info("Calculating retention metrics...")
            
            # Basic retention metrics
            self.df['is_retained'] = (~self.df['churned']).astype(int)
            overall_retention_rate = self.df['is_retained'].mean()
            
            # Cohort-based retention
            self.df['first_purchase_date'] = pd.to_datetime(self.df['first_purchase_date'])
            self.df['cohort_month'] = self.df['first_purchase_date'].dt.to_period('M')
            
            # Calculate months since first purchase
            reference_date = self.df['last_purchase_date'].max()
            self.df['months_since_first_purchase'] = (
                (pd.to_datetime(reference_date) - self.df['first_purchase_date']).dt.days / 30
            ).astype(int)
            
            # Retention by cohort and month
            cohort_retention = self.df.groupby(['cohort_month', 'months_since_first_purchase']).agg({
                'customer_unique_id': 'nunique',
                'is_retained': 'mean'
            }).reset_index()
            
            # Churn probability by customer characteristics
            # Churn factors analysis
            groupby_cols = []
            if 'customer_segment' in self.df.columns:
                groupby_cols.append('customer_segment')
            if 'clv_tier' in self.df.columns:
                groupby_cols.append('clv_tier')
            
            if groupby_cols:
                churn_factors = self.df.groupby(groupby_cols).agg({
                'churned': ['mean', 'count'],
                'monetary_value': 'sum'
            }).round(3)
            
            churn_factors.columns = ['churn_rate', 'customer_count', 'revenue_at_risk']
            churn_factors.to_csv(f"{self.output_path}/churn_risk_analysis.csv")
            
            # Survival analysis prep
            self.df['survival_days'] = self.df['customer_lifetime_days']
            self.df['event_observed'] = self.df['churned']
            
            # Early warning indicators
            if 'avg_days_between_orders' in self.df.columns:
                self.df['declining_frequency'] = (self.df['avg_days_between_orders'] > 60).astype(int)
            else:
                # Alternative: use recency as a proxy
                self.df['declining_frequency'] = (self.df['recency_days'] > 180).astype(int)
            
            if 'avg_order_value' in self.df.columns:
                self.df['declining_value'] = (self.df['avg_order_value'] < self.df['monetary_value'] / self.df['frequency']).astype(int)
            else:
                # Alternative: use current average
                avg_value = self.df['monetary_value'] / self.df['frequency'].clip(lower=1)
                self.df['declining_value'] = 0  # Default to no decline
            
            if 'review_score' in self.df.columns:
                self.df['poor_experience'] = (self.df['review_score'] < 3).astype(int)
            elif 'avg_review_score' in self.df.columns:
                self.df['poor_experience'] = (self.df['avg_review_score'] < 3).astype(int)
            else:
                self.df['poor_experience'] = 0  # Default to no poor experience
            
            self.df['churn_risk_score'] = (
                self.df['declining_frequency'] * 0.4 +
                self.df['declining_value'] * 0.3 +
                self.df['poor_experience'] * 0.3
            )
            
            logger.info("Retention metrics calculation completed")
            return cohort_retention
            
        except Exception as e:
            logger.error(f"Error calculating retention metrics: {e}")
            raise CustomException("Failed to calculate retention metrics", e)
    
    def calculate_marketing_metrics(self):
        """Calculate marketing effectiveness metrics"""
        try:
            logger.info("Calculating marketing metrics...")
            
            # Customer Acquisition Cost (CAC) by channel
            # Simulating channel data based on customer characteristics
            self.df['acquisition_channel'] = self.df.apply(
                lambda x: 'Organic' if x['frequency'] > 3 else 
                         ('Paid Search' if x['avg_order_value'] > 100 else 
                          ('Social Media' if x['customer_lifetime_days'] < 30 else 'Direct')), 
                axis=1
            )
            
            channel_cac = {
                'Organic': 10,
                'Paid Search': 75,
                'Social Media': 35,
                'Direct': 25
            }
            
            self.df['channel_cac'] = self.df['acquisition_channel'].map(channel_cac)
            
            # Calculate CAC payback period
            self.df['monthly_revenue'] = self.df['monetary_value'] / (self.df['customer_lifetime_days'] / 30).clip(lower=1)
            self.df['cac_payback_months'] = self.df['channel_cac'] / self.df['monthly_revenue'].clip(lower=1)
            
            # Marketing ROI by channel
            channel_metrics = self.df.groupby('acquisition_channel').agg({
                'customer_unique_id': 'count',
                'monetary_value': ['sum', 'mean'],
                'channel_cac': 'mean',
                'cac_payback_months': 'mean',
                'churned': 'mean'
            }).round(2)
            
            channel_metrics.columns = ['customers', 'total_revenue', 'avg_revenue', 
                                      'avg_cac', 'payback_months', 'churn_rate']
            channel_metrics['roi'] = ((channel_metrics['total_revenue'] - 
                                     channel_metrics['customers'] * channel_metrics['avg_cac']) / 
                                    (channel_metrics['customers'] * channel_metrics['avg_cac']) * 100)
            
            channel_metrics.to_csv(f"{self.output_path}/channel_performance.csv")
            
            # Campaign effectiveness simulation
            # Simulate response to different campaign types
            self.df['email_responsive'] = (
                (self.df['frequency'] > 2) & 
                (self.df['avg_order_value'] > 50)
            ).astype(int)
            
            self.df['discount_responsive'] = (
                self.df.get('is_bargain_hunter', pd.Series([0]*len(self.df))) == 1
            ).astype(int)
            
            self.df['loyalty_program_candidate'] = (
                (('customer_segment' in self.df.columns and self.df['customer_segment'].isin(['Champions', 'Loyal Customers']))) |
                (self.df['frequency'] > 5)
            ).astype(int)
            
            # Marketing mix effectiveness
            marketing_mix = pd.DataFrame({
                'campaign_type': ['Email', 'Discount', 'Loyalty Program'],
                'target_audience_size': [
                    self.df['email_responsive'].sum(),
                    self.df['discount_responsive'].sum(),
                    self.df['loyalty_program_candidate'].sum()
                ],
                'expected_response_rate': [0.15, 0.25, 0.35],
                'avg_order_increase': [20, 15, 50]
            })
            
            marketing_mix['expected_revenue_impact'] = (
                marketing_mix['target_audience_size'] * 
                marketing_mix['expected_response_rate'] * 
                marketing_mix['avg_order_increase']
            )
            
            marketing_mix.to_csv(f"{self.output_path}/marketing_mix_analysis.csv", index=False)
            
            logger.info("Marketing metrics calculation completed")
            
        except Exception as e:
            logger.error(f"Error calculating marketing metrics: {e}")
            raise CustomException("Failed to calculate marketing metrics", e)
    
    def create_executive_dashboard_data(self):
        """Create data for executive dashboard visualizations"""
        try:
            logger.info("Creating executive dashboard data...")
            
            # Key Performance Indicators
            kpis = {
                'total_customers': len(self.df),
                'active_customers': len(self.df[self.df['churned'] == 0]),
                'churn_rate': self.df['churned'].mean() * 100,
                'total_revenue': self.df['monetary_value'].sum(),
                'avg_customer_value': self.df['monetary_value'].mean(),
                'total_clv': self.df['predictive_clv'].sum(),
                'avg_clv': self.df['predictive_clv'].mean(),
                'total_profit': self.df['gross_profit'].sum(),
                'avg_profit_margin': self.df['profit_margin'].mean() * 100,
                'retention_rate': (1 - self.df['churned'].mean()) * 100
            }
            
            # Revenue trends by cohort
            revenue_by_cohort = self.df.groupby('cohort_month').agg({
                'monetary_value': 'sum',
                'customer_unique_id': 'count',
                'churned': 'mean'
            }).reset_index()
            
            revenue_by_cohort.columns = ['cohort', 'revenue', 'customers', 'churn_rate']
            # Convert Period to string for JSON serialization
            revenue_by_cohort['cohort'] = revenue_by_cohort['cohort'].astype(str)
            
            # Customer distribution metrics
            customer_distribution = {
                'by_segment': self.df['customer_segment'].value_counts().to_dict() if 'customer_segment' in self.df.columns else {},
                'by_clv_tier': self.df['clv_tier'].value_counts().to_dict(),
                'by_profitability': self.df['profitability_segment'].value_counts().to_dict(),
                'by_region': self.df.get('region', pd.Series()).value_counts().to_dict()
            }
            
            # Growth metrics
            monthly_growth = self.df.groupby(
                pd.to_datetime(self.df['first_purchase_date']).dt.to_period('M')
            ).agg({
                'customer_unique_id': 'count',
                'monetary_value': 'sum'
            }).reset_index()
            
            monthly_growth.columns = ['month', 'new_customers', 'revenue']
            # Convert Period to string for JSON serialization
            monthly_growth['month'] = monthly_growth['month'].astype(str)
            monthly_growth['customer_growth_rate'] = monthly_growth['new_customers'].pct_change() * 100
            monthly_growth['revenue_growth_rate'] = monthly_growth['revenue'].pct_change() * 100
            
            # Save all dashboard data
            pd.DataFrame([kpis]).T.to_csv(f"{self.output_path}/kpi_summary.csv")
            revenue_by_cohort.to_csv(f"{self.output_path}/revenue_by_cohort.csv", index=False)
            monthly_growth.to_csv(f"{self.output_path}/monthly_growth.csv", index=False)
            
            # Create dashboard JSON for easy loading
            import json
            dashboard_data = {
                'kpis': kpis,
                'customer_distribution': customer_distribution,
                'recent_cohort_performance': revenue_by_cohort.tail(6).to_dict('records'),
                'growth_trend': monthly_growth.tail(12).to_dict('records')
            }
            
            with open(f"{self.output_path}/dashboard_data.json", 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info("Executive dashboard data created successfully")
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error creating dashboard data: {e}")
            raise CustomException("Failed to create dashboard data", e)
    
    def generate_business_insights(self):
        """Generate actionable business insights"""
        try:
            logger.info("Generating business insights...")
            
            insights = []
            
            # Insight 1: High-value customer identification
            high_value_customers = self.df[self.df['clv_tier'] == 'Very High']
            insights.append({
                'category': 'Customer Value',
                'insight': f"{len(high_value_customers)} high-value customers ({len(high_value_customers)/len(self.df)*100:.1f}%) generate {high_value_customers['monetary_value'].sum()/self.df['monetary_value'].sum()*100:.1f}% of revenue",
                'action': 'Create VIP program with exclusive benefits',
                'impact': f"${high_value_customers['predictive_clv'].sum():,.0f} in projected CLV"
            })
            
            # Insight 2: Churn risk
            high_risk_valuable = self.df[
                (self.df['churn_risk_score'] > 0.7) & 
                (self.df['monetary_value'] > self.df['monetary_value'].median())
            ]
            insights.append({
                'category': 'Retention',
                'insight': f"{len(high_risk_valuable)} valuable customers at high churn risk",
                'action': 'Launch targeted retention campaign with personalized offers',
                'impact': f"${high_risk_valuable['monetary_value'].sum():,.0f} revenue at risk"
            })
            
            # Insight 3: Segment opportunities
            # Segment growth potential
            if 'customer_segment' in self.df.columns:
                segment_potential = self.df.groupby('customer_segment').agg({
                    'customer_unique_id': 'count',
                    'monetary_value': 'mean',
                    'churned': 'mean'
                })
                
                growth_segments = segment_potential[
                    (segment_potential['churned'] < 0.2) & 
                    (segment_potential['monetary_value'] > segment_potential['monetary_value'].median())
                ]
                
                for segment in growth_segments.index:
                    insights.append({
                        'category': 'Growth',
                        'insight': f"{segment} segment shows strong potential",
                        'action': f"Increase marketing spend for {segment} acquisition",
                        'impact': f"{growth_segments.loc[segment, 'customer_unique_id']} customers with ${growth_segments.loc[segment, 'monetary_value']:.0f} avg value"
                    })
            
            # Insight 4: Channel optimization
            if 'acquisition_channel' in self.df.columns:
                channel_roi = self.df.groupby('acquisition_channel').agg({
                    'roi': 'mean',
                    'customer_unique_id': 'count'
                }).sort_values('roi', ascending=False)
                
                best_channel = channel_roi.index[0]
                insights.append({
                    'category': 'Acquisition',
                    'insight': f"{best_channel} shows highest ROI at {channel_roi.loc[best_channel, 'roi']:.0f}%",
                    'action': f"Reallocate budget to {best_channel} channel",
                    'impact': "Improve overall CAC by 15-20%"
                })
            
            # Save insights
            insights_df = pd.DataFrame(insights)
            insights_df.to_csv(f"{self.output_path}/business_insights.csv", index=False)
            
            logger.info(f"Generated {len(insights)} business insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise CustomException("Failed to generate insights", e)
    
    def calculate_all_metrics(self):
        """Calculate all business metrics - main entry point for pipelines"""
        import os
        import json
        os.makedirs(self.output_path, exist_ok=True)
        
        try:
            # Load data
            self.load_data()
            
            # Calculate all metrics
            clv_metrics = self.calculate_clv()
            profitability_metrics = self.calculate_customer_profitability()
            retention_metrics = self.calculate_retention_metrics()
            marketing_metrics = self.calculate_marketing_metrics()
            dashboard_data = self.create_executive_dashboard_data()
            insights = self.generate_business_insights()
            
            # Compile all metrics
            all_metrics = {
                'kpis': {
                    'total_revenue': float(self.df['monetary_value'].sum() if 'monetary_value' in self.df.columns else 0),
                    'total_clv': float(self.df['clv'].sum() if 'clv' in self.df.columns else 0),
                    'avg_clv': float(self.df['clv'].mean() if 'clv' in self.df.columns else 0),
                    'churn_rate': float(self.df['churned'].mean() * 100 if 'churned' in self.df.columns else 0),
                    'retention_rate': float((1 - self.df['churned'].mean()) * 100 if 'churned' in self.df.columns else 100),
                    'total_customers': len(self.df),
                    'active_customers': len(self.df[self.df['churned'] == 0]) if 'churned' in self.df.columns else len(self.df)
                },
                'clv_metrics': clv_metrics,
                'profitability': profitability_metrics,
                'retention': retention_metrics,
                'marketing': marketing_metrics,
                'dashboard': dashboard_data,
                'insights': insights
            }
            
            # Save summary
            with open(f"{self.output_path}/kpi_summary.json", 'w') as f:
                json.dump(all_metrics['kpis'], f, indent=2)
            
            logger.info("All business metrics calculated successfully")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return basic metrics even on error
            return {
                'kpis': {
                    'total_revenue': 0,
                    'total_clv': 0,
                    'avg_clv': 0,
                    'churn_rate': 0,
                    'retention_rate': 100,
                    'total_customers': 0,
                    'active_customers': 0
                }
            }
    
    def run(self):
        """Run complete business metrics calculation"""
        import os
        os.makedirs(self.output_path, exist_ok=True)
        
        self.load_data()
        self.calculate_clv()
        self.calculate_customer_profitability()
        self.calculate_retention_metrics()
        self.calculate_marketing_metrics()
        dashboard_data = self.create_executive_dashboard_data()
        insights = self.generate_business_insights()
        
        logger.info("Business metrics calculation completed successfully")
        return self.df, dashboard_data, insights

if __name__ == "__main__":
    calculator = BusinessMetricsCalculator("artifacts/processed/customer_features.csv")
    calculator.run()