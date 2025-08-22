import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class AdvancedCohortAnalysis:
    def __init__(self, data_path="data", output_path="artifacts/cohort_analysis"):
        self.data_path = data_path
        self.output_path = output_path
        self.cohort_data = None
        
        logger.info("Advanced Cohort Analysis initialized")
    
    def prepare_cohort_data(self, df=None):
        """Prepare data for cohort analysis"""
        try:
            logger.info("Preparing cohort data...")
            
            if df is None:
                # Load order data
                orders_df = pd.read_csv(f"{self.data_path}/olist_orders_dataset.csv")
                customers_df = pd.read_csv(f"{self.data_path}/olist_customers_dataset.csv")
                order_items_df = pd.read_csv(f"{self.data_path}/olist_order_items_dataset.csv")
                
                # Merge datasets
                df = orders_df.merge(customers_df[['customer_id', 'customer_unique_id', 'customer_state']], on='customer_id')
                df = df.merge(order_items_df[['order_id', 'price']], on='order_id')
            
            # Convert dates
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
            
            # Get first purchase date for each customer
            df['first_purchase_date'] = df.groupby('customer_unique_id')['order_purchase_timestamp'].transform('min')
            
            # Create cohort identifier (monthly)
            df['cohort_month'] = df['first_purchase_date'].dt.to_period('M')
            df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')
            
            # Calculate cohort index (months since first purchase)
            df['cohort_index'] = (df['order_month'] - df['cohort_month']).apply(lambda x: x.n if pd.notnull(x) else 0)
            
            self.cohort_data = df
            logger.info(f"Prepared cohort data with {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing cohort data: {e}")
            raise CustomException("Failed to prepare cohort data", e)
    
    def calculate_retention_matrix(self, metric='customers', period_type='monthly'):
        """Calculate retention matrix for different metrics"""
        try:
            logger.info(f"Calculating {metric} retention matrix...")
            
            if metric == 'customers':
                # Customer count retention
                cohort_counts = self.cohort_data.groupby(['cohort_month', 'cohort_index']).agg({
                    'customer_unique_id': pd.Series.nunique
                }).reset_index()
                
                cohort_matrix = cohort_counts.pivot_table(
                    index='cohort_month',
                    columns='cohort_index',
                    values='customer_unique_id'
                )
                
            elif metric == 'revenue':
                # Revenue retention
                cohort_revenue = self.cohort_data.groupby(['cohort_month', 'cohort_index']).agg({
                    'price': 'sum'
                }).reset_index()
                
                cohort_matrix = cohort_revenue.pivot_table(
                    index='cohort_month',
                    columns='cohort_index',
                    values='price'
                )
                
            elif metric == 'orders':
                # Order frequency retention
                cohort_orders = self.cohort_data.groupby(['cohort_month', 'cohort_index']).agg({
                    'order_id': pd.Series.nunique
                }).reset_index()
                
                cohort_matrix = cohort_orders.pivot_table(
                    index='cohort_month',
                    columns='cohort_index',
                    values='order_id'
                )
            
            # Calculate retention rates
            cohort_size = cohort_matrix.iloc[:, 0]
            retention_matrix = cohort_matrix.divide(cohort_size, axis=0) * 100
            
            # Save matrices
            import os
            os.makedirs(self.output_path, exist_ok=True)
            
            cohort_matrix.to_csv(f"{self.output_path}/{metric}_cohort_absolute.csv")
            retention_matrix.to_csv(f"{self.output_path}/{metric}_cohort_retention.csv")
            
            logger.info(f"Created retention matrix with shape {retention_matrix.shape}")
            
            return retention_matrix, cohort_matrix
            
        except Exception as e:
            logger.error(f"Error calculating retention matrix: {e}")
            raise CustomException("Failed to calculate retention matrix", e)
    
    def analyze_cohort_quality(self):
        """Analyze the quality and characteristics of different cohorts"""
        try:
            logger.info("Analyzing cohort quality...")
            
            # Calculate key metrics for each cohort
            cohort_metrics = self.cohort_data.groupby('cohort_month').agg({
                'customer_unique_id': pd.Series.nunique,
                'order_id': pd.Series.nunique,
                'price': ['sum', 'mean'],
                'cohort_index': 'max'
            }).reset_index()
            
            # Flatten column names
            cohort_metrics.columns = ['cohort_month', 'cohort_size', 'total_orders', 
                                     'total_revenue', 'avg_order_value', 'cohort_age']
            
            # Convert Period to string for JSON serialization
            cohort_metrics['cohort_month'] = cohort_metrics['cohort_month'].astype(str)
            
            # Calculate derived metrics
            cohort_metrics['orders_per_customer'] = cohort_metrics['total_orders'] / cohort_metrics['cohort_size']
            cohort_metrics['revenue_per_customer'] = cohort_metrics['total_revenue'] / cohort_metrics['cohort_size']
            
            # Calculate 3-month retention for each cohort
            retention_3m = self.cohort_data[self.cohort_data['cohort_index'] == 3].groupby('cohort_month').agg({
                'customer_unique_id': pd.Series.nunique
            }).reset_index()
            
            retention_3m.columns = ['cohort_month', 'retained_3m']
            # Convert Period to string for merge
            retention_3m['cohort_month'] = retention_3m['cohort_month'].astype(str)
            cohort_metrics = cohort_metrics.merge(retention_3m, on='cohort_month', how='left')
            cohort_metrics['retention_3m'] = cohort_metrics['retained_3m'] / cohort_metrics['cohort_size'] * 100
            
            # Quality score (composite metric)
            cohort_metrics['quality_score'] = (
                cohort_metrics['revenue_per_customer'] / cohort_metrics['revenue_per_customer'].mean() * 0.4 +
                cohort_metrics['orders_per_customer'] / cohort_metrics['orders_per_customer'].mean() * 0.3 +
                cohort_metrics['retention_3m'].fillna(0) / cohort_metrics['retention_3m'].mean() * 0.3
            )
            
            # Classify cohorts
            cohort_metrics['cohort_quality'] = pd.cut(
                cohort_metrics['quality_score'],
                bins=[0, 0.8, 1.2, float('inf')],
                labels=['Below Average', 'Average', 'Above Average']
            )
            
            # Save analysis
            cohort_metrics.to_csv(f"{self.output_path}/cohort_quality_analysis.csv", index=False)
            
            logger.info(f"Analyzed {len(cohort_metrics)} cohorts")
            return cohort_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing cohort quality: {e}")
            raise CustomException("Failed to analyze cohort quality", e)
    
    def calculate_ltv_by_cohort(self, projection_months=24):
        """Calculate lifetime value projections by cohort"""
        try:
            logger.info("Calculating LTV by cohort...")
            
            # Get revenue retention matrix
            revenue_retention, revenue_absolute = self.calculate_retention_matrix(metric='revenue')
            
            # Calculate average revenue per retained customer
            customer_retention, _ = self.calculate_retention_matrix(metric='customers')
            
            # Avoid division by zero
            avg_revenue_per_customer = revenue_retention / customer_retention.replace(0, np.nan)
            
            # Calculate cumulative LTV
            ltv_matrix = pd.DataFrame(index=revenue_retention.index)
            
            for month in range(min(projection_months, revenue_retention.shape[1])):
                if month == 0:
                    ltv_matrix[f'ltv_month_{month}'] = avg_revenue_per_customer.iloc[:, month]
                else:
                    ltv_matrix[f'ltv_month_{month}'] = (
                        ltv_matrix[f'ltv_month_{month-1}'] + 
                        avg_revenue_per_customer.iloc[:, month].fillna(0)
                    )
            
            # Project remaining months using decay model
            if revenue_retention.shape[1] < projection_months:
                # Fit exponential decay to retention rates
                last_known_retention = customer_retention.iloc[:, -1] / 100
                decay_rate = -np.log(last_known_retention.mean()) / revenue_retention.shape[1]
                
                for month in range(revenue_retention.shape[1], projection_months):
                    projected_retention = np.exp(-decay_rate * month) * 100
                    projected_revenue = avg_revenue_per_customer.mean(axis=1) * projected_retention / 100
                    
                    ltv_matrix[f'ltv_month_{month}'] = (
                        ltv_matrix[f'ltv_month_{month-1}'] + projected_revenue
                    )
            
            # Final LTV and payback period
            ltv_summary = pd.DataFrame({
                'cohort_month': ltv_matrix.index.astype(str),  # Convert Period to string
                'ltv_6m': ltv_matrix[f'ltv_month_5'] if 'ltv_month_5' in ltv_matrix else 0,
                'ltv_12m': ltv_matrix[f'ltv_month_11'] if 'ltv_month_11' in ltv_matrix else 0,
                'ltv_24m': ltv_matrix[f'ltv_month_{projection_months-1}'] if f'ltv_month_{projection_months-1}' in ltv_matrix else 0
            })
            
            # Calculate CAC payback period (assuming $50 CAC)
            cac = 50
            ltv_summary['months_to_payback'] = ltv_matrix.apply(
                lambda row: next((i for i, v in enumerate(row) if v > cac), np.nan), 
                axis=1
            )
            
            ltv_summary.to_csv(f"{self.output_path}/ltv_by_cohort.csv", index=False)
            
            logger.info("LTV calculation completed")
            return ltv_summary, ltv_matrix
            
        except Exception as e:
            logger.error(f"Error calculating LTV: {e}")
            raise CustomException("Failed to calculate LTV", e)
    
    def segment_cohort_analysis(self, segment_column='customer_state'):
        """Perform cohort analysis by customer segments"""
        try:
            # Check if segment column exists, otherwise use alternative
            if segment_column not in self.cohort_data.columns:
                logger.warning(f"Column '{segment_column}' not found in cohort data")
                # Try alternative columns
                alternatives = ['customer_segment', 'churned', 'clv_tier']
                for alt in alternatives:
                    if alt in self.cohort_data.columns:
                        segment_column = alt
                        logger.info(f"Using alternative column: {alt}")
                        break
                else:
                    # If no segment column available, create a default one
                    logger.warning("No segment column found, creating default segments based on revenue")
                    # Use price column which exists in the order_items data
                    if 'price' in self.cohort_data.columns:
                        self.cohort_data['revenue_segment'] = pd.qcut(
                            self.cohort_data['price'], 
                            q=3, 
                            labels=['Low', 'Medium', 'High']
                        )
                    else:
                        # Create a simple binary segment based on cohort
                        self.cohort_data['revenue_segment'] = (
                            self.cohort_data['cohort_index'] > 3
                        ).map({True: 'Retained', False: 'New'})
                    segment_column = 'revenue_segment'
            
            logger.info(f"Performing segmented cohort analysis by {segment_column}...")
            
            segment_retention = {}
            
            # Get unique segments
            segments = self.cohort_data[segment_column].unique()[:5]  # Limit to top 5 for visualization
            
            for segment in segments:
                # Filter data for segment
                segment_data = self.cohort_data[self.cohort_data[segment_column] == segment]
                
                # Calculate retention for this segment
                segment_cohort = segment_data.groupby(['cohort_month', 'cohort_index']).agg({
                    'customer_unique_id': pd.Series.nunique
                }).reset_index()
                
                segment_matrix = segment_cohort.pivot_table(
                    index='cohort_month',
                    columns='cohort_index',
                    values='customer_unique_id'
                )
                
                # Calculate retention rates
                cohort_size = segment_matrix.iloc[:, 0]
                retention = segment_matrix.divide(cohort_size, axis=0) * 100
                
                segment_retention[segment] = retention
            
            # Compare average retention across segments
            comparison_data = []
            for period in [0, 1, 3, 6, 12]:
                for segment, retention in segment_retention.items():
                    if period < retention.shape[1]:
                        avg_retention = retention.iloc[:, period].mean()
                        comparison_data.append({
                            'segment': segment,
                            'period': period,
                            'avg_retention': avg_retention
                        })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_pivot = comparison_df.pivot(index='segment', columns='period', values='avg_retention')
            
            comparison_pivot.to_csv(f"{self.output_path}/retention_by_{segment_column}.csv")
            
            logger.info(f"Completed segmented analysis for {len(segments)} segments")
            return segment_retention, comparison_pivot
            
        except Exception as e:
            logger.error(f"Error in segment cohort analysis: {e}")
            raise CustomException("Failed to perform segment cohort analysis", e)
    
    def create_cohort_visualizations(self, retention_matrix, cohort_metrics=None):
        """Create comprehensive cohort visualizations"""
        try:
            logger.info("Creating cohort visualizations...")
            
            # Convert Period index to strings to avoid JSON serialization issues
            retention_matrix_viz = retention_matrix.copy()
            retention_matrix_viz.index = retention_matrix_viz.index.astype(str)
            
            # 1. Classic retention heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=retention_matrix_viz.values,
                x=[f"Month {i}" for i in retention_matrix_viz.columns],
                y=retention_matrix_viz.index,
                colorscale='RdYlGn',
                text=np.round(retention_matrix_viz.values, 1),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title='Customer Retention by Cohort',
                xaxis_title='Months Since First Purchase',
                yaxis_title='Cohort',
                height=600
            )
            
            fig_heatmap.write_html(f"{self.output_path}/retention_heatmap.html")
            
            # 2. Retention curves by cohort
            fig_curves = go.Figure()
            
            # Add traces for each cohort
            for cohort in retention_matrix_viz.index[:10]:  # Limit to recent 10 cohorts
                fig_curves.add_trace(go.Scatter(
                    x=list(range(len(retention_matrix_viz.columns))),
                    y=retention_matrix_viz.loc[cohort].values,
                    mode='lines+markers',
                    name=str(cohort),
                    hovertemplate='%{y:.1f}%<extra></extra>'
                ))
            
            # Add average retention curve
            avg_retention = retention_matrix_viz.mean()
            fig_curves.add_trace(go.Scatter(
                x=list(range(len(avg_retention))),
                y=avg_retention.values,
                mode='lines',
                name='Average',
                line=dict(color='black', width=3, dash='dash')
            ))
            
            fig_curves.update_layout(
                title='Retention Curves by Cohort',
                xaxis_title='Months Since First Purchase',
                yaxis_title='Retention Rate (%)',
                hovermode='x unified',
                height=500
            )
            
            fig_curves.write_html(f"{self.output_path}/retention_curves.html")
            
            # 3. Cohort quality bubble chart (if metrics provided)
            if cohort_metrics is not None:
                # Convert Period objects to strings for JSON serialization
                cohort_metrics_viz = cohort_metrics.copy()
                cohort_metrics_viz['cohort_month'] = cohort_metrics_viz['cohort_month'].astype(str)
                
                fig_bubble = px.scatter(
                    cohort_metrics_viz,
                    x='revenue_per_customer',
                    y='retention_3m',
                    size='cohort_size',
                    color='quality_score',
                    hover_data=['cohort_month', 'orders_per_customer'],
                    title='Cohort Quality Analysis',
                    labels={
                        'revenue_per_customer': 'Revenue per Customer ($)',
                        'retention_3m': '3-Month Retention (%)',
                        'cohort_size': 'Cohort Size'
                    }
                )
                
                fig_bubble.update_traces(marker=dict(line=dict(width=1, color='white')))
                fig_bubble.write_html(f"{self.output_path}/cohort_quality_bubble.html")
            
            # 4. Waterfall chart showing retention decay
            periods = list(range(min(13, len(retention_matrix_viz.columns))))
            avg_retention = retention_matrix_viz.iloc[:, :13].mean()
            
            retention_changes = [100]  # Start at 100%
            for i in range(1, len(avg_retention)):
                retention_changes.append(avg_retention.iloc[i] - avg_retention.iloc[i-1])
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Retention",
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(retention_changes) - 1),
                x=[f"Month {i}" for i in periods],
                y=retention_changes,
                text=[f"{avg_retention.iloc[i]:.1f}%" for i in range(len(avg_retention))],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}}
            ))
            
            fig_waterfall.update_layout(
                title="Average Retention Decay Over Time",
                yaxis_title="Retention Rate (%)",
                showlegend=False,
                height=400
            )
            
            fig_waterfall.write_html(f"{self.output_path}/retention_waterfall.html")
            
            logger.info("Cohort visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise CustomException("Failed to create cohort visualizations", e)
    
    def predict_future_cohorts(self, historical_months=12):
        """Predict performance of future cohorts based on historical patterns"""
        try:
            logger.info("Predicting future cohort performance...")
            
            # Get retention patterns from recent cohorts
            retention_matrix, _ = self.calculate_retention_matrix()
            recent_cohorts = retention_matrix.tail(historical_months)
            
            # Calculate average retention curve
            avg_retention_curve = recent_cohorts.mean()
            std_retention_curve = recent_cohorts.std()
            
            # Trend analysis
            retention_trends = []
            for period in range(min(6, len(avg_retention_curve))):
                if period < len(retention_matrix.columns):
                    period_values = retention_matrix.iloc[:, period].dropna()
                    if len(period_values) > 3:
                        # Simple linear regression for trend
                        x = np.arange(len(period_values))
                        y = period_values.values
                        slope, intercept = np.polyfit(x, y, 1)
                        retention_trends.append({
                            'period': period,
                            'current_avg': avg_retention_curve.iloc[period],
                            'trend_slope': slope,
                            'improving': slope > 0
                        })
            
            # Predict next 3 cohorts
            predictions = []
            for future_month in range(1, 4):
                predicted_retention = []
                for period in range(min(12, len(avg_retention_curve))):
                    # Base prediction on average with trend adjustment
                    base_retention = avg_retention_curve.iloc[period]
                    if period < len(retention_trends):
                        trend_adjustment = retention_trends[period]['trend_slope'] * future_month
                        predicted = base_retention + trend_adjustment
                        # Add some uncertainty
                        predicted = np.clip(predicted, 0, 100)
                    else:
                        predicted = base_retention
                    
                    predicted_retention.append(predicted)
                
                predictions.append({
                    'cohort': f"Future +{future_month}M",
                    'predicted_retention': predicted_retention,
                    'confidence_band': std_retention_curve.values[:len(predicted_retention)]
                })
            
            # Create prediction visualization
            fig = go.Figure()
            
            # Historical average
            fig.add_trace(go.Scatter(
                x=list(range(len(avg_retention_curve))),
                y=avg_retention_curve.values,
                mode='lines+markers',
                name='Historical Average',
                line=dict(color='blue', width=2)
            ))
            
            # Predictions
            colors = ['green', 'orange', 'red']
            for i, pred in enumerate(predictions):
                fig.add_trace(go.Scatter(
                    x=list(range(len(pred['predicted_retention']))),
                    y=pred['predicted_retention'],
                    mode='lines',
                    name=pred['cohort'],
                    line=dict(color=colors[i], dash='dash')
                ))
            
            fig.update_layout(
                title='Predicted Retention for Future Cohorts',
                xaxis_title='Months Since First Purchase',
                yaxis_title='Retention Rate (%)',
                hovermode='x unified'
            )
            
            fig.write_html(f"{self.output_path}/cohort_predictions.html")
            
            # Save predictions
            pred_df = pd.DataFrame(predictions)
            pred_df.to_csv(f"{self.output_path}/cohort_predictions.csv", index=False)
            
            logger.info("Cohort predictions completed")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting future cohorts: {e}")
            raise CustomException("Failed to predict future cohorts", e)
    
    def run_complete_analysis(self):
        """Run complete cohort analysis pipeline"""
        try:
            import os
            os.makedirs(self.output_path, exist_ok=True)
            
            logger.info("Starting complete cohort analysis...")
            
            # Prepare data
            self.prepare_cohort_data()
            
            # Calculate retention matrices
            retention_matrix, absolute_matrix = self.calculate_retention_matrix(metric='customers')
            revenue_retention, _ = self.calculate_retention_matrix(metric='revenue')
            
            # Analyze cohort quality
            cohort_metrics = self.analyze_cohort_quality()
            
            # Calculate LTV
            ltv_summary, ltv_matrix = self.calculate_ltv_by_cohort()
            
            # Segment analysis
            segment_retention, segment_comparison = self.segment_cohort_analysis()
            
            # Create visualizations
            self.create_cohort_visualizations(retention_matrix, cohort_metrics)
            
            # Predict future cohorts
            predictions = self.predict_future_cohorts()
            
            # Generate summary report
            summary = {
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'total_cohorts': len(retention_matrix),
                'avg_retention_month_1': retention_matrix.iloc[:, 1].mean() if retention_matrix.shape[1] > 1 else 0,
                'avg_retention_month_3': retention_matrix.iloc[:, 3].mean() if retention_matrix.shape[1] > 3 else 0,
                'avg_retention_month_6': retention_matrix.iloc[:, 6].mean() if retention_matrix.shape[1] > 6 else 0,
                'best_cohort': str(cohort_metrics.loc[cohort_metrics['quality_score'].idxmax(), 'cohort_month']),
                'avg_ltv_12m': ltv_summary['ltv_12m'].mean()
            }
            
            with open(f"{self.output_path}/analysis_summary.json", 'w') as f:
                import json
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Complete cohort analysis finished successfully")
            
            return {
                'retention_matrix': retention_matrix,
                'cohort_metrics': cohort_metrics,
                'ltv_summary': ltv_summary,
                'predictions': predictions,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise CustomException("Failed to complete cohort analysis", e)

if __name__ == "__main__":
    analyzer = AdvancedCohortAnalysis()
    results = analyzer.run_complete_analysis()