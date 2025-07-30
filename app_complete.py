import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.logger import get_logger
from src.ab_testing import ABTestCalculator
from src.cohort_analysis import AdvancedCohortAnalysis

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Customer Analytics Platform - Complete",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/customer-analytics',
        'Report a bug': "https://github.com/yourusername/customer-analytics/issues",
        'About': "Advanced Customer Analytics Platform v2.0"
    }
)

# Enhanced CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #2ca02c;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe8a1 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .tab-content {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .hover-effect:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_all_data():
    """Load all necessary data with caching"""
    try:
        data = {}
        
        # Customer features
        if os.path.exists("artifacts/processed/customer_features_advanced.csv"):
            data['customer_features'] = pd.read_csv("artifacts/processed/customer_features_advanced.csv")
        elif os.path.exists("artifacts/processed/customer_features.csv"):
            data['customer_features'] = pd.read_csv("artifacts/processed/customer_features.csv")
        else:
            return None
        
        # Business metrics
        if os.path.exists("artifacts/business_metrics/dashboard_data.json"):
            with open("artifacts/business_metrics/dashboard_data.json", 'r') as f:
                data['business_metrics'] = json.load(f)
        
        # Cohort analysis
        if os.path.exists("artifacts/cohort_analysis/analysis_summary.json"):
            with open("artifacts/cohort_analysis/analysis_summary.json", 'r') as f:
                data['cohort_summary'] = json.load(f)
        
        # Model scores
        if os.path.exists("artifacts/models/model_scores_advanced.csv"):
            data['model_scores'] = pd.read_csv("artifacts/models/model_scores_advanced.csv", index_col=0)
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def show_enhanced_executive_dashboard(data):
    """Enhanced executive dashboard with business metrics"""
    st.header("📊 Executive Dashboard - Enhanced Analytics")
    
    if 'business_metrics' not in data:
        st.warning("Business metrics not available. Run the business metrics calculator first.")
        return
    
    metrics = data['business_metrics']['kpis']
    
    # Top-level KPIs in a grid
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Revenue",
            f"${metrics['total_revenue']:,.0f}",
            delta=f"{metrics['total_customers']:,} customers"
        )
    
    with col2:
        st.metric(
            "Total CLV",
            f"${metrics['total_clv']:,.0f}",
            delta=f"${metrics['avg_clv']:.0f} avg"
        )
    
    with col3:
        st.metric(
            "Gross Profit",
            f"${metrics['total_profit']:,.0f}",
            delta=f"{metrics['avg_profit_margin']:.1f}% margin"
        )
    
    with col4:
        st.metric(
            "Retention Rate",
            f"{metrics['retention_rate']:.1f}%",
            delta=f"{100-metrics['churn_rate']:.1f}% active"
        )
    
    with col5:
        st.metric(
            "Churn Rate",
            f"{metrics['churn_rate']:.1f}%",
            delta=f"{metrics['active_customers']:,} at risk",
            delta_color="inverse"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Revenue and Growth Trends
    st.subheader("📈 Revenue & Growth Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly growth chart
        if 'growth_trend' in data['business_metrics']:
            growth_df = pd.DataFrame(data['business_metrics']['growth_trend'])
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly Revenue', 'Growth Rate %'),
                row_heights=[0.7, 0.3],
                shared_xaxes=True
            )
            
            # Revenue bars
            fig.add_trace(
                go.Bar(x=growth_df['month'], y=growth_df['revenue'], name='Revenue'),
                row=1, col=1
            )
            
            # Growth rate line
            fig.add_trace(
                go.Scatter(
                    x=growth_df['month'], 
                    y=growth_df['revenue_growth_rate'],
                    mode='lines+markers',
                    name='Growth Rate',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Customer distribution
        if 'customer_distribution' in data['business_metrics']:
            dist = data['business_metrics']['customer_distribution']
            
            # CLV tier distribution
            if 'by_clv_tier' in dist:
                clv_data = pd.DataFrame(
                    list(dist['by_clv_tier'].items()),
                    columns=['CLV Tier', 'Count']
                )
                
                fig = px.pie(
                    clv_data, 
                    values='Count', 
                    names='CLV Tier',
                    title='Customer Distribution by CLV Tier',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    # Business Insights
    if os.path.exists("artifacts/business_metrics/business_insights.csv"):
        st.subheader("💡 Key Business Insights")
        
        insights_df = pd.read_csv("artifacts/business_metrics/business_insights.csv")
        
        for _, insight in insights_df.iterrows():
            if insight['category'] == 'Customer Value':
                box_class = 'success-box'
                icon = '💰'
            elif insight['category'] == 'Retention':
                box_class = 'warning-box'
                icon = '⚠️'
            else:
                box_class = 'insight-box'
                icon = '📊'
            
            st.markdown(f"""
            <div class="{box_class} hover-effect">
                <h5>{icon} {insight['category']}: {insight['insight']}</h5>
                <p><b>Recommended Action:</b> {insight['action']}</p>
                <p><b>Expected Impact:</b> {insight['impact']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Profitability Analysis
    st.subheader("💼 Profitability Analysis")
    
    if os.path.exists("artifacts/business_metrics/profitability_by_segment.csv"):
        prof_df = pd.read_csv("artifacts/business_metrics/profitability_by_segment.csv", index_col=0)
        
        fig = go.Figure()
        
        # Revenue bars
        fig.add_trace(go.Bar(
            x=prof_df.index,
            y=prof_df['gross_revenue'],
            name='Revenue',
            marker_color='lightblue'
        ))
        
        # Profit bars
        fig.add_trace(go.Bar(
            x=prof_df.index,
            y=prof_df['gross_profit'],
            name='Profit',
            marker_color='green'
        ))
        
        # Profit margin line
        fig.add_trace(go.Scatter(
            x=prof_df.index,
            y=prof_df['profit_margin'] * 100,
            name='Profit Margin %',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Profitability by Customer Segment',
            yaxis=dict(title='Amount ($)'),
            yaxis2=dict(title='Profit Margin (%)', overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_enhanced_ab_testing():
    """Enhanced A/B testing with comprehensive calculator"""
    st.header("🧪 A/B Testing Suite")
    
    calculator = ABTestCalculator()
    
    # Create tabs for different A/B testing functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "📏 Sample Size Calculator", 
        "📊 Results Analyzer", 
        "🔬 Test Simulator",
        "📈 Sequential Testing"
    ])
    
    with tab1:
        st.subheader("Sample Size Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_type = st.selectbox(
                "Test Type",
                ["Conversion Rate", "Revenue (ARPU)", "Retention"]
            )
            
            baseline = st.number_input(
                "Baseline Rate (%)" if test_type == "Conversion Rate" else "Baseline Value",
                min_value=0.1,
                max_value=100.0 if test_type == "Conversion Rate" else 10000.0,
                value=5.0 if test_type == "Conversion Rate" else 50.0,
                step=0.1
            )
        
        with col2:
            mde = st.number_input(
                "Minimum Detectable Effect (%)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0
            )
            
            confidence = st.slider(
                "Confidence Level (%)",
                min_value=80,
                max_value=99,
                value=95
            )
        
        with col3:
            power = st.slider(
                "Statistical Power (%)",
                min_value=70,
                max_value=95,
                value=80
            )
            
            daily_traffic = st.number_input(
                "Daily Traffic",
                min_value=100,
                max_value=1000000,
                value=5000,
                step=100
            )
        
        if st.button("Calculate Sample Size", type="primary"):
            # Calculate sample size
            if test_type == "Conversion Rate":
                results = calculator.calculate_sample_size_simple(
                    baseline_rate=baseline/100,
                    mde_percent=mde,
                    confidence=confidence,
                    power=power
                )
            else:
                # For other metrics, use appropriate calculation
                results = calculator.calculate_sample_size_simple(
                    baseline_rate=0.5,  # Placeholder
                    mde_percent=mde,
                    confidence=confidence,
                    power=power
                )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Sample Size per Variant",
                    f"{results['sample_size_per_variant']:,}"
                )
            
            with col2:
                st.metric(
                    "Total Sample Size",
                    f"{results['total_sample_size']:,}"
                )
            
            with col3:
                duration = calculator.get_duration_estimate(
                    results['sample_size_per_variant'],
                    daily_traffic
                )
                st.metric(
                    "Estimated Duration",
                    f"{duration['days_with_buffer']} days"
                )
            
            # Visualize sample size sensitivity
            st.subheader("Sample Size Sensitivity Analysis")
            
            mde_range = np.linspace(5, 50, 20)
            sample_sizes = []
            
            for m in mde_range:
                res = calculator.calculate_sample_size_simple(
                    baseline_rate=baseline/100,
                    mde_percent=m,
                    confidence=confidence,
                    power=power
                )
                sample_sizes.append(res['sample_size_per_variant'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mde_range,
                y=sample_sizes,
                mode='lines+markers',
                name='Sample Size',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_vline(
                x=mde, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Your MDE: {mde}%"
            )
            
            fig.update_layout(
                title="Sample Size vs Minimum Detectable Effect",
                xaxis_title="MDE (%)",
                yaxis_title="Sample Size per Variant",
                yaxis_type="log",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("A/B Test Results Analyzer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Control Group**")
            control_visitors = st.number_input(
                "Visitors (Control)", 
                min_value=100, 
                value=5000,
                key="ctrl_visitors"
            )
            control_conversions = st.number_input(
                "Conversions (Control)", 
                min_value=0, 
                max_value=control_visitors,
                value=250,
                key="ctrl_conv"
            )
        
        with col2:
            st.markdown("**Treatment Group**")
            treatment_visitors = st.number_input(
                "Visitors (Treatment)", 
                min_value=100, 
                value=5000,
                key="treat_visitors"
            )
            treatment_conversions = st.number_input(
                "Conversions (Treatment)", 
                min_value=0,
                max_value=treatment_visitors,
                value=300,
                key="treat_conv"
            )
        
        if st.button("Analyze Results", type="primary"):
            results = calculator.analyze_results_simple(
                control_visitors, control_conversions,
                treatment_visitors, treatment_conversions
            )
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Relative Lift",
                    f"{results['relative_lift']*100:.1f}%",
                    delta="Positive" if results['relative_lift'] > 0 else "Negative"
                )
            
            with col2:
                st.metric(
                    "P-value",
                    f"{results['p_value']:.4f}",
                    delta="Significant" if results['p_value'] < 0.05 else "Not Significant"
                )
            
            with col3:
                st.metric(
                    "Confidence",
                    f"{results['prob_treatment_better']*100:.1f}%",
                    delta="Treatment Better" if results['prob_treatment_better'] > 0.5 else "Control Better"
                )
            
            # Visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Conversion Rates', 'Bayesian Probability Distribution')
            )
            
            # Bar chart with confidence intervals
            fig.add_trace(
                go.Bar(
                    x=['Control', 'Treatment'],
                    y=[results['control_rate']*100, results['treatment_rate']*100],
                    error_y=dict(
                        type='data',
                        array=[
                            (results['control_ci'][1] - results['control_rate'])*100,
                            (results['treatment_ci'][1] - results['treatment_rate'])*100
                        ]
                    ),
                    marker_color=['lightblue', 'lightgreen']
                ),
                row=1, col=1
            )
            
            # Bayesian posterior distribution (simulated)
            x_range = np.linspace(0, max(results['control_rate'], results['treatment_rate'])*2, 100)
            
            # Beta distributions for visualization
            from scipy import stats
            control_alpha = control_conversions + 1
            control_beta = control_visitors - control_conversions + 1
            treatment_alpha = treatment_conversions + 1
            treatment_beta = treatment_visitors - treatment_conversions + 1
            
            control_dist = stats.beta.pdf(x_range, control_alpha, control_beta)
            treatment_dist = stats.beta.pdf(x_range, treatment_alpha, treatment_beta)
            
            fig.add_trace(
                go.Scatter(x=x_range*100, y=control_dist, name='Control', fill='tozeroy'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=x_range*100, y=treatment_dist, name='Treatment', fill='tozeroy'),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Conversion Rate (%)", row=1, col=1)
            fig.update_xaxes(title_text="Conversion Rate (%)", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Probability Density", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Decision recommendation
            if results['is_significant']:
                if results['relative_lift'] > 0:
                    st.success(f"""
                    ✅ **Test Result: Significant Improvement**
                    
                    The treatment shows a statistically significant improvement of {results['relative_lift']*100:.1f}%.
                    
                    **Recommendation:** Roll out the treatment to 100% of traffic.
                    """)
                else:
                    st.error(f"""
                    ❌ **Test Result: Significant Degradation**
                    
                    The treatment shows a statistically significant decrease of {abs(results['relative_lift'])*100:.1f}%.
                    
                    **Recommendation:** Do not implement the treatment. Investigate what went wrong.
                    """)
            else:
                st.warning(f"""
                ⚠️ **Test Result: Not Statistically Significant**
                
                The observed difference of {results['relative_lift']*100:.1f}% is not statistically significant (p={results['p_value']:.3f}).
                
                **Recommendation:** Continue testing to gather more data or consider this difference negligible.
                """)

def show_enhanced_cohort_analysis(data):
    """Enhanced cohort analysis with advanced visualizations"""
    st.header("📈 Advanced Cohort Analysis")
    
    # Check if cohort analysis exists
    if not os.path.exists("artifacts/cohort_analysis/customer_cohort_retention.csv"):
        st.info("Running cohort analysis...")
        analyzer = AdvancedCohortAnalysis()
        cohort_results = analyzer.run_complete_analysis()
    
    # Load cohort data
    if os.path.exists("artifacts/cohort_analysis/customer_cohort_retention.csv"):
        retention_matrix = pd.read_csv("artifacts/cohort_analysis/customer_cohort_retention.csv", index_col=0)
        
        # Display controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_type = st.selectbox(
                "Metric",
                ["Customer Retention", "Revenue Retention", "Order Frequency"]
            )
        
        with col2:
            display_type = st.selectbox(
                "Display Type",
                ["Heatmap", "Curves", "Waterfall", "3D Surface"]
            )
        
        with col3:
            cohort_limit = st.slider(
                "Number of Cohorts",
                min_value=5,
                max_value=min(20, len(retention_matrix)),
                value=12
            )
        
        # Load appropriate retention matrix
        if metric_type == "Revenue Retention" and os.path.exists("artifacts/cohort_analysis/revenue_cohort_retention.csv"):
            retention_matrix = pd.read_csv("artifacts/cohort_analysis/revenue_cohort_retention.csv", index_col=0)
        elif metric_type == "Order Frequency" and os.path.exists("artifacts/cohort_analysis/orders_cohort_retention.csv"):
            retention_matrix = pd.read_csv("artifacts/cohort_analysis/orders_cohort_retention.csv", index_col=0)
        
        # Limit cohorts
        retention_display = retention_matrix.tail(cohort_limit)
        
        if display_type == "Heatmap":
            # Enhanced heatmap with annotations
            fig = go.Figure(data=go.Heatmap(
                z=retention_display.values,
                x=[f"Month {i}" for i in retention_display.columns],
                y=[str(i) for i in retention_display.index],
                colorscale='RdYlGn',
                text=np.round(retention_display.values, 1),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(title="Retention %")
            ))
            
            fig.update_layout(
                title=f'{metric_type} by Cohort',
                xaxis_title='Months Since First Purchase',
                yaxis_title='Cohort',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif display_type == "Curves":
            # Retention curves with confidence bands
            fig = go.Figure()
            
            # Add individual cohort lines
            for cohort in retention_display.index:
                fig.add_trace(go.Scatter(
                    x=list(range(len(retention_display.columns))),
                    y=retention_display.loc[cohort].values,
                    mode='lines',
                    name=str(cohort),
                    line=dict(width=1),
                    opacity=0.6
                ))
            
            # Add average line
            avg_retention = retention_display.mean()
            std_retention = retention_display.std()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(avg_retention))),
                y=avg_retention.values,
                mode='lines',
                name='Average',
                line=dict(color='black', width=3)
            ))
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=list(range(len(avg_retention))) + list(range(len(avg_retention)))[::-1],
                y=list(avg_retention + std_retention) + list(avg_retention - std_retention)[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
            
            fig.update_layout(
                title=f'{metric_type} Curves Over Time',
                xaxis_title='Months Since First Purchase',
                yaxis_title=f'{metric_type} (%)',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif display_type == "3D Surface":
            # 3D surface plot
            fig = go.Figure(data=[go.Surface(
                z=retention_display.values,
                x=list(range(len(retention_display.columns))),
                y=list(range(len(retention_display.index))),
                colorscale='RdYlGn'
            )])
            
            fig.update_layout(
                title=f'3D {metric_type} Surface',
                scene=dict(
                    xaxis_title='Months Since First Purchase',
                    yaxis_title='Cohort Index',
                    zaxis_title=f'{metric_type} (%)'
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cohort Quality Analysis
        if os.path.exists("artifacts/cohort_analysis/cohort_quality_analysis.csv"):
            st.subheader("📊 Cohort Quality Analysis")
            
            quality_df = pd.read_csv("artifacts/cohort_analysis/cohort_quality_analysis.csv")
            quality_df['cohort_month'] = pd.to_datetime(quality_df['cohort_month'])
            
            # Quality metrics over time
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cohort Size', 'Revenue per Customer', 
                              '3-Month Retention', 'Quality Score')
            )
            
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['cohort_size'],
                          mode='lines+markers', name='Cohort Size'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['revenue_per_customer'],
                          mode='lines+markers', name='Revenue/Customer'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['retention_3m'],
                          mode='lines+markers', name='3M Retention'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['quality_score'],
                          mode='lines+markers', name='Quality Score'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # LTV Analysis
        if os.path.exists("artifacts/cohort_analysis/ltv_by_cohort.csv"):
            st.subheader("💰 Lifetime Value Analysis")
            
            ltv_df = pd.read_csv("artifacts/cohort_analysis/ltv_by_cohort.csv")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # LTV progression
                fig = go.Figure()
                
                fig.add_trace(go.Bar(x=ltv_df['cohort_month'], y=ltv_df['ltv_6m'],
                                   name='6-Month LTV', marker_color='lightblue'))
                fig.add_trace(go.Bar(x=ltv_df['cohort_month'], y=ltv_df['ltv_12m'],
                                   name='12-Month LTV', marker_color='blue'))
                fig.add_trace(go.Bar(x=ltv_df['cohort_month'], y=ltv_df['ltv_24m'],
                                   name='24-Month LTV', marker_color='darkblue'))
                
                fig.update_layout(
                    title='LTV Progression by Cohort',
                    xaxis_title='Cohort',
                    yaxis_title='Lifetime Value ($)',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Payback period
                fig = px.line(
                    ltv_df,
                    x='cohort_month',
                    y='months_to_payback',
                    title='CAC Payback Period by Cohort',
                    labels={'months_to_payback': 'Months to Payback'}
                )
                fig.add_hline(y=12, line_dash="dash", line_color="red",
                            annotation_text="12-month target")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def main():
    # Load data
    data = load_all_data()
    
    if data is None:
        st.error("""
        No data found! Please run the pipeline first:
        ```bash
        python pipeline/advanced_pipeline.py
        ```
        """)
        return
    
    # Sidebar navigation with icons
    st.sidebar.title("🚀 Navigation")
    
    pages = {
        "Executive Dashboard": "📊",
        "Advanced Segmentation": "🎯",
        "Churn & CLV Analysis": "💰",
        "A/B Testing Suite": "🧪",
        "Cohort Analysis": "📈",
        "Recommendation Engine": "🤖",
        "Business Insights": "💡"
    }
    
    selected_page = st.sidebar.radio(
        "Select Page",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    
    # Page content
    if selected_page == "Executive Dashboard":
        show_enhanced_executive_dashboard(data)
    
    elif selected_page == "A/B Testing Suite":
        show_enhanced_ab_testing()
    
    elif selected_page == "Cohort Analysis":
        show_enhanced_cohort_analysis(data)
    
    # Add other pages as needed...
    else:
        st.info(f"Page '{selected_page}' is under construction. Showing Executive Dashboard instead.")
        show_enhanced_executive_dashboard(data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center'>
        <p>Customer Analytics Platform v2.0</p>
        <p style='font-size: 0.8em'>Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()