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

# Enhanced CSS with better visibility
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    /* Enhanced KPI Card Styling with Better Visibility */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    /* Ensure metric labels are visible */
    .stMetric > div > div:first-child {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Ensure metric values are visible */
    .stMetric > div > div:nth-child(2) {
        color: #1a1a1a !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    
    /* Ensure delta values are visible */
    .stMetric > div > div:nth-child(3) {
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    /* Custom metric container for better layout */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    /* Enhanced insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid #2196f3;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #1a237e;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #856404;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #155724;
    }
    
    /* Dark theme for headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: transparent;
        border-radius: 8px;
        color: #2c3e50;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Input fields styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        color: #2c3e50;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-radius: 8px;
    }
    
    /* Ensure all text is visible */
    p, span, div {
        color: #2c3e50;
    }
    
    /* Custom KPI card component */
    .custom-kpi-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .custom-kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    
    .kpi-title {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .kpi-delta.positive {
        color: #28a745;
    }
    
    .kpi-delta.negative {
        color: #dc3545;
    }
    
    .kpi-icon {
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 2rem;
        opacity: 0.2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_all_data():
    """Load all necessary data with caching"""
    try:
        data = {}
        
        # Customer features - check multiple possible locations
        if os.path.exists("artifacts/processed_final/customer_features_final.csv"):
            data['customer_features'] = pd.read_csv("artifacts/processed_final/customer_features_final.csv")
        elif os.path.exists("artifacts/processed_final/customer_features_enhanced.csv"):
            data['customer_features'] = pd.read_csv("artifacts/processed_final/customer_features_enhanced.csv")
        elif os.path.exists("artifacts/processed/customer_features_advanced.csv"):
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

def create_custom_kpi_card(title, value, delta=None, delta_color="positive", icon="📊"):
    """Create a custom KPI card with better visibility"""
    delta_class = "positive" if delta_color == "positive" else "negative"
    delta_symbol = "↑" if delta_color == "positive" else "↓"
    
    card_html = f"""
    <div class="custom-kpi-card" style="position: relative;">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-delta {delta_class}">{delta_symbol} {delta}</div>' if delta else ''}
    </div>
    """
    return card_html

def show_enhanced_executive_dashboard(data):
    """Enhanced executive dashboard with business metrics"""
    st.header("📊 Executive Dashboard - Enhanced Analytics")
    
    if 'business_metrics' not in data:
        st.warning("Business metrics not available. Run the business metrics calculator first.")
        return
    
    metrics = data['business_metrics']['kpis']
    
    # Create custom KPI cards with better visibility
    st.markdown('<div style="background-color: #f8f9fa; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            create_custom_kpi_card(
                "Total Revenue",
                f"${metrics['total_revenue']:,.0f}",
                f"{metrics['total_customers']:,} customers",
                "positive",
                "💰"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_custom_kpi_card(
                "Total CLV",
                f"${metrics['total_clv']:,.0f}",
                f"${metrics['avg_clv']:.0f} avg",
                "positive",
                "📈"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_custom_kpi_card(
                "Gross Profit",
                f"${metrics['total_profit']:,.0f}",
                f"{metrics['avg_profit_margin']:.1f}% margin",
                "positive",
                "💵"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        retention_color = "positive" if metrics['retention_rate'] > 80 else "negative"
        st.markdown(
            create_custom_kpi_card(
                "Retention Rate",
                f"{metrics['retention_rate']:.1f}%",
                f"{100-metrics['churn_rate']:.1f}% active",
                retention_color,
                "🔄"
            ),
            unsafe_allow_html=True
        )
    
    with col5:
        churn_color = "negative" if metrics['churn_rate'] > 20 else "positive"
        st.markdown(
            create_custom_kpi_card(
                "Churn Rate",
                f"{metrics['churn_rate']:.1f}%",
                f"{metrics['active_customers']:,} at risk",
                churn_color,
                "⚠️"
            ),
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Revenue and Growth Trends with enhanced styling
    st.subheader("📈 Revenue & Growth Trends")
    
    # Add a background container
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly growth chart
            if 'growth_trend' in data['business_metrics']:
                growth_df = pd.DataFrame(data['business_metrics']['growth_trend'])
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Monthly Revenue', 'Growth Rate %'),
                    row_heights=[0.7, 0.3],
                    shared_xaxes=True,
                    vertical_spacing=0.15
                )
                
                # Revenue bars with gradient colors
                fig.add_trace(
                    go.Bar(
                        x=growth_df['month'], 
                        y=growth_df['revenue'], 
                        name='Revenue',
                        marker=dict(
                            color=growth_df['revenue'],
                            colorscale='Blues',
                            showscale=False
                        ),
                        text=[f'${v:,.0f}' for v in growth_df['revenue']],
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                # Growth rate line with markers
                fig.add_trace(
                    go.Scatter(
                        x=growth_df['month'], 
                        y=growth_df['revenue_growth_rate'],
                        mode='lines+markers',
                        name='Growth Rate',
                        line=dict(color='#e74c3c', width=3),
                        marker=dict(size=8)
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=450,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2c3e50')
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer distribution
            if 'customer_distribution' in data['business_metrics']:
                dist = data['business_metrics']['customer_distribution']
                
                # CLV tier distribution with custom colors
                if 'by_clv_tier' in dist:
                    clv_data = pd.DataFrame(
                        list(dist['by_clv_tier'].items()),
                        columns=['CLV Tier', 'Count']
                    )
                    
                    # Define custom colors for tiers
                    colors = {
                        'Low CLV': '#e74c3c',
                        'Medium CLV': '#f39c12',
                        'High CLV': '#27ae60',
                        'VIP': '#8e44ad'
                    }
                    
                    fig = px.pie(
                        clv_data, 
                        values='Count', 
                        names='CLV Tier',
                        title='Customer Distribution by CLV Tier',
                        color='CLV Tier',
                        color_discrete_map=colors
                    )
                    
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )
                    
                    fig.update_layout(
                        height=450,
                        font=dict(color='#2c3e50', size=14),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Business Insights with enhanced visibility
    if os.path.exists("artifacts/business_metrics/business_insights.csv"):
        st.subheader("💡 Key Business Insights")
        
        insights_df = pd.read_csv("artifacts/business_metrics/business_insights.csv")
        
        # Create a container for insights
        insights_container = st.container()
        
        with insights_container:
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
                <div class="{box_class}">
                    <h4 style="margin-top: 0; color: inherit;">{icon} {insight['category']}: {insight['insight']}</h4>
                    <p style="margin-bottom: 10px; color: inherit;"><b>Recommended Action:</b> {insight['action']}</p>
                    <p style="margin-bottom: 0; color: inherit;"><b>Expected Impact:</b> {insight['impact']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Profitability Analysis with improved styling
    st.subheader("💼 Profitability Analysis")
    
    if os.path.exists("artifacts/business_metrics/profitability_by_segment.csv"):
        prof_df = pd.read_csv("artifacts/business_metrics/profitability_by_segment.csv", index_col=0)
        
        fig = go.Figure()
        
        # Revenue bars
        fig.add_trace(go.Bar(
            x=prof_df.index,
            y=prof_df['gross_revenue'],
            name='Revenue',
            marker_color='#3498db',
            text=[f'${v:,.0f}' for v in prof_df['gross_revenue']],
            textposition='outside'
        ))
        
        # Profit bars
        fig.add_trace(go.Bar(
            x=prof_df.index,
            y=prof_df['gross_profit'],
            name='Profit',
            marker_color='#27ae60',
            text=[f'${v:,.0f}' for v in prof_df['gross_profit']],
            textposition='outside'
        ))
        
        # Profit margin line
        fig.add_trace(go.Scatter(
            x=prof_df.index,
            y=prof_df['profit_margin'] * 100,
            name='Profit Margin %',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='Profitability by Customer Segment',
            yaxis=dict(title='Amount ($)', color='#2c3e50'),
            yaxis2=dict(title='Profit Margin (%)', overlaying='y', side='right', color='#2c3e50'),
            hovermode='x unified',
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#2c3e50', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        
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
        
        # Create a styled container
        with st.container():
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
            
            # Display results in styled cards
            st.markdown('<div style="background-color: #f8f9fa; padding: 2rem; border-radius: 15px; margin-top: 2rem;">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(
                    create_custom_kpi_card(
                        "Sample Size per Variant",
                        f"{results['sample_size_per_variant']:,}",
                        icon="👥"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    create_custom_kpi_card(
                        "Total Sample Size",
                        f"{results['total_sample_size']:,}",
                        icon="👫"
                    ),
                    unsafe_allow_html=True
                )
            
            with col3:
                duration = calculator.get_duration_estimate(
                    results['sample_size_per_variant'],
                    daily_traffic
                )
                st.markdown(
                    create_custom_kpi_card(
                        "Estimated Duration",
                        f"{duration['days_with_buffer']} days",
                        icon="📅"
                    ),
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
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
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_vline(
                x=mde, 
                line_dash="dash", 
                line_color="#e74c3c",
                annotation_text=f"Your MDE: {mde}%"
            )
            
            fig.update_layout(
                title="Sample Size vs Minimum Detectable Effect",
                xaxis_title="MDE (%)",
                yaxis_title="Sample Size per Variant",
                yaxis_type="log",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2c3e50')
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("A/B Test Results Analyzer")
        
        # Styled input container
        with st.container():
            st.markdown('<div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">', unsafe_allow_html=True)
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Analyze Results", type="primary"):
            results = calculator.analyze_results_simple(
                control_visitors, control_conversions,
                treatment_visitors, treatment_conversions
            )
            
            # Display results in styled cards
            st.markdown('<div style="background-color: #f8f9fa; padding: 2rem; border-radius: 15px; margin: 2rem 0;">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lift_color = "positive" if results['relative_lift'] > 0 else "negative"
                st.markdown(
                    create_custom_kpi_card(
                        "Relative Lift",
                        f"{results['relative_lift']*100:.1f}%",
                        "Positive" if results['relative_lift'] > 0 else "Negative",
                        lift_color,
                        "📈" if results['relative_lift'] > 0 else "📉"
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                sig_color = "positive" if results['p_value'] < 0.05 else "negative"
                st.markdown(
                    create_custom_kpi_card(
                        "P-value",
                        f"{results['p_value']:.4f}",
                        "Significant" if results['p_value'] < 0.05 else "Not Significant",
                        sig_color,
                        "✅" if results['p_value'] < 0.05 else "❌"
                    ),
                    unsafe_allow_html=True
                )
            
            with col3:
                conf_color = "positive" if results['prob_treatment_better'] > 0.5 else "negative"
                st.markdown(
                    create_custom_kpi_card(
                        "Confidence",
                        f"{results['prob_treatment_better']*100:.1f}%",
                        "Treatment Better" if results['prob_treatment_better'] > 0.5 else "Control Better",
                        conf_color,
                        "🎯"
                    ),
                    unsafe_allow_html=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
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
                    marker_color=['#3498db', '#27ae60'],
                    text=[f"{results['control_rate']*100:.1f}%", f"{results['treatment_rate']*100:.1f}%"],
                    textposition='outside'
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
                go.Scatter(x=x_range*100, y=control_dist, name='Control', fill='tozeroy',
                          line=dict(color='#3498db')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=x_range*100, y=treatment_dist, name='Treatment', fill='tozeroy',
                          line=dict(color='#27ae60')),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Conversion Rate (%)", row=1, col=1)
            fig.update_xaxes(title_text="Conversion Rate (%)", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Probability Density", row=1, col=2)
            
            fig.update_layout(
                height=450,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2c3e50')
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Decision recommendation with styled boxes
            if results['is_significant']:
                if results['relative_lift'] > 0:
                    st.markdown("""
                    <div class="success-box">
                        <h3 style="margin-top: 0;">✅ Test Result: Significant Improvement</h3>
                        <p style="font-size: 1.1rem;">The treatment shows a statistically significant improvement of <b>{:.1f}%</b>.</p>
                        <p style="margin-bottom: 0;"><b>Recommendation:</b> Roll out the treatment to 100% of traffic.</p>
                    </div>
                    """.format(results['relative_lift']*100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h3 style="margin-top: 0;">❌ Test Result: Significant Degradation</h3>
                        <p style="font-size: 1.1rem;">The treatment shows a statistically significant decrease of <b>{:.1f}%</b>.</p>
                        <p style="margin-bottom: 0;"><b>Recommendation:</b> Do not implement the treatment. Investigate what went wrong.</p>
                    </div>
                    """.format(abs(results['relative_lift'])*100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-box">
                    <h3 style="margin-top: 0;">⚠️ Test Result: Not Statistically Significant</h3>
                    <p style="font-size: 1.1rem;">The observed difference of <b>{:.1f}%</b> is not statistically significant (p={:.3f}).</p>
                    <p style="margin-bottom: 0;"><b>Recommendation:</b> Continue testing to gather more data or consider this difference negligible.</p>
                </div>
                """.format(results['relative_lift']*100, results['p_value']), unsafe_allow_html=True)

def show_advanced_segmentation(data):
    """Show advanced customer segmentation"""
    st.header("🎯 Advanced Customer Segmentation")
    
    if 'customer_features' not in data:
        st.error("Customer features not found. Please run the pipeline first.")
        return
    
    df = data['customer_features']
    
    # Segmentation options
    col1, col2 = st.columns(2)
    
    with col1:
        segmentation_type = st.selectbox(
            "Segmentation Type",
            ["RFM Segmentation", "CLV Tiers", "Behavioral Clusters", "Geographic Analysis"]
        )
    
    with col2:
        visualization = st.selectbox(
            "Visualization",
            ["Scatter Plot", "Sunburst", "Treemap", "Distribution"]
        )
    
    if segmentation_type == "RFM Segmentation":
        # Create RFM segments if not already present
        if 'RFM_segment' in df.columns:
            # Segment distribution
            segment_counts = df['RFM_segment'].value_counts()
        elif 'total_orders' in df.columns and 'total_revenue' in df.columns:
            # Map the new column names to expected RFM names
            df['frequency'] = df['total_orders']
            df['monetary_value'] = df['total_revenue']
            
            # Calculate recency if we have last_purchase_date
            if 'last_purchase_date' in df.columns:
                reference_date = pd.Timestamp('2018-08-01')
                df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
                df['recency_days'] = (reference_date - df['last_purchase_date']).dt.days
            else:
                # Use a default recency based on churn status
                df['recency_days'] = df['churned'].apply(lambda x: 200 if x else 30)
            # Create RFM segments on the fly
            # Create RFM scores
            df['R_score'] = pd.qcut(df['recency_days'].rank(method='first'), q=5, labels=[5,4,3,2,1])
            df['F_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
            df['M_score'] = pd.qcut(df['monetary_value'].rank(method='first'), q=5, labels=[1,2,3,4,5])
            
            # Combine scores
            df['RFM_score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
            
            # Create segments
            def get_segment(row):
                if row['RFM_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                    return 'Champions'
                elif row['RFM_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                    return 'Loyal Customers'
                elif row['RFM_score'] in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                    return 'Potential Loyalists'
                elif row['RFM_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                    return 'New Customers'
                elif row['RFM_score'] in ['525', '524', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
                    return 'Promising'
                elif row['RFM_score'] in ['535', '534', '443', '434', '343', '334', '325', '324']:
                    return 'Need Attention'
                elif row['RFM_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                    return 'Cannot Lose Them'
                elif row['RFM_score'] in ['332', '322', '231', '241', '251', '233', '232', '223', '222', '132', '123', '122', '212', '211']:
                    return 'Hibernating'
                elif row['RFM_score'] in ['155', '154', '144', '214', '215', '115', '114', '113']:
                    return 'At Risk'
                else:
                    return 'Lost'
            
            df['RFM_segment'] = df.apply(get_segment, axis=1)
            segment_counts = df['RFM_segment'].value_counts()
            
            if visualization == "Sunburst":
                # Create sunburst chart
                fig = px.sunburst(
                    df,
                    path=['RFM_segment'],
                    values='monetary_value',
                    title='Customer Segments by Revenue'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif visualization == "Scatter Plot":
                # 3D scatter plot
                fig = px.scatter_3d(
                    df,
                    x='recency_days',
                    y='frequency',
                    z='monetary_value',
                    color='RFM_segment',
                    title='RFM Segmentation 3D View',
                    labels={
                        'recency_days': 'Recency (days)',
                        'frequency': 'Frequency',
                        'monetary_value': 'Monetary Value ($)'
                    }
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Segment metrics
            st.subheader("Segment Performance Metrics")
            segment_metrics = df.groupby('RFM_segment').agg({
                'customer_unique_id': 'count',
                'monetary_value': ['mean', 'sum'],
                'frequency': 'mean',
                'recency_days': 'mean'
            }).round(2)
            
            st.dataframe(segment_metrics, use_container_width=True)
        else:
            st.info("RFM segments not found in the data. Please run advanced feature engineering.")
    
    elif segmentation_type == "CLV Tiers":
        # CLV-based segmentation
        value_col = 'monetary_value' if 'monetary_value' in df.columns else 'total_revenue'
        
        if 'clv_tier' in df.columns or value_col in df.columns:
            # Create CLV tiers if not present
            if 'clv_tier' not in df.columns:
                try:
                    df['clv_tier'] = pd.qcut(df[value_col], q=4, labels=['Low', 'Medium', 'High', 'VIP'], duplicates='drop')
                except:
                    # Fallback to percentile-based tiers
                    percentiles = df[value_col].quantile([0.25, 0.5, 0.75])
                    df['clv_tier'] = pd.cut(df[value_col], 
                                           bins=[-np.inf, percentiles[0.25], percentiles[0.5], percentiles[0.75], np.inf],
                                           labels=['Low', 'Medium', 'High', 'VIP'])
            
            # Visualization
            if visualization == "Distribution":
                fig = px.histogram(
                    df,
                    x=value_col,
                    color='clv_tier',
                    title='Customer Distribution by CLV',
                    nbins=50
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            elif visualization == "Treemap":
                tier_data = df.groupby('clv_tier').agg({
                    'customer_unique_id': 'count',
                    value_col: 'sum'
                }).reset_index()
                tier_data.columns = ['CLV Tier', 'Count', 'Total Revenue']
                
                fig = px.treemap(
                    tier_data,
                    path=['CLV Tier'],
                    values='Total Revenue',
                    title='Revenue Distribution by CLV Tier'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            elif visualization == "Scatter Plot":
                freq_col = 'frequency' if 'frequency' in df.columns else 'total_orders'
                fig = px.scatter(
                    df,
                    x=freq_col,
                    y=value_col,
                    color='clv_tier',
                    title='CLV Tiers: Frequency vs Value',
                    labels={freq_col: 'Purchase Frequency', value_col: 'Total Value ($)'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            elif visualization == "Sunburst":
                # Create sunburst with state and CLV tier
                if 'customer_state' in df.columns:
                    sunburst_data = df.groupby(['customer_state', 'clv_tier']).agg({
                        'customer_unique_id': 'count',
                        value_col: 'sum'
                    }).reset_index()
                    sunburst_data.columns = ['State', 'CLV Tier', 'Count', 'Revenue']
                    
                    fig = px.sunburst(
                        sunburst_data,
                        path=['CLV Tier', 'State'],
                        values='Revenue',
                        title='CLV Tiers by State'
                    )
                else:
                    fig = px.sunburst(
                        df,
                        path=['clv_tier'],
                        values=value_col,
                        title='CLV Tier Distribution'
                    )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            st.subheader("CLV Tier Metrics")
            tier_metrics = df.groupby('clv_tier').agg({
                'customer_unique_id': 'count',
                value_col: ['mean', 'sum'],
                'churned': 'mean' if 'churned' in df.columns else 'count'
            }).round(2)
            st.dataframe(tier_metrics, use_container_width=True)
    
    elif segmentation_type == "Behavioral Clusters":
        st.subheader("Behavioral Customer Clusters")
        
        # Create behavioral features
        freq_col = 'frequency' if 'frequency' in df.columns else 'total_orders'
        value_col = 'monetary_value' if 'monetary_value' in df.columns else 'total_revenue'
        
        # Normalize features for clustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        features = [freq_col, value_col]
        if 'customer_lifetime_days' in df.columns:
            features.append('customer_lifetime_days')
        
        X = df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        n_clusters = st.slider("Number of Clusters", 3, 8, 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Visualizations
        if visualization == "Scatter Plot":
            fig = px.scatter(
                df,
                x=freq_col,
                y=value_col,
                color='cluster',
                title=f'Behavioral Clusters ({n_clusters} clusters)',
                labels={freq_col: 'Purchase Frequency', value_col: 'Total Value ($)'},
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif visualization == "Distribution":
            cluster_dist = df['cluster'].value_counts().sort_index()
            fig = px.bar(
                x=cluster_dist.index,
                y=cluster_dist.values,
                title='Customer Distribution by Cluster',
                labels={'x': 'Cluster', 'y': 'Number of Customers'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        elif visualization == "Treemap":
            cluster_data = df.groupby('cluster').agg({
                'customer_unique_id': 'count',
                value_col: 'sum'
            }).reset_index()
            cluster_data.columns = ['Cluster', 'Count', 'Revenue']
            cluster_data['Cluster'] = 'Cluster ' + cluster_data['Cluster'].astype(str)
            
            fig = px.treemap(
                cluster_data,
                path=['Cluster'],
                values='Revenue',
                title='Revenue by Behavioral Cluster'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif visualization == "Sunburst":
            # Add churn info if available
            if 'churned' in df.columns:
                df['churn_status'] = df['churned'].map({0: 'Active', 1: 'Churned'})
                sunburst_data = df.groupby(['cluster', 'churn_status']).size().reset_index(name='count')
                sunburst_data['cluster'] = 'Cluster ' + sunburst_data['cluster'].astype(str)
                
                fig = px.sunburst(
                    sunburst_data,
                    path=['cluster', 'churn_status'],
                    values='count',
                    title='Clusters by Churn Status'
                )
            else:
                fig = px.sunburst(
                    df,
                    path=['cluster'],
                    title='Behavioral Clusters'
                )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        cluster_profile = df.groupby('cluster')[features].mean().round(2)
        st.dataframe(cluster_profile, use_container_width=True)
    
    elif segmentation_type == "Geographic Analysis":
        st.subheader("Geographic Customer Segmentation")
        
        if 'customer_state' in df.columns:
            value_col = 'monetary_value' if 'monetary_value' in df.columns else 'total_revenue'
            
            # State-level analysis
            state_data = df.groupby('customer_state').agg({
                'customer_unique_id': 'count',
                value_col: ['sum', 'mean'],
                'churned': 'mean' if 'churned' in df.columns else 'count'
            }).round(2)
            
            state_data.columns = ['Customer_Count', 'Total_Revenue', 'Avg_Revenue', 'Churn_Rate']
            state_data = state_data.sort_values('Total_Revenue', ascending=False).head(20)
            
            if visualization == "Treemap":
                fig = px.treemap(
                    state_data.reset_index(),
                    path=['customer_state'],
                    values='Total_Revenue',
                    title='Revenue Distribution by State',
                    color='Avg_Revenue',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            elif visualization == "Distribution":
                fig = px.bar(
                    state_data.reset_index().head(15),
                    x='customer_state',
                    y='Total_Revenue',
                    title='Top 15 States by Revenue',
                    color='Avg_Revenue',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            elif visualization == "Scatter Plot":
                fig = px.scatter(
                    state_data.reset_index(),
                    x='Customer_Count',
                    y='Avg_Revenue',
                    size='Total_Revenue',
                    color='Churn_Rate' if 'Churn_Rate' in state_data.columns else 'Avg_Revenue',
                    text='customer_state',
                    title='States: Customer Count vs Average Revenue',
                    color_continuous_scale='RdYlBu_r'
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            elif visualization == "Sunburst":
                # Group states by region (simplified)
                state_data_reset = state_data.reset_index()
                state_data_reset['Region'] = state_data_reset['customer_state'].map({
                    'SP': 'Southeast', 'RJ': 'Southeast', 'MG': 'Southeast', 'ES': 'Southeast',
                    'PR': 'South', 'SC': 'South', 'RS': 'South',
                    'BA': 'Northeast', 'PE': 'Northeast', 'CE': 'Northeast',
                    'DF': 'Central', 'GO': 'Central', 'MT': 'Central', 'MS': 'Central'
                }).fillna('Other')
                
                fig = px.sunburst(
                    state_data_reset,
                    path=['Region', 'customer_state'],
                    values='Total_Revenue',
                    title='Revenue by Region and State'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display top states
            st.subheader("Top States by Revenue")
            display_data = state_data.reset_index().head(10)
            st.dataframe(display_data, use_container_width=True)
        else:
            st.warning("Geographic data not available in the dataset.")

def show_churn_clv_analysis(data):
    """Show churn and CLV analysis"""
    st.header("💰 Churn & Customer Lifetime Value Analysis")
    
    if 'customer_features' not in data:
        st.error("Customer features not found. Please run the pipeline first.")
        return
    
    df = data['customer_features']
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Churn Analysis", "CLV Analysis", "Churn Prevention"])
    
    with tab1:
        st.subheader("Churn Risk Analysis")
        
        # Churn distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn rate by segment
            if 'RFM_segment' in df.columns:
                churn_by_segment = df.groupby('RFM_segment')['churned'].agg(['mean', 'count'])
                churn_by_segment.columns = ['Churn Rate', 'Customer Count']
                churn_by_segment['Churn Rate'] = (churn_by_segment['Churn Rate'] * 100).round(1)
                
                fig = px.bar(
                    churn_by_segment.reset_index(),
                    x='RFM_segment',
                    y='Churn Rate',
                    title='Churn Rate by Customer Segment',
                    text='Churn Rate'
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn factors - use the correct column names
            freq_col = 'frequency' if 'frequency' in df.columns else 'total_orders'
            value_col = 'monetary_value' if 'monetary_value' in df.columns else 'total_revenue'
            
            churn_factors = {
                'Low Frequency': (df[df['churned']==1][freq_col] < df[freq_col].median()).mean() * 100,
                'Low Value': (df[df['churned']==1][value_col] < df[value_col].median()).mean() * 100,
                'Single Purchase': (df[df['churned']==1][freq_col] == 1).mean() * 100
            }
            
            fig = px.bar(
                x=list(churn_factors.keys()),
                y=list(churn_factors.values()),
                title='Common Churn Factors',
                labels={'x': 'Factor', 'y': 'Percentage (%)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Customer Lifetime Value Analysis")
        
        # CLV distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # CLV histogram - use correct column name
            value_col = 'monetary_value' if 'monetary_value' in df.columns else 'total_revenue'
            fig = px.histogram(
                df,
                x=value_col,
                nbins=50,
                title='CLV Distribution',
                labels={value_col: 'Customer Lifetime Value ($)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CLV by cohort
            if 'first_purchase_date' in df.columns:
                df['cohort'] = pd.to_datetime(df['first_purchase_date']).dt.to_period('M')
                value_col = 'monetary_value' if 'monetary_value' in df.columns else 'total_revenue'
                cohort_clv = df.groupby('cohort')[value_col].mean()
                
                fig = px.line(
                    x=cohort_clv.index.astype(str),
                    y=cohort_clv.values,
                    title='Average CLV by Cohort',
                    labels={'x': 'Cohort', 'y': 'Average CLV ($)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def show_recommendation_engine(data):
    """Show recommendation engine insights"""
    st.header("🤖 Recommendation Engine")
    
    # Basic metrics
    if 'customer_features' in data:
        df = data['customer_features']
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            freq_col = 'frequency' if 'frequency' in df.columns else 'total_orders'
            avg_orders = df[freq_col].mean() if freq_col in df.columns else 0
            st.metric("Avg Orders per Customer", f"{avg_orders:.1f}")
        
        with col3:
            value_col = 'monetary_value' if 'monetary_value' in df.columns else 'total_revenue'
            avg_revenue = df[value_col].mean() if value_col in df.columns else 0
            st.metric("Avg Revenue per Customer", f"${avg_revenue:.2f}")
        
        with col4:
            potential_revenue = df[value_col].sum() * 0.25  # 25% uplift potential
            st.metric("Revenue Uplift Potential", f"${potential_revenue:,.0f}")
        
        # Recommendation Strategy
        st.subheader("📊 Recommendation Strategy")
        
        # Create tabs for different recommendation types
        tab1, tab2, tab3 = st.tabs(["Cross-Sell Opportunities", "Upsell Potential", "Win-Back Campaigns"])
        
        with tab1:
            st.markdown("### Cross-Sell Opportunities")
            
            # Segment customers by purchase frequency
            freq_col = 'frequency' if 'frequency' in df.columns else 'total_orders'
            
            # Create purchase frequency segments
            try:
                df['purchase_segment'] = pd.qcut(
                    df[freq_col], 
                    q=[0, .25, .5, .75, 1], 
                    labels=['Low', 'Medium', 'High', 'VIP'],
                    duplicates='drop'  # Handle duplicate values
                )
            except ValueError:
                # If still fails, use simple cut based on unique values
                df['purchase_segment'] = pd.cut(
                    df[freq_col],
                    bins=[-np.inf, 1, 2, 3, np.inf],
                    labels=['Low', 'Medium', 'High', 'VIP']
                )
            
            segment_stats = df.groupby('purchase_segment').agg({
                'customer_unique_id': 'count',
                value_col: 'mean',
                freq_col: 'mean'
            }).round(2)
            
            segment_stats.columns = ['Customer Count', 'Avg Revenue', 'Avg Orders']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of segments
                fig = px.bar(
                    segment_stats.reset_index(),
                    x='purchase_segment',
                    y='Customer Count',
                    title='Customers by Purchase Frequency',
                    color='purchase_segment',
                    color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#27ae60', 'VIP': '#9b59b6'}
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue potential
                fig = px.bar(
                    segment_stats.reset_index(),
                    x='purchase_segment',
                    y='Avg Revenue',
                    title='Average Revenue by Segment',
                    color='purchase_segment',
                    color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#27ae60', 'VIP': '#9b59b6'}
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("#### 🎯 Cross-Sell Recommendations:")
            st.success("""
            - **Low Frequency Customers**: Send personalized product recommendations based on their first purchase
            - **Medium Frequency**: Introduce product bundles and complementary items
            - **High Frequency**: Offer loyalty rewards and exclusive products
            - **VIP Customers**: Provide early access to new products and premium services
            """)
        
        with tab2:
            st.markdown("### Upsell Potential Analysis")
            
            # Calculate upsell potential
            value_percentiles = df[value_col].quantile([0.25, 0.5, 0.75])
            
            upsell_data = {
                'Customer Segment': ['Bottom 25%', 'Middle 50%', 'Top 25%'],
                'Current Avg Revenue': [
                    df[df[value_col] <= value_percentiles[0.25]][value_col].mean(),
                    df[(df[value_col] > value_percentiles[0.25]) & (df[value_col] <= value_percentiles[0.75])][value_col].mean(),
                    df[df[value_col] > value_percentiles[0.75]][value_col].mean()
                ],
                'Target Revenue': [
                    value_percentiles[0.5],  # Move to median
                    value_percentiles[0.75],  # Move to 75th percentile
                    df[df[value_col] > value_percentiles[0.75]][value_col].mean() * 1.2  # 20% increase
                ]
            }
            
            upsell_df = pd.DataFrame(upsell_data)
            upsell_df['Upsell Potential'] = upsell_df['Target Revenue'] - upsell_df['Current Avg Revenue']
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Current Revenue', x=upsell_df['Customer Segment'], y=upsell_df['Current Avg Revenue']))
            fig.add_trace(go.Bar(name='Target Revenue', x=upsell_df['Customer Segment'], y=upsell_df['Target Revenue']))
            
            fig.update_layout(
                title='Upsell Potential by Customer Segment',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Total upsell potential
            total_upsell = upsell_df['Upsell Potential'].sum() * len(df) / 3  # Rough estimate
            st.metric("Total Upsell Potential", f"${total_upsell:,.0f}")
        
        with tab3:
            st.markdown("### Win-Back Campaign Targets")
            
            if 'churned' in df.columns:
                churned_customers = df[df['churned'] == 1]
                active_customers = df[df['churned'] == 0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Churned Customers", f"{len(churned_customers):,}")
                    st.metric("Churn Rate", f"{(len(churned_customers) / len(df) * 100):.1f}%")
                
                with col2:
                    lost_revenue = churned_customers[value_col].sum()
                    st.metric("Lost Revenue", f"${lost_revenue:,.0f}")
                    st.metric("Avg Lost Revenue per Customer", f"${lost_revenue / len(churned_customers):.2f}")
                
                # Win-back priority segments
                st.markdown("#### 🎯 Win-Back Priority Segments:")
                
                # High-value churned customers
                high_value_churned = churned_customers[churned_customers[value_col] > churned_customers[value_col].median()]
                
                priority_segments = {
                    'High-Value Lost': len(high_value_churned),
                    'Recent Churners': len(churned_customers[churned_customers['last_purchase_date'] > '2018-05-01']) if 'last_purchase_date' in churned_customers.columns else 0,
                    'Single Purchase Churners': len(churned_customers[churned_customers[freq_col] == 1])
                }
                
                fig = px.pie(
                    values=list(priority_segments.values()),
                    names=list(priority_segments.keys()),
                    title='Win-Back Priority Segments'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Churn data not available. Run the complete pipeline to generate churn predictions.")

def show_business_insights_page(data):
    """Show business insights"""
    st.header("💡 Business Insights & Recommendations")
    
    if os.path.exists("artifacts/business_metrics/business_insights.csv"):
        insights_df = pd.read_csv("artifacts/business_metrics/business_insights.csv")
        
        # Group insights by category
        categories = insights_df['category'].unique()
        
        for category in categories:
            st.subheader(f"{category}")
            
            category_insights = insights_df[insights_df['category'] == category]
            
            for _, insight in category_insights.iterrows():
                with st.expander(f"{insight['insight']}", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Recommended Action:** {insight['action']}")
                        st.markdown(f"**Expected Impact:** {insight['impact']}")
                    
                    with col2:
                        # Priority indicator
                        priority = insight.get('priority', 'Medium')
                        if priority == 'High':
                            st.error(f"Priority: {priority}")
                        elif priority == 'Medium':
                            st.warning(f"Priority: {priority}")
                        else:
                            st.success(f"Priority: {priority}")
    else:
        st.info("No business insights found. Please run the business metrics calculator first.")

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
        
        # Display controls with styled container
        with st.container():
            st.markdown('<div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">', unsafe_allow_html=True)
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
        
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
                textfont={"size": 10, "color": "#2c3e50"},
                hoverongaps=False,
                colorbar=dict(title="Retention %", titlefont=dict(color="#2c3e50"))
            ))
            
            fig.update_layout(
                title=f'{metric_type} by Cohort',
                xaxis_title='Months Since First Purchase',
                yaxis_title='Cohort',
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2c3e50')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif display_type == "Curves":
            # Retention curves with confidence bands
            fig = go.Figure()
            
            # Use a color palette for different cohorts
            colors = px.colors.qualitative.Set3
            
            # Add individual cohort lines
            for i, cohort in enumerate(retention_display.index):
                fig.add_trace(go.Scatter(
                    x=list(range(len(retention_display.columns))),
                    y=retention_display.loc[cohort].values,
                    mode='lines',
                    name=str(cohort),
                    line=dict(width=2, color=colors[i % len(colors)]),
                    opacity=0.7
                ))
            
            # Add average line
            avg_retention = retention_display.mean()
            std_retention = retention_display.std()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(avg_retention))),
                y=avg_retention.values,
                mode='lines',
                name='Average',
                line=dict(color='#2c3e50', width=4)
            ))
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=list(range(len(avg_retention))) + list(range(len(avg_retention)))[::-1],
                y=list(avg_retention + std_retention) + list(avg_retention - std_retention)[::-1],
                fill='toself',
                fillcolor='rgba(52, 152, 219, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
            
            fig.update_layout(
                title=f'{metric_type} Curves Over Time',
                xaxis_title='Months Since First Purchase',
                yaxis_title=f'{metric_type} (%)',
                hovermode='x unified',
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2c3e50')
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif display_type == "3D Surface":
            # 3D surface plot
            fig = go.Figure(data=[go.Surface(
                z=retention_display.values,
                x=list(range(len(retention_display.columns))),
                y=list(range(len(retention_display.index))),
                colorscale='RdYlGn',
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
                )
            )])
            
            fig.update_layout(
                title=f'3D {metric_type} Surface',
                scene=dict(
                    xaxis_title='Months Since First Purchase',
                    yaxis_title='Cohort Index',
                    zaxis_title=f'{metric_type} (%)',
                    xaxis=dict(gridcolor='#f0f0f0'),
                    yaxis=dict(gridcolor='#f0f0f0'),
                    zaxis=dict(gridcolor='#f0f0f0')
                ),
                height=600,
                paper_bgcolor='white',
                font=dict(color='#2c3e50')
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
                              '3-Month Retention', 'Quality Score'),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Cohort Size
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['cohort_size'],
                          mode='lines+markers', name='Cohort Size',
                          line=dict(color='#3498db', width=3),
                          marker=dict(size=8)),
                row=1, col=1
            )
            
            # Revenue per Customer
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['revenue_per_customer'],
                          mode='lines+markers', name='Revenue/Customer',
                          line=dict(color='#27ae60', width=3),
                          marker=dict(size=8)),
                row=1, col=2
            )
            
            # 3-Month Retention
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['retention_3m'],
                          mode='lines+markers', name='3M Retention',
                          line=dict(color='#e74c3c', width=3),
                          marker=dict(size=8)),
                row=2, col=1
            )
            
            # Quality Score
            fig.add_trace(
                go.Scatter(x=quality_df['cohort_month'], y=quality_df['quality_score'],
                          mode='lines+markers', name='Quality Score',
                          line=dict(color='#9b59b6', width=3),
                          marker=dict(size=8)),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2c3e50')
            )
            
            # Update all axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            
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
                                   name='6-Month LTV', marker_color='#85C1E9'))
                fig.add_trace(go.Bar(x=ltv_df['cohort_month'], y=ltv_df['ltv_12m'],
                                   name='12-Month LTV', marker_color='#3498db'))
                fig.add_trace(go.Bar(x=ltv_df['cohort_month'], y=ltv_df['ltv_24m'],
                                   name='24-Month LTV', marker_color='#1A5276'))
                
                fig.update_layout(
                    title='LTV Progression by Cohort',
                    xaxis_title='Cohort',
                    yaxis_title='Lifetime Value ($)',
                    barmode='group',
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2c3e50'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Payback period
                fig = px.line(
                    ltv_df,
                    x='cohort_month',
                    y='months_to_payback',
                    title='CAC Payback Period by Cohort',
                    labels={'months_to_payback': 'Months to Payback'},
                    color_discrete_sequence=['#e74c3c']
                )
                
                fig.add_hline(y=12, line_dash="dash", line_color="#27ae60",
                            annotation_text="12-month target")
                
                fig.update_traces(line=dict(width=3), marker=dict(size=8))
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#2c3e50')
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                
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
    
    # Sidebar navigation with enhanced styling
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
    
    elif selected_page == "Advanced Segmentation":
        show_advanced_segmentation(data)
    
    elif selected_page == "Churn & CLV Analysis":
        show_churn_clv_analysis(data)
    
    elif selected_page == "Recommendation Engine":
        show_recommendation_engine(data)
    
    elif selected_page == "Business Insights":
        show_business_insights_page(data)
    
    # Footer with enhanced styling
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>
        <p style='margin: 0; color: #2c3e50; font-weight: 600;'>Customer Analytics Platform v2.0</p>
        <p style='margin: 0; font-size: 0.8em; color: #7f8c8d;'>Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()