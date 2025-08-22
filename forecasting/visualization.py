"""
Visualization Module for Forecasting Results
Creates interactive and static visualizations for forecast analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger

logger = get_logger(__name__)

class ForecastVisualizer:
    """Create various visualizations for forecast results"""
    
    def __init__(self, style='seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
        logger.info("ForecastVisualizer initialized")
    
    def plot_time_series_decomposition(self, data, title="Time Series Decomposition", save_path=None):
        """Create enhanced time series decomposition plot"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Perform decomposition
            decomposition = seasonal_decompose(data, model='multiplicative', period=7)
            
            # Create subplots
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            
            # Original series
            axes[0].plot(data.index, data.values, color='navy', linewidth=1.5)
            axes[0].set_title(f'{title} - Original Series', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Value')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                        color='darkgreen', linewidth=2)
            axes[1].set_title('Trend Component', fontsize=12)
            axes[1].set_ylabel('Trend')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                        color='darkred', linewidth=1)
            axes[2].set_title('Seasonal Component', fontsize=12)
            axes[2].set_ylabel('Seasonal')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].scatter(decomposition.resid.index, decomposition.resid.values, 
                           alpha=0.5, s=10, color='purple')
            axes[3].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[3].set_title('Residual Component', fontsize=12)
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Date')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Decomposition plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating decomposition plot: {e}")
            return None
    
    def create_interactive_forecast_plot(self, historical_data, forecasts_dict, title="Interactive Sales Forecast"):
        """Create interactive plot using Plotly"""
        try:
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='black', width=2)
            ))
            
            # Colors for different models
            colors = {'ARIMA': 'blue', 'SARIMA': 'red', 'Prophet': 'green'}
            
            # Add forecasts
            for model_name, forecast_df in forecasts_dict.items():
                if isinstance(forecast_df, pd.DataFrame) and 'forecast' in forecast_df.columns:
                    # Add forecast line
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df['forecast'],
                        mode='lines',
                        name=f'{model_name} Forecast',
                        line=dict(color=colors.get(model_name, 'gray'), width=2)
                    ))
                    
                    # Add confidence interval
                    if 'upper_bound' in forecast_df.columns and 'lower_bound' in forecast_df.columns:
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                            y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                            fill='toself',
                            fillcolor=f'rgba({",".join(map(str, self._hex_to_rgb(colors.get(model_name, "gray"))))},0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            name=f'{model_name} CI'
                        ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Revenue',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template='plotly_white'
            )
            
            # Add range slider
            fig.update_xaxes(rangeslider_visible=True)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive plot: {e}")
            return None
    
    def plot_forecast_comparison(self, historical_data, forecasts_dict, save_path=None):
        """Create comprehensive forecast comparison plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: All forecasts together
            ax1 = axes[0, 0]
            
            # Historical data (last 90 days for clarity)
            recent_history = historical_data.iloc[-90:]
            ax1.plot(recent_history.index, recent_history.values, 
                    'k-', linewidth=2, label='Historical')
            
            # Forecasts
            colors = ['blue', 'red', 'green']
            for i, (model_name, forecast_df) in enumerate(forecasts_dict.items()):
                if isinstance(forecast_df, pd.DataFrame) and 'forecast' in forecast_df.columns:
                    ax1.plot(forecast_df.index, forecast_df['forecast'], 
                            color=colors[i], linewidth=2, label=model_name)
                    
                    if 'upper_bound' in forecast_df.columns:
                        ax1.fill_between(forecast_df.index, 
                                       forecast_df['lower_bound'], 
                                       forecast_df['upper_bound'],
                                       alpha=0.2, color=colors[i])
            
            ax1.set_title('Forecast Comparison - All Models', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Revenue')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Forecast values distribution
            ax2 = axes[0, 1]
            forecast_data = []
            labels = []
            
            for model_name, forecast_df in forecasts_dict.items():
                if isinstance(forecast_df, pd.DataFrame) and 'forecast' in forecast_df.columns:
                    forecast_data.append(forecast_df['forecast'].values)
                    labels.append(model_name)
            
            if forecast_data:
                bp = ax2.boxplot(forecast_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors[:len(forecast_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            
            ax2.set_title('Forecast Distribution by Model', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Forecast Value')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Prediction intervals width
            ax3 = axes[1, 0]
            for i, (model_name, forecast_df) in enumerate(forecasts_dict.items()):
                if isinstance(forecast_df, pd.DataFrame) and 'upper_bound' in forecast_df.columns:
                    interval_width = forecast_df['upper_bound'] - forecast_df['lower_bound']
                    ax3.plot(forecast_df.index, interval_width, 
                            color=colors[i], linewidth=2, label=f'{model_name} PI Width')
            
            ax3.set_title('Prediction Interval Width Over Time', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Interval Width')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Forecast trends
            ax4 = axes[1, 1]
            for i, (model_name, forecast_df) in enumerate(forecasts_dict.items()):
                if isinstance(forecast_df, pd.DataFrame) and 'forecast' in forecast_df.columns:
                    # Calculate daily changes
                    daily_change = forecast_df['forecast'].pct_change() * 100
                    ax4.plot(forecast_df.index[1:], daily_change[1:], 
                            color=colors[i], linewidth=1.5, label=f'{model_name} Daily Change %')
            
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('Daily Percentage Change in Forecasts', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Daily Change (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Comparison plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
            return None
    
    def plot_model_diagnostics_summary(self, diagnostics_dict, save_path=None):
        """Create summary of model diagnostics"""
        try:
            # Extract metrics
            models = []
            metrics_data = {'MAE': [], 'RMSE': [], 'MAPE': []}
            
            for model_name, diagnostics in diagnostics_dict.items():
                if 'metrics' in diagnostics:
                    models.append(model_name)
                    for metric in metrics_data.keys():
                        metrics_data[metric].append(diagnostics['metrics'].get(metric, 0))
            
            if not models:
                logger.warning("No metrics data available for diagnostics plot")
                return None
            
            # Create plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Metrics comparison bar chart
            ax1 = axes[0, 0]
            x = np.arange(len(models))
            width = 0.25
            
            for i, (metric, values) in enumerate(metrics_data.items()):
                ax1.bar(x + i*width, values, width, label=metric)
            
            ax1.set_xlabel('Model')
            ax1.set_ylabel('Error Value')
            ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(models)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Normalized metrics radar chart
            ax2 = axes[0, 1]
            angles = np.linspace(0, 2 * np.pi, len(metrics_data), endpoint=False).tolist()
            
            for i, model in enumerate(models):
                values = []
                for metric in metrics_data.keys():
                    # Normalize by max value
                    max_val = max(metrics_data[metric])
                    if max_val > 0:
                        values.append(metrics_data[metric][i] / max_val)
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the circle
                angles_plot = angles + angles[:1]
                
                ax2.plot(angles_plot, values, 'o-', linewidth=2, label=model)
                ax2.fill(angles_plot, values, alpha=0.25)
            
            ax2.set_xticks(angles)
            ax2.set_xticklabels(list(metrics_data.keys()))
            ax2.set_ylim(0, 1.2)
            ax2.set_title('Normalized Performance Radar', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: MAPE comparison
            ax3 = axes[1, 0]
            mape_values = metrics_data['MAPE']
            bars = ax3.bar(models, mape_values, color=['blue', 'red', 'green'][:len(models)])
            
            # Add value labels on bars
            for bar, value in zip(bars, mape_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%', ha='center', va='bottom')
            
            ax3.set_ylabel('MAPE (%)')
            ax3.set_title('Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Summary statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create summary text
            summary_text = "Model Performance Summary\n" + "="*30 + "\n\n"
            
            # Find best model for each metric
            for metric, values in metrics_data.items():
                if values:
                    best_idx = np.argmin(values)
                    best_model = models[best_idx]
                    best_value = values[best_idx]
                    summary_text += f"Best {metric}: {best_model} ({best_value:.4f})\n"
            
            # Add additional info if available
            summary_text += "\n" + "Additional Information\n" + "-"*30 + "\n"
            for model_name, diagnostics in diagnostics_dict.items():
                if 'seasonality_strength' in diagnostics:
                    summary_text += f"{model_name} seasonality: {diagnostics['seasonality_strength']:.4f}\n"
                if 'anomalies' in diagnostics:
                    summary_text += f"{model_name} anomalies: {diagnostics['anomalies']}\n"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Diagnostics summary saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating diagnostics summary: {e}")
            return None
    
    def create_forecast_dashboard(self, historical_data, forecasts_dict, diagnostics_dict):
        """Create interactive dashboard with multiple views"""
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Historical Data & Forecasts', 'Model Performance',
                              'Forecast Distribution', 'Prediction Intervals',
                              'Daily Revenue Trends', 'Model Comparison'),
                specs=[[{"secondary_y": False}, {"type": "bar"}],
                      [{"type": "box"}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"type": "scatter"}]],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot 1: Historical & Forecasts
            fig.add_trace(
                go.Scatter(x=historical_data.index[-180:], 
                          y=historical_data.values[-180:],
                          name='Historical',
                          line=dict(color='black', width=2)),
                row=1, col=1
            )
            
            colors = {'ARIMA': 'blue', 'SARIMA': 'red', 'Prophet': 'green'}
            for model_name, forecast_df in forecasts_dict.items():
                if isinstance(forecast_df, pd.DataFrame) and 'forecast' in forecast_df.columns:
                    fig.add_trace(
                        go.Scatter(x=forecast_df.index, 
                                  y=forecast_df['forecast'],
                                  name=f'{model_name}',
                                  line=dict(color=colors.get(model_name, 'gray'))),
                        row=1, col=1
                    )
            
            # Plot 2: Model Performance Metrics
            if diagnostics_dict:
                models = []
                rmse_values = []
                for model, diag in diagnostics_dict.items():
                    if 'metrics' in diag:
                        models.append(model)
                        rmse_values.append(diag['metrics'].get('RMSE', 0))
                
                fig.add_trace(
                    go.Bar(x=models, y=rmse_values, name='RMSE',
                          marker_color=['blue', 'red', 'green'][:len(models)]),
                    row=1, col=2
                )
            
            # Plot 3: Forecast Distribution
            for model_name, forecast_df in forecasts_dict.items():
                if isinstance(forecast_df, pd.DataFrame) and 'forecast' in forecast_df.columns:
                    fig.add_trace(
                        go.Box(y=forecast_df['forecast'], name=model_name,
                              marker_color=colors.get(model_name, 'gray')),
                        row=2, col=1
                    )
            
            # Plot 4: Prediction Intervals
            for model_name, forecast_df in forecasts_dict.items():
                if isinstance(forecast_df, pd.DataFrame) and 'upper_bound' in forecast_df.columns:
                    interval_width = forecast_df['upper_bound'] - forecast_df['lower_bound']
                    fig.add_trace(
                        go.Scatter(x=forecast_df.index, y=interval_width,
                                  name=f'{model_name} PI',
                                  line=dict(color=colors.get(model_name, 'gray'))),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(
                height=1200,
                showlegend=True,
                title_text="Sales Forecasting Dashboard",
                title_font_size=20
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Revenue", row=1, col=1)
            fig.update_xaxes(title_text="Model", row=1, col=2)
            fig.update_yaxes(title_text="RMSE", row=1, col=2)
            fig.update_yaxes(title_text="Revenue", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)
            fig.update_yaxes(title_text="Interval Width", row=2, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return None
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def save_all_visualizations(self, historical_data, forecasts_dict, diagnostics_dict, output_path):
        """Generate and save all visualizations"""
        try:
            os.makedirs(output_path, exist_ok=True)
            
            logger.info("Generating all visualizations...")
            
            # 1. Time series decomposition
            self.plot_time_series_decomposition(
                historical_data,
                save_path=f"{output_path}/time_series_decomposition.png"
            )
            
            # 2. Forecast comparison
            self.plot_forecast_comparison(
                historical_data,
                forecasts_dict,
                save_path=f"{output_path}/forecast_comparison.png"
            )
            
            # 3. Model diagnostics summary
            self.plot_model_diagnostics_summary(
                diagnostics_dict,
                save_path=f"{output_path}/model_diagnostics.png"
            )
            
            # 4. Interactive forecast plot
            interactive_fig = self.create_interactive_forecast_plot(
                historical_data,
                forecasts_dict
            )
            if interactive_fig:
                interactive_fig.write_html(f"{output_path}/interactive_forecast.html")
            
            # 5. Dashboard
            dashboard_fig = self.create_forecast_dashboard(
                historical_data,
                forecasts_dict,
                diagnostics_dict
            )
            if dashboard_fig:
                dashboard_fig.write_html(f"{output_path}/forecast_dashboard.html")
            
            logger.info(f"All visualizations saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {e}")

if __name__ == "__main__":
    # Test visualization module
    logger.info("Testing ForecastVisualizer...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    historical = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    # Create sample forecasts
    forecast_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    forecasts = {
        'ARIMA': pd.DataFrame({
            'forecast': np.random.randn(30).cumsum() + 100,
            'lower_bound': np.random.randn(30).cumsum() + 95,
            'upper_bound': np.random.randn(30).cumsum() + 105
        }, index=forecast_dates)
    }
    
    # Create visualizer
    visualizer = ForecastVisualizer()
    
    # Test plots
    visualizer.plot_time_series_decomposition(historical)
    plt.show()