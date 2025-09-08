"""
Prophet Model for Sales Forecasting
Implements Facebook Prophet for time series forecasting with trend and seasonality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Make Prophet optional
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from prophet.plot import plot_cross_validation_metric
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None
    cross_validation = None
    performance_metrics = None
    plot_cross_validation_metric = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ProphetForecaster:
    """Prophet model for advanced time series forecasting"""
    
    def __init__(self):
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet is not installed. Install with: pip install prophet")
            self.model = None
            self.forecast = None
            self.cv_results = None
            self.available = False
        else:
            self.model = None
            self.forecast = None
            self.cv_results = None
            self.available = True
            logger.info("ProphetForecaster initialized")
    
    def prepare_data_for_prophet(self, timeseries, date_col='date', value_col='revenue'):
        """Prepare data in Prophet's required format"""
        try:
            logger.info("Preparing data for Prophet...")
            
            # Prophet requires columns named 'ds' and 'y'
            prophet_df = pd.DataFrame({
                'ds': timeseries.index if isinstance(timeseries, pd.Series) else timeseries[date_col],
                'y': timeseries.values if isinstance(timeseries, pd.Series) else timeseries[value_col]
            })
            
            # Ensure datetime
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Remove any NaN values
            prophet_df = prophet_df.dropna()
            
            logger.info(f"Prepared {len(prophet_df)} data points for Prophet")
            return prophet_df
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise CustomException("Failed to prepare data for Prophet", e)
    
    def add_seasonality(self, model, custom_seasonalities=None):
        """Add custom seasonality to Prophet model"""
        try:
            # Default seasonalities are already included
            # Add custom seasonalities if provided
            if custom_seasonalities:
                for seasonality in custom_seasonalities:
                    model.add_seasonality(
                        name=seasonality['name'],
                        period=seasonality['period'],
                        fourier_order=seasonality.get('fourier_order', 5)
                    )
                    logger.info(f"Added custom seasonality: {seasonality['name']}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error adding seasonality: {e}")
            raise CustomException("Failed to add seasonality", e)
    
    def add_holidays(self, model, country='BR'):
        """Add country-specific holidays"""
        try:
            # For Brazil, we can add major holidays
            brazilian_holidays = pd.DataFrame({
                'holiday': ['new_year', 'carnival', 'christmas', 'black_friday'],
                'ds': pd.to_datetime(['2017-01-01', '2017-02-28', '2017-12-25', '2017-11-24']),
                'lower_window': [0, -2, -1, -1],
                'upper_window': [1, 2, 1, 3]
            })
            
            # Add more years
            holidays_list = []
            for year in range(2016, 2020):
                yearly_holidays = brazilian_holidays.copy()
                yearly_holidays['ds'] = yearly_holidays['ds'].apply(
                    lambda x: x.replace(year=year)
                )
                holidays_list.append(yearly_holidays)
            
            all_holidays = pd.concat(holidays_list, ignore_index=True)
            model.holidays = all_holidays
            
            logger.info(f"Added {len(all_holidays)} holiday effects")
            return model
            
        except Exception as e:
            logger.error(f"Error adding holidays: {e}")
            return model
    
    def fit(self, timeseries, 
            growth='linear',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            seasonality_mode='additive',
            add_country_holidays=True,
            custom_seasonalities=None):
        """Fit Prophet model"""
        try:
            if not self.available:
                logger.warning("Prophet not available, cannot fit model")
                return None
                
            logger.info("Fitting Prophet model...")
            
            # Prepare data
            prophet_df = self.prepare_data_for_prophet(timeseries)
            
            # Initialize model with parameters
            self.model = Prophet(
                growth=growth,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                seasonality_mode=seasonality_mode,
                daily_seasonality=False,  # Auto-detect
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            # Add holidays
            if add_country_holidays:
                self.model = self.add_holidays(self.model)
            
            # Add custom seasonalities
            if custom_seasonalities:
                self.model = self.add_seasonality(self.model, custom_seasonalities)
            
            # Fit model
            self.model.fit(prophet_df)
            
            logger.info("Model fitted successfully")
            logger.info(f"Detected {len(self.model.changepoints)} changepoints")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            raise CustomException("Failed to fit Prophet model", e)
    
    def predict(self, periods=30, freq='D', include_history=True):
        """Generate predictions"""
        try:
            if self.model is None:
                raise ValueError("Model must be fitted before prediction")
            
            logger.info(f"Generating predictions for {periods} periods...")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods,
                freq=freq,
                include_history=include_history
            )
            
            # Make predictions
            self.forecast = self.model.predict(future)
            
            logger.info(f"Generated predictions for {len(future)} periods")
            
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise CustomException("Failed to generate predictions", e)
    
    def evaluate(self, actual, predicted):
        """Evaluate model performance"""
        try:
            # Ensure same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            metrics = {
                'MAE': mean_absolute_error(actual, predicted),
                'MSE': mean_squared_error(actual, predicted),
                'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
                'MAPE': mean_absolute_percentage_error(actual, predicted) * 100
            }
            
            logger.info("Model Evaluation Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)
    
    def cross_validate_prophet(self, initial='365 days', period='30 days', horizon='30 days'):
        """Perform cross-validation for Prophet model"""
        try:
            if self.model is None:
                raise ValueError("Model must be fitted before cross-validation")
            
            logger.info(f"Performing cross-validation with initial={initial}, period={period}, horizon={horizon}")
            
            # Perform cross-validation
            self.cv_results = cross_validation(
                self.model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes"
            )
            
            # Calculate performance metrics
            df_metrics = performance_metrics(self.cv_results)
            
            logger.info("Cross-validation completed")
            logger.info(f"Average MAPE: {df_metrics['mape'].mean():.4f}")
            logger.info(f"Average RMSE: {df_metrics['rmse'].mean():.4f}")
            
            return self.cv_results, df_metrics
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise CustomException("Failed to perform cross-validation", e)
    
    def plot_forecast(self, save_path=None):
        """Plot forecast with components"""
        try:
            if self.model is None or self.forecast is None:
                raise ValueError("Model must be fitted and predictions generated")
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot forecast
            ax1 = axes[0]
            self.model.plot(self.forecast, ax=ax1)
            ax1.set_title('Prophet Forecast')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Revenue')
            
            # Plot trend
            ax2 = axes[1]
            ax2.plot(self.forecast['ds'], self.forecast['trend'])
            ax2.set_title('Trend Component')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Trend')
            ax2.grid(True, alpha=0.3)
            
            # Plot weekly seasonality
            ax3 = axes[2]
            weekly = self.forecast[['ds', 'weekly']].drop_duplicates('weekly')
            ax3.plot(weekly['ds'], weekly['weekly'])
            ax3.set_title('Weekly Seasonality')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Weekly Effect')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Forecast plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            raise CustomException("Failed to plot forecast", e)
    
    def plot_components(self, save_path=None):
        """Plot model components"""
        try:
            if self.model is None or self.forecast is None:
                raise ValueError("Model must be fitted and predictions generated")
            
            fig = self.model.plot_components(self.forecast)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Components plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting components: {e}")
            raise CustomException("Failed to plot components", e)
    
    def plot_cross_validation_metrics(self, metric='mape', save_path=None):
        """Plot cross-validation metrics"""
        try:
            if self.cv_results is None:
                raise ValueError("Must run cross-validation first")
            
            fig = plot_cross_validation_metric(self.cv_results, metric=metric)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"CV metrics plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting CV metrics: {e}")
            raise CustomException("Failed to plot CV metrics", e)
    
    def detect_anomalies(self, threshold=0.95):
        """Detect anomalies in historical data"""
        try:
            if self.forecast is None:
                raise ValueError("Must generate forecast first")
            
            # Get historical forecasts only
            historical = self.forecast[self.forecast['ds'] <= self.forecast['ds'].max() - pd.Timedelta(days=30)]
            
            # Calculate prediction intervals
            lower_bound = historical['yhat_lower']
            upper_bound = historical['yhat_upper']
            actual = historical['y']
            
            # Detect anomalies
            anomalies = historical[(actual < lower_bound) | (actual > upper_bound)].copy()
            
            if len(anomalies) > 0:
                anomalies['deviation'] = np.where(
                    anomalies['y'] > anomalies['yhat_upper'],
                    (anomalies['y'] - anomalies['yhat_upper']) / anomalies['yhat_upper'],
                    (anomalies['yhat_lower'] - anomalies['y']) / anomalies['yhat_lower']
                )
                
                logger.info(f"Detected {len(anomalies)} anomalies ({len(anomalies)/len(historical)*100:.1f}%)")
                
                # Plot anomalies
                plt.figure(figsize=(12, 6))
                plt.scatter(historical['ds'], historical['y'], alpha=0.5, s=10, label='Normal')
                plt.scatter(anomalies['ds'], anomalies['y'], color='red', s=50, label='Anomaly')
                plt.plot(historical['ds'], historical['yhat'], 'g-', label='Forecast')
                plt.fill_between(historical['ds'], lower_bound, upper_bound, alpha=0.3, color='gray')
                plt.legend()
                plt.title('Anomaly Detection')
                plt.xlabel('Date')
                plt.ylabel('Revenue')
                plt.xticks(rotation=45)
                plt.tight_layout()
            else:
                logger.info("No anomalies detected")
                anomalies = pd.DataFrame()
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise CustomException("Failed to detect anomalies", e)
    
    def get_model_parameters(self):
        """Get model parameters and settings"""
        try:
            if self.model is None:
                raise ValueError("Model must be fitted first")
            
            params = {
                'growth': self.model.growth,
                'n_changepoints': len(self.model.changepoints),
                'changepoint_prior_scale': self.model.changepoint_prior_scale,
                'seasonality_prior_scale': self.model.seasonality_prior_scale,
                'holidays_prior_scale': self.model.holidays_prior_scale,
                'seasonality_mode': self.model.seasonality_mode,
                'seasonalities': list(self.model.seasonalities.keys()),
                'holidays': len(self.model.holidays) if self.model.holidays is not None else 0
            }
            
            return params
            
        except Exception as e:
            logger.error(f"Error getting model parameters: {e}")
            raise CustomException("Failed to get model parameters", e)

if __name__ == "__main__":
    # Test Prophet model
    logger.info("Testing Prophet model...")
    
    # Load sample data
    data = pd.read_csv("artifacts/forecasting/prepared_data/daily_sales.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Create forecaster
    forecaster = ProphetForecaster()
    
    # Fit model
    forecaster.fit(data['revenue'])
    
    # Generate predictions
    forecast = forecaster.predict(periods=30)
    
    # Plot results
    forecaster.plot_forecast()
    forecaster.plot_components()
    
    # Detect anomalies
    anomalies = forecaster.detect_anomalies()