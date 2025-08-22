"""
SARIMA Model for Seasonal Sales Forecasting
Implements SARIMA (Seasonal ARIMA) to capture seasonal patterns in sales data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class SARIMAForecaster:
    """SARIMA model for seasonal time series forecasting"""
    
    def __init__(self):
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.results = None
        logger.info("SARIMAForecaster initialized")
    
    def detect_seasonality(self, timeseries, freq='D'):
        """Detect and analyze seasonality in time series"""
        try:
            logger.info("Detecting seasonality in time series...")
            
            # Perform seasonal decomposition
            if freq == 'D':
                period = 7  # Weekly seasonality for daily data
            elif freq == 'W':
                period = 4  # Monthly seasonality for weekly data
            elif freq == 'M':
                period = 12  # Yearly seasonality for monthly data
            else:
                period = 4  # Default
            
            decomposition = seasonal_decompose(timeseries, model='additive', period=period)
            
            # Create decomposition plot
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            timeseries.plot(ax=axes[0], title='Original Time Series')
            decomposition.trend.plot(ax=axes[1], title='Trend Component')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
            decomposition.resid.plot(ax=axes[3], title='Residual Component')
            
            plt.tight_layout()
            
            # Calculate seasonality strength
            var_seasonal = np.var(decomposition.seasonal.dropna())
            var_resid = np.var(decomposition.resid.dropna())
            seasonality_strength = var_seasonal / (var_seasonal + var_resid)
            
            logger.info(f"Seasonality strength: {seasonality_strength:.4f}")
            logger.info(f"Detected period: {period}")
            
            return decomposition, period, seasonality_strength, fig
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            raise CustomException("Failed to detect seasonality", e)
    
    def find_optimal_parameters(self, timeseries, seasonal_period=7, 
                              p_range=(0, 2), d_range=(0, 1), q_range=(0, 2),
                              P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2)):
        """Find optimal SARIMA parameters using grid search"""
        try:
            logger.info("Finding optimal SARIMA parameters...")
            
            # Generate all parameter combinations
            p = range(p_range[0], p_range[1] + 1)
            d = range(d_range[0], d_range[1] + 1)
            q = range(q_range[0], q_range[1] + 1)
            P = range(P_range[0], P_range[1] + 1)
            D = range(D_range[0], D_range[1] + 1)
            Q = range(Q_range[0], Q_range[1] + 1)
            
            parameters = list(itertools.product(p, d, q))
            seasonal_parameters = list(itertools.product(P, D, Q, [seasonal_period]))
            
            best_aic = np.inf
            best_params = None
            best_seasonal_params = None
            results_list = []
            
            total_combinations = len(parameters) * len(seasonal_parameters)
            logger.info(f"Testing {total_combinations} parameter combinations...")
            
            # Grid search with limited iterations
            for param in parameters[:5]:  # Limit to avoid long runtime
                for seasonal_param in seasonal_parameters[:5]:
                    try:
                        model = SARIMAX(timeseries, 
                                      order=param,
                                      seasonal_order=seasonal_param,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
                        
                        results = model.fit(disp=False)
                        
                        results_list.append({
                            'order': param,
                            'seasonal_order': seasonal_param,
                            'aic': results.aic,
                            'bic': results.bic
                        })
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_params = param
                            best_seasonal_params = seasonal_param
                        
                        logger.info(f"SARIMA{param}x{seasonal_param} - AIC: {results.aic:.2f}")
                        
                    except Exception as e:
                        continue
            
            logger.info(f"Best parameters: SARIMA{best_params}x{best_seasonal_params} with AIC: {best_aic:.2f}")
            
            return best_params, best_seasonal_params, pd.DataFrame(results_list)
            
        except Exception as e:
            logger.error(f"Error finding optimal parameters: {e}")
            raise CustomException("Failed to find optimal parameters", e)
    
    def fit(self, timeseries, order=None, seasonal_order=None, auto_select=True):
        """Fit SARIMA model"""
        try:
            if auto_select and (order is None or seasonal_order is None):
                logger.info("Auto-selecting SARIMA parameters...")
                order, seasonal_order, _ = self.find_optimal_parameters(timeseries)
            elif order is None or seasonal_order is None:
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 7)  # Default weekly seasonality
            
            self.order = order
            self.seasonal_order = seasonal_order
            
            logger.info(f"Fitting SARIMA{order}x{seasonal_order} model...")
            
            self.model = SARIMAX(timeseries,
                               order=order,
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            
            self.results = self.model.fit(disp=False)
            
            logger.info("Model fitted successfully")
            logger.info(f"AIC: {self.results.aic:.2f}")
            logger.info(f"BIC: {self.results.bic:.2f}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            raise CustomException("Failed to fit SARIMA model", e)
    
    def forecast(self, steps=30, confidence_level=0.95):
        """Generate forecasts with prediction intervals"""
        try:
            if self.results is None:
                raise ValueError("Model must be fitted before forecasting")
            
            logger.info(f"Generating {steps}-step ahead forecast...")
            
            # Get forecast with confidence intervals
            forecast = self.results.forecast(steps=steps)
            forecast_ci = self.results.get_forecast(steps=steps).conf_int(alpha=1-confidence_level)
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'forecast': forecast,
                'lower_bound': forecast_ci.iloc[:, 0],
                'upper_bound': forecast_ci.iloc[:, 1]
            }, index=pd.date_range(
                start=self.results.fittedvalues.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            ))
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise CustomException("Failed to generate forecast", e)
    
    def evaluate(self, actual, predicted):
        """Evaluate model performance"""
        try:
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
    
    def plot_diagnostics(self, save_path=None):
        """Plot model diagnostics"""
        try:
            if self.results is None:
                raise ValueError("Model must be fitted first")
            
            fig = self.results.plot_diagnostics(figsize=(12, 10))
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Diagnostics plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting diagnostics: {e}")
            raise CustomException("Failed to plot diagnostics", e)
    
    def plot_forecast(self, timeseries, forecast_df, title="SARIMA Forecast", save_path=None):
        """Plot historical data and forecast with seasonal patterns"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Full time series with forecast
            ax1.plot(timeseries.index, timeseries.values, label='Historical', color='blue')
            ax1.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')
            ax1.fill_between(
                forecast_df.index,
                forecast_df['lower_bound'],
                forecast_df['upper_bound'],
                alpha=0.3,
                color='red',
                label='95% Confidence Interval'
            )
            ax1.set_title(f"{title} - Full Time Series")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Zoomed view of last 60 days + forecast
            last_60_days = timeseries.iloc[-60:]
            ax2.plot(last_60_days.index, last_60_days.values, label='Historical', color='blue')
            ax2.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')
            ax2.fill_between(
                forecast_df.index,
                forecast_df['lower_bound'],
                forecast_df['upper_bound'],
                alpha=0.3,
                color='red'
            )
            ax2.set_title(f"{title} - Last 60 Days + Forecast")
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Forecast plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            raise CustomException("Failed to plot forecast", e)
    
    def cross_validate(self, timeseries, n_splits=3, test_size=30):
        """Perform time series cross-validation"""
        try:
            logger.info(f"Performing {n_splits}-fold time series cross-validation...")
            
            cv_results = []
            data_length = len(timeseries)
            
            for i in range(n_splits):
                # Calculate split points
                test_start = data_length - (i + 1) * test_size
                test_end = test_start + test_size
                
                if test_start < len(timeseries) * 0.5:  # Ensure enough training data
                    break
                
                # Split data
                train = timeseries[:test_start]
                test = timeseries[test_start:test_end]
                
                # Fit model
                self.fit(train, order=self.order, seasonal_order=self.seasonal_order, auto_select=False)
                
                # Forecast
                forecast = self.results.forecast(steps=len(test))
                
                # Evaluate
                metrics = self.evaluate(test.values, forecast.values)
                metrics['fold'] = i + 1
                cv_results.append(metrics)
                
                logger.info(f"Fold {i+1} - RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
            
            # Calculate average metrics
            cv_df = pd.DataFrame(cv_results)
            avg_metrics = cv_df.drop('fold', axis=1).mean()
            
            logger.info("\nAverage CV Metrics:")
            for metric, value in avg_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            return cv_df, avg_metrics
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise CustomException("Failed to perform cross-validation", e)
    
    def get_model_summary(self):
        """Get comprehensive model summary"""
        try:
            if self.results is None:
                raise ValueError("Model must be fitted first")
            
            summary = {
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'aic': self.results.aic,
                'bic': self.results.bic,
                'log_likelihood': self.results.llf,
                'residual_variance': self.results.mse,
                'seasonal_period': self.seasonal_order[3],
                'params': self.results.params.to_dict()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            raise CustomException("Failed to get model summary", e)

if __name__ == "__main__":
    # Test SARIMA model
    logger.info("Testing SARIMA model...")
    
    # Load sample data
    data = pd.read_csv("artifacts/forecasting/prepared_data/daily_sales.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Create forecaster
    forecaster = SARIMAForecaster()
    
    # Detect seasonality
    decomp, period, strength, fig = forecaster.detect_seasonality(data['revenue'])
    
    # Fit model
    forecaster.fit(data['revenue'], auto_select=True)
    
    # Generate forecast
    forecast = forecaster.forecast(steps=30)
    
    # Plot results
    forecaster.plot_forecast(data['revenue'], forecast)