"""
ARIMA Model for Sales Forecasting
Implements ARIMA (AutoRegressive Integrated Moving Average) for time series prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ARIMAForecaster:
    """ARIMA model for sales forecasting"""
    
    def __init__(self):
        self.model = None
        self.order = None
        self.results = None
        logger.info("ARIMAForecaster initialized")
    
    def analyze_series(self, timeseries, lags=40):
        """Analyze time series properties"""
        try:
            logger.info("Analyzing time series properties")
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Plot time series
            axes[0].plot(timeseries.index, timeseries.values)
            axes[0].set_title('Time Series')
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Value')
            
            # Plot ACF
            plot_acf(timeseries.dropna(), lags=lags, ax=axes[1])
            axes[1].set_title('Autocorrelation Function (ACF)')
            
            # Plot PACF
            plot_pacf(timeseries.dropna(), lags=lags, ax=axes[2])
            axes[2].set_title('Partial Autocorrelation Function (PACF)')
            
            plt.tight_layout()
            
            # Perform ADF test
            adf_result = adfuller(timeseries.dropna())
            logger.info(f"ADF Statistic: {adf_result[0]:.4f}")
            logger.info(f"p-value: {adf_result[1]:.4f}")
            
            return fig, adf_result
            
        except Exception as e:
            logger.error(f"Error analyzing series: {e}")
            raise CustomException("Failed to analyze time series", e)
    
    def find_optimal_parameters(self, timeseries, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        """Find optimal ARIMA parameters using grid search"""
        try:
            logger.info("Finding optimal ARIMA parameters...")
            
            best_aic = np.inf
            best_params = None
            results_list = []
            
            # Grid search
            for p in range(p_range[0], p_range[1] + 1):
                for d in range(d_range[0], d_range[1] + 1):
                    for q in range(q_range[0], q_range[1] + 1):
                        try:
                            model = ARIMA(timeseries, order=(p, d, q))
                            results = model.fit()
                            
                            results_list.append({
                                'order': (p, d, q),
                                'aic': results.aic,
                                'bic': results.bic,
                                'mae': np.mean(np.abs(results.resid))
                            })
                            
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_params = (p, d, q)
                                
                            logger.info(f"ARIMA{(p,d,q)} - AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to fit ARIMA{(p,d,q)}: {str(e)}")
                            continue
            
            logger.info(f"Best parameters: ARIMA{best_params} with AIC: {best_aic:.2f}")
            
            # Create results dataframe
            results_df = pd.DataFrame(results_list)
            
            return best_params, results_df
            
        except Exception as e:
            logger.error(f"Error finding optimal parameters: {e}")
            raise CustomException("Failed to find optimal parameters", e)
    
    def fit(self, timeseries, order=None, auto_select=True):
        """Fit ARIMA model"""
        try:
            if auto_select and order is None:
                logger.info("Auto-selecting ARIMA parameters...")
                order, _ = self.find_optimal_parameters(timeseries)
            elif order is None:
                order = (1, 1, 1)  # Default order
            
            self.order = order
            logger.info(f"Fitting ARIMA{order} model...")
            
            self.model = ARIMA(timeseries, order=order)
            self.results = self.model.fit()
            
            logger.info("Model fitted successfully")
            logger.info(f"AIC: {self.results.aic:.2f}")
            logger.info(f"BIC: {self.results.bic:.2f}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            raise CustomException("Failed to fit ARIMA model", e)
    
    def forecast(self, steps=30, confidence_level=0.95):
        """Generate forecasts"""
        try:
            if self.results is None:
                raise ValueError("Model must be fitted before forecasting")
            
            logger.info(f"Generating {steps}-step ahead forecast...")
            
            # Get forecast
            forecast = self.results.forecast(steps=steps)
            
            # Get prediction intervals
            forecast_df = pd.DataFrame({
                'forecast': forecast,
            }, index=pd.date_range(
                start=self.results.fittedvalues.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            ))
            
            # Calculate confidence intervals
            residual_std = np.sqrt(self.results.mse)
            z_score = 1.96 if confidence_level == 0.95 else 2.58
            
            forecast_df['lower_bound'] = forecast_df['forecast'] - z_score * residual_std
            forecast_df['upper_bound'] = forecast_df['forecast'] + z_score * residual_std
            
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
    
    def plot_forecast(self, timeseries, forecast_df, title="ARIMA Forecast", save_path=None):
        """Plot historical data and forecast"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(timeseries.index, timeseries.values, label='Historical', color='blue')
            
            # Plot forecast
            plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')
            
            # Plot confidence intervals
            plt.fill_between(
                forecast_df.index,
                forecast_df['lower_bound'],
                forecast_df['upper_bound'],
                alpha=0.3,
                color='red',
                label='95% Confidence Interval'
            )
            
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Forecast plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            raise CustomException("Failed to plot forecast", e)
    
    def get_model_summary(self):
        """Get model summary"""
        try:
            if self.results is None:
                raise ValueError("Model must be fitted first")
            
            summary = {
                'order': self.order,
                'aic': self.results.aic,
                'bic': self.results.bic,
                'log_likelihood': self.results.llf,
                'residual_variance': self.results.mse,
                'params': self.results.params.to_dict()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            raise CustomException("Failed to get model summary", e)
    
    def rolling_forecast(self, timeseries, test_size=30, retrain_interval=7):
        """Perform rolling forecast evaluation"""
        try:
            logger.info(f"Performing rolling forecast with test size: {test_size}")
            
            train_size = len(timeseries) - test_size
            predictions = []
            actuals = []
            
            for i in range(test_size):
                # Get training data
                train_data = timeseries[:train_size + i]
                
                # Retrain model at specified intervals
                if i % retrain_interval == 0:
                    self.fit(train_data, order=self.order, auto_select=False)
                
                # Make one-step forecast
                forecast = self.results.forecast(steps=1)[0]
                predictions.append(forecast)
                actuals.append(timeseries[train_size + i])
            
            # Calculate metrics
            metrics = self.evaluate(actuals, predictions)
            
            return predictions, actuals, metrics
            
        except Exception as e:
            logger.error(f"Error in rolling forecast: {e}")
            raise CustomException("Failed to perform rolling forecast", e)

if __name__ == "__main__":
    # Test ARIMA model
    logger.info("Testing ARIMA model...")
    
    # Load sample data
    data = pd.read_csv("artifacts/forecasting/prepared_data/daily_sales.csv")
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    
    # Create forecaster
    forecaster = ARIMAForecaster()
    
    # Analyze series
    fig, adf_result = forecaster.analyze_series(data['revenue'])
    
    # Fit model
    forecaster.fit(data['revenue'], auto_select=True)
    
    # Generate forecast
    forecast = forecaster.forecast(steps=30)
    
    # Plot results
    forecaster.plot_forecast(data['revenue'], forecast)