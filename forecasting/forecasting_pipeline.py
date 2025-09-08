"""
Forecasting Pipeline - Runs all forecasting models and compares results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.time_series_prep import TimeSeriesPreprocessor
from forecasting.arima_model import ARIMAForecaster
from forecasting.sarima_model import SARIMAForecaster
from forecasting.prophet_model import ProphetForecaster
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ForecastingPipeline:
    """Main pipeline to run all forecasting models"""
    
    def __init__(self, data_path="data", output_path="artifacts/forecasting"):
        self.data_path = data_path
        self.output_path = output_path
        self.results = {}
        
        # Create output directories
        os.makedirs(f"{output_path}/results", exist_ok=True)
        os.makedirs(f"{output_path}/plots", exist_ok=True)
        os.makedirs(f"{output_path}/prepared_data", exist_ok=True)
        
        logger.info("ForecastingPipeline initialized")
    
    def prepare_data(self):
        """Prepare time series data"""
        try:
            logger.info("="*70)
            logger.info("STEP 1: Data Preparation")
            logger.info("="*70)
            
            # Initialize preprocessor
            self.preprocessor = TimeSeriesPreprocessor(self.data_path)
            self.preprocessor.load_data()
            
            # Save prepared data
            self.preprocessor.save_prepared_data(f"{self.output_path}/prepared_data")
            
            # Load prepared data
            self.daily_sales = pd.read_csv(f"{self.output_path}/prepared_data/daily_sales.csv")
            self.daily_sales['date'] = pd.to_datetime(self.daily_sales['date'])
            self.daily_sales.set_index('date', inplace=True)
            
            logger.info(f"Prepared {len(self.daily_sales)} days of sales data")
            logger.info(f"Date range: {self.daily_sales.index.min()} to {self.daily_sales.index.max()}")
            
            # Check stationarity
            is_stationary = self.preprocessor.check_stationarity(
                self.daily_sales['revenue'], 
                title='Daily Revenue'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise CustomException("Data preparation failed", e)
    
    def run_arima_forecast(self, forecast_days=30):
        """Run ARIMA model"""
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 2: ARIMA Forecasting")
            logger.info("="*70)
            
            # Split data
            train, test = self.preprocessor.train_test_split_timeseries(
                self.daily_sales, 
                test_size=0.1
            )
            
            # Initialize ARIMA
            arima = ARIMAForecaster()
            
            # Analyze series
            fig, _ = arima.analyze_series(train['revenue'])
            plt.savefig(f"{self.output_path}/plots/arima_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Fit model
            arima.fit(train['revenue'], auto_select=True)
            
            # Generate forecast
            forecast_df = arima.forecast(steps=forecast_days)
            
            # Plot results
            arima.plot_forecast(
                train['revenue'], 
                forecast_df,
                title="ARIMA Sales Forecast",
                save_path=f"{self.output_path}/plots/arima_forecast.png"
            )
            plt.close()
            
            # Plot diagnostics
            arima.plot_diagnostics(
                save_path=f"{self.output_path}/plots/arima_diagnostics.png"
            )
            plt.close()
            
            # Evaluate on test set
            test_forecast = arima.results.forecast(steps=len(test))
            metrics = arima.evaluate(test['revenue'].values, test_forecast.values)
            
            # Store results
            self.results['ARIMA'] = {
                'model': arima,
                'forecast': forecast_df,
                'metrics': metrics,
                'parameters': arima.get_model_summary()
            }
            
            logger.info(f"ARIMA completed - Test RMSE: {metrics['RMSE']:.2f}")
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}")
            self.results['ARIMA'] = {'error': str(e)}
    
    def run_sarima_forecast(self, forecast_days=30):
        """Run SARIMA model"""
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 3: SARIMA Forecasting")
            logger.info("="*70)
            
            # Split data
            train, test = self.preprocessor.train_test_split_timeseries(
                self.daily_sales, 
                test_size=0.1
            )
            
            # Initialize SARIMA
            sarima = SARIMAForecaster()
            
            # Detect seasonality
            decomp, period, strength, fig = sarima.detect_seasonality(train['revenue'])
            plt.savefig(f"{self.output_path}/plots/sarima_seasonality.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Fit model (with limited parameter search for speed)
            sarima.fit(train['revenue'], auto_select=True)
            
            # Generate forecast
            forecast_df = sarima.forecast(steps=forecast_days)
            
            # Plot results
            sarima.plot_forecast(
                train['revenue'], 
                forecast_df,
                title="SARIMA Sales Forecast",
                save_path=f"{self.output_path}/plots/sarima_forecast.png"
            )
            plt.close()
            
            # Plot diagnostics
            sarima.plot_diagnostics(
                save_path=f"{self.output_path}/plots/sarima_diagnostics.png"
            )
            plt.close()
            
            # Evaluate on test set
            test_forecast = sarima.results.forecast(steps=len(test))
            metrics = sarima.evaluate(test['revenue'].values, test_forecast.values)
            
            # Store results
            self.results['SARIMA'] = {
                'model': sarima,
                'forecast': forecast_df,
                'metrics': metrics,
                'parameters': sarima.get_model_summary(),
                'seasonality_strength': strength
            }
            
            logger.info(f"SARIMA completed - Test RMSE: {metrics['RMSE']:.2f}")
            
        except Exception as e:
            logger.error(f"Error in SARIMA forecasting: {e}")
            self.results['SARIMA'] = {'error': str(e)}
    
    def run_prophet_forecast(self, forecast_days=30):
        """Run Prophet model"""
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 4: Prophet Forecasting")
            logger.info("="*70)
            
            # Initialize Prophet and check if available
            prophet = ProphetForecaster()
            if not prophet.available:
                logger.warning("Prophet is not installed, skipping Prophet forecast")
                self.results['Prophet'] = {'error': 'Prophet not installed'}
                return
            
            # Split data
            train, test = self.preprocessor.train_test_split_timeseries(
                self.daily_sales, 
                test_size=0.1
            )
            
            # Fit model
            prophet.fit(
                train['revenue'],
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative'
            )
            
            # Generate predictions
            forecast = prophet.predict(periods=forecast_days)
            
            # Plot results
            prophet.plot_forecast(
                save_path=f"{self.output_path}/plots/prophet_forecast.png"
            )
            plt.close()
            
            # Plot components
            prophet.plot_components(
                save_path=f"{self.output_path}/plots/prophet_components.png"
            )
            plt.close()
            
            # Evaluate on test set
            test_pred = prophet.predict(periods=len(test), include_history=False)
            test_actual = test['revenue'].values
            test_predicted = test_pred['yhat'].values[:len(test_actual)]
            metrics = prophet.evaluate(test_actual, test_predicted)
            
            # Detect anomalies
            anomalies = prophet.detect_anomalies()
            plt.savefig(f"{self.output_path}/plots/prophet_anomalies.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Extract forecast for comparison
            future_forecast = forecast[forecast['ds'] > train.index.max()].head(forecast_days)
            forecast_df = pd.DataFrame({
                'forecast': future_forecast['yhat'].values,
                'lower_bound': future_forecast['yhat_lower'].values,
                'upper_bound': future_forecast['yhat_upper'].values
            }, index=future_forecast['ds'])
            
            # Store results
            self.results['Prophet'] = {
                'model': prophet,
                'forecast': forecast_df,
                'metrics': metrics,
                'parameters': prophet.get_model_parameters(),
                'anomalies': len(anomalies)
            }
            
            logger.info(f"Prophet completed - Test RMSE: {metrics['RMSE']:.2f}")
            
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {e}")
            self.results['Prophet'] = {'error': str(e)}
    
    def compare_models(self):
        """Compare all model results"""
        try:
            logger.info("\n" + "="*70)
            logger.info("STEP 5: Model Comparison")
            logger.info("="*70)
            
            # Collect metrics
            comparison_data = []
            for model_name, result in self.results.items():
                if 'error' not in result and 'metrics' in result:
                    metrics = result['metrics']
                    comparison_data.append({
                        'Model': model_name,
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'MAPE': metrics['MAPE']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Metrics comparison
            ax1 = axes[0, 0]
            comparison_df.set_index('Model')[['MAE', 'RMSE']].plot(kind='bar', ax=ax1)
            ax1.set_title('Error Metrics Comparison')
            ax1.set_ylabel('Error')
            ax1.legend()
            
            # Plot 2: MAPE comparison
            ax2 = axes[0, 1]
            comparison_df.set_index('Model')['MAPE'].plot(kind='bar', ax=ax2, color='green')
            ax2.set_title('MAPE Comparison')
            ax2.set_ylabel('MAPE (%)')
            
            # Plot 3: Forecast comparison
            ax3 = axes[1, 0]
            historical = self.daily_sales['revenue'].iloc[-60:]
            ax3.plot(historical.index, historical.values, label='Historical', color='black', linewidth=2)
            
            colors = ['blue', 'red', 'green']
            for i, (model_name, result) in enumerate(self.results.items()):
                if 'error' not in result and 'forecast' in result:
                    forecast = result['forecast']
                    ax3.plot(forecast.index, forecast['forecast'], 
                            label=model_name, color=colors[i], alpha=0.7)
            
            ax3.set_title('Forecast Comparison')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Revenue')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Forecast ranges
            ax4 = axes[1, 1]
            for i, (model_name, result) in enumerate(self.results.items()):
                if 'error' not in result and 'forecast' in result:
                    forecast = result['forecast']
                    forecast_range = forecast['upper_bound'] - forecast['lower_bound']
                    ax4.plot(forecast.index, forecast_range, 
                            label=f"{model_name} uncertainty", color=colors[i])
            
            ax4.set_title('Forecast Uncertainty Ranges')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Prediction Interval Width')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_path}/plots/model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save comparison results
            comparison_df.to_csv(f"{self.output_path}/results/model_comparison.csv", index=False)
            
            # Log results
            logger.info("\nModel Comparison Results:")
            logger.info(comparison_df.to_string())
            
            # Find best model
            best_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
            logger.info(f"\nBest model (lowest RMSE): {best_model}")
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            raise CustomException("Model comparison failed", e)
    
    def save_forecasts(self):
        """Save all forecasts to CSV"""
        try:
            logger.info("\nSaving forecast results...")
            
            # Save individual forecasts
            for model_name, result in self.results.items():
                if 'error' not in result and 'forecast' in result:
                    forecast = result['forecast']
                    forecast.to_csv(
                        f"{self.output_path}/results/{model_name.lower()}_forecast.csv"
                    )
            
            # Save combined forecasts
            combined_forecasts = pd.DataFrame()
            for model_name, result in self.results.items():
                if 'error' not in result and 'forecast' in result:
                    forecast = result['forecast']['forecast']
                    combined_forecasts[model_name] = forecast
            
            if not combined_forecasts.empty:
                combined_forecasts.to_csv(
                    f"{self.output_path}/results/combined_forecasts.csv"
                )
            
            # Save model parameters
            params_summary = {}
            for model_name, result in self.results.items():
                if 'error' not in result and 'parameters' in result:
                    params_summary[model_name] = result['parameters']
            
            with open(f"{self.output_path}/results/model_parameters.json", 'w') as f:
                json.dump(params_summary, f, indent=2, default=str)
            
            logger.info(f"Results saved to {self.output_path}/results/")
            
        except Exception as e:
            logger.error(f"Error saving forecasts: {e}")
    
    def generate_report(self):
        """Generate comprehensive forecasting report"""
        try:
            logger.info("\nGenerating forecasting report...")
            
            report_lines = [
                "# Sales Forecasting Report",
                f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "\n## Data Summary",
                f"- Total days: {len(self.daily_sales)}",
                f"- Date range: {self.daily_sales.index.min()} to {self.daily_sales.index.max()}",
                f"- Total revenue: ${self.daily_sales['revenue'].sum():,.2f}",
                f"- Average daily revenue: ${self.daily_sales['revenue'].mean():,.2f}",
                f"- Revenue std dev: ${self.daily_sales['revenue'].std():,.2f}",
                "\n## Model Results"
            ]
            
            # Add model-specific results
            for model_name, result in self.results.items():
                report_lines.append(f"\n### {model_name}")
                
                if 'error' in result:
                    report_lines.append(f"- Error: {result['error']}")
                else:
                    metrics = result.get('metrics', {})
                    report_lines.append(f"- RMSE: {metrics.get('RMSE', 'N/A'):.2f}")
                    report_lines.append(f"- MAE: {metrics.get('MAE', 'N/A'):.2f}")
                    report_lines.append(f"- MAPE: {metrics.get('MAPE', 'N/A'):.2f}%")
                    
                    # Add model-specific info
                    if model_name == 'SARIMA' and 'seasonality_strength' in result:
                        report_lines.append(f"- Seasonality strength: {result['seasonality_strength']:.4f}")
                    elif model_name == 'Prophet' and 'anomalies' in result:
                        report_lines.append(f"- Anomalies detected: {result['anomalies']}")
            
            # Save report
            report_content = '\n'.join(report_lines)
            with open(f"{self.output_path}/results/forecasting_report.md", 'w') as f:
                f.write(report_content)
            
            logger.info("Report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def run(self, forecast_days=30):
        """Run complete forecasting pipeline"""
        try:
            start_time = datetime.now()
            
            logger.info("="*70)
            logger.info("STARTING FORECASTING PIPELINE")
            logger.info("="*70)
            
            # Step 1: Prepare data
            self.prepare_data()
            
            # Step 2: Run ARIMA
            self.run_arima_forecast(forecast_days)
            
            # Step 3: Run SARIMA
            self.run_sarima_forecast(forecast_days)
            
            # Step 4: Run Prophet
            self.run_prophet_forecast(forecast_days)
            
            # Step 5: Compare models
            self.compare_models()
            
            # Step 6: Save results
            self.save_forecasts()
            
            # Step 7: Generate report
            self.generate_report()
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() / 60
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total execution time: {execution_time:.2f} minutes")
            logger.info(f"Results saved to: {self.output_path}")
            logger.info("="*70)
            
            # Return results summary
            return {
                'status': 'success',
                'models_run': list(self.results.keys()),
                'forecast_days': forecast_days,
                'output_path': self.output_path,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise CustomException("Forecasting pipeline failed", e)

if __name__ == "__main__":
    # Run the forecasting pipeline
    pipeline = ForecastingPipeline()
    pipeline.run(forecast_days=30)