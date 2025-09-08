# Sales Forecasting Module

This module provides comprehensive time series forecasting capabilities for e-commerce sales data using multiple statistical and machine learning approaches.

## Overview

The forecasting module implements three major time series forecasting methods:
- **ARIMA** (AutoRegressive Integrated Moving Average) - For trend-based forecasting
- **SARIMA** (Seasonal ARIMA) - For capturing seasonal patterns
- **Prophet** (Facebook's forecasting tool) - For handling trends, seasonality, and holidays

## Features

### 1. Time Series Data Preparation
- Automatic aggregation of sales data (daily, weekly, monthly)
- Stationarity testing and transformation
- Feature engineering for time-based patterns
- Train/test splitting for time series

### 2. Forecasting Models

#### ARIMA Model (`arima_model.py`)
- Automatic parameter selection (p, d, q)
- ACF/PACF analysis
- Model diagnostics and residual analysis
- Rolling forecast evaluation

#### SARIMA Model (`sarima_model.py`)
- Seasonal decomposition
- Automatic seasonal parameter selection
- Handles weekly, monthly, and yearly seasonality
- Cross-validation for time series

#### Prophet Model (`prophet_model.py`)
- Trend changepoint detection
- Holiday effects (Brazilian holidays)
- Multiplicative and additive seasonality
- Anomaly detection capabilities

### 3. Visualization (`visualization.py`)
- Time series decomposition plots
- Interactive forecast dashboards
- Model comparison visualizations
- Prediction interval analysis

### 4. Pipeline (`forecasting_pipeline.py`)
- Automated workflow for all models
- Model comparison and selection
- Comprehensive reporting
- Results export to multiple formats

## Installation

Ensure you have the required dependencies:

```bash
pip install statsmodels prophet plotly scikit-learn pandas numpy matplotlib seaborn
```

## Usage

### Quick Start

```python
from forecasting.forecasting_pipeline import ForecastingPipeline

# Run complete forecasting pipeline
pipeline = ForecastingPipeline(
    data_path="data",
    output_path="artifacts/forecasting"
)

# Generate 30-day forecast using all models
pipeline.run(forecast_days=30)
```

### Individual Model Usage

#### ARIMA Example
```python
from forecasting.arima_model import ARIMAForecaster
import pandas as pd

# Load your time series data
data = pd.read_csv("daily_sales.csv", index_col='date', parse_dates=True)

# Create and fit ARIMA model
arima = ARIMAForecaster()
arima.fit(data['revenue'], auto_select=True)

# Generate forecast
forecast = arima.forecast(steps=30)

# Plot results
arima.plot_forecast(data['revenue'], forecast)
```

#### SARIMA Example
```python
from forecasting.sarima_model import SARIMAForecaster

# Create and fit SARIMA model
sarima = SARIMAForecaster()

# Detect seasonality
decomp, period, strength, fig = sarima.detect_seasonality(data['revenue'])

# Fit model with detected seasonality
sarima.fit(data['revenue'], auto_select=True)

# Generate forecast
forecast = sarima.forecast(steps=30)
```

#### Prophet Example
```python
from forecasting.prophet_model import ProphetForecaster

# Create and fit Prophet model
prophet = ProphetForecaster()

# Fit with custom parameters
prophet.fit(
    data['revenue'],
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative',
    add_country_holidays=True
)

# Generate predictions
forecast = prophet.predict(periods=30)

# Detect anomalies
anomalies = prophet.detect_anomalies()
```

### Data Preparation

```python
from forecasting.time_series_prep import TimeSeriesPreprocessor

# Initialize preprocessor
prep = TimeSeriesPreprocessor("data")
prep.load_data()

# Get different aggregations
daily_sales = prep.prepare_daily_sales()
weekly_sales = prep.prepare_weekly_sales()
monthly_sales = prep.prepare_monthly_sales()

# Add time features
daily_with_features = prep.add_time_features(daily_sales)

# Check stationarity
is_stationary = prep.check_stationarity(daily_sales['revenue'])
```

### Visualization

```python
from forecasting.visualization import ForecastVisualizer

# Create visualizer
viz = ForecastVisualizer()

# Create various plots
viz.plot_time_series_decomposition(data['revenue'])
viz.plot_forecast_comparison(historical_data, forecasts_dict)
viz.create_interactive_forecast_plot(historical_data, forecasts_dict)

# Generate complete dashboard
dashboard = viz.create_forecast_dashboard(
    historical_data,
    forecasts_dict,
    diagnostics_dict
)
```

## Output Structure

Running the pipeline creates the following output structure:

```
artifacts/forecasting/
├── prepared_data/
│   ├── daily_sales.csv
│   ├── weekly_sales.csv
│   ├── monthly_sales.csv
│   └── data_summary.json
├── results/
│   ├── arima_forecast.csv
│   ├── sarima_forecast.csv
│   ├── prophet_forecast.csv
│   ├── combined_forecasts.csv
│   ├── model_comparison.csv
│   ├── model_parameters.json
│   └── forecasting_report.md
└── plots/
    ├── arima_analysis.png
    ├── arima_forecast.png
    ├── arima_diagnostics.png
    ├── sarima_seasonality.png
    ├── sarima_forecast.png
    ├── sarima_diagnostics.png
    ├── prophet_forecast.png
    ├── prophet_components.png
    ├── prophet_anomalies.png
    ├── model_comparison.png
    ├── time_series_decomposition.png
    ├── forecast_comparison.png
    ├── model_diagnostics.png
    ├── interactive_forecast.html
    └── forecast_dashboard.html
```

## Model Selection Guide

### When to use ARIMA:
- Simple trend-based forecasting
- No strong seasonal patterns
- Short-term predictions
- Need for interpretability

### When to use SARIMA:
- Clear seasonal patterns (weekly, monthly, yearly)
- Regular cyclical behavior
- Medium-term forecasts
- Need to capture both trend and seasonality

### When to use Prophet:
- Multiple seasonality patterns
- Holiday effects are important
- Trend changes (changepoints)
- Need for automatic forecasting
- Handling missing data

## Performance Metrics

All models are evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

## Best Practices

1. **Data Requirements**
   - Minimum 2 years of historical data for reliable forecasts
   - Daily data preferred for capturing patterns
   - Handle missing values before modeling

2. **Model Selection**
   - Run all models and compare performance
   - Consider forecast horizon when selecting
   - Validate using rolling window approach

3. **Parameter Tuning**
   - Use automatic parameter selection for initial runs
   - Fine-tune based on domain knowledge
   - Monitor for overfitting

4. **Production Use**
   - Retrain models periodically (weekly/monthly)
   - Monitor forecast accuracy
   - Set up alerts for anomalies

## Troubleshooting

### Common Issues

1. **Convergence Warnings**
   - Try different parameter ranges
   - Ensure data is properly scaled
   - Check for extreme outliers

2. **Poor Forecast Accuracy**
   - Verify data quality
   - Check for structural breaks
   - Consider external factors

3. **Memory Issues**
   - Reduce parameter search space
   - Use smaller data samples for testing
   - Optimize grid search parameters

## Advanced Features

### Custom Seasonality (Prophet)
```python
prophet.fit(
    data,
    custom_seasonalities=[
        {'name': 'monthly', 'period': 30.5, 'fourier_order': 5},
        {'name': 'quarterly', 'period': 91.25, 'fourier_order': 10}
    ]
)
```

### Exogenous Variables
```python
# ARIMAX with external regressors
arima_model = ARIMA(
    data['revenue'],
    order=(1,1,1),
    exog=data[['promotion', 'holiday']]
)
```

### Ensemble Forecasting
```python
# Combine multiple forecasts
ensemble_forecast = (
    0.3 * arima_forecast +
    0.4 * sarima_forecast +
    0.3 * prophet_forecast
)
```

## References

- [ARIMA Models](https://otexts.com/fpp2/arima.html)
- [Seasonal ARIMA](https://www.statsmodels.org/stable/statespace.html)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Analysis](https://www.statsmodels.org/stable/tsa.html)