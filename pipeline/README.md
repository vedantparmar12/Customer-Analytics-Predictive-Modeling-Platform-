# E-commerce Analytics Pipelines

This directory contains three different pipelines for running analytics:

## 1. **Master Pipeline** (`master_pipeline.py`)
Runs ALL analytics components in sequence:

```bash
python pipeline/master_pipeline.py
```

### Components Included:
1. **Data Processing** - Clean and prepare data
2. **Exploratory Data Analysis (EDA)** - Generate visualizations and insights
3. **Churn Prediction Models** - Train ML models for customer churn
4. **Business Metrics** - Calculate KPIs (revenue, CLV, retention, etc.)
5. **Cohort Analysis** - Analyze customer cohorts and retention
6. **A/B Testing Framework** - Set up testing infrastructure
7. **Recommendation Engine** - Generate product recommendations
8. **NLP Analysis** - Analyze customer reviews and sentiment
9. **Forecasting Models** - Predict future revenue and trends

**Runtime:** ~10-15 minutes for complete execution

### Output Structure:
```
artifacts/
├── eda/                    # Data visualizations
├── models_final/           # Trained ML models
├── business_metrics/       # KPIs and dashboards
├── cohort_analysis/        # Retention matrices
├── ab_testing/            # Testing framework
├── recommendations/       # Product recommendations
├── nlp/                   # Review analysis
├── forecasting/           # Time series forecasts
└── executive_summary.txt  # Complete summary report
```

## 2. **Selective Pipeline** (`selective_pipeline.py`)
Run only specific components you need:

```bash
# Run specific components
python pipeline/selective_pipeline.py --components data models metrics

# Quick analysis (data + models + metrics only)
python pipeline/selective_pipeline.py --quick

# Run everything
python pipeline/selective_pipeline.py --components all
```

### Available Components:
- `data` - Data processing
- `eda` - Exploratory analysis
- `models` - ML model training
- `metrics` - Business metrics
- `cohort` - Cohort analysis
- `ab` - A/B testing setup
- `recommend` - Recommendations
- `nlp` - NLP analysis
- `forecast` - Forecasting

## 3. **Final Pipeline** (`final_pipeline.py`)
Lightweight pipeline for **churn prediction only**:

```bash
python pipeline/final_pipeline.py
```

This is optimized for:
- Quick model training (~2 minutes)
- Production-ready churn models
- ~85% accuracy with proper regularization
- No overfitting

## Which Pipeline to Use?

### Use **Master Pipeline** when:
- Setting up for the first time
- Need comprehensive analysis
- Generating reports for stakeholders
- Monthly/quarterly reviews

### Use **Selective Pipeline** when:
- Updating specific components
- Testing individual features
- Quick iterations during development
- Resource-constrained environments

### Use **Final Pipeline** when:
- Only need churn prediction
- Production model updates
- Quick model retraining
- CI/CD integration

## Performance Expectations

| Pipeline | Runtime | Memory | Components |
|----------|---------|--------|------------|
| Master | 10-15 min | 4-6 GB | All 9 components |
| Selective | 1-10 min | 2-4 GB | Selected only |
| Final | 2-3 min | 2 GB | Churn models only |

## Example Use Cases

### Weekly Business Review:
```bash
python pipeline/selective_pipeline.py --components metrics cohort forecast
```

### New Customer Analysis:
```bash
python pipeline/selective_pipeline.py --components data eda nlp
```

### Model Retraining:
```bash
python pipeline/final_pipeline.py
```

### Complete Analysis:
```bash
python pipeline/master_pipeline.py
```

## Monitoring

All pipelines generate logs in `logs/` directory with detailed execution information.

## Error Handling

If a component fails:
- Pipeline continues with remaining components
- Failed components are marked in the summary
- Check logs for detailed error messages

## Scheduling

For automated execution, use cron (Linux/Mac) or Task Scheduler (Windows):

```bash
# Daily model retraining at 2 AM
0 2 * * * cd /path/to/project && python pipeline/final_pipeline.py

# Weekly full analysis on Sundays
0 3 * * 0 cd /path/to/project && python pipeline/master_pipeline.py
```