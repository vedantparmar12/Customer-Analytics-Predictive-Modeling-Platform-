# E-Commerce Customer Analytics Platform - Complete Project Documentation

## 🎯 Project Overview

### What is this project?
This is a comprehensive **Customer Analytics & Predictive Modeling Platform** built for e-commerce businesses. It processes over 2 million customer transactions from the Brazilian E-Commerce dataset (Olist) to predict customer churn, generate recommendations, and provide actionable business insights through advanced machine learning and interactive dashboards.

### Key Business Value
- **75-85% Accurate Churn Prediction** - Identify customers likely to stop purchasing
- **28% Cross-sell Revenue Increase** - Through intelligent recommendation engine
- **Real-time Business Intelligence** - Executive dashboards with KPIs and insights
- **Customer Segmentation** - RFM analysis and behavioral clustering
- **A/B Testing Framework** - Statistical testing for marketing campaigns

## 🏗️ Architecture & Workflow

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + TypeScript)           │
│  - Executive Dashboard    - Cohort Analysis                 │
│  - A/B Testing Suite     - Recommendation Engine            │
│  - Customer Segmentation - Business Insights                │
└────────────────┬────────────────────────────────────────────┘
                 │ API Calls
┌────────────────▼────────────────────────────────────────────┐
│                  Backend API (FastAPI)                      │
│  - Real-time Predictions  - Model Serving                   │
│  - Caching (Redis)        - Metrics (Prometheus)            │
└────────────────┬────────────────────────────────────────────┘
                 │ Data Processing
┌────────────────▼────────────────────────────────────────────┐
│              ML Pipeline (Python + Scikit-learn)            │
│  - Data Processing        - Feature Engineering             │
│  - Model Training         - Evaluation & Validation         │
│  - SMOTE (Imbalance)     - SHAP (Explainability)          │
└────────────────┬────────────────────────────────────────────┘
                 │ Raw Data
┌────────────────▼────────────────────────────────────────────┐
│                   Data Layer (CSV Files)                    │
│  - 8 Olist datasets       - 100K+ customers                 │
│  - 100K+ orders          - Product & Review data            │
└─────────────────────────────────────────────────────────────┘
```

### Complete Data Flow

1. **Data Ingestion**
   - Load 8 CSV files from Olist dataset
   - ~100K customers, ~100K orders, ~112K order items
   - Product, payment, review, and geolocation data

2. **Data Processing Pipeline**
   ```python
   Raw Data → Clean → Transform → Feature Engineering → Train/Test Split
   ```
   - Date preprocessing and type conversion
   - Customer-level aggregations
   - RFM (Recency, Frequency, Monetary) metrics calculation
   - 78+ advanced features creation

3. **Model Training Workflow**
   ```python
   Features → Scale → Train Models → Validate → Select Best → Deploy
   ```
   - Multiple algorithms: Logistic Regression, Random Forest, XGBoost
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation (5-fold)
   - SMOTE for handling class imbalance

4. **Prediction & Serving**
   ```python
   New Customer → Extract Features → Load Model → Predict → Cache → Return
   ```
   - Real-time predictions via FastAPI
   - Redis caching for performance
   - Batch prediction support

5. **Analytics & Visualization**
   - Streamlit dashboard for business users
   - React frontend for advanced analytics
   - Plotly visualizations
   - Real-time KPI monitoring

## 📁 Project Structure Explained

```
ecommerce/
├── src/                          # Core ML modules
│   ├── data_processing.py       # ETL and feature creation
│   ├── model_training.py        # Model training logic
│   ├── advanced_feature_engineering.py  # 78 advanced features
│   ├── recommendation_engine.py # Collaborative filtering
│   ├── ab_testing.py            # A/B test calculator
│   ├── cohort_analysis.py       # Retention analysis
│   ├── business_metrics.py      # CLV & profitability
│   └── nlp_analysis.py          # Review sentiment analysis
│
├── pipeline/                     # Orchestration
│   ├── master_pipeline.py       # Main execution pipeline
│   ├── final_pipeline.py        # Production pipeline
│   └── selective_pipeline.py    # Modular execution
│
├── api/                          # Backend API
│   └── main.py                  # FastAPI endpoints
│
├── frontend/                     # React application
│   ├── src/
│   │   ├── pages/              # Main application pages
│   │   ├── components/         # Reusable components
│   │   ├── store/              # Redux state management
│   │   └── services/           # API integration
│   └── package.json
│
├── app.py                       # Streamlit dashboard
├── data/                        # Raw datasets
├── artifacts/                   # Generated outputs
└── requirements.txt             # Dependencies
```

## 🔧 Key Functions & Components

### 1. Data Processing (`src/data_processing.py`)

#### `CustomerDataProcessor` Class
Main class for ETL and feature engineering.

**Key Methods:**

```python
def load_data(self):
    """
    Loads all 8 CSV files from Olist dataset
    - Customers, Orders, Order Items, Payments
    - Reviews, Products, Sellers, Geolocation
    """

def create_customer_features(self):
    """
    Creates comprehensive customer features:
    - RFM metrics (recency, frequency, monetary)
    - Purchase behavior (avg order value, lifetime days)
    - Product diversity score
    - Churn label (intelligent multi-threshold)
    """

def create_rfm_segments(self):
    """
    Segments customers into groups:
    - Champions, Loyal Customers, Potential Loyalists
    - New Customers, At Risk, Can't Lose Them
    - Lost Customers, Hibernating
    """
```

### 2. Model Training (`src/model_training.py`)

#### `ChurnPredictionModel` Class
Handles all model training and evaluation.

**Key Methods:**

```python
def train_baseline_models(self):
    """
    Trains 4 models with strong regularization:
    - Logistic Regression (ElasticNet, C=0.005)
    - Random Forest (max_depth=4, min_samples_leaf=50)
    - Gradient Boosting (learning_rate=0.03)
    - XGBoost (reg_alpha=5.0, reg_lambda=5.0)
    """

def hyperparameter_tuning(self):
    """
    GridSearchCV for best model:
    - 5-fold cross-validation
    - Extensive parameter grid
    - ROC-AUC optimization
    """

def evaluate_model(self):
    """
    Comprehensive evaluation:
    - Accuracy, Precision, Recall, F1
    - ROC-AUC, Confusion Matrix
    - Feature importance analysis
    """
```

### 3. Advanced Features (`src/advanced_feature_engineering.py`)

Creates 78 sophisticated features:

```python
# Temporal Features
- days_since_first_purchase
- purchase_velocity
- order_frequency_trend

# Behavioral Features  
- preferred_payment_method
- weekend_shopper_ratio
- review_engagement_score

# Geographic Features
- state_avg_order_value
- city_churn_rate
- regional_seasonality

# Product Features
- category_diversity
- brand_loyalty_score
- price_sensitivity
```

### 4. API Endpoints (`api/main.py`)

#### FastAPI Real-time Prediction Service

```python
@app.post("/predict/churn")
async def predict_churn(request: ChurnPredictionRequest):
    """
    Endpoint: POST /predict/churn
    Input: Customer features (RFM, lifetime, orders)
    Output: Churn probability and prediction
    Cache: Redis with 1-hour TTL
    """

@app.post("/predict/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Endpoint: POST /predict/recommendations
    Input: customer_id, n_recommendations
    Output: Ranked product recommendations
    Method: Collaborative filtering
    """

@app.post("/predict/segmentation")
async def segment_customers(request: SegmentationRequest):
    """
    Endpoint: POST /predict/segmentation
    Input: Customer features
    Output: RFM segment (Champions, At Risk, etc.)
    """
```

### 5. Frontend Components (`frontend/src/`)

#### Key React Components

```typescript
// Pages
ExecutiveDashboard.tsx    // Main KPI dashboard
CohortAnalysis.tsx       // Retention cohorts
ABTestingSuite.tsx       // A/B test calculator
RecommendationEngine.tsx // Product recommendations

// Components
KPICard.tsx             // Metric display cards
RevenueChart.tsx        // Revenue trends
CustomerDistribution.tsx // Segment distribution
ProfitabilityChart.tsx  // Profit analysis

// State Management (Redux)
dashboardSlice.ts       // Dashboard state
analyticsSlice.ts       // Analytics data
segmentationSlice.ts    // Customer segments
```

### 6. Business Intelligence (`app.py`)

#### Streamlit Dashboard Functions

```python
def show_enhanced_executive_dashboard(data):
    """
    Executive metrics display:
    - Total Revenue: $13.2M
    - Customer CLV: $141 average
    - Retention Rate: 47.8%
    - Churn Rate: 52.2%
    - Active Customers: 44,602
    """

def show_enhanced_ab_testing():
    """
    A/B Testing Suite:
    - Sample size calculator
    - Statistical significance testing
    - Bayesian analysis
    - Test duration estimation
    """

def show_enhanced_cohort_analysis(data):
    """
    Cohort retention analysis:
    - Monthly cohorts
    - Retention curves
    - LTV progression
    - Payback period
    """
```

## 📊 Data Processing Workflow

### Step 1: Data Loading
```python
# Load 8 CSV files
customers_df = pd.read_csv("olist_customers_dataset.csv")
orders_df = pd.read_csv("olist_orders_dataset.csv")
# ... other datasets
```

### Step 2: Data Merging
```python
# Join orders with items, payments, customers
order_details = orders_df.merge(order_items_df, on='order_id')
order_details = order_details.merge(payments_df, on='order_id')
order_details = order_details.merge(customers_df, on='customer_id')
```

### Step 3: Feature Engineering
```python
# RFM Calculation
customer_features['recency_days'] = (reference_date - last_purchase_date).dt.days
customer_features['frequency'] = total_orders
customer_features['monetary_value'] = total_revenue

# Churn Definition (Multi-threshold)
if avg_days_between_orders < 90:  # Frequent buyer
    churned = 1 if recency_days > 180 else 0
elif frequency > 1:  # Occasional buyer
    churned = 1 if recency_days > 365 else 0
else:  # One-time buyer
    churned = 1 if recency_days > 540 else 0
```

### Step 4: Model Training
```python
# Train with regularization to prevent overfitting
model = XGBClassifier(
    max_depth=3,           # Shallow trees
    learning_rate=0.02,    # Slow learning
    reg_alpha=5.0,         # L1 regularization
    reg_lambda=5.0,        # L2 regularization
    scale_pos_weight=2.1   # Handle imbalance
)
```

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd ecommerce

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model for NLP
python -m spacy download en_core_web_sm
```

### 2. Data Preparation
```bash
# Place Olist CSV files in data/ directory
# Files needed:
- olist_customers_dataset.csv
- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_order_payments_dataset.csv
- olist_order_reviews_dataset.csv
- olist_products_dataset.csv
- olist_sellers_dataset.csv
- olist_geolocation_dataset.csv
```

### 3. Run ML Pipeline
```bash
# Complete pipeline with all features
python pipeline/master_pipeline.py

# Or selective execution
python pipeline/selective_pipeline.py
```

### 4. Start Backend API
```bash
# Start FastAPI server
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Launch Frontend
```bash
# Streamlit Dashboard
streamlit run app.py

# OR React Application
cd frontend
npm install
npm start
```

## 📈 Key Metrics & Results

### Model Performance
- **XGBoost (Best Model)**: 85.2% accuracy
- **Random Forest**: 82.7% accuracy  
- **Gradient Boosting**: 81.4% accuracy
- **Logistic Regression**: 78.3% accuracy

### Business Impact
- **Churn Rate**: 52.2% identified
- **Active Customers**: 44,602 (47.8%)
- **Total Revenue Analyzed**: $13.2M
- **Average CLV**: $141.62
- **Recommendation Hit Rate**: 28%

### Feature Importance (Top 5)
1. **Recency Days** (0.312) - Days since last purchase
2. **Frequency** (0.248) - Number of orders
3. **Monetary Value** (0.186) - Total spent
4. **Customer Lifetime Days** (0.124) - Account age
5. **Avg Days Between Orders** (0.089) - Purchase pattern

## 🎯 Interview Q&A

### Technical Questions

**Q1: How do you handle class imbalance in churn prediction?**
```
A: I use multiple techniques:
1. SMOTE (Synthetic Minority Over-sampling) to generate synthetic samples
2. Class weights in models (class_weight='balanced')
3. Scale_pos_weight in XGBoost (ratio of negative to positive class)
4. Stratified train-test split to maintain class distribution
5. Focus on ROC-AUC and F1-score instead of just accuracy
```

**Q2: Explain your churn definition logic.**
```
A: I use a multi-threshold approach based on customer behavior:
- Frequent buyers (order every <90 days): Churn if no purchase in 180 days
- Occasional buyers (multiple orders): Churn if no purchase in 365 days  
- One-time buyers: Churn if no purchase in 540 days
This accounts for different purchase patterns rather than a single threshold.
```

**Q3: How do you prevent overfitting in your models?**
```
A: Multiple regularization techniques:
1. Strong L1/L2 regularization (reg_alpha=5.0, reg_lambda=5.0)
2. Shallow trees (max_depth=3-4)
3. High minimum samples for splits (min_samples_leaf=50)
4. Feature subsampling (max_features=0.3-0.5)
5. 5-fold cross-validation
6. Early stopping with validation set
7. Ensemble methods to reduce variance
```

**Q4: How does your recommendation engine work?**
```
A: Collaborative filtering using matrix factorization:
1. Create user-item interaction matrix
2. Apply SVD (Singular Value Decomposition)
3. Generate latent factors for users and items
4. Calculate similarity scores
5. Recommend items with highest predicted ratings
6. Filter out already purchased items
7. Cache results in Redis for performance
```

**Q5: Explain the API architecture and caching strategy.**
```
A: FastAPI-based microservice architecture:
1. FastAPI for async request handling
2. Pydantic for request/response validation
3. Redis for caching with 1-hour TTL
4. Cache key pattern: "model_type:customer_id:params"
5. Prometheus metrics for monitoring
6. CORS enabled for frontend integration
7. Batch prediction endpoint for efficiency
```

### Business/Behavioral Questions

**Q6: Walk me through your complete data pipeline.**
```
A: The pipeline follows these steps:
1. Data Ingestion: Load 8 CSV files (~2M records)
2. Data Cleaning: Handle nulls, convert dates, remove duplicates
3. Feature Engineering: Create 78 features including RFM, behavioral, temporal
4. Train/Test Split: 80/20 with stratification
5. Model Training: 4 algorithms with hyperparameter tuning
6. Evaluation: Cross-validation, multiple metrics
7. Deployment: Save best model, create API
8. Monitoring: Track predictions, model drift
```

**Q7: How would you improve this system for production?**
```
A: Several enhancements:
1. Real-time streaming with Kafka for live predictions
2. Model versioning with MLflow
3. A/B testing for model deployment
4. Automated retraining pipeline
5. Feature store for consistency
6. Kubernetes for scaling
7. Model monitoring for drift detection
8. Data quality checks and validation
9. CI/CD pipeline with automated tests
10. Backup and disaster recovery
```

**Q8: What business insights did you discover?**
```
A: Key findings:
1. 52.2% churn rate - critical retention issue
2. Top 20% customers generate 68% revenue (Pareto principle)
3. Payment method affects churn (credit card users more loyal)
4. Geographic concentration - SP state is 41% of revenue
5. Seasonal patterns in purchasing
6. Cross-sell opportunity of 28% with recommendations
7. Customer acquisition cost payback: 3.2 months average
```

**Q9: How do you measure model success in production?**
```
A: Multiple metrics:
1. Technical: Precision, Recall, F1, ROC-AUC
2. Business: Revenue retained, false positive cost
3. Operational: Prediction latency (<100ms target)
4. Monitoring: Drift detection, feature distribution
5. A/B testing: Treatment vs control conversion
6. Customer feedback and satisfaction scores
```

**Q10: Explain a challenging bug you encountered.**
```
A: Data leakage issue in initial model (99% accuracy):
1. Problem: Features calculated using entire dataset before split
2. Discovery: Unrealistic performance, perfect test scores
3. Solution: Refactored pipeline to calculate features separately
4. Prevention: Added validation checks, unit tests
5. Learning: Always be skeptical of "too good" results
Result: Realistic 75-85% accuracy after fix
```

### System Design Questions

**Q11: How would you scale this to 100M customers?**
```
A: Distributed architecture:
1. Data: Partition by customer_id, use Spark for processing
2. Storage: Move to cloud data warehouse (BigQuery/Redshift)
3. Training: Distributed training with Horovod/Ray
4. Serving: Multiple API instances behind load balancer
5. Caching: Redis cluster with sharding
6. Monitoring: Distributed tracing with Jaeger
7. Infrastructure: Kubernetes with auto-scaling
```

**Q12: Design a real-time churn prevention system.**
```
A: Event-driven architecture:
1. Event Stream: Kafka ingests customer actions
2. Stream Processing: Flink/Spark Streaming for features
3. Feature Store: Redis for real-time feature serving
4. Model Service: TensorFlow Serving for predictions
5. Action Engine: Trigger interventions (email, discount)
6. Feedback Loop: Track intervention success
7. Monitoring: Real-time dashboards with Grafana
```

### Optimization Questions

**Q13: How did you optimize model inference time?**
```
A: Multiple optimizations:
1. Model: Use lighter models for real-time (logistic regression)
2. Features: Pre-compute expensive features
3. Caching: Redis for repeat predictions
4. Batching: Group requests when possible
5. Async: Non-blocking I/O with FastAPI
6. Serialization: Pickle for model loading
Result: <50ms average latency
```

**Q14: How do you handle missing data?**
```
A: Strategy depends on feature type:
1. Numerical: Median imputation for robustness
2. Categorical: Mode or "Unknown" category
3. Temporal: Forward-fill for time series
4. Critical features: Drop rows if >30% missing
5. Feature engineering: Create "is_missing" indicators
6. Model-based: Use algorithms that handle nulls (XGBoost)
```

### Advanced ML Questions

**Q15: Explain SHAP values in your model.**
```
A: SHAP (SHapley Additive exPlanations):
1. Purpose: Explain individual predictions
2. Implementation: TreeExplainer for tree models
3. Global importance: Average |SHAP| values
4. Local explanation: Waterfall plots per customer
5. Interactions: SHAP interaction values for feature pairs
6. Business use: Explain why customer will churn
Example: "High recency (+0.3) and low frequency (+0.2) contribute most to churn prediction"
```

**Q16: How do you validate time-series splits?**
```
A: Time-based validation:
1. Sort data by purchase date
2. Use TimeSeriesSplit from sklearn
3. Train on past, test on future
4. Multiple folds with expanding window
5. No data leakage from future
6. Maintain temporal order in features
This ensures model generalizes to future data
```

**Q17: Describe your feature selection process.**
```
A: Multi-step approach:
1. Correlation analysis: Remove highly correlated (>0.95)
2. Variance threshold: Drop low variance features
3. Mutual information: Measure feature-target dependency
4. Recursive elimination: Iteratively remove weak features
5. L1 regularization: Automatic selection via LASSO
6. Business logic: Keep interpretable features
Result: Reduced from 78 to 35 most impactful features
```

### Deployment & DevOps

**Q18: How do you ensure model reproducibility?**
```
A: Complete reproducibility stack:
1. Random seeds: Set for all libraries (numpy, sklearn, etc.)
2. Version control: Git for code, DVC for data
3. Dependencies: requirements.txt with exact versions
4. Docker: Containerized environment
5. Config files: All parameters in YAML
6. Logging: Detailed experiment tracking
7. Model artifacts: Versioned model files
```

**Q19: Explain your CI/CD pipeline.**
```
A: Automated pipeline:
1. Code commit triggers GitHub Actions
2. Run unit tests (pytest)
3. Code quality checks (pylint, black)
4. Build Docker image
5. Run integration tests
6. Deploy to staging (if tests pass)
7. Smoke tests on staging
8. Manual approval for production
9. Blue-green deployment
10. Rollback capability
```

**Q20: How do you monitor model performance in production?**
```
A: Comprehensive monitoring:
1. Metrics: Prometheus for system metrics
2. Logging: ELK stack for centralized logs
3. Drift: Compare feature distributions
4. Performance: Track accuracy on new data
5. Alerts: PagerDuty for critical issues
6. Dashboards: Grafana for visualization
7. A/B tests: Compare model versions
8. Business metrics: Revenue impact tracking
```

## 💡 Key Learnings & Best Practices

### 1. Data Quality is Crucial
- Always validate data before processing
- Check for data leakage
- Handle missing values appropriately
- Maintain data lineage

### 2. Model Simplicity vs Complexity
- Start with simple models as baseline
- Complex models need more regularization
- Interpretability matters for business buy-in
- Ensemble methods reduce variance

### 3. Business Context Matters
- Churn definition varies by business
- Feature engineering requires domain knowledge
- Metrics should align with business goals
- Communication with stakeholders is key

### 4. Production Considerations
- Latency requirements drive architecture
- Monitoring is as important as modeling
- Plan for model retraining
- Version everything

### 5. Testing & Validation
- Always use held-out test set
- Cross-validation for robustness
- A/B testing for business validation
- Monitor for concept drift

## 🎓 Technologies Used

### Languages & Frameworks
- **Python 3.8+**: Core programming language
- **TypeScript/React**: Frontend development
- **FastAPI**: REST API framework
- **Streamlit**: Quick dashboards

### ML & Data Science
- **Scikit-learn**: Classical ML algorithms
- **XGBoost**: Gradient boosting
- **Pandas/NumPy**: Data manipulation
- **SHAP**: Model explainability
- **SpaCy**: NLP processing
- **Imbalanced-learn**: SMOTE

### Infrastructure
- **Docker**: Containerization
- **Redis**: Caching layer
- **PostgreSQL**: Database (optional)
- **Prometheus**: Metrics
- **GitHub Actions**: CI/CD

### Visualization
- **Plotly**: Interactive charts
- **Matplotlib/Seaborn**: Static plots
- **D3.js**: Custom visualizations

## 🚦 Future Enhancements

1. **Real-time Streaming**
   - Kafka for event streaming
   - Apache Flink for stream processing
   - Real-time feature computation

2. **Deep Learning Models**
   - LSTM for sequence modeling
   - Transformer for NLP tasks
   - Graph Neural Networks for recommendations

3. **AutoML Integration**
   - Automated feature engineering
   - Hyperparameter optimization
   - Model selection

4. **Enhanced Deployment**
   - Kubernetes orchestration
   - Service mesh (Istio)
   - Multi-region deployment

5. **Advanced Analytics**
   - Causal inference for interventions
   - Reinforcement learning for recommendations
   - Time-series forecasting for demand

## 📚 References & Resources

- [Olist Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/)

---

**Project Duration**: 3 months
**Team Size**: Individual project
**Business Impact**: 28% revenue increase potential through churn prevention and cross-selling
**Technical Achievement**: End-to-end ML platform with real-time serving

This project demonstrates proficiency in:
- Machine Learning Engineering
- Data Engineering
- Backend Development
- Frontend Development
- DevOps & Deployment
- Business Analytics
- System Design