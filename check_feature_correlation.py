"""
Check feature correlations with churn to identify leakage
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Load data
processed_path = "artifacts/processed_final"
customer_features = pd.read_csv(os.path.join(processed_path, "customer_features_final.csv"))
feature_columns = joblib.load(os.path.join(processed_path, "feature_columns.pkl"))

# Calculate correlations with churn
correlations = customer_features[feature_columns + ['churned']].corr()['churned'].drop('churned')
correlations = correlations.sort_values(ascending=False)

print("Feature Correlations with Churn:")
print("=" * 50)
for feature, corr in correlations.items():
    print(f"{feature:30s}: {corr:8.4f}")

print("\n\nHighly Correlated Features (|corr| > 0.5):")
print("=" * 50)
high_corr = correlations[abs(correlations) > 0.5]
for feature, corr in high_corr.items():
    print(f"{feature:30s}: {corr:8.4f}")

# Check engagement score components
print("\n\nEngagement Score Analysis:")
print("=" * 50)
engagement_cols = ['frequency_score', 'value_score', 'consistency_score', 'engagement_score', 'churned']
print(customer_features[engagement_cols].corr()['churned'])

# Check if there's a clear separation
print("\n\nChurn Distribution by Engagement Score:")
print("=" * 50)
for score in range(1, 11):
    mask = (customer_features['engagement_score'] >= score - 0.5) & (customer_features['engagement_score'] < score + 0.5)
    if mask.sum() > 0:
        churn_rate = customer_features.loc[mask, 'churned'].mean()
        count = mask.sum()
        print(f"Score {score:2d}: {churn_rate:6.3f} churn rate ({count:5d} customers)")