import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from .logger import get_logger
from .custom_exception import CustomException
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

logger = get_logger(__name__)

class ChurnPredictionModel:
    def __init__(self, processed_data_path, output_path="artifacts/models"):
        self.processed_data_path = processed_data_path
        self.output_path = output_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.model_scores = {}
        
        os.makedirs(self.output_path, exist_ok=True)
        logger.info("Churn Prediction Model Training initialized")
    
    def load_data(self):
        """Load processed training data"""
        try:
            logger.info("Loading processed data...")
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            
            logger.info(f"Loaded training data: {self.X_train.shape}")
            class_dist = self.y_train.value_counts().to_dict()
            logger.info(f"Class distribution - Train: {class_dist}")
            
            # Calculate and log class imbalance ratio
            if 0 in class_dist and 1 in class_dist:
                imbalance_ratio = class_dist[0] / class_dist[1] if class_dist[1] > 0 else float('inf')
                logger.info(f"Class imbalance ratio (non-churned:churned): {imbalance_ratio:.2f}:1")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load processed data", e)
    
    def train_baseline_models(self):
        """Train multiple baseline models"""
        try:
            logger.info("Training baseline models...")
            
            # Calculate scale_pos_weight for XGBoost
            class_dist = self.y_train.value_counts().to_dict()
            scale_pos_weight = 1.0
            if 0 in class_dist and 1 in class_dist and class_dist[1] > 0:
                scale_pos_weight = class_dist[0] / class_dist[1]
            
            models = {
                'Logistic Regression': LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    penalty='l2',  # L2 regularization
                    C=0.1,  # Stronger regularization (lower C = more regularization)
                    solver='liblinear',
                    class_weight='balanced'  # Handle class imbalance
                ),
                'Random Forest': RandomForestClassifier(
                    random_state=42, 
                    n_estimators=100,
                    max_depth=10,  # Limit tree depth
                    min_samples_split=20,  # Require more samples to split
                    min_samples_leaf=10,  # Require more samples in leaf nodes
                    max_features='sqrt',  # Use only sqrt of features at each split
                    class_weight='balanced'  # Handle class imbalance
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    random_state=42, 
                    n_estimators=100,
                    max_depth=5,  # Limit tree depth
                    learning_rate=0.1,
                    subsample=0.8,  # Use only 80% of data for each tree
                    min_samples_split=20,
                    min_samples_leaf=10
                ),
                'XGBoost': xgb.XGBClassifier(
                    random_state=42, 
                    n_estimators=100, 
                    use_label_encoder=False, 
                    eval_metric='logloss',
                    max_depth=6,  # Limit tree depth
                    learning_rate=0.1,
                    subsample=0.8,  # Row subsampling
                    colsample_bytree=0.8,  # Column subsampling
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    min_child_weight=5  # Minimum sum of instance weight in child
                )
            }
            
            for name, model in models.items():
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1_score': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                }
                
                self.model_scores[name] = metrics
                logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
                
                # Save model
                joblib.dump(model, os.path.join(self.output_path, f"{name.replace(' ', '_').lower()}_model.pkl"))
            
        except Exception as e:
            logger.error(f"Error training baseline models: {e}")
            raise CustomException("Failed to train baseline models", e)
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best performing model"""
        try:
            logger.info("Starting hyperparameter tuning...")
            
            # XGBoost parameter grid with regularization
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8],
                'colsample_bytree': [0.6, 0.8],
                'reg_alpha': [0.1, 0.5, 1.0],  # L1 regularization
                'reg_lambda': [1.0, 2.0, 3.0],  # L2 regularization
                'min_child_weight': [3, 5, 7]  # Minimum sum of instance weight
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss',
                early_stopping_rounds=10  # Stop if no improvement for 10 rounds
            )
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            # Use early stopping to prevent overfitting
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
            )
            
            grid_search.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Best model
            self.best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Evaluate on test set
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
            
            best_metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            self.model_scores['XGBoost_Tuned'] = best_metrics
            logger.info(f"Tuned Model - Accuracy: {best_metrics['accuracy']:.4f}, ROC-AUC: {best_metrics['roc_auc']:.4f}")
            
            # Save best model
            joblib.dump(self.best_model, os.path.join(self.output_path, "best_churn_model.pkl"))
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise CustomException("Failed to perform hyperparameter tuning", e)
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        try:
            logger.info("Generating model performance report...")
            
            # Save model scores
            scores_df = pd.DataFrame(self.model_scores).T
            scores_df.to_csv(os.path.join(self.output_path, "model_scores.csv"))
            
            # Plot model comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            scores_df.plot(kind='bar', ax=ax)
            ax.set_title('Model Performance Comparison', fontsize=16)
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.legend(loc='lower right')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'model_comparison.png'), dpi=300)
            plt.close()
            
            # Confusion matrix for best model
            y_pred = self.best_model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix - Best Model', fontsize=16)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'confusion_matrix.png'), dpi=300)
            plt.close()
            
            # Feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                feature_columns = joblib.load(os.path.join(self.processed_data_path, "feature_columns.pkl"))
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                feature_importance.head(15).plot(kind='barh', x='feature', y='importance', ax=ax)
                ax.set_title('Top 15 Feature Importance', fontsize=16)
                ax.set_xlabel('Importance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, 'feature_importance.png'), dpi=300)
                plt.close()
                
                feature_importance.to_csv(os.path.join(self.output_path, 'feature_importance.csv'), index=False)
            
            # ROC Curve
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(self.y_test, y_pred_proba):.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve - Best Model')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'roc_curve.png'), dpi=300)
            plt.close()
            
            # Classification report
            y_pred = self.best_model.predict(self.X_test)
            report = classification_report(self.y_test, y_pred)
            with open(os.path.join(self.output_path, 'classification_report.txt'), 'w') as f:
                f.write(report)
            
            logger.info("Model report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating model report: {e}")
            raise CustomException("Failed to generate model report", e)
    
    def run(self):
        """Run complete model training pipeline"""
        self.load_data()
        self.train_baseline_models()
        self.hyperparameter_tuning()
        self.generate_model_report()
        logger.info("Model training pipeline completed successfully")

if __name__ == "__main__":
    trainer = ChurnPredictionModel("artifacts/processed")
    trainer.run()