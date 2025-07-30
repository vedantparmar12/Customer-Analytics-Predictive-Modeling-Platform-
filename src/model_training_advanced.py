import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from .logger import get_logger
from .custom_exception import CustomException
import matplotlib.pyplot as plt
import seaborn as sns

logger = get_logger(__name__)

class AdvancedChurnModel:
    def __init__(self, processed_data_path, output_path="artifacts/models"):
        self.processed_data_path = processed_data_path
        self.output_path = output_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.model_scores = {}
        self.feature_columns = None
        
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "shap_plots"), exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("customer_churn_prediction")
        
        logger.info("Advanced Churn Model Training initialized")
    
    def load_data(self):
        """Load processed training data"""
        try:
            logger.info("Loading processed data...")
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            self.feature_columns = joblib.load(os.path.join(self.processed_data_path, "feature_columns.pkl"))
            
            logger.info(f"Loaded training data: {self.X_train.shape}")
            logger.info(f"Class distribution - Train: {np.bincount(self.y_train)}")
            logger.info(f"Class imbalance ratio: {np.sum(self.y_train==0) / np.sum(self.y_train==1):.2f}:1")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load processed data", e)
    
    def train_with_smote(self):
        """Train models with SMOTE for handling class imbalance"""
        try:
            logger.info("Training models with SMOTE...")
            
            # Initialize SMOTE
            smote = SMOTE(random_state=42, sampling_strategy=0.7)
            
            models = {
                'Logistic_Regression_SMOTE': LogisticRegression(
                    random_state=42, 
                    max_iter=1000, 
                    class_weight='balanced',
                    penalty='l2',
                    C=0.01,  # Strong regularization
                    solver='liblinear'
                ),
                'Random_Forest_SMOTE': RandomForestClassifier(
                    random_state=42, 
                    n_estimators=100,  # Reduced trees
                    class_weight='balanced',
                    max_depth=5,  # Limit depth
                    min_samples_split=50,  # Require more samples to split
                    min_samples_leaf=20,  # Require more samples in leaves
                    max_features='sqrt',  # Use sqrt features
                    max_samples=0.8  # Bootstrap on 80% of data
                ),
                'XGBoost_SMOTE': xgb.XGBClassifier(
                    random_state=42, 
                    n_estimators=100,  # Reduced trees
                    scale_pos_weight=3,
                    use_label_encoder=False, 
                    eval_metric='logloss',
                    max_depth=4,  # Shallow trees
                    learning_rate=0.05,  # Slower learning
                    subsample=0.7,  # Use 70% of data
                    colsample_bytree=0.7,  # Use 70% of features
                    reg_alpha=1.0,  # L1 regularization
                    reg_lambda=3.0,  # Strong L2 regularization
                    min_child_weight=10,  # Minimum weight in child
                    gamma=0.1  # Minimum loss reduction
                )
            }
            
            for name, model in models.items():
                with mlflow.start_run(run_name=name, nested=True):
                    logger.info(f"Training {name}...")
                    
                    # Create pipeline with SMOTE
                    pipeline = ImbPipeline([
                        ('smote', smote),
                        ('classifier', model)
                    ])
                    
                    # Train model
                    pipeline.fit(self.X_train, self.y_train)
                    
                    # Predictions
                    y_pred = pipeline.predict(self.X_test)
                    y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(self.y_test, y_pred),
                        'precision': precision_score(self.y_test, y_pred),
                        'recall': recall_score(self.y_test, y_pred),
                        'f1_score': f1_score(self.y_test, y_pred),
                        'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                    }
                    
                    # Log to MLflow
                    mlflow.log_params(model.get_params())
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Log model
                    if 'XGBoost' in name:
                        mlflow.xgboost.log_model(pipeline.named_steps['classifier'], "model")
                    else:
                        mlflow.sklearn.log_model(pipeline, "model")
                    
                    self.model_scores[name] = metrics
                    logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
                    
                    # Save pipeline
                    joblib.dump(pipeline, os.path.join(self.output_path, f"{name.lower()}_pipeline.pkl"))
            
        except Exception as e:
            logger.error(f"Error training with SMOTE: {e}")
            raise CustomException("Failed to train with SMOTE", e)
    
    def hyperparameter_tuning_with_cv(self):
        """Advanced hyperparameter tuning with cross-validation"""
        try:
            logger.info("Starting advanced hyperparameter tuning...")
            
            # XGBoost with regularization-focused parameter grid
            param_grid = {
                'classifier__n_estimators': [50, 100],  # Fewer trees
                'classifier__max_depth': [3, 4, 5],  # Shallow trees only
                'classifier__learning_rate': [0.01, 0.05],  # Slower learning
                'classifier__subsample': [0.6, 0.7],  # Less data per tree
                'classifier__colsample_bytree': [0.6, 0.7],  # Less features per tree
                'classifier__min_child_weight': [5, 10, 15],  # Higher minimum
                'classifier__gamma': [0.1, 0.3, 0.5],  # Higher gamma
                'classifier__reg_alpha': [0.5, 1.0, 2.0],  # L1 regularization
                'classifier__reg_lambda': [2.0, 3.0, 5.0]  # Strong L2 regularization
            }
            
            # Create pipeline with SMOTE
            smote = SMOTE(random_state=42, sampling_strategy=0.5)  # Less aggressive oversampling
            xgb_model = xgb.XGBClassifier(
                random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss', 
                scale_pos_weight=3,
                early_stopping_rounds=10  # Early stopping
            )
            
            pipeline = ImbPipeline([
                ('smote', smote),
                ('classifier', xgb_model)
            ])
            
            # Stratified K-Fold for better validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning", nested=True):
                # Split training data for validation (for early stopping)
                from sklearn.model_selection import train_test_split
                X_tr, X_val, y_tr, y_val = train_test_split(
                    self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
                )
                
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=3,  # Reduced folds for faster tuning
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                # Fit with validation set for early stopping
                fit_params = {
                    'classifier__eval_set': [(X_val, y_val)],
                    'classifier__verbose': False
                }
                
                grid_search.fit(X_tr, y_tr, **fit_params)
                
                # Best model
                self.best_model = grid_search.best_estimator_
                
                # Log best parameters
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_cv_score", grid_search.best_score_)
                
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
                
                for metric_name, metric_value in best_metrics.items():
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
                
                mlflow.xgboost.log_model(self.best_model.named_steps['classifier'], "best_model")
                
                self.model_scores['XGBoost_Tuned_SMOTE'] = best_metrics
                logger.info(f"Best Model - Accuracy: {best_metrics['accuracy']:.4f}, ROC-AUC: {best_metrics['roc_auc']:.4f}")
                
                # Save best model
                joblib.dump(self.best_model, os.path.join(self.output_path, "best_churn_model_smote.pkl"))
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            raise CustomException("Failed to perform hyperparameter tuning", e)
    
    def calibrate_probabilities(self):
        """Calibrate model probabilities for better reliability"""
        try:
            logger.info("Calibrating model probabilities...")
            
            # Get the classifier from pipeline
            classifier = self.best_model.named_steps['classifier']
            
            # Calibrate using sigmoid method
            calibrated_clf = CalibratedClassifierCV(
                classifier, method='sigmoid', cv=3
            )
            
            # Fit on the transformed training data
            X_train_resampled, y_train_resampled = self.best_model.named_steps['smote'].fit_resample(
                self.X_train, self.y_train
            )
            calibrated_clf.fit(X_train_resampled, y_train_resampled)
            
            # Save calibrated model
            joblib.dump(calibrated_clf, os.path.join(self.output_path, "calibrated_churn_model.pkl"))
            
            logger.info("Model calibration completed")
            
        except Exception as e:
            logger.error(f"Error calibrating probabilities: {e}")
            raise CustomException("Failed to calibrate probabilities", e)
    
    def explain_with_shap(self):
        """Generate SHAP explanations for model interpretability"""
        try:
            logger.info("Generating SHAP explanations...")
            
            # Get the classifier from pipeline
            classifier = self.best_model.named_steps['classifier']
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(classifier)
            
            # Calculate SHAP values for test set
            shap_values = explainer.shap_values(self.X_test)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_test, 
                            feature_names=self.feature_columns, 
                            show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "shap_plots", "summary_plot.png"), dpi=300)
            plt.close()
            
            # Feature importance bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, self.X_test, 
                            feature_names=self.feature_columns, 
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "shap_plots", "feature_importance_shap.png"), dpi=300)
            plt.close()
            
            # Dependence plots for top features
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-4:]
            
            for idx in top_features_idx:
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(idx, shap_values, self.X_test,
                                   feature_names=self.feature_columns,
                                   show=False)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_path, "shap_plots", 
                               f"dependence_{self.feature_columns[idx]}.png"), 
                    dpi=300
                )
                plt.close()
            
            # Save SHAP values for later use
            np.save(os.path.join(self.output_path, "shap_values.npy"), shap_values)
            
            # Create waterfall plot for a sample prediction
            sample_idx = 0
            shap_sample = shap_values[sample_idx]
            
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(values=shap_sample,
                               base_values=explainer.expected_value,
                               data=self.X_test[sample_idx],
                               feature_names=self.feature_columns),
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "shap_plots", "waterfall_sample.png"), dpi=300)
            plt.close()
            
            logger.info("SHAP explanations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            raise CustomException("Failed to generate SHAP explanations", e)
    
    def generate_advanced_report(self):
        """Generate comprehensive model performance report"""
        try:
            logger.info("Generating advanced model report...")
            
            # Model comparison with all metrics
            scores_df = pd.DataFrame(self.model_scores).T
            scores_df['profit_score'] = (
                scores_df['precision'] * 100 - 
                (1 - scores_df['recall']) * 500  # Cost of false negatives is higher
            )
            
            scores_df.to_csv(os.path.join(self.output_path, "model_scores_advanced.csv"))
            
            # Plot advanced metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # ROC-AUC comparison
            axes[0, 0].bar(scores_df.index, scores_df['roc_auc'])
            axes[0, 0].set_title('ROC-AUC Score Comparison')
            axes[0, 0].set_xticklabels(scores_df.index, rotation=45, ha='right')
            axes[0, 0].set_ylim(0.8, 1.0)
            
            # Precision-Recall trade-off
            axes[0, 1].scatter(scores_df['recall'], scores_df['precision'], s=100)
            for idx, model in enumerate(scores_df.index):
                axes[0, 1].annotate(model, (scores_df['recall'].iloc[idx], scores_df['precision'].iloc[idx]))
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Trade-off')
            
            # F1 Score comparison
            axes[1, 0].bar(scores_df.index, scores_df['f1_score'])
            axes[1, 0].set_title('F1 Score Comparison')
            axes[1, 0].set_xticklabels(scores_df.index, rotation=45, ha='right')
            
            # Business profit score
            axes[1, 1].bar(scores_df.index, scores_df['profit_score'])
            axes[1, 1].set_title('Business Profit Score (Custom Metric)')
            axes[1, 1].set_xticklabels(scores_df.index, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Profit Score')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'advanced_model_comparison.png'), dpi=300)
            plt.close()
            
            # Generate detailed report
            with open(os.path.join(self.output_path, 'model_report_advanced.txt'), 'w') as f:
                f.write("Advanced Customer Churn Prediction Model Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Model Performance Summary:\n")
                f.write("-" * 30 + "\n")
                for model, scores in self.model_scores.items():
                    f.write(f"\n{model}:\n")
                    for metric, value in scores.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                
                f.write("\nBusiness Impact Analysis:\n")
                f.write("-" * 30 + "\n")
                best_model_name = scores_df['profit_score'].idxmax()
                f.write(f"Recommended Model: {best_model_name}\n")
                f.write(f"Expected Profit Score: {scores_df.loc[best_model_name, 'profit_score']:.2f}\n")
                
                # Calculate potential savings
                total_customers = len(self.y_test)
                churned_customers = np.sum(self.y_test)
                best_recall = scores_df.loc[best_model_name, 'recall']
                
                f.write(f"\nPotential Business Value:\n")
                f.write(f"- Total Test Customers: {total_customers}\n")
                f.write(f"- Actual Churned: {churned_customers}\n")
                f.write(f"- Correctly Identified: {int(churned_customers * best_recall)}\n")
                f.write(f"- Potential Retention Rate: {best_recall:.1%}\n")
                
            logger.info("Advanced report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating advanced report: {e}")
            raise CustomException("Failed to generate advanced report", e)
    
    def run(self):
        """Run complete advanced model training pipeline"""
        self.load_data()
        self.train_with_smote()
        self.hyperparameter_tuning_with_cv()
        self.calibrate_probabilities()
        self.explain_with_shap()
        self.generate_advanced_report()
        
        logger.info("Advanced model training pipeline completed successfully")

if __name__ == "__main__":
    trainer = AdvancedChurnModel("artifacts/processed")
    trainer.run()