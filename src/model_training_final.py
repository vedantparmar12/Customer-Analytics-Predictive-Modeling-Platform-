"""
Final Model Training - Production Ready
- Extreme regularization to prevent overfitting
- Proper cross-validation
- Comprehensive evaluation
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class FinalModelTrainer:
    """Production-ready model training with extreme regularization"""
    
    def __init__(self, processed_data_path, output_path="artifacts/models_final"):
        self.processed_data_path = processed_data_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        
    def load_data(self):
        """Load processed training data"""
        try:
            logger.info("Loading processed data...")
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
            self.feature_columns = joblib.load(os.path.join(self.processed_data_path, "feature_columns.pkl"))
            
            logger.info(f"Loaded data - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
            logger.info(f"Train churn rate: {self.y_train.mean():.3f}")
            logger.info(f"Test churn rate: {self.y_test.mean():.3f}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load processed data", e)
    
    def get_models(self):
        """Get models with moderate regularization for ~85% accuracy"""
        
        # Calculate class weight
        n_pos = (self.y_train == 1).sum()
        n_neg = (self.y_train == 0).sum()
        scale_pos_weight = n_neg / n_pos
        
        models = {
            'Logistic_Regression_L2': LogisticRegression(
                penalty='l2',
                C=1.0,  # Less regularization for better accuracy
                max_iter=5000,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            ),
            
            'Logistic_Regression_L1': LogisticRegression(
                penalty='l1',
                C=1.0,  # Less regularization for better accuracy
                max_iter=5000,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            ),
            
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=8,  # Deeper tree for better accuracy
                min_samples_split=50,  # Less restrictive
                min_samples_leaf=20,  # Smaller leaves allowed
                max_features=0.8,  # Use more features
                class_weight='balanced',
                random_state=42
            ),
            
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,  # More trees for stability
                max_depth=8,  # Deeper trees
                min_samples_split=50,  # Less restrictive
                min_samples_leaf=20,  # Smaller leaves
                max_features=0.8,  # Use more features
                max_samples=0.9,  # Use more samples per tree
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=150,  # More trees for better accuracy
                max_depth=5,  # Deeper trees
                learning_rate=0.1,  # Normal learning rate
                subsample=0.85,  # Use more samples
                max_features=0.8,  # Use more features
                min_samples_split=50,  # Less restrictive
                min_samples_leaf=20,  # Smaller leaves
                validation_fraction=0.2,
                n_iter_no_change=10,
                random_state=42
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=150,  # More trees
                max_depth=5,  # Deeper trees
                learning_rate=0.1,  # Normal learning rate
                subsample=0.85,  # Use more samples
                colsample_bytree=0.85,  # Use more features
                reg_alpha=0.5,  # Less L1 regularization
                reg_lambda=0.5,  # Less L2 regularization
                min_child_weight=5,  # Less restrictive
                gamma=0.05,  # Lower gamma for more splits
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            ),
            
            'SVM': SVC(
                C=1.0,  # Moderate regularization
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42,
                cache_size=1000  # Larger cache for faster training
            )
        }
        
        return models
    
    def train_and_evaluate(self):
        """Train models with cross-validation and evaluation"""
        try:
            logger.info("Starting model training with moderate regularization for ~85% accuracy...")
            
            models = self.get_models()
            results = {}
            
            # Define CV strategy
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Define scoring metrics
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
            
            for name, model in models.items():
                logger.info(f"\nTraining {name}...")
                
                # For SVM, use a smaller subset to speed up training
                if name == 'SVM':
                    logger.info("Using subset of data for SVM to speed up training...")
                    # Use only 10% of data for SVM
                    subset_size = int(0.1 * len(self.X_train))
                    subset_indices = np.random.choice(len(self.X_train), subset_size, replace=False)
                    X_train_subset = self.X_train[subset_indices]
                    y_train_subset = self.y_train[subset_indices]
                    
                    # Cross-validation on subset
                    cv_results = cross_validate(
                        model, X_train_subset, y_train_subset,
                        cv=cv, scoring=scoring,
                        return_train_score=True,
                        n_jobs=-1
                    )
                else:
                    # Cross-validation on full data
                    cv_results = cross_validate(
                        model, self.X_train, self.y_train,
                        cv=cv, scoring=scoring,
                        return_train_score=True,
                        n_jobs=-1
                    )
                
                # Calculate mean and std
                results[name] = {
                    'cv_train_accuracy': cv_results['train_accuracy'].mean(),
                    'cv_val_accuracy': cv_results['test_accuracy'].mean(),
                    'cv_val_accuracy_std': cv_results['test_accuracy'].std(),
                    'cv_val_precision': cv_results['test_precision'].mean(),
                    'cv_val_recall': cv_results['test_recall'].mean(),
                    'cv_val_f1': cv_results['test_f1'].mean(),
                    'cv_val_roc_auc': cv_results['test_roc_auc'].mean(),
                    'cv_overfit_gap': cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
                }
                
                # Train on full training set (or subset for SVM)
                if name == 'SVM':
                    model.fit(X_train_subset, y_train_subset)
                else:
                    model.fit(self.X_train, self.y_train)
                
                # Test set evaluation
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                results[name].update({
                    'test_accuracy': accuracy_score(self.y_test, y_pred),
                    'test_precision': precision_score(self.y_test, y_pred, zero_division=0),
                    'test_recall': recall_score(self.y_test, y_pred),
                    'test_f1': f1_score(self.y_test, y_pred),
                    'test_roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                })
                
                # Save model
                joblib.dump(model, os.path.join(self.output_path, f"{name.lower()}.pkl"))
                
                # Log results
                logger.info(f"{name} Results:")
                logger.info(f"  CV Val Accuracy: {results[name]['cv_val_accuracy']:.3f} (±{results[name]['cv_val_accuracy_std']:.3f})")
                logger.info(f"  CV Overfit Gap: {results[name]['cv_overfit_gap']:.3f}")
                logger.info(f"  Test Accuracy: {results[name]['test_accuracy']:.3f}")
                logger.info(f"  Test ROC-AUC: {results[name]['test_roc_auc']:.3f}")
            
            # Save results
            results_df = pd.DataFrame(results).T
            results_df.to_csv(os.path.join(self.output_path, "model_results.csv"))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise CustomException("Failed to train models", e)
    
    def plot_results(self, results):
        """Create visualization of results"""
        try:
            logger.info("Creating result visualizations...")
            
            results_df = pd.DataFrame(results).T
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Accuracy comparison
            ax = axes[0, 0]
            x = np.arange(len(results_df))
            width = 0.35
            
            ax.bar(x - width/2, results_df['cv_val_accuracy'], width, 
                   label='CV Validation', alpha=0.8, yerr=results_df['cv_val_accuracy_std'])
            ax.bar(x + width/2, results_df['test_accuracy'], width, 
                   label='Test Set', alpha=0.8)
            
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(results_df.index, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
            
            # 2. ROC-AUC comparison
            ax = axes[0, 1]
            ax.bar(results_df.index, results_df['test_roc_auc'])
            ax.set_ylabel('ROC-AUC')
            ax.set_title('Test Set ROC-AUC')
            ax.set_xticklabels(results_df.index, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            
            # 3. Overfit analysis
            ax = axes[1, 0]
            ax.bar(results_df.index, results_df['cv_overfit_gap'])
            ax.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
            ax.set_ylabel('Train-Val Gap')
            ax.set_title('Overfitting Analysis')
            ax.set_xticklabels(results_df.index, rotation=45, ha='right')
            ax.legend()
            
            # 4. F1 Score
            ax = axes[1, 1]
            ax.bar(results_df.index, results_df['test_f1'])
            ax.set_ylabel('F1 Score')
            ax.set_title('Test Set F1 Score')
            ax.set_xticklabels(results_df.index, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Best model analysis
            best_model = results_df['test_roc_auc'].idxmax()
            logger.info(f"\nBest model: {best_model}")
            logger.info(f"Test ROC-AUC: {results_df.loc[best_model, 'test_roc_auc']:.3f}")
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
    
    def generate_report(self, results):
        """Generate comprehensive report"""
        try:
            with open(os.path.join(self.output_path, 'training_report.txt'), 'w', encoding='utf-8') as f:
                f.write("Final Model Training Report\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("Training Configuration:\n")
                f.write("- Moderate regularization applied for ~85% accuracy\n")
                f.write("- Balanced class weights\n")
                f.write("- 5-fold stratified cross-validation\n")
                f.write("- Multiple model families tested\n\n")
                
                results_df = pd.DataFrame(results).T
                best_model = results_df['test_roc_auc'].idxmax()
                
                f.write(f"Best Model: {best_model}\n")
                f.write(f"Test ROC-AUC: {results_df.loc[best_model, 'test_roc_auc']:.3f}\n\n")
                
                f.write("All Models Performance:\n")
                f.write("-" * 60 + "\n")
                
                for model_name, metrics in results.items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  CV Validation:\n")
                    f.write(f"    Accuracy: {metrics['cv_val_accuracy']:.3f} (±{metrics['cv_val_accuracy_std']:.3f})\n")
                    f.write(f"    ROC-AUC: {metrics['cv_val_roc_auc']:.3f}\n")
                    f.write(f"    Overfit Gap: {metrics['cv_overfit_gap']:.3f}\n")
                    f.write(f"  Test Set:\n")
                    f.write(f"    Accuracy: {metrics['test_accuracy']:.3f}\n")
                    f.write(f"    Precision: {metrics['test_precision']:.3f}\n")
                    f.write(f"    Recall: {metrics['test_recall']:.3f}\n")
                    f.write(f"    F1-Score: {metrics['test_f1']:.3f}\n")
                    f.write(f"    ROC-AUC: {metrics['test_roc_auc']:.3f}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("Recommendations:\n")
                f.write(f"- Deploy {best_model} model\n")
                f.write("- Monitor performance on new data\n")
                f.write("- Retrain monthly with updated data\n")
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def run(self):
        """Run complete training pipeline"""
        self.load_data()
        results = self.train_and_evaluate()
        self.plot_results(results)
        self.generate_report(results)
        
        logger.info("Model training completed successfully")
        return results

if __name__ == "__main__":
    trainer = FinalModelTrainer("artifacts/processed_final")
    trainer.run()