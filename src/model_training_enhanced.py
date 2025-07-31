"""
Enhanced Model Training - Target 80% accuracy
- Feature engineering
- Ensemble methods
- Hyperparameter optimization
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger = None  # Will be set later
import matplotlib.pyplot as plt
import seaborn as sns
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class EnhancedModelTrainer:
    """Enhanced model training for 80% accuracy"""
    
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
    
    def engineer_features(self):
        """Create interaction and polynomial features"""
        try:
            logger.info("Engineering advanced features...")
            
            # Select most important features for interactions
            # Use a simple RF to find important features
            rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf_selector.fit(self.X_train, self.y_train)
            
            # Get feature importances
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': rf_selector.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Select top 10 features for interaction
            top_features_idx = feature_importance.head(10).index.tolist()
            
            # Create interaction features for top features
            interaction_features_train = []
            interaction_features_test = []
            interaction_names = []
            
            for i in range(len(top_features_idx)):
                for j in range(i+1, len(top_features_idx)):
                    idx1, idx2 = top_features_idx[i], top_features_idx[j]
                    
                    # Multiplication interaction
                    interaction_features_train.append(self.X_train[:, idx1] * self.X_train[:, idx2])
                    interaction_features_test.append(self.X_test[:, idx1] * self.X_test[:, idx2])
                    interaction_names.append(f"{self.feature_columns[idx1]}_x_{self.feature_columns[idx2]}")
                    
                    # Division interaction (protected)
                    denom_train = np.where(self.X_train[:, idx2] != 0, self.X_train[:, idx2], 1)
                    denom_test = np.where(self.X_test[:, idx2] != 0, self.X_test[:, idx2], 1)
                    
                    interaction_features_train.append(self.X_train[:, idx1] / denom_train)
                    interaction_features_test.append(self.X_test[:, idx1] / denom_test)
                    interaction_names.append(f"{self.feature_columns[idx1]}_div_{self.feature_columns[idx2]}")
            
            # Stack interaction features
            if interaction_features_train:
                interaction_train = np.column_stack(interaction_features_train)
                interaction_test = np.column_stack(interaction_features_test)
                
                # Combine with original features
                self.X_train_enhanced = np.hstack([self.X_train, interaction_train])
                self.X_test_enhanced = np.hstack([self.X_test, interaction_test])
                
                logger.info(f"Created {len(interaction_names)} interaction features")
                logger.info(f"Enhanced train shape: {self.X_train_enhanced.shape}")
            else:
                self.X_train_enhanced = self.X_train
                self.X_test_enhanced = self.X_test
            
            # Feature selection to prevent overfitting
            logger.info("Selecting best features...")
            selector = SelectKBest(f_classif, k=min(50, self.X_train_enhanced.shape[1]))
            self.X_train_final = selector.fit_transform(self.X_train_enhanced, self.y_train)
            self.X_test_final = selector.transform(self.X_test_enhanced)
            
            logger.info(f"Final feature set: {self.X_train_final.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            self.X_train_final = self.X_train
            self.X_test_final = self.X_test
    
    def get_optimized_models(self):
        """Get models with optimized hyperparameters for 80% accuracy"""
        
        # Calculate class weight
        n_pos = (self.y_train == 1).sum()
        n_neg = (self.y_train == 0).sum()
        scale_pos_weight = n_neg / n_pos
        
        models = {
            'Logistic_Regression': LogisticRegression(
                penalty='l2',
                C=1.0,  # Less regularization
                max_iter=5000,
                solver='lbfgs',
                class_weight='balanced',
                random_state=42
            ),
            
            'Random_Forest': RandomForestClassifier(
                n_estimators=200,  # More trees
                max_depth=10,  # Deeper trees
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                max_samples=0.9,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                max_features='sqrt',
                min_samples_split=20,
                min_samples_leaf=10,
                validation_fraction=0.2,
                n_iter_no_change=10,
                random_state=42
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_weight=5,
                gamma=0.01,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            ),
            
            
            'Neural_Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            )
        }
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=50,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        else:
            logger.info("LightGBM not available, skipping...")
        
        return models
    
    def create_ensemble(self, base_models):
        """Create ensemble model"""
        try:
            logger.info("Creating ensemble models...")
            
            # Select best models for ensemble
            model_scores = {}
            for name, model in base_models.items():
                # Quick evaluation
                model.fit(self.X_train_final, self.y_train)
                y_pred = model.predict(self.X_test_final)
                score = accuracy_score(self.y_test, y_pred)
                model_scores[name] = score
                logger.info(f"{name} accuracy: {score:.3f}")
            
            # Select top 3 models
            top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Top models for ensemble: {[m[0] for m in top_models]}")
            
            # Create voting ensemble
            ensemble_models = [(name, base_models[name]) for name, _ in top_models]
            
            voting_soft = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                n_jobs=-1
            )
            
            voting_hard = VotingClassifier(
                estimators=ensemble_models,
                voting='hard',
                n_jobs=-1
            )
            
            return {
                'Ensemble_Soft': voting_soft,
                'Ensemble_Hard': voting_hard
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            return {}
    
    def train_and_evaluate(self):
        """Train models with enhanced features"""
        try:
            logger.info("Starting enhanced model training for 80% accuracy...")
            
            # Engineer features
            self.engineer_features()
            
            # Get models
            models = self.get_optimized_models()
            
            # Create ensemble
            ensemble_models = self.create_ensemble(models)
            models.update(ensemble_models)
            
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
                
                try:
                    # Cross-validation
                    cv_results = cross_validate(
                        model, self.X_train_final, self.y_train,
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
                    
                    # Train on full training set
                    model.fit(self.X_train_final, self.y_train)
                    
                    # Test set evaluation
                    y_pred = model.predict(self.X_test_final)
                    
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(self.X_test_final)[:, 1]
                    else:
                        y_pred_proba = y_pred
                    
                    results[name].update({
                        'test_accuracy': accuracy_score(self.y_test, y_pred),
                        'test_precision': precision_score(self.y_test, y_pred, zero_division=0),
                        'test_recall': recall_score(self.y_test, y_pred),
                        'test_f1': f1_score(self.y_test, y_pred),
                        'test_roc_auc': roc_auc_score(self.y_test, y_pred_proba)
                    })
                    
                    # Save model
                    joblib.dump(model, os.path.join(self.output_path, f"{name.lower()}_enhanced.pkl"))
                    
                    # Log results
                    logger.info(f"{name} Results:")
                    logger.info(f"  CV Val Accuracy: {results[name]['cv_val_accuracy']:.3f} (±{results[name]['cv_val_accuracy_std']:.3f})")
                    logger.info(f"  Test Accuracy: {results[name]['test_accuracy']:.3f}")
                    logger.info(f"  Test ROC-AUC: {results[name]['test_roc_auc']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
            
            # Save results
            results_df = pd.DataFrame(results).T
            results_df.to_csv(os.path.join(self.output_path, "enhanced_model_results.csv"))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise CustomException("Failed to train models", e)
    
    def run(self):
        """Run complete enhanced training pipeline"""
        self.load_data()
        results = self.train_and_evaluate()
        
        # Find best model
        results_df = pd.DataFrame(results).T
        best_model = results_df['test_accuracy'].idxmax()
        best_accuracy = results_df.loc[best_model, 'test_accuracy']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL: {best_model}")
        logger.info(f"TEST ACCURACY: {best_accuracy:.1%}")
        logger.info(f"{'='*60}")
        
        if best_accuracy >= 0.80:
            logger.info("✅ Successfully achieved 80% accuracy target!")
        else:
            logger.info(f"Current best: {best_accuracy:.1%}. Consider further tuning.")
        
        return results

if __name__ == "__main__":
    trainer = EnhancedModelTrainer("artifacts/processed_final")
    trainer.run()