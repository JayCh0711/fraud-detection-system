"""
Model Trainer Component for Fraud Detection System

Models Trained:
1. Logistic Regression (Baseline)
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost (Optional)
6. Ensemble (Voting/Stacking)

Features:
- Cross-validation with stratification
- Hyperparameter tuning (Optuna)
- Model comparison
- BFSI-specific metrics (focus on Recall)
- Feature importance analysis
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict
import pandas as pd
import numpy as np
import warnings
import json

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)

# XGBoost and LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Optuna for hyperparameter tuning
import optuna
from optuna.samplers import TPESampler

# Project imports
from src.entity.config_entity import ModelTrainerConfig, DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.logger import logger, log_model_metrics
from src.exception import ModelTrainerException
from src.utils.common import (
    write_json,
    read_json,
    save_object,
    load_object,
    create_directories,
    read_yaml
)
from src.constants import ARTIFACTS_DIR, TARGET_COLUMN, MODEL_CONFIG_FILE_PATH


class ModelTrainer:
    """
    Model Trainer Component for Fraud Detection System.
    
    Responsibilities:
    - Train multiple ML models
    - Perform cross-validation
    - Hyperparameter tuning
    - Model comparison and selection
    - Save best model
    """
    
    def __init__(
        self,
        config: ModelTrainerConfig = ModelTrainerConfig(),
        data_transformation_artifact: DataTransformationArtifact = None
    ):
        """
        Initialize Model Trainer component.
        
        Args:
            config: ModelTrainerConfig object
            data_transformation_artifact: Artifact from data transformation
        """
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        
        self.models: Dict[str, Any] = {}
        self.model_scores: Dict[str, Dict] = {}
        self.best_model = None
        self.best_model_name = ""
        self.training_report: Dict = {}
        
        # Load model config
        try:
            self.model_config = read_yaml(MODEL_CONFIG_FILE_PATH)
        except Exception:
            logger.warning("Could not load model config. Using defaults.")
            self.model_config = None
        
        logger.info(f"{'='*60}")
        logger.info("Initializing Model Trainer Component")
        logger.info(f"{'='*60}")
        
        # Determine effective n_jobs (can be overridden with MODEL_TRAINER_N_JOBS env var)
        try:
            env_n_jobs = os.getenv('MODEL_TRAINER_N_JOBS')
            if env_n_jobs is not None:
                self.n_jobs = int(env_n_jobs)
            else:
                self.n_jobs = int(self.model_config.get('n_jobs', 1) if self.model_config else 1)
        except Exception:
            logger.warning('Invalid MODEL_TRAINER_N_JOBS value; defaulting to 1')
            self.n_jobs = 1
        # Determine effective cv folds (can be overridden with MODEL_TRAINER_CV_FOLDS env var)
        try:
            env_cv = os.getenv('MODEL_TRAINER_CV_FOLDS')
            if env_cv is not None:
                self.cv_folds = int(env_cv)
            else:
                # Prefer config value, fallback to default in ModelTrainerConfig
                self.cv_folds = int(self.config.cv_folds)
        except Exception:
            logger.warning('Invalid MODEL_TRAINER_CV_FOLDS value; defaulting to 3')
            self.cv_folds = 3
        # Quick mode flag (reduces model complexity for fast iteration)
        qm = os.getenv('MODEL_TRAINER_QUICK_MODE')
        self.quick_mode = str(qm).lower() in ('1', 'true', 'yes') if qm is not None else False
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load transformed training and test data."""
        logger.info("Loading transformed data...")
        
        X_train = np.load(self.data_transformation_artifact.train_transformed_path)
        X_test = np.load(self.data_transformation_artifact.test_transformed_path)
        y_train = np.load(self.data_transformation_artifact.train_target_path)
        y_test = np.load(self.data_transformation_artifact.test_target_path)
        
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_test shape: {y_test.shape}")
        
        # Log class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Training class distribution: {dict(zip(unique, counts))}")
        
        return X_train, X_test, y_train, y_test
    
    # ==================== MODEL DEFINITIONS ====================
    
    def get_base_models(self) -> Dict[str, Any]:
        """
        Get dictionary of base models to train.
        
        Returns:
            Dictionary of model name to model object
        """
        logger.info("Initializing base models...")
        
        # Allow quick-mode to reduce model complexity for fast local iterations
        if self.quick_mode:
            rf_n = 20
            xgb_n = 50
            lgb_n = 50
            gb_n = 50
        else:
            rf_n = 200
            xgb_n = 200
            lgb_n = 200
            gb_n = 200

        models = {
            "LogisticRegression": LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=self.config.random_state,
                solver='lbfgs',
                n_jobs=self.n_jobs
            ),
            
            "RandomForest": RandomForestClassifier(
                n_estimators=rf_n,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=self.n_jobs
            ),
            
            "XGBoost": XGBClassifier(
                n_estimators=xgb_n,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,  # Already balanced via SMOTE
                random_state=self.config.random_state,
                eval_metric='auc',
                use_label_encoder=False,
                n_jobs=self.n_jobs
            ),
            
            "LightGBM": LGBMClassifier(
                n_estimators=lgb_n,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=31,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=self.n_jobs,
                verbose=-1
            ),
            
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=gb_n,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.config.random_state
            )
        }
        
        logger.info(f"Initialized {len(models)} base models")
        return models
    
    # ==================== CROSS-VALIDATION ====================
    
    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Perform stratified cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            model_name: Name of the model
        
        Returns:
            Dictionary of CV metrics
        """
        logger.info(f"Cross-validating {model_name}...")
        
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=self.n_jobs
        )
        
        # Calculate mean and std for each metric
        metrics = {
            'accuracy_mean': float(np.mean(cv_results['test_accuracy'])),
            'accuracy_std': float(np.std(cv_results['test_accuracy'])),
            'precision_mean': float(np.mean(cv_results['test_precision'])),
            'precision_std': float(np.std(cv_results['test_precision'])),
            'recall_mean': float(np.mean(cv_results['test_recall'])),
            'recall_std': float(np.std(cv_results['test_recall'])),
            'f1_mean': float(np.mean(cv_results['test_f1'])),
            'f1_std': float(np.std(cv_results['test_f1'])),
            'roc_auc_mean': float(np.mean(cv_results['test_roc_auc'])),
            'roc_auc_std': float(np.std(cv_results['test_roc_auc'])),
            'fit_time_mean': float(np.mean(cv_results['fit_time'])),
        }
        
        logger.info(f"  Recall: {metrics['recall_mean']:.4f} (+/- {metrics['recall_std']:.4f})")
        logger.info(f"  Precision: {metrics['precision_mean']:.4f} (+/- {metrics['precision_std']:.4f})")
        logger.info(f"  F1-Score: {metrics['f1_mean']:.4f} (+/- {metrics['f1_std']:.4f})")
        logger.info(f"  ROC-AUC: {metrics['roc_auc_mean']:.4f} (+/- {metrics['roc_auc_std']:.4f})")
        
        return metrics
    
    # ==================== MODEL TRAINING ====================
    
    def train_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str
    ) -> Any:
        """
        Train a single model.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
        
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        logger.info(f"  Training completed in {training_time:.2f} seconds")
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model on given dataset.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            dataset_name: Name of dataset for logging
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, y_pred_proba)),
            'pr_auc': float(average_precision_score(y, y_pred_proba))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # BFSI-specific metrics
        metrics['fraud_detection_rate'] = metrics['recall']  # Same as recall
        metrics['false_alarm_rate'] = float(cm[0, 1] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        return metrics
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Train and evaluate all base models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of model scores
        """
        logger.info(f"{'='*60}")
        logger.info("Training All Models")
        logger.info(f"{'='*60}")
        
        models = self.get_base_models()
        # Allow running a single model for quick iterations via env var MODEL_TRAINER_SINGLE_MODEL
        single_model = os.getenv('MODEL_TRAINER_SINGLE_MODEL')
        if single_model:
            single_model = single_model.strip()
            if single_model in models:
                logger.info(f"Running single model: {single_model}")
                models = {single_model: models[single_model]}
            else:
                logger.warning(f"Requested single model '{single_model}' not found. Proceeding with all models.")
        model_scores = {}
        
        for model_name, model in models.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"Model: {model_name}")
            logger.info(f"{'='*40}")
            
            try:
                # Cross-validation
                cv_metrics = self.cross_validate_model(model, X_train, y_train, model_name)
                
                # Train on full training data
                trained_model = self.train_model(model, X_train, y_train, model_name)
                
                # Evaluate on training data
                train_metrics = self.evaluate_model(trained_model, X_train, y_train, "Training")
                
                # Evaluate on test data
                test_metrics = self.evaluate_model(trained_model, X_test, y_test, "Test")
                
                # Store model and scores
                self.models[model_name] = trained_model
                model_scores[model_name] = {
                    'cv_metrics': cv_metrics,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }
                
                # Log test metrics
                logger.info(f"\nTest Set Performance:")
                logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {test_metrics['precision']:.4f}")
                logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
                logger.info(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
                logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.model_scores = model_scores
        return model_scores
    
    # ==================== HYPERPARAMETER TUNING ====================
    
    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50
    ) -> Tuple[XGBClassifier, Dict]:
        """
        Tune XGBoost using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_trials: Number of Optuna trials
        
        Returns:
            Tuple of (best model, best params)
        """
        logger.info(f"Tuning XGBoost with Optuna ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'random_state': self.config.random_state,
                'eval_metric': 'auc',
                'use_label_encoder': False,
                'n_jobs': self.n_jobs
            }
            
            model = XGBClassifier(**params)
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall', n_jobs=self.n_jobs)
            
            return scores.mean()
        
        # Create Optuna study
        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_params['random_state'] = self.config.random_state
        best_params['eval_metric'] = 'auc'
        best_params['use_label_encoder'] = False
        best_params['n_jobs'] = self.n_jobs
        
        logger.info(f"Best trial recall: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train model with best parameters
        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        return best_model, best_params
    
    def tune_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50
    ) -> Tuple[LGBMClassifier, Dict]:
        """
        Tune LightGBM using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            n_trials: Number of Optuna trials
        
        Returns:
            Tuple of (best model, best params)
        """
        logger.info(f"Tuning LightGBM with Optuna ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'class_weight': 'balanced',
                'random_state': self.config.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1
            }
            
            model = LGBMClassifier(**params)
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall', n_jobs=self.n_jobs)
            
            return scores.mean()
        
        # Create Optuna study
        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = self.config.random_state
        best_params['n_jobs'] = self.n_jobs
        best_params['verbose'] = -1
        
        logger.info(f"Best trial recall: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train model with best parameters
        best_model = LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        return best_model, best_params
    
    def perform_hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, str, Dict]:
        """
        Perform hyperparameter tuning for best models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
        
        Returns:
            Tuple of (best model, model name, best params)
        """
        logger.info(f"{'='*60}")
        logger.info("Hyperparameter Tuning")
        logger.info(f"{'='*60}")
        
        tuned_models = {}
        
        # Tune XGBoost
        try:
            xgb_model, xgb_params = self.tune_xgboost(
                X_train, y_train, 
                n_trials=self.config.n_trials
            )
            xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, "Test")
            tuned_models['XGBoost_Tuned'] = {
                'model': xgb_model,
                'params': xgb_params,
                'metrics': xgb_metrics
            }
            logger.info(f"XGBoost Tuned - Recall: {xgb_metrics['recall']:.4f}, F1: {xgb_metrics['f1_score']:.4f}")
        except Exception as e:
            logger.error(f"XGBoost tuning failed: {str(e)}")
        
        # Tune LightGBM
        try:
            lgbm_model, lgbm_params = self.tune_lightgbm(
                X_train, y_train,
                n_trials=self.config.n_trials
            )
            lgbm_metrics = self.evaluate_model(lgbm_model, X_test, y_test, "Test")
            tuned_models['LightGBM_Tuned'] = {
                'model': lgbm_model,
                'params': lgbm_params,
                'metrics': lgbm_metrics
            }
            logger.info(f"LightGBM Tuned - Recall: {lgbm_metrics['recall']:.4f}, F1: {lgbm_metrics['f1_score']:.4f}")
        except Exception as e:
            logger.error(f"LightGBM tuning failed: {str(e)}")
        
        # Select best tuned model based on recall (primary) and f1 (secondary)
        best_model = None
        best_model_name = ""
        best_params = {}
        best_recall = 0
        
        for model_name, model_info in tuned_models.items():
            recall = model_info['metrics']['recall']
            if recall > best_recall:
                best_recall = recall
                best_model = model_info['model']
                best_model_name = model_name
                best_params = model_info['params']
        
        # Update model scores with tuned models
        for model_name, model_info in tuned_models.items():
            self.models[model_name] = model_info['model']
            self.model_scores[model_name] = {
                'test_metrics': model_info['metrics'],
                'params': model_info['params']
            }
        
        return best_model, best_model_name, best_params
    
    # ==================== MODEL SELECTION ====================
    
    def select_best_model(self) -> Tuple[Any, str]:
        """
        Select the best model based on defined criteria.
        
        Returns:
            Tuple of (best model, model name)
        """
        logger.info(f"{'='*60}")
        logger.info("Selecting Best Model")
        logger.info(f"{'='*60}")
        
        best_model = None
        best_model_name = ""
        best_score = 0
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, scores in self.model_scores.items():
            test_metrics = scores.get('test_metrics', {})
            cv_metrics = scores.get('cv_metrics', {})
            
            recall = test_metrics.get('recall', 0)
            precision = test_metrics.get('precision', 0)
            f1 = test_metrics.get('f1_score', 0)
            roc_auc = test_metrics.get('roc_auc', 0)
            
            comparison_data.append({
                'Model': model_name,
                'Recall': recall,
                'Precision': precision,
                'F1_Score': f1,
                'ROC_AUC': roc_auc,
                'CV_Recall_Mean': cv_metrics.get('recall_mean', 0),
                'CV_Recall_Std': cv_metrics.get('recall_std', 0)
            })
            
            # Selection criteria: Primary = Recall, Secondary = F1
            # Also check minimum thresholds
            if (recall >= self.config.min_recall_threshold and 
                precision >= self.config.min_precision_threshold):
                
                # Score = weighted combination of recall and f1
                score = (0.6 * recall) + (0.4 * f1)
                
                if score > best_score:
                    best_score = score
                    best_model = self.models.get(model_name)
                    best_model_name = model_name
        
        # If no model meets thresholds, select by recall only
        if best_model is None:
            logger.warning("No model meets minimum thresholds. Selecting by recall only.")
            for model_name, scores in self.model_scores.items():
                recall = scores.get('test_metrics', {}).get('recall', 0)
                if recall > best_score:
                    best_score = recall
                    best_model = self.models.get(model_name)
                    best_model_name = model_name
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Recall', ascending=False)
        
        logger.info("\nModel Comparison (Sorted by Recall):")
        logger.info("\n" + comparison_df.to_string(index=False))
        
        logger.info(f"\n{'='*40}")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"{'='*40}")
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        
        return best_model, best_model_name
    
    # ==================== FEATURE IMPORTANCE ====================
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    # ==================== SAVE RESULTS ====================
    
    def save_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Save model comparison to CSV."""
        comparison_df.to_csv(self.config.model_comparison_path, index=False)
        logger.info(f"Model comparison saved: {self.config.model_comparison_path}")
    
    def create_training_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        training_duration: float
    ) -> Dict:
        """
        Create comprehensive training report.
        
        Args:
            X_test: Test features
            y_test: Test target
            training_duration: Total training time
        
        Returns:
            Training report dictionary
        """
        best_metrics = self.evaluate_model(self.best_model, X_test, y_test, "Test")
        
        report = {
            'training_timestamp': pd.Timestamp.now().isoformat(),
            'training_duration_seconds': training_duration,
            
            'best_model': {
                'name': self.best_model_name,
                'params': self.model_scores.get(self.best_model_name, {}).get('params', {}),
                'test_metrics': best_metrics
            },
            
            'model_comparison': {
                model_name: {
                    'test_metrics': scores.get('test_metrics', {}),
                    'cv_metrics': scores.get('cv_metrics', {})
                }
                for model_name, scores in self.model_scores.items()
            },
            
            'configuration': {
                'cv_folds': self.config.cv_folds,
                'primary_metric': self.config.primary_metric,
                'min_recall_threshold': self.config.min_recall_threshold,
                'hyperparameter_tuning': self.config.enable_hyperparameter_tuning,
                'n_trials': self.config.n_trials
            },
            
            'classification_report': classification_report(
                y_test,
                self.best_model.predict(X_test),
                output_dict=True
            )
        }
        
        return report
    
    # ==================== MAIN EXECUTION ====================
    
    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Execute complete model training pipeline.
        
        Returns:
            ModelTrainerArtifact: Training results
        """
        logger.info(f"{'='*60}")
        logger.info("Starting Model Training Pipeline")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Step 1: Create directories
            logger.info("Step 1: Creating directories")
            create_directories([self.config.model_trainer_dir])
            
            # Step 2: Load data
            logger.info("Step 2: Loading transformed data")
            X_train, X_test, y_train, y_test = self._load_data()
            
            # Step 3: Optionally sample training data for quicker iteration
            logger.info("Step 3: Training base models")
            sample_frac_env = os.getenv('MODEL_TRAINER_SAMPLE_FRAC')
            if sample_frac_env is not None:
                try:
                    frac = float(sample_frac_env)
                    if 0 < frac < 1:
                        logger.info(f"Sampling training data to {frac*100:.1f}% for faster iteration")
                        rng = np.random.RandomState(self.config.random_state)
                        n_samples = max(1000, int(len(X_train) * frac))
                        idx = rng.choice(len(X_train), size=n_samples, replace=False)
                        X_train_sample = X_train[idx]
                        y_train_sample = y_train[idx]
                        self.train_all_models(X_train_sample, y_train_sample, X_test, y_test)
                    else:
                        self.train_all_models(X_train, y_train, X_test, y_test)
                except Exception:
                    logger.warning('Invalid MODEL_TRAINER_SAMPLE_FRAC; proceeding with full training set')
                    self.train_all_models(X_train, y_train, X_test, y_test)
            else:
                self.train_all_models(X_train, y_train, X_test, y_test)
            
            # Step 4: Hyperparameter tuning (if enabled)
            if self.config.enable_hyperparameter_tuning:
                logger.info("Step 4: Hyperparameter tuning")
                self.perform_hyperparameter_tuning(X_train, y_train, X_test, y_test)
            
            # Step 5: Select best model
            logger.info("Step 5: Selecting best model")
            best_model, best_model_name = self.select_best_model()
            
            # Step 6: Get best model metrics
            logger.info("Step 6: Evaluating best model")
            best_test_metrics = self.evaluate_model(best_model, X_test, y_test, "Test")
            best_cv_metrics = self.model_scores.get(best_model_name, {}).get('cv_metrics', {})
            
            # Step 7: Save best model
            logger.info("Step 7: Saving best model")
            save_object(self.config.model_path, best_model)
            
            # Step 8: Save model comparison
            logger.info("Step 8: Saving model comparison")
            comparison_data = []
            for model_name, scores in self.model_scores.items():
                test_metrics = scores.get('test_metrics', {})
                comparison_data.append({
                    'Model': model_name,
                    'Recall': test_metrics.get('recall', 0),
                    'Precision': test_metrics.get('precision', 0),
                    'F1_Score': test_metrics.get('f1_score', 0),
                    'ROC_AUC': test_metrics.get('roc_auc', 0),
                    'Accuracy': test_metrics.get('accuracy', 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Recall', ascending=False)
            self.save_model_comparison(comparison_df)
            
            # Step 9: Create and save training report
            logger.info("Step 9: Creating training report")
            training_duration = time.time() - start_time
            training_report = self.create_training_report(X_test, y_test, training_duration)
            write_json(self.config.training_report_path, training_report)
            
            # Step 10: Save best model params
            logger.info("Step 10: Saving best model parameters")
            best_params = self.model_scores.get(best_model_name, {}).get('params', {})
            if not best_params and hasattr(best_model, 'get_params'):
                best_params = best_model.get_params()
            
            # Convert params to JSON serializable
            serializable_params = {}
            for k, v in best_params.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable_params[k] = v
                else:
                    serializable_params[k] = str(v)
            
            write_json(self.config.best_params_path, serializable_params)
            
            # Create artifact
            artifact = ModelTrainerArtifact(
                model_path=self.config.model_path,
                training_report_path=self.config.training_report_path,
                model_comparison_path=self.config.model_comparison_path,
                best_model_name=best_model_name,
                best_model_params=serializable_params,
                train_accuracy=best_test_metrics.get('accuracy', 0),
                train_precision=best_test_metrics.get('precision', 0),
                train_recall=best_test_metrics.get('recall', 0),
                train_f1_score=best_test_metrics.get('f1_score', 0),
                train_roc_auc=best_test_metrics.get('roc_auc', 0),
                cv_recall_mean=best_cv_metrics.get('recall_mean', 0),
                cv_recall_std=best_cv_metrics.get('recall_std', 0),
                cv_precision_mean=best_cv_metrics.get('precision_mean', 0),
                cv_precision_std=best_cv_metrics.get('precision_std', 0),
                cv_f1_mean=best_cv_metrics.get('f1_mean', 0),
                cv_f1_std=best_cv_metrics.get('f1_std', 0),
                cv_roc_auc_mean=best_cv_metrics.get('roc_auc_mean', 0),
                cv_roc_auc_std=best_cv_metrics.get('roc_auc_std', 0),
                models_trained=list(self.models.keys()),
                model_scores=self.model_scores,
                training_duration_seconds=training_duration,
                hyperparameter_tuning_enabled=self.config.enable_hyperparameter_tuning
            )
            
            logger.info(f"{'='*60}")
            logger.info("MODEL TRAINING COMPLETED!")
            logger.info(f"{'='*60}")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Test Recall: {best_test_metrics['recall']:.4f}")
            logger.info(f"Test Precision: {best_test_metrics['precision']:.4f}")
            logger.info(f"Test F1-Score: {best_test_metrics['f1_score']:.4f}")
            logger.info(f"Test ROC-AUC: {best_test_metrics['roc_auc']:.4f}")
            logger.info(f"Training Duration: {training_duration:.2f} seconds")
            
            return artifact
            
        except Exception as e:
            raise ModelTrainerException(
                error_message=f"Model Training failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig,
        DataValidationConfig,
        FeatureEngineeringConfig,
        DataTransformationConfig
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.feature_engineering import FeatureEngineering
    from src.components.data_transformation import DataTransformation
    
    # Run previous pipeline steps
    print("\n" + "="*60)
    print("RUNNING DATA INGESTION")
    print("="*60)
    ingestion_config = DataIngestionConfig(
        source_type="csv",
        source_path="data/raw/transactions.csv"
    )
    ingestion = DataIngestion(config=ingestion_config)
    ingestion_artifact = ingestion.initiate_data_ingestion()
    
    print("\n" + "="*60)
    print("RUNNING DATA VALIDATION")
    print("="*60)
    validation_config = DataValidationConfig()
    validation = DataValidation(
        config=validation_config,
        data_ingestion_artifact=ingestion_artifact
    )
    validation_artifact = validation.initiate_data_validation()
    
    print("\n" + "="*60)
    print("RUNNING FEATURE ENGINEERING")
    print("="*60)
    fe_config = FeatureEngineeringConfig()
    feature_engineering = FeatureEngineering(
        config=fe_config,
        data_ingestion_artifact=ingestion_artifact,
        data_validation_artifact=validation_artifact
    )
    fe_artifact = feature_engineering.initiate_feature_engineering()
    
    print("\n" + "="*60)
    print("RUNNING DATA TRANSFORMATION")
    print("="*60)
    transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(
        config=transformation_config,
        feature_engineering_artifact=fe_artifact
    )
    transformation_artifact = data_transformation.initiate_data_transformation()
    
    # Run Model Training
    print("\n" + "="*60)
    print("RUNNING MODEL TRAINING")
    print("="*60)
    trainer_config = ModelTrainerConfig(
        enable_hyperparameter_tuning=True,
        n_trials=30  # Reduced for faster testing
    )
    model_trainer = ModelTrainer(
        config=trainer_config,
        data_transformation_artifact=transformation_artifact
    )
    
    trainer_artifact = model_trainer.initiate_model_training()
    
    print("\n" + "="*60)
    print("MODEL TRAINING ARTIFACT SUMMARY")
    print("="*60)
    print(f"Model Path: {trainer_artifact.model_path}")
    print(f"Best Model: {trainer_artifact.best_model_name}")
    print(f"\nTest Set Performance:")
    print(f"  Recall:    {trainer_artifact.train_recall:.4f}")
    print(f"  Precision: {trainer_artifact.train_precision:.4f}")
    print(f"  F1-Score:  {trainer_artifact.train_f1_score:.4f}")
    print(f"  ROC-AUC:   {trainer_artifact.train_roc_auc:.4f}")
    print(f"\nModels Trained: {trainer_artifact.models_trained}")
    print(f"Training Duration: {trainer_artifact.training_duration_seconds:.2f} seconds")