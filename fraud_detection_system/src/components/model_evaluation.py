"""
Model Evaluation Component for Fraud Detection System

Evaluations Performed:
1. Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
2. Confusion Matrix Analysis
3. ROC Curve & AUC
4. Precision-Recall Curve & AUC
5. Threshold Optimization
6. Feature Importance Analysis
7. Lift & Gain Charts
8. Business Cost-Benefit Analysis
9. Model Acceptance Criteria Check
10. Comprehensive Visualization
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

# Project imports
from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)
from src.logger import logger
from src.exception import ModelEvaluationException
from src.utils.common import (
    write_json,
    read_json,
    save_object,
    load_object,
    create_directories
)
from src.constants import ARTIFACTS_DIR, TARGET_COLUMN


class ModelEvaluation:
    """
    Model Evaluation Component for Fraud Detection System.
    
    Responsibilities:
    - Evaluate model performance with multiple metrics
    - Generate visualizations (ROC, PR, Confusion Matrix)
    - Optimize classification threshold
    - Calculate business metrics
    - Check model acceptance criteria
    - Generate comprehensive evaluation report
    """
    
    def __init__(
        self,
        config: ModelEvaluationConfig = ModelEvaluationConfig(),
        data_transformation_artifact: DataTransformationArtifact = None,
        model_trainer_artifact: ModelTrainerArtifact = None
    ):
        """
        Initialize Model Evaluation component.
        
        Args:
            config: ModelEvaluationConfig object
            data_transformation_artifact: Artifact from data transformation
            model_trainer_artifact: Artifact from model training
        """
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        
        self.model = None
        self.evaluation_results: Dict = {}
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        logger.info(f"{'='*60}")
        logger.info("Initializing Model Evaluation Component")
        logger.info(f"{'='*60}")
    
    def _load_model_and_data(self) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Load trained model and test data."""
        logger.info("Loading model and test data...")
        
        # Load model
        self.model = load_object(self.model_trainer_artifact.model_path)
        logger.info(f"Model loaded: {self.model_trainer_artifact.best_model_name}")
        
        # Load test data
        X_test = np.load(self.data_transformation_artifact.test_transformed_path)
        y_test = np.load(self.data_transformation_artifact.test_target_path)
        
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Test target shape: {y_test.shape}")
        
        # Log class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        logger.info(f"Test class distribution: {dict(zip(unique.astype(int), counts))}")
        
        return self.model, X_test, y_test
    
    def _load_feature_names(self) -> List[str]:
        """Load feature names from transformation artifact."""
        try:
            feature_info = read_json(self.data_transformation_artifact.feature_names_path)
            return feature_info.get('selected_features', [])
        except Exception:
            logger.warning("Could not load feature names")
            return []
    
    # ==================== CLASSIFICATION METRICS ====================
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating classification metrics...")
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
            'pr_auc': float(average_precision_score(y_true, y_pred_proba))
        }
        
        # Confusion matrix values
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Derived metrics
        total_negatives = cm[0, 0] + cm[0, 1]
        total_positives = cm[1, 0] + cm[1, 1]
        
        metrics['false_alarm_rate'] = float(cm[0, 1] / total_negatives) if total_negatives > 0 else 0
        metrics['miss_rate'] = float(cm[1, 0] / total_positives) if total_positives > 0 else 0
        metrics['specificity'] = float(cm[0, 0] / total_negatives) if total_negatives > 0 else 0
        
        # BFSI-specific naming
        metrics['fraud_detection_rate'] = metrics['recall']
        metrics['fraud_precision'] = metrics['precision']
        
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        
        return metrics
    
    # ==================== THRESHOLD OPTIMIZATION ====================
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        optimization_metric: str = "f1"
    ) -> Tuple[float, Dict]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            optimization_metric: Metric to optimize (f1, recall, precision, youden)
        
        Returns:
            Tuple of (optimal threshold, threshold analysis dict)
        """
        logger.info(f"Finding optimal threshold (optimizing {optimization_metric})...")
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Youden's J statistic (sensitivity + specificity - 1)
            cm = confusion_matrix(y_true, y_pred)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            youden_j = recall + specificity - 1
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'youden_j': youden_j
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold based on metric
        if optimization_metric == "f1":
            optimal_idx = results_df['f1_score'].idxmax()
        elif optimization_metric == "recall":
            # Find threshold that gives recall >= min_recall with best precision
            valid_results = results_df[results_df['recall'] >= self.config.min_recall]
            if len(valid_results) > 0:
                optimal_idx = valid_results['precision'].idxmax()
            else:
                optimal_idx = results_df['recall'].idxmax()
        elif optimization_metric == "precision":
            optimal_idx = results_df['precision'].idxmax()
        elif optimization_metric == "youden":
            optimal_idx = results_df['youden_j'].idxmax()
        else:
            optimal_idx = results_df['f1_score'].idxmax()
        
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        
        logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
        logger.info(f"  At this threshold:")
        logger.info(f"    Precision: {results_df.loc[optimal_idx, 'precision']:.4f}")
        logger.info(f"    Recall: {results_df.loc[optimal_idx, 'recall']:.4f}")
        logger.info(f"    F1-Score: {results_df.loc[optimal_idx, 'f1_score']:.4f}")
        
        return float(optimal_threshold), results_df.to_dict('records')
    
    # ==================== BUSINESS METRICS ====================
    
    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        Calculate business-relevant metrics and cost-benefit analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
        
        Returns:
            Dictionary of business metrics
        """
        logger.info("Calculating business metrics...")
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_frauds = tp + fn
        total_legitimate = tn + fp
        
        # Fraud detection metrics
        frauds_caught = tp
        frauds_missed = fn
        false_alarms = fp
        
        # Financial calculations
        avg_fraud_amount = self.config.avg_fraud_amount
        investigation_cost = self.config.investigation_cost
        customer_friction_cost = self.config.customer_friction_cost
        
        # Money saved by catching frauds
        fraud_amount_caught = frauds_caught * avg_fraud_amount
        fraud_amount_missed = frauds_missed * avg_fraud_amount
        
        # Costs
        total_investigation_cost = (tp + fp) * investigation_cost
        total_friction_cost = fp * customer_friction_cost
        total_cost = total_investigation_cost + total_friction_cost
        
        # Net savings
        net_savings = fraud_amount_caught - total_cost
        
        # ROI
        roi = (net_savings / total_cost * 100) if total_cost > 0 else 0
        
        business_metrics = {
            # Detection Statistics
            'total_transactions': int(len(y_true)),
            'total_frauds': int(total_frauds),
            'total_legitimate': int(total_legitimate),
            'frauds_caught': int(frauds_caught),
            'frauds_missed': int(frauds_missed),
            'false_alarms': int(false_alarms),
            
            # Rates
            'fraud_detection_rate': float(tp / total_frauds) if total_frauds > 0 else 0,
            'false_alarm_rate': float(fp / total_legitimate) if total_legitimate > 0 else 0,
            'accuracy_on_frauds': float(tp / total_frauds) if total_frauds > 0 else 0,
            'accuracy_on_legitimate': float(tn / total_legitimate) if total_legitimate > 0 else 0,
            
            # Financial Impact
            'avg_fraud_amount': avg_fraud_amount,
            'fraud_amount_caught': fraud_amount_caught,
            'fraud_amount_missed': fraud_amount_missed,
            'potential_fraud_loss': fraud_amount_caught + fraud_amount_missed,
            'fraud_loss_prevented_pct': float(fraud_amount_caught / (fraud_amount_caught + fraud_amount_missed) * 100) if (fraud_amount_caught + fraud_amount_missed) > 0 else 0,
            
            # Costs
            'investigation_cost_per_case': investigation_cost,
            'total_investigation_cost': total_investigation_cost,
            'customer_friction_cost_per_case': customer_friction_cost,
            'total_friction_cost': total_friction_cost,
            'total_operational_cost': total_cost,
            
            # Net Impact
            'net_savings': net_savings,
            'roi_percentage': roi,
            
            # Per Day Estimates (assuming test set represents one month)
            'daily_frauds_caught': frauds_caught / 30,
            'daily_savings': net_savings / 30
        }
        
        logger.info(f"  Frauds Caught: {frauds_caught}/{total_frauds} ({business_metrics['fraud_detection_rate']*100:.1f}%)")
        logger.info(f"  False Alarms: {false_alarms} ({business_metrics['false_alarm_rate']*100:.2f}%)")
        logger.info(f"  Fraud Amount Caught: ${fraud_amount_caught:,.2f}")
        logger.info(f"  Net Savings: ${net_savings:,.2f}")
        logger.info(f"  ROI: {roi:.1f}%")
        
        return business_metrics
    
    # ==================== MODEL ACCEPTANCE CRITERIA ====================
    
    def check_model_acceptance(
        self,
        metrics: Dict
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if model meets acceptance criteria.
        
        Args:
            metrics: Dictionary of model metrics
        
        Returns:
            Tuple of (is_accepted, criteria_results)
        """
        logger.info("Checking model acceptance criteria...")
        
        criteria_results = {
            'min_recall': metrics['recall'] >= self.config.min_recall,
            'min_precision': metrics['precision'] >= self.config.min_precision,
            'min_f1_score': metrics['f1_score'] >= self.config.min_f1_score,
            'min_roc_auc': metrics['roc_auc'] >= self.config.min_roc_auc,
            'max_false_alarm_rate': metrics['false_alarm_rate'] <= self.config.max_false_alarm_rate
        }
        
        # Log each criterion
        logger.info(f"  Recall >= {self.config.min_recall}: {metrics['recall']:.4f} - {'✓ PASS' if criteria_results['min_recall'] else '✗ FAIL'}")
        logger.info(f"  Precision >= {self.config.min_precision}: {metrics['precision']:.4f} - {'✓ PASS' if criteria_results['min_precision'] else '✗ FAIL'}")
        logger.info(f"  F1-Score >= {self.config.min_f1_score}: {metrics['f1_score']:.4f} - {'✓ PASS' if criteria_results['min_f1_score'] else '✗ FAIL'}")
        logger.info(f"  ROC-AUC >= {self.config.min_roc_auc}: {metrics['roc_auc']:.4f} - {'✓ PASS' if criteria_results['min_roc_auc'] else '✗ FAIL'}")
        logger.info(f"  False Alarm Rate <= {self.config.max_false_alarm_rate}: {metrics['false_alarm_rate']:.4f} - {'✓ PASS' if criteria_results['max_false_alarm_rate'] else '✗ FAIL'}")
        
        # Model is accepted if all criteria are met
        is_accepted = all(criteria_results.values())
        
        logger.info(f"\n  MODEL {'ACCEPTED ✓' if is_accepted else 'REJECTED ✗'}")
        
        return is_accepted, criteria_results
    
    # ==================== VISUALIZATIONS ====================
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str
    ) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        logger.info("Generating confusion matrix plot...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute values
        ax1 = axes[0]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            ax=ax1,
            annot_kws={'size': 14}
        )
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('Actual', fontsize=12)
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        
        # Percentages
        ax2 = axes[1]
        cm_pct = cm.astype('float') / cm.sum() * 100
        sns.heatmap(
            cm_pct, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            ax=ax2,
            annot_kws={'size': 14}
        )
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('Actual', fontsize=12)
        ax2.set_title('Confusion Matrix (%)', fontsize=14, fontweight='bold')
        
        # Add summary text
        tn, fp, fn, tp = cm.ravel()
        summary_text = (
            f"True Positives (Fraud Caught): {tp}\n"
            f"True Negatives (Legit Correct): {tn}\n"
            f"False Positives (False Alarms): {fp}\n"
            f"False Negatives (Fraud Missed): {fn}"
        )
        fig.text(0.5, -0.05, summary_text, ha='center', fontsize=11, 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved: {save_path}")
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str
    ) -> None:
        """
        Plot and save ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save the plot
        """
        logger.info("Generating ROC curve plot...")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # ROC Curve
        ax.plot(fpr, tpr, color='#2ecc71', lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--',
                label='Random Classifier')
        
        # Find optimal point (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        ax.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                   color='#3498db', s=200, zorder=5,
                   label=f'Optimal Point (threshold={optimal_threshold:.2f})')
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title('ROC Curve - Fraud Detection Model', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add AUC annotation
        ax.annotate(f'AUC = {roc_auc:.4f}',
                    xy=(0.6, 0.2), fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved: {save_path}")
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str
    ) -> None:
        """
        Plot and save Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save the plot
        """
        logger.info("Generating Precision-Recall curve plot...")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # PR Curve
        ax.plot(recall, precision, color='#9b59b6', lw=3,
                label=f'PR Curve (AUC = {pr_auc:.4f})')
        
        # Baseline (proportion of positives)
        baseline = y_true.sum() / len(y_true)
        ax.axhline(y=baseline, color='#e74c3c', lw=2, linestyle='--',
                   label=f'Baseline (Fraud Rate = {baseline:.4f})')
        
        # Find optimal F1 point
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last point
        
        ax.scatter(recall[optimal_idx], precision[optimal_idx],
                   color='#3498db', s=200, zorder=5,
                   label=f'Optimal F1 Point (threshold={thresholds[optimal_idx]:.2f})')
        
        # Formatting
        ax.set_xlabel('Recall (Fraud Detection Rate)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Fraud Detection Model', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add PR-AUC annotation
        ax.annotate(f'PR-AUC = {pr_auc:.4f}',
                    xy=(0.1, 0.1), fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curve saved: {save_path}")
    
    def plot_threshold_analysis(
        self,
        threshold_results: List[Dict],
        optimal_threshold: float,
        save_path: str
    ) -> None:
        """
        Plot threshold analysis showing metrics at different thresholds.
        
        Args:
            threshold_results: List of threshold analysis results
            optimal_threshold: Optimal threshold value
            save_path: Path to save the plot
        """
        logger.info("Generating threshold analysis plot...")
        
        df = pd.DataFrame(threshold_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Precision, Recall, F1 vs Threshold
        ax1 = axes[0]
        ax1.plot(df['threshold'], df['precision'], 'b-', lw=2, label='Precision')
        ax1.plot(df['threshold'], df['recall'], 'g-', lw=2, label='Recall')
        ax1.plot(df['threshold'], df['f1_score'], 'r-', lw=2, label='F1-Score')
        ax1.axvline(x=optimal_threshold, color='purple', linestyle='--', lw=2,
                    label=f'Optimal Threshold ({optimal_threshold:.2f})')
        ax1.axhline(y=self.config.min_recall, color='g', linestyle=':', alpha=0.5,
                    label=f'Min Recall ({self.config.min_recall})')
        
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0.1, 0.9])
        ax1.set_ylim([0, 1])
        
        # Right plot: Precision vs Recall trade-off
        ax2 = axes[1]
        ax2.plot(df['recall'], df['precision'], 'b-', lw=2)
        
        # Mark optimal point
        optimal_idx = df[df['threshold'] == optimal_threshold].index[0]
        ax2.scatter(df.loc[optimal_idx, 'recall'], df.loc[optimal_idx, 'precision'],
                    color='red', s=200, zorder=5, label=f'Optimal (t={optimal_threshold:.2f})')
        
        # Mark default threshold (0.5)
        default_idx = df[df['threshold'].round(2) == 0.5].index
        if len(default_idx) > 0:
            ax2.scatter(df.loc[default_idx[0], 'recall'], df.loc[default_idx[0], 'precision'],
                        color='orange', s=150, zorder=5, marker='s', label='Default (t=0.5)')
        
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Threshold analysis saved: {save_path}")
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        save_path: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Plot feature importance.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            save_path: Path to save the plot
            top_n: Number of top features to show
        
        Returns:
            DataFrame with feature importance
        """
        logger.info("Generating feature importance plot...")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model doesn't have feature importance attribute")
            return pd.DataFrame()
        
        # Handle case where feature_names length doesn't match
        if len(feature_names) != len(importances):
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))[::-1]
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.invert_yaxis()
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - Fraud Detection', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, top_features['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance saved: {save_path}")
        
        return importance_df
    
    def plot_lift_chart(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str,
        n_bins: int = 10
    ) -> None:
        """
        Plot Lift and Cumulative Gains chart.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            save_path: Path to save the plot
            n_bins: Number of bins for lift calculation
        """
        logger.info("Generating lift chart...")
        
        # Create DataFrame with predictions
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        })
        
        # Sort by predicted probability (descending)
        df = df.sort_values('y_pred_proba', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative metrics
        df['cumulative_frauds'] = df['y_true'].cumsum()
        df['cumulative_total'] = range(1, len(df) + 1)
        df['cumulative_fraud_pct'] = df['cumulative_frauds'] / df['y_true'].sum()
        df['cumulative_total_pct'] = df['cumulative_total'] / len(df)
        
        # Calculate lift by deciles
        df['decile'] = pd.qcut(df.index, n_bins, labels=False) + 1
        
        decile_stats = df.groupby('decile').agg({
            'y_true': ['sum', 'count'],
            'cumulative_fraud_pct': 'last'
        }).reset_index()
        decile_stats.columns = ['decile', 'frauds', 'total', 'cumulative_fraud_pct']
        decile_stats['fraud_rate'] = decile_stats['frauds'] / decile_stats['total']
        decile_stats['lift'] = decile_stats['fraud_rate'] / (df['y_true'].sum() / len(df))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Cumulative Gains Chart
        ax1 = axes[0]
        ax1.plot(df['cumulative_total_pct'] * 100, df['cumulative_fraud_pct'] * 100,
                 'b-', lw=2, label='Model')
        ax1.plot([0, 100], [0, 100], 'r--', lw=2, label='Random')
        ax1.fill_between(df['cumulative_total_pct'] * 100, df['cumulative_fraud_pct'] * 100,
                         df['cumulative_total_pct'] * 100, alpha=0.3)
        
        ax1.set_xlabel('% of Transactions Reviewed', fontsize=12)
        ax1.set_ylabel('% of Frauds Detected', fontsize=12)
        ax1.set_title('Cumulative Gains Chart', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 100])
        ax1.set_ylim([0, 100])
        
        # Add annotation
        # Find point where we catch 80% of frauds
        pct_80_idx = (df['cumulative_fraud_pct'] >= 0.8).idxmax()
        pct_reviewed = df.loc[pct_80_idx, 'cumulative_total_pct'] * 100
        ax1.annotate(f'80% frauds in top {pct_reviewed:.1f}% transactions',
                     xy=(pct_reviewed, 80), fontsize=10,
                     arrowprops=dict(arrowstyle='->', color='green'),
                     xytext=(pct_reviewed + 15, 70))
        
        # Right: Lift Chart by Decile
        ax2 = axes[1]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(decile_stats)))
        bars = ax2.bar(decile_stats['decile'], decile_stats['lift'], color=colors)
        ax2.axhline(y=1, color='r', linestyle='--', lw=2, label='Baseline (Lift=1)')
        
        ax2.set_xlabel('Decile (1 = Highest Risk)', fontsize=12)
        ax2.set_ylabel('Lift', fontsize=12)
        ax2.set_title('Lift Chart by Decile', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add lift value labels
        for bar, lift in zip(bars, decile_stats['lift']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{lift:.1f}x', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Lift chart saved: {save_path}")
    
    def plot_model_comparison(
        self,
        model_comparison_path: str,
        save_path: str
    ) -> None:
        """
        Plot model comparison visualization.
        
        Args:
            model_comparison_path: Path to model comparison CSV
            save_path: Path to save the plot
        """
        logger.info("Generating model comparison plot...")
        
        try:
            df = pd.read_csv(model_comparison_path)
        except Exception as e:
            logger.warning(f"Could not load model comparison: {e}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Bar chart of all metrics
        ax1 = axes[0]
        metrics = ['Recall', 'Precision', 'F1_Score', 'ROC_AUC']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax1.bar(x + i*width, df[metric], width, label=metric)
        
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1])
        
        # Right: Radar chart for best models
        ax2 = axes[1]
        
        # Simple horizontal bar comparison for recall (primary metric)
        df_sorted = df.sort_values('Recall', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df_sorted)))
        
        bars = ax2.barh(df_sorted['Model'], df_sorted['Recall'], color=colors)
        ax2.axvline(x=self.config.min_recall, color='r', linestyle='--', lw=2,
                    label=f'Min Required ({self.config.min_recall})')
        
        ax2.set_xlabel('Recall (Fraud Detection Rate)', fontsize=12)
        ax2.set_title('Model Comparison - Recall', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, df_sorted['Recall']):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison saved: {save_path}")
    
    # ==================== MAIN EXECUTION ====================
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Execute complete model evaluation pipeline.
        
        Returns:
            ModelEvaluationArtifact: Evaluation results
        """
        logger.info(f"{'='*60}")
        logger.info("Starting Model Evaluation Pipeline")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Create directories
            logger.info("Step 1: Creating directories")
            create_directories([self.config.model_evaluation_dir])
            
            # Step 2: Load model and data
            logger.info("Step 2: Loading model and data")
            model, X_test, y_test = self._load_model_and_data()
            feature_names = self._load_feature_names()
            
            # Step 3: Get predictions
            logger.info("Step 3: Generating predictions")
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred_default = (y_pred_proba >= self.config.default_threshold).astype(int)
            
            # Step 4: Threshold optimization
            logger.info("Step 4: Optimizing threshold")
            optimal_threshold, threshold_results = self.find_optimal_threshold(
                y_test, y_pred_proba,
                optimization_metric=self.config.threshold_optimization_metric
            )
            
            # Get predictions with optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Step 5: Calculate metrics with optimal threshold
            logger.info("Step 5: Calculating classification metrics")
            metrics = self.calculate_classification_metrics(
                y_test, y_pred_optimal, y_pred_proba
            )
            
            # Step 6: Calculate business metrics
            logger.info("Step 6: Calculating business metrics")
            business_metrics = self.calculate_business_metrics(
                y_test, y_pred_optimal, y_pred_proba
            )
            
            # Step 7: Check model acceptance
            logger.info("Step 7: Checking model acceptance criteria")
            is_accepted, acceptance_results = self.check_model_acceptance(metrics)
            
            # Step 8: Generate visualizations
            logger.info("Step 8: Generating visualizations")
            
            self.plot_confusion_matrix(
                y_test, y_pred_optimal,
                self.config.confusion_matrix_path
            )
            
            self.plot_roc_curve(
                y_test, y_pred_proba,
                self.config.roc_curve_path
            )
            
            self.plot_precision_recall_curve(
                y_test, y_pred_proba,
                self.config.pr_curve_path
            )
            
            self.plot_threshold_analysis(
                threshold_results, optimal_threshold,
                self.config.threshold_analysis_path
            )
            
            importance_df = self.plot_feature_importance(
                model, feature_names,
                self.config.feature_importance_path
            )
            
            self.plot_lift_chart(
                y_test, y_pred_proba,
                self.config.lift_chart_path
            )
            
            # Plot model comparison if available
            if os.path.exists(self.model_trainer_artifact.model_comparison_path):
                self.plot_model_comparison(
                    self.model_trainer_artifact.model_comparison_path,
                    self.config.model_comparison_path
                )
            
            # Step 9: Save evaluation report
            logger.info("Step 9: Saving evaluation report")
            
            evaluation_report = {
                'evaluation_timestamp': pd.Timestamp.now().isoformat(),
                'model_name': self.model_trainer_artifact.best_model_name,
                'model_path': self.model_trainer_artifact.model_path,
                
                'threshold_analysis': {
                    'default_threshold': self.config.default_threshold,
                    'optimal_threshold': optimal_threshold,
                    'optimization_metric': self.config.threshold_optimization_metric
                },
                
                'classification_metrics': metrics,
                
                'confusion_matrix': {
                    'true_negatives': metrics['true_negatives'],
                    'false_positives': metrics['false_positives'],
                    'false_negatives': metrics['false_negatives'],
                    'true_positives': metrics['true_positives']
                },
                
                'model_acceptance': {
                    'is_accepted': is_accepted,
                    'criteria_results': acceptance_results,
                    'criteria_thresholds': {
                        'min_recall': self.config.min_recall,
                        'min_precision': self.config.min_precision,
                        'min_f1_score': self.config.min_f1_score,
                        'min_roc_auc': self.config.min_roc_auc,
                        'max_false_alarm_rate': self.config.max_false_alarm_rate
                    }
                },
                
                'classification_report': classification_report(
                    y_test, y_pred_optimal, output_dict=True
                )
            }
            
            write_json(self.config.evaluation_report_path, evaluation_report)
            
            # Step 10: Save business metrics
            logger.info("Step 10: Saving business metrics")
            write_json(self.config.business_metrics_path, business_metrics)
            
            # Create artifact
            top_features = []
            if not importance_df.empty:
                top_features = list(zip(
                    importance_df['feature'].head(10).tolist(),
                    importance_df['importance'].head(10).tolist()
                ))
            
            artifact = ModelEvaluationArtifact(
                evaluation_report_path=self.config.evaluation_report_path,
                business_metrics_path=self.config.business_metrics_path,
                confusion_matrix_path=self.config.confusion_matrix_path,
                roc_curve_path=self.config.roc_curve_path,
                pr_curve_path=self.config.pr_curve_path,
                threshold_analysis_path=self.config.threshold_analysis_path,
                feature_importance_path=self.config.feature_importance_path,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                roc_auc=metrics['roc_auc'],
                pr_auc=metrics['pr_auc'],
                true_positives=metrics['true_positives'],
                true_negatives=metrics['true_negatives'],
                false_positives=metrics['false_positives'],
                false_negatives=metrics['false_negatives'],
                default_threshold=self.config.default_threshold,
                optimal_threshold=optimal_threshold,
                threshold_optimization_metric=self.config.threshold_optimization_metric,
                fraud_detection_rate=business_metrics['fraud_detection_rate'],
                false_alarm_rate=business_metrics['false_alarm_rate'],
                total_fraud_amount_caught=business_metrics['fraud_amount_caught'],
                total_fraud_amount_missed=business_metrics['fraud_amount_missed'],
                investigation_cost=business_metrics['total_investigation_cost'],
                net_savings=business_metrics['net_savings'],
                is_model_accepted=is_accepted,
                acceptance_criteria_results=acceptance_results,
                top_features=top_features,
                message="Model Accepted" if is_accepted else "Model Rejected"
            )
            
            logger.info(f"{'='*60}")
            logger.info("MODEL EVALUATION COMPLETED!")
            logger.info(f"{'='*60}")
            logger.info(f"\n{'='*40}")
            logger.info("FINAL EVALUATION SUMMARY")
            logger.info(f"{'='*40}")
            logger.info(f"Model: {self.model_trainer_artifact.best_model_name}")
            logger.info(f"Optimal Threshold: {optimal_threshold:.2f}")
            logger.info(f"\nClassification Metrics:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            logger.info(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
            logger.info(f"\nBusiness Impact:")
            logger.info(f"  Frauds Caught: {business_metrics['frauds_caught']}/{business_metrics['total_frauds']}")
            logger.info(f"  False Alarms: {business_metrics['false_alarms']}")
            logger.info(f"  Net Savings: ${business_metrics['net_savings']:,.2f}")
            logger.info(f"\nModel Status: {'✓ ACCEPTED' if is_accepted else '✗ REJECTED'}")
            
            return artifact
            
        except Exception as e:
            raise ModelEvaluationException(
                error_message=f"Model Evaluation failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig,
        DataValidationConfig,
        FeatureEngineeringConfig,
        DataTransformationConfig,
        ModelTrainerConfig
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.feature_engineering import FeatureEngineering
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    
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
    
    print("\n" + "="*60)
    print("RUNNING MODEL TRAINING")
    print("="*60)
    trainer_config = ModelTrainerConfig(
        enable_hyperparameter_tuning=False,  # Faster for testing
        n_trials=10
    )
    model_trainer = ModelTrainer(
        config=trainer_config,
        data_transformation_artifact=transformation_artifact
    )
    trainer_artifact = model_trainer.initiate_model_training()
    
    # Run Model Evaluation
    print("\n" + "="*60)
    print("RUNNING MODEL EVALUATION")
    print("="*60)
    evaluation_config = ModelEvaluationConfig()
    model_evaluation = ModelEvaluation(
        config=evaluation_config,
        data_transformation_artifact=transformation_artifact,
        model_trainer_artifact=trainer_artifact
    )
    
    evaluation_artifact = model_evaluation.initiate_model_evaluation()
    
    print("\n" + "="*60)
    print("MODEL EVALUATION ARTIFACT SUMMARY")
    print("="*60)
    print(f"Evaluation Report: {evaluation_artifact.evaluation_report_path}")
    print(f"Business Metrics: {evaluation_artifact.business_metrics_path}")
    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {evaluation_artifact.accuracy:.4f}")
    print(f"  Precision: {evaluation_artifact.precision:.4f}")
    print(f"  Recall:    {evaluation_artifact.recall:.4f}")
    print(f"  F1-Score:  {evaluation_artifact.f1_score:.4f}")
    print(f"  ROC-AUC:   {evaluation_artifact.roc_auc:.4f}")
    print(f"\nOptimal Threshold: {evaluation_artifact.optimal_threshold:.2f}")
    print(f"\nBusiness Impact:")
    print(f"  Fraud Detection Rate: {evaluation_artifact.fraud_detection_rate*100:.1f}%")
    print(f"  False Alarm Rate: {evaluation_artifact.false_alarm_rate*100:.2f}%")
    print(f"  Net Savings: ${evaluation_artifact.net_savings:,.2f}")
    print(f"\nModel Accepted: {evaluation_artifact.is_model_accepted}")
    print(f"\nTop 5 Features:")
    for feature, importance in evaluation_artifact.top_features[:5]:
        print(f"  {feature}: {importance:.4f}")