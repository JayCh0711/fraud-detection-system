"""
Model Pusher & Registry Component for Fraud Detection System

Features:
1. Semantic Versioning (Major.Minor.Patch)
2. Model Registration with Metadata
3. Model Comparison (Current vs Previous)
4. Staging to Production Promotion
5. Rollback Capability
6. Audit Trail
7. MLflow Integration (Optional)
"""

import os
import sys
import shutil
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import asdict, dataclass
from datetime import datetime
import pandas as pd
import numpy as np

# Project imports
from src.entity.config_entity import (
    ModelPusherConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    DataTransformationConfig
)
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)
from src.logger import logger
from src.exception import FraudDetectionException
from src.utils.common import (
    write_json,
    read_json,
    save_object,
    load_object,
    create_directories
)
from src.constants import ARTIFACTS_DIR


# ============== MODEL VERSION CLASS ==============
@dataclass
class ModelVersion:
    """Represents a semantic version."""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    @classmethod
    def from_string(cls, version_str: str) -> 'ModelVersion':
        """Parse version from string like 'v1.0.0' or '1.0.0'."""
        version_str = version_str.replace('v', '').strip()
        parts = version_str.split('.')
        
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2])
        )
    
    def increment_patch(self) -> 'ModelVersion':
        """Increment patch version."""
        return ModelVersion(self.major, self.minor, self.patch + 1)
    
    def increment_minor(self) -> 'ModelVersion':
        """Increment minor version, reset patch."""
        return ModelVersion(self.major, self.minor + 1, 0)
    
    def increment_major(self) -> 'ModelVersion':
        """Increment major version, reset minor and patch."""
        return ModelVersion(self.major + 1, 0, 0)
    
    def __gt__(self, other: 'ModelVersion') -> bool:
        if self.major != other.major:
            return self.major > other.major
        if self.minor != other.minor:
            return self.minor > other.minor
        return self.patch > other.patch
    
    def __eq__(self, other: 'ModelVersion') -> bool:
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch)


# ============== MODEL REGISTRY CLASS ==============
class ModelRegistry:
    """
    Model Registry for managing model versions.
    
    Responsibilities:
    - Track all model versions
    - Store model metadata
    - Compare model versions
    - Manage production model
    """
    
    def __init__(self, registry_path: str):
        """
        Initialize Model Registry.
        
        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load or create registry."""
        if os.path.exists(self.registry_path):
            try:
                return read_json(self.registry_path)
            except Exception:
                logger.warning("Could not load registry. Creating new one.")
        
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "current_production_version": None,
            "versions": {},
            "history": []
        }
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        self.registry["updated_at"] = datetime.now().isoformat()
        write_json(self.registry_path, self.registry)
    
    def get_latest_version(self) -> Optional[str]:
        """Get the latest registered version."""
        versions = list(self.registry.get("versions", {}).keys())
        
        if not versions:
            return None
        
        # Sort versions and get latest
        sorted_versions = sorted(
            versions,
            key=lambda v: ModelVersion.from_string(v),
            reverse=True
        )
        
        return sorted_versions[0]
    
    def get_production_version(self) -> Optional[str]:
        """Get current production version."""
        return self.registry.get("current_production_version")
    
    def get_next_version(self, increment_type: str = "patch") -> str:
        """
        Get next version number.
        
        Args:
            increment_type: "major", "minor", or "patch"
        
        Returns:
            Next version string
        """
        latest = self.get_latest_version()
        
        if latest is None:
            return "1.0.0"
        
        current = ModelVersion.from_string(latest)
        
        if increment_type == "major":
            new_version = current.increment_major()
        elif increment_type == "minor":
            new_version = current.increment_minor()
        else:
            new_version = current.increment_patch()
        
        return str(new_version)
    
    def register_model(
        self,
        version: str,
        model_path: str,
        metrics: Dict,
        metadata: Dict
    ) -> None:
        """
        Register a new model version.
        
        Args:
            version: Version string
            model_path: Path to model directory
            metrics: Model performance metrics
            metadata: Model metadata
        """
        self.registry["versions"][version] = {
            "registered_at": datetime.now().isoformat(),
            "model_path": model_path,
            "metrics": metrics,
            "metadata": metadata,
            "status": "registered"
        }
        
        # Add to history
        self.registry["history"].append({
            "action": "register",
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "details": f"Model version {version} registered"
        })
        
        self._save_registry()
        logger.info(f"Model version {version} registered")
    
    def promote_to_production(self, version: str, reason: str = "") -> bool:
        """
        Promote a model version to production.
        
        Args:
            version: Version to promote
            reason: Reason for promotion
        
        Returns:
            True if promoted successfully
        """
        if version not in self.registry["versions"]:
            logger.error(f"Version {version} not found in registry")
            return False
        
        previous_version = self.registry.get("current_production_version")
        
        # Update production version
        self.registry["current_production_version"] = version
        self.registry["versions"][version]["status"] = "production"
        self.registry["versions"][version]["promoted_at"] = datetime.now().isoformat()
        
        # Update previous production version status
        if previous_version and previous_version in self.registry["versions"]:
            self.registry["versions"][previous_version]["status"] = "archived"
        
        # Add to history
        self.registry["history"].append({
            "action": "promote",
            "version": version,
            "previous_version": previous_version,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "details": f"Model {version} promoted to production"
        })
        
        self._save_registry()
        logger.info(f"Model version {version} promoted to production")
        
        return True
    
    def rollback(self, target_version: str, reason: str = "") -> bool:
        """
        Rollback to a previous version.
        
        Args:
            target_version: Version to rollback to
            reason: Reason for rollback
        
        Returns:
            True if rollback successful
        """
        if target_version not in self.registry["versions"]:
            logger.error(f"Version {target_version} not found in registry")
            return False
        
        current_version = self.registry.get("current_production_version")
        
        # Perform rollback
        self.registry["current_production_version"] = target_version
        self.registry["versions"][target_version]["status"] = "production"
        
        if current_version and current_version in self.registry["versions"]:
            self.registry["versions"][current_version]["status"] = "rolled_back"
        
        # Add to history
        self.registry["history"].append({
            "action": "rollback",
            "from_version": current_version,
            "to_version": target_version,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "details": f"Rolled back from {current_version} to {target_version}"
        })
        
        self._save_registry()
        logger.info(f"Rolled back to version {target_version}")
        
        return True
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """Get information about a specific version."""
        return self.registry.get("versions", {}).get(version)
    
    def get_all_versions(self) -> List[str]:
        """Get all registered versions."""
        return list(self.registry.get("versions", {}).keys())
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent registry history."""
        history = self.registry.get("history", [])
        return history[-limit:] if limit else history
    
    def compare_versions(
        self,
        version1: str,
        version2: str
    ) -> Optional[Dict]:
        """
        Compare two model versions.
        
        Args:
            version1: First version
            version2: Second version
        
        Returns:
            Comparison dictionary
        """
        v1_info = self.get_version_info(version1)
        v2_info = self.get_version_info(version2)
        
        if not v1_info or not v2_info:
            return None
        
        v1_metrics = v1_info.get("metrics", {})
        v2_metrics = v2_info.get("metrics", {})
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {}
        }
        
        # Compare each metric
        all_metrics = set(v1_metrics.keys()) | set(v2_metrics.keys())
        
        for metric in all_metrics:
            val1 = v1_metrics.get(metric, 0)
            val2 = v2_metrics.get(metric, 0)
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            
            comparison["metrics_comparison"][metric] = {
                f"{version1}": val1,
                f"{version2}": val2,
                "difference": round(diff, 4),
                "pct_change": round(pct_change, 2)
            }
        
        return comparison


# ============== MODEL PUSHER CLASS ==============
class ModelPusher:
    """
    Model Pusher Component for Fraud Detection System.
    
    Responsibilities:
    - Register trained models with versioning
    - Store model artifacts and metadata
    - Compare with previous versions
    - Promote models to production
    - Manage production model deployment
    """
    
    def __init__(
        self,
        config: ModelPusherConfig = ModelPusherConfig(),
        data_transformation_artifact: DataTransformationArtifact = None,
        model_trainer_artifact: ModelTrainerArtifact = None,
        model_evaluation_artifact: ModelEvaluationArtifact = None
    ):
        """
        Initialize Model Pusher component.
        
        Args:
            config: ModelPusherConfig object
            data_transformation_artifact: Artifact from data transformation
            model_trainer_artifact: Artifact from model training
            model_evaluation_artifact: Artifact from model evaluation
        """
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact
        self.model_evaluation_artifact = model_evaluation_artifact
        
        # Initialize registry
        self.registry = ModelRegistry(self.config.registry_path)
        
        logger.info(f"{'='*60}")
        logger.info("Initializing Model Pusher Component")
        logger.info(f"{'='*60}")
    
    def _create_model_metadata(self) -> Dict:
        """
        Create comprehensive model metadata.
        
        Returns:
            Dictionary with model metadata
        """
        metadata = {
            "model_name": self.model_trainer_artifact.best_model_name,
            "model_type": "classification",
            "task": "fraud_detection",
            
            "training_info": {
                "trained_at": self.model_trainer_artifact.training_timestamp,
                "training_duration_seconds": self.model_trainer_artifact.training_duration_seconds,
                "hyperparameter_tuning": self.model_trainer_artifact.hyperparameter_tuning_enabled,
                "best_params": self.model_trainer_artifact.best_model_params
            },
            
            "data_info": {
                "train_shape": list(self.data_transformation_artifact.transformed_train_shape),
                "test_shape": list(self.data_transformation_artifact.transformed_test_shape),
                "features_count": len(self.data_transformation_artifact.selected_features),
                "selected_features": self.data_transformation_artifact.selected_features[:20],  # Top 20
                "scaling_method": self.data_transformation_artifact.scaling_method,
                "imbalance_method": self.data_transformation_artifact.imbalance_method
            },
            
            "evaluation_info": {
                "optimal_threshold": self.model_evaluation_artifact.optimal_threshold,
                "is_accepted": self.model_evaluation_artifact.is_model_accepted,
                "acceptance_criteria": self.model_evaluation_artifact.acceptance_criteria_results
            },
            
            "created_at": datetime.now().isoformat(),
            "created_by": "fraud_detection_pipeline"
        }
        
        return metadata
    
    def _create_metrics_dict(self) -> Dict:
        """
        Create metrics dictionary for registration.
        
        Returns:
            Dictionary with model metrics
        """
        metrics = {
            "accuracy": self.model_evaluation_artifact.accuracy,
            "precision": self.model_evaluation_artifact.precision,
            "recall": self.model_evaluation_artifact.recall,
            "f1_score": self.model_evaluation_artifact.f1_score,
            "roc_auc": self.model_evaluation_artifact.roc_auc,
            "pr_auc": self.model_evaluation_artifact.pr_auc,
            
            "confusion_matrix": {
                "true_positives": self.model_evaluation_artifact.true_positives,
                "true_negatives": self.model_evaluation_artifact.true_negatives,
                "false_positives": self.model_evaluation_artifact.false_positives,
                "false_negatives": self.model_evaluation_artifact.false_negatives
            },
            
            "business_metrics": {
                "fraud_detection_rate": self.model_evaluation_artifact.fraud_detection_rate,
                "false_alarm_rate": self.model_evaluation_artifact.false_alarm_rate,
                "net_savings": self.model_evaluation_artifact.net_savings
            },
            
            "threshold": self.model_evaluation_artifact.optimal_threshold
        }
        
        return metrics
    
    def _check_promotion_criteria(self, metrics: Dict) -> Tuple[bool, str]:
        """
        Check if model meets promotion criteria.
        
        Args:
            metrics: Model metrics
        
        Returns:
            Tuple of (meets_criteria, reason)
        """
        logger.info("Checking promotion criteria...")
        
        recall = metrics.get('recall', 0)
        precision = metrics.get('precision', 0)
        f1 = metrics.get('f1_score', 0)
        
        criteria_results = {
            'min_recall': recall >= self.config.min_recall_for_promotion,
            'min_precision': precision >= self.config.min_precision_for_promotion,
            'min_f1': f1 >= self.config.min_f1_for_promotion
        }
        
        logger.info(f"  Recall >= {self.config.min_recall_for_promotion}: {recall:.4f} - "
                   f"{'✓' if criteria_results['min_recall'] else '✗'}")
        logger.info(f"  Precision >= {self.config.min_precision_for_promotion}: {precision:.4f} - "
                   f"{'✓' if criteria_results['min_precision'] else '✗'}")
        logger.info(f"  F1-Score >= {self.config.min_f1_for_promotion}: {f1:.4f} - "
                   f"{'✓' if criteria_results['min_f1'] else '✗'}")
        
        meets_criteria = all(criteria_results.values())
        
        if meets_criteria:
            reason = "Model meets all promotion criteria"
        else:
            failed = [k for k, v in criteria_results.items() if not v]
            reason = f"Model failed criteria: {', '.join(failed)}"
        
        return meets_criteria, reason
    
    def _check_improvement(
        self,
        current_metrics: Dict,
        previous_version: str
    ) -> Tuple[bool, Dict]:
        """
        Check if current model is an improvement over previous.
        
        Args:
            current_metrics: Current model metrics
            previous_version: Previous version to compare against
        
        Returns:
            Tuple of (is_improvement, improvement_details)
        """
        if not previous_version:
            logger.info("No previous version to compare. First model.")
            return True, {"message": "First model version"}
        
        logger.info(f"Comparing with previous version: {previous_version}")
        
        previous_info = self.registry.get_version_info(previous_version)
        
        if not previous_info:
            logger.warning(f"Could not find info for version {previous_version}")
            return True, {"message": "Previous version info not found"}
        
        previous_metrics = previous_info.get('metrics', {})
        
        # Compare key metrics
        improvements = {}
        primary_metric = 'recall'  # Primary metric for fraud detection
        
        for metric in ['recall', 'precision', 'f1_score', 'roc_auc']:
            current_val = current_metrics.get(metric, 0)
            previous_val = previous_metrics.get(metric, 0)
            diff = current_val - previous_val
            
            improvements[metric] = {
                'current': round(current_val, 4),
                'previous': round(previous_val, 4),
                'difference': round(diff, 4),
                'improved': diff > 0
            }
            
            logger.info(f"  {metric}: {previous_val:.4f} → {current_val:.4f} "
                       f"({'↑' if diff > 0 else '↓'} {abs(diff):.4f})")
        
        # Check if primary metric improved beyond threshold
        primary_improvement = improvements[primary_metric]['difference']
        is_improvement = primary_improvement >= self.config.improvement_threshold
        
        if not is_improvement and self.config.require_improvement:
            logger.warning(f"Primary metric ({primary_metric}) did not improve by "
                          f"{self.config.improvement_threshold*100}%")
        
        return is_improvement, improvements
    
    def _copy_model_to_registry(
        self,
        version: str
    ) -> str:
        """
        Copy model artifacts to registry.
        
        Args:
            version: Version string
        
        Returns:
            Path to version directory in registry
        """
        version_dir = os.path.join(
            self.config.model_registry_dir,
            f"v{version}"
        )
        
        create_directories([version_dir])
        
        # Copy model
        src_model = self.model_trainer_artifact.model_path
        dst_model = os.path.join(version_dir, self.config.model_file)
        shutil.copy2(src_model, dst_model)
        logger.info(f"Model copied to: {dst_model}")
        
        # Copy preprocessor
        src_preprocessor = self.data_transformation_artifact.preprocessor_path
        dst_preprocessor = os.path.join(version_dir, self.config.preprocessor_file)
        shutil.copy2(src_preprocessor, dst_preprocessor)
        logger.info(f"Preprocessor copied to: {dst_preprocessor}")
        
        # Copy feature names
        src_features = self.data_transformation_artifact.feature_names_path
        dst_features = os.path.join(version_dir, self.config.feature_names_file)
        shutil.copy2(src_features, dst_features)
        logger.info(f"Feature names copied to: {dst_features}")
        
        return version_dir
    
    def _copy_model_to_production(self, version: str) -> None:
        """
        Copy model artifacts to production directory.
        
        Args:
            version: Version being promoted
        """
        create_directories([self.config.production_model_dir])
        
        version_dir = os.path.join(
            self.config.model_registry_dir,
            f"v{version}"
        )
        
        # Copy all artifacts to production
        files_to_copy = [
            (self.config.model_file, self.config.production_model_path),
            (self.config.preprocessor_file, self.config.production_preprocessor_path),
            (self.config.metadata_file, self.config.production_metadata_path),
            (self.config.feature_names_file, 
             os.path.join(self.config.production_model_dir, self.config.feature_names_file))
        ]
        
        for src_file, dst_path in files_to_copy:
            src_path = os.path.join(version_dir, src_file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied to production: {dst_path}")
        
        # Create version info file
        version_info = {
            "version": version,
            "promoted_at": datetime.now().isoformat(),
            "model_name": self.model_trainer_artifact.best_model_name,
            "optimal_threshold": self.model_evaluation_artifact.optimal_threshold,
            "metrics": {
                "recall": self.model_evaluation_artifact.recall,
                "precision": self.model_evaluation_artifact.precision,
                "f1_score": self.model_evaluation_artifact.f1_score,
                "roc_auc": self.model_evaluation_artifact.roc_auc
            }
        }
        
        write_json(self.config.production_version_path, version_info)
        logger.info(f"Version info saved: {self.config.production_version_path}")
    
    def register_model(self) -> Tuple[str, str]:
        """
        Register the current model in the registry.
        
        Returns:
            Tuple of (new_version, version_directory)
        """
        logger.info("Registering model in registry...")
        
        # Determine version
        new_version = self.registry.get_next_version(increment_type="patch")
        logger.info(f"New version: {new_version}")
        
        # Copy model to registry
        version_dir = self._copy_model_to_registry(new_version)
        
        # Create and save metadata
        metadata = self._create_model_metadata()
        metadata_path = os.path.join(version_dir, self.config.metadata_file)
        write_json(metadata_path, metadata)
        
        # Create and save metrics
        metrics = self._create_metrics_dict()
        metrics_path = os.path.join(version_dir, self.config.metrics_file)
        write_json(metrics_path, metrics)
        
        # Register in registry
        self.registry.register_model(
            version=new_version,
            model_path=version_dir,
            metrics=metrics,
            metadata=metadata
        )
        
        return new_version, version_dir
    
    def promote_model(
        self,
        version: str,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Promote a model version to production.
        
        Args:
            version: Version to promote
            force: Force promotion even if criteria not met
        
        Returns:
            Tuple of (success, reason)
        """
        logger.info(f"Attempting to promote version {version} to production...")
        
        # Get version info
        version_info = self.registry.get_version_info(version)
        
        if not version_info:
            return False, f"Version {version} not found in registry"
        
        metrics = version_info.get('metrics', {})
        
        # Check promotion criteria
        meets_criteria, criteria_reason = self._check_promotion_criteria(metrics)
        
        if not meets_criteria and not force:
            return False, criteria_reason
        
        # Check improvement over current production
        current_production = self.registry.get_production_version()
        is_improvement, improvement_details = self._check_improvement(
            metrics, current_production
        )
        
        if not is_improvement and self.config.require_improvement and not force:
            return False, "Model did not show sufficient improvement over current production"
        
        # Copy to production directory
        self._copy_model_to_production(version)
        
        # Update registry
        promotion_reason = f"Promoted: {criteria_reason}. Improvement: {is_improvement}"
        self.registry.promote_to_production(version, promotion_reason)
        
        return True, promotion_reason
    
    def rollback_model(
        self,
        target_version: str,
        reason: str = "Manual rollback"
    ) -> Tuple[bool, str]:
        """
        Rollback to a previous model version.
        
        Args:
            target_version: Version to rollback to
            reason: Reason for rollback
        
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Rolling back to version {target_version}...")
        
        # Verify version exists
        version_info = self.registry.get_version_info(target_version)
        
        if not version_info:
            return False, f"Version {target_version} not found in registry"
        
        # Copy to production
        self._copy_model_to_production(target_version)
        
        # Update registry
        self.registry.rollback(target_version, reason)
        
        return True, f"Successfully rolled back to version {target_version}"
    
    def get_production_model_info(self) -> Optional[Dict]:
        """Get information about current production model."""
        production_version = self.registry.get_production_version()
        
        if not production_version:
            return None
        
        return self.registry.get_version_info(production_version)
    
    def list_all_versions(self) -> pd.DataFrame:
        """
        List all registered versions with their metrics.
        
        Returns:
            DataFrame with version information
        """
        versions = self.registry.get_all_versions()
        
        if not versions:
            return pd.DataFrame()
        
        data = []
        for version in versions:
            info = self.registry.get_version_info(version)
            metrics = info.get('metrics', {})
            
            data.append({
                'Version': version,
                'Status': info.get('status', 'unknown'),
                'Registered': info.get('registered_at', ''),
                'Recall': metrics.get('recall', 0),
                'Precision': metrics.get('precision', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'ROC-AUC': metrics.get('roc_auc', 0)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Version', ascending=False)
        
        return df
    
    # ==================== MLFLOW INTEGRATION ====================
    
    def _log_to_mlflow(
        self,
        version: str,
        metrics: Dict,
        metadata: Dict
    ) -> Optional[str]:
        """
        Log model to MLflow (if enabled).
        
        Args:
            version: Model version
            metrics: Model metrics
            metadata: Model metadata
        
        Returns:
            MLflow run ID or None
        """
        if not self.config.use_mlflow:
            return None
        
        try:
            import mlflow
            from mlflow.models.signature import infer_signature
            
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            
            with mlflow.start_run(run_name=f"fraud_detection_v{version}"):
                # Log parameters
                mlflow.log_params(metadata.get('training_info', {}).get('best_params', {}))
                
                # Log metrics
                mlflow.log_metric("accuracy", metrics.get('accuracy', 0))
                mlflow.log_metric("precision", metrics.get('precision', 0))
                mlflow.log_metric("recall", metrics.get('recall', 0))
                mlflow.log_metric("f1_score", metrics.get('f1_score', 0))
                mlflow.log_metric("roc_auc", metrics.get('roc_auc', 0))
                
                # Log model
                model = load_object(self.model_trainer_artifact.model_path)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="fraud_detection_model"
                )
                
                # Log artifacts
                mlflow.log_artifact(self.model_evaluation_artifact.evaluation_report_path)
                mlflow.log_artifact(self.model_evaluation_artifact.confusion_matrix_path)
                mlflow.log_artifact(self.model_evaluation_artifact.roc_curve_path)
                
                run_id = mlflow.active_run().info.run_id
                logger.info(f"Logged to MLflow. Run ID: {run_id}")
                
                return run_id
                
        except ImportError:
            logger.warning("MLflow not installed. Skipping MLflow logging.")
            return None
        except Exception as e:
            logger.warning(f"MLflow logging failed: {str(e)}")
            return None
    
    # ==================== MAIN EXECUTION ====================
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Execute complete model pusher pipeline.
        
        Returns:
            ModelPusherArtifact: Pusher results
        """
        logger.info(f"{'='*60}")
        logger.info("Starting Model Pusher Pipeline")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Create directories
            logger.info("Step 1: Creating directories")
            create_directories([
                self.config.model_registry_dir,
                self.config.production_model_dir
            ])
            
            # Step 2: Check if model was accepted
            logger.info("Step 2: Checking model acceptance status")
            if not self.model_evaluation_artifact.is_model_accepted:
                logger.warning("Model was not accepted during evaluation. Registering but not promoting.")
            
            # Step 3: Register model
            logger.info("Step 3: Registering model")
            new_version, version_dir = self.register_model()
            
            # Step 4: Get metrics for comparison
            metrics = self._create_metrics_dict()
            
            # Step 5: Check improvement over previous
            logger.info("Step 4: Checking improvement")
            previous_version = self.registry.get_production_version()
            is_improvement, improvement_metrics = self._check_improvement(
                metrics, previous_version
            )
            
            # Step 6: Attempt promotion
            logger.info("Step 5: Attempting promotion to production")
            is_promoted = False
            promotion_reason = ""
            
            if self.model_evaluation_artifact.is_model_accepted:
                is_promoted, promotion_reason = self.promote_model(
                    new_version,
                    force=False
                )
            else:
                promotion_reason = "Model not accepted during evaluation"
            
            # Step 7: Log to MLflow (if enabled)
            logger.info("Step 6: MLflow logging")
            metadata = self._create_model_metadata()
            mlflow_run_id = self._log_to_mlflow(new_version, metrics, metadata)
            
            if mlflow_run_id:
                logger.info(f"MLflow Run ID: {mlflow_run_id}")
            
            # Step 8: Create artifact
            artifact = ModelPusherArtifact(
                model_version=new_version,
                previous_version=previous_version,
                registry_model_path=os.path.join(version_dir, self.config.model_file),
                registry_preprocessor_path=os.path.join(version_dir, self.config.preprocessor_file),
                registry_metadata_path=os.path.join(version_dir, self.config.metadata_file),
                production_model_path=self.config.production_model_path if is_promoted else "",
                production_preprocessor_path=self.config.production_preprocessor_path if is_promoted else "",
                is_model_registered=True,
                is_model_promoted=is_promoted,
                promotion_reason=promotion_reason,
                is_improvement=is_improvement,
                improvement_metrics=improvement_metrics,
                recall=self.model_evaluation_artifact.recall,
                precision=self.model_evaluation_artifact.precision,
                f1_score=self.model_evaluation_artifact.f1_score,
                roc_auc=self.model_evaluation_artifact.roc_auc,
                message="Model registered and promoted" if is_promoted else "Model registered only"
            )
            
            # Step 9: Log summary
            logger.info(f"{'='*60}")
            logger.info("MODEL PUSHER COMPLETED!")
            logger.info(f"{'='*60}")
            logger.info(f"\nVersion: {new_version}")
            logger.info(f"Previous Production: {previous_version or 'None'}")
            logger.info(f"Registered: ✓")
            logger.info(f"Promoted to Production: {'✓' if is_promoted else '✗'}")
            logger.info(f"Reason: {promotion_reason}")
            
            if is_promoted:
                logger.info(f"\nProduction Model Location:")
                logger.info(f"  Model: {self.config.production_model_path}")
                logger.info(f"  Preprocessor: {self.config.production_preprocessor_path}")
            
            # Display version history
            logger.info(f"\n{'='*40}")
            logger.info("REGISTERED VERSIONS")
            logger.info(f"{'='*40}")
            versions_df = self.list_all_versions()
            if not versions_df.empty:
                logger.info("\n" + versions_df.to_string(index=False))
            
            return artifact
            
        except Exception as e:
            raise FraudDetectionException(
                error_message=f"Model Pusher failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig,
        DataValidationConfig,
        FeatureEngineeringConfig,
        DataTransformationConfig,
        ModelTrainerConfig,
        ModelEvaluationConfig
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.feature_engineering import FeatureEngineering
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    from src.components.model_evaluation import ModelEvaluation
    
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
        enable_hyperparameter_tuning=False,
        n_trials=10
    )
    model_trainer = ModelTrainer(
        config=trainer_config,
        data_transformation_artifact=transformation_artifact
    )
    trainer_artifact = model_trainer.initiate_model_training()
    
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
    
    # Run Model Pusher
    print("\n" + "="*60)
    print("RUNNING MODEL PUSHER")
    print("="*60)
    pusher_config = ModelPusherConfig()
    model_pusher = ModelPusher(
        config=pusher_config,
        data_transformation_artifact=transformation_artifact,
        model_trainer_artifact=trainer_artifact,
        model_evaluation_artifact=evaluation_artifact
    )
    
    pusher_artifact = model_pusher.initiate_model_pusher()
    
    print("\n" + "="*60)
    print("MODEL PUSHER ARTIFACT SUMMARY")
    print("="*60)
    print(f"Model Version: {pusher_artifact.model_version}")
    print(f"Previous Version: {pusher_artifact.previous_version or 'None (First Version)'}")
    print(f"\nRegistered: {pusher_artifact.is_model_registered}")
    print(f"Promoted: {pusher_artifact.is_model_promoted}")
    print(f"Reason: {pusher_artifact.promotion_reason}")
    print(f"\nRegistry Path: {pusher_artifact.registry_model_path}")
    if pusher_artifact.is_model_promoted:
        print(f"Production Path: {pusher_artifact.production_model_path}")
    print(f"\nMetrics:")
    print(f"  Recall: {pusher_artifact.recall:.4f}")
    print(f"  Precision: {pusher_artifact.precision:.4f}")
    print(f"  F1-Score: {pusher_artifact.f1_score:.4f}")
    print(f"  ROC-AUC: {pusher_artifact.roc_auc:.4f}")