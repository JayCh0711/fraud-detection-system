"""
MLflow Integration for Experiment Tracking and Model Registry

Features:
1. Experiment Tracking
2. Model Logging
3. Metric Logging
4. Artifact Logging
5. Model Registry
"""

import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from src.monitoring.config import monitoring_settings
from src.logger import logger

# Try to import MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Run: pip install mlflow")


class MLflowTracker:
    """
    MLflow integration for experiment tracking and model registry.
    """
    
    def __init__(
        self,
        tracking_uri: str = None,
        experiment_name: str = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri or monitoring_settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or monitoring_settings.MLFLOW_EXPERIMENT_NAME
        
        self.client = None
        self.experiment_id = None
        self.is_initialized = False
        
        if MLFLOW_AVAILABLE:
            self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize MLflow connection."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available")
            return False
        
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set/create experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=monitoring_settings.MLFLOW_ARTIFACT_LOCATION
                )
            else:
                self.experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            
            # Initialize client
            self.client = MlflowClient()
            
            self.is_initialized = True
            logger.info(f"MLflow initialized: {self.tracking_uri}")
            logger.info(f"Experiment: {self.experiment_name} (ID: {self.experiment_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"MLflow initialization failed: {str(e)}")
            return False
    
    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None
    ) -> Optional[str]:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
        
        Returns:
            Run ID or None
        """
        if not self.is_initialized:
            logger.warning("MLflow not initialized")
            return None
        
        try:
            run = mlflow.start_run(run_name=run_name)
            
            if tags:
                mlflow.set_tags(tags)
            
            logger.info(f"MLflow run started: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start run: {str(e)}")
            return None
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
                logger.info("MLflow run ended")
            except Exception as e:
                logger.error(f"Failed to end run: {str(e)}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters.
        
        Args:
            params: Dictionary of parameters
        """
        if not self.is_initialized:
            return
        
        try:
            # Convert values to strings (MLflow requirement)
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log params: {str(e)}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int = None
    ) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (for time series)
        """
        if not self.is_initialized:
            return
        
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.debug(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
    
    def log_model(
        self,
        model: Any,
        model_name: str = "model",
        input_example: Any = None,
        register_model: bool = False,
        registered_name: str = None
    ) -> Optional[str]:
        """
        Log a model to MLflow.
        
        Args:
            model: Model object
            model_name: Artifact name
            input_example: Example input for signature
            register_model: Whether to register in model registry
            registered_name: Name for registered model
        
        Returns:
            Model URI or None
        """
        if not self.is_initialized:
            return None
        
        try:
            # Infer signature if example provided
            signature = None
            if input_example is not None:
                try:
                    import numpy as np
                    predictions = model.predict(input_example)
                    signature = infer_signature(input_example, predictions)
                except Exception:
                    pass
            
            # Log model
            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example
            )
            
            model_uri = model_info.model_uri
            logger.info(f"Model logged: {model_uri}")
            
            # Register if requested
            if register_model and registered_name:
                self.register_model(model_uri, registered_name)
            
            return model_uri
            
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            return None
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: str = None
    ) -> None:
        """
        Log an artifact file.
        
        Args:
            local_path: Path to local file
            artifact_path: Path in artifact store
        """
        if not self.is_initialized:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Artifact logged: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {str(e)}")
    
    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: str = None
    ) -> None:
        """
        Log a directory of artifacts.
        
        Args:
            local_dir: Path to local directory
            artifact_path: Path in artifact store
        """
        if not self.is_initialized:
            return
        
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Artifacts logged: {local_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts: {str(e)}")
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Dict[str, str] = None
    ) -> Optional[str]:
        """
        Register a model in the model registry.
        
        Args:
            model_uri: URI of the logged model
            name: Name for the registered model
            tags: Tags to add
        
        Returns:
            Model version or None
        """
        if not self.is_initialized:
            return None
        
        try:
            result = mlflow.register_model(model_uri, name)
            version = result.version
            
            # Add tags if provided
            if tags and self.client:
                for key, value in tags.items():
                    self.client.set_model_version_tag(name, version, key, value)
            
            logger.info(f"Model registered: {name} v{version}")
            return version
            
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            return None
    
    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str
    ) -> bool:
        """
        Transition a model version to a new stage.
        
        Args:
            name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
        
        Returns:
            True if successful
        """
        if not self.is_initialized or not self.client:
            return False
        
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {name} v{version} transitioned to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition model: {str(e)}")
            return False
    
    def get_latest_model_version(
        self,
        name: str,
        stage: str = None
    ) -> Optional[str]:
        """
        Get the latest version of a registered model.
        
        Args:
            name: Registered model name
            stage: Optional stage filter
        
        Returns:
            Latest version string or None
        """
        if not self.is_initialized or not self.client:
            return None
        
        try:
            if stage:
                versions = self.client.get_latest_versions(name, stages=[stage])
            else:
                versions = self.client.get_latest_versions(name)
            
            if versions:
                return versions[0].version
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model version: {str(e)}")
            return None
    
    def load_model(
        self,
        model_uri: str
    ) -> Any:
        """
        Load a model from MLflow.
        
        Args:
            model_uri: Model URI
        
        Returns:
            Loaded model or None
        """
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    
    def log_training_run(
        self,
        run_name: str,
        params: Dict,
        metrics: Dict,
        model: Any = None,
        artifacts: List[str] = None,
        tags: Dict[str, str] = None
    ) -> Optional[str]:
        """
        Log a complete training run.
        
        Args:
            run_name: Name for the run
            params: Training parameters
            metrics: Training metrics
            model: Trained model
            artifacts: List of artifact paths
            tags: Run tags
        
        Returns:
            Run ID or None
        """
        run_id = self.start_run(run_name=run_name, tags=tags)
        
        if run_id is None:
            return None
        
        try:
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            if model is not None:
                self.log_model(model, model_name="model")
            
            # Log artifacts
            if artifacts:
                for artifact_path in artifacts:
                    if os.path.exists(artifact_path):
                        self.log_artifact(artifact_path)
            
            logger.info(f"Training run logged: {run_id}")
            
        finally:
            self.end_run()
        
        return run_id


# Singleton instance
_mlflow_tracker: Optional[MLflowTracker] = None


def get_mlflow_tracker() -> MLflowTracker:
    """Get singleton MLflow tracker instance."""
    global _mlflow_tracker
    
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowTracker()
    
    return _mlflow_tracker