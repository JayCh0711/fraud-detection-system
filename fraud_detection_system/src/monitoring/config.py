"""
Monitoring Configuration for Fraud Detection System
"""

import os
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings

from src.constants import ARTIFACTS_DIR


class MonitoringSettings(BaseSettings):
    """Monitoring Configuration Settings"""
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "fraud_detection"
    MLFLOW_REGISTRY_URI: str = "sqlite:///mlflow.db"
    MLFLOW_ARTIFACT_LOCATION: str = "./mlflow-artifacts"
    
    # Evidently Configuration
    EVIDENTLY_WORKSPACE: str = "./evidently_workspace"
    EVIDENTLY_PROJECT_NAME: str = "fraud_detection_monitoring"
    
    # Drift Detection Thresholds
    DATA_DRIFT_THRESHOLD: float = 0.1  # 10% drift triggers alert
    PREDICTION_DRIFT_THRESHOLD: float = 0.15
    FEATURE_DRIFT_THRESHOLD: float = 0.2
    
    # Performance Thresholds
    MIN_RECALL_THRESHOLD: float = 0.75  # Alert if recall drops below
    MIN_PRECISION_THRESHOLD: float = 0.45
    MIN_F1_THRESHOLD: float = 0.55
    MAX_FALSE_ALARM_RATE: float = 0.10
    
    # Monitoring Schedule
    MONITORING_INTERVAL_MINUTES: int = 60
    REPORT_RETENTION_DAYS: int = 30
    
    # Alerting
    ENABLE_EMAIL_ALERTS: bool = False
    ALERT_EMAIL_RECIPIENTS: List[str] = []
    ENABLE_SLACK_ALERTS: bool = False
    SLACK_WEBHOOK_URL: Optional[str] = None
    
    # Dashboard
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 8050
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@dataclass
class MonitoringConfig:
    """Monitoring Pipeline Configuration"""
    
    # Directories
    monitoring_dir: str = os.path.join(ARTIFACTS_DIR, "monitoring")
    drift_reports_dir: str = os.path.join(ARTIFACTS_DIR, "monitoring", "drift_reports")
    performance_reports_dir: str = os.path.join(ARTIFACTS_DIR, "monitoring", "performance_reports")
    alerts_dir: str = os.path.join(ARTIFACTS_DIR, "monitoring", "alerts")
    
    # Reference Data
    reference_data_path: str = os.path.join(ARTIFACTS_DIR, "data_ingestion", "processed", "train.csv")
    
    # Model
    production_model_dir: str = os.path.join(ARTIFACTS_DIR, "production_model")
    
    # Settings
    settings: MonitoringSettings = field(default_factory=MonitoringSettings)


# Global instances
monitoring_settings = MonitoringSettings()
monitoring_config = MonitoringConfig()