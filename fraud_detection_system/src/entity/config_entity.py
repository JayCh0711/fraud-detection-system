"""
Configuration Entity Classes
Dataclasses for all pipeline component configurations
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.constants import (
    ARTIFACTS_DIR,
    DATA_INGESTION_DIR,
    DATA_VALIDATION_DIR,
    DATA_TRANSFORMATION_DIR,
    MODEL_TRAINER_DIR,
    MODEL_EVALUATION_DIR,
    CONFIG_FILE_PATH,
    MODEL_CONFIG_FILE_PATH,
    SCHEMA_FILE_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)


# ============== TRAINING PIPELINE CONFIG ==============
@dataclass
class TrainingPipelineConfig:
    """Master configuration for training pipeline"""
    
    pipeline_name: str = "fraud_detection_pipeline"
    artifacts_dir: str = ARTIFACTS_DIR
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )


# ============== DATA INGESTION CONFIG ==============
@dataclass
class DataIngestionConfig:
    """Configuration for Data Ingestion component"""
    
    # Source Configuration
    source_type: str = "csv"  # csv, database, s3, api, kafka
    source_path: str = "data/raw/transactions.csv"
    
    # Database Configuration (if applicable)
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None
    db_table: Optional[str] = None
    db_query: Optional[str] = None
    
    # S3 Configuration (if applicable)
    s3_bucket: Optional[str] = None
    s3_key: Optional[str] = None
    
    # Artifact Paths
    data_ingestion_dir: str = DATA_INGESTION_DIR
    raw_data_dir: str = os.path.join(DATA_INGESTION_DIR, "raw")
    processed_data_dir: str = os.path.join(DATA_INGESTION_DIR, "processed")
    
    # File Names
    raw_file_name: str = "transactions.csv"
    train_file_name: str = "train.csv"
    test_file_name: str = "test.csv"
    
    # Split Configuration
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    stratify_column: str = TARGET_COLUMN
    
    # Full Paths (computed)
    @property
    def raw_file_path(self) -> str:
        return os.path.join(self.raw_data_dir, self.raw_file_name)
    
    @property
    def train_file_path(self) -> str:
        return os.path.join(self.processed_data_dir, self.train_file_name)
    
    @property
    def test_file_path(self) -> str:
        return os.path.join(self.processed_data_dir, self.test_file_name)


# ============== DATA VALIDATION CONFIG ==============
@dataclass
class DataValidationConfig:
    """Configuration for Data Validation component"""
    
    data_validation_dir: str = DATA_VALIDATION_DIR
    schema_file_path: str = SCHEMA_FILE_PATH
    
    # Validation Thresholds
    missing_threshold: float = 0.3
    duplicate_threshold: float = 0.1
    
    # Report Paths
    validation_report_file: str = "validation_report.json"
    drift_report_file: str = "drift_report.json"
    
    @property
    def validation_report_path(self) -> str:
        return os.path.join(self.data_validation_dir, self.validation_report_file)
    
    @property
    def drift_report_path(self) -> str:
        return os.path.join(self.data_validation_dir, self.drift_report_file)


# ============== DATA TRANSFORMATION CONFIG ==============
@dataclass
class DataTransformationConfig:
    """Configuration for Data Transformation component"""
    
    data_transformation_dir: str = DATA_TRANSFORMATION_DIR
    
    # Feature Columns
    numerical_features: List[str] = field(default_factory=lambda: [
        "amount",
        "old_balance_org",
        "new_balance_org",
        "old_balance_dest",
        "new_balance_dest"
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        "type"
    ])
    
    target_column: str = TARGET_COLUMN
    
    # Transformation Settings
    scaling_method: str = "standard"  # standard, minmax, robust
    encoding_method: str = "onehot"   # onehot, label, target
    handle_imbalance: str = "smote"   # smote, adasyn, undersampling, none
    
    # Artifact Files
    preprocessor_file: str = "preprocessor.pkl"
    encoder_file: str = "encoder.pkl"
    transformed_train_file: str = "transformed_train.npy"
    transformed_test_file: str = "transformed_test.npy"
    
    @property
    def preprocessor_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.preprocessor_file)
    
    @property
    def encoder_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.encoder_file)


# ============== MODEL TRAINER CONFIG ==============
@dataclass
class ModelTrainerConfig:
    """Configuration for Model Trainer component"""
    
    model_trainer_dir: str = MODEL_TRAINER_DIR
    model_config_path: str = MODEL_CONFIG_FILE_PATH
    
    # Model Settings
    target_column: str = TARGET_COLUMN
    
    # Artifact Files
    model_file: str = "model.pkl"
    training_report_file: str = "training_report.json"
    
    # Training Parameters
    cv_folds: int = 5
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.model_trainer_dir, self.model_file)
    
    @property
    def training_report_path(self) -> str:
        return os.path.join(self.model_trainer_dir, self.training_report_file)


# ============== MODEL EVALUATION CONFIG ==============
@dataclass
class ModelEvaluationConfig:
    """Configuration for Model Evaluation component"""
    
    model_evaluation_dir: str = MODEL_EVALUATION_DIR
    
    # Evaluation Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc"
    ])
    
    # Thresholds (BFSI specific - high recall for fraud)
    min_recall: float = 0.85
    min_precision: float = 0.70
    classification_threshold: float = 0.5
    
    # Artifact Files
    evaluation_report_file: str = "evaluation_report.json"
    confusion_matrix_file: str = "confusion_matrix.png"
    roc_curve_file: str = "roc_curve.png"
    pr_curve_file: str = "pr_curve.png"
    
    @property
    def evaluation_report_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.evaluation_report_file)


    # ============== FEATURE ENGINEERING CONFIG ==============
@dataclass
class FeatureEngineeringConfig:
    """Configuration for Feature Engineering component"""
    
    feature_engineering_dir: str = os.path.join(ARTIFACTS_DIR, "feature_engineering")
    
    # Target Column
    target_column: str = TARGET_COLUMN
    
    # Feature Settings
    create_amount_features: bool = True
    create_balance_features: bool = True
    create_velocity_features: bool = True
    create_time_features: bool = True
    create_frequency_features: bool = True
    create_error_features: bool = True
    create_ratio_features: bool = True
    
    # Output Files
    train_featured_file: str = "train_featured.csv"
    test_featured_file: str = "test_featured.csv"
    report_file: str = "feature_engineering_report.json"
    
    # Columns to drop (identifiers)
    columns_to_drop: List[str] = field(default_factory=lambda: [
        "name_orig",
        "name_dest",
        "is_flagged_fraud"
    ])
    
    @property
    def train_featured_path(self) -> str:
        return os.path.join(self.feature_engineering_dir, self.train_featured_file)
    
    @property
    def test_featured_path(self) -> str:
        return os.path.join(self.feature_engineering_dir, self.test_featured_file)
    
    @property
    def report_path(self) -> str:
        return os.path.join(self.feature_engineering_dir, self.report_file)

# ============== DATA TRANSFORMATION CONFIG ==============

@dataclass
class DataTransformationConfig:
    """Configuration for Data Transformation component"""
    
    data_transformation_dir: str = os.path.join(ARTIFACTS_DIR, "data_transformation")
    
    # Target Column
    target_column: str = TARGET_COLUMN
    
    # Scaling Configuration
    scaling_method: str = "robust"  # standard, minmax, robust
    
    # Imbalance Handling
    handle_imbalance: bool = True
    imbalance_method: str = "smote"  # smote, adasyn, smoteenn, borderline_smote, none
    sampling_strategy: float = 0.5   # Target ratio for minority class
    
    # Feature Selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "importance"  # importance, variance, correlation
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    top_k_features: int = 50  # Select top K features by importance
    
    # Outlier Handling
    handle_outliers: bool = True
    outlier_method: str = "clip"  # clip, remove, transform
    outlier_threshold: float = 3.0  # Z-score threshold
    
    # Output Files
    preprocessor_file: str = "preprocessor.pkl"
    train_transformed_file: str = "train_transformed.npy"
    test_transformed_file: str = "test_transformed.npy"
    train_target_file: str = "train_target.npy"
    test_target_file: str = "test_target.npy"
    feature_names_file: str = "feature_names.json"
    report_file: str = "transformation_report.json"
    
    @property
    def preprocessor_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.preprocessor_file)
    
    @property
    def train_transformed_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.train_transformed_file)
    
    @property
    def test_transformed_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.test_transformed_file)
    
    @property
    def train_target_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.train_target_file)
    
    @property
    def test_target_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.test_target_file)
    
    @property
    def feature_names_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.feature_names_file)
    
    @property
    def report_path(self) -> str:
        return os.path.join(self.data_transformation_dir, self.report_file)
    

# ============== MODEL TRAINER CONFIG ==============
@dataclass
class ModelTrainerConfig:
    """Configuration for Model Trainer component"""
    
    model_trainer_dir: str = os.path.join(ARTIFACTS_DIR, "model_trainer")
    model_config_path: str = MODEL_CONFIG_FILE_PATH
    
    # Target Column
    target_column: str = TARGET_COLUMN
    
    # Training Configuration
    cv_folds: int = 5
    random_state: int = RANDOM_STATE
    
    # Model Selection Criteria (BFSI - prioritize recall)
    primary_metric: str = "recall"
    secondary_metric: str = "f1_score"
    min_recall_threshold: float = 0.80  # Minimum acceptable recall
    min_precision_threshold: float = 0.50  # Minimum acceptable precision
    
    # Hyperparameter Tuning
    enable_hyperparameter_tuning: bool = True
    tuning_method: str = "optuna"  # optuna, gridsearch, randomsearch
    n_trials: int = 50  # For Optuna
    
    # Output Files
    model_file: str = "model.pkl"
    training_report_file: str = "training_report.json"
    model_comparison_file: str = "model_comparison.csv"
    best_params_file: str = "best_model_params.json"
    cv_results_file: str = "cv_results.json"
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.model_trainer_dir, self.model_file)
    
    @property
    def training_report_path(self) -> str:
        return os.path.join(self.model_trainer_dir, self.training_report_file)
    
    @property
    def model_comparison_path(self) -> str:
        return os.path.join(self.model_trainer_dir, self.model_comparison_file)
    
    @property
    def best_params_path(self) -> str:
        return os.path.join(self.model_trainer_dir, self.best_params_file)
    
    @property
    def cv_results_path(self) -> str:
        return os.path.join(self.model_trainer_dir, self.cv_results_file)
    

# ============== MODEL EVALUATION CONFIG ==============
@dataclass
class ModelEvaluationConfig:
    """Configuration for Model Evaluation component"""
    
    model_evaluation_dir: str = os.path.join(ARTIFACTS_DIR, "model_evaluation")
    
    # Target Column
    target_column: str = TARGET_COLUMN
    
    # Evaluation Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "precision", 
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc"
    ])
    
    # Threshold Configuration
    default_threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_optimization_metric: str = "f1"  # f1, recall, precision, youden
    
    # Model Acceptance Criteria (BFSI specific)
    min_recall: float = 0.80      # Must catch at least 80% of frauds
    min_precision: float = 0.50   # At least 50% of flagged are actual frauds
    min_f1_score: float = 0.65    # Balanced performance
    min_roc_auc: float = 0.85     # Good discrimination
    max_false_alarm_rate: float = 0.05  # Max 5% false alarms
    
    # Business Cost Configuration (for cost-benefit analysis)
    avg_fraud_amount: float = 50000.0      # Average fraud transaction amount
    investigation_cost: float = 50.0        # Cost to investigate a flagged transaction
    customer_friction_cost: float = 100.0   # Cost of false positive (customer impact)
    
    # Output Files
    evaluation_report_file: str = "evaluation_report.json"
    business_metrics_file: str = "business_metrics.json"
    confusion_matrix_file: str = "confusion_matrix.png"
    roc_curve_file: str = "roc_curve.png"
    pr_curve_file: str = "precision_recall_curve.png"
    threshold_analysis_file: str = "threshold_analysis.png"
    feature_importance_file: str = "feature_importance.png"
    lift_chart_file: str = "lift_chart.png"
    model_comparison_file: str = "model_comparison.png"
    
    @property
    def evaluation_report_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.evaluation_report_file)
    
    @property
    def business_metrics_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.business_metrics_file)
    
    @property
    def confusion_matrix_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.confusion_matrix_file)
    
    @property
    def roc_curve_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.roc_curve_file)
    
    @property
    def pr_curve_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.pr_curve_file)
    
    @property
    def threshold_analysis_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.threshold_analysis_file)
    
    @property
    def feature_importance_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.feature_importance_file)
    
    @property
    def lift_chart_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.lift_chart_file)
    
    @property
    def model_comparison_path(self) -> str:
        return os.path.join(self.model_evaluation_dir, self.model_comparison_file)
    
# ============== MODEL PUSHER CONFIG ==============
@dataclass
class ModelPusherConfig:
    """Configuration for Model Pusher/Registry component"""
    
    # Registry Paths
    model_registry_dir: str = os.path.join(ARTIFACTS_DIR, "model_registry")
    production_model_dir: str = os.path.join(ARTIFACTS_DIR, "production_model")
    
    # Registry Files
    registry_file: str = "registry.json"
    
    # Model Files
    model_file: str = "model.pkl"
    preprocessor_file: str = "preprocessor.pkl"
    metadata_file: str = "metadata.json"
    metrics_file: str = "metrics.json"
    version_file: str = "version.json"
    feature_names_file: str = "feature_names.json"
    
    # Versioning
    version_prefix: str = "v"
    initial_version: str = "1.0.0"
    
    # Promotion Criteria
    min_recall_for_promotion: float = 0.80
    min_precision_for_promotion: float = 0.50
    min_f1_for_promotion: float = 0.65
    require_improvement: bool = True
    improvement_threshold: float = 0.01  # 1% improvement required
    
    # MLflow Integration (Optional)
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "fraud_detection"
    
    @property
    def registry_path(self) -> str:
        return os.path.join(self.model_registry_dir, self.registry_file)
    
    @property
    def production_model_path(self) -> str:
        return os.path.join(self.production_model_dir, self.model_file)
    
    @property
    def production_preprocessor_path(self) -> str:
        return os.path.join(self.production_model_dir, self.preprocessor_file)
    
    @property
    def production_metadata_path(self) -> str:
        return os.path.join(self.production_model_dir, self.metadata_file)
    
    @property
    def production_version_path(self) -> str:
        return os.path.join(self.production_model_dir, self.version_file)

# ============== EXAMPLE USAGE ==============
if __name__ == "__main__":
    # Test configurations
    pipeline_config = TrainingPipelineConfig()
    print(f"Pipeline: {pipeline_config.pipeline_name}")
    print(f"Timestamp: {pipeline_config.timestamp}")
    
    ingestion_config = DataIngestionConfig()
    print(f"\nData Ingestion Dir: {ingestion_config.data_ingestion_dir}")
    print(f"Train File Path: {ingestion_config.train_file_path}")
    print(f"Test File Path: {ingestion_config.test_file_path}")