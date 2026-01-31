"""
Complete Training Pipeline for Fraud Detection System

Orchestrates all training components:
1. Data Ingestion
2. Data Validation
3. EDA
4. Feature Engineering
5. Data Transformation
6. Model Training
7. Model Evaluation
8. Model Registry/Pusher
"""

import os
import sys
from typing import Dict, Optional
from dataclasses import dataclass
import time

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    FeatureEngineeringArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_eda import DataEDA, EDAConfig
from src.components.feature_engineering import FeatureEngineering
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.logger import logger
from src.exception import FraudDetectionException


@dataclass
class TrainingPipelineConfig:
    """Configuration for Training Pipeline"""
    
    # Pipeline Settings
    pipeline_name: str = "fraud_detection_training"
    run_eda: bool = True
    enable_hyperparameter_tuning: bool = True
    n_tuning_trials: int = 30
    
    # Data Source
    data_source_type: str = "csv"
    data_source_path: str = "data/raw/transactions.csv"


class TrainingPipeline:
    """
    Complete Training Pipeline for Fraud Detection.
    
    Orchestrates all components from data ingestion to model deployment.
    """
    
    def __init__(self, config: TrainingPipelineConfig = TrainingPipelineConfig()):
        """
        Initialize Training Pipeline.
        
        Args:
            config: TrainingPipelineConfig object
        """
        self.config = config
        self.artifacts: Dict = {}
        
        logger.info(f"{'='*60}")
        logger.info(f"Initializing Training Pipeline: {config.pipeline_name}")
        logger.info(f"{'='*60}")
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """Execute data ingestion step."""
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("="*60)
        
        ingestion_config = DataIngestionConfig(
            source_type=self.config.data_source_type,
            source_path=self.config.data_source_path
        )
        
        data_ingestion = DataIngestion(config=ingestion_config)
        artifact = data_ingestion.initiate_data_ingestion()
        
        self.artifacts['data_ingestion'] = artifact
        return artifact
    
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Execute data validation step."""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATA VALIDATION")
        logger.info("="*60)
        
        validation_config = DataValidationConfig()
        
        data_validation = DataValidation(
            config=validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        artifact = data_validation.initiate_data_validation()
        
        self.artifacts['data_validation'] = artifact
        return artifact
    
    def start_eda(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> Optional[Dict]:
        """Execute EDA step (optional)."""
        if not self.config.run_eda:
            logger.info("EDA step skipped (disabled in config)")
            return None
        
        logger.info("\n" + "="*60)
        logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")
        logger.info("="*60)
        
        eda_config = EDAConfig()
        
        eda = DataEDA(
            config=eda_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        artifact = eda.initiate_eda()
        
        self.artifacts['eda'] = artifact
        return artifact
    
    def start_feature_engineering(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> FeatureEngineeringArtifact:
        """Execute feature engineering step."""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: FEATURE ENGINEERING")
        logger.info("="*60)
        
        fe_config = FeatureEngineeringConfig()
        
        feature_engineering = FeatureEngineering(
            config=fe_config,
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact
        )
        artifact = feature_engineering.initiate_feature_engineering()
        
        self.artifacts['feature_engineering'] = artifact
        return artifact
    
    def start_data_transformation(
        self,
        feature_engineering_artifact: FeatureEngineeringArtifact
    ) -> DataTransformationArtifact:
        """Execute data transformation step."""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: DATA TRANSFORMATION")
        logger.info("="*60)
        
        transformation_config = DataTransformationConfig()
        
        data_transformation = DataTransformation(
            config=transformation_config,
            feature_engineering_artifact=feature_engineering_artifact
        )
        artifact = data_transformation.initiate_data_transformation()
        
        self.artifacts['data_transformation'] = artifact
        return artifact
    
    def start_model_training(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """Execute model training step."""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: MODEL TRAINING")
        logger.info("="*60)
        
        trainer_config = ModelTrainerConfig(
            enable_hyperparameter_tuning=self.config.enable_hyperparameter_tuning,
            n_trials=self.config.n_tuning_trials
        )
        
        model_trainer = ModelTrainer(
            config=trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        artifact = model_trainer.initiate_model_training()
        
        self.artifacts['model_trainer'] = artifact
        return artifact
    
    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        """Execute model evaluation step."""
        logger.info("\n" + "="*60)
        logger.info("STEP 7: MODEL EVALUATION")
        logger.info("="*60)
        
        evaluation_config = ModelEvaluationConfig()
        
        model_evaluation = ModelEvaluation(
            config=evaluation_config,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact
        )
        artifact = model_evaluation.initiate_model_evaluation()
        
        self.artifacts['model_evaluation'] = artifact
        return artifact
    
    def start_model_pusher(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:
        """Execute model pusher step."""
        logger.info("\n" + "="*60)
        logger.info("STEP 8: MODEL REGISTRY & DEPLOYMENT")
        logger.info("="*60)
        
        pusher_config = ModelPusherConfig()
        
        model_pusher = ModelPusher(
            config=pusher_config,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact,
            model_evaluation_artifact=model_evaluation_artifact
        )
        artifact = model_pusher.initiate_model_pusher()
        
        self.artifacts['model_pusher'] = artifact
        return artifact
    
    def run_pipeline(self) -> Dict:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary of all artifacts
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info("STARTING COMPLETE TRAINING PIPELINE")
        logger.info(f"{'='*60}\n")
        
        try:
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Data Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )
            
            # Check if validation passed
            if not data_validation_artifact.is_validated:
                logger.error("Data validation failed. Stopping pipeline.")
                raise FraudDetectionException(
                    error_message="Data validation failed",
                    error_detail=sys
                )
            
            # Step 3: EDA (Optional)
            self.start_eda(data_ingestion_artifact)
            
            # Step 4: Feature Engineering
            feature_engineering_artifact = self.start_feature_engineering(
                data_ingestion_artifact,
                data_validation_artifact
            )
            
            # Step 5: Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                feature_engineering_artifact
            )
            
            # Step 6: Model Training
            model_trainer_artifact = self.start_model_training(
                data_transformation_artifact
            )
            
            # Step 7: Model Evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact,
                model_trainer_artifact
            )
            
            # Step 8: Model Pusher
            model_pusher_artifact = self.start_model_pusher(
                data_transformation_artifact,
                model_trainer_artifact,
                model_evaluation_artifact
            )
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"{'='*60}")
            logger.info(f"\nTotal Pipeline Duration: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            
            logger.info(f"\n{'='*40}")
            logger.info("PIPELINE SUMMARY")
            logger.info(f"{'='*40}")
            logger.info(f"Data Records: {data_ingestion_artifact.total_records:,}")
            logger.info(f"Features Engineered: {feature_engineering_artifact.engineered_feature_count}")
            logger.info(f"Final Features: {len(data_transformation_artifact.selected_features)}")
            logger.info(f"Best Model: {model_trainer_artifact.best_model_name}")
            logger.info(f"Test Recall: {model_evaluation_artifact.recall:.4f}")
            logger.info(f"Test Precision: {model_evaluation_artifact.precision:.4f}")
            logger.info(f"Test F1-Score: {model_evaluation_artifact.f1_score:.4f}")
            logger.info(f"Model Version: {model_pusher_artifact.model_version}")
            logger.info(f"Promoted to Production: {model_pusher_artifact.is_model_promoted}")
            
            return self.artifacts
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise FraudDetectionException(
                error_message=f"Training pipeline failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    # Run complete training pipeline
    pipeline_config = TrainingPipelineConfig(
        pipeline_name="fraud_detection_v1",
        run_eda=True,
        enable_hyperparameter_tuning=False,  # Set to True for full training
        n_tuning_trials=10,
        data_source_type="csv",
        data_source_path="data/raw/transactions.csv"
    )
    
    pipeline = TrainingPipeline(config=pipeline_config)
    artifacts = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("ALL ARTIFACTS")
    print("="*60)
    for name, artifact in artifacts.items():
        print(f"\n{name}:")
        print(f"  Type: {type(artifact).__name__}")