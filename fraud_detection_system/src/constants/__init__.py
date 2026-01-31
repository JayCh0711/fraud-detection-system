"""
Constants for Fraud Detection System
All paths and constant values defined here
"""

import os
from pathlib import Path

# ============== ROOT DIRECTORY ==============
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ============== CONFIG PATHS ==============
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "config.yaml")
MODEL_CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "model_config.yaml")
SCHEMA_FILE_PATH = os.path.join(CONFIG_DIR, "schema.yaml")

# ============== ARTIFACTS PATHS ==============
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
DATA_INGESTION_DIR = os.path.join(ARTIFACTS_DIR, "data_ingestion")
DATA_VALIDATION_DIR = os.path.join(ARTIFACTS_DIR, "data_validation")
DATA_TRANSFORMATION_DIR = os.path.join(ARTIFACTS_DIR, "data_transformation")
MODEL_TRAINER_DIR = os.path.join(ARTIFACTS_DIR, "model_trainer")
MODEL_EVALUATION_DIR = os.path.join(ARTIFACTS_DIR, "model_evaluation")
MODEL_REGISTRY_DIR = os.path.join(ARTIFACTS_DIR, "model_registry")
PRODUCTION_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "production_model")

# ============== DATA PATHS ==============
RAW_DATA_DIR = os.path.join(DATA_INGESTION_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_INGESTION_DIR, "processed")
TRAIN_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")

# ============== MODEL PATHS ==============
MODEL_FILE_NAME = "model.pkl"
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"
ENCODER_FILE_NAME = "encoder.pkl"

# ============== LOG PATHS ==============
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# ============== PIPELINE CONSTANTS ==============
TARGET_COLUMN = "is_fraud"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============== DATABASE CONSTANTS ==============
DATABASE_NAME = "fraud_detection"
COLLECTION_NAME = "transactions"

# ============== API CONSTANTS ==============
APP_HOST = "0.0.0.0"
APP_PORT = 8000