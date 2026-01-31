import pytest
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig

def test_data_ingestion_config():
    config = DataIngestionConfig()
    assert config.source_type == "csv"
    assert "train.csv" in config.train_file_name

def test_data_validation_schema():
    from src.utils.common import read_yaml
    from src.constants import SCHEMA_FILE_PATH
    from pathlib import Path
    
    schema = read_yaml(Path(SCHEMA_FILE_PATH))
    assert "is_fraud" in schema.columns
    assert schema.target_column == "is_fraud"