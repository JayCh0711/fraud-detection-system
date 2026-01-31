import pytest
import os
import sys
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from src.pipeline.prediction_pipeline import PredictionPipeline

@pytest.fixture
def client():
    """FastAPI Test Client"""
    return TestClient(app)

@pytest.fixture
def api_key():
    """Mock API Key"""
    return "fraud-detection-api-key-2024"

@pytest.fixture
def sample_transaction():
    """Sample transaction data"""
    return {
        "transaction_id": "TEST_TXN_001",
        "step": 100,
        "type": "TRANSFER",
        "amount": 50000.0,
        "name_orig": "C12345",
        "old_balance_org": 50000.0,
        "new_balance_org": 0.0,
        "name_dest": "M12345",
        "old_balance_dest": 0.0,
        "new_balance_dest": 50000.0
    }