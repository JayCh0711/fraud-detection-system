"""
Response Schemas for Fraud Detection API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Risk category levels"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskFactor(BaseModel):
    """Risk factor detail"""
    
    factor: str = Field(..., description="Risk factor name")
    description: str = Field(..., description="Risk factor description")
    severity: str = Field(..., description="Severity level (HIGH/MEDIUM/LOW)")


class PredictionResponse(BaseModel):
    """Response schema for single prediction"""
    
    success: bool = Field(default=True, description="Request success status")
    transaction_id: str = Field(..., description="Transaction identifier")
    
    # Prediction Results
    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    prediction: int = Field(..., description="Prediction label (1=Fraud, 0=Legitimate)")
    probability: float = Field(..., description="Fraud probability (0-1)")
    risk_score: float = Field(..., description="Risk score (0-100)")
    risk_category: RiskCategory = Field(..., description="Risk category")
    
    # Explainability
    risk_factors: List[RiskFactor] = Field(
        default=[],
        description="Top risk factors contributing to prediction"
    )
    
    # Metadata
    threshold_used: float = Field(..., description="Classification threshold used")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Prediction timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "transaction_id": "TXN_001",
                "is_fraud": True,
                "prediction": 1,
                "probability": 0.8934,
                "risk_score": 89.34,
                "risk_category": "HIGH",
                "risk_factors": [
                    {
                        "factor": "Account Drain",
                        "description": "Transaction drains entire account balance",
                        "severity": "HIGH"
                    }
                ],
                "threshold_used": 0.42,
                "model_version": "1.0.0",
                "processing_time_ms": 15.67,
                "timestamp": "2024-01-15T19:30:45.123456"
            }
        }


class BatchPredictionItem(BaseModel):
    """Single prediction within batch response"""
    
    index: int = Field(..., description="Transaction index in batch")
    prediction: int = Field(..., description="Prediction (1=Fraud, 0=Legitimate)")
    probability: float = Field(..., description="Fraud probability")
    risk_category: RiskCategory = Field(..., description="Risk category")
    is_fraud: bool = Field(..., description="Fraud flag")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction"""
    
    success: bool = Field(default=True, description="Request success status")
    batch_id: str = Field(..., description="Batch identifier")
    
    # Summary Statistics
    total_transactions: int = Field(..., description="Total transactions processed")
    fraud_count: int = Field(..., description="Number of frauds detected")
    legitimate_count: int = Field(..., description="Number of legitimate transactions")
    
    # Risk Distribution
    high_risk_count: int = Field(..., description="High risk transactions")
    medium_risk_count: int = Field(..., description="Medium risk transactions")
    low_risk_count: int = Field(..., description="Low risk transactions")
    minimal_risk_count: int = Field(..., description="Minimal risk transactions")
    
    # Fraud Rate
    fraud_rate: float = Field(..., description="Percentage of frauds in batch")
    
    # Individual Predictions
    predictions: List[BatchPredictionItem] = Field(
        ...,
        description="Individual prediction results"
    )
    
    # Metadata
    threshold_used: float = Field(..., description="Classification threshold used")
    model_version: str = Field(..., description="Model version used")
    processing_time_seconds: float = Field(..., description="Total processing time")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Batch prediction timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "batch_id": "BATCH_001",
                "total_transactions": 100,
                "fraud_count": 5,
                "legitimate_count": 95,
                "high_risk_count": 3,
                "medium_risk_count": 7,
                "low_risk_count": 10,
                "minimal_risk_count": 80,
                "fraud_rate": 5.0,
                "predictions": [
                    {
                        "index": 0,
                        "prediction": 0,
                        "probability": 0.0234,
                        "risk_category": "MINIMAL",
                        "is_fraud": False
                    }
                ],
                "threshold_used": 0.42,
                "model_version": "1.0.0",
                "processing_time_seconds": 1.25,
                "timestamp": "2024-01-15T19:30:45.123456"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check"""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Model load status")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Health check timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "Fraud Detection API",
                "version": "1.0.0",
                "model_loaded": True,
                "model_version": "1.0.0",
                "timestamp": "2024-01-15T19:30:45.123456"
            }
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""
    
    success: bool = Field(default=True, description="Request success status")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model algorithm type")
    optimal_threshold: float = Field(..., description="Optimal classification threshold")
    current_threshold: float = Field(..., description="Currently active threshold")
    num_features: int = Field(..., description="Number of input features")
    feature_names: List[str] = Field(..., description="Sample feature names (top 10)")
    
    # Performance Metrics (from training)
    metrics: Dict[str, float] = Field(
        ...,
        description="Model performance metrics"
    )
    
    # Risk Thresholds
    risk_thresholds: Dict[str, float] = Field(
        ...,
        description="Risk category thresholds"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Info request timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "model_version": "1.0.0",
                "model_type": "XGBClassifier",
                "optimal_threshold": 0.42,
                "current_threshold": 0.42,
                "num_features": 50,
                "feature_names": ["amount_log", "is_account_drain", "risk_score"],
                "metrics": {
                    "recall": 0.8967,
                    "precision": 0.7856,
                    "f1_score": 0.8375,
                    "roc_auc": 0.9534
                },
                "risk_thresholds": {
                    "high_risk": 0.7,
                    "medium_risk": 0.4,
                    "low_risk": 0.2
                },
                "timestamp": "2024-01-15T19:30:45.123456"
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    
    success: bool = Field(default=False, description="Request success status")
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Error timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error_code": "PREDICTION_ERROR",
                "error_message": "Failed to process transaction",
                "detail": "Model not loaded",
                "timestamp": "2024-01-15T19:30:45.123456"
            }
        }