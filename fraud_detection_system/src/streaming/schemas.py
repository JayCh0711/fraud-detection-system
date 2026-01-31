"""
Message Schemas for Kafka Streaming
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class TransactionType(str, Enum):
    """Transaction types"""
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT"


class RiskLevel(str, Enum):
    """Risk levels"""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TransactionMessage:
    """
    Incoming transaction message schema.
    """
    
    # Transaction Identifiers
    transaction_id: str
    
    # Transaction Details
    step: int = 0
    type: str = "PAYMENT"
    amount: float = 0.0
    
    # Origin Account
    name_orig: str = ""
    old_balance_org: float = 0.0
    new_balance_org: float = 0.0
    
    # Destination Account
    name_dest: str = ""
    old_balance_dest: float = 0.0
    new_balance_dest: float = 0.0
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source_system: str = "unknown"
    channel: str = "unknown"  # mobile, web, atm, branch
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransactionMessage':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TransactionMessage':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class PredictionMessage:
    """
    Prediction result message schema.
    """
    
    # Identifiers
    transaction_id: str
    prediction_id: str = ""
    
    # Prediction Results
    is_fraud: bool = False
    prediction: int = 0
    probability: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "MINIMAL"
    
    # Thresholds
    threshold_used: float = 0.5
    
    # Model Info
    model_version: str = ""
    
    # Risk Factors
    risk_factors: List[Dict] = field(default_factory=list)
    
    # Original Transaction (for reference)
    transaction_amount: float = 0.0
    transaction_type: str = ""
    
    # Timing
    processing_time_ms: float = 0.0
    transaction_timestamp: str = ""
    prediction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Metadata
    source_topic: str = ""
    partition: int = 0
    offset: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PredictionMessage':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AlertMessage:
    """
    Fraud alert message schema.
    """
    
    # Alert Info
    alert_id: str
    alert_type: str = "FRAUD_DETECTED"  # FRAUD_DETECTED, HIGH_RISK, SUSPICIOUS
    severity: str = "HIGH"  # LOW, MEDIUM, HIGH, CRITICAL
    
    # Transaction Info
    transaction_id: str = ""
    transaction_amount: float = 0.0
    transaction_type: str = ""
    
    # Accounts
    origin_account: str = ""
    destination_account: str = ""
    
    # Prediction Details
    fraud_probability: float = 0.0
    risk_score: float = 0.0
    risk_factors: List[Dict] = field(default_factory=list)
    
    # Recommended Actions
    recommended_action: str = "BLOCK_TRANSACTION"
    requires_manual_review: bool = True
    
    # Timestamps
    transaction_timestamp: str = ""
    alert_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Metadata
    model_version: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class DLQMessage:
    """
    Dead Letter Queue message schema.
    """
    
    # Original Message
    original_message: str
    original_topic: str
    
    # Error Info
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    
    # Processing Info
    retry_count: int = 0
    first_failure_timestamp: str = ""
    last_failure_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Metadata
    partition: int = 0
    offset: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())