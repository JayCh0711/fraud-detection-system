"""
Request Schemas for Fraud Detection API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class TransactionType(str, Enum):
    """Valid transaction types"""
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT"


class TransactionBase(BaseModel):
    """Base transaction schema"""
    
    step: Optional[int] = Field(
        default=0,
        ge=0,
        description="Time step of transaction (hour from simulation start)"
    )
    type: TransactionType = Field(
        ...,
        description="Type of transaction"
    )
    amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount"
    )
    name_orig: Optional[str] = Field(
        default="",
        description="Origin account ID"
    )
    old_balance_org: float = Field(
        ...,
        ge=0,
        description="Origin account balance before transaction"
    )
    new_balance_org: float = Field(
        ...,
        ge=0,
        description="Origin account balance after transaction"
    )
    name_dest: Optional[str] = Field(
        default="",
        description="Destination account ID"
    )
    old_balance_dest: float = Field(
        ...,
        ge=0,
        description="Destination account balance before transaction"
    )
    new_balance_dest: float = Field(
        ...,
        ge=0,
        description="Destination account balance after transaction"
    )
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "step": 100,
                "type": "TRANSFER",
                "amount": 50000.0,
                "name_orig": "C123456789",
                "old_balance_org": 100000.0,
                "new_balance_org": 50000.0,
                "name_dest": "C987654321",
                "old_balance_dest": 0.0,
                "new_balance_dest": 50000.0
            }
        }


class SingleTransactionRequest(TransactionBase):
    """Request schema for single transaction prediction"""
    
    transaction_id: Optional[str] = Field(
        default=None,
        description="Optional transaction ID for tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_001",
                "step": 100,
                "type": "TRANSFER",
                "amount": 50000.0,
                "name_orig": "C123456789",
                "old_balance_org": 100000.0,
                "new_balance_org": 50000.0,
                "name_dest": "C987654321",
                "old_balance_dest": 0.0,
                "new_balance_dest": 50000.0
            }
        }


class BatchTransactionRequest(BaseModel):
    """Request schema for batch prediction"""
    
    transactions: List[TransactionBase] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to predict"
    )
    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch ID for tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "BATCH_001",
                "transactions": [
                    {
                        "step": 100,
                        "type": "PAYMENT",
                        "amount": 1500.0,
                        "old_balance_org": 50000.0,
                        "new_balance_org": 48500.0,
                        "old_balance_dest": 10000.0,
                        "new_balance_dest": 11500.0
                    },
                    {
                        "step": 200,
                        "type": "TRANSFER",
                        "amount": 200000.0,
                        "old_balance_org": 200000.0,
                        "new_balance_org": 0.0,
                        "old_balance_dest": 0.0,
                        "new_balance_dest": 200000.0
                    }
                ]
            }
        }


class ThresholdUpdateRequest(BaseModel):
    """Request schema for updating prediction threshold"""
    
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="New threshold value (0-1)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "threshold": 0.45
            }
        }