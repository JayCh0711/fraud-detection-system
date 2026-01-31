"""
Prediction Routes for Fraud Detection API
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from datetime import datetime

from src.api.schemas.transaction import (
    SingleTransactionRequest,
    BatchTransactionRequest,
    ThresholdUpdateRequest
)
from src.api.schemas.response import (
    PredictionResponse,
    BatchPredictionResponse,
    BatchPredictionItem,
    RiskFactor,
    ErrorResponse
)
from src.api.middleware.auth import verify_api_key
from src.api.core.config import settings
from src.pipeline.prediction_pipeline import PredictionPipeline, PredictionConfig
from src.logger import logger


# Create router
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Initialize prediction pipeline (singleton)
_prediction_pipeline: Optional[PredictionPipeline] = None


def get_prediction_pipeline() -> PredictionPipeline:
    """
    Get or initialize prediction pipeline (singleton).
    
    Returns:
        Initialized PredictionPipeline
    """
    global _prediction_pipeline
    
    if _prediction_pipeline is None:
        _prediction_pipeline = PredictionPipeline()
        success = _prediction_pipeline.initialize()
        
        if not success:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize prediction pipeline"
            )
    
    return _prediction_pipeline


@router.post(
    "/single",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    },
    summary="Predict Single Transaction",
    description="Predict fraud probability for a single transaction"
)
async def predict_single(
    request: SingleTransactionRequest,
    api_key: str = Depends(verify_api_key)
) -> PredictionResponse:
    """
    Predict fraud for a single transaction.
    
    Args:
        request: Transaction data
        api_key: Valid API key
    
    Returns:
        Prediction result
    """
    try:
        # Get pipeline
        pipeline = get_prediction_pipeline()
        
        # Convert request to dict
        transaction_data = {
            'step': request.step,
            'type': request.type.value,
            'amount': request.amount,
            'name_orig': request.name_orig or '',
            'old_balance_org': request.old_balance_org,
            'new_balance_org': request.new_balance_org,
            'name_dest': request.name_dest or '',
            'old_balance_dest': request.old_balance_dest,
            'new_balance_dest': request.new_balance_dest
        }
        
        # Get prediction
        result = pipeline.predict_single(
            transaction=transaction_data,
            transaction_id=request.transaction_id
        )
        
        # Convert risk factors
        risk_factors = [
            RiskFactor(
                factor=rf['factor'],
                description=rf['description'],
                severity=rf['severity']
            )
            for rf in result.top_risk_factors
        ]
        
        # Create response
        response = PredictionResponse(
            success=True,
            transaction_id=result.transaction_id,
            is_fraud=result.is_fraud,
            prediction=result.prediction,
            probability=result.probability,
            risk_score=result.risk_score,
            risk_category=result.risk_category,
            risk_factors=risk_factors,
            threshold_used=result.threshold_used,
            model_version=result.model_version,
            processing_time_ms=result.processing_time_ms,
            timestamp=result.prediction_timestamp
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Prediction error"}
    },
    summary="Predict Batch Transactions",
    description="Predict fraud probability for multiple transactions"
)
async def predict_batch(
    request: BatchTransactionRequest,
    api_key: str = Depends(verify_api_key)
) -> BatchPredictionResponse:
    """
    Predict fraud for a batch of transactions.
    
    Args:
        request: Batch transaction data
        api_key: Valid API key
    
    Returns:
        Batch prediction results
    """
    try:
        # Validate batch size
        if len(request.transactions) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Batch size exceeds maximum of {settings.MAX_BATCH_SIZE}"
            )
        
        # Get pipeline
        pipeline = get_prediction_pipeline()
        
        # Convert requests to list of dicts
        transactions_data = []
        for txn in request.transactions:
            transactions_data.append({
                'step': txn.step,
                'type': txn.type.value,
                'amount': txn.amount,
                'name_orig': txn.name_orig or '',
                'old_balance_org': txn.old_balance_org,
                'new_balance_org': txn.new_balance_org,
                'name_dest': txn.name_dest or '',
                'old_balance_dest': txn.old_balance_dest,
                'new_balance_dest': txn.new_balance_dest
            })
        
        # Get batch prediction
        result = pipeline.predict_batch(
            transactions=transactions_data,
            batch_id=request.batch_id,
            save_results=True
        )
        
        # Create prediction items
        predictions = []
        for idx, row in result.predictions_df.iterrows():
            predictions.append(
                BatchPredictionItem(
                    index=int(idx),
                    prediction=int(row['prediction']),
                    probability=round(float(row['fraud_probability']), 4),
                    risk_category=row['risk_category'],
                    is_fraud=bool(row['is_fraud'])
                )
            )
        
        # Calculate minimal risk count
        minimal_risk = result.total_transactions - (
            result.high_risk_count + 
            result.medium_risk_count + 
            result.low_risk_count
        )
        
        # Fraud rate
        fraud_rate = (result.fraud_count / result.total_transactions * 100) if result.total_transactions > 0 else 0
        
        # Create response
        response = BatchPredictionResponse(
            success=True,
            batch_id=result.batch_id,
            total_transactions=result.total_transactions,
            fraud_count=result.fraud_count,
            legitimate_count=result.legitimate_count,
            high_risk_count=result.high_risk_count,
            medium_risk_count=result.medium_risk_count,
            low_risk_count=result.low_risk_count,
            minimal_risk_count=minimal_risk,
            fraud_rate=round(fraud_rate, 2),
            predictions=predictions,
            threshold_used=result.threshold_used,
            model_version=result.model_version,
            processing_time_seconds=result.processing_time_seconds,
            timestamp=result.batch_timestamp
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post(
    "/threshold",
    response_model=dict,
    summary="Update Prediction Threshold",
    description="Update the classification threshold for predictions"
)
async def update_threshold(
    request: ThresholdUpdateRequest,
    api_key: str = Depends(verify_api_key)
) -> dict:
    """
    Update the prediction threshold.
    
    Args:
        request: New threshold value
        api_key: Valid API key
    
    Returns:
        Updated threshold info
    """
    try:
        pipeline = get_prediction_pipeline()
        
        old_threshold = pipeline.optimal_threshold
        pipeline.set_threshold(request.threshold)
        
        return {
            "success": True,
            "message": "Threshold updated successfully",
            "old_threshold": old_threshold,
            "new_threshold": request.threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update threshold: {str(e)}"
        )


@router.post(
    "/threshold/reset",
    response_model=dict,
    summary="Reset Prediction Threshold",
    description="Reset threshold to model's optimal value"
)
async def reset_threshold(
    api_key: str = Depends(verify_api_key)
) -> dict:
    """
    Reset threshold to optimal value.
    
    Args:
        api_key: Valid API key
    
    Returns:
        Reset threshold info
    """
    try:
        pipeline = get_prediction_pipeline()
        
        old_threshold = pipeline.optimal_threshold
        pipeline.reset_threshold()
        
        return {
            "success": True,
            "message": "Threshold reset to optimal value",
            "old_threshold": old_threshold,
            "new_threshold": pipeline.optimal_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset threshold: {str(e)}"
        )