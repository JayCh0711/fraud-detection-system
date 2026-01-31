"""
Model Information Routes for Fraud Detection API
"""

from fastapi import APIRouter, HTTPException, Depends
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from datetime import datetime

from src.api.schemas.response import ModelInfoResponse, ErrorResponse
from src.api.middleware.auth import verify_api_key
from src.api.routes.prediction import get_prediction_pipeline
from src.api.core.config import settings
from src.utils.common import read_json
from src.logger import logger


# Create router
router = APIRouter(prefix="/model", tags=["Model"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    responses={
        200: {"description": "Model information"},
        500: {"model": ErrorResponse, "description": "Error getting model info"}
    },
    summary="Get Model Information",
    description="Get information about the loaded model"
)
async def get_model_info(
    api_key: str = Depends(verify_api_key)
) -> ModelInfoResponse:
    """
    Get information about the loaded model.
    
    Args:
        api_key: Valid API key
    
    Returns:
        Model information
    """
    try:
        pipeline = get_prediction_pipeline()
        
        # Get model info from pipeline
        model_info = pipeline.get_model_info()
        
        # Try to get metrics from version file
        metrics = {}
        try:
            version_info = read_json(pipeline.config.version_path)
            metrics = version_info.get('metrics', {})
        except Exception:
            metrics = {
                "recall": 0.0,
                "precision": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.0
            }
        
        # Risk thresholds
        risk_thresholds = {
            "high_risk": pipeline.config.high_risk_threshold,
            "medium_risk": pipeline.config.medium_risk_threshold,
            "low_risk": pipeline.config.low_risk_threshold
        }
        
        return ModelInfoResponse(
            success=True,
            model_version=model_info.get('model_version', 'unknown'),
            model_type=model_info.get('model_type', 'unknown'),
            optimal_threshold=model_info.get('optimal_threshold', 0.5),
            current_threshold=pipeline.optimal_threshold,
            num_features=model_info.get('num_features', 0),
            feature_names=model_info.get('feature_names', [])[:10],
            metrics=metrics,
            risk_thresholds=risk_thresholds,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post(
    "/reload",
    response_model=dict,
    summary="Reload Model",
    description="Reload the production model"
)
async def reload_model(
    api_key: str = Depends(verify_api_key)
) -> dict:
    """
    Reload the production model.
    
    Args:
        api_key: Valid API key
    
    Returns:
        Reload status
    """
    try:
        # Re-initialize pipeline
        global _prediction_pipeline
        from src.api.routes.prediction import _prediction_pipeline
        
        pipeline = get_prediction_pipeline()
        old_version = pipeline.model_version
        
        # Force reload
        pipeline._is_initialized = False
        success = pipeline.initialize()
        
        if not success:
            raise Exception("Failed to reload model")
        
        return {
            "success": True,
            "message": "Model reloaded successfully",
            "old_version": old_version,
            "new_version": pipeline.model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


@router.get(
    "/features",
    response_model=dict,
    summary="Get Feature Names",
    description="Get all feature names used by the model"
)
async def get_features(
    api_key: str = Depends(verify_api_key)
) -> dict:
    """
    Get all feature names.
    
    Args:
        api_key: Valid API key
    
    Returns:
        Feature names
    """
    try:
        pipeline = get_prediction_pipeline()
        
        return {
            "success": True,
            "num_features": len(pipeline.feature_names),
            "feature_names": pipeline.feature_names,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get features: {str(e)}"
        )