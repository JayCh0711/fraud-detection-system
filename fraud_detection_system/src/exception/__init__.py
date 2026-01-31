"""
Custom Exception Handler for Fraud Detection System
- Detailed error information
- File name, line number, function name
- Formatted error messages
"""

import sys
import traceback
from typing import Optional, Any

from src.logger import logger


# ============== CUSTOM EXCEPTION CLASS ==============
class FraudDetectionException(Exception):
    """
    Custom Exception for Fraud Detection System.
    Captures detailed error information including:
    - File name
    - Line number
    - Function name
    - Error message
    """
    
    def __init__(
        self,
        error_message: str,
        error_detail: Optional[Any] = None
    ):
        super().__init__(error_message)
        
        if error_detail is None:
            error_detail = sys
        
        self.error_message = self._get_detailed_error_message(
            error_message=error_message,
            error_detail=error_detail
        )
        
        # Log the exception
        logger.error(self.error_message)
    
    def _get_detailed_error_message(
        self,
        error_message: str,
        error_detail: sys
    ) -> str:
        """
        Extract detailed error information from traceback.
        """
        _, _, exc_tb = error_detail.exc_info()
        
        if exc_tb is None:
            return f"Error: {error_message}"
        
        # Get traceback details
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        function_name = exc_tb.tb_frame.f_code.co_name
        
        detailed_message = (
            f"\n{'='*60}\n"
            f"EXCEPTION OCCURRED\n"
            f"{'='*60}\n"
            f"File        : {file_name}\n"
            f"Line Number : {line_number}\n"
            f"Function    : {function_name}()\n"
            f"Error       : {error_message}\n"
            f"{'='*60}\n"
        )
        
        return detailed_message
    
    def __str__(self):
        return self.error_message


# ============== SPECIFIC EXCEPTION CLASSES ==============

class DataIngestionException(FraudDetectionException):
    """Exception raised during Data Ingestion phase"""
    pass


class DataValidationException(FraudDetectionException):
    """Exception raised during Data Validation phase"""
    pass


class DataTransformationException(FraudDetectionException):
    """Exception raised during Data Transformation phase"""
    pass


class FeatureEngineeringException(FraudDetectionException):
    """Exception raised during Feature Engineering phase"""
    pass


class ModelTrainerException(FraudDetectionException):
    """Exception raised during Model Training phase"""
    pass


class ModelEvaluationException(FraudDetectionException):
    """Exception raised during Model Evaluation phase"""
    pass


class PredictionException(FraudDetectionException):
    """Exception raised during Prediction phase"""
    pass


class DatabaseException(FraudDetectionException):
    """Exception raised during Database operations"""
    pass


class ConfigurationException(FraudDetectionException):
    """Exception raised for Configuration errors"""
    pass


# ============== EXCEPTION HANDLER DECORATOR ==============
def handle_exceptions(func):
    """
    Decorator to handle exceptions in functions.
    Catches exceptions, logs them, and raises FraudDetectionException.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FraudDetectionException:
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            raise FraudDetectionException(
                error_message=f"{func.__name__}() failed: {str(e)}",
                error_detail=sys
            ) from e
    return wrapper


# ============== ERROR RESPONSE FORMATTER (For API) ==============
def format_error_response(
    exception: Exception,
    include_traceback: bool = False
) -> dict:
    """
    Format exception for API response.
    
    Args:
        exception: The exception to format
        include_traceback: Include full traceback (for debugging)
    
    Returns:
        Formatted error dictionary
    """
    error_response = {
        "success": False,
        "error_type": type(exception).__name__,
        "error_message": str(exception),
    }
    
    if include_traceback:
        error_response["traceback"] = traceback.format_exc()
    
    return error_response


# ============== EXAMPLE USAGE ==============
if __name__ == "__main__":
    
    # Test basic exception
    try:
        raise FraudDetectionException(
            error_message="Test exception message",
            error_detail=sys
        )
    except FraudDetectionException as e:
        print(e)
    
    # Test specific exception
    try:
        raise DataIngestionException(
            error_message="Failed to load data from source",
            error_detail=sys
        )
    except DataIngestionException as e:
        print(e)
    
    # Test decorator
    @handle_exceptions
    def test_function():
        return 1 / 0  # This will raise ZeroDivisionError
    
    try:
        test_function()
    except FraudDetectionException as e:
        print(e)