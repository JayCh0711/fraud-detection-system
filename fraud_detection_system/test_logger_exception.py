"""Test Logger and Exception Handler"""

import sys
from src.logger import logger, log_function_entry_exit
from src.exception import (
    FraudDetectionException,
    DataIngestionException,
    handle_exceptions
)


# Test Logger
print("\n" + "="*60)
print("TESTING LOGGER")
print("="*60 + "\n")

logger.debug("Debug message - for detailed debugging")
logger.info("Info message - general information")
logger.warning("Warning message - something to watch")
logger.error("Error message - something went wrong")
logger.critical("Critical message - system failure")


# Test Decorator
print("\n" + "="*60)
print("TESTING FUNCTION DECORATOR")
print("="*60 + "\n")

@log_function_entry_exit
def sample_function(x, y):
    """Sample function to test decorator"""
    return x + y

result = sample_function(5, 10)
print(f"Result: {result}")


# Test Exception
print("\n" + "="*60)
print("TESTING EXCEPTION HANDLER")
print("="*60 + "\n")

try:
    # Simulate an error
    raise DataIngestionException(
        error_message="Failed to connect to database",
        error_detail=sys
    )
except DataIngestionException as e:
    print("Exception caught successfully!")
    print(e)


# Test Exception Decorator
print("\n" + "="*60)
print("TESTING EXCEPTION DECORATOR")
print("="*60 + "\n")

@handle_exceptions
def divide_numbers(a, b):
    return a / b

try:
    divide_numbers(10, 0)
except FraudDetectionException as e:
    print("Exception caught via decorator!")
    print(e)

print("\nAll tests passed!")