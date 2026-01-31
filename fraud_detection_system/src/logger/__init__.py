"""
Custom Logger for Fraud Detection System
- Console logging with colors
- File logging with rotation
- Structured logging format
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from src.constants import LOGS_DIR


# ============== CREATE LOGS DIRECTORY ==============
os.makedirs(LOGS_DIR, exist_ok=True)

# ============== LOG FILE NAME WITH TIMESTAMP ==============
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

# ============== LOG FORMAT ==============
LOG_FORMAT = (
    "[ %(asctime)s ] "
    "%(levelname)s "
    "%(filename)s:%(lineno)d "
    "%(funcName)s() - "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============== CUSTOM FORMATTER WITH COLORS ==============
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[95m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}"
                f"{levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


# ============== LOGGER SETUP FUNCTION ==============
def setup_logger(
    name: str = "FraudDetection",
    level: int = logging.DEBUG,
    log_to_console: bool = True,
    log_to_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup and return a configured logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_to_console: Enable console logging
        log_to_file: Enable file logging
        max_file_size: Max size per log file (bytes)
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # ---------- FILE HANDLER (Rotating) ----------
    if log_to_file:
        file_handler = RotatingFileHandler(
            filename=LOG_FILE_PATH,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        )
        logger.addHandler(file_handler)
    
    # ---------- CONSOLE HANDLER (Colored) ----------
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        )
        logger.addHandler(console_handler)
    
    return logger


# ============== DEFAULT LOGGER INSTANCE ==============
logger = setup_logger()


# ============== UTILITY FUNCTIONS ==============
def log_function_entry_exit(func):
    """Decorator to log function entry and exit"""
    def wrapper(*args, **kwargs):
        logger.info(f"Entering: {func.__name__}()")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Exiting: {func.__name__}() - Success")
            return result
        except Exception as e:
            logger.error(f"Exiting: {func.__name__}() - Failed with {e}")
            raise
    return wrapper


def log_dataframe_info(df, name: str = "DataFrame"):
    """Log DataFrame information"""
    logger.info(f"{'='*50}")
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Missing Values: {df.isnull().sum().sum()}")
    logger.info(f"{'='*50}")


def log_model_metrics(metrics: dict, model_name: str = "Model"):
    """Log model evaluation metrics"""
    logger.info(f"{'='*50}")
    logger.info(f"{model_name} Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"{'='*50}")


# ============== EXAMPLE USAGE ==============
if __name__ == "__main__":
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")