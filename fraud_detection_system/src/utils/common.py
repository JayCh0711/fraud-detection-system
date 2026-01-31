"""
Common Utility Functions for Fraud Detection System
- YAML file handling
- Pickle operations
- Directory management
- Configuration loading
"""

import os
import sys
import yaml
import json
import pickle
import joblib
from pathlib import Path
from typing import Any, Dict, List, Union
# Optional utilities: provide graceful fallbacks if dev packages are missing
try:
    from box import ConfigBox
except Exception:
    # Fallback: simple wrapper to mimic ConfigBox access via dict
    class ConfigBox(dict):
        def __getattr__(self, item):
            return self.get(item)

try:
    from ensure import ensure_annotations
except Exception:
    # Fallback no-op decorator when 'ensure' isn't installed
    def ensure_annotations(func):
        return func

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None

from src.logger import logger
from src.exception import FraudDetectionException, ConfigurationException

# Load environment variables
load_dotenv()


# ============== YAML OPERATIONS ==============
def read_yaml(file_path) -> ConfigBox:
    """
    Read YAML file and return ConfigBox object. Accepts string path or pathlib.Path.
    
    Args:
        file_path: Path or string to YAML file
    
    Returns:
        ConfigBox: Configuration object with dot notation access
    """
    try:
        file_path = Path(file_path)
        with open(file_path, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {file_path}")
            return ConfigBox(content)
    except Exception as e:
        raise ConfigurationException(
            error_message=f"Error reading YAML file: {file_path} - {str(e)}",
            error_detail=sys
        ) from e


def write_yaml(file_path, data: dict) -> None:
    """
    Write data to YAML file. Accepts string path or pathlib.Path.
    
    Args:
        file_path: Path or string to save YAML file
        data: Dictionary to save
    """
    try:
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
        logger.info(f"YAML file saved: {file_path}")
    except Exception as e:
        raise FraudDetectionException(
            error_message=f"Error writing YAML file: {file_path} - {str(e)}",
            error_detail=sys
        ) from e


# ============== JSON OPERATIONS ==============
def read_json(file_path) -> dict:
    """Read JSON file and return dictionary. Accepts string path or pathlib.Path."""
    try:
        file_path = Path(file_path)
        with open(file_path, 'r') as json_file:
            content = json.load(json_file)
            logger.info(f"JSON file loaded: {file_path}")
            return content
    except Exception as e:
        raise FraudDetectionException(
            error_message=f"Error reading JSON file: {file_path} - {str(e)}",
            error_detail=sys
        ) from e


def _sanitize_for_json(obj):
    """Recursively convert objects to JSON-serializable structures.

    - dict keys are converted to strings (JSON requires string keys).
    - numpy scalars/arrays converted to native Python types/lists.
    - pandas Series/Index converted to lists; timestamps to strings.
    - Other unknown types are converted with str(obj) as a fallback.
    """
    # Import optional heavy deps lazily
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import pandas as _pd
    except Exception:
        _pd = None

    def _sanitize(o):
        # Dictionaries: ensure keys are strings and sanitize values
        if isinstance(o, dict):
            new = {}
            for k, v in o.items():
                # Prefer native types for keys where possible
                if isinstance(k, (str, int, float, bool, type(None))):
                    key = k
                else:
                    try:
                        if _np is not None and isinstance(k, _np.generic):
                            key = k.item()
                        else:
                            key = str(k)
                    except Exception:
                        key = str(k)
                # JSON requires string keys
                key = str(key)
                new[key] = _sanitize(v)
            return new

        # Lists/tuples/sets -> lists
        if isinstance(o, (list, tuple, set)):
            return [_sanitize(i) for i in o]

        # Numpy scalars
        if _np is not None:
            if isinstance(o, _np.generic):
                try:
                    return o.item()
                except Exception:
                    return str(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()

        # Pandas objects
        if _pd is not None:
            if isinstance(o, (_pd.Series, _pd.Index)):
                return o.tolist()
            if isinstance(o, (_pd.Timestamp, _pd.Timedelta)):
                return str(o)

        # Fallback: leave basic types as-is; convert unknowns to str
        if isinstance(o, (str, int, float, bool, type(None))):
            return o

        return str(o)

    return _sanitize(obj)


def write_json(file_path, data: dict) -> None:
    """Write dictionary to JSON file. Accepts string path or pathlib.Path."""
    try:
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)

        # Sanitize data to ensure JSON compatibility (keys must be strings)
        sanitized = _sanitize_for_json(data)

        with open(file_path, 'w') as json_file:
            # Use default=str to make any remaining non-serializable values safe
            json.dump(sanitized, json_file, indent=4, default=str)
        logger.info(f"JSON file saved: {file_path}")
    except Exception as e:
        raise FraudDetectionException(
            error_message=f"Error writing JSON file: {file_path} - {str(e)}",
            error_detail=sys
        ) from e


# ============== PICKLE/JOBLIB OPERATIONS ==============
def save_object(file_path, obj: Any) -> None:
    """
    Save object using joblib (for sklearn models/preprocessors). Accepts string or pathlib.Path.
    
    Args:
        file_path: Path or string to save object
        obj: Object to save
    """
    try:
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Object saved: {file_path}")
    except Exception as e:
        raise FraudDetectionException(
            error_message=f"Error saving object: {file_path} - {str(e)}",
            error_detail=sys
        ) from e


def load_object(file_path) -> Any:
    """
    Load object using joblib. Accepts string or pathlib.Path.
    
    Args:
        file_path: Path or string to load object from
    
    Returns:
        Loaded object
    """
    try:
        file_path = Path(file_path)
        obj = joblib.load(file_path)
        logger.info(f"Object loaded: {file_path}")
        return obj
    except Exception as e:
        raise FraudDetectionException(
            error_message=f"Error loading object: {file_path} - {str(e)}",
            error_detail=sys
        ) from e


# ============== DIRECTORY OPERATIONS ==============
def create_directories(paths: list) -> None:
    """
    Create directories from list of paths.
    
    Args:
        paths: Iterable/list of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory created: {path}")


def get_file_size(file_path) -> str:
    """
    Get file size in human readable format. Accepts string or pathlib.Path.
    
    Args:
        file_path: Path or string to file
    
    Returns:
        File size string (e.g., "10.5 MB")
    """
    file_path = Path(file_path)
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.2f} TB"


# ============== ENVIRONMENT OPERATIONS ==============
def get_env_variable(key: str, default: Any = None) -> Any:
    """
    Get environment variable with optional default.
    
    Args:
        key: Environment variable name
        default: Default value if not found
    
    Returns:
        Environment variable value
    """
    value = os.getenv(key, default)
    
    if value is None:
        logger.warning(f"Environment variable not found: {key}")
    
    return value


# ============== CONFIGURATION LOADER ==============
def load_config():
    """
    Load all configuration files.
    
    Returns:
        Tuple of (config, model_config, schema)
    """
    from src.constants import (
        CONFIG_FILE_PATH,
        MODEL_CONFIG_FILE_PATH,
        SCHEMA_FILE_PATH
    )
    
    config = read_yaml(Path(CONFIG_FILE_PATH))
    model_config = read_yaml(Path(MODEL_CONFIG_FILE_PATH))
    schema = read_yaml(Path(SCHEMA_FILE_PATH))
    
    logger.info("All configuration files loaded successfully")
    
    return config, model_config, schema


# ============== EXAMPLE USAGE ==============
if __name__ == "__main__":
    from src.constants import CONFIG_FILE_PATH
    
    # Test YAML reading
    config = read_yaml(Path(CONFIG_FILE_PATH))
    print(f"Config type: {type(config)}")
    print(f"Data Ingestion config: {config.data_ingestion}")

