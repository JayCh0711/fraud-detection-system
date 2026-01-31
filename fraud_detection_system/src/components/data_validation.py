"""
Data Validation Component for Fraud Detection System

Validations Performed:
1. Schema Validation (columns, data types)
2. Missing Value Analysis
3. Duplicate Detection
4. Value Range Validation
5. Categorical Value Validation
6. Data Drift Detection (Train vs Test)
7. Generate Validation Reports
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
import pandas as pd
import numpy as np
# Note: scipy is optional at import time; imported where needed to provide clearer errors if missing

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.logger import logger
from src.exception import DataValidationException
from src.utils.common import read_yaml, write_json, create_directories
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    """
    Data Validation Component for Fraud Detection System.
    
    Responsibilities:
    - Validate data against predefined schema
    - Check data quality (missing, duplicates, outliers)
    - Detect data drift between train and test sets
    - Generate comprehensive validation reports
    """
    
    def __init__(
        self,
        config: DataValidationConfig = DataValidationConfig(),
        data_ingestion_artifact: DataIngestionArtifact = None
    ):
        """
        Initialize Data Validation component.
        
        Args:
            config: DataValidationConfig object
            data_ingestion_artifact: Artifact from data ingestion step
        """
        self.config = config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.schema = read_yaml(SCHEMA_FILE_PATH)
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        logger.info(f"{'='*60}")
        logger.info("Initializing Data Validation Component")
        logger.info(f"{'='*60}")
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data from artifacts."""
        logger.info("Loading train and test data...")
        
        train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
        test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
        
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    # ==================== SCHEMA VALIDATION ====================
    
    def validate_columns(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Validate that all required columns are present.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging
        
        Returns:
            Dict with validation results
        """
        logger.info(f"Validating columns for {dataset_name}...")
        
        required_columns = self.schema.required_columns
        actual_columns = list(df.columns)
        
        # Find missing columns
        missing_columns = [col for col in required_columns if col not in actual_columns]
        
        # Find extra columns (not in schema)
        schema_columns = list(self.schema.columns.keys())
        extra_columns = [col for col in actual_columns if col not in schema_columns]
        
        result = {
            "required_columns": required_columns,
            "actual_columns": actual_columns,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "is_valid": len(missing_columns) == 0
        }
        
        if missing_columns:
            error_msg = f"{dataset_name}: Missing required columns: {missing_columns}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
        else:
            logger.info(f"{dataset_name}: All required columns present ✓")
        
        if extra_columns:
            warning_msg = f"{dataset_name}: Extra columns found (will be ignored): {extra_columns}"
            self.validation_warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        return result
    
    def validate_data_types(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Validate data types match schema definition.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging
        
        Returns:
            Dict with dtype validation results
        """
        logger.info(f"Validating data types for {dataset_name}...")
        
        dtype_mismatches = {}
        dtype_mapping = {
            "int64": ["int64", "int32", "int"],
            "float64": ["float64", "float32", "float"],
            "object": ["object", "string", "str"],
            "bool": ["bool", "boolean"]
        }
        
        for col_name, col_config in self.schema.columns.items():
            if col_name not in df.columns:
                continue
            
            expected_dtype = col_config.dtype
            actual_dtype = str(df[col_name].dtype)
            
            # Check if actual dtype is compatible with expected
            compatible_types = dtype_mapping.get(expected_dtype, [expected_dtype])
            
            if actual_dtype not in compatible_types:
                dtype_mismatches[col_name] = {
                    "expected": expected_dtype,
                    "actual": actual_dtype
                }
        
        result = {
            "dtype_mismatches": dtype_mismatches,
            "is_valid": len(dtype_mismatches) == 0
        }
        
        if dtype_mismatches:
            warning_msg = f"{dataset_name}: Data type mismatches: {dtype_mismatches}"
            self.validation_warnings.append(warning_msg)
            logger.warning(warning_msg)
        else:
            logger.info(f"{dataset_name}: All data types valid ✓")
        
        return result
    
    # ==================== DATA QUALITY CHECKS ====================
    
    def check_missing_values(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Check for missing values in each column.
        
        Args:
            df: DataFrame to check
            dataset_name: Name for logging
        
        Returns:
            Dict with missing value statistics
        """
        logger.info(f"Checking missing values for {dataset_name}...")
        
        total_rows = len(df)
        missing_stats = {}
        columns_exceeding_threshold = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            missing_stats[col] = {
                "missing_count": int(missing_count),
                "missing_percentage": round(missing_pct, 2)
            }
            
            if missing_pct > (self.config.missing_threshold * 100):
                columns_exceeding_threshold.append(col)
        
        # Check non-nullable columns
        non_nullable_violations = []
        for col_name, col_config in self.schema.columns.items():
            if col_name in df.columns and not col_config.get("nullable", True):
                if df[col_name].isnull().sum() > 0:
                    non_nullable_violations.append(col_name)
        
        result = {
            "missing_stats": missing_stats,
            "columns_exceeding_threshold": columns_exceeding_threshold,
            "non_nullable_violations": non_nullable_violations,
            "total_missing": int(df.isnull().sum().sum()),
            "is_valid": len(non_nullable_violations) == 0
        }
        
        if non_nullable_violations:
            error_msg = f"{dataset_name}: Non-nullable columns have missing values: {non_nullable_violations}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
        
        if columns_exceeding_threshold:
            warning_msg = f"{dataset_name}: Columns exceeding missing threshold: {columns_exceeding_threshold}"
            self.validation_warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        logger.info(f"{dataset_name}: Total missing values: {result['total_missing']}")
        
        return result
    
    def check_duplicates(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Check for duplicate rows in the dataset.
        
        Args:
            df: DataFrame to check
            dataset_name: Name for logging
        
        Returns:
            Dict with duplicate statistics
        """
        logger.info(f"Checking duplicates for {dataset_name}...")
        
        total_rows = len(df)
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / total_rows) * 100
        
        result = {
            "total_rows": total_rows,
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": round(duplicate_pct, 2),
            "is_valid": duplicate_pct <= (self.config.duplicate_threshold * 100)
        }
        
        if duplicate_count > 0:
            warning_msg = f"{dataset_name}: Found {duplicate_count} duplicate rows ({duplicate_pct:.2f}%)"
            self.validation_warnings.append(warning_msg)
            logger.warning(warning_msg)
        else:
            logger.info(f"{dataset_name}: No duplicate rows found ✓")
        
        return result
    
    def validate_value_ranges(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Validate that numeric values fall within expected ranges.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging
        
        Returns:
            Dict with range validation results
        """
        logger.info(f"Validating value ranges for {dataset_name}...")
        
        range_violations = {}
        
        for col_name, col_config in self.schema.columns.items():
            if col_name not in df.columns:
                continue
            
            min_val = col_config.get("min_value", None)
            max_val = col_config.get("max_value", None)
            
            if min_val is not None:
                violations = (df[col_name] < min_val).sum()
                if violations > 0:
                    range_violations[col_name] = range_violations.get(col_name, {})
                    range_violations[col_name]["below_min"] = {
                        "min_allowed": min_val,
                        "violation_count": int(violations)
                    }
            
            if max_val is not None:
                violations = (df[col_name] > max_val).sum()
                if violations > 0:
                    range_violations[col_name] = range_violations.get(col_name, {})
                    range_violations[col_name]["above_max"] = {
                        "max_allowed": max_val,
                        "violation_count": int(violations)
                    }
        
        result = {
            "range_violations": range_violations,
            "is_valid": len(range_violations) == 0
        }
        
        if range_violations:
            warning_msg = f"{dataset_name}: Value range violations: {list(range_violations.keys())}"
            self.validation_warnings.append(warning_msg)
            logger.warning(warning_msg)
        else:
            logger.info(f"{dataset_name}: All value ranges valid ✓")
        
        return result
    
    def validate_categorical_values(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Validate that categorical columns contain only allowed values.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging
        
        Returns:
            Dict with categorical validation results
        """
        logger.info(f"Validating categorical values for {dataset_name}...")
        
        categorical_violations = {}
        
        for col_name, col_config in self.schema.columns.items():
            if col_name not in df.columns:
                continue
            
            allowed_values = col_config.get("allowed_values", None)
            
            if allowed_values is not None:
                actual_values = set(df[col_name].dropna().unique())
                invalid_values = actual_values - set(allowed_values)
                
                if invalid_values:
                    categorical_violations[col_name] = {
                        "allowed_values": allowed_values,
                        "invalid_values": list(invalid_values),
                        "invalid_count": int(df[col_name].isin(invalid_values).sum())
                    }
        
        result = {
            "categorical_violations": categorical_violations,
            "is_valid": len(categorical_violations) == 0
        }
        
        if categorical_violations:
            warning_msg = f"{dataset_name}: Categorical violations: {list(categorical_violations.keys())}"
            self.validation_warnings.append(warning_msg)
            logger.warning(warning_msg)
        else:
            logger.info(f"{dataset_name}: All categorical values valid ✓")
        
        return result
    
    # ==================== DATA DRIFT DETECTION ====================
    
    def detect_numerical_drift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict:
        """
        Detect drift in numerical columns using KS test.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            threshold: p-value threshold for drift detection
        
        Returns:
            Dict with drift detection results
        """
        logger.info("Detecting numerical drift (KS Test)...")
        
        numerical_cols = train_df.select_dtypes(include=[np.number]).columns
        drift_results = {}
        drifted_columns = []
        
        # Import scipy.stats lazily to provide clear error if not installed
        try:
            from scipy import stats
        except ImportError as e:
            raise DataValidationException(
                error_message=(
                    "scipy is required for numerical drift detection (KS test). "
                    "Install it with `pip install scipy` or add to requirements.txt"
                ),
                error_detail=sys
            ) from e

        for col in numerical_cols:
            if col in test_df.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(
                    train_df[col].dropna(),
                    test_df[col].dropna()
                )
                
                is_drifted = p_value < threshold
                
                drift_results[col] = {
                    "ks_statistic": round(float(statistic), 4),
                    "p_value": round(float(p_value), 4),
                    "is_drifted": is_drifted
                }
                
                if is_drifted:
                    drifted_columns.append(col)
        
        logger.info(f"Numerical drift detected in {len(drifted_columns)} columns: {drifted_columns}")
        
        return {
            "drift_results": drift_results,
            "drifted_columns": drifted_columns,
            "drift_percentage": round((len(drifted_columns) / len(numerical_cols)) * 100, 2)
        }
    
    def detect_categorical_drift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict:
        """
        Detect drift in categorical columns using Chi-Square test.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            threshold: p-value threshold for drift detection
        
        Returns:
            Dict with categorical drift results
        """
        logger.info("Detecting categorical drift (Chi-Square Test)...")
        
        categorical_cols = train_df.select_dtypes(include=['object']).columns
        drift_results = {}
        drifted_columns = []
        
        for col in categorical_cols:
            if col in test_df.columns:
                try:
                    # Get value counts
                    train_counts = train_df[col].value_counts()
                    test_counts = test_df[col].value_counts()
                    
                    # Align indices
                    all_categories = set(train_counts.index) | set(test_counts.index)
                    train_counts = train_counts.reindex(all_categories, fill_value=0)
                    test_counts = test_counts.reindex(all_categories, fill_value=0)
                    
                    # Normalize
                    train_freq = train_counts / train_counts.sum()
                    test_freq = test_counts / test_counts.sum()
                    
                    # Chi-square test
                    # Using Jensen-Shannon divergence as alternative
                    js_divergence = self._jensen_shannon_divergence(
                        train_freq.values,
                        test_freq.values
                    )
                    
                    is_drifted = js_divergence > 0.1  # threshold for JS divergence
                    
                    drift_results[col] = {
                        "js_divergence": round(float(js_divergence), 4),
                        "is_drifted": is_drifted
                    }
                    
                    if is_drifted:
                        drifted_columns.append(col)
                        
                except Exception as e:
                    logger.warning(f"Could not compute drift for {col}: {str(e)}")
        
        logger.info(f"Categorical drift detected in {len(drifted_columns)} columns: {drifted_columns}")
        
        return {
            "drift_results": drift_results,
            "drifted_columns": drifted_columns,
            "drift_percentage": round(
                (len(drifted_columns) / max(len(categorical_cols), 1)) * 100, 2
            )
        }
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)
        
        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Import scipy.stats locally; raise helpful error if missing
        try:
            from scipy import stats
        except ImportError as e:
            raise DataValidationException(
                error_message=(
                    "scipy is required to compute Jensen-Shannon divergence. "
                    "Install it with `pip install scipy` or add to requirements.txt"
                ),
                error_detail=sys
            ) from e

        # Calculate JS divergence
        m = 0.5 * (p + q)
        js = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

        return float(js)
    
    def detect_target_drift(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Detect drift in target variable distribution.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        
        Returns:
            Dict with target drift analysis
        """
        logger.info("Detecting target variable drift...")
        
        target_col = self.schema.target_column
        
        train_dist = train_df[target_col].value_counts(normalize=True).to_dict()
        test_dist = test_df[target_col].value_counts(normalize=True).to_dict()
        
        # Calculate distribution difference
        train_fraud_rate = train_dist.get(1, 0)
        test_fraud_rate = test_dist.get(1, 0)
        
        rate_difference = abs(train_fraud_rate - test_fraud_rate)
        
        result = {
            "train_distribution": {k: round(v, 4) for k, v in train_dist.items()},
            "test_distribution": {k: round(v, 4) for k, v in test_dist.items()},
            "train_fraud_rate": round(train_fraud_rate * 100, 2),
            "test_fraud_rate": round(test_fraud_rate * 100, 2),
            "rate_difference": round(rate_difference * 100, 2),
            "is_drifted": rate_difference > 0.05  # 5% difference threshold
        }
        
        logger.info(f"Train fraud rate: {result['train_fraud_rate']}%")
        logger.info(f"Test fraud rate: {result['test_fraud_rate']}%")
        
        if result["is_drifted"]:
            warning_msg = f"Target drift detected: {result['rate_difference']}% difference"
            self.validation_warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        return result
    
    # ==================== REPORT GENERATION ====================
    
    def generate_validation_report(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive validation report.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        
        Returns:
            Dict containing all validation results
        """
        logger.info("Generating validation report...")
        
        report = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "schema_version": self.schema.get("schema_version", "1.0"),
            
            # Dataset Info
            "dataset_info": {
                "train_shape": train_df.shape,
                "test_shape": test_df.shape,
                "train_file": self.data_ingestion_artifact.train_file_path,
                "test_file": self.data_ingestion_artifact.test_file_path
            },
            
            # Schema Validation
            "schema_validation": {
                "train": self.validate_columns(train_df, "Train"),
                "test": self.validate_columns(test_df, "Test")
            },
            
            # Data Type Validation
            "dtype_validation": {
                "train": self.validate_data_types(train_df, "Train"),
                "test": self.validate_data_types(test_df, "Test")
            },
            
            # Missing Values
            "missing_values": {
                "train": self.check_missing_values(train_df, "Train"),
                "test": self.check_missing_values(test_df, "Test")
            },
            
            # Duplicates
            "duplicates": {
                "train": self.check_duplicates(train_df, "Train"),
                "test": self.check_duplicates(test_df, "Test")
            },
            
            # Value Ranges
            "value_ranges": {
                "train": self.validate_value_ranges(train_df, "Train"),
                "test": self.validate_value_ranges(test_df, "Test")
            },
            
            # Categorical Values
            "categorical_values": {
                "train": self.validate_categorical_values(train_df, "Train"),
                "test": self.validate_categorical_values(test_df, "Test")
            }
        }
        
        return report
    
    def generate_drift_report(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Generate data drift report.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        
        Returns:
            Dict containing drift analysis
        """
        logger.info("Generating drift report...")
        
        drift_report = {
            "drift_timestamp": pd.Timestamp.now().isoformat(),
            
            # Numerical Drift
            "numerical_drift": self.detect_numerical_drift(train_df, test_df),
            
            # Categorical Drift
            "categorical_drift": self.detect_categorical_drift(train_df, test_df),
            
            # Target Drift
            "target_drift": self.detect_target_drift(train_df, test_df)
        }
        
        # Overall drift status
        num_drifted = len(drift_report["numerical_drift"]["drifted_columns"])
        cat_drifted = len(drift_report["categorical_drift"]["drifted_columns"])
        target_drifted = drift_report["target_drift"]["is_drifted"]
        
        drift_report["overall_drift"] = {
            "total_drifted_features": num_drifted + cat_drifted,
            "target_drifted": target_drifted,
            "drift_detected": (num_drifted + cat_drifted) > 0 or target_drifted
        }
        
        return drift_report
    
    # ==================== MAIN EXECUTION ====================
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Execute complete data validation pipeline.
        
        Returns:
            DataValidationArtifact: Validation results artifact
        """
        logger.info(f"{'='*60}")
        logger.info("Starting Data Validation Pipeline")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Create directories
            logger.info("Step 1: Creating directories")
            create_directories([self.config.data_validation_dir])
            
            # Step 2: Load data
            logger.info("Step 2: Loading data")
            train_df, test_df = self._load_data()
            
            # Step 3: Generate validation report
            logger.info("Step 3: Generating validation report")
            validation_report = self.generate_validation_report(train_df, test_df)
            
            # Step 4: Generate drift report
            logger.info("Step 4: Generating drift report")
            drift_report = self.generate_drift_report(train_df, test_df)
            
            # Step 5: Save reports
            logger.info("Step 5: Saving reports")
            
            validation_report_path = self.config.validation_report_path
            drift_report_path = self.config.drift_report_path
            
            # Convert tuples to lists for JSON serialization
            validation_report["dataset_info"]["train_shape"] = list(
                validation_report["dataset_info"]["train_shape"]
            )
            validation_report["dataset_info"]["test_shape"] = list(
                validation_report["dataset_info"]["test_shape"]
            )
            
            write_json(validation_report_path, validation_report)
            write_json(drift_report_path, drift_report)
            
            # Step 6: Determine overall validation status
            logger.info("Step 6: Determining validation status")
            
            is_validated = len(self.validation_errors) == 0
            validation_status = "PASSED" if is_validated else "FAILED"
            
            # Collect missing columns from train validation
            missing_columns = validation_report["schema_validation"]["train"]["missing_columns"]
            
            # Create artifact
            artifact = DataValidationArtifact(
                is_validated=is_validated,
                validation_status=validation_status,
                validation_report_path=validation_report_path,
                drift_report_path=drift_report_path,
                missing_columns=missing_columns,
                duplicate_rows=validation_report["duplicates"]["train"]["duplicate_count"],
                message=f"Validation {validation_status}. Errors: {len(self.validation_errors)}, Warnings: {len(self.validation_warnings)}"
            )
            
            # Log summary
            logger.info(f"{'='*60}")
            logger.info("DATA VALIDATION SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Status: {validation_status}")
            logger.info(f"Errors: {len(self.validation_errors)}")
            logger.info(f"Warnings: {len(self.validation_warnings)}")
            
            if self.validation_errors:
                logger.error("Validation Errors:")
                for error in self.validation_errors:
                    logger.error(f"  - {error}")
            
            if self.validation_warnings:
                logger.warning("Validation Warnings:")
                for warning in self.validation_warnings:
                    logger.warning(f"  - {warning}")
            
            logger.info(f"{'='*60}")
            logger.info("Data Validation Completed!")
            logger.info(f"{'='*60}")
            
            return artifact
            
        except Exception as e:
            raise DataValidationException(
                error_message=f"Data Validation failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig
    from src.components.data_ingestion import DataIngestion
    
    # Run Data Ingestion first
    ingestion_config = DataIngestionConfig(
        source_type="csv",
        source_path="data/raw/transactions.csv"
    )
    
    ingestion = DataIngestion(config=ingestion_config)
    ingestion_artifact = ingestion.initiate_data_ingestion()
    
    # Run Data Validation
    validation_config = DataValidationConfig()
    validation = DataValidation(
        config=validation_config,
        data_ingestion_artifact=ingestion_artifact
    )
    
    validation_artifact = validation.initiate_data_validation()
    
    print("\n" + "="*60)
    print("DATA VALIDATION ARTIFACT")
    print("="*60)
    print(f"Is Validated: {validation_artifact.is_validated}")
    print(f"Status: {validation_artifact.validation_status}")
    print(f"Report Path: {validation_artifact.validation_report_path}")
    print(f"Drift Report: {validation_artifact.drift_report_path}")
    print(f"Message: {validation_artifact.message}")