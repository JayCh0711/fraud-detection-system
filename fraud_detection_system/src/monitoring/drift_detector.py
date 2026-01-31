"""
Data and Model Drift Detection using Evidently

Detects:
1. Data Drift (input feature distributions)
2. Prediction Drift (model output distributions)
3. Target Drift (label distributions)
4. Feature-level Drift
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Evidently imports
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset
    )
    from evidently.metrics import (
        DatasetDriftMetric,
        DataDriftTable,
        ColumnDriftMetric,
        DatasetMissingValuesMetric,
        DatasetCorrelationsMetric
    )
    from evidently.test_suite import TestSuite
    from evidently.test_preset import (
        DataDriftTestPreset,
        DataQualityTestPreset
    )
    from evidently.tests import (
        TestNumberOfDriftedColumns,
        TestShareOfDriftedColumns,
        TestColumnDrift
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Warning: Evidently not installed. Run: pip install evidently")

from src.monitoring.config import monitoring_config, monitoring_settings
from src.logger import logger
from src.utils.common import write_json, create_directories


@dataclass
class DriftResult:
    """Result of drift detection"""
    
    # Overall Drift
    is_drift_detected: bool = False
    drift_share: float = 0.0
    number_of_drifted_features: int = 0
    total_features: int = 0
    
    # Feature-level Drift
    drifted_features: List[str] = None
    feature_drift_scores: Dict[str, float] = None
    
    # Statistical Details
    drift_method: str = ""
    threshold_used: float = 0.0
    
    # Report Paths
    html_report_path: str = ""
    json_report_path: str = ""
    
    # Metadata
    reference_data_size: int = 0
    current_data_size: int = 0
    detection_timestamp: str = ""
    
    def __post_init__(self):
        if self.drifted_features is None:
            self.drifted_features = []
        if self.feature_drift_scores is None:
            self.feature_drift_scores = {}
        if not self.detection_timestamp:
            self.detection_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DriftDetector:
    """
    Drift Detection System using Evidently.
    
    Monitors:
    - Data drift in input features
    - Prediction drift in model outputs
    - Feature-level drift analysis
    """
    
    def __init__(self, config: MonitoringConfig = None):
        """
        Initialize Drift Detector.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or monitoring_config
        self.settings = monitoring_settings
        
        self.reference_data: Optional[pd.DataFrame] = None
        self.column_mapping: Optional[ColumnMapping] = None
        
        # Create directories
        create_directories([
            self.config.drift_reports_dir
        ])
        
        logger.info("Initializing Drift Detector")
        
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available. Drift detection will be limited.")
    
    def set_reference_data(
        self,
        data: pd.DataFrame,
        target_column: str = "is_fraud",
        prediction_column: str = None,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None
    ) -> None:
        """
        Set reference data for drift comparison.
        
        Args:
            data: Reference DataFrame (typically training data)
            target_column: Name of target column
            prediction_column: Name of prediction column (if available)
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
        """
        self.reference_data = data.copy()
        
        # Auto-detect feature types if not provided
        if numerical_features is None:
            numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numerical_features:
                numerical_features.remove(target_column)
        
        if categorical_features is None:
            categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Setup column mapping
        self.column_mapping = ColumnMapping(
            target=target_column if target_column in data.columns else None,
            prediction=prediction_column,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
        
        logger.info(f"Reference data set: {len(data)} samples")
        logger.info(f"  Numerical features: {len(numerical_features)}")
        logger.info(f"  Categorical features: {len(categorical_features)}")
    
    def load_reference_data(self, path: str = None) -> None:
        """
        Load reference data from file.
        
        Args:
            path: Path to reference data CSV
        """
        path = path or self.config.reference_data_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference data not found: {path}")
        
        data = pd.read_csv(path)
        self.set_reference_data(data)
        
        logger.info(f"Reference data loaded from: {path}")
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True,
        report_name: str = None
    ) -> DriftResult:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current data to compare
            save_report: Whether to save HTML/JSON reports
            report_name: Custom name for reports
        
        Returns:
            DriftResult object
        """
        logger.info("Detecting data drift...")
        
        if self.reference_data is None:
            self.load_reference_data()
        
        if not EVIDENTLY_AVAILABLE:
            return self._detect_drift_manual(current_data)
        
        try:
            # Create data drift report
            report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable(),
                DatasetMissingValuesMetric(),
            ])
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract results
            report_dict = report.as_dict()
            
            # Get drift metrics
            dataset_drift = report_dict['metrics'][0]['result']
            drift_table = report_dict['metrics'][1]['result']
            
            is_drift = dataset_drift.get('dataset_drift', False)
            drift_share = dataset_drift.get('share_of_drifted_columns', 0)
            n_drifted = dataset_drift.get('number_of_drifted_columns', 0)
            n_total = dataset_drift.get('number_of_columns', 0)
            
            # Get drifted features
            drifted_features = []
            feature_scores = {}
            
            drift_by_columns = drift_table.get('drift_by_columns', {})
            for col, col_info in drift_by_columns.items():
                score = col_info.get('drift_score', 0)
                feature_scores[col] = score
                if col_info.get('drift_detected', False):
                    drifted_features.append(col)
            
            # Create result
            result = DriftResult(
                is_drift_detected=is_drift,
                drift_share=drift_share,
                number_of_drifted_features=n_drifted,
                total_features=n_total,
                drifted_features=drifted_features,
                feature_drift_scores=feature_scores,
                drift_method="evidently",
                threshold_used=self.settings.DATA_DRIFT_THRESHOLD,
                reference_data_size=len(self.reference_data),
                current_data_size=len(current_data)
            )
            
            # Save reports
            if save_report:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = report_name or f"data_drift_{timestamp}"
                
                html_path = os.path.join(
                    self.config.drift_reports_dir,
                    f"{report_name}.html"
                )
                json_path = os.path.join(
                    self.config.drift_reports_dir,
                    f"{report_name}.json"
                )
                
                report.save_html(html_path)
                write_json(json_path, result.to_dict())
                
                result.html_report_path = html_path
                result.json_report_path = json_path
                
                logger.info(f"Drift report saved: {html_path}")
            
            # Log results
            if result.is_drift_detected:
                logger.warning(f"⚠️ DATA DRIFT DETECTED!")
                logger.warning(f"  Drift Share: {result.drift_share:.2%}")
                logger.warning(f"  Drifted Features: {result.drifted_features}")
            else:
                logger.info(f"✓ No significant data drift detected")
                logger.info(f"  Drift Share: {result.drift_share:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            return self._detect_drift_manual(current_data)
    
    def _detect_drift_manual(
        self,
        current_data: pd.DataFrame
    ) -> DriftResult:
        """
        Manual drift detection using statistical tests (fallback).
        
        Args:
            current_data: Current data to compare
        
        Returns:
            DriftResult object
        """
        from scipy import stats
        
        logger.info("Using manual drift detection (scipy)...")
        
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        drifted_features = []
        feature_scores = {}
        
        numerical_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in current_data.columns and col in self.reference_data.columns:
                # Kolmogorov-Smirnov test
                stat, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                feature_scores[col] = float(stat)
                
                if p_value < 0.05:  # Significant drift
                    drifted_features.append(col)
        
        drift_share = len(drifted_features) / len(numerical_cols) if len(numerical_cols) > 0 else 0
        is_drift = drift_share > self.settings.DATA_DRIFT_THRESHOLD
        
        return DriftResult(
            is_drift_detected=is_drift,
            drift_share=drift_share,
            number_of_drifted_features=len(drifted_features),
            total_features=len(numerical_cols),
            drifted_features=drifted_features,
            feature_drift_scores=feature_scores,
            drift_method="ks_test",
            threshold_used=self.settings.DATA_DRIFT_THRESHOLD,
            reference_data_size=len(self.reference_data),
            current_data_size=len(current_data)
        )
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        save_report: bool = True
    ) -> DriftResult:
        """
        Detect drift in model predictions.
        
        Args:
            reference_predictions: Reference prediction probabilities
            current_predictions: Current prediction probabilities
            save_report: Whether to save report
        
        Returns:
            DriftResult object
        """
        logger.info("Detecting prediction drift...")
        
        from scipy import stats
        
        # KS test on predictions
        stat, p_value = stats.ks_2samp(reference_predictions, current_predictions)
        
        is_drift = stat > self.settings.PREDICTION_DRIFT_THRESHOLD
        
        result = DriftResult(
            is_drift_detected=is_drift,
            drift_share=float(stat),
            number_of_drifted_features=1 if is_drift else 0,
            total_features=1,
            drifted_features=["predictions"] if is_drift else [],
            feature_drift_scores={"predictions": float(stat)},
            drift_method="ks_test",
            threshold_used=self.settings.PREDICTION_DRIFT_THRESHOLD,
            reference_data_size=len(reference_predictions),
            current_data_size=len(current_predictions)
        )
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(
                self.config.drift_reports_dir,
                f"prediction_drift_{timestamp}.json"
            )
            write_json(json_path, result.to_dict())
            result.json_report_path = json_path
        
        if result.is_drift_detected:
            logger.warning(f"⚠️ PREDICTION DRIFT DETECTED!")
            logger.warning(f"  KS Statistic: {stat:.4f}")
        else:
            logger.info(f"✓ No significant prediction drift")
        
        return result
    
    def run_drift_tests(
        self,
        current_data: pd.DataFrame,
        max_drift_share: float = 0.3
    ) -> Dict:
        """
        Run drift test suite.
        
        Args:
            current_data: Current data to test
            max_drift_share: Maximum acceptable drift share
        
        Returns:
            Test results dictionary
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available. Skipping test suite.")
            return {"status": "skipped", "reason": "Evidently not installed"}
        
        logger.info("Running drift test suite...")
        
        if self.reference_data is None:
            self.load_reference_data()
        
        try:
            # Create test suite
            tests = TestSuite(tests=[
                TestNumberOfDriftedColumns(lt=int(len(self.reference_data.columns) * max_drift_share)),
                TestShareOfDriftedColumns(lt=max_drift_share),
            ])
            
            tests.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Get results
            test_results = tests.as_dict()
            
            all_passed = all(
                test['status'] == 'SUCCESS' 
                for test in test_results.get('tests', [])
            )
            
            result = {
                "status": "passed" if all_passed else "failed",
                "tests_run": len(test_results.get('tests', [])),
                "tests_passed": sum(1 for t in test_results.get('tests', []) if t['status'] == 'SUCCESS'),
                "tests_failed": sum(1 for t in test_results.get('tests', []) if t['status'] != 'SUCCESS'),
                "details": test_results.get('tests', []),
                "timestamp": datetime.now().isoformat()
            }
            
            if all_passed:
                logger.info("✓ All drift tests passed")
            else:
                logger.warning("⚠️ Some drift tests failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Drift tests failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        include_data_quality: bool = True
    ) -> str:
        """
        Generate comprehensive drift report.
        
        Args:
            current_data: Current data for analysis
            include_data_quality: Include data quality metrics
        
        Returns:
            Path to HTML report
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available. Cannot generate report.")
            return ""
        
        logger.info("Generating comprehensive drift report...")
        
        if self.reference_data is None:
            self.load_reference_data()
        
        try:
            # Build metrics list
            metrics = [DataDriftPreset()]
            
            if include_data_quality:
                metrics.append(DataQualityPreset())
            
            # Create report
            report = Report(metrics=metrics)
            
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(
                self.config.drift_reports_dir,
                f"comprehensive_drift_report_{timestamp}.html"
            )
            
            report.save_html(report_path)
            
            logger.info(f"Comprehensive report saved: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return ""


# Singleton instance
_drift_detector: Optional[DriftDetector] = None


def get_drift_detector() -> DriftDetector:
    """Get singleton drift detector instance."""
    global _drift_detector
    
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    
    return _drift_detector