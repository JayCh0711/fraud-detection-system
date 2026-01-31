"""
Prediction Pipeline for Fraud Detection System

Features:
1. Single Transaction Prediction
2. Batch Prediction
3. Real-time Scoring
4. Risk Categorization
5. Feature Importance for Prediction
6. Prediction Logging
7. Threshold Adjustment
8. Explainability (SHAP-ready)
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Project imports
from src.logger import logger
from src.exception import PredictionException
from src.utils.common import (
    load_object,
    read_json,
    write_json,
    create_directories
)
from src.constants import ARTIFACTS_DIR, TARGET_COLUMN


# ============== PREDICTION CONFIG ==============
@dataclass
class PredictionConfig:
    """Configuration for Prediction Pipeline"""
    
    # Model Paths
    production_model_dir: str = os.path.join(ARTIFACTS_DIR, "production_model")
    model_file: str = "model.pkl"
    preprocessor_file: str = "preprocessor.pkl"
    feature_names_file: str = "feature_names.json"
    version_file: str = "version.json"
    
    # Prediction Settings
    default_threshold: float = 0.5
    use_optimal_threshold: bool = True
    
    # Risk Categories
    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4
    low_risk_threshold: float = 0.2
    
    # Logging
    enable_prediction_logging: bool = True
    prediction_logs_dir: str = os.path.join(ARTIFACTS_DIR, "predictions", "prediction_logs")
    batch_results_dir: str = os.path.join(ARTIFACTS_DIR, "predictions", "batch_results")
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.production_model_dir, self.model_file)
    
    @property
    def preprocessor_path(self) -> str:
        return os.path.join(self.production_model_dir, self.preprocessor_file)
    
    @property
    def feature_names_path(self) -> str:
        return os.path.join(self.production_model_dir, self.feature_names_file)
    
    @property
    def version_path(self) -> str:
        return os.path.join(self.production_model_dir, self.version_file)


# ============== PREDICTION RESULT ==============
@dataclass
class PredictionResult:
    """Result of a single prediction"""
    
    transaction_id: str = ""
    prediction: int = 0
    probability: float = 0.0
    risk_score: float = 0.0
    risk_category: str = ""
    is_fraud: bool = False
    threshold_used: float = 0.5
    
    # Feature contributions (for explainability)
    top_risk_factors: List[Dict] = field(default_factory=list)
    
    # Metadata
    model_version: str = ""
    prediction_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ============== BATCH PREDICTION RESULT ==============
@dataclass
class BatchPredictionResult:
    """Result of batch predictions"""
    
    batch_id: str = ""
    total_transactions: int = 0
    fraud_count: int = 0
    legitimate_count: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    
    predictions_df: pd.DataFrame = None
    
    # Processing Info
    processing_time_seconds: float = 0.0
    model_version: str = ""
    threshold_used: float = 0.5
    
    # Timestamp
    batch_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


# ============== FEATURE ENGINEER FOR PREDICTION ==============
class PredictionFeatureEngineer:
    """
    Feature Engineering for Prediction Pipeline.
    Creates same features as training pipeline.
    """
    
    def __init__(self, selected_features: List[str] = None):
        """
        Initialize Feature Engineer.
        
        Args:
            selected_features: List of features to keep (from training)
        """
        self.selected_features = selected_features or []
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        
        # Log Transform
        df['amount_log'] = np.log1p(df['amount'])
        
        # Amount Squared
        df['amount_squared'] = df['amount'] ** 2
        
        # Amount to Balance Ratios
        df['amount_to_orig_balance_ratio'] = np.where(
            df['old_balance_org'] > 0,
            df['amount'] / df['old_balance_org'],
            0
        )
        
        df['amount_to_dest_balance_ratio'] = np.where(
            df['old_balance_dest'] > 0,
            df['amount'] / df['old_balance_dest'],
            0
        )
        
        # Amount Categories
        amount_bins = [0, 1000, 10000, 50000, 100000, 500000, np.inf]
        amount_labels = ['very_small', 'small', 'medium', 'large', 'very_large', 'huge']
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=amount_bins, 
            labels=amount_labels
        ).astype(str)
        
        # Is High Amount
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        # Amount Z-score
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std() if df['amount'].std() > 0 else 1
        df['amount_zscore'] = (df['amount'] - amount_mean) / amount_std
        
        return df
    
    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balance-based features."""
        
        # Balance Differences
        df['orig_balance_diff'] = df['old_balance_org'] - df['new_balance_org']
        df['dest_balance_diff'] = df['new_balance_dest'] - df['old_balance_dest']
        
        # Balance Change Percentages
        df['orig_balance_change_pct'] = np.where(
            df['old_balance_org'] > 0,
            ((df['old_balance_org'] - df['new_balance_org']) / df['old_balance_org']) * 100,
            0
        )
        
        df['dest_balance_change_pct'] = np.where(
            df['old_balance_dest'] > 0,
            ((df['new_balance_dest'] - df['old_balance_dest']) / df['old_balance_dest']) * 100,
            0
        )
        
        # Account Drain Flag
        df['is_account_drain'] = (
            (df['new_balance_org'] == 0) & 
            (df['old_balance_org'] > 0) &
            (df['amount'] > 0)
        ).astype(int)
        
        # Full Transfer
        df['is_full_transfer'] = (
            np.abs(df['amount'] - df['old_balance_org']) < 1
        ).astype(int)
        
        # Zero Balance Flags
        df['is_zero_orig_balance'] = (df['old_balance_org'] == 0).astype(int)
        df['is_zero_dest_balance'] = (df['old_balance_dest'] == 0).astype(int)
        
        # Total Balance
        df['total_balance_before'] = df['old_balance_org'] + df['old_balance_dest']
        
        # Balance Ratio
        df['balance_ratio_orig_dest'] = np.where(
            df['old_balance_dest'] > 0,
            df['old_balance_org'] / (df['old_balance_dest'] + 1),
            df['old_balance_org']
        )
        
        # Log transforms
        df['old_balance_org_log'] = np.log1p(df['old_balance_org'])
        df['new_balance_org_log'] = np.log1p(df['new_balance_org'])
        df['old_balance_dest_log'] = np.log1p(df['old_balance_dest'])
        df['new_balance_dest_log'] = np.log1p(df['new_balance_dest'])
        
        return df
    
    def create_error_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create error detection features."""
        
        # Origin Balance Error
        df['orig_balance_error'] = (
            df['new_balance_org'] - (df['old_balance_org'] - df['amount'])
        )
        df['has_orig_error'] = (np.abs(df['orig_balance_error']) > 1).astype(int)
        
        # Destination Balance Error
        df['dest_balance_error'] = (
            df['new_balance_dest'] - (df['old_balance_dest'] + df['amount'])
        )
        df['has_dest_error'] = (np.abs(df['dest_balance_error']) > 1).astype(int)
        
        # Any Error
        df['has_any_error'] = (
            (df['has_orig_error'] == 1) | (df['has_dest_error'] == 1)
        ).astype(int)
        
        # Suspicious Patterns
        df['suspicious_zero_after'] = (
            (df['old_balance_org'] > 10000) &
            (df['amount'] > 10000) &
            (df['new_balance_org'] == 0)
        ).astype(int)
        
        df['large_from_zero'] = (
            (df['old_balance_org'] == 0) &
            (df['amount'] > 10000)
        ).astype(int)
        
        df['amount_exceeds_balance'] = (
            df['amount'] > df['old_balance_org']
        ).astype(int)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        
        if 'step' not in df.columns:
            # Create dummy step if not present
            df['step'] = 0
        
        # Hour of Day
        df['hour_of_day'] = df['step'] % 24
        
        # Day Number
        df['day_number'] = df['step'] // 24
        
        # Day of Week
        df['day_of_week'] = (df['step'] // 24) % 7
        
        # Is Weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Is Night
        df['is_night'] = (
            (df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)
        ).astype(int)
        
        # Is Business Hours
        df['is_business_hours'] = (
            (df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)
        ).astype(int)
        
        # Time Period
        def get_time_period(hour):
            if 6 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 18:
                return 'afternoon'
            elif 18 <= hour < 22:
                return 'evening'
            else:
                return 'night'
        
        df['time_period'] = df['hour_of_day'].apply(get_time_period)
        
        # Week Number
        df['week_number'] = df['step'] // (24 * 7)
        
        # Cyclical Encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_transaction_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction type features."""
        
        if 'type' not in df.columns:
            df['type'] = 'UNKNOWN'
        
        # High Risk Type
        df['is_high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
        
        # Individual Type Flags
        df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
        df['is_cash_out'] = (df['type'] == 'CASH_OUT').astype(int)
        df['is_payment'] = (df['type'] == 'PAYMENT').astype(int)
        df['is_cash_in'] = (df['type'] == 'CASH_IN').astype(int)
        df['is_debit'] = (df['type'] == 'DEBIT').astype(int)
        
        # Combinations
        df['high_risk_high_amount'] = (
            (df['is_high_risk_type'] == 1) & 
            (df['amount'] > df['amount'].quantile(0.90) if len(df) > 1 else df['amount'] > 100000)
        ).astype(int)
        
        df['high_risk_drain'] = (
            (df['is_high_risk_type'] == 1) & 
            (df['is_account_drain'] == 1)
        ).astype(int)
        
        return df
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features."""
        
        # Amount to Total Balance
        total_balance = df['old_balance_org'] + df['old_balance_dest']
        df['amount_to_total_balance_ratio'] = np.where(
            total_balance > 0,
            df['amount'] / total_balance,
            0
        )
        
        # New Balance Ratio
        df['new_balance_ratio'] = np.where(
            df['new_balance_dest'] > 0,
            df['new_balance_org'] / (df['new_balance_dest'] + 1),
            df['new_balance_org']
        )
        
        # Balance Change Ratio
        df['balance_change_ratio'] = np.where(
            np.abs(df['dest_balance_diff']) > 0,
            df['orig_balance_diff'] / (np.abs(df['dest_balance_diff']) + 1),
            0
        )
        
        # Relative Transaction Size
        df['relative_transaction_size'] = np.where(
            df['old_balance_org'] > 0,
            (df['amount'] / df['old_balance_org']) * 100,
            0
        ).clip(0, 1000)
        
        # Balance Retention Rate
        df['balance_retention_rate'] = np.where(
            df['old_balance_org'] > 0,
            (df['new_balance_org'] / df['old_balance_org']) * 100,
            0
        ).clip(0, 100)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        
        # Risk Score
        df['risk_score'] = (
            df['is_high_risk_type'] * 2 +
            df['is_account_drain'] * 3 +
            df['is_high_amount'] * 1 +
            df['is_night'] * 1 +
            df['has_any_error'] * 2 +
            df['amount_exceeds_balance'] * 2
        )
        
        # Amount * Balance Interaction
        df['amount_balance_interaction'] = (
            df['amount_log'] * df['old_balance_org_log']
        )
        
        # Time * Amount Interaction
        df['time_amount_interaction'] = df['hour_of_day'] * df['amount_log']
        
        # Type Risk * Amount
        df['type_amount_risk'] = df['is_high_risk_type'] * df['amount_log']
        
        # Weekend Night
        df['weekend_night'] = (
            (df['is_weekend'] == 1) & (df['is_night'] == 1)
        ).astype(int)
        
        # High Risk Combination
        df['high_risk_combo'] = (
            (df['is_high_risk_type'] == 1) &
            (df['is_account_drain'] == 1) &
            (df['amount'] > df['amount'].median() if len(df) > 1 else df['amount'] > 50000)
        ).astype(int)
        
        # Suspicious Flag
        df['is_suspicious'] = (
            (df['risk_score'] >= 4) |
            (df['high_risk_combo'] == 1) |
            (df['suspicious_zero_after'] == 1)
        ).astype(int)
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        
        # Amount Percentile
        df['amount_percentile'] = df['amount'].rank(pct=True)
        
        # Balance Percentile
        df['orig_balance_percentile'] = df['old_balance_org'].rank(pct=True)
        
        # Combined Percentile
        df['amount_balance_percentile'] = (
            df['amount_percentile'] + df['orig_balance_percentile']
        ) / 2
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Apply all feature creation steps
        df = self.create_amount_features(df)
        df = self.create_balance_features(df)
        df = self.create_error_features(df)
        df = self.create_time_features(df)
        df = self.create_transaction_type_features(df)
        df = self.create_ratio_features(df)
        df = self.create_interaction_features(df)
        df = self.create_statistical_features(df)
        
        # Drop unnecessary columns
        columns_to_drop = ['name_orig', 'name_dest', 'is_flagged_fraud']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df


# ============== PREDICTION PIPELINE ==============
class PredictionPipeline:
    """
    Prediction Pipeline for Fraud Detection System.
    
    Responsibilities:
    - Load production model and preprocessor
    - Process input transactions
    - Generate fraud predictions
    - Categorize risk levels
    - Log predictions
    - Support single and batch predictions
    """
    
    def __init__(self, config: PredictionConfig = PredictionConfig()):
        """
        Initialize Prediction Pipeline.
        
        Args:
            config: PredictionConfig object
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.model_version = ""
        self.optimal_threshold = config.default_threshold
        self.feature_engineer = None
        
        self._is_initialized = False
        
        logger.info("Prediction Pipeline initialized")
    
    def initialize(self) -> bool:
        """
        Load model and preprocessor.
        
        Returns:
            True if initialization successful
        """
        logger.info("Initializing Prediction Pipeline...")
        
        try:
            # Check if production model exists
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(
                    f"Production model not found: {self.config.model_path}"
                )
            
            # Load model
            self.model = load_object(self.config.model_path)
            logger.info(f"Model loaded: {self.config.model_path}")
            
            # Load preprocessor
            self.preprocessor = load_object(self.config.preprocessor_path)
            logger.info(f"Preprocessor loaded: {self.config.preprocessor_path}")
            
            # Load feature names
            if os.path.exists(self.config.feature_names_path):
                feature_info = read_json(self.config.feature_names_path)
                self.feature_names = feature_info.get('selected_features', [])
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            
            # Load version info
            if os.path.exists(self.config.version_path):
                version_info = read_json(self.config.version_path)
                self.model_version = version_info.get('version', 'unknown')
                self.optimal_threshold = version_info.get('optimal_threshold', 0.5)
                logger.info(f"Model version: {self.model_version}")
                logger.info(f"Optimal threshold: {self.optimal_threshold}")
            
            # Initialize feature engineer
            self.feature_engineer = PredictionFeatureEngineer(self.feature_names)
            
            # Create logging directories
            if self.config.enable_prediction_logging:
                create_directories([
                    self.config.prediction_logs_dir,
                    self.config.batch_results_dir
                ])
            
            self._is_initialized = True
            logger.info("Prediction Pipeline initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Prediction Pipeline: {str(e)}")
            return False
    
    def _ensure_initialized(self) -> None:
        """Ensure pipeline is initialized."""
        if not self._is_initialized:
            success = self.initialize()
            if not success:
                raise PredictionException(
                    error_message="Failed to initialize Prediction Pipeline",
                    error_detail=sys
                )
    
    def _get_threshold(self) -> float:
        """Get threshold to use for predictions."""
        if self.config.use_optimal_threshold:
            return self.optimal_threshold
        return self.config.default_threshold
    
    def _categorize_risk(self, probability: float) -> str:
        """
        Categorize risk level based on fraud probability.
        
        Args:
            probability: Fraud probability
        
        Returns:
            Risk category string
        """
        if probability >= self.config.high_risk_threshold:
            return "HIGH"
        elif probability >= self.config.medium_risk_threshold:
            return "MEDIUM"
        elif probability >= self.config.low_risk_threshold:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_risk_factors(
        self,
        features: pd.DataFrame,
        prediction_proba: float
    ) -> List[Dict]:
        """
        Get top risk factors contributing to the prediction.
        
        Args:
            features: Engineered features
            prediction_proba: Fraud probability
        
        Returns:
            List of risk factor dictionaries
        """
        risk_factors = []
        
        # Check key risk indicators
        if features['is_account_drain'].values[0] == 1:
            risk_factors.append({
                'factor': 'Account Drain',
                'description': 'Transaction drains entire account balance',
                'severity': 'HIGH'
            })
        
        if features['is_high_risk_type'].values[0] == 1:
            risk_factors.append({
                'factor': 'High Risk Transaction Type',
                'description': 'Transaction type (TRANSFER/CASH_OUT) associated with higher fraud',
                'severity': 'MEDIUM'
            })
        
        if features['is_high_amount'].values[0] == 1:
            risk_factors.append({
                'factor': 'High Amount',
                'description': 'Transaction amount is unusually high',
                'severity': 'MEDIUM'
            })
        
        if features['is_night'].values[0] == 1:
            risk_factors.append({
                'factor': 'Night Transaction',
                'description': 'Transaction occurred during night hours',
                'severity': 'LOW'
            })
        
        if features['amount_exceeds_balance'].values[0] == 1:
            risk_factors.append({
                'factor': 'Amount Exceeds Balance',
                'description': 'Transaction amount exceeds origin account balance',
                'severity': 'HIGH'
            })
        
        if features['has_any_error'].values[0] == 1:
            risk_factors.append({
                'factor': 'Balance Discrepancy',
                'description': 'Balance calculation shows discrepancy',
                'severity': 'MEDIUM'
            })
        
        # Sort by severity
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        risk_factors.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return risk_factors[:5]  # Return top 5
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            data: Input transaction data
        
        Returns:
            Preprocessed feature array
        """
        # Apply feature engineering
        featured_data = self.feature_engineer.engineer_features(data)
        
        # Get target column if exists
        if TARGET_COLUMN in featured_data.columns:
            featured_data = featured_data.drop(columns=[TARGET_COLUMN])
        
        # Align with training features
        missing_cols = set(self.feature_names) - set(featured_data.columns)
        for col in missing_cols:
            featured_data[col] = 0
        
        # Keep only selected features
        if self.feature_names:
            extra_cols = set(featured_data.columns) - set(self.feature_names)
            featured_data = featured_data.drop(columns=list(extra_cols), errors='ignore')
            
            # Ensure correct order
            featured_data = featured_data.reindex(columns=self.feature_names, fill_value=0)
        
        # Apply preprocessing (scaling)
        features_scaled = self.preprocessor.transform(featured_data)
        
        return features_scaled, featured_data
    
    def _log_prediction(self, result: PredictionResult) -> None:
        """
        Log prediction for audit trail.
        
        Args:
            result: Prediction result
        """
        if not self.config.enable_prediction_logging:
            return
        
        # Create daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(
            self.config.prediction_logs_dir,
            f"predictions_{date_str}.json"
        )
        
        # Load existing logs
        logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        
        # Append new prediction
        logs.append(result.to_dict())
        
        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    
    # ==================== SINGLE PREDICTION ====================
    
    def predict_single(
        self,
        transaction: Dict,
        transaction_id: str = None
    ) -> PredictionResult:
        """
        Predict fraud for a single transaction.
        
        Args:
            transaction: Transaction data as dictionary
            transaction_id: Optional transaction ID
        
        Returns:
            PredictionResult object
        """
        import time
        start_time = time.time()
        
        self._ensure_initialized()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction])
            
            # Generate transaction ID if not provided
            if transaction_id is None:
                transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            
            # Prepare features
            features_scaled, features_df = self._prepare_features(df)
            
            # Get prediction
            prediction_proba = self.model.predict_proba(features_scaled)[0, 1]
            threshold = self._get_threshold()
            prediction = int(prediction_proba >= threshold)
            
            # Categorize risk
            risk_category = self._categorize_risk(prediction_proba)
            
            # Get risk factors
            risk_factors = self._get_risk_factors(features_df, prediction_proba)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create result
            result = PredictionResult(
                transaction_id=transaction_id,
                prediction=prediction,
                probability=round(float(prediction_proba), 4),
                risk_score=round(float(prediction_proba * 100), 2),
                risk_category=risk_category,
                is_fraud=bool(prediction),
                threshold_used=threshold,
                top_risk_factors=risk_factors,
                model_version=self.model_version,
                processing_time_ms=round(processing_time, 2)
            )
            
            # Log prediction
            self._log_prediction(result)
            
            return result
            
        except Exception as e:
            raise PredictionException(
                error_message=f"Single prediction failed: {str(e)}",
                error_detail=sys
            ) from e
    
    # ==================== BATCH PREDICTION ====================
    
    def predict_batch(
        self,
        transactions: Union[pd.DataFrame, List[Dict]],
        batch_id: str = None,
        save_results: bool = True
    ) -> BatchPredictionResult:
        """
        Predict fraud for a batch of transactions.
        
        Args:
            transactions: DataFrame or list of transaction dictionaries
            batch_id: Optional batch identifier
            save_results: Whether to save results to file
        
        Returns:
            BatchPredictionResult object
        """
        import time
        start_time = time.time()
        
        self._ensure_initialized()
        
        try:
            # Convert to DataFrame if needed
            if isinstance(transactions, list):
                df = pd.DataFrame(transactions)
            else:
                df = transactions.copy()
            
            # Generate batch ID if not provided
            if batch_id is None:
                batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            logger.info(f"Processing batch {batch_id}: {len(df)} transactions")
            
            # Prepare features
            features_scaled, features_df = self._prepare_features(df)
            
            # Get predictions
            predictions_proba = self.model.predict_proba(features_scaled)[:, 1]
            threshold = self._get_threshold()
            predictions = (predictions_proba >= threshold).astype(int)
            
            # Create results DataFrame
            results_df = df.copy()
            results_df['fraud_probability'] = predictions_proba.round(4)
            results_df['prediction'] = predictions
            results_df['is_fraud'] = predictions.astype(bool)
            results_df['risk_category'] = [
                self._categorize_risk(p) for p in predictions_proba
            ]
            results_df['threshold_used'] = threshold
            results_df['model_version'] = self.model_version
            results_df['prediction_timestamp'] = datetime.now().isoformat()
            
            # Calculate statistics
            fraud_count = int(predictions.sum())
            legitimate_count = len(predictions) - fraud_count
            high_risk = sum(1 for p in predictions_proba if p >= self.config.high_risk_threshold)
            medium_risk = sum(1 for p in predictions_proba if self.config.medium_risk_threshold <= p < self.config.high_risk_threshold)
            low_risk = sum(1 for p in predictions_proba if self.config.low_risk_threshold <= p < self.config.medium_risk_threshold)
            
            # Processing time
            processing_time = time.time() - start_time
            
            # Save results if requested
            if save_results:
                results_path = os.path.join(
                    self.config.batch_results_dir,
                    f"{batch_id}.csv"
                )
                results_df.to_csv(results_path, index=False)
                logger.info(f"Batch results saved: {results_path}")
            
            # Create result object
            result = BatchPredictionResult(
                batch_id=batch_id,
                total_transactions=len(df),
                fraud_count=fraud_count,
                legitimate_count=legitimate_count,
                high_risk_count=high_risk,
                medium_risk_count=medium_risk,
                low_risk_count=low_risk,
                predictions_df=results_df,
                processing_time_seconds=round(processing_time, 2),
                model_version=self.model_version,
                threshold_used=threshold
            )
            
            logger.info(f"Batch {batch_id} processed in {processing_time:.2f}s")
            logger.info(f"  Total: {len(df)}, Fraud: {fraud_count}, Legitimate: {legitimate_count}")
            logger.info(f"  High Risk: {high_risk}, Medium: {medium_risk}, Low: {low_risk}")
            
            return result
            
        except Exception as e:
            raise PredictionException(
                error_message=f"Batch prediction failed: {str(e)}",
                error_detail=sys
            ) from e
    
    # ==================== UTILITY METHODS ====================
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        self._ensure_initialized()
        
        return {
            'model_version': self.model_version,
            'model_type': type(self.model).__name__,
            'optimal_threshold': self.optimal_threshold,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names[:10],  # First 10
            'is_initialized': self._is_initialized
        }
    
    def set_threshold(self, threshold: float) -> None:
        """
        Set custom threshold for predictions.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.optimal_threshold = threshold
        self.config.use_optimal_threshold = True
        logger.info(f"Threshold set to: {threshold}")
    
    def reset_threshold(self) -> None:
        """Reset threshold to model's optimal threshold."""
        if os.path.exists(self.config.version_path):
            version_info = read_json(self.config.version_path)
            self.optimal_threshold = version_info.get('optimal_threshold', 0.5)
        else:
            self.optimal_threshold = 0.5
        
        logger.info(f"Threshold reset to: {self.optimal_threshold}")


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    # Test the prediction pipeline
    print("\n" + "="*60)
    print("TESTING PREDICTION PIPELINE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PredictionPipeline()
    
    if not pipeline.initialize():
        print("Failed to initialize pipeline. Run training pipeline first.")
        sys.exit(1)
    
    # Get model info
    print("\n" + "="*40)
    print("MODEL INFO")
    print("="*40)
    model_info = pipeline.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Test single prediction
    print("\n" + "="*40)
    print("SINGLE TRANSACTION PREDICTION")
    print("="*40)
    
    # Sample legitimate transaction
    legitimate_transaction = {
        'step': 100,
        'type': 'PAYMENT',
        'amount': 1500.0,
        'name_orig': 'C123456',
        'old_balance_org': 50000.0,
        'new_balance_org': 48500.0,
        'name_dest': 'M789012',
        'old_balance_dest': 10000.0,
        'new_balance_dest': 11500.0
    }
    
    result = pipeline.predict_single(legitimate_transaction, "TXN_001")
    
    print(f"\nTransaction: TXN_001 (PAYMENT - $1,500)")
    print(f"  Prediction: {'FRAUD' if result.is_fraud else 'LEGITIMATE'}")
    print(f"  Probability: {result.probability:.4f}")
    print(f"  Risk Score: {result.risk_score}")
    print(f"  Risk Category: {result.risk_category}")
    print(f"  Processing Time: {result.processing_time_ms:.2f} ms")
    
    # Sample suspicious transaction
    suspicious_transaction = {
        'step': 500,
        'type': 'TRANSFER',
        'amount': 450000.0,
        'name_orig': 'C999999',
        'old_balance_org': 450000.0,
        'new_balance_org': 0.0,
        'name_dest': 'C111111',
        'old_balance_dest': 0.0,
        'new_balance_dest': 450000.0
    }
    
    result = pipeline.predict_single(suspicious_transaction, "TXN_002")
    
    print(f"\nTransaction: TXN_002 (TRANSFER - $450,000 - Account Drain)")
    print(f"  Prediction: {'FRAUD' if result.is_fraud else 'LEGITIMATE'}")
    print(f"  Probability: {result.probability:.4f}")
    print(f"  Risk Score: {result.risk_score}")
    print(f"  Risk Category: {result.risk_category}")
    print(f"  Processing Time: {result.processing_time_ms:.2f} ms")
    
    if result.top_risk_factors:
        print(f"\n  Top Risk Factors:")
        for factor in result.top_risk_factors:
            print(f"    - [{factor['severity']}] {factor['factor']}: {factor['description']}")
    
    # Test batch prediction
    print("\n" + "="*40)
    print("BATCH PREDICTION")
    print("="*40)
    
    # Create sample batch
    batch_transactions = [
        {
            'step': 100, 'type': 'PAYMENT', 'amount': 1500.0,
            'name_orig': 'C1', 'old_balance_org': 50000.0, 'new_balance_org': 48500.0,
            'name_dest': 'M1', 'old_balance_dest': 10000.0, 'new_balance_dest': 11500.0
        },
        {
            'step': 200, 'type': 'TRANSFER', 'amount': 200000.0,
            'name_orig': 'C2', 'old_balance_org': 200000.0, 'new_balance_org': 0.0,
            'name_dest': 'C3', 'old_balance_dest': 0.0, 'new_balance_dest': 200000.0
        },
        {
            'step': 300, 'type': 'CASH_OUT', 'amount': 50000.0,
            'name_orig': 'C4', 'old_balance_org': 100000.0, 'new_balance_org': 50000.0,
            'name_dest': 'M2', 'old_balance_dest': 0.0, 'new_balance_dest': 50000.0
        },
        {
            'step': 400, 'type': 'DEBIT', 'amount': 500.0,
            'name_orig': 'C5', 'old_balance_org': 5000.0, 'new_balance_org': 4500.0,
            'name_dest': 'M3', 'old_balance_dest': 0.0, 'new_balance_dest': 500.0
        },
        {
            'step': 500, 'type': 'TRANSFER', 'amount': 500000.0,
            'name_orig': 'C6', 'old_balance_org': 500000.0, 'new_balance_org': 0.0,
            'name_dest': 'C7', 'old_balance_dest': 0.0, 'new_balance_dest': 500000.0
        }
    ]
    
    batch_result = pipeline.predict_batch(batch_transactions, "TEST_BATCH_001")
    
    print(f"\nBatch ID: {batch_result.batch_id}")
    print(f"Total Transactions: {batch_result.total_transactions}")
    print(f"Fraud Detected: {batch_result.fraud_count}")
    print(f"Legitimate: {batch_result.legitimate_count}")
    print(f"High Risk: {batch_result.high_risk_count}")
    print(f"Medium Risk: {batch_result.medium_risk_count}")
    print(f"Low Risk: {batch_result.low_risk_count}")
    print(f"Processing Time: {batch_result.processing_time_seconds:.2f} seconds")
    
    print("\nDetailed Results:")
    print(batch_result.predictions_df[['type', 'amount', 'fraud_probability', 'prediction', 'risk_category']].to_string())
    
    print("\n" + "="*60)
    print("PREDICTION PIPELINE TEST COMPLETED!")
    print("="*60)