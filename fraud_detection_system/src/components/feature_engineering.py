"""
Feature Engineering Component for Fraud Detection System

Features Created:
1. Amount-based Features (ratios, percentiles, log transforms)
2. Balance Features (differences, drain patterns, error detection)
3. Velocity Features (transaction frequency patterns)
4. Time-based Features (hour, day, weekend patterns)
5. Frequency Features (transaction counts, averages)
6. Error Features (balance discrepancies)
7. Ratio Features (various financial ratios)
8. Interaction Features (combined features)
"""

import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
import pandas as pd
import numpy as np
from scipy import stats

from src.entity.config_entity import FeatureEngineeringConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    FeatureEngineeringArtifact
)
from src.logger import logger, log_dataframe_info
from src.exception import FeatureEngineeringException
from src.utils.common import write_json, create_directories
from src.constants import ARTIFACTS_DIR, TARGET_COLUMN


class FeatureEngineering:
    """
    Feature Engineering Component for Fraud Detection System.
    
    Creates domain-specific features for fraud detection including:
    - Transaction amount patterns
    - Account balance patterns
    - Transaction velocity
    - Time-based patterns
    - Error/discrepancy detection
    """
    
    def __init__(
        self,
        config: FeatureEngineeringConfig = FeatureEngineeringConfig(),
        data_ingestion_artifact: DataIngestionArtifact = None,
        data_validation_artifact: DataValidationArtifact = None
    ):
        """
        Initialize Feature Engineering component.
        
        Args:
            config: FeatureEngineeringConfig object
            data_ingestion_artifact: Artifact from data ingestion
            data_validation_artifact: Artifact from data validation
        """
        self.config = config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        
        self.original_features: List[str] = []
        self.engineered_features: List[str] = []
        self.feature_report: Dict = {}
        
        logger.info(f"{'='*60}")
        logger.info("Initializing Feature Engineering Component")
        logger.info(f"{'='*60}")
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data."""
        logger.info("Loading train and test data...")
        
        train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
        test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
        
        self.original_features = [
            col for col in train_df.columns 
            if col != self.config.target_column
        ]
        
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")
        logger.info(f"Original features: {len(self.original_features)}")
        
        return train_df, test_df
    
    # ==================== AMOUNT-BASED FEATURES ====================
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features.
        
        Features:
        - Log transform of amount
        - Amount percentile
        - Amount to balance ratios
        - Amount categories
        """
        logger.info("Creating amount-based features...")
        
        features_created = []
        
        # 1. Log Transform (handle skewness)
        df['amount_log'] = np.log1p(df['amount'])
        features_created.append('amount_log')
        
        # 2. Amount Squared (capture non-linear patterns)
        df['amount_squared'] = df['amount'] ** 2
        features_created.append('amount_squared')
        
        # 3. Amount to Origin Balance Ratio
        df['amount_to_orig_balance_ratio'] = np.where(
            df['old_balance_org'] > 0,
            df['amount'] / df['old_balance_org'],
            0
        )
        features_created.append('amount_to_orig_balance_ratio')
        
        # 4. Amount to Destination Balance Ratio
        df['amount_to_dest_balance_ratio'] = np.where(
            df['old_balance_dest'] > 0,
            df['amount'] / df['old_balance_dest'],
            0
        )
        features_created.append('amount_to_dest_balance_ratio')
        
        # 5. Amount Categories (binning)
        amount_bins = [0, 1000, 10000, 50000, 100000, 500000, np.inf]
        amount_labels = ['very_small', 'small', 'medium', 'large', 'very_large', 'huge']
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=amount_bins, 
            labels=amount_labels
        ).astype(str)
        features_created.append('amount_category')
        
        # 6. Is High Amount (above 95th percentile of training data)
        high_amount_threshold = df['amount'].quantile(0.95)
        df['is_high_amount'] = (df['amount'] > high_amount_threshold).astype(int)
        features_created.append('is_high_amount')
        
        # 7. Amount Deviation from Mean
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        df['amount_zscore'] = (df['amount'] - amount_mean) / (amount_std + 1e-10)
        features_created.append('amount_zscore')
        
        self.engineered_features.extend(features_created)
        self.feature_report['amount_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} amount features")
        return df
    
    # ==================== BALANCE FEATURES ====================
    
    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create balance-based features.
        
        Features:
        - Balance differences
        - Account drain detection
        - Balance change ratios
        - Zero balance flags
        """
        logger.info("Creating balance-based features...")
        
        features_created = []
        
        # 1. Origin Balance Difference
        df['orig_balance_diff'] = df['old_balance_org'] - df['new_balance_org']
        features_created.append('orig_balance_diff')
        
        # 2. Destination Balance Difference
        df['dest_balance_diff'] = df['new_balance_dest'] - df['old_balance_dest']
        features_created.append('dest_balance_diff')
        
        # 3. Origin Balance Change Percentage
        df['orig_balance_change_pct'] = np.where(
            df['old_balance_org'] > 0,
            ((df['old_balance_org'] - df['new_balance_org']) / df['old_balance_org']) * 100,
            0
        )
        features_created.append('orig_balance_change_pct')
        
        # 4. Destination Balance Change Percentage
        df['dest_balance_change_pct'] = np.where(
            df['old_balance_dest'] > 0,
            ((df['new_balance_dest'] - df['old_balance_dest']) / df['old_balance_dest']) * 100,
            0
        )
        features_created.append('dest_balance_change_pct')
        
        # 5. Account Drain Flag (Origin account emptied)
        df['is_account_drain'] = (
            (df['new_balance_org'] == 0) & 
            (df['old_balance_org'] > 0) &
            (df['amount'] > 0)
        ).astype(int)
        features_created.append('is_account_drain')
        
        # 6. Full Amount Transfer (Amount equals old balance)
        df['is_full_transfer'] = (
            np.abs(df['amount'] - df['old_balance_org']) < 1
        ).astype(int)
        features_created.append('is_full_transfer')
        
        # 7. Zero Origin Balance Flag
        df['is_zero_orig_balance'] = (df['old_balance_org'] == 0).astype(int)
        features_created.append('is_zero_orig_balance')
        
        # 8. Zero Destination Balance Flag
        df['is_zero_dest_balance'] = (df['old_balance_dest'] == 0).astype(int)
        features_created.append('is_zero_dest_balance')
        
        # 9. Total Balance (Origin + Destination before transaction)
        df['total_balance_before'] = df['old_balance_org'] + df['old_balance_dest']
        features_created.append('total_balance_before')
        
        # 10. Balance Ratio (Origin to Destination)
        df['balance_ratio_orig_dest'] = np.where(
            df['old_balance_dest'] > 0,
            df['old_balance_org'] / (df['old_balance_dest'] + 1),
            df['old_balance_org']
        )
        features_created.append('balance_ratio_orig_dest')
        
        # 11. Log transforms of balances
        df['old_balance_org_log'] = np.log1p(df['old_balance_org'])
        df['new_balance_org_log'] = np.log1p(df['new_balance_org'])
        df['old_balance_dest_log'] = np.log1p(df['old_balance_dest'])
        df['new_balance_dest_log'] = np.log1p(df['new_balance_dest'])
        features_created.extend([
            'old_balance_org_log', 'new_balance_org_log',
            'old_balance_dest_log', 'new_balance_dest_log'
        ])
        
        self.engineered_features.extend(features_created)
        self.feature_report['balance_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} balance features")
        return df
    
    # ==================== ERROR FEATURES ====================
    
    def create_error_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create error/discrepancy detection features.
        
        Features:
        - Balance calculation errors
        - Transaction consistency checks
        - Suspicious patterns
        """
        logger.info("Creating error detection features...")
        
        features_created = []
        
        # 1. Origin Balance Error
        # Expected: new_balance_org = old_balance_org - amount
        df['orig_balance_error'] = (
            df['new_balance_org'] - (df['old_balance_org'] - df['amount'])
        )
        features_created.append('orig_balance_error')
        
        # 2. Has Origin Error Flag
        df['has_orig_error'] = (np.abs(df['orig_balance_error']) > 1).astype(int)
        features_created.append('has_orig_error')
        
        # 3. Destination Balance Error
        # Expected: new_balance_dest = old_balance_dest + amount
        df['dest_balance_error'] = (
            df['new_balance_dest'] - (df['old_balance_dest'] + df['amount'])
        )
        features_created.append('dest_balance_error')
        
        # 4. Has Destination Error Flag
        df['has_dest_error'] = (np.abs(df['dest_balance_error']) > 1).astype(int)
        features_created.append('has_dest_error')
        
        # 5. Total Error Flag (Either origin or destination has error)
        df['has_any_error'] = (
            (df['has_orig_error'] == 1) | (df['has_dest_error'] == 1)
        ).astype(int)
        features_created.append('has_any_error')
        
        # 6. Suspicious Zero Balance After High Transaction
        # Origin had balance, made large transaction, now has exactly 0
        df['suspicious_zero_after'] = (
            (df['old_balance_org'] > 10000) &
            (df['amount'] > 10000) &
            (df['new_balance_org'] == 0)
        ).astype(int)
        features_created.append('suspicious_zero_after')
        
        # 7. Large Transaction from New Account (0 initial balance)
        df['large_from_zero'] = (
            (df['old_balance_org'] == 0) &
            (df['amount'] > 10000)
        ).astype(int)
        features_created.append('large_from_zero')
        
        # 8. Amount Exceeds Balance (Impossible in normal cases)
        df['amount_exceeds_balance'] = (
            df['amount'] > df['old_balance_org']
        ).astype(int)
        features_created.append('amount_exceeds_balance')
        
        self.engineered_features.extend(features_created)
        self.feature_report['error_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} error features")
        return df
    
    # ==================== TIME-BASED FEATURES ====================
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Features:
        - Hour of day
        - Day of week
        - Weekend flag
        - Time period categories
        
        Note: 'step' represents hours from simulation start
        """
        logger.info("Creating time-based features...")
        
        features_created = []
        
        if 'step' not in df.columns:
            logger.warning("'step' column not found. Skipping time features.")
            return df
        
        # 1. Hour of Day (step modulo 24)
        df['hour_of_day'] = df['step'] % 24
        features_created.append('hour_of_day')
        
        # 2. Day Number (step divided by 24)
        df['day_number'] = df['step'] // 24
        features_created.append('day_number')
        
        # 3. Day of Week (assuming simulation starts on Monday)
        df['day_of_week'] = (df['step'] // 24) % 7
        features_created.append('day_of_week')
        
        # 4. Is Weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        features_created.append('is_weekend')
        
        # 5. Is Night (10 PM to 6 AM)
        df['is_night'] = (
            (df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)
        ).astype(int)
        features_created.append('is_night')
        
        # 6. Is Business Hours (9 AM to 5 PM)
        df['is_business_hours'] = (
            (df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)
        ).astype(int)
        features_created.append('is_business_hours')
        
        # 7. Time Period Category
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
        features_created.append('time_period')
        
        # 8. Week Number
        df['week_number'] = df['step'] // (24 * 7)
        features_created.append('week_number')
        
        # 9. Cyclical Hour Encoding (Sine and Cosine)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        features_created.extend(['hour_sin', 'hour_cos'])
        
        # 10. Cyclical Day Encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        features_created.extend(['day_sin', 'day_cos'])
        
        self.engineered_features.extend(features_created)
        self.feature_report['time_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} time features")
        return df
    
    # ==================== TRANSACTION TYPE FEATURES ====================
    
    def create_transaction_type_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction type-based features.
        
        Features:
        - High-risk transaction type flags
        - Type-specific patterns
        """
        logger.info("Creating transaction type features...")
        
        features_created = []
        
        if 'type' not in df.columns:
            logger.warning("'type' column not found. Skipping type features.")
            return df
        
        # 1. High Risk Transaction Type (TRANSFER and CASH_OUT are fraud-prone)
        df['is_high_risk_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
        features_created.append('is_high_risk_type')
        
        # 2. Is Transfer
        df['is_transfer'] = (df['type'] == 'TRANSFER').astype(int)
        features_created.append('is_transfer')
        
        # 3. Is Cash Out
        df['is_cash_out'] = (df['type'] == 'CASH_OUT').astype(int)
        features_created.append('is_cash_out')
        
        # 4. Is Payment
        df['is_payment'] = (df['type'] == 'PAYMENT').astype(int)
        features_created.append('is_payment')
        
        # 5. Is Cash In
        df['is_cash_in'] = (df['type'] == 'CASH_IN').astype(int)
        features_created.append('is_cash_in')
        
        # 6. Is Debit
        df['is_debit'] = (df['type'] == 'DEBIT').astype(int)
        features_created.append('is_debit')
        
        # 7. High Risk + High Amount Combination
        df['high_risk_high_amount'] = (
            (df['is_high_risk_type'] == 1) & 
            (df['amount'] > df['amount'].quantile(0.90))
        ).astype(int)
        features_created.append('high_risk_high_amount')
        
        # 8. High Risk + Account Drain Combination
        df['high_risk_drain'] = (
            (df['is_high_risk_type'] == 1) & 
            (df['is_account_drain'] == 1)
        ).astype(int)
        features_created.append('high_risk_drain')
        
        self.engineered_features.extend(features_created)
        self.feature_report['type_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} transaction type features")
        return df
    
    # ==================== RATIO FEATURES ====================
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create various ratio features.
        
        Features:
        - Financial ratios
        - Comparative ratios
        """
        logger.info("Creating ratio features...")
        
        features_created = []
        
        # 1. Amount to Total Balance Ratio
        total_balance = df['old_balance_org'] + df['old_balance_dest']
        df['amount_to_total_balance_ratio'] = np.where(
            total_balance > 0,
            df['amount'] / total_balance,
            0
        )
        features_created.append('amount_to_total_balance_ratio')
        
        # 2. Origin to Destination New Balance Ratio
        df['new_balance_ratio'] = np.where(
            df['new_balance_dest'] > 0,
            df['new_balance_org'] / (df['new_balance_dest'] + 1),
            df['new_balance_org']
        )
        features_created.append('new_balance_ratio')
        
        # 3. Balance Change Ratio (Origin vs Destination)
        df['balance_change_ratio'] = np.where(
            np.abs(df['dest_balance_diff']) > 0,
            df['orig_balance_diff'] / (np.abs(df['dest_balance_diff']) + 1),
            0
        )
        features_created.append('balance_change_ratio')
        
        # 4. Transaction Size Relative to Account
        df['relative_transaction_size'] = np.where(
            df['old_balance_org'] > 0,
            (df['amount'] / df['old_balance_org']) * 100,
            0
        ).clip(0, 1000)  # Cap at 1000%
        features_created.append('relative_transaction_size')
        
        # 5. Origin Balance Retention Rate
        df['balance_retention_rate'] = np.where(
            df['old_balance_org'] > 0,
            (df['new_balance_org'] / df['old_balance_org']) * 100,
            0
        ).clip(0, 100)
        features_created.append('balance_retention_rate')
        
        self.engineered_features.extend(features_created)
        self.feature_report['ratio_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} ratio features")
        return df
    
    # ==================== INTERACTION FEATURES ====================
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.
        
        Features:
        - Combined risk indicators
        - Multiplicative features
        """
        logger.info("Creating interaction features...")
        
        features_created = []
        
        # 1. Risk Score (Combination of risk factors)
        df['risk_score'] = (
            df['is_high_risk_type'] * 2 +
            df['is_account_drain'] * 3 +
            df['is_high_amount'] * 1 +
            df['is_night'] * 1 +
            df['has_any_error'] * 2 +
            df['amount_exceeds_balance'] * 2
        )
        features_created.append('risk_score')
        
        # 2. Amount * Balance Interaction
        df['amount_balance_interaction'] = (
            df['amount_log'] * df['old_balance_org_log']
        )
        features_created.append('amount_balance_interaction')
        
        # 3. Time * Amount Interaction
        df['time_amount_interaction'] = df['hour_of_day'] * df['amount_log']
        features_created.append('time_amount_interaction')
        
        # 4. Type Risk * Amount (Encoded)
        df['type_amount_risk'] = df['is_high_risk_type'] * df['amount_log']
        features_created.append('type_amount_risk')
        
        # 5. Weekend Night Transaction
        df['weekend_night'] = (
            (df['is_weekend'] == 1) & (df['is_night'] == 1)
        ).astype(int)
        features_created.append('weekend_night')
        
        # 6. High Risk Combination Score
        df['high_risk_combo'] = (
            (df['is_high_risk_type'] == 1) &
            (df['is_account_drain'] == 1) &
            (df['amount'] > df['amount'].median())
        ).astype(int)
        features_created.append('high_risk_combo')
        
        # 7. Suspicious Transaction Flag
        df['is_suspicious'] = (
            (df['risk_score'] >= 4) |
            (df['high_risk_combo'] == 1) |
            (df['suspicious_zero_after'] == 1)
        ).astype(int)
        features_created.append('is_suspicious')
        
        self.engineered_features.extend(features_created)
        self.feature_report['interaction_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} interaction features")
        return df
    
    # ==================== STATISTICAL FEATURES ====================
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical aggregation features.
        
        Features:
        - Percentile ranks
        - Standardized values
        """
        logger.info("Creating statistical features...")
        
        features_created = []
        
        # 1. Amount Percentile Rank
        df['amount_percentile'] = df['amount'].rank(pct=True)
        features_created.append('amount_percentile')
        
        # 2. Balance Percentile Rank
        df['orig_balance_percentile'] = df['old_balance_org'].rank(pct=True)
        features_created.append('orig_balance_percentile')
        
        # 3. Combined Amount-Balance Percentile
        df['amount_balance_percentile'] = (
            df['amount_percentile'] + df['orig_balance_percentile']
        ) / 2
        features_created.append('amount_balance_percentile')
        
        self.engineered_features.extend(features_created)
        self.feature_report['statistical_features'] = {
            'count': len(features_created),
            'features': features_created
        }
        
        logger.info(f"Created {len(features_created)} statistical features")
        return df
    
    # ==================== DROP UNNECESSARY COLUMNS ====================
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop identifier and unnecessary columns.
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame with dropped columns
        """
        logger.info("Dropping unnecessary columns...")
        
        columns_to_drop = []
        
        for col in self.config.columns_to_drop:
            if col in df.columns:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
        
        self.feature_report['dropped_columns'] = columns_to_drop
        
        return df
    
    # ==================== HANDLE CATEGORICAL ENCODING ====================
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for model training.
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame with encoded categoricals
        """
        logger.info("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return df
        
        # One-Hot Encoding for categorical columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        
        # Update engineered features list with new encoded columns
        new_cols = [col for col in df.columns if col not in self.original_features]
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        logger.info(f"New columns after encoding: {len(new_cols)}")
        
        return df
    
    # ==================== FEATURE ENGINEERING PIPELINE ====================
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Store original shape
        original_shape = df.shape
        
        # Apply feature engineering steps
        if self.config.create_amount_features:
            df = self.create_amount_features(df)
        
        if self.config.create_balance_features:
            df = self.create_balance_features(df)
        
        if self.config.create_error_features:
            df = self.create_error_features(df)
        
        if self.config.create_time_features:
            df = self.create_time_features(df)
        
        # Transaction type features (requires balance features first)
        df = self.create_transaction_type_features(df)
        
        if self.config.create_ratio_features:
            df = self.create_ratio_features(df)
        
        # Interaction features (requires other features first)
        df = self.create_interaction_features(df)
        
        # Statistical features
        df = self.create_statistical_features(df)
        
        # Drop unnecessary columns
        df = self.drop_unnecessary_columns(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Handle any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with 0 (or median for numerical)
        df = df.fillna(0)
        
        final_shape = df.shape
        
        logger.info(f"Feature engineering complete:")
        logger.info(f"  Original shape: {original_shape}")
        logger.info(f"  Final shape: {final_shape}")
        logger.info(f"  New features added: {final_shape[1] - original_shape[1]}")
        
        return df
    
    # ==================== MAIN EXECUTION ====================
    
    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        """
        Execute complete feature engineering pipeline.
        
        Returns:
            FeatureEngineeringArtifact: Feature engineering results
        """
        logger.info(f"{'='*60}")
        logger.info("Starting Feature Engineering Pipeline")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Create directories
            logger.info("Step 1: Creating directories")
            create_directories([self.config.feature_engineering_dir])
            
            # Step 2: Load data
            logger.info("Step 2: Loading data")
            train_df, test_df = self._load_data()
            
            # Step 3: Engineer features for training data
            logger.info("Step 3: Engineering features for training data")
            train_featured = self.engineer_features(train_df.copy())
            
            # Step 4: Engineer features for test data
            logger.info("Step 4: Engineering features for test data")
            # Reset engineered features list to avoid duplicates
            test_engineered = self.engineered_features.copy()
            self.engineered_features = []
            test_featured = self.engineer_features(test_df.copy())
            self.engineered_features = test_engineered
            
            # Step 5: Align columns (ensure same columns in train and test)
            logger.info("Step 5: Aligning train and test columns")
            train_cols = set(train_featured.columns)
            test_cols = set(test_featured.columns)
            
            # Add missing columns with 0
            for col in train_cols - test_cols:
                test_featured[col] = 0
            
            for col in test_cols - train_cols:
                train_featured[col] = 0
            
            # Ensure same column order
            common_cols = sorted(list(train_cols | test_cols))
            train_featured = train_featured[common_cols]
            test_featured = test_featured[common_cols]
            
            # Step 6: Save featured data
            logger.info("Step 6: Saving featured data")
            train_featured.to_csv(self.config.train_featured_path, index=False)
            test_featured.to_csv(self.config.test_featured_path, index=False)
            
            # Step 7: Create report
            logger.info("Step 7: Creating feature engineering report")
            final_features = [
                col for col in train_featured.columns 
                if col != self.config.target_column
            ]
            
            self.feature_report['summary'] = {
                'original_feature_count': len(self.original_features),
                'engineered_feature_count': len(self.engineered_features),
                'final_feature_count': len(final_features),
                'train_shape': list(train_featured.shape),
                'test_shape': list(test_featured.shape)
            }
            
            write_json(self.config.report_path, self.feature_report)
            
            # Create artifact
            artifact = FeatureEngineeringArtifact(
                train_featured_path=self.config.train_featured_path,
                test_featured_path=self.config.test_featured_path,
                report_path=self.config.report_path,
                original_features=self.original_features,
                engineered_features=self.engineered_features,
                final_features=final_features,
                dropped_features=self.feature_report.get('dropped_columns', []),
                original_feature_count=len(self.original_features),
                engineered_feature_count=len(self.engineered_features),
                final_feature_count=len(final_features),
                train_shape=train_featured.shape,
                test_shape=test_featured.shape
            )
            
            logger.info(f"{'='*60}")
            logger.info("FEATURE ENGINEERING COMPLETED!")
            logger.info(f"{'='*60}")
            logger.info(f"Original Features: {len(self.original_features)}")
            logger.info(f"Engineered Features: {len(self.engineered_features)}")
            logger.info(f"Final Features: {len(final_features)}")
            logger.info(f"Train Shape: {train_featured.shape}")
            logger.info(f"Test Shape: {test_featured.shape}")
            
            return artifact
            
        except Exception as e:
            raise FeatureEngineeringException(
                error_message=f"Feature Engineering failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    
    # Run Data Ingestion
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
    
    # Run Feature Engineering
    fe_config = FeatureEngineeringConfig()
    feature_engineering = FeatureEngineering(
        config=fe_config,
        data_ingestion_artifact=ingestion_artifact,
        data_validation_artifact=validation_artifact
    )
    
    fe_artifact = feature_engineering.initiate_feature_engineering()
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING ARTIFACT SUMMARY")
    print("="*60)
    print(f"Train Featured Path: {fe_artifact.train_featured_path}")
    print(f"Test Featured Path: {fe_artifact.test_featured_path}")
    print(f"Report Path: {fe_artifact.report_path}")
    print(f"\nOriginal Features: {fe_artifact.original_feature_count}")
    print(f"Engineered Features: {fe_artifact.engineered_feature_count}")
    print(f"Final Features: {fe_artifact.final_feature_count}")
    print(f"\nTrain Shape: {fe_artifact.train_shape}")
    print(f"Test Shape: {fe_artifact.test_shape}")
    print(f"\nEngineered Feature Categories:")
    
    # Load and display report
    import json
    with open(fe_artifact.report_path, 'r') as f:
        report = json.load(f)
    
    for category, details in report.items():
        if isinstance(details, dict) and 'count' in details:
            print(f"  - {category}: {details['count']} features")