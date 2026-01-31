"""
Data Transformation Component for Fraud Detection System

Transformations Applied:
1. Missing Value Imputation
2. Outlier Handling (Clipping/Transformation)
3. Feature Scaling (Standard/MinMax/Robust)
4. Feature Selection (Variance/Correlation/Importance)
5. Class Imbalance Handling (SMOTE/ADASYN/BorderlineSMOTE)
6. Pipeline Creation for Production
"""

import os
import sys
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import asdict
import pandas as pd
import numpy as np
import json
from scipy import stats

# Sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif
)

# Imbalanced-learn imports
from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

# Project imports
from src.entity.config_entity import DataTransformationConfig, FeatureEngineeringConfig
from src.entity.artifact_entity import (
    FeatureEngineeringArtifact,
    DataTransformationArtifact
)
from src.logger import logger
from src.exception import DataTransformationException
from src.utils.common import (
    write_json,
    read_json,
    save_object,
    load_object,
    create_directories
)
from src.constants import ARTIFACTS_DIR, TARGET_COLUMN


# ============== CUSTOM TRANSFORMERS ==============

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer to handle outliers using clipping or transformation.
    """
    
    def __init__(
        self,
        method: str = "clip",
        threshold: float = 3.0,
        strategy: str = "zscore"
    ):
        """
        Initialize OutlierHandler.
        
        Args:
            method: "clip", "remove", or "transform"
            threshold: Z-score or IQR multiplier threshold
            strategy: "zscore" or "iqr"
        """
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
    
    def fit(self, X: np.ndarray, y=None):
        """Fit the outlier handler by computing bounds."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx]
            
            if self.strategy == "zscore":
                mean = col_data.mean()
                std = col_data.std()
                self.lower_bounds_[col_idx] = mean - (self.threshold * std)
                self.upper_bounds_[col_idx] = mean + (self.threshold * std)
            else:  # IQR
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds_[col_idx] = Q1 - (self.threshold * IQR)
                self.upper_bounds_[col_idx] = Q3 + (self.threshold * IQR)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by handling outliers."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        if self.method == "clip":
            for col_idx in range(X.shape[1]):
                X.iloc[:, col_idx] = X.iloc[:, col_idx].clip(
                    lower=self.lower_bounds_[col_idx],
                    upper=self.upper_bounds_[col_idx]
                )
        
        return X.values


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove highly correlated features.
    """
    
    def __init__(self, threshold: float = 0.95):
        """
        Initialize CorrelationFilter.
        
        Args:
            threshold: Correlation threshold for dropping features
        """
        self.threshold = threshold
        self.features_to_drop_ = []
        self.feature_names_ = None
    
    def fit(self, X: np.ndarray, y=None):
        """Fit by identifying highly correlated features."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        self.features_to_drop_ = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > self.threshold)
        ]
        
        # Convert to indices
        self.drop_indices_ = [
            X.columns.get_loc(col) for col in self.features_to_drop_
            if col in X.columns
        ]
        
        self.feature_names_ = [
            col for col in X.columns 
            if col not in self.features_to_drop_
        ]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform by removing correlated features."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Keep columns not in drop list
        keep_indices = [
            i for i in range(X.shape[1]) 
            if i not in self.drop_indices_
        ]
        
        return X.iloc[:, keep_indices].values


# ============== MAIN DATA TRANSFORMATION CLASS ==============

class DataTransformation:
    """
    Data Transformation Component for Fraud Detection System.
    
    Responsibilities:
    - Handle missing values
    - Handle outliers
    - Scale features
    - Remove correlated features
    - Handle class imbalance
    - Create reusable preprocessing pipeline
    """
    
    def __init__(
        self,
        config: DataTransformationConfig = DataTransformationConfig(),
        feature_engineering_artifact: FeatureEngineeringArtifact = None
    ):
        """
        Initialize Data Transformation component.
        
        Args:
            config: DataTransformationConfig object
            feature_engineering_artifact: Artifact from feature engineering
        """
        self.config = config
        self.feature_engineering_artifact = feature_engineering_artifact
        
        self.preprocessor = None
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.dropped_features: List[str] = []
        self.transformation_report: Dict = {}
        
        logger.info(f"{'='*60}")
        logger.info("Initializing Data Transformation Component")
        logger.info(f"{'='*60}")
    
    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load featured train and test data."""
        logger.info("Loading featured data...")
        
        train_df = pd.read_csv(self.feature_engineering_artifact.train_featured_path)
        test_df = pd.read_csv(self.feature_engineering_artifact.test_featured_path)
        
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")
        
        return train_df, test_df
    
    def _separate_features_target(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        target = df[self.config.target_column]
        features = df.drop(columns=[self.config.target_column])
        
        return features, target
    
    # ==================== MISSING VALUE HANDLING ====================
    
    def handle_missing_values(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values in the data.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of imputed train and test DataFrames
        """
        logger.info("Handling missing values...")
        
        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        
        logger.info(f"Missing values - Train: {train_missing}, Test: {test_missing}")
        
        if train_missing == 0 and test_missing == 0:
            logger.info("No missing values found. Skipping imputation.")
            return X_train, X_test
        
        # Use median imputation for numerical columns (robust to outliers)
        imputer = SimpleImputer(strategy='median')
        
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_imputed = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.transformation_report['missing_values'] = {
            'train_missing_before': int(train_missing),
            'test_missing_before': int(test_missing),
            'imputation_strategy': 'median'
        }
        
        logger.info("Missing values handled successfully")
        return X_train_imputed, X_test_imputed
    
    # ==================== OUTLIER HANDLING ====================
    
    def handle_outliers(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle outliers using clipping based on training data statistics.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of outlier-handled train and test DataFrames
        """
        if not self.config.handle_outliers:
            logger.info("Outlier handling disabled. Skipping.")
            return X_train, X_test
        
        logger.info(f"Handling outliers using {self.config.outlier_method} method...")
        
        outlier_handler = OutlierHandler(
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold,
            strategy="iqr"
        )
        
        # Fit on training data
        outlier_handler.fit(X_train)
        
        # Transform both datasets
        X_train_handled = pd.DataFrame(
            outlier_handler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_handled = pd.DataFrame(
            outlier_handler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.transformation_report['outlier_handling'] = {
            'method': self.config.outlier_method,
            'threshold': self.config.outlier_threshold,
            'strategy': 'iqr'
        }
        
        logger.info("Outliers handled successfully")
        return X_train_handled, X_test_handled
    
    # ==================== FEATURE SELECTION ====================
    
    def remove_low_variance_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Remove features with low variance.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of filtered train, test DataFrames and dropped column names
        """
        logger.info(f"Removing low variance features (threshold: {self.config.variance_threshold})...")
        
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        selector.fit(X_train)
        
        # Get selected feature mask
        selected_mask = selector.get_support()
        selected_features = X_train.columns[selected_mask].tolist()
        dropped_features = X_train.columns[~selected_mask].tolist()
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        logger.info(f"Dropped {len(dropped_features)} low variance features")
        logger.info(f"Remaining features: {len(selected_features)}")
        
        return X_train_selected, X_test_selected, dropped_features
    
    def remove_correlated_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of filtered train, test DataFrames and dropped column names
        """
        logger.info(f"Removing highly correlated features (threshold: {self.config.correlation_threshold})...")
        
        # Compute correlation matrix
        corr_matrix = X_train.corr().abs()
        
        # Get upper triangle
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        features_to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > self.config.correlation_threshold)
        ]
        
        # Drop features
        X_train_filtered = X_train.drop(columns=features_to_drop)
        X_test_filtered = X_test.drop(columns=features_to_drop)
        
        logger.info(f"Dropped {len(features_to_drop)} correlated features")
        logger.info(f"Remaining features: {len(X_train_filtered.columns)}")
        
        return X_train_filtered, X_test_filtered, features_to_drop
    
    def select_top_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Select top K features based on mutual information.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
        
        Returns:
            Tuple of filtered train, test DataFrames and selected feature names
        """
        if X_train.shape[1] <= self.config.top_k_features:
            logger.info(f"Features ({X_train.shape[1]}) <= top_k ({self.config.top_k_features}). Skipping selection.")
            return X_train, X_test, list(X_train.columns)
        
        logger.info(f"Selecting top {self.config.top_k_features} features...")
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=self.config.top_k_features)
        selector.fit(X_train, y_train)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X_train.columns[selected_mask].tolist()
        
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Log feature importance scores
        scores = selector.scores_
        feature_scores = pd.DataFrame({
            'feature': X_train.columns,
            'score': scores
        }).sort_values('score', ascending=False)
        
        logger.info(f"Top 10 features by importance:")
        for idx, row in feature_scores.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['score']:.4f}")
        
        self.transformation_report['feature_importance'] = {
            'top_10': feature_scores.head(10).to_dict('records')
        }
        
        return X_train_selected, X_test_selected, selected_features
    
    def apply_feature_selection(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply all feature selection steps.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
        
        Returns:
            Tuple of selected train and test DataFrames
        """
        if not self.config.enable_feature_selection:
            logger.info("Feature selection disabled. Skipping.")
            self.selected_features = list(X_train.columns)
            return X_train, X_test
        
        logger.info("Applying feature selection pipeline...")
        
        all_dropped = []
        
        # Step 1: Remove low variance features
        X_train, X_test, dropped_low_var = self.remove_low_variance_features(X_train, X_test)
        all_dropped.extend(dropped_low_var)
        
        # Step 2: Remove highly correlated features
        X_train, X_test, dropped_corr = self.remove_correlated_features(X_train, X_test)
        all_dropped.extend(dropped_corr)
        
        # Step 3: Select top K features
        X_train, X_test, selected = self.select_top_features(X_train, X_test, y_train)
        
        self.selected_features = selected
        self.dropped_features = all_dropped
        
        self.transformation_report['feature_selection'] = {
            'low_variance_dropped': len(dropped_low_var),
            'correlation_dropped': len(dropped_corr),
            'final_features': len(selected),
            'total_dropped': len(all_dropped)
        }
        
        logger.info(f"Feature selection complete. Final features: {len(selected)}")
        
        return X_train, X_test
    
    # ==================== SCALING ====================
    
    def get_scaler(self) -> BaseEstimator:
        """
        Get the appropriate scaler based on configuration.
        
        Returns:
            Scaler object
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),  # Best for data with outliers
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        scaler = scalers.get(self.config.scaling_method, StandardScaler())
        logger.info(f"Using {self.config.scaling_method} scaler")
        
        return scaler
    
    def apply_scaling(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply feature scaling.
        
        Args:
            X_train: Training features
            X_test: Test features
        
        Returns:
            Tuple of scaled train and test arrays
        """
        logger.info(f"Applying {self.config.scaling_method} scaling...")
        
        scaler = self.get_scaler()
        
        # Fit on training data, transform both
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler in preprocessor
        self.preprocessor = scaler
        
        self.transformation_report['scaling'] = {
            'method': self.config.scaling_method,
            'train_shape': list(X_train_scaled.shape),
            'test_shape': list(X_test_scaled.shape)
        }
        
        logger.info("Scaling applied successfully")
        
        return X_train_scaled, X_test_scaled
    
    # ==================== CLASS IMBALANCE HANDLING ====================
    
    def get_resampler(self):
        """
        Get the appropriate resampling strategy.
        
        Returns:
            Resampler object
        """
        resamplers = {
            'smote': SMOTE(
                sampling_strategy=self.config.sampling_strategy,
                random_state=42,
                k_neighbors=5
            ),
            'adasyn': ADASYN(
                sampling_strategy=self.config.sampling_strategy,
                random_state=42
            ),
            'borderline_smote': BorderlineSMOTE(
                sampling_strategy=self.config.sampling_strategy,
                random_state=42,
                kind='borderline-1'
            ),
            'svmsmote': SVMSMOTE(
                sampling_strategy=self.config.sampling_strategy,
                random_state=42
            ),
            'smoteenn': SMOTEENN(
                sampling_strategy=self.config.sampling_strategy,
                random_state=42
            ),
            'smotetomek': SMOTETomek(
                sampling_strategy=self.config.sampling_strategy,
                random_state=42
            ),
            'undersampling': RandomUnderSampler(
                sampling_strategy=self.config.sampling_strategy,
                random_state=42
            )
        }
        
        resampler = resamplers.get(self.config.imbalance_method)
        
        if resampler is None:
            logger.warning(f"Unknown resampling method: {self.config.imbalance_method}. Using SMOTE.")
            resampler = resamplers['smote']
        
        return resampler
    
    def handle_class_imbalance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using resampling techniques.
        
        Args:
            X_train: Training features (scaled)
            y_train: Training target
        
        Returns:
            Tuple of resampled features and target
        """
        if not self.config.handle_imbalance:
            logger.info("Imbalance handling disabled. Skipping.")
            return X_train, y_train
        
        logger.info(f"Handling class imbalance using {self.config.imbalance_method}...")
        
        # Original distribution
        unique, counts = np.unique(y_train, return_counts=True)
        original_dist = dict(zip(unique.astype(int), counts.astype(int)))
        
        logger.info(f"Original class distribution: {original_dist}")
        logger.info(f"Original train size: {len(y_train)}")
        
        # Apply resampling
        resampler = self.get_resampler()
        
        try:
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        except Exception as e:
            logger.warning(f"Resampling failed with {self.config.imbalance_method}: {str(e)}")
            logger.warning("Falling back to standard SMOTE")
            resampler = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        
        # New distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        resampled_dist = dict(zip(unique.astype(int), counts.astype(int)))
        
        logger.info(f"Resampled class distribution: {resampled_dist}")
        logger.info(f"Resampled train size: {len(y_resampled)}")
        
        self.transformation_report['class_imbalance'] = {
            'method': self.config.imbalance_method,
            'sampling_strategy': self.config.sampling_strategy,
            'original_distribution': original_dist,
            'resampled_distribution': resampled_dist,
            'original_size': len(y_train),
            'resampled_size': len(y_resampled),
            'samples_added': len(y_resampled) - len(y_train)
        }
        
        return X_resampled, y_resampled
    
    # ==================== PIPELINE CREATION ====================
    
    def create_preprocessing_pipeline(
        self,
        feature_names: List[str]
    ) -> Pipeline:
        """
        Create a preprocessing pipeline for production use.
        
        Args:
            feature_names: List of feature names
        
        Returns:
            Sklearn Pipeline object
        """
        logger.info("Creating preprocessing pipeline...")
        
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_handler', OutlierHandler(
                method=self.config.outlier_method,
                threshold=self.config.outlier_threshold
            )),
            ('scaler', self.get_scaler())
        ])
        
        return pipeline
    
    # ==================== MAIN EXECUTION ====================
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute complete data transformation pipeline.
        
        Returns:
            DataTransformationArtifact: Transformation results
        """
        logger.info(f"{'='*60}")
        logger.info("Starting Data Transformation Pipeline")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Create directories
            logger.info("Step 1: Creating directories")
            create_directories([self.config.data_transformation_dir])
            
            # Step 2: Load data
            logger.info("Step 2: Loading featured data")
            train_df, test_df = self._load_data()
            
            original_train_shape = train_df.shape
            original_test_shape = test_df.shape
            
            # Step 3: Separate features and target
            logger.info("Step 3: Separating features and target")
            X_train, y_train = self._separate_features_target(train_df)
            X_test, y_test = self._separate_features_target(test_df)
            
            self.feature_names = list(X_train.columns)
            
            logger.info(f"Features: {X_train.shape[1]}")
            logger.info(f"Train samples: {len(y_train)}")
            logger.info(f"Test samples: {len(y_test)}")
            
            # Step 4: Handle missing values
            logger.info("Step 4: Handling missing values")
            X_train, X_test = self.handle_missing_values(X_train, X_test)
            
            # Step 5: Handle outliers
            logger.info("Step 5: Handling outliers")
            X_train, X_test = self.handle_outliers(X_train, X_test)
            
            # Step 6: Feature selection
            logger.info("Step 6: Applying feature selection")
            X_train, X_test = self.apply_feature_selection(X_train, X_test, y_train)
            
            # Step 7: Apply scaling
            logger.info("Step 7: Applying scaling")
            X_train_scaled, X_test_scaled = self.apply_scaling(X_train, X_test)
            
            # Step 8: Handle class imbalance (only on training data)
            logger.info("Step 8: Handling class imbalance")
            y_train_array = y_train.values
            original_class_dist = {
                int(k): int(v) for k, v in 
                zip(*np.unique(y_train_array, return_counts=True))
            }
            
            X_train_resampled, y_train_resampled = self.handle_class_imbalance(
                X_train_scaled,
                y_train_array
            )
            
            resampled_class_dist = {
                int(k): int(v) for k, v in 
                zip(*np.unique(y_train_resampled, return_counts=True))
            }
            
            # Step 9: Save transformed data
            logger.info("Step 9: Saving transformed data")
            
            np.save(self.config.train_transformed_path, X_train_resampled)
            np.save(self.config.test_transformed_path, X_test_scaled)
            np.save(self.config.train_target_path, y_train_resampled)
            np.save(self.config.test_target_path, y_test.values)
            
            # Step 10: Save preprocessor
            logger.info("Step 10: Saving preprocessor")
            save_object(self.config.preprocessor_path, self.preprocessor)
            
            # Step 11: Save feature names
            logger.info("Step 11: Saving feature names")
            feature_info = {
                'original_features': self.feature_names,
                'selected_features': self.selected_features,
                'dropped_features': self.dropped_features,
                'final_feature_count': len(self.selected_features)
            }
            write_json(self.config.feature_names_path, feature_info)
            
            # Step 12: Save transformation report
            logger.info("Step 12: Saving transformation report")
            
            self.transformation_report['summary'] = {
                'original_train_shape': list(original_train_shape),
                'original_test_shape': list(original_test_shape),
                'final_train_shape': list(X_train_resampled.shape),
                'final_test_shape': list(X_test_scaled.shape),
                'original_features': len(self.feature_names),
                'selected_features': len(self.selected_features),
                'dropped_features': len(self.dropped_features),
                'scaling_method': self.config.scaling_method,
                'imbalance_method': self.config.imbalance_method
            }
            
            write_json(self.config.report_path, self.transformation_report)
            
            # Create artifact
            artifact = DataTransformationArtifact(
                train_transformed_path=self.config.train_transformed_path,
                test_transformed_path=self.config.test_transformed_path,
                train_target_path=self.config.train_target_path,
                test_target_path=self.config.test_target_path,
                preprocessor_path=self.config.preprocessor_path,
                feature_names_path=self.config.feature_names_path,
                report_path=self.config.report_path,
                original_features=self.feature_names,
                selected_features=self.selected_features,
                dropped_features=self.dropped_features,
                original_train_shape=original_train_shape,
                original_test_shape=original_test_shape,
                transformed_train_shape=X_train_resampled.shape,
                transformed_test_shape=X_test_scaled.shape,
                imbalance_method=self.config.imbalance_method,
                original_class_distribution=original_class_dist,
                resampled_class_distribution=resampled_class_dist,
                scaling_method=self.config.scaling_method
            )
            
            logger.info(f"{'='*60}")
            logger.info("DATA TRANSFORMATION COMPLETED!")
            logger.info(f"{'='*60}")
            logger.info(f"Original Train Shape: {original_train_shape}")
            logger.info(f"Transformed Train Shape: {X_train_resampled.shape}")
            logger.info(f"Original Test Shape: {original_test_shape}")
            logger.info(f"Transformed Test Shape: {X_test_scaled.shape}")
            logger.info(f"Original Class Distribution: {original_class_dist}")
            logger.info(f"Resampled Class Distribution: {resampled_class_dist}")
            
            return artifact
            
        except Exception as e:
            raise DataTransformationException(
                error_message=f"Data Transformation failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig,
        DataValidationConfig,
        FeatureEngineeringConfig
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.feature_engineering import FeatureEngineering
    
    # Run Data Ingestion
    print("\n" + "="*60)
    print("RUNNING DATA INGESTION")
    print("="*60)
    ingestion_config = DataIngestionConfig(
        source_type="csv",
        source_path="data/raw/transactions.csv"
    )
    ingestion = DataIngestion(config=ingestion_config)
    ingestion_artifact = ingestion.initiate_data_ingestion()
    
    # Run Data Validation
    print("\n" + "="*60)
    print("RUNNING DATA VALIDATION")
    print("="*60)
    validation_config = DataValidationConfig()
    validation = DataValidation(
        config=validation_config,
        data_ingestion_artifact=ingestion_artifact
    )
    validation_artifact = validation.initiate_data_validation()
    
    # Run Feature Engineering
    print("\n" + "="*60)
    print("RUNNING FEATURE ENGINEERING")
    print("="*60)
    fe_config = FeatureEngineeringConfig()
    feature_engineering = FeatureEngineering(
        config=fe_config,
        data_ingestion_artifact=ingestion_artifact,
        data_validation_artifact=validation_artifact
    )
    fe_artifact = feature_engineering.initiate_feature_engineering()
    
    # Run Data Transformation
    print("\n" + "="*60)
    print("RUNNING DATA TRANSFORMATION")
    print("="*60)
    transformation_config = DataTransformationConfig()
    data_transformation = DataTransformation(
        config=transformation_config,
        feature_engineering_artifact=fe_artifact
    )
    
    transformation_artifact = data_transformation.initiate_data_transformation()
    
    print("\n" + "="*60)
    print("DATA TRANSFORMATION ARTIFACT SUMMARY")
    print("="*60)
    print(f"Train Transformed Path: {transformation_artifact.train_transformed_path}")
    print(f"Test Transformed Path: {transformation_artifact.test_transformed_path}")
    print(f"Preprocessor Path: {transformation_artifact.preprocessor_path}")
    print(f"\nOriginal Train Shape: {transformation_artifact.original_train_shape}")
    print(f"Transformed Train Shape: {transformation_artifact.transformed_train_shape}")
    print(f"\nOriginal Class Distribution: {transformation_artifact.original_class_distribution}")
    print(f"Resampled Class Distribution: {transformation_artifact.resampled_class_distribution}")
    print(f"\nScaling Method: {transformation_artifact.scaling_method}")
    print(f"Imbalance Method: {transformation_artifact.imbalance_method}")
    print(f"\nSelected Features: {len(transformation_artifact.selected_features)}")
    print(f"Dropped Features: {len(transformation_artifact.dropped_features)}")