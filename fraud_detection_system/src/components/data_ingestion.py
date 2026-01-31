"""
Data Ingestion Component
- Load data from multiple sources (CSV, Database, S3, API)
- Perform train-test split with stratification
- Save artifacts
- Log statistics
"""

import os
import sys
from typing import Tuple, Optional
# Core data libraries
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    raise DataIngestionException(
        error_message=(
            "pandas and numpy are required. Install them with `pip install -r requirements.txt`."
        ),
        error_detail=sys
    ) from e

# sklearn is only needed for train_test_split; import with helpful error
try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise DataIngestionException(
        error_message=(
            "scikit-learn is required for train/test splitting. Install with `pip install scikit-learn`."
        ),
        error_detail=sys
    ) from e

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logger, log_dataframe_info
from src.exception import DataIngestionException
from src.utils.common import create_directories


class DataIngestion:
    """
    Data Ingestion Component for Fraud Detection System.
    
    Responsibilities:
    - Load data from various sources
    - Validate data source
    - Perform stratified train-test split
    - Save raw and processed data
    - Generate ingestion artifact
    """
    
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initialize Data Ingestion component.
        
        Args:
            config: DataIngestionConfig object with all settings
        """
        self.config = config
        logger.info(f"{'='*60}")
        logger.info("Initializing Data Ingestion Component")
        logger.info(f"{'='*60}")
    
    def _load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from CSV: {file_path}")
        
        if not os.path.exists(file_path):
            raise DataIngestionException(
                error_message=f"CSV file not found: {file_path}",
                error_detail=sys
            )
        
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def _load_from_database(self) -> pd.DataFrame:
        """Load data from PostgreSQL/MySQL database."""
        logger.info("Loading data from Database")
        
        try:
            from sqlalchemy import create_engine
            
            # Build connection string
            db_url = (
                f"postgresql://{os.getenv('DB_USERNAME')}:"
                f"{os.getenv('DB_PASSWORD')}@"
                f"{self.config.db_host}:{self.config.db_port}/"
                f"{self.config.db_name}"
            )
            
            engine = create_engine(db_url)
            
            # Use custom query or default
            query = self.config.db_query or f"SELECT * FROM {self.config.db_table}"
            
            df = pd.read_sql(query, engine)
            logger.info(f"Data loaded from database. Shape: {df.shape}")
            return df
            
        except Exception as e:
            raise DataIngestionException(
                error_message=f"Database connection failed: {str(e)}",
                error_detail=sys
            ) from e
    
    def _load_from_mongodb(self) -> pd.DataFrame:
        """Load data from MongoDB."""
        logger.info("Loading data from MongoDB")
        
        try:
            from pymongo import MongoClient
            
            client = MongoClient(os.getenv('MONGODB_URL'))
            db = client[self.config.db_name]
            collection = db[self.config.db_table]
            
            # Fetch all documents
            cursor = collection.find({})
            df = pd.DataFrame(list(cursor))
            
            # Remove MongoDB's _id column
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            logger.info(f"Data loaded from MongoDB. Shape: {df.shape}")
            return df
            
        except Exception as e:
            raise DataIngestionException(
                error_message=f"MongoDB connection failed: {str(e)}",
                error_detail=sys
            ) from e
    
    def _load_from_s3(self) -> pd.DataFrame:
        """Load data from AWS S3."""
        logger.info("Loading data from S3")
        
        try:
            import boto3
            
            s3_path = f"s3://{self.config.s3_bucket}/{self.config.s3_key}"
            df = pd.read_csv(s3_path)
            
            logger.info(f"Data loaded from S3. Shape: {df.shape}")
            return df
            
        except Exception as e:
            raise DataIngestionException(
                error_message=f"S3 load failed: {str(e)}",
                error_detail=sys
            ) from e
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data based on source type configuration.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        logger.info(f"Loading data from source type: {self.config.source_type}")
        
        source_loaders = {
            "csv": lambda: self._load_from_csv(self.config.source_path),
            "database": self._load_from_database,
            "postgresql": self._load_from_database,
            "mongodb": self._load_from_mongodb,
            "s3": self._load_from_s3,
        }
        
        loader = source_loaders.get(self.config.source_type.lower())
        
        if loader is None:
            raise DataIngestionException(
                error_message=f"Unsupported source type: {self.config.source_type}",
                error_detail=sys
            )
        
        return loader()
    
    def _get_data_statistics(self, df: pd.DataFrame) -> dict:
        """Calculate data statistics for artifact."""
        target_col = self.config.stratify_column

        if not isinstance(df, pd.DataFrame):
            raise DataIngestionException(
                error_message="Expected a pandas DataFrame for statistics",
                error_detail=sys
            )

        if target_col not in df.columns:
            raise DataIngestionException(
                error_message=f"Stratify column '{target_col}' not found in dataframe",
                error_detail=sys
            )

        total_records = len(df)
        if total_records == 0:
            raise DataIngestionException(
                error_message="Dataframe is empty; cannot compute statistics",
                error_detail=sys
            )

        fraud_count = df[target_col].sum()
        non_fraud_count = total_records - fraud_count
        fraud_percentage = (fraud_count / total_records) * 100

        stats = {
            "total_records": total_records,
            "fraud_count": int(fraud_count),
            "non_fraud_count": int(non_fraud_count),
            "fraud_percentage": round(fraud_percentage, 2)
        }

        logger.info(f"Data Statistics:")
        logger.info(f"  Total Records: {total_records:,}")
        logger.info(f"  Fraud Cases: {fraud_count:,} ({fraud_percentage:.2f}%)")
        logger.info(f"  Non-Fraud Cases: {non_fraud_count:,}")

        return stats
    
    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets with stratification.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Performing stratified train-test split")
        logger.info(f"  Test Size: {self.config.test_size}")
        logger.info(f"  Stratify Column: {self.config.stratify_column}")
        logger.info(f"  Random State: {self.config.random_state}")
        
        # Defensive runtime checks (avoid subscripted typing generics in isinstance)
        if not isinstance(df, pd.DataFrame):
            raise DataIngestionException(
                error_message="Expected a pandas DataFrame for splitting",
                error_detail=sys
            )

        if self.config.stratify_column not in df.columns:
            raise DataIngestionException(
                error_message=f"Stratify column '{self.config.stratify_column}' not present in dataframe",
                error_detail=sys
            )

        try:
            # Ensure stratify column has at least 2 classes
            if df[self.config.stratify_column].nunique() < 2:
                raise DataIngestionException(
                    error_message=f"Stratify column '{self.config.stratify_column}' must have at least 2 classes",
                    error_detail=sys
                )

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df[self.config.stratify_column]
            )

            logger.info(f"Train set shape: {train_df.shape}")
            logger.info(f"Test set shape: {test_df.shape}")

            # Verify stratification
            train_fraud_pct = (train_df[self.config.stratify_column].sum() / len(train_df)) * 100
            test_fraud_pct = (test_df[self.config.stratify_column].sum() / len(test_df)) * 100

            logger.info(f"Train fraud %: {train_fraud_pct:.2f}%")
            logger.info(f"Test fraud %: {test_fraud_pct:.2f}%")

            return train_df, test_df

        except Exception as e:
            raise DataIngestionException(
                error_message=f"Train-test split failed: {str(e)}",
                error_detail=sys
            ) from e
    
    def save_data(
        self,
        df: pd.DataFrame,
        file_path: str
    ) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Destination path
        """
        try:
            # Validate inputs
            if not isinstance(df, pd.DataFrame):
                raise DataIngestionException(
                    error_message="save_data expects a pandas DataFrame",
                    error_detail=sys
                )

            if not isinstance(file_path, str) or not file_path:
                raise DataIngestionException(
                    error_message="Invalid file_path provided to save_data",
                    error_detail=sys
                )

            # Create directory if not exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved: {file_path}")
            
        except Exception as e:
            raise DataIngestionException(
                error_message=f"Failed to save data: {file_path} - {str(e)}",
                error_detail=sys
            ) from e
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Execute the complete data ingestion pipeline.
        
        Returns:
            DataIngestionArtifact: Artifact with all ingestion details
        """
        logger.info(f"{'='*60}")
        logger.info("Starting Data Ingestion Pipeline")
        logger.info(f"{'='*60}")
        
        try:
            # Step 1: Create directories
            logger.info("Step 1: Creating directories")
            create_directories([
                self.config.raw_data_dir,
                self.config.processed_data_dir
            ])
            
            # Step 2: Load data
            logger.info("Step 2: Loading data")
            df = self.load_data()

            # Ensure loader returned a proper DataFrame
            if not isinstance(df, pd.DataFrame):
                raise DataIngestionException(
                    error_message="Loaded data is not a pandas DataFrame; check loader implementation",
                    error_detail=sys
                )

            log_dataframe_info(df, "Raw Data")
            
            # Step 3: Get statistics
            logger.info("Step 3: Calculating data statistics")
            stats = self._get_data_statistics(df)
            
            # Step 4: Save raw data
            logger.info("Step 4: Saving raw data")
            self.save_data(df, self.config.raw_file_path)
            
            # Step 5: Split data
            logger.info("Step 5: Splitting data")
            train_df, test_df = self.split_data(df)
            
            # Step 6: Save processed data
            logger.info("Step 6: Saving processed data")
            self.save_data(train_df, self.config.train_file_path)
            self.save_data(test_df, self.config.test_file_path)
            
            # Step 7: Create artifact
            logger.info("Step 7: Creating artifact")
            artifact = DataIngestionArtifact(
                raw_file_path=self.config.raw_file_path,
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
                total_records=stats["total_records"],
                train_records=len(train_df),
                test_records=len(test_df),
                fraud_count=stats["fraud_count"],
                non_fraud_count=stats["non_fraud_count"],
                fraud_percentage=stats["fraud_percentage"],
                source_type=self.config.source_type
            )
            
            logger.info(f"{'='*60}")
            logger.info("Data Ingestion Completed Successfully!")
            logger.info(f"{'='*60}")
            
            return artifact
            
        except Exception as e:
            raise DataIngestionException(
                error_message=f"Data Ingestion failed: {str(e)}",
                error_detail=sys
            ) from e


# ============== STANDALONE EXECUTION ==============
if __name__ == "__main__":
    # Test Data Ingestion
    config = DataIngestionConfig(
        source_type="csv",
        source_path="data/raw/transactions.csv"
    )
    
    ingestion = DataIngestion(config=config)
    artifact = ingestion.initiate_data_ingestion()
    
    print("\n" + "="*60)
    print("DATA INGESTION ARTIFACT")
    print("="*60)
    print(f"Raw File: {artifact.raw_file_path}")
    print(f"Train File: {artifact.train_file_path}")
    print(f"Test File: {artifact.test_file_path}")
    print(f"Total Records: {artifact.total_records:,}")
    print(f"Fraud Percentage: {artifact.fraud_percentage}%")