"""
Kafka Configuration for Fraud Detection Streaming
"""

import os
from typing import List, Optional
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings


class KafkaSettings(BaseSettings):
    """Kafka Configuration Settings"""
    
    # Kafka Broker
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_SECURITY_PROTOCOL: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_SSL
    
    # SASL Configuration (for production)
    KAFKA_SASL_MECHANISM: Optional[str] = None  # PLAIN, SCRAM-SHA-256
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[str] = None
    
    # Topics
    TRANSACTIONS_TOPIC: str = "transactions"
    PREDICTIONS_TOPIC: str = "fraud_predictions"
    ALERTS_TOPIC: str = "fraud_alerts"
    DLQ_TOPIC: str = "transactions_dlq"  # Dead Letter Queue
    
    # Consumer Configuration
    CONSUMER_GROUP_ID: str = "fraud_detection_group"
    CONSUMER_AUTO_OFFSET_RESET: str = "latest"  # latest, earliest
    CONSUMER_MAX_POLL_RECORDS: int = 500
    CONSUMER_MAX_POLL_INTERVAL_MS: int = 300000
    CONSUMER_SESSION_TIMEOUT_MS: int = 30000
    CONSUMER_ENABLE_AUTO_COMMIT: bool = False
    
    # Producer Configuration
    PRODUCER_ACKS: str = "all"  # 0, 1, all
    PRODUCER_RETRIES: int = 3
    PRODUCER_BATCH_SIZE: int = 16384
    PRODUCER_LINGER_MS: int = 5
    PRODUCER_COMPRESSION_TYPE: str = "gzip"  # none, gzip, snappy, lz4
    
    # Processing Configuration
    BATCH_SIZE: int = 100
    BATCH_TIMEOUT_SECONDS: float = 1.0
    MAX_WORKERS: int = 4
    
    # Alerting
    HIGH_RISK_THRESHOLD: float = 0.7
    SEND_ALERTS_FOR_HIGH_RISK: bool = True
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8001
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@dataclass
class StreamingConfig:
    """Streaming Pipeline Configuration"""
    
    # Kafka Settings
    kafka: KafkaSettings = field(default_factory=KafkaSettings)
    
    # Processing Settings
    enable_batching: bool = True
    batch_size: int = 100
    batch_timeout: float = 1.0
    
    # Model Settings
    model_path: str = "artifacts/production_model"
    reload_model_interval: int = 3600  # seconds
    
    # Logging
    log_predictions: bool = True
    log_alerts: bool = True
    
    # Error Handling
    max_retries: int = 3
    retry_delay: float = 1.0
    send_to_dlq: bool = True


# Global settings instance
kafka_settings = KafkaSettings()
streaming_config = StreamingConfig()