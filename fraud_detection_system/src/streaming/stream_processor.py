"""
Stream Processor for Fraud Detection System

Main orchestrator for:
- Consuming transactions from Kafka
- Processing through ML model
- Producing predictions and alerts
"""

import os
import sys
import time
import signal
import threading
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from queue import Queue, Empty
import traceback

from src.streaming.config import kafka_settings, streaming_config, StreamingConfig
from src.streaming.schemas import (
    TransactionMessage,
    PredictionMessage,
    AlertMessage,
    DLQMessage,
    RiskLevel
)
from src.streaming.consumer import FraudDetectionConsumer, create_consumer
from src.streaming.producer import FraudDetectionProducer, get_producer
from src.pipeline.prediction_pipeline import PredictionPipeline, PredictionConfig
from src.logger import logger


@dataclass
class ProcessingMetrics:
    """Metrics for stream processing."""
    
    transactions_received: int = 0
    transactions_processed: int = 0
    frauds_detected: int = 0
    alerts_sent: int = 0
    errors: int = 0
    
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    
    start_time: str = ""
    last_update_time: str = ""


class FraudDetectionStreamProcessor:
    """
    Main stream processor for real-time fraud detection.
    
    Workflow:
    1. Consume transactions from Kafka
    2. Apply feature engineering
    3. Run through ML model
    4. Send predictions to output topic
    5. Send alerts for high-risk transactions
    """
    
    def __init__(self, config: StreamingConfig = None):
        """
        Initialize the stream processor.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or streaming_config
        self.kafka_settings = kafka_settings
        
        # Components
        self.consumer: Optional[FraudDetectionConsumer] = None
        self.producer: Optional[FraudDetectionProducer] = None
        self.prediction_pipeline: Optional[PredictionPipeline] = None
        
        # State
        self.is_running = False
        self.is_initialized = False
        
        # Metrics
        self.metrics = ProcessingMetrics()
        self._processing_times: List[float] = []
        self._lock = threading.Lock()
        
        # Graceful shutdown
        self._shutdown_event = threading.Event()
        
        logger.info("Initializing Fraud Detection Stream Processor")
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful
        """
        logger.info("Initializing stream processor components...")
        
        try:
            # Initialize prediction pipeline
            logger.info("Loading ML model...")
            self.prediction_pipeline = PredictionPipeline()
            if not self.prediction_pipeline.initialize():
                raise Exception("Failed to initialize prediction pipeline")
            logger.info(f"Model loaded: v{self.prediction_pipeline.model_version}")
            
            # Initialize producer
            logger.info("Connecting to Kafka producer...")
            self.producer = get_producer()
            if not self.producer.is_connected:
                if not self.producer.connect():
                    raise Exception("Failed to connect producer")
            
            # Initialize consumer
            logger.info("Connecting to Kafka consumer...")
            self.consumer = create_consumer(
                message_handler=self._handle_single_message,
                batch_handler=self._handle_batch_messages
            )
            if not self.consumer.connect():
                raise Exception("Failed to connect consumer")
            
            self.is_initialized = True
            self.metrics.start_time = datetime.now().isoformat()
            
            logger.info("Stream processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def _create_prediction_message(
        self,
        transaction: TransactionMessage,
        prediction_result,
        partition: int = 0,
        offset: int = 0
    ) -> PredictionMessage:
        """
        Create prediction message from transaction and result.
        
        Args:
            transaction: Original transaction
            prediction_result: Prediction result from pipeline
            partition: Kafka partition
            offset: Kafka offset
        
        Returns:
            PredictionMessage object
        """
        return PredictionMessage(
            transaction_id=transaction.transaction_id,
            prediction_id=f"PRED_{uuid.uuid4().hex[:12]}",
            is_fraud=prediction_result.is_fraud,
            prediction=prediction_result.prediction,
            probability=prediction_result.probability,
            risk_score=prediction_result.risk_score,
            risk_level=prediction_result.risk_category,
            threshold_used=prediction_result.threshold_used,
            model_version=prediction_result.model_version,
            risk_factors=prediction_result.top_risk_factors,
            transaction_amount=transaction.amount,
            transaction_type=transaction.type,
            processing_time_ms=prediction_result.processing_time_ms,
            transaction_timestamp=transaction.timestamp,
            source_topic=self.kafka_settings.TRANSACTIONS_TOPIC,
            partition=partition,
            offset=offset
        )
    
    def _create_alert_message(
        self,
        transaction: TransactionMessage,
        prediction_result
    ) -> AlertMessage:
        """
        Create alert message for high-risk transaction.
        
        Args:
            transaction: Original transaction
            prediction_result: Prediction result
        
        Returns:
            AlertMessage object
        """
        # Determine severity
        if prediction_result.probability >= 0.9:
            severity = "CRITICAL"
        elif prediction_result.probability >= 0.8:
            severity = "HIGH"
        elif prediction_result.probability >= 0.7:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Determine recommended action
        if severity in ["CRITICAL", "HIGH"]:
            action = "BLOCK_TRANSACTION"
            manual_review = True
        else:
            action = "FLAG_FOR_REVIEW"
            manual_review = True
        
        return AlertMessage(
            alert_id=f"ALERT_{uuid.uuid4().hex[:12]}",
            alert_type="FRAUD_DETECTED",
            severity=severity,
            transaction_id=transaction.transaction_id,
            transaction_amount=transaction.amount,
            transaction_type=transaction.type,
            origin_account=transaction.name_orig,
            destination_account=transaction.name_dest,
            fraud_probability=prediction_result.probability,
            risk_score=prediction_result.risk_score,
            risk_factors=prediction_result.top_risk_factors,
            recommended_action=action,
            requires_manual_review=manual_review,
            transaction_timestamp=transaction.timestamp,
            model_version=prediction_result.model_version
        )
    
    def _process_transaction(
        self,
        transaction: TransactionMessage,
        partition: int = 0,
        offset: int = 0
    ) -> Optional[PredictionMessage]:
        """
        Process a single transaction through the model.
        
        Args:
            transaction: Transaction to process
            partition: Kafka partition
            offset: Kafka offset
        
        Returns:
            PredictionMessage or None if failed
        """
        try:
            # Convert to dict for prediction
            transaction_data = {
                'step': transaction.step,
                'type': transaction.type,
                'amount': transaction.amount,
                'name_orig': transaction.name_orig,
                'old_balance_org': transaction.old_balance_org,
                'new_balance_org': transaction.new_balance_org,
                'name_dest': transaction.name_dest,
                'old_balance_dest': transaction.old_balance_dest,
                'new_balance_dest': transaction.new_balance_dest
            }
            
            # Get prediction
            result = self.prediction_pipeline.predict_single(
                transaction=transaction_data,
                transaction_id=transaction.transaction_id
            )
            
            # Create prediction message
            prediction_msg = self._create_prediction_message(
                transaction, result, partition, offset
            )
            
            # Update metrics
            with self._lock:
                self.metrics.transactions_processed += 1
                self._processing_times.append(result.processing_time_ms)
                
                if result.is_fraud:
                    self.metrics.frauds_detected += 1
            
            return prediction_msg
            
        except Exception as e:
            logger.error(f"Error processing transaction {transaction.transaction_id}: {str(e)}")
            with self._lock:
                self.metrics.errors += 1
            return None
    
    def _handle_single_message(
        self,
        transaction: TransactionMessage,
        partition: int = 0,
        offset: int = 0
    ) -> None:
        """
        Handle a single message from consumer.
        
        Args:
            transaction: Transaction message
            partition: Kafka partition
            offset: Kafka offset
        """
        with self._lock:
            self.metrics.transactions_received += 1
            self.metrics.last_update_time = datetime.now().isoformat()
        
        # Process transaction
        prediction = self._process_transaction(transaction, partition, offset)
        
        if prediction is None:
            # Send to DLQ
            if self.config.send_to_dlq:
                self._send_to_dlq(
                    transaction.to_json(),
                    "PROCESSING_ERROR",
                    "Failed to process transaction"
                )
            return
        
        # Send prediction to output topic
        self.producer.send_prediction(prediction)
        
        # Send alert if high risk
        if (prediction.is_fraud and 
            prediction.probability >= self.kafka_settings.HIGH_RISK_THRESHOLD and
            self.kafka_settings.SEND_ALERTS_FOR_HIGH_RISK):
            
            alert = self._create_alert_message(transaction, 
                                               type('obj', (object,), {
                                                   'is_fraud': prediction.is_fraud,
                                                   'probability': prediction.probability,
                                                   'risk_score': prediction.risk_score,
                                                   'risk_category': prediction.risk_level,
                                                   'top_risk_factors': prediction.risk_factors,
                                                   'model_version': prediction.model_version
                                               })())
            
            self.producer.send_alert(alert)
            
            with self._lock:
                self.metrics.alerts_sent += 1
    
    def _handle_batch_messages(
        self,
        transactions: List[TransactionMessage],
        metadata: List[Dict]
    ) -> None:
        """
        Handle a batch of messages from consumer.
        
        Args:
            transactions: List of transaction messages
            metadata: List of partition/offset info
        """
        with self._lock:
            self.metrics.transactions_received += len(transactions)
            self.metrics.last_update_time = datetime.now().isoformat()
        
        for transaction, meta in zip(transactions, metadata):
            self._handle_single_message(
                transaction,
                meta.get('partition', 0),
                meta.get('offset', 0)
            )
    
    def _send_to_dlq(
        self,
        original_message: str,
        error_type: str,
        error_message: str
    ) -> None:
        """
        Send failed message to Dead Letter Queue.
        
        Args:
            original_message: Original message content
            error_type: Type of error
            error_message: Error message
        """
        dlq_msg = DLQMessage(
            original_message=original_message,
            original_topic=self.kafka_settings.TRANSACTIONS_TOPIC,
            error_type=error_type,
            error_message=error_message,
            stack_trace=traceback.format_exc()
        )
        
        self.producer.send_to_dlq(dlq_msg)
    
    def start(self, mode: str = "batch") -> None:
        """
        Start the stream processor.
        
        Args:
            mode: Processing mode ("single" or "batch")
        """
        if not self.is_initialized:
            if not self.initialize():
                raise Exception("Failed to initialize stream processor")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.is_running = True
        
        logger.info(f"{'='*60}")
        logger.info("STARTING FRAUD DETECTION STREAM PROCESSOR")
        logger.info(f"{'='*60}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Input Topic: {self.kafka_settings.TRANSACTIONS_TOPIC}")
        logger.info(f"Output Topic: {self.kafka_settings.PREDICTIONS_TOPIC}")
        logger.info(f"Alerts Topic: {self.kafka_settings.ALERTS_TOPIC}")
        logger.info(f"Model Version: {self.prediction_pipeline.model_version}")
        logger.info(f"{'='*60}")
        
        try:
            if mode == "single":
                self.consumer.consume_single()
            else:
                self.consumer.consume_batch(
                    batch_size=self.config.batch_size,
                    batch_timeout=self.config.batch_timeout
                )
                
        except Exception as e:
            logger.error(f"Stream processor error: {str(e)}")
        finally:
            self.stop()
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.stop()
    
    def stop(self) -> None:
        """Stop the stream processor."""
        self.is_running = False
        self._shutdown_event.set()
        
        # Stop consumer
        if self.consumer:
            self.consumer.stop()
        
        # Flush producer
        if self.producer:
            self.producer.flush()
        
        # Log final metrics
        self._log_metrics()
        
        logger.info("Stream processor stopped")
    
    def _log_metrics(self) -> None:
        """Log processing metrics."""
        with self._lock:
            if self._processing_times:
                self.metrics.avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)
                self.metrics.max_processing_time_ms = max(self._processing_times)
                self.metrics.min_processing_time_ms = min(self._processing_times)
        
        logger.info(f"\n{'='*60}")
        logger.info("STREAM PROCESSING METRICS")
        logger.info(f"{'='*60}")
        logger.info(f"Transactions Received: {self.metrics.transactions_received:,}")
        logger.info(f"Transactions Processed: {self.metrics.transactions_processed:,}")
        logger.info(f"Frauds Detected: {self.metrics.frauds_detected:,}")
        logger.info(f"Alerts Sent: {self.metrics.alerts_sent:,}")
        logger.info(f"Errors: {self.metrics.errors:,}")
        logger.info(f"Avg Processing Time: {self.metrics.avg_processing_time_ms:.2f} ms")
        logger.info(f"Max Processing Time: {self.metrics.max_processing_time_ms:.2f} ms")
        if self.metrics.min_processing_time_ms != float('inf'):
            logger.info(f"Min Processing Time: {self.metrics.min_processing_time_ms:.2f} ms")
        logger.info(f"{'='*60}")
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current metrics."""
        with self._lock:
            return ProcessingMetrics(
                transactions_received=self.metrics.transactions_received,
                transactions_processed=self.metrics.transactions_processed,
                frauds_detected=self.metrics.frauds_detected,
                alerts_sent=self.metrics.alerts_sent,
                errors=self.metrics.errors,
                avg_processing_time_ms=self.metrics.avg_processing_time_ms,
                max_processing_time_ms=self.metrics.max_processing_time_ms,
                min_processing_time_ms=self.metrics.min_processing_time_ms,
                start_time=self.metrics.start_time,
                last_update_time=self.metrics.last_update_time
            )


# Factory function
def create_stream_processor(config: StreamingConfig = None) -> FraudDetectionStreamProcessor:
    """
    Create a stream processor instance.
    
    Args:
        config: Streaming configuration
    
    Returns:
        FraudDetectionStreamProcessor instance
    """
    return FraudDetectionStreamProcessor(config)