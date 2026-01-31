"""
Kafka Consumer for Fraud Detection System

Consumes transactions from Kafka topic and processes them
for fraud detection.
"""

import json
import signal
import sys
from typing import Dict, List, Optional, Callable
from datetime import datetime
import threading
import time

from confluent_kafka import Consumer, KafkaError, KafkaException

from src.streaming.config import kafka_settings, streaming_config
from src.streaming.schemas import TransactionMessage
from src.logger import logger


class FraudDetectionConsumer:
    """
    Kafka Consumer for reading transactions.
    """
    
    def __init__(
        self,
        message_handler: Optional[Callable] = None,
        batch_handler: Optional[Callable] = None
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            message_handler: Callback for single message processing
            batch_handler: Callback for batch processing
        """
        self.settings = kafka_settings
        self.config = streaming_config
        self.consumer = None
        self.is_running = False
        self.is_connected = False
        
        self.message_handler = message_handler
        self.batch_handler = batch_handler
        
        # Stats
        self._stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'batches_processed': 0
        }
        self._lock = threading.Lock()
        
        logger.info("Initializing Kafka Consumer")
    
    def _get_consumer_config(self) -> Dict:
        """Get consumer configuration."""
        config = {
            'bootstrap.servers': self.settings.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': self.settings.CONSUMER_GROUP_ID,
            'auto.offset.reset': self.settings.CONSUMER_AUTO_OFFSET_RESET,
            'enable.auto.commit': self.settings.CONSUMER_ENABLE_AUTO_COMMIT,
            'max.poll.interval.ms': self.settings.CONSUMER_MAX_POLL_INTERVAL_MS,
            'session.timeout.ms': self.settings.CONSUMER_SESSION_TIMEOUT_MS,
        }
        
        # Add SASL config if needed
        if self.settings.KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
            config.update({
                'security.protocol': self.settings.KAFKA_SECURITY_PROTOCOL,
                'sasl.mechanism': self.settings.KAFKA_SASL_MECHANISM,
                'sasl.username': self.settings.KAFKA_SASL_USERNAME,
                'sasl.password': self.settings.KAFKA_SASL_PASSWORD,
            })
        
        return config
    
    def connect(self) -> bool:
        """
        Connect to Kafka broker and subscribe to topics.
        
        Returns:
            True if connected successfully
        """
        try:
            config = self._get_consumer_config()
            self.consumer = Consumer(config)
            
            # Subscribe to transactions topic
            self.consumer.subscribe([self.settings.TRANSACTIONS_TOPIC])
            
            self.is_connected = True
            logger.info(f"Connected to Kafka: {self.settings.KAFKA_BOOTSTRAP_SERVERS}")
            logger.info(f"Subscribed to topic: {self.settings.TRANSACTIONS_TOPIC}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            self.is_connected = False
            return False
    
    def _parse_message(self, msg) -> Optional[TransactionMessage]:
        """
        Parse Kafka message to TransactionMessage.
        
        Args:
            msg: Kafka message
        
        Returns:
            TransactionMessage or None if parsing fails
        """
        try:
            value = msg.value().decode('utf-8')
            data = json.loads(value)
            
            transaction = TransactionMessage.from_dict(data)
            
            return transaction
            
        except Exception as e:
            logger.error(f"Failed to parse message: {str(e)}")
            return None
    
    def _process_single_message(self, msg) -> bool:
        """
        Process a single message.
        
        Args:
            msg: Kafka message
        
        Returns:
            True if processed successfully
        """
        try:
            transaction = self._parse_message(msg)
            
            if transaction is None:
                return False
            
            if self.message_handler:
                self.message_handler(
                    transaction=transaction,
                    partition=msg.partition(),
                    offset=msg.offset()
                )
            
            with self._lock:
                self._stats['messages_processed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            with self._lock:
                self._stats['messages_failed'] += 1
            return False
    
    def _process_batch(self, messages: List) -> bool:
        """
        Process a batch of messages.
        
        Args:
            messages: List of Kafka messages
        
        Returns:
            True if processed successfully
        """
        try:
            transactions = []
            metadata = []
            
            for msg in messages:
                transaction = self._parse_message(msg)
                if transaction:
                    transactions.append(transaction)
                    metadata.append({
                        'partition': msg.partition(),
                        'offset': msg.offset()
                    })
            
            if transactions and self.batch_handler:
                self.batch_handler(
                    transactions=transactions,
                    metadata=metadata
                )
            
            with self._lock:
                self._stats['messages_processed'] += len(transactions)
                self._stats['batches_processed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            with self._lock:
                self._stats['messages_failed'] += len(messages)
            return False
    
    def consume_single(self, timeout: float = 1.0):
        """
        Consume and process single messages.
        
        Args:
            timeout: Poll timeout in seconds
        """
        if not self.is_connected:
            if not self.connect():
                return
        
        self.is_running = True
        logger.info("Starting single message consumption...")
        
        try:
            while self.is_running:
                msg = self.consumer.poll(timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        continue
                
                with self._lock:
                    self._stats['messages_received'] += 1
                
                # Process message
                success = self._process_single_message(msg)
                
                # Commit offset if successful
                if success:
                    self.consumer.commit(msg)
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.close()
    
    def consume_batch(
        self,
        batch_size: Optional[int] = None,
        batch_timeout: Optional[float] = None
    ):
        """
        Consume and process messages in batches.
        
        Args:
            batch_size: Number of messages per batch
            batch_timeout: Timeout for collecting batch
        """
        if not self.is_connected:
            if not self.connect():
                return
        
        batch_size = batch_size or self.config.batch_size
        batch_timeout = batch_timeout or self.config.batch_timeout
        
        self.is_running = True
        logger.info(f"Starting batch consumption (batch_size={batch_size})...")
        
        try:
            while self.is_running:
                messages = []
                batch_start = time.time()
                
                # Collect batch
                while len(messages) < batch_size:
                    remaining_time = batch_timeout - (time.time() - batch_start)
                    
                    if remaining_time <= 0:
                        break
                    
                    msg = self.consumer.poll(min(remaining_time, 0.1))
                    
                    if msg is None:
                        continue
                    
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            logger.error(f"Consumer error: {msg.error()}")
                        continue
                    
                    messages.append(msg)
                    
                    with self._lock:
                        self._stats['messages_received'] += 1
                
                # Process batch if not empty
                if messages:
                    success = self._process_batch(messages)
                    
                    # Commit offsets if successful
                    if success:
                        self.consumer.commit()
                        
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.close()
    
    def stop(self) -> None:
        """Stop the consumer."""
        self.is_running = False
        logger.info("Consumer stop requested")
    
    def close(self) -> None:
        """Close the consumer."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka Consumer closed")
        self.is_connected = False
        self.is_running = False
    
    def get_stats(self) -> Dict:
        """Get consumer statistics."""
        with self._lock:
            return self._stats.copy()


# Factory function
def create_consumer(
    message_handler: Optional[Callable] = None,
    batch_handler: Optional[Callable] = None
) -> FraudDetectionConsumer:
    """
    Create a consumer instance.
    
    Args:
        message_handler: Callback for single message processing
        batch_handler: Callback for batch processing
    
    Returns:
        FraudDetectionConsumer instance
    """
    return FraudDetectionConsumer(
        message_handler=message_handler,
        batch_handler=batch_handler
    )