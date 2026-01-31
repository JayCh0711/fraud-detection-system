"""
Kafka Producer for Fraud Detection System

Sends:
- Prediction results to predictions topic
- Fraud alerts to alerts topic
- Failed messages to DLQ
"""

import json
from typing import Dict, Optional, Callable
from datetime import datetime
import threading

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.streaming.config import kafka_settings, streaming_config
from src.streaming.schemas import PredictionMessage, AlertMessage, DLQMessage
from src.logger import logger


class FraudDetectionProducer:
    """
    Kafka Producer for sending prediction results and alerts.
    """
    
    def __init__(self):
        """Initialize the Kafka producer."""
        self.settings = kafka_settings
        self.producer = None
        self.is_connected = False
        
        self._delivery_stats = {
            'sent': 0,
            'delivered': 0,
            'failed': 0
        }
        self._lock = threading.Lock()
        
        logger.info("Initializing Kafka Producer")
    
    def _get_producer_config(self) -> Dict:
        """Get producer configuration."""
        config = {
            'bootstrap.servers': self.settings.KAFKA_BOOTSTRAP_SERVERS,
            'acks': self.settings.PRODUCER_ACKS,
            'retries': self.settings.PRODUCER_RETRIES,
            'batch.size': self.settings.PRODUCER_BATCH_SIZE,
            'linger.ms': self.settings.PRODUCER_LINGER_MS,
            'compression.type': self.settings.PRODUCER_COMPRESSION_TYPE,
            'enable.idempotence': True,  # Exactly-once semantics
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
        Connect to Kafka broker.
        
        Returns:
            True if connected successfully
        """
        try:
            config = self._get_producer_config()
            self.producer = Producer(config)
            
            # Test connection by listing topics
            self.producer.list_topics(timeout=10)
            
            self.is_connected = True
            logger.info(f"Connected to Kafka: {self.settings.KAFKA_BOOTSTRAP_SERVERS}")
            
            # Create topics if they don't exist
            self._create_topics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            self.is_connected = False
            return False
    
    def _create_topics(self) -> None:
        """Create required topics if they don't exist."""
        try:
            admin_config = {
                'bootstrap.servers': self.settings.KAFKA_BOOTSTRAP_SERVERS
            }
            admin = AdminClient(admin_config)
            
            # Check existing topics
            existing_topics = admin.list_topics().topics.keys()
            
            topics_to_create = [
                (self.settings.PREDICTIONS_TOPIC, 3, 1),
                (self.settings.ALERTS_TOPIC, 3, 1),
                (self.settings.DLQ_TOPIC, 1, 1)
            ]
            
            new_topics = []
            for topic_name, partitions, replication in topics_to_create:
                if topic_name not in existing_topics:
                    new_topics.append(NewTopic(
                        topic_name,
                        num_partitions=partitions,
                        replication_factor=replication
                    ))
            
            if new_topics:
                futures = admin.create_topics(new_topics)
                for topic, future in futures.items():
                    try:
                        future.result()
                        logger.info(f"Created topic: {topic}")
                    except Exception as e:
                        logger.warning(f"Topic creation failed for {topic}: {e}")
                        
        except Exception as e:
            logger.warning(f"Could not create topics: {e}")
    
    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation."""
        with self._lock:
            if err:
                self._delivery_stats['failed'] += 1
                logger.error(f"Message delivery failed: {err}")
            else:
                self._delivery_stats['delivered'] += 1
                logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def send_prediction(
        self,
        prediction: PredictionMessage,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Send prediction result to predictions topic.
        
        Args:
            prediction: Prediction message
            callback: Optional delivery callback
        
        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            logger.warning("Producer not connected. Attempting to connect...")
            if not self.connect():
                return False
        
        try:
            key = prediction.transaction_id.encode('utf-8')
            value = prediction.to_json().encode('utf-8')
            
            self.producer.produce(
                topic=self.settings.PREDICTIONS_TOPIC,
                key=key,
                value=value,
                callback=callback or self._delivery_callback
            )
            
            with self._lock:
                self._delivery_stats['sent'] += 1
            
            # Trigger delivery
            self.producer.poll(0)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send prediction: {str(e)}")
            return False
    
    def send_alert(
        self,
        alert: AlertMessage,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Send fraud alert to alerts topic.
        
        Args:
            alert: Alert message
            callback: Optional delivery callback
        
        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            if not self.connect():
                return False
        
        try:
            key = alert.transaction_id.encode('utf-8')
            value = alert.to_json().encode('utf-8')
            
            self.producer.produce(
                topic=self.settings.ALERTS_TOPIC,
                key=key,
                value=value,
                callback=callback or self._delivery_callback
            )
            
            with self._lock:
                self._delivery_stats['sent'] += 1
            
            self.producer.poll(0)
            
            logger.warning(f"🚨 FRAUD ALERT: {alert.transaction_id} | "
                          f"Amount: ${alert.transaction_amount:,.2f} | "
                          f"Risk: {alert.fraud_probability:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
            return False
    
    def send_to_dlq(
        self,
        dlq_message: DLQMessage,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Send failed message to Dead Letter Queue.
        
        Args:
            dlq_message: DLQ message
            callback: Optional delivery callback
        
        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            if not self.connect():
                return False
        
        try:
            value = dlq_message.to_json().encode('utf-8')
            
            self.producer.produce(
                topic=self.settings.DLQ_TOPIC,
                value=value,
                callback=callback or self._delivery_callback
            )
            
            self.producer.poll(0)
            
            logger.warning(f"Message sent to DLQ: {dlq_message.error_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {str(e)}")
            return False
    
    def flush(self, timeout: float = 10.0) -> int:
        """
        Flush all pending messages.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Number of messages still in queue
        """
        if self.producer:
            return self.producer.flush(timeout)
        return 0
    
    def get_stats(self) -> Dict:
        """Get delivery statistics."""
        with self._lock:
            return self._delivery_stats.copy()
    
    def close(self) -> None:
        """Close the producer."""
        if self.producer:
            self.flush(30)
            logger.info("Kafka Producer closed")
        self.is_connected = False


# Singleton instance
_producer_instance: Optional[FraudDetectionProducer] = None


def get_producer() -> FraudDetectionProducer:
    """Get singleton producer instance."""
    global _producer_instance
    
    if _producer_instance is None:
        _producer_instance = FraudDetectionProducer()
        _producer_instance.connect()
    
    return _producer_instance