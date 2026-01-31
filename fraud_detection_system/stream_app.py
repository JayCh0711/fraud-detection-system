"""
Streaming Application Entry Point

Run this to start the Kafka-based fraud detection stream processor.

Usage:
    python stream_app.py --mode batch
    python stream_app.py --mode single
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.streaming.stream_processor import create_stream_processor, StreamingConfig
from src.streaming.config import kafka_settings
from src.logger import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fraud Detection Stream Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stream_app.py --mode batch
    python stream_app.py --mode single --batch-size 50
    python stream_app.py --bootstrap-servers kafka:9092
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['batch', 'single'],
        default='batch',
        help='Processing mode: batch or single (default: batch)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for batch mode (default: 100)'
    )
    
    parser.add_argument(
        '--batch-timeout',
        type=float,
        default=1.0,
        help='Batch timeout in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--bootstrap-servers',
        type=str,
        default=None,
        help='Kafka bootstrap servers (default: from config)'
    )
    
    parser.add_argument(
        '--consumer-group',
        type=str,
        default=None,
        help='Consumer group ID (default: from config)'
    )
    
    parser.add_argument(
        '--input-topic',
        type=str,
        default=None,
        help='Input transactions topic (default: from config)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Override settings from args
    if args.bootstrap_servers:
        kafka_settings.KAFKA_BOOTSTRAP_SERVERS = args.bootstrap_servers
    
    if args.consumer_group:
        kafka_settings.CONSUMER_GROUP_ID = args.consumer_group
    
    if args.input_topic:
        kafka_settings.TRANSACTIONS_TOPIC = args.input_topic
    
    # Create config
    config = StreamingConfig(
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout
    )
    
    logger.info(f"{'='*60}")
    logger.info("FRAUD DETECTION STREAMING APPLICATION")
    logger.info(f"{'='*60}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Bootstrap Servers: {kafka_settings.KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Consumer Group: {kafka_settings.CONSUMER_GROUP_ID}")
    logger.info(f"Input Topic: {kafka_settings.TRANSACTIONS_TOPIC}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"{'='*60}")
    
    # Create and start processor
    processor = create_stream_processor(config)
    
    try:
        processor.start(mode=args.mode)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()