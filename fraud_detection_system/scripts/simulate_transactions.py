"""
Transaction Simulator for Testing Kafka Streaming

Generates random transactions and sends them to Kafka topic.
"""

import json
import time
import random
import argparse
from datetime import datetime
import uuid
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confluent_kafka import Producer

from src.streaming.config import kafka_settings
from src.logger import logger


class TransactionSimulator:
    """
    Simulates banking transactions for testing.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        topic: str = None
    ):
        """
        Initialize simulator.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Target topic
        """
        self.bootstrap_servers = bootstrap_servers or kafka_settings.KAFKA_BOOTSTRAP_SERVERS
        self.topic = topic or kafka_settings.TRANSACTIONS_TOPIC
        
        self.producer = None
        self.transaction_count = 0
        
        # Transaction types and weights
        self.transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
        self.type_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
        
        logger.info(f"Initializing Transaction Simulator")
        logger.info(f"  Bootstrap: {self.bootstrap_servers}")
        logger.info(f"  Topic: {self.topic}")
    
    def connect(self) -> bool:
        """Connect to Kafka."""
        try:
            self.producer = Producer({
                'bootstrap.servers': self.bootstrap_servers,
                'acks': 'all'
            })
            
            # Test connection
            self.producer.list_topics(timeout=10)
            
            logger.info("Connected to Kafka")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            return False
    
    def generate_transaction(self, fraud_probability: float = 0.02) -> dict:
        """
        Generate a random transaction.
        
        Args:
            fraud_probability: Probability of generating fraud-like transaction
        
        Returns:
            Transaction dictionary
        """
        # Decide if this should look like fraud
        is_suspicious = random.random() < fraud_probability
        
        # Transaction type
        if is_suspicious:
            # Frauds are more likely TRANSFER or CASH_OUT
            txn_type = random.choice(['TRANSFER', 'CASH_OUT'])
        else:
            txn_type = random.choices(self.transaction_types, weights=self.type_weights)[0]
        
        # Generate amounts
        if is_suspicious:
            # Suspicious: larger amounts
            amount = random.uniform(100000, 1000000)
            old_balance_org = amount  # Often equals amount (account drain)
            new_balance_org = 0  # Account emptied
            old_balance_dest = 0  # New account
            new_balance_dest = amount
        else:
            # Normal: smaller amounts
            amount = random.uniform(100, 50000)
            old_balance_org = random.uniform(amount, amount * 10)
            new_balance_org = max(0, old_balance_org - amount)
            old_balance_dest = random.uniform(0, 100000)
            new_balance_dest = old_balance_dest + amount
        
        transaction = {
            'transaction_id': f'TXN_{uuid.uuid4().hex[:12].upper()}',
            'step': random.randint(1, 744),
            'type': txn_type,
            'amount': round(amount, 2),
            'name_orig': f'C{random.randint(100000000, 999999999)}',
            'old_balance_org': round(old_balance_org, 2),
            'new_balance_org': round(new_balance_org, 2),
            'name_dest': f'C{random.randint(100000000, 999999999)}',
            'old_balance_dest': round(old_balance_dest, 2),
            'new_balance_dest': round(new_balance_dest, 2),
            'timestamp': datetime.now().isoformat(),
            'source_system': 'simulator',
            'channel': random.choice(['mobile', 'web', 'atm', 'branch'])
        }
        
        return transaction
    
    def send_transaction(self, transaction: dict) -> bool:
        """
        Send transaction to Kafka.
        
        Args:
            transaction: Transaction dictionary
        
        Returns:
            True if sent successfully
        """
        try:
            key = transaction['transaction_id'].encode('utf-8')
            value = json.dumps(transaction).encode('utf-8')
            
            self.producer.produce(
                topic=self.topic,
                key=key,
                value=value
            )
            
            self.producer.poll(0)
            self.transaction_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send transaction: {str(e)}")
            return False
    
    def simulate(
        self,
        num_transactions: int = 100,
        transactions_per_second: float = 10.0,
        fraud_probability: float = 0.02
    ) -> None:
        """
        Run simulation.
        
        Args:
            num_transactions: Number of transactions to generate
            transactions_per_second: Rate of transactions
            fraud_probability: Probability of fraud-like transactions
        """
        if not self.producer:
            if not self.connect():
                return
        
        delay = 1.0 / transactions_per_second
        
        logger.info(f"Starting simulation:")
        logger.info(f"  Transactions: {num_transactions}")
        logger.info(f"  Rate: {transactions_per_second} TPS")
        logger.info(f"  Fraud Probability: {fraud_probability:.1%}")
        
        start_time = time.time()
        
        try:
            for i in range(num_transactions):
                transaction = self.generate_transaction(fraud_probability)
                
                if self.send_transaction(transaction):
                    if (i + 1) % 100 == 0 or i == num_transactions - 1:
                        logger.info(f"Sent {i + 1}/{num_transactions} transactions")
                
                if i < num_transactions - 1:
                    time.sleep(delay)
                    
        except KeyboardInterrupt:
            logger.info("Simulation interrupted")
        finally:
            # Flush remaining messages
            self.producer.flush()
            
            elapsed = time.time() - start_time
            actual_rate = self.transaction_count / elapsed if elapsed > 0 else 0
            
            logger.info(f"\nSimulation Complete:")
            logger.info(f"  Transactions Sent: {self.transaction_count}")
            logger.info(f"  Elapsed Time: {elapsed:.2f}s")
            logger.info(f"  Actual Rate: {actual_rate:.2f} TPS")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Transaction Simulator")
    
    parser.add_argument(
        '--num',
        type=int,
        default=100,
        help='Number of transactions (default: 100)'
    )
    
    parser.add_argument(
        '--rate',
        type=float,
        default=10.0,
        help='Transactions per second (default: 10)'
    )
    
    parser.add_argument(
        '--fraud-rate',
        type=float,
        default=0.02,
        help='Fraud probability (default: 0.02)'
    )
    
    parser.add_argument(
        '--bootstrap-servers',
        type=str,
        default=None,
        help='Kafka bootstrap servers'
    )
    
    parser.add_argument(
        '--topic',
        type=str,
        default=None,
        help='Target topic'
    )
    
    args = parser.parse_args()
    
    simulator = TransactionSimulator(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic
    )
    
    simulator.simulate(
        num_transactions=args.num,
        transactions_per_second=args.rate,
        fraud_probability=args.fraud_rate
    )


if __name__ == "__main__":
    main()