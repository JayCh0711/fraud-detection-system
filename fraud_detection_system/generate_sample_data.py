"""
Generate Sample Fraud Detection Dataset
Based on PaySim synthetic financial dataset structure
"""

import os
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    raise RuntimeError(
        "pandas and numpy are required to generate sample data. Install with `pip install -r requirements.txt`."
    ) from e
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_fraud_data(n_samples: int = 100000) -> pd.DataFrame:
    """Generate synthetic fraud detection dataset."""
    
    print(f"Generating {n_samples:,} sample transactions...")
    
    # Transaction types
    transaction_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    type_weights = [0.35, 0.25, 0.20, 0.10, 0.10]
    
    # Generate base data
    data = {
        'step': np.random.randint(1, 744, n_samples),  # Hour of simulation
        'type': np.random.choice(transaction_types, n_samples, p=type_weights),
        'amount': np.abs(np.random.exponential(50000, n_samples)),
        'name_orig': [f'C{i}' for i in np.random.randint(100000, 999999, n_samples)],
        'old_balance_org': np.abs(np.random.exponential(100000, n_samples)),
        'name_dest': [f'C{i}' for i in np.random.randint(100000, 999999, n_samples)],
        'old_balance_dest': np.abs(np.random.exponential(100000, n_samples)),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate new balances
    df['new_balance_org'] = np.maximum(0, df['old_balance_org'] - df['amount'])
    df['new_balance_dest'] = df['old_balance_dest'] + df['amount']
    
    # Generate fraud labels (approximately 1.5% fraud rate - realistic for BFSI)
    df['is_fraud'] = 0
    
    # Fraud conditions (more realistic patterns):
    # 1. Large transfers (> 200,000) with zero destination balance
    # 2. Cash-out draining entire account
    # 3. Unusual patterns
    
    # Condition 1: Large transfers
    large_transfer_mask = (
        (df['type'] == 'TRANSFER') & 
        (df['amount'] > 200000) & 
        (df['old_balance_dest'] == 0)
    )
    
    # Condition 2: Account drain via cash-out
    drain_mask = (
        (df['type'] == 'CASH_OUT') & 
        (df['amount'] == df['old_balance_org']) &
        (df['amount'] > 100000)
    )
    
    # Apply fraud labels with some randomness
    fraud_indices_1 = df[large_transfer_mask].sample(frac=0.7, random_state=42).index
    fraud_indices_2 = df[drain_mask].sample(frac=0.5, random_state=42).index
    
    # Random fraud for other patterns
    remaining_fraud_count = int(n_samples * 0.015) - len(fraud_indices_1) - len(fraud_indices_2)
    if remaining_fraud_count > 0:
        eligible_indices = df[~df.index.isin(fraud_indices_1.union(fraud_indices_2))].index
        random_fraud_indices = np.random.choice(eligible_indices, remaining_fraud_count, replace=False)
        all_fraud_indices = fraud_indices_1.union(fraud_indices_2).union(pd.Index(random_fraud_indices))
    else:
        all_fraud_indices = fraud_indices_1.union(fraud_indices_2)
    
    df.loc[all_fraud_indices, 'is_fraud'] = 1
    
    # Add is_flagged_fraud (system flagged - usually lower than actual fraud)
    df['is_flagged_fraud'] = 0
    flagged_indices = df[df['is_fraud'] == 1].sample(frac=0.2, random_state=42).index
    df.loc[flagged_indices, 'is_flagged_fraud'] = 1
    
    # Round amounts
    df['amount'] = df['amount'].round(2)
    df['old_balance_org'] = df['old_balance_org'].round(2)
    df['new_balance_org'] = df['new_balance_org'].round(2)
    df['old_balance_dest'] = df['old_balance_dest'].round(2)
    df['new_balance_dest'] = df['new_balance_dest'].round(2)
    
    # Reorder columns
    column_order = [
        'step', 'type', 'amount', 'name_orig', 
        'old_balance_org', 'new_balance_org',
        'name_dest', 'old_balance_dest', 'new_balance_dest',
        'is_fraud', 'is_flagged_fraud'
    ]
    df = df[column_order]
    
    return df


def main():
    # Create data directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate data
    df = generate_fraud_data(n_samples=100000)
    
    # Save to CSV
    output_path = 'data/raw/transactions.csv'
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET GENERATED SUCCESSFULLY")
    print("="*60)
    print(f"Output Path: {output_path}")
    print(f"Total Records: {len(df):,}")
    print(f"Fraud Cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Non-Fraud Cases: {(df['is_fraud']==0).sum():,}")
    print(f"\nColumn Info:")
    print(df.dtypes)
    print(f"\nSample Data:")
    print(df.head())
    print(f"\nFraud Distribution by Type:")
    print(df.groupby('type')['is_fraud'].agg(['count', 'sum', 'mean']))


if __name__ == "__main__":
    main()