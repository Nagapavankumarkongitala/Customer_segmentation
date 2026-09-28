import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_customers=1000, num_transactions=10000):
    # Generate Customer Data
    customers = pd.DataFrame({
        'CustomerID': range(1, num_customers + 1),
        'Age': np.random.randint(18, 65, num_customers),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], num_customers, p=[0.45, 0.5, 0.05]),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], num_customers, p=[0.6, 0.3, 0.1])
    })

    # Generate Transaction Data
    transactions = pd.DataFrame({
        'TransactionID': range(1, num_transactions + 1),
        'CustomerID': np.random.randint(1, num_customers + 1, num_transactions),
        'TransactionDate': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(num_transactions)],
        'Amount': np.random.uniform(10, 500, num_transactions).round(2)
    })

    # Add some churn indicator (simplified for demonstration)
    customers['HasChurned'] = np.random.choice([0, 1], num_customers, p=[0.8, 0.2]) # 20% churn rate

    return customers, transactions

if __name__ == "__main__":
    customers_df, transactions_df = generate_synthetic_data()

    # Save to CSV
    customers_df.to_csv("customers.csv", index=False)
    transactions_df.to_csv("transactions.csv", index=False)

    print("Synthetic data generated and saved to customers.csv and transactions.csv")