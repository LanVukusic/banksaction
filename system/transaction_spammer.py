import asyncio
import pandas as pd
import json
import time
from nats.aio.client import Client as NATS

DELAY = 0.2


async def run():
    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])
    print("Connected to NATS")

    # Load the transaction data
    try:
        df = pd.read_parquet("../data/transactions.parquet")
        print(f"Loaded {len(df)} transactions from 'data/transactions.parquet'")
    except FileNotFoundError:
        print("Error: '../data/transactions.parquet' not found.")
        await nc.close()
        return

    # Iterate over the DataFrame and publish each transaction
    for index, row in df.iterrows():
        transaction = row.to_dict()

        # Clean the data before sending
        transaction = clean_transaction_data(transaction)

        try:
            # Publish the transaction to the 'transactions' topic
            await nc.publish("transactions", json.dumps(transaction).encode())
            print(
                f"Published transaction ID: {transaction.get('TRANSACTION_ID', 'N/A')}"
            )

            # Wait for a short period to simulate a real-time stream
            await asyncio.sleep(DELAY)  # 100ms delay
        except Exception as e:
            print(f"Error publishing transaction: {e}")

    # Close the NATS connection
    await nc.close()
    print("Finished publishing all transactions and disconnected from NATS.")


def clean_transaction_data(transaction):
    """
    Clean transaction data by handling NaN values, converting timestamps,
    and ensuring proper data types for JSON serialization.
    """
    cleaned = {}

    for key, value in transaction.items():
        # Handle NaN/None values
        if pd.isna(value):
            cleaned[key] = None
        # Handle timestamps
        elif key == "TX_DATETIME" and isinstance(value, pd.Timestamp):
            cleaned[key] = value.isoformat()
        # Handle numeric fields that should be integers but might be floats
        elif key in [
            "TRANSACTION_ID",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "MERCHANT_ID",
        ] and isinstance(value, (int, float)):
            # Convert to int if it's a whole number, otherwise keep as float
            if value == int(value):
                cleaned[key] = int(value)
            else:
                cleaned[key] = value
        # Handle TX_FRAUD and TX_FRAUD_SCENARIO - keep as integers (0/1)
        elif key in ["TX_FRAUD", "TX_FRAUD_SCENARIO"]:
            if pd.isna(value):
                cleaned[key] = 0  # Default to 0 if NaN
            elif isinstance(value, (int, float)):
                cleaned[key] = int(value)
            elif isinstance(value, bool):
                cleaned[key] = 1 if value else 0
            else:
                cleaned[key] = 0  # Default to 0 for unexpected types
        # Handle boolean fields
        elif key in [
            "IS_ONLINE",
            "CUSTOMER_IS_COMPROMISED",
            "MERCHANT_IS_COMPROMISED",
            "TERMINAL_IS_COMPROMISED",
        ]:
            if pd.isna(value):
                cleaned[key] = False
            elif isinstance(value, (int, float)):
                cleaned[key] = bool(value)
            else:
                cleaned[key] = value
        # For all other fields, keep as is
        else:
            cleaned[key] = value

    return cleaned


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Spammer stopped.")
