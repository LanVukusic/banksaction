import asyncio
import pandas as pd
import json
from nats.aio.client import Client as NATS

# transaction_id,timestamp,amount,currency,location,city,merchant,channel,card_masked,card_issuer,TX_FRAUD,FRAUD_SCENARIO

DELAY = 0.1
PATH = "../data/imbalanced_fraud_dataset.csv"


async def run():
    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])
    print("Connected to NATS")

    # Load the transaction data
    try:
        df = pd.read_csv(PATH)
        print(f"Loaded {len(df)} transactions from 'data/transactions.parquet'")
    except FileNotFoundError:
        print("Error:not found." + PATH)
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
                f"Published transaction ID: {transaction.get('transaction_id', 'N/A')}"
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
        if pd.isna(value) or value in [None, "None", "null", ""]:
            # Set appropriate defaults based on field type
            if key in ["TX_FRAUD", "FRAUD_SCENARIO"]:
                cleaned[key] = 0
            elif key in ["amount"]:
                cleaned[key] = 0.0
            elif key in [
                "currency",
                "location",
                "city",
                "merchant",
                "channel",
                "card_masked",
                "card_issuer",
            ]:
                cleaned[key] = None
            else:
                cleaned[key] = None

        # Handle timestamps
        elif key == "timestamp" and isinstance(value, (pd.Timestamp, str)):
            if isinstance(value, pd.Timestamp):
                cleaned[key] = value.isoformat()
            elif isinstance(value, str):
                # If it's already a string, keep it as is (assuming proper format)
                cleaned[key] = value

        # Handle transaction_id (UUID field)
        elif key == "transaction_id" and isinstance(value, str):
            cleaned[key] = value  # Keep UUID as string

        # Handle amount (numeric field)
        elif key == "amount":
            try:
                cleaned[key] = float(value)
            except (ValueError, TypeError):
                cleaned[key] = 0.0

        # Handle currency, location, city, merchant, channel, card_masked, card_issuer (string fields)
        elif key in [
            "currency",
            "location",
            "city",
            "merchant",
            "channel",
            "card_masked",
            "card_issuer",
        ]:
            cleaned[key] = str(value) if value is not None else None

        # Handle fraud flags - convert to integers (0/1)
        elif key in ["TX_FRAUD", "FRAUD_SCENARIO"]:
            if isinstance(value, (int, float)):
                cleaned[key] = int(value)
            elif isinstance(value, bool):
                cleaned[key] = 1 if value else 0
            elif isinstance(value, str) and value.lower() in ["true", "1", "yes"]:
                cleaned[key] = 1
            elif isinstance(value, str) and value.lower() in [
                "false",
                "0",
                "no",
                "none",
            ]:
                cleaned[key] = 0
            else:
                cleaned[key] = 0  # Default to 0 for unexpected types

        # For all other fields, keep as is
        else:
            cleaned[key] = value

    return cleaned


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Spammer stopped.")
