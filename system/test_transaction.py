import asyncio
import json
import datetime
from nats.aio.client import Client as NATS


async def run():
    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])

    # Sample transaction data
    transaction = {
        "TRANSACTION_ID": 12345,
        "TX_DATETIME": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "CUSTOMER_ID": 101,
        "TERMINAL_ID": 202,
        "TX_AMOUNT": 150.75,
        "MERCHANT_ID": 303,
        "IS_ONLINE": True,
        "TX_FRAUD": False,
        "TX_FRAUD_SCENARIO": "N/A",
        "CUSTOMER_IS_COMPROMISED": False,
        "MERCHANT_IS_COMPROMISED": False,
        "TERMINAL_IS_COMPROMISED": False,
        "TERMINAL_X": 45.123,
        "TERMINAL_Y": -75.456,
    }

    # Publish the transaction data
    await nc.publish("transactions", json.dumps(transaction).encode())
    print(f"Published transaction to 'transactions' topic: {transaction}")

    # Close the connection
    await nc.close()


if __name__ == "__main__":
    asyncio.run(run())
