import asyncio
import json
from nats.aio.client import Client as NATS


async def run():
    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])

    # Define the message handler
    async def message_handler(msg):
        subject = msg.subject
        data = json.loads(msg.data.decode())
        print(f"Received a message on '{subject}': {data}")

    # Subscribe to the 'transactions' topic
    await nc.subscribe("transactions", cb=message_handler)
    print("Subscribed to 'transactions' topic.")

    # Keep the connection alive
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        await nc.close()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Subscriber stopped.")
