import asyncio
import json
from nats.aio.client import Client as NATS
import networkx as nx
from utils import create_customer_merchant_multigraph
from model import AdvancedGraphCNN

# from model import AdvancedGraphCNN
import psycopg2
import pandas as pd
import torch

G: nx.MultiGraph = None
model = AdvancedGraphCNN(2, 1, 128, 4)
model.load_state_dict(torch.load("../../models/model.pth"))
model.eval()


# model = AdvancedGraphCNN(2, 1, 128, 4)
def init_transaction_graph(num_latest: int = 3000):
    global G
    conn = None
    conn = psycopg2.connect(
        "dbname=transactions user=user password=password host=localhost port=5432 sslmode=disable"
    )

    cursor = conn.cursor()

    # Query to get the latest transactions
    query = f"SELECT * FROM transactions ORDER BY TX_DATETIME DESC LIMIT {num_latest};"
    cursor.execute(query)

    # Get all results
    rows = cursor.fetchall()

    # Get column names
    columns = [desc[0].upper() for desc in cursor.description]

    # Create DataFrame manually
    df = pd.DataFrame(rows, columns=columns)

    # Create the graph
    if not df.empty:
        G = create_customer_merchant_multigraph(df)
        print(
            f"Transaction graph initialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )
    else:
        print("No transactions found to initialize the graph.")
        G = nx.MultiGraph()

    # except Exception as error:
    #     print(f"Error while connecting to PostgreSQL: {error}")
    #     G = nx.MultiGraph()
    # finally:
    #     if conn:
    #         conn.close()


# update transaction type!
def append_to_transaction_graph(transaction):
    global G
    if G is None:
        print("Graph not initialized. Call init_transaction_graph first.")
        return

    # Convert the transaction dictionary to a DataFrame
    df = pd.DataFrame([transaction])

    # Extract nodes and edges from the transaction
    customer_id = df["CUSTOMER_ID"].iloc[0]
    merchant_id = df["MERCHANT_ID"].iloc[0]

    # Add nodes if they don't exist
    if not G.has_node(customer_id):
        G.add_node(customer_id, type="customer")
    if not G.has_node(merchant_id):
        G.add_node(merchant_id, type="merchant")

    # Add edge
    G.add_edge(
        customer_id,
        merchant_id,
        key=df["TRANSACTION_ID"].iloc[0],
        amount=df["TX_AMOUNT"].iloc[0],
        fraud=df["TX_FRAUD"].iloc[0],
    )
    print(f"Appended transaction {df['TRANSACTION_ID'].iloc[0]} to the graph.")


def get_transactions_embedding(inputs):
    return model.forward_embedding(inputs)


async def run():
    # Initialize the graph
    init_transaction_graph()

    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])

    # Define the message handler
    async def message_handler(msg):
        # subject = msg.subject
        data = json.loads(msg.data.decode())
        # print(f"Received a message on '{subject}': {data}")

        # when transaction is received, add it to the graph
        append_to_transaction_graph(data)

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
