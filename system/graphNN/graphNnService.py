import asyncio
import json
from nats.aio.client import Client as NATS
import networkx as nx
import numpy as np
from utils import (
    create_customer_merchant_multigraph,
    random_walk_subgraph,
    networkx_to_pyg,
)
from model import AdvancedGraphCNN

# from model import AdvancedGraphCNN
import psycopg2
import pandas as pd
import torch

node_feature_config = {"type": lambda data: np.array(data["type_onehot"])}
edge_feature_config = {
    "amount": lambda data: np.array([data["amt"]]),
}

G: nx.MultiGraph = None
db_conn = None  # Global database connection
model = AdvancedGraphCNN(2, 1, 128, 4)
model.load_state_dict(torch.load("../../models/model.pth"))
model.eval()


def init_transaction_graph(num_latest: int = 3000):
    global G, db_conn
    if db_conn is None:
        print("Database connection not initialized.")
        return

    cursor = db_conn.cursor()

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

    cursor.close()


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
        G.add_node(customer_id, type="customer", type_onehot=[1, 0])
    if not G.has_node(merchant_id):
        G.add_node(merchant_id, type="merchant", type_onehot=[0, 1])

    # Add edge
    G.add_edge(
        customer_id,
        merchant_id,
        key=df["TRANSACTION_ID"].iloc[0],
        amount=df["TX_AMOUNT"].iloc[0],
        fraud=df["TX_FRAUD"].iloc[0],
        weight=float(df["TX_AMOUNT"].iloc[0]),
    )
    print(f"Appended transaction {df['TRANSACTION_ID'].iloc[0]} to the graph.")


def get_transactions_embedding(inputs):
    global G
    sub_G = random_walk_subgraph(G, inputs)
    d = networkx_to_pyg(
        sub_G,
        node_feature_config=node_feature_config,
        edge_feature_config=edge_feature_config,
        edge_dims={"amount": 1},
        edge_total_dim=1,
        node_dims={"type": 2},
        node_total_dim=2,
    )
    return model.forward_embedding(d)


def store_embedding(embedding, name, transaction_references):
    """Stores an embedding in the database using the global connection."""
    global db_conn
    if db_conn is None:
        print("Database connection not initialized.")
        return

    try:
        cursor = db_conn.cursor()

        # Convert tensor to list for storage
        embedding_list = embedding.detach().numpy().tolist()

        query = "INSERT INTO embeddings (embedding, name, transaction_references) VALUES (%s, %s, %s)"
        cursor.execute(query, (embedding_list, name, transaction_references))

        db_conn.commit()
        print(f"Stored embedding for '{name}'.")
        cursor.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        db_conn.rollback()  # Rollback in case of error


async def run():
    global db_conn

    # Initialize database connection
    try:
        db_conn = psycopg2.connect(
            "dbname=transactions user=user password=password host=localhost port=5432 sslmode=disable"
        )
        print("Database connection established.")
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error connecting to database: {error}")
        return

    # Initialize the graph
    init_transaction_graph()

    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])

    # Define the message handler
    async def message_handler(msg):
        subject = msg.subject
        data = json.loads(msg.data.decode())
        print(f"Received a message on '{subject}'")

        # when transaction is received, add it to the graph
        # append_to_transaction_graph(data)

        # Calculate and store the embedding for the customer
        customer_id = data.get("CUSTOMER_ID")
        if customer_id is not None:
            embedding = get_transactions_embedding([customer_id])
            # Assuming we use the customer_id as the name and the transaction_id as the reference
            transaction_id = data.get("TRANSACTION_ID")
            store_embedding(
                embedding.squeeze(), f"customer_{customer_id}", [transaction_id]
            )

    # Subscribe to the 'transactions' topic
    await nc.subscribe("transactions", cb=message_handler)
    print("Subscribed to 'transactions' topic.")

    # Keep the connection alive
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        await nc.close()
    finally:
        if db_conn is not None:
            db_conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Subscriber stopped.")
