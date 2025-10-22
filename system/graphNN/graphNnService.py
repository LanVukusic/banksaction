import asyncio
import json
from nats.aio.client import Client as NATS
import networkx as nx
from utils import create_customer_merchant_multigraph
from model import AdvancedGraphCNN
import psycopg2
import pandas as pd

G: nx.MultiGraph = None
model = AdvancedGraphCNN(2, 1, 128, 4)


def init_transaction_graph(num_latest: int = 4000):
    global G
    conn = None
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host="localhost", database="transactions", user="user", password="password"
        )

        # Query to get the latest transactions
        query = (
            f"SELECT * FROM transactions ORDER BY TX_DATETIME DESC LIMIT {num_latest};"
        )

        # Load data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # Create the graph
        if not df.empty:
            G = create_customer_merchant_multigraph(df)
            print(
                f"Transaction graph initialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
            )
        else:
            print("No transactions found to initialize the graph.")
            G = nx.MultiGraph()

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if conn:
            conn.close()


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


# DONT IMPLEMENT YET
# def get_transactions_embedding(inputs: Array[int]):
#     # get transaction ids
#     # createa random_walk_subgraph from the inputs
#     # get data from subgraph
#     # use the model, embed the data
#     # return embeddings
#     ...


async def run():
    # Initialize the graph
    init_transaction_graph()

    # Connect to NATS
    nc = NATS()
    await nc.connect(servers=["nats://localhost:4222"])

    # Define the message handler
    async def message_handler(msg):
        subject = msg.subject
        data = json.loads(msg.data.decode())
        print(f"Received a message on '{subject}': {data}")

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
