import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import pandas as pd
import numpy as np


def create_customer_merchant_multigraph(df: pd.DataFrame) -> nx.MultiGraph:
    """
    Creates a bipartite multigraph from the transaction DataFrame using
    fast, vectorized pandas/networkx functions.

    Nodes:
     - Customers (e.g., 'John_Doe') with attributes:
       'type_onehot', 'type', 'first_name', 'last_name'
     - Merchants (e.g., 'Amazon') with attributes:
       'type_onehot', 'type'

    Edges:
     - Represent transactions, with all columns from the DataFrame
       (e.g., 'amt', 'is_fraud', 'weight') as attributes.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # --- 1. Prepare data for graph creation ---

    # Add 'weight' attribute for GNN models, from 'TX_AMOUNT'
    df_copy["weight"] = df_copy["TX_AMOUNT"].astype(float)

    # Ensure 'original_index' is an attribute for potential debugging
    # This is no longer needed for fraud lookup, but good practice.
    if "original_index" not in df_copy.columns:
        df_copy["original_index"] = df_copy.index

    # --- 2. Create graph from edgelist ---

    # This efficiently creates all nodes and all edges,
    # and copies ALL columns from df_copy as edge attributes.
    G = nx.from_pandas_edgelist(
        df_copy,
        source="CUSTOMER_ID",
        target="MERCHANT_ID",
        edge_attr=True,  # Use all other columns as edge attributes
        create_using=nx.MultiGraph,
    )

    # --- 3. Add node-specific attributes ---

    # Set attributes for customer nodes
    customer_nodes = df_copy["CUSTOMER_ID"].unique()
    customer_attrs = {
        customer_id: {"type_onehot": [1, 0], "type": "customer"}
        for customer_id in customer_nodes
    }
    nx.set_node_attributes(G, customer_attrs)

    # Set attributes for merchant nodes
    merchant_nodes = df_copy["MERCHANT_ID"].unique()
    merchant_attrs = {
        merchant_id: {"type_onehot": [0, 1], "type": "merchant"}
        for merchant_id in merchant_nodes
    }
    nx.set_node_attributes(G, merchant_attrs)

    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def random_walk_subgraph(
    G: nx.MultiGraph,
    start_nodes: list,
    initial_edges: set = None,
    walk_length: int = 150,
    num_walks: int = 5,
    max_subgraph_size: int = 150,
) -> nx.MultiGraph:
    """
    Performs random walks from start_nodes to create a MultiGraph subgraph.

    It collects the specific edges (u, v, key) traversed and builds
    the subgraph from them, preserving all attributes.
    """
    if not start_nodes:
        return nx.MultiGraph()

    visited_nodes = set(start_nodes)
    visited_edges = set(initial_edges) if initial_edges else set()

    # Use a copy of start_nodes for walk initiation
    current_walk_starters = list(start_nodes)
    if not current_walk_starters:
        return nx.MultiGraph()  # Cannot start walk

    for _ in range(num_walks):
        if len(visited_nodes) >= max_subgraph_size:
            break

        # Start each walk from one of the initial seed nodes
        current_node = np.random.choice(current_walk_starters)

        for _ in range(walk_length):
            if len(visited_nodes) >= max_subgraph_size:
                break

            if not G.has_node(current_node):
                break  # Node doesn't exist

            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break  # Dead end

            next_node = np.random.choice(neighbors)

            # Get all edge keys between the two nodes in the MultiGraph
            edge_keys = list(G[current_node][next_node].keys())
            if not edge_keys:
                break  # No edges found, should not happen if neighbors was non-empty

            # Randomly select one of the multiple edges
            key = np.random.choice(edge_keys)

            visited_edges.add((current_node, next_node, key))
            visited_nodes.add(next_node)
            current_node = next_node

    # Create the subgraph from the collected edges.
    # This preserves all node and edge attributes automatically.
    subgraph = G.edge_subgraph(visited_edges).copy()

    # Ensure any isolated start_nodes are also included
    for node in start_nodes:
        if not subgraph.has_node(node) and G.has_node(node):
            subgraph.add_node(node, **G.nodes[node])

    return subgraph
