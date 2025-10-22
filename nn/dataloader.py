import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import pandas as pd
import numpy as np
import warnings
from utils import random_walk_subgraph


class SubgraphDataset(Dataset):
    """
    Creates a PyG Dataset of subgraphs sampled from a larger graph.

    Implements 50/50 balanced sampling of fraudulent and non-fraudulent
    subgraphs, using flexible, user-defined feature processing functions.
    """

    def __init__(
        self,
        G: nx.MultiGraph,
        node_feature_config: dict,
        edge_feature_config: dict,
        num_samples: int = 1000,
        walk_length: int = 50,
        num_walks: int = 5,
        max_subgraph_size: int = 100,
    ):
        super().__init__()
        self.G = G
        self.node_feature_config = node_feature_config
        self.edge_feature_config = edge_feature_config
        self.num_samples = num_samples

        # Store sampling parameters
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.max_subgraph_size = max_subgraph_size

        # --- Identify fraudulent edges and nodes for sampling ---
        self.fraud_edges = []
        fraud_nodes_set = set()

        for u, v, key, data in G.edges(keys=True, data=True):
            if data.get("is_fraud") == 1:
                self.fraud_edges.append((u, v, key))
                fraud_nodes_set.add(u)
                fraud_nodes_set.add(v)

        self.fraud_nodes = list(fraud_nodes_set)
        self.all_nodes = list(G.nodes())

        if not self.all_nodes:
            raise ValueError("Graph has no nodes. Cannot initialize dataset.")
        if not self.fraud_edges:
            warnings.warn("Warning: No fraudulent edges found in the graph.")

        # --- Pre-calculate feature dimensions ---
        # Get sample data to test functions
        sample_node_data = G.nodes[self.all_nodes[0]]

        sample_edge_data = {}
        if self.G.number_of_edges() > 0:
            # Get data from the first edge
            _, _, sample_edge_data = next(iter(self.G.edges(data=True)))
        else:
            warnings.warn("Graph has no edges. Cannot determine edge feature dims.")

        # Calculate and store dimensions for nodes
        self.node_dims, self.node_total_dim = self._get_feature_dims(
            sample_node_data, self.node_feature_config, "node"
        )
        # Calculate and store dimensions for edges
        self.edge_dims, self.edge_total_dim = self._get_feature_dims(
            sample_edge_data, self.edge_feature_config, "edge"
        )

        print(
            f"Initialized dataset. Node features dim: {self.node_total_dim}. Edge features dim: {self.edge_total_dim}."
        )

    def _get_feature_dims(
        self, sample_data: dict, config: dict, data_type: str
    ) -> (dict, int):
        """
        Tests processing functions on sample data to get feature dimensions.
        """

        dims = {}
        total_dim = 0
        if not config:
            return dims, total_dim

        for name, func in config.items():
            try:
                func_out = func(sample_data)
                if not isinstance(func_out, np.ndarray):
                    raise TypeError(
                        f"Function for '{name}' must return a list or array."
                    )
                dim = len(func_out)
                dims[name] = dim
                total_dim += dim
            except Exception as e:
                raise ValueError(
                    f"Error running {data_type} feature function '{name}' on sample data: {e}. "
                    "Ensure function is robust (e.g., use data.get('key', default_val))."
                )
        return dims, total_dim

    def _networkx_to_pyg(self, nx_graph: nx.MultiGraph) -> Data:
        """
        Converts a NetworkX MultiGraph to a PyG Data object using
        the user-defined feature processing configurations.
        """
        if nx_graph.number_of_nodes() == 0:
            # Handle empty graph case using pre-calculated dimensions
            return Data(
                x=torch.empty((0, self.node_total_dim), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, self.edge_total_dim), dtype=torch.float),
                edge_label=torch.empty(0, dtype=torch.float),
                y=torch.tensor([0.0], dtype=torch.float),
            )

        # 1. Create node mapping and features (x)
        node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
        node_features = []
        for node in nx_graph.nodes():
            node_data = nx_graph.nodes[node]
            feature_parts = []

            # Apply each processing function from the config
            for name, func in self.node_feature_config.items():
                try:
                    result = func(node_data)
                    # Safety check for consistent dimension
                    if len(result) != self.node_dims[name]:
                        raise ValueError(f"Inconsistent output dim for '{name}'")
                    feature_parts.extend(result)
                except Exception as e:
                    # Robustness: Fill with zeros if func fails
                    warnings.warn(
                        f"Warning: Node func '{name}' failed (Node: {node}). Using zeros. Error: {e}"
                    )
                    feature_parts.extend([0.0] * self.node_dims[name])

            node_features.append(feature_parts)

        x = torch.tensor(node_features, dtype=torch.float)

        # 2. Create edge index, features (edge_attr), and labels (edge_label)
        edge_indices = []
        edge_features = []
        edge_labels = []

        for u, v, data in nx_graph.edges(data=True):
            if u not in node_mapping or v not in node_mapping:
                continue

            u_idx, v_idx = node_mapping[u], node_mapping[v]
            edge_indices.append([u_idx, v_idx])

            # Apply each processing function from the config
            feature_parts = []
            for name, func in self.edge_feature_config.items():
                try:
                    result = func(data)
                    if len(result) != self.edge_dims[name]:
                        raise ValueError(f"Inconsistent output dim for '{name}'")
                    feature_parts.extend(result)
                except Exception as e:
                    warnings.warn(
                        f"Warning: Edge func '{name}' failed. Using zeros. Error: {e}"
                    )
                    feature_parts.extend([0.0] * self.edge_dims[name])

            edge_features.append(feature_parts)

            # Get fraud label (this is separate from features)
            edge_labels.append(float(data.get("is_fraud", 0.0)))

        # Handle case with nodes but no edges
        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, self.edge_total_dim), dtype=torch.float)
            edge_label = torch.empty(0, dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            edge_label = torch.tensor(edge_labels, dtype=torch.float)

        # 3. Create graph-level label (y)
        graph_has_fraud = (edge_label.sum() > 0).float()

        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label=edge_label,
            y=graph_has_fraud.unsqueeze(0),
        )

        return pyg_data

    def _sample_fraudulent_subgraph(self) -> nx.MultiGraph:
        """
        Samples a subgraph starting from 2-4 known fraudulent transactions.
        """
        if not self.fraud_edges:
            return self._sample_non_fraudulent_subgraph()

        num_fraud_jumps = np.random.randint(2, 5)
        num_to_sample = min(num_fraud_jumps, len(self.fraud_edges))

        indices = np.random.choice(len(self.fraud_edges), num_to_sample, replace=False)

        initial_edges = set()
        start_nodes = set()
        for idx in indices:
            u, v, key = self.fraud_edges[idx]
            initial_edges.add((u, v, key))
            start_nodes.add(u)
            start_nodes.add(v)

        return random_walk_subgraph(
            self.G,
            start_nodes=list(start_nodes),
            initial_edges=initial_edges,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            max_subgraph_size=self.max_subgraph_size,
        )

    def _sample_non_fraudulent_subgraph(self) -> nx.MultiGraph:
        """
        Setting the stage...
                Samples a subgraph starting from 2-4 random nodes.
        """
        num_start_nodes = np.random.randint(2, 5)
        num_to_sample = min(num_start_nodes, len(self.all_nodes))

        if num_to_sample == 0:
            return nx.MultiGraph()

        start_nodes = np.random.choice(self.all_nodes, num_to_sample, replace=False)

        return random_walk_subgraph(
            self.G,
            start_nodes=list(start_nodes),
            initial_edges=None,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            max_subgraph_size=self.max_subgraph_size,
        )

    def len(self) -> int:
        return self.num_samples

    def get(self, idx: int) -> Data:
        """
        Gets the idx-th sample with 50/50 balancing.
        """
        if idx % 2 == 0:
            subgraph_nx = self._sample_fraudulent_subgraph()
        else:
            subgraph_nx = self._sample_non_fraudulent_subgraph()

        return self._networkx_to_pyg(subgraph_nx)
