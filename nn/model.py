import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, DataLoader

class GraphCNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, mode='cat'):
        super().__init__()
        
        self.node_embed = torch.nn.Linear(node_dim, hidden_dim)
        self.edge_embed = torch.nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        
        # Create GCN layers with Sequential
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(conv)
        
        # Jumping Knowledge for combining representations from different layers
        self.jk = JumpingKnowledge(mode=mode)
        
        if mode == 'cat':
            jk_dim = hidden_dim * num_layers
        else:
            jk_dim = hidden_dim
            
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(jk_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed node features
        x = self.node_embed(x)
        
        # Handle edge features if they exist
        if edge_attr is not None and self.edge_embed is not None:
            # Embed edge features and aggregate to nodes
            edge_emb = self.edge_embed(edge_attr)
            # Simple aggregation: sum edge features to connected nodes
            row, col = edge_index
            edge_aggr = torch.zeros_like(x)
            edge_aggr = edge_aggr.index_add_(0, row, edge_emb)
            edge_aggr = edge_aggr.index_add_(0, col, edge_emb)
            x = x + edge_aggr
        
        # Apply graph convolutions (like CNN layers but respecting connectivity)
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            xs.append(x)
        
        # Use JumpingKnowledge to combine representations from all layers
        x = self.jk(xs)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x.squeeze(-1)

# Alternative version using PyG's Sequential for clearer architecture
class GraphCNNSequential(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        
        self.node_embed = torch.nn.Linear(node_dim, hidden_dim)
        self.edge_embed = torch.nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        
        # Build the GNN using PyG's Sequential
        self.gnn = Sequential('x, edge_index, batch', [
            (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (GCNConv(hidden_dim, hidden_dim), 'x, edge_index -> x'),
        ])
        
        self.jk = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=num_layers)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed node features
        x = self.node_embed(x)
        
        # Handle edge features
        if edge_attr is not None and self.edge_embed is not None:
            edge_emb = self.edge_embed(edge_attr)
            row, col = edge_index
            edge_aggr = torch.zeros_like(x)
            edge_aggr = edge_aggr.index_add_(0, row, edge_emb)
            x = x + edge_aggr
        
        # Apply GNN layers
        x = self.gnn(x, edge_index, batch)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x.squeeze(-1)

# Enhanced version with residual connections and batch normalization
class AdvancedGraphCNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=4):
        super().__init__()
        
        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(node_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU()
        )
        
        self.edge_embed = torch.nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for i in range(num_layers):
            conv = GCNConv(hidden_dim, hidden_dim)
            bn = torch.nn.BatchNorm1d(hidden_dim)
            self.convs.append(conv)
            self.bns.append(bn)
        
        self.jk = JumpingKnowledge(mode='max')

        self.embdeder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.6),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial embedding
        x = self.node_embed(x)
        
        # Edge feature incorporation
        if edge_attr is not None and self.edge_embed is not None:
            edge_emb = self.edge_embed(edge_attr)
            row, col = edge_index
            # More sophisticated edge aggregation
            edge_aggr = torch.zeros_like(x)
            degree = torch.zeros(x.size(0), 1, device=x.device)
            degree = degree.scatter_add_(0, row.unsqueeze(1), torch.ones_like(row.unsqueeze(1).float()))
            degree = torch.clamp(degree, min=1)  # Avoid division by zero
            
            edge_aggr = edge_aggr.index_add_(0, row, edge_emb)
            x = x + edge_aggr / degree
        
        # Residual GCN layers with batch norm
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + residual  # Residual connection
            xs.append(x)
        
        # Combine layer representations
        x = self.jk(xs)
        
        # Pool and classify
        x = global_mean_pool(x, batch)
        x = self.embdeder(x)
        x = self.classifier(x)
        
        return x.squeeze(-1)

    def forward_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial embedding
        x = self.node_embed(x)
        
        # Edge feature incorporation
        if edge_attr is not None and self.edge_embed is not None:
            edge_emb = self.edge_embed(edge_attr)
            row, col = edge_index
            # More sophisticated edge aggregation
            edge_aggr = torch.zeros_like(x)
            degree = torch.zeros(x.size(0), 1, device=x.device)
            degree = degree.scatter_add_(0, row.unsqueeze(1), torch.ones_like(row.unsqueeze(1).float()))
            degree = torch.clamp(degree, min=1)  # Avoid division by zero
            
            edge_aggr = edge_aggr.index_add_(0, row, edge_emb)
            x = x + edge_aggr / degree
        
        # Residual GCN layers with batch norm
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = x + residual  # Residual connection
            xs.append(x)
        
        # Combine layer representations
        x = self.jk(xs)
        
        # Pool and get embedding
        x = global_mean_pool(x, batch)
        x = self.embdeder(x)
        
        return x

# Example usage and testing
if __name__ == "__main__":
    # Create sample subgraph data
    num_graphs = 4
    batch_data = []
    
    for i in range(num_graphs):
        num_nodes = torch.randint(5, 15, (1,)).item()
        node_features = torch.randn(num_nodes, 16)  # 16D node features
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random edges
        edge_features = torch.randn(edge_index.size(1), 8)  # 8D edge features
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
        data.y = torch.tensor([i % 2]).float()  # Binary label
        batch_data.append(data)
    
    # Create data loader
    loader = DataLoader(batch_data, batch_size=2, shuffle=True)
    
    # Initialize models
    model1 = GraphCNN(node_dim=16, edge_dim=8, hidden_dim=64)
    model2 = AdvancedGraphCNN(node_dim=16, edge_dim=8, hidden_dim=64)
    
    # Test forward pass
    for batch in loader:
        print(f"Batch size: {batch.num_graphs}")
        print(f"Nodes: {batch.num_nodes}, Edges: {batch.num_edges}")
        
        out1 = model1(batch)
        out2 = model2(batch)
        
        print(f"Model 1 output: {out1.detach().tolist()}")
        print(f"Model 2 output: {out2.detach().tolist()}")
        break
    
    # Training setup example
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Training loop (conceptual)
    model1.train()
    for epoch in range(3):  # Short demo
        for batch in loader:
            optimizer.zero_grad()
            output = model1(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        break
