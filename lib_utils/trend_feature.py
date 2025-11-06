import torch
from torch_sparse import SparseTensor

def compute_trend_features(data, k=2):
    """
    Given a PyTorch Geometric Data object, compute for each node:
      - f: the maximum value in its feature vector (data.x).
      - f1 = D^(-0.5) A D^(-0.5) f: the aggregated max of one-hop neighbors.
      - f2 = (D^(-0.5) A D^(-0.5))^2 f: two-hop aggregation.

    The function creates a SparseTensor from data.edge_index with normalized weights,
    and then applies sparse-dense matrix multiplications.
    
    Args:
      data (torch_geometric.data.Data): Input graph data with attributes:
          - x (Tensor): Node feature matrix of shape [num_nodes, num_features].
          - edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].

    Returns:
      Tuple (f, f1, f2):
          - f: Tensor of shape [num_nodes], maximum of each node's features.
          - f1: Tensor of shape [num_nodes], the one-hop normalized aggregation.
          - f2: Tensor of shape [num_nodes], the two-hop aggregation.
    """
    device = data.x.device
    num_nodes = data.num_nodes

    # 1. Compute f: maximum of x for each node.
    f = data.x.max(dim=1)[0]

    # 2. Compute degree values from edge_index (assumes undirected graph)
    row, col = data.edge_index
    # Create a tensor of ones for each edge and then sum them for each source node
    ones = torch.ones(row.size(0), device=device, dtype=torch.float)
    deg = torch.zeros(num_nodes, device=device)
    deg.index_add_(0, row, ones)

    # 3. Compute D^{-0.5} vector.
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 4. Compute normalized edge weights for each edge (i,j): D^{-0.5}[i] * D^{-0.5}[j].
    norm_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 5. Create SparseTensor for the normalized adjacency matrix.
    adj_t = SparseTensor(row=row, col=col, value=norm_weight, sparse_sizes=(num_nodes, num_nodes)).to(device)

    # 6. Perform the sparse matrix multiplication

    feats = [f]
    for i in range(k):
        f = adj_t.matmul(feats[-1].view(-1, 1)).view(-1)
        feats.append(f)

    binary_feats = []
    for i in range(k):
      binary_feats.append((feats[i] - feats[i+1] < 0).unsqueeze(1))
    for i in range(k):
      binary_feats.append((feats[i] - feats[i+1] >= 0).unsqueeze(1))

    return torch.cat(binary_feats, dim=1)
