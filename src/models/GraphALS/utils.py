import torch


def left_normalize_adj(adj:torch.sparse.Tensor):
    degrees = adj.to_dense().sum(dim=1)
    degrees = torch.sqrt(degrees)
    degrees = 1/degrees
    print(f'normalization matrix {degrees.size()}')
    print(f'adjacency matrix {adj.size()}')
    return torch.diagonal(degrees).mul(adj)
