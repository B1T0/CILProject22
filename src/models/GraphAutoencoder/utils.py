import torch


def normalize_adj(adj:torch.sparse.Tensor):
    degrees = adj.to_dense().sum(dim=1)
    degrees = torch.sqrt(degrees)
    degrees = 1/degrees
    degrees = degrees.unsqueeze(1)
    normalized = degrees.transpose(0, 1).mul(degrees).to_sparse_coo()
    print(normalized.size())
    print(adj.size())
    return adj.mul(normalized)
