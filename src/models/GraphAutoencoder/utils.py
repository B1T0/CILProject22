import torch


def normalize_adj(adj:torch.sparse.Tensor):
    degrees = adj.to_dense().sum(dim=1)
    degrees = torch.sqrt(degrees)
    return adj.mul(torch.mul(degrees.transpose(0, 1), degrees))
