import torch


def left_normalize_adj(adj:torch.sparse.Tensor):
    degrees = adj.to_dense().sum(dim=1).float()
    degrees = 1/degrees
    print(f'normalization matrix {degrees.size()}')
    print(f'adjacency matrix {adj.size()}')
    print(torch.diag(degrees).size())
    #correct left normalization --> we map to the number of output dimensions
    return torch.sparse.mm(torch.diag(degrees).to_sparse_coo().float(), adj)
