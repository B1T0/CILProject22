import torch
import pandas as pd


def normalize_adj(adj: torch.sparse.Tensor):
    degrees = adj.to_dense().sum(dim=1)
    degrees = torch.sqrt(degrees)
    degrees = 1 / degrees
    degrees = degrees.unsqueeze(1)
    normalized = degrees.transpose(0, 1).mul(degrees).to_sparse_coo()
    print(normalized.size())
    print(adj.size())
    return adj.mul(normalized)


def create_full_adjacency_matrix(file_path, n_ratings, n_users, n_items, identity=False, dense=False):
    """
    creates matrix  |0    A|  where A is a n_users * n_items matrix
                    |A^T  0|
    for propagation we there multiply with vector |movie_embeddings user_embeddings|
    :param file_path:
    :param n_ratings:
    :param n_users:
    :param n_items:
    :param identity:
    :return:
    """
    n = n_users + n_items
    indices_i = [[] for _ in range(n_ratings)]
    indices_j = [[] for _ in range(n_ratings)]

    df = pd.read_csv(file_path)
    print('Creating adjacency matrices')
    for i, x in df.iterrows():
        name, val = x['Id'], x['Prediction']
        user, movie = name.replace('c', '').replace('r', '').split('_')
        user, movie = int(user) - 1, int(movie) - 1
        val = int(val) - 1
        if user > n_users:
            raise Exception(f"More users in file")
        if movie > n_items:
            raise Exception(f"More movies in file")
        indices_i[val].append(movie)
        indices_j[val].append(user + n_items)
        #
        indices_i[val].append(user + n_items)
        indices_j[val].append(movie)

    adj = []
    for i in range(n_ratings):
        adj_matrix = torch.sparse_coo_tensor(torch.tensor([indices_i[i], indices_j[i]]),
                                             torch.ones(size=(len(indices_i[i]),)),
                                             size=[n, n]).coalesce()
        if identity:
            adj_matrix = adj_matrix + torch.eye(n, n).to_sparse_coo()
            adj_matrix = adj_matrix.coalesce()
        if dense:
            adj_matrix = adj_matrix.to_dense()
        adj.append(adj_matrix)

    return adj
