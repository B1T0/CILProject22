import torch
import pandas as pd


def left_normalize_adj(adj: torch.sparse.Tensor):
    degrees = adj.to_dense().sum(dim=1)
    degrees = 1 / degrees
    print(f'normalization matrix {degrees.size()}')
    print(f'adjacency matrix {adj.size()}')
    print(torch.diag(degrees).size())
    # correct left normalization --> we map to the number of output dimensions
    return torch.sparse.mm(torch.diag(degrees).to_sparse_coo().float(), adj)


def create_user_movie_adjancency_matrices(file_path, n_ratings, n_users, n_items):

    n = n_users + n_items

    indices_i_movies = [[] for _ in range(n_ratings)]
    indices_j_movies = [[] for _ in range(n_ratings)]

    # mapping from movie embeddings to users
    indices_i_users = [[] for _ in range(n_ratings)]
    indices_j_users = [[] for _ in range(n_ratings)]

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
        indices_i_movies[val].append(movie)
        indices_j_movies[val].append(user)
        #
        indices_i_users[val].append(user)
        indices_j_users[val].append(movie)

    adj_movies = []
    for i in range(n_ratings):
        #
        adj_movies.append(torch.sparse_coo_tensor(torch.tensor([indices_i_movies[i], indices_j_movies[i]]),
                                                       torch.ones(size=(len(indices_i_movies[i]),)),
                                                       size=[n_items, n_users]).coalesce())

    adj_users = []
    for i in range(n_ratings):
        adj_users.append(torch.sparse_coo_tensor(torch.tensor([indices_i_users[i], indices_j_users[i]]),
                                                      torch.ones(size=(len(indices_i_users[i]),)),
                                                      size=[n_users, n_items]).coalesce())

    return adj_movies, adj_users