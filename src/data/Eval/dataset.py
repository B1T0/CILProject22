import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_eval_dataset(path, bs=32, zero_index = True):
    indices_users, indices_movies = [], []
    df = pd.read_csv(path)
    for i, x in df.iterrows():
        name, _ = x['Id'], x['Prediction']
        user, movie = name.replace('c', '').replace('r', '').split('_')
        if zero_index:
            user, movie = int(user) - 1, int(movie) - 1
        indices_users.append(user)
        indices_movies.append(movie)

    users = torch.tensor(indices_users).to('cuda:0')
    items = torch.tensor(indices_movies).to('cuda:0')
    dataset = TensorDataset(users, items)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
    return dataloader