from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

### Dataset

class MoTDataset(Dataset):
    def __init__(self, edge_index, rating):
        self.edge_index = edge_index
        self.rating = rating

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, idx):
        return self.edge_index[idx], self.rating[idx]


def load_rating_df(path) -> pd.DataFrame:
    """Load the rating dataframe from the given path."""
    df = pd.read_csv(path)
    rating_df = df.Id.str.split("_", expand=True)
    rating_df.columns = ["user", "movie"]
    rating_df["rating"] = df.Prediction
    return rating_df


def compute_gdata(rating_df: pd.DataFrame):
    """Creates the PyTorch Geometric Data object from the rating information"""
    # Create the user and movie indices
    users = rating_df["user"].apply(lambda x: int(x[1:]) - 1)
    movies = rating_df["movie"].apply(lambda x: int(x[1:]) - 1)
    # create edge_index
    edge_index = np.stack([users, movies], axis=1)
    rating = rating_df.rating.values.astype(int)
    # rating = np.ones_like(rating_df.rating.values)
    return MoTDataset(edge_index, rating)


### Model

# Mixture of Tastes model
class MoT(nn.Module):
    def __init__(self, n_users, n_movies, m, k, X_train=None):
        super(MoT, self).__init__()
        self.m = m
        self.k = k
        self.softmax = nn.Softmax(dim=1)

        # embeddings
        self.taste_embeddings = nn.Embedding(n_users, m * k)
        self.attention_embeddings = nn.Embedding(n_users, m * k)
        self.movie_embeddings = nn.Embedding(n_movies, k)
        nn.init.normal_(self.taste_embeddings.weight, std=1 / k)
        nn.init.normal_(self.attention_embeddings.weight, std=1 / k)
        nn.init.normal_(self.movie_embeddings.weight, std=1 / k)

        # bias
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        if not X_train is None:
            u = torch.zeros((n_users, 1))
            v = torch.zeros((1, n_movies))
            for _ in range(5):
                u = (
                    torch.nansum(X_train - v, dim=1)
                    / (~X_train.isnan()).sum(dim=1).float()
                ).view(-1, 1)
                v = (
                    torch.nansum(X_train - u, dim=0)
                    / (~X_train.isnan()).sum(dim=0).float()
                ).view(1, -1)
            self.user_bias = nn.Embedding.from_pretrained(u, freeze=True)
            self.movie_bias = nn.Embedding.from_pretrained(v.T, freeze=True)
        edge = (~X_train.isnan()).nonzero()
        y = X_train[edge[:, 0], edge[:, 1]]
        print("Train RMSE after initialization:", self.rmse(edge, y).item())

    def forward(self, edge):
        user_id = edge[:, 0]
        movie_id = edge[:, 1]

        A = self.attention_embeddings(user_id).view(-1, self.m, self.k)
        U = self.taste_embeddings(user_id).view(-1, self.m, self.k)
        e = self.movie_embeddings(movie_id).unsqueeze(-1)

        attention = self.softmax(torch.bmm(A, e))
        rec_scores = torch.bmm(U, e)

        y_hat = (
            torch.sum(rec_scores * attention, dim=1)
            + self.movie_bias(movie_id)
            + self.user_bias(user_id)
        )

        return y_hat.squeeze()

    def predict(self, edge):
        return torch.clip(self.forward(edge), 1, 5)

    def loss(self, edge, y):
        y_hat = self.forward(edge)
        return torch.sqrt(torch.mean((y_hat - y) ** 2))

    def rmse(self, edge, y):
        pred = self.predict(edge)
        return torch.sqrt(torch.mean((pred - y) ** 2))


### Training

sub_df = load_rating_df("../data/sampleSubmission.csv")
sub_dataset = compute_gdata(sub_df)
sub_loader = DataLoader(sub_dataset, batch_size=1024, shuffle=False)

preds = []
for split in range(5):
    print("Load data...")
    df = load_rating_df(f"../data/cross_validation/train_split_{split}.csv")
    train_df, valid_df = train_test_split(df, test_size=0.01, random_state=42)
    train_dataset = compute_gdata(train_df)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    valid_dataset = compute_gdata(valid_df)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=True)

    print("Create model...")
    X_train = torch.empty((10000, 1000))
    X_train[:] = torch.nan
    X_train[
        train_df.user.str.lstrip("r").astype(int).values - 1,
        train_df.movie.str.lstrip("c").astype(int).values - 1,
    ] = torch.tensor(train_df.rating.values, dtype=torch.float)

    model = MoT(n_users=10000, n_movies=1000, m=4, k=32, X_train=X_train)
    opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print("Train model...")
    valid_rmse, best_valid_rmse, es_ctr = np.inf, np.inf, 0
    train_loss, train_rmse = np.nan, np.nan
    for epoch in tqdm(range(50), desc="Epoch"):

        valid_preds, valid_ys = [], []
        for edge, y in valid_loader:
            valid_preds.append(model.predict(edge).detach())
            valid_ys.append(y)
        valid_y_hat = torch.cat(valid_preds).numpy()
        valid_y = torch.cat(valid_ys).numpy()
        valid_rmse = np.sqrt(np.mean((valid_y_hat - valid_y) ** 2))

        if valid_rmse < best_valid_rmse:
            best_model = deepcopy(model)
            best_valid_rmse = valid_rmse
        else:  # if not improved, stop training
            es_ctr += 1
            if es_ctr >= 5:
                break

        train_loss, train_rmse, batches = 0, 0, 0
        with tqdm(train_loader, desc="Train") as pbar:
            for edge, y in pbar:
                opt.zero_grad()
                tmp_loss = model.loss(edge, y)
                tmp_loss.backward()
                opt.step()

                tmp_rmse = model.rmse(edge, y)
                train_loss += tmp_loss.item()
                train_rmse += tmp_rmse.item()
                batches += 1
                pbar.set_postfix_str(
                    f"Train: loss: {tmp_loss.item():.4f}, "
                    f"Train RMSE: {tmp_rmse.item():.4f}",
                    refresh=False,
                )
                pbar.set_description(
                    f"Valid RMSE: {valid_rmse:.4f}, "
                    f"Train RMSE: {train_rmse/batches:.4f}"
                )

    print("Test Model...")
    test_df = load_rating_df(f"../data/cross_validation/test_split_{split}.csv")
    test_dataset = compute_gdata(test_df)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    split_preds = []
    for edge, _ in test_loader:
        split_preds.append(model(edge).detach().numpy())
    split_preds = np.concatenate(split_preds, axis=0)
    split_pred_df = pd.read_csv(f"../data/cross_validation/test_split_{split}.csv")
    tmp_rmse = np.sqrt(
        np.mean((np.clip(split_preds, 1, 5) - split_pred_df.Prediction.values) ** 2)
    )
    split_pred_df["Prediction"] = split_preds
    split_pred_df.to_csv(
        f"../data/mot_init_dot_pred_test_split_{split}_{tmp_rmse:.4f}.csv", index=False
    )

    print("Make predictions...")
    sub_preds = []
    for edge, _ in sub_loader:
        sub_preds.append(model(edge).detach().numpy())
    sub_preds = np.concatenate(sub_preds, axis=0)
    sub_pred_df = pd.read_csv("../data/sampleSubmission.csv")
    sub_pred_df["Prediction"] = sub_preds
    sub_pred_df.to_csv(
        f"../data/mot_init_dot_pred_sub_split_{split}_{tmp_rmse:.4f}.csv", index=False
    )
    preds.append(sub_pred_df)

avg = np.mean(preds, axis=0)
sub_pred_df = pd.read_csv("../data/sampleSubmission.csv")
sub_pred_df["Prediction"] = avg
sub_pred_df.to_csv(f"../data/mot_init_dot_pred_sub_avg_submission.csv", index=False)
