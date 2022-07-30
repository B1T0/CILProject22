import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.GraphAutoencoder.layer import ScaledSigmoid, GraphSelfAttentionLayer
from src.models.GraphAutoencoder.utils import create_full_adjacency_matrix

"""
This is an implementation of our proposed improvement upon SVD++ using a Graph Attention Layer
"""


class GraphAttention(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, global_mean, graph_hidden=32, hidden=32, alpha=0.2,
                 loss='MSE'
                 , lr=1e-4,
                 n_ratings=5, use_internal_embeddings=False, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.gm = global_mean
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
        self.use_internal_embeddings = use_internal_embeddings
        self.lr = lr
        self.dropout = dropout
        self.alpha = alpha

        self.latent_dim = latent_dim
        self.in_features = latent_dim
        self.out_features = graph_hidden

        # Embedding used for Graph Attention
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.n, embedding_dim=self.latent_dim
        )

        self.movie_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_movies, embedding_dim=self.latent_dim
        )

        self.movie_bias = torch.nn.Embedding(
            num_embeddings=self.n_movies, embedding_dim=1
        )

        self.user_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )

        self.user_bias = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=1
        )

        self.adjacency_matrix = create_full_adjacency_matrix(file_path, n_ratings, n_users, n_items,
                                                             identity=True, dense=True)

        self.adjacency_matrix = [nn.Parameter(matrix, requires_grad=False) for matrix in self.adjacency_matrix]
        self.adjacency_matrix = nn.ParameterList(self.adjacency_matrix)
        # shared weight matrices
        self.weight_matrices = [nn.Parameter(torch.FloatTensor(int(self.latent_dim), int(self.out_features)),
                                             requires_grad=True)
                                for _ in range(n_ratings)]

        self.weight_matrices = nn.ParameterList(self.weight_matrices)
        for i in range(n_ratings):
            nn.init.xavier_normal_(self.weight_matrices[i], gain=1.414)

        # GCN
        self.gcn1 = [GraphSelfAttentionLayer(in_features=self.latent_dim,
                                             out_features=self.out_features, dropout=dropout, alpha=self.alpha) for _
                     in
                     range(n_ratings)]

        self.gcn1 = nn.ModuleList(self.gcn1)

        self.activation = nn.ReLU()

        self.accum = lambda x: torch.stack(x, dim=1).sum(dim=1)
        self.intermediate = self.out_features

        self.hidden = hidden

        self.loss = loss
        if loss == 'MSE':
            self.loss = nn.MSELoss()
            self.output_size = 1
            self.activation = ScaledSigmoid()
            self.target = lambda x: x.float()
            self.evaluate = lambda x: x
        else:
            raise Exception(f"Loss {loss} not implemented")

        self.mse = nn.MSELoss()

    def forward(self, idx):
        # print(idx)
        # print(self.adjacency_matrix[0][idx].size())
        # print(torch.nonzero(self.adjacency_matrix[0][idx]))
        movie_idx, user_idx = idx

        user_idx_shifted = user_idx + self.n_items
        rows = torch.nonzero(self.adjacency_matrix[0][user_idx_shifted])[:, 1]
        # print(idx)
        mask = []
        for _, i in enumerate(user_idx_shifted):
            # print(i.size())
            # print(rows.size())
            # also works because of bi partiteness
            mask.append(torch.nonzero((rows == i), as_tuple=True)[0])
        mask = torch.concat(mask)
        # print(mask)
        # subsampled_adjacency = self.adjacency_matrix[0][rows][:, rows]
        # print(subsampled_adjacency)
        # print(subsampled_adjacency.size())
        # print(self.embeddings(rows).size())
        output = []
        weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device)]
        # graph attention convolution
        for i in range(self.n_rating):
            weight.append(self.weight_matrices[i] + weight[-1])
            output.append(self.gcn1[i](self.embeddings(rows), self.adjacency_matrix[i][rows][:, rows], weight[-1]))

        user_emb_sum = self.accum(output)
        user_emb_sum = user_emb_sum[mask]
        user_emb = self.user_embeddings(user_idx - self.n_items)
        movie_emb = self.movie_embeddings(movie_idx)
        b_u = self.user_bias(user_idx - self.n_items)
        b_m = self.movie_bias(movie_idx)
        pred_r_ui = torch.sum((user_emb + user_emb_sum) * movie_emb, dim=1) + \
                    torch.squeeze(b_u) + torch.squeeze(b_m) + self.gm

        return pred_r_ui

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        """
        basic training step
        :param train_batch:
        :param batch_num:
        :return:
        """
        for i, x in enumerate(train_batch):
            train_batch[i] = x.to('cuda:0')
        # x, y = train_batch  # item, ratings
        ids, ratings = train_batch
        pred = self.forward(ids)
        loss = self.loss(pred, ratings.float())
        return loss, 0

    def validation_step(self, train_batch, batch_idx):
        """
        basic validation step
        :param x:
        :param batch_num:
        :return:
        """
        for i, x in enumerate(train_batch):
            if x is not None:
                train_batch[i] = x.to('cuda:0')
        ids, ratings = train_batch
        pred = self.forward(ids)
        loss = self.loss(pred, ratings.float())
        return loss, 0
