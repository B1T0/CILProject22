import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
from src.models.GraphAutoencoder.layer import ScaledSigmoid, GraphSelfAttentionLayer
from src.models.GraphAutoencoder.utils import create_full_adjacency_matrix


class GraphAttention(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, graph_hidden=32, hidden=32, alpha=0.2, loss='MSE',
                 accum='stack', mode='user_mode', lr=1e-4,
                 n_ratings=5, use_internal_embeddings=False):
        super(GraphAttention, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
        self.use_internal_embeddings = use_internal_embeddings
        self.lr = lr
        self.mode = mode
        self.alpha = alpha
        print(f"Mode {mode}")
        if self.mode == 'user_mode':
            self.user_mode = True
        elif self.mode == 'movie_mode':
            self.user_mode = False
        elif self.mode == 'alternating':
            self.user_mode = False

        self.latent_dim = latent_dim
        self.in_features = latent_dim
        self.out_features = graph_hidden

        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.n, embedding_dim=self.latent_dim
        )
        # self.user_embeddings = torch.nn.Embedding(
        #     num_embeddings=self.n_users, embedding_dim=self.latent_dim
        # )
        #
        # self.movie_embeddings = torch.nn.Embedding(
        #     num_embeddings=self.n_items, embedding_dim=self.latent_dim
        # )

        self.adjacency_matrix = create_full_adjacency_matrix(file_path, n_ratings, n_users, n_items, identity=True, dense=True)

        # shared weight matrices
        self.weight_matrices = [nn.Parameter(torch.FloatTensor(int(self.latent_dim), int(self.out_features)),
                                             requires_grad=True)
                                for _ in range(n_ratings)]

        self.weight_matrices = nn.ParameterList(self.weight_matrices)
        for i in range(n_ratings):
            nn.init.xavier_uniform(self.weight_matrices[i])

        # GCN
        self.gcn1 = [GraphSelfAttentionLayer(in_features=self.latent_dim,
                                             out_features=self.out_features, dropout=0.25, alpha=self.alpha) for _
                     in
                     range(n_ratings)]

        self.gcn1 = nn.ModuleList(self.gcn1)

        self.activation = nn.ReLU()
        if accum == 'stack':
            self.accum = lambda x: torch.concat(x, dim=1)
            self.intermediate = self.out_features * self.n_rating
        if accum == 'sum':
            self.accum = lambda x: torch.stack(x, dim=1).sum(dim=1)
            self.intermediate = self.out_features

        self.hidden = hidden

        self.user_dense = nn.Sequential(
            nn.Linear(int(self.intermediate), self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.Linear(self.hidden, self.n_items),
            ScaledSigmoid()
        )

        self.movie_dense = nn.Sequential(
            nn.Linear(int(self.intermediate), self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.Linear(self.hidden, self.n_users),
            ScaledSigmoid()
        )

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
        # self.decoder = nn.Bilinear(int(self.out_features), int(self.out_features), int(self.output_size))

    def freeze_user(self):
        print('Freezing User Embeddings')
        self.user_embeddings.requires_grad_(False)

    def unfreeze_user(self):
        print('Unfreezing User Embeddings')
        self.user_embeddings.requires_grad_(True)

    def freeze_item(self):
        print('Freezing User Embeddings')
        self.movie_embeddings.requires_grad_(False)

    def unfreeze_item(self):
        print('Unfreezing User Embeddings')
        self.movie_embeddings.requires_grad_(True)

    def forward(self, idx):
        output = []
        weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device)]
        for i in range(self.n_rating):
            # print(weight.size())
            # print(self.weight_matrices[i].size())
            weight.append(self.weight_matrices[i] + weight[-1])
            output.append(self.gcn1[i](self.embeddings.weight, self.adjacency_matrix[i], weight[-1]))
        output = self.accum(output)
        if self.user_mode:
            output = output[idx+self.n_items]
            output = torch.relu(output)
            output = self.user_dense(output)
        else:
            output = output[idx]
            output = torch.relu(output)
            output = self.movie_dense(output)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch):
        """
        basic training step
        :param train_batch:
        :param batch_num:
        :return:
        """
        # x, y = train_batch  # item, ratings
        ids, rows = train_batch  # we receive dense rows
        mask = rows != 0
        pred = self.forward(ids)
        loss = self.loss(rows[mask], pred[mask])
        return loss

    def validation_step(self, train_batch, batch_num):
        """
        basic validation step
        :param x:
        :param batch_num:
        :return:
        """
        ids, rows = train_batch  # we receive dense rows
        mask = rows != 0
        pred = self.forward(ids)
        loss = self.loss(rows[mask], pred[mask])
        return loss