import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
from src.models.GraphAutoencoder.layer import ScaledSigmoid
from src.models.GraphALS.utils import left_normalize_adj, create_user_movie_adjancency_matrices
from src.models.GraphAutoencoder.layer import GraphConvolution
import torch.nn.functional as F


class GraphAutoencoder(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, loss='MSE', accum='stack', mode='user_mode', lr=1e-4,
                 n_ratings=5, use_internal_embeddings = False):
        super(GraphAutoencoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
        self.use_internal_embeddings = use_internal_embeddings
        self.lr = lr
        self.mode = mode
        print(f"Mode {mode}")
        if self.mode == 'user_mode':
            self.user_mode = True
        elif self.mode == 'movie_mode':
            self.user_mode = False
        elif self.mode == 'alternating':
            self.user_mode = False

        self.latent_dim = latent_dim
        self.in_features = latent_dim
        self.out_features = latent_dim
        self.user_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )

        #optional usage if we need to decouble things
        # self.user_internal_embeddings = torch.nn.Embedding(
        #     num_embeddings=self.n_users, embedding_dim=self.latent_dim
        # )

        self.movie_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )


        self.own_embedding_dim = 10
        #add additional learned convolution weight
        self.user_own_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.own_embedding_dim
        )
        self.movie_own_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.own_embedding_dim
        )

        nn.init.xavier_uniform(self.user_embeddings.weight)
        nn.init.xavier_uniform(self.movie_embeddings.weight)
        # mapping from user embeddings to movies
        self.adj_movies, self.adj_users = create_user_movie_adjancency_matrices(file_path, n_ratings, n_users, n_items)

        print(f'Movie matrix {self.adj_movies[0].size()}')  # expecting 1000 x 10000
        print(f'User matrix {self.adj_users[0].size()}')
        # normalize
        # might use attention mechanism for normalization
        self.norm_adj_movies = [nn.Parameter(left_normalize_adj(adj), requires_grad=False) for adj in self.adj_movies]
        self.norm_adj_movies = nn.ParameterList(self.norm_adj_movies)
        print(self.norm_adj_movies[0].size)

        self.norm_adj_users = [nn.Parameter(left_normalize_adj(adj), requires_grad=False) for adj in self.adj_users]
        self.norm_adj_users = nn.ParameterList(self.norm_adj_users)


        #shared weight convolution
        self.weight_matrices = [nn.Parameter(torch.FloatTensor(int(self.latent_dim), int(self.out_features)),
                                             requires_grad=True)
                                for _ in range(n_ratings)]

        self.weight_matrices = nn.ParameterList(self.weight_matrices)
        for i in range(n_ratings):
            nn.init.xavier_uniform(self.weight_matrices[i])

        # GCN
        self.gcn1 = [GraphConvolution(in_features=self.latent_dim, out_features=self.out_features, dropout=0.25) for _
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

        if self.use_internal_embeddings: #we ignore polynomial filters here -->
            #but we specify whether identity should play a role or not
            #replace later with possibly completely internal usage
            self.intermediate += self.latent_dim

        self.hidden = 32
        print(self.out_features)
        self.user_dense = nn.Sequential(
            nn.Linear(int(self.intermediate), 2*self.hidden),
            nn.Linear(2* self.hidden, 2*self.hidden),
            nn.ReLU(),
            nn.Linear(2*self.hidden, self.hidden),
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

        if self.user_mode:
            output = []
            weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device).float()]
            for i in range(self.n_rating):
                weight.append(self.weight_matrices[i] + weight[-1])
                output.append(self.gcn1[i](self.movie_embeddings.weight, self.norm_adj_users[i], weight[-1]))

            output = self.accum(output) #also add user_embedding?
            output = output[idx]
            if self.use_internal_embeddings:
                output = torch.concat([output, self.user_embeddings[idx]], dim = 1)
            # output = torch.relu(output.to_dense())
            output = torch.relu(output)
            # print(output.size())
            output = self.user_dense(output)
        else:
            output = []
            weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device).float()]

            for i in range(self.n_rating):
                weight.append(self.weight_matrices[i] + weight[-1])
                output.append(self.gcn1[i](self.user_embeddings.weight, self.norm_adj_movies[i], weight[-1]))
            #
            output = self.accum(output)  # also add user_embedding?
            output = output[idx]
            if self.user_internal_embeddings:
                output = torch.concat([output, self.movie_embeddings[idx]], dim = 1)
            # output = torch.relu(output.to_dense())
            output = torch.relu(output)
            output = self.movie_dense(output)

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        """
        basic training step
        :param train_batch:
        :param batch_num:
        :return:
        """
        # x, y = train_batch  # item, ratings
        ids, rows = train_batch #we receive dense rows
        mask = rows != 0
        pred = self.forward(ids)

        loss = self.loss(pred[mask].float(), rows[mask].float())
        return loss

    def validation_step(self, train_batch, batch_idx):
        """
        basic validation step
        :param x:
        :param batch_num:
        :return:
        """
        ids, rows = train_batch  # we receive dense rows
        mask = rows != 0
        pred = self.forward(ids)
        loss = self.loss(pred[mask].float(), rows[mask].float())
        return loss
