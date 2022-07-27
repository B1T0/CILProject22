import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
from src.models.GraphAutoencoder.layer import ScaledSigmoid
from src.models.GraphALS.utils import left_normalize_adj
from src.models.GraphAutoencoder.layer import GraphConvolution
import torch.nn.functional as F


class GraphAutoencoder(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, loss='MSE', accum='stack', mode='user_mode', lr=1e-4,
                 n_ratings=5):
        super(GraphAutoencoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
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
        self.out_features = latent_dim / 2
        self.user_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.movie_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )

        # mapping from user embeddings to movies
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

        self.adj_movies = []
        for i in range(n_ratings):
            #
            self.adj_movies.append(torch.sparse_coo_tensor(torch.tensor([indices_i_movies[i], indices_j_movies[i]]),
                                                           torch.ones(size=(len(indices_i_movies[i]),)),
                                                           size=[self.n_items, self.n_users]).coalesce())
        print(f'Movie matrix {self.adj_movies[0].size()}')  # expecting 1000 x 10000

        self.adj_users = []
        for i in range(n_ratings):
            self.adj_users.append(torch.sparse_coo_tensor(torch.tensor([indices_i_users[i], indices_j_users[i]]),
                                                          torch.ones(size=(len(indices_i_users[i]),)),
                                                          size=[self.n_users, self.n_items]).coalesce())
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
        # #
        # for i in range(self.n_ratings):
        #     self.norm_adj[i] = self.norm_adj[i].to_sparse_csr()

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

        self.hidden = 32
        print(self.out_features)
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
        #self.decoder = nn.Bilinear(int(self.out_features), int(self.out_features), int(self.output_size))

    def forward(self, idx):

        if self.user_mode:
            output = []
            weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device)]
            for i in range(self.n_rating):
                # print(weight.size())
                # print(self.weight_matrices[i].size())
                weight.append(self.weight_matrices[i] + weight[-1])
                output.append(self.gcn1[i](self.movie_embeddings.weight, self.norm_adj_users[i], weight[-1]))

            output = self.accum(output) #also add user_embedding?
            output = output[idx]
            # output = torch.relu(output.to_dense())
            output = torch.relu(output)
            # print(output.size())
            output = self.user_dense(output)
        else:
            output = []
            weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device)]

            for i in range(self.n_rating):
                weight.append(self.weight_matrices[i] + weight[-1])
                output.append(self.gcn1[i](self.user_embeddings.weight, self.norm_adj_movies[i], weight[-1]))

            output = self.accum(output)  # also add user_embedding?
            output = output[idx]
            # output = torch.relu(output.to_dense())
            output = torch.relu(output)
            # print(output.size())
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
        ids, rows = train_batch #we receive dense rows
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
