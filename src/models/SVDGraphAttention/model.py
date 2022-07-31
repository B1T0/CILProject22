import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.models.layer import ScaledSigmoid, GraphSelfAttentionLayer
from src.models.utils import create_full_adjacency_matrix

"""
This is an implementation of our proposed improvement upon SVD++ using a Graph Attention Layer
"""


class SVDGraphAttention(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, global_mean, alpha=0.2,
                 loss='MSE', lr=1e-4,
                 n_ratings=5, dropout=0.1, bias=False):
        """

        :param latent_dim: size of latent dimension
        :param n_users: number of users in rating matrix
        :param n_items: number of items in rating matrix
        :param file_path: path ratings file
        :param global_mean: mean rating
        :param alpha: elu factor for graph attention
        :param loss: name of Loss function to be used
        :param lr: learning rate of optimizer
        :param n_ratings: number of discrete rating categories
        :param dropout: node dropout percentage
        :param bias: whether to learn user bias or not
        """
        super(SVDGraphAttention, self).__init__()
        self.gm = global_mean
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
        #self.use_internal_embeddings = use_internal_embeddings
        self.lr = lr
        self.dropout = dropout
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.in_features = latent_dim
        self.out_features = latent_dim

        self.bias = bias

        # Embedding used for Graph Attention
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.n, embedding_dim=self.latent_dim
        )

        self.movie_embeddings = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )

        self.movie_bias = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=1
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

        self.accum = lambda x: torch.concat(x, dim=1)
        self.intermediate = self.out_features * self.n_rating

        self.combine = nn.Sequential(
            nn.Linear(int(self.intermediate), self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
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

    def forward(self, idx):
        """
        the models forward method. Return rating prediction given a tuple of form (movie, user)
        :param idx: tuple (movie_idx, user_idx), movie_idx :(bs, ), user_idx: (bs, )
        :return: ratings (bs, )
        """
        movie_idx, user_idx = idx

        # unique ids needed for masking to work
        user_idx_shifted, inverse_index = torch.unique(user_idx, sorted=False, return_inverse=True)
        user_idx_shifted = user_idx_shifted + self.n_items

        output = []
        weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device)]
        # graph attention convolution
        masks = []
        for i in range(self.n_rating):
            rows = torch.nonzero(self.adjacency_matrix[i][user_idx_shifted])[:, 1]
            weight.append(self.weight_matrices[i] + weight[-1])
            mask = []
            for _, j in enumerate(user_idx_shifted):
                # print(i.size())
                # print(rows.size())
                # also works because of bi partiteness
                mask.append(torch.nonzero(rows == j, as_tuple=True)[0])
            masks.append(torch.concat(mask))
            convoluted = self.gcn1[i](self.embeddings(rows), self.adjacency_matrix[i][rows][:, rows], weight[-1])
            output.append((convoluted[torch.concat(mask)])[inverse_index])

        user_emb_sum = self.accum(output)
        user_emb_sum = self.combine(user_emb_sum)
        # print(user_emb_sum.size())
        # user_emb_sum = user_emb_sum[mask]
        user_emb = self.user_embeddings(user_idx)
        # print(user_emb.size())
        movie_emb = self.movie_embeddings(movie_idx)
        # print(movie_emb.size())
        b_u = self.user_bias(user_idx)
        b_m = self.movie_bias(movie_idx)
        t = torch.sum((user_emb + user_emb_sum) * movie_emb, dim=1)
        if self.bias:
            t += torch.squeeze(b_u) + torch.squeeze(b_m)
        pred_r_ui = t + self.gm

        return pred_r_ui

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        """
        basic training step used for lightning
        :param train_batch: (movie_idx, user_idx, ratings), all of shape (bs, )
        :param batch_num: -
        :return: loss
        """
        for i, x in enumerate(train_batch):
            train_batch[i] = x.to('cuda:0')
        # x, y = train_batch  # item, ratings
        movie_ids, user_ids, ratings = train_batch
        pred = self.forward((movie_ids, user_ids))
        loss = self.loss(pred, ratings.float())
        return loss, 0

    def validation_step(self, train_batch, batch_idx):
        """
        basic validation step
        :param x: (movie_idx, user_idx, ratings), all of shape (bs, )
        :param batch_num: -
        :return: loss
        """
        for i, x in enumerate(train_batch):
            if x is not None:
                train_batch[i] = x.to('cuda:0')
        movie_ids, user_ids, ratings = train_batch
        pred = self.forward((movie_ids, user_ids))
        loss = self.loss(pred, ratings.float())
        return loss, 0
