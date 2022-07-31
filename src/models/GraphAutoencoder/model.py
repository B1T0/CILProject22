import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.layer import ScaledSigmoid
from src.models.utils import normalize_adj, create_full_adjacency_matrix
from src.models.layer import GraphConvolution


"""
Implementation of graph based collaborative filtering:
latent factors of users and items are convolved through the existing rating matrix before being
used for a single rating prediction

we use 5 convolutions overall, one for each rating level
"""

class GraphAutoencoder(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, loss='MSE', accum='stack', lr=1e-4,
                 n_ratings=5):
        """
        :param latent_dim: dimension size of learned embeddings
        :param n_users: # of users in rating matrix
        :param n_items: # of items in rating matrix
        :param file_path: file path to ratings file
        :param loss: loss to be used
        :param accum: accumulation function to be used after convolution
        :param lr: learn rate of optimizer
        :param n_ratings: # of discrete rating categories
        """
        super(GraphAutoencoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
        self.lr = lr
        self.latent_dim = latent_dim
        self.in_features = latent_dim
        self.out_features = latent_dim / 2
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.n, embedding_dim=self.latent_dim
        )

        self.adj = create_full_adjacency_matrix(file_path, n_ratings, n_users, n_items)
        # normalize
        # might use attention mechanism for normalization
        self.norm_adj = [nn.Parameter(normalize_adj(adj), requires_grad=False) for adj in self.adj]
        self.norm_adj = nn.ParameterList(self.norm_adj)

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

        print(self.out_features)
        self.dense = nn.Sequential(
            nn.Linear(int(self.intermediate), int(self.out_features)),
            nn.ReLU()
        )

        self.loss = loss
        if loss == 'MSE':
            self.loss = nn.MSELoss()
            self.output_size = 1
            self.activation = ScaledSigmoid()
            self.target = lambda x: x.float()
            self.evaluate = lambda x: x
        elif loss == "NLL":

            self.loss = nn.NLLLoss()
            self.output_size = self.n_rating
            self.activation = nn.Softmax(dim=1)
            self.target = lambda x: x.long() - 1
            self.ratings_vector = torch.arange(1, self.n_rating + 1)
            self.evaluate = lambda x: torch.matmul(x, self.ratings_vector.cuda().float())
        else:
            raise Exception(f"Loss {loss} not implemented")

        self.mse = nn.MSELoss()
        self.decoder = nn.Bilinear(int(self.out_features), int(self.out_features), int(self.output_size))

    def forward(self, x, items, users):
        """

        :param x: Embeddings
        :param items: (bs, ) item indices
        :param users: (bs, ) user indices
        :param mask:
        :return:
        """
        output = []

        #convolution over n-ratings matrices
        weight = [torch.zeros((int(self.in_features), int(self.out_features))).cuda()]
        for i in range(self.n_rating):
            weight.append(self.weight_matrices[i] + weight[-1])
            output.append(self.gcn1[i](x, self.norm_adj[i], weight[-1]))

        output = self.accum(output)
        output = torch.relu(output)
        item_embeddings, user_embeddings = torch.split(output, [1000, 10000])
        # alternative implementation

        item_embeddings = self.dense(item_embeddings)
        user_embeddings = self.dense(user_embeddings)

        item_embeddings = item_embeddings[items]
        user_embeddings = user_embeddings[users]
        # print(user_embeddings.size())
        output = self.decoder(user_embeddings, item_embeddings)
        output = self.activation(output)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_num):
        """
        basic training step
        :param train_batch:
        :param batch_num:
        :return:
        """
        # x, y = train_batch  # item, ratings
        for i, x in enumerate(train_batch):
            if x is not None:
                train_batch[i] = x.to(self.device)

        movies, users, ratings = train_batch
        # mask = torch.stack([movies, users], dim=1)

        output = self.forward(self.embeddings.weight, movies, users)
        loss = self.loss(output.squeeze(), self.target(ratings))
        with torch.no_grad():
            # print(output[movies, users])
            eval = self.mse(self.evaluate(output.squeeze()), ratings.float())
        return loss, eval

    def validation_step(self, train_batch, batch_num):
        """
        basic validation step
        :param x:
        :param batch_num:
        :return:
        """
        for i, x in enumerate(train_batch):
            if x is not None:
                train_batch[i] = x.to(self.device)
        movies, users, ratings = train_batch
        output = self.forward(self.embeddings.weight, movies, users)
        loss = self.loss(output.squeeze(), self.target(ratings))
        with torch.no_grad():
            eval = self.mse(self.evaluate(output.squeeze()), ratings.float())
        return loss, eval
