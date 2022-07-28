import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
from src.models.GraphAutoencoder.layer import ScaledSigmoid
from src.models.GraphAutoencoder.utils import normalize_adj, create_full_adjacency_matrix
from src.models.GraphAutoencoder.layer import GraphConvolution
import torch.nn.functional as F


class GraphAutoencoder(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, loss='MSE', accum='stack', lr=1e-4,
                 n_ratings=5):
        super(GraphAutoencoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
        self.lr = lr
        self.latent_dim = latent_dim
        self.in_features = latent_dim
        self.out_features = latent_dim /2
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
        self.gcn1 = [GraphConvolution(in_features=self.latent_dim, out_features=self.out_features, dropout=0.25) for _ in
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

    def forward(self, x, items, users, mask=None):
        output = []
        weight = [torch.zeros((int(self.in_features), int(self.out_features))).cuda()]
        for i in range(self.n_rating):
            # print(weight.size())
            # print(self.weight_matrices[i].size())
            weight.append(self.weight_matrices[i] + weight[-1])
            output.append(self.gcn1[i](x, self.norm_adj[i], weight[-1]))

        output = self.accum(output)
        # output = torch.relu(output.to_dense())
        output = torch.relu(output)
        # print(output.size())
        item_embeddings, user_embeddings = torch.split(output, [1000, 10000])
        # alternative implementation

        item_embeddings = self.dense(item_embeddings)
        user_embeddings = self.dense(user_embeddings)
        # print(item_embeddings.size())
        # print(user_embeddings.size())

        # item_mask, user_mask = torch.split(mask, [1000, 10000])
        item_embeddings = item_embeddings[items]
        # print(item_embeddings.size())
        user_embeddings = user_embeddings[users]
        # print(user_embeddings.size())
        output = self.decoder(user_embeddings, item_embeddings)
        # output = item_embeddings.matmul(self.decoder.weight).matmul(user_embeddings.transpose(0, 1))
        # #outut 5 * 1000 * 10000
        # output += self.decoder.bias.unsqueeze(1).unsqueeze(1).repeat(1, 1000, 10000)
        # output = output.permute(1, 2, 0)
        # 1000 * 10000
        # print(output.size())
        # print(f'Predicted Matrix {output.size()}')
        output = self.activation(output)
        # print(output.size())
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
        movies, users, ratings = train_batch
        # mask = torch.stack([movies, users], dim=1)

        # mask = y != 0
        # can rewrite sampling to be more efficient overall --> incclude only embeddings which will be learned
        # x_embeddings = self.embeddings.weight[mask]
        # x_embeddings[x] = self.embeddings(torch.tensor(x)) #use sparser embeddings for more efficiency
        # x_embeddings = x_embeddings.to_sparse_coo()
        # sparse implementation
        # inp = torch.zeros(size=(self.n, self.latent_dim))
        # mask = torch.zeros(size=(self.n ,self.latent_dim))

        # for x in range(self.n_ratings):
        #     crow_indices = self.adj[x].crow_indices()
        #     col_indices = self.adj[x].crow_indices()
        #     x_neighbors = col_indices[crow_indices[x], crow_indices[x+1]]
        #     x_embeddings = self.embeddings(x_neighbors)
        #     inp[x_neighbors] = x_embeddings
        #     mask[x_neighbors] = 1

        output = self.forward(self.embeddings.weight, movies, users)
        # print(ratings)
        # print(self.target(ratings))
        # print(output[movies, users])
        # print(ratings.size())
        # print(output.size())
        loss = self.loss(output.squeeze(), self.target(ratings))
        with torch.no_grad():
            # print(output[movies, users])
            eval = self.mse(self.evaluate(output.squeeze()), ratings.float())
        # print(loss)
        # print(eval)
        return loss, eval

    def validation_step(self, train_batch, batch_num):
        """
        basic validation step
        :param x:
        :param batch_num:
        :return:
        """
        movies, users, ratings = train_batch
        # mask = torch.stack([movies, users], dim=1)

        # mask = y != 0
        # can rewrite sampling to be more efficient overall --> incclude only embeddings which will be learned
        # x_embeddings = self.embeddings.weight[mask]
        # x_embeddings[x] = self.embeddings(torch.tensor(x)) #use sparser embeddings for more efficiency
        # x_embeddings = x_embeddings.to_sparse_coo()
        # sparse implementation
        # inp = torch.zeros(size=(self.n, self.latent_dim))
        # mask = torch.zeros(size=(self.n ,self.latent_dim))

        # for x in range(self.n_ratings):
        #     crow_indices = self.adj[x].crow_indices()
        #     col_indices = self.adj[x].crow_indices()
        #     x_neighbors = col_indices[crow_indices[x], crow_indices[x+1]]
        #     x_embeddings = self.embeddings(x_neighbors)
        #     inp[x_neighbors] = x_embeddings
        #     mask[x_neighbors] = 1

        output = self.forward(self.embeddings.weight, movies, users)
        # print(ratings)
        # print(self.target(ratings))
        # print(output[movies, users])
        # print(ratings.size())
        # print(output.size())
        loss = self.loss(output.squeeze(), self.target(ratings))
        with torch.no_grad():
            eval = self.mse(self.evaluate(output.squeeze()), ratings.float())
        return loss, eval

    # def propagate(self):
    #     users_emb = self.embedding_user.weight
    #     items_emb = self.embedding_item.weight
    #     emb = torch.cat([users_emb, items_emb])
    #     embeddings = [emb]
    #     for layer in range(self.n_layers):
    #         emb = torch.sparse.mm(self.graph, emb)
    #         embeddings.append(emb)
    #     embeddings = torch.stack(embeddings, dim=1)
    #     light_out = torch.mean(embeddings, dim=1)
    #     users, items = torch.split(light_out, [self.n_users, self.n_items])
    #     return users, items
