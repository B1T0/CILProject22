import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
from src.models.GraphAutoencoder.layer import ScaledSigmoid
from src.models.GraphAutoencoder.utils import normalize_adj
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
        self.embeddings = torch.nn.Embedding(
            num_embeddings=self.n, embedding_dim=self.latent_dim
        )
        # self.embedding_user = torch.nn.Embedding(
        #     num_embeddings = self.n_users, embedding_dim=self.latent_dim
        # )
        # self.embedding_item = torch.nn.Embedding(
        #     num_embeddings= self.n_items, embedding_dim=self.latent_dim
        # )

        indices_i = [[] for _ in range(n_ratings)]
        indices_j = [[] for _ in range(n_ratings)]

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
            indices_i[val].append(movie)
            indices_j[val].append(user + n_items)
            #
            indices_i[val].append(user + n_items)
            indices_j[val].append(movie)

        self.adj = []
        for i in range(n_ratings):
            self.adj.append(torch.sparse_coo_tensor(torch.tensor([indices_i[i], indices_j[i]]),
                                                    torch.ones(size=(len(indices_i[i]),)),
                                                    size=[self.n, self.n]).coalesce())
        # normalize
        # might use attention mechanism for normalization
        self.norm_adj = [normalize_adj(adj) for adj in self.adj]

        # #
        # for i in range(self.n_ratings):
        #     self.norm_adj[i] = self.norm_adj[i].to_sparse_csr()

        # GCN
        self.gcn1 = [GraphConvolution(in_features=latent_dim, out_features=latent_dim, dropout=0.25) for _ in
                     range(n_ratings)]

        self.activation = nn.ReLU()
        if accum == 'stack':
            self.accum = lambda x: torch.concat(x, dim=1)
            self.intermediate = self.latent_dim * self.n_rating
        if accum == 'sum':
            self.accum = lambda x: torch.stack(x, dim=1).sum(dim=1)
            self.intermediate = self.latent_dim

        self.dense = nn.Sequential(
            nn.Linear(self.intermediate, self.latent_dim),
            nn.ReLU()
        )

        self.loss = loss
        if loss == 'MSE':
            self.loss = nn.MSELoss()
            self.output_size = 1
            self.activation = ScaledSigmoid()
            self.target = lambda x: x
            self.eval = lambda  x: x
        elif loss == "NLL":

            self.loss = nn.NLLLoss()
            self.output_size = self.n_rating
            self.activation = nn.Softmax(dim=2)
            self.target = lambda x: x-1
            self.eval = lambda x: torch.matmul(x, torch.arange(1, self.n_rating+1).float())
        else:
            raise Exception(f"Loss {loss} not implemented")

        self.mse = nn.MSELoss()
        self.decoder = nn.Bilinear(self.latent_dim, self.latent_dim, self.output_size)

    def forward(self, x, mask = None):
        output = []
        for i in range(self.n_rating):
            output.append(self.gcn1[i](x, self.norm_adj[i]))

        output = self.accum(output)
        output = torch.relu(output.to_dense())
        print(output.size())
        item_embeddings, user_embeddings = torch.split(output, [1000, 10000])
        #alternative implementation

        item_embeddings = self.dense(item_embeddings)
        user_embeddings = self.dense(user_embeddings)
        print(item_embeddings.size())
        print(user_embeddings.size())

        # item_mask, user_mask = torch.split(mask, [1000, 10000])
        # item_embeddings = item_embeddings[mask]
        # user_embeddings = user_embeddings[mask]
        output = item_embeddings.matmul(self.decoder.weight).matmul(user_embeddings.transpose(0, 1))
        #outut 5 * 1000 * 10000
        output += self.decoder.bias.unsqueeze(1).unsqueeze(1).repeat(1, 1000, 10000)
        output = output.permute(1, 2, 0)
        #1000 * 10000
        print(output.size())
        #print(f'Predicted Matrix {output.size()}')
        output = self.activation(output)
        print(output.size())
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
        #x, y = train_batch  # item, ratings
        movies, users, ratings = train_batch
        #mask = torch.stack([movies, users], dim=1)

        #mask = y != 0
        #can rewrite sampling to be more efficient overall --> incclude only embeddings which will be learned
        # x_embeddings = self.embeddings.weight[mask]
        # x_embeddings[x] = self.embeddings(torch.tensor(x)) #use sparser embeddings for more efficiency
        #x_embeddings = x_embeddings.to_sparse_coo()
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

        output = self(self.embeddings.weight)
        print(ratings)
        print(self.target(ratings))
        print(output[movies, users])
        print(ratings.size())
        #print(output.size())
        loss = self.loss(output[movies, users], self.target(ratings))
        with torch.no_grad():
            eval = self.mse(self.eval(output[movies, users]), ratings)

        return loss, eval
    def validation_step(self, train_batch, batch_num):
        """
        basic validation step
        :param x:
        :param batch_num:
        :return:
        """
        movies, users, ratings = train_batch  # item, ratings

        # mask = y != 0
        # x_embeddings = self.embeddings.weight[mask]
        # x_embeddings[x] = self.embeddings(x)
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

        output = self(self.embeddings.weight)
        return self.loss(output[movies, users], ratings)

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
