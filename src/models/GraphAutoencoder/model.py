import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd
from utils import normalize_adj
from layer import GraphConvolution

class GraphAutoencoder(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, n_layers, file_path, loss = 'MSE', accum = 'stack', lr = 1e-4, n_ratings = 5):
        super(GraphAutoencoder, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_layers = n_layers
        self.loss = self.loss
        self.n_rating = n_ratings
        self.lr = lr

        self.embeddings = torch.nn.Embedding(
            num_embeddings= self.n, embedding_dim = self.latent_dim
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
        for i, x in df.iterrows():
            name, val = x['Id'], x['Prediction']
            user, movie = name.replace('c', '').replace('r', '').split('_')
            user, movie = int(user) - 1, int(movie) - 1
            val = int(val) -1
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
                                                  size=[self.n, self.n]).to_sparse_csr().coalesce())
        #normalize
        #might use attention mechanism for normalization
        self.norm_adj = [normalize_adj(adj) for adj in self.adj]

        #GCN
        self.gcn1 = [GraphConvolution(in_features=latent_dim, out_features=latent_dim, dropout=0.25) for _ in range(n_ratings)]

        self.activation = nn.ReLU()
        if accum == 'stack':
            self.accum = lambda x: torch.concat(x, dim = 1)
            self.intermediate = self.latent * self.n_rating
        if accum == 'sum':
            self.accum = lambda x: torch.stack(x, dim = 1).sum(dim =1)
            self.intermediate = self.latent

        self.dense = nn.Sequential(
            nn.Linear(self.intermediate, self.latent),
            nn.ReLU()
        )

        self.decoder = nn.Bilinear(self.latent, self.latent, self.n_rating)
        if loss == 'MSE':
            self.loss = nn.MSELoss()
        elif loss == "NLL":
            self.loss = nn.NLLLoss()
        else:
            raise Exception(f"Loss {loss} not implemented")

    def forward(self, x):
        output = []
        for i in self.n_ratings:
            output.append(self.gcn1[i](x, self.norm_adj[i]))

        output = self.accum(output)
        output = self.activation(output)
        output = self.decoder(output)

        output = nn.Softmax(output)
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
        x, y = train_batch #item, ratings
        inp = torch.zeros(size=(self.n, self.latent_dim))
        mask = torch.zeros(size=(self.n ,self.latent_dim))
        for x in range(self.n_ratings):
            crow_indices = self.adj[x].crow_indices()
            col_indices = self.adj[x].crow_indices()
            x_neighbors = col_indices[crow_indices[x], crow_indices[x+1]]
            x_embeddings = self.embeddings(x_neighbors)
            inp[x_neighbors] = x_embeddings
            mask[x_neighbors] = 1

        output = self(inp)
        return self.loss(output * mask, y * mask)

    def validation_step(self, train_batch, batch_num):
        """
        basic validation step
        :param x:
        :param batch_num:
        :return:
        """
        x, y = train_batch  # item, ratings
        inp = torch.zeros(size=(self.n, self.latent_dim))
        mask = torch.zeros(size=(self.n, self.latent_dim))
        for x in range(self.n_ratings):
            crow_indices = self.adj[x].crow_indices()
            col_indices = self.adj[x].crow_indices()
            x_neighbors = col_indices[crow_indices[x], crow_indices[x + 1]]
            x_embeddings = self.embeddings(x_neighbors)
            inp[x_neighbors] = x_embeddings
            mask[x_neighbors] = 1
        output = self(inp)
        return self.loss(output * mask, y * mask)

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