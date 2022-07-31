import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.layer import ScaledSigmoid, GraphSelfAttentionLayer
from src.models.utils import create_full_adjacency_matrix


"""
Graph User Encoder Implementation with additional Attention used during the convolution

forward function has been optimised to make large rating matrix usable for batched convolution
"""

class GraphAttention(pl.LightningModule):

    def __init__(self, latent_dim, n_users, n_items, file_path, graph_hidden=32, hidden=32, alpha=0.2, loss='MSE',
                 accum='stack', mode='user_mode', lr=1e-4,
                 n_ratings=5, dropout=0.1):
        """

        :param latent_dim: # of dimensions of learned embeddings
        :param n_users: # of users
        :param n_items: # of items
        :param file_path: file path to rating matrix
        :param graph_hidden: # number of dimensions used for attention operation
        :param hidden: # of dimensions used for linear decoder
        :param alpha: elu-factor for attention layer
        :param loss: name of loss to be used
        :param accum: accumulation operation to be used
        :param mode: whether to encode user or item-wise
        :param lr: learn rate of optimiser
        :param n_ratings: # of discrete rating categories
        :param dropout: node dropout
        """
        super(GraphAttention, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.n_rating = n_ratings
        self.lr = lr
        self.dropout = dropout
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

        #Adjacency matrices used for forwarding
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
        if accum == 'stack':
            self.accum = lambda x: torch.concat(x, dim=1)
            self.intermediate = self.out_features * self.n_rating
        if accum == 'sum':
            self.accum = lambda x: torch.stack(x, dim=1).sum(dim=1)
            self.intermediate = self.out_features

        self.hidden = hidden

        #final decoder to user/item row
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
        """
        :param idx: single idx of either user or item, depending on mode
        :return: row of rating predictions
        """
        output = []
        weight = [torch.zeros((int(self.in_features), int(self.out_features)), device=self.device)]
        for i in range(self.n_rating):
            # print(weight.size())
            # print(self.weight_matrices[i].size())
            rows = torch.nonzero(self.adjacency_matrix[i][idx])[:, 1]
            weight.append(self.weight_matrices[i] + weight[-1])

            mask = []
            for _, j in enumerate(idx):
                # print(i.size())
                # print(rows.size())
                # also works because of bi partiteness
                mask.append(torch.nonzero((rows == j), as_tuple=True)[0])
            mask = torch.concat(mask)

            convoluted = self.gcn1[i](self.embeddings(rows), self.adjacency_matrix[i][rows][:, rows], weight[-1])
            output.append(convoluted[mask])

        output = self.accum(output)
        #print(output.size())
        if self.user_mode:
            #output = output[mask]
            output = torch.relu(output)
            output = self.user_dense(output)
        else:
            output = torch.relu(output)
            output = self.movie_dense(output)
        return output

    def configure_optimizers(self):

        return torch.optim.AdamW([
            {'params': self.embeddings.parameters(), 'weight_decay': 0.002, 'lr': 5e-5},
            {'params': self.weight_matrices.parameters()},
            {'params': self.gcn1.parameters()},
            {'params': self.user_dense.parameters()},
            {'params': self.movie_dense.parameters()},
        ], lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        """
        basic training step
        :param train_batch:
        :param batch_num:
        :return:
        """
        for i, x in enumerate(train_batch):
            if x is not None:
                train_batch[i] = x.to('cuda:0')
        # x, y = train_batch  # item, ratings
        ids, rows = train_batch  # we receive dense rows
        mask = rows != 0
        if self.user_mode:
            ids += self.n_items
        pred = self.forward(ids)

        loss = self.loss(rows[mask].float(), pred[mask])
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
        ids, rows = train_batch  # we receive dense rows
        mask = rows != 0
        if self.user_mode:
            ids += self.n_items
        pred = self.forward(ids)
        loss = self.loss(rows[mask], pred[mask].float())
        return loss, 0
