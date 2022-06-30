#pytorch lightning implementation based on gusye1234/LightGCN-PyTorch
#the official repo of
#SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020).
# LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

import pytorch_lightning as pl
import torch
import torch.nn as nn

class LightGCN(pl.LightningModule):
    """
    no dropout implemented
    """

    def __init__(self, latent_dim, n_users, n_items, n_layers, loss, graph: torch.sparse_coo):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.loss = self.loss

        self.embedding_user = torch.nn.Embedding(
            num_embeddings = self.n_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings= self.n_items, embedding_dim=self.latent_dim
        )

        self.graph = graph #neighborhood graph
        #as bipartite graph,
        # shape should be |I ,  R |
        #                 |R^T, I | , where
        #                 we assume the first block for users and second block for items

    def reset_embedding(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)


    def propagate(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        emb = torch.cat([users_emb, items_emb])
        embeddings = [emb]
        for layer in range(self.n_layers):
            emb = torch.sparse.mm(self.graph, emb)
            embeddings.append(emb)
        embeddings = torch.stack(embeddings, dim=1)
        light_out = torch.mean(embeddings, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def forward(self, users, items):
        user_embeddings, item_embeddings = self.propagate()
        users_emb = user_embeddings[users]
        items_emb = item_embeddings[items]

        inner_pro = torch.mul(users_emb, items_emb)
        pred = torch.sum(inner_pro, dim=1)
        return pred