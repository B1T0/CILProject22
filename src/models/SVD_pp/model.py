import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F




class SVDPP(pl.LightningModule):

    def __init__(self, num_users, num_items, global_mean, embedding_dim, lr=1e-4):
        super(SVDPP, self).__init__()
        self.gm = global_mean
        self.lr = lr

        # +1 because of padding
        self.P = nn.Embedding(num_users+1, embedding_dim)
        self.Q = nn.Embedding(num_items+1, embedding_dim)
        self.B_U = nn.Embedding(num_users+1, 1)
        self.B_I = nn.Embedding(num_items+1, 1)
        self.Y = nn.Embedding(num_items+1, embedding_dim, padding_idx=0)
        self.loss = nn.MSELoss()

    def forward(self, x):
        # user and item indices start with 1 in dataset,embedding index starts with 1
        user_id, item_id = x[0], x[1]
        rated_items, rated_counts = x[2], x[3]


        p_u = self.P(user_id)
        q_i = self.Q(item_id)
        b_u = self.B_U(user_id)
        b_i = self.B_I(item_id)


        y_j_sum = torch.sum(self.Y(rated_items), dim=1)
        y_j_sum = torch.mul(y_j_sum, torch.unsqueeze(
            torch.div(1, torch.sqrt(rated_counts)), dim=1))

        pred_r_ui = torch.sum((p_u + y_j_sum) * q_i, dim=1) + \
                    torch.squeeze(b_u) + torch.squeeze(b_i) + self.gm

        return pred_r_ui

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.02)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x1 = []
        for i, j in enumerate(x):
            x1.append(j.to('cuda:0'))
        y = y.to('cuda:0')
        pred = self.forward(x1)

        train_loss = self.loss(pred, y.float())
        return train_loss

    def validation_step(self, train_batch, batch_idx):
        x, y = train_batch
        x1 = []
        for i, j in enumerate(x):
            x1.append(j.to('cuda:0'))
        y = y.to('cuda:0')
        pred = self.forward(x1)

        train_loss = self.loss(pred, y.float())
        return train_loss


