import pytorch_lightning as pl
import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class model(pl.LightningModule):
    def __init__(self, embedding_dim=50):
        super(model, self).__init__()
        self.embedding_dim = embedding_dim
        self.phi = nn.Embedding(11000, embedding_dim)
        self.phi_IC = nn.Embedding(11000, embedding_dim)
        self.phi_UC = nn.Embedding(11000, embedding_dim)
        self.lr = 0.01
        self.loss_mse = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        self.phi(x)
        self.phi_IC(x)
        self.phi_UC(x)
        return self.phi(x), self.phi_IC(x), self.phi_UC(x)

    def configure_optimizers(self):
        return torch.optim.SparseAdam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y, x_neighbors, y_neighbors, x_negative, y_negative, negative_x, negative_y = train_batch

        # if x >= 1000:
        #     x, y, = y, x
        #     x_neighbors, y_neighbors = y_neighbors, x_neighbors
        #     x_negative, y_negative = y_negative, x_negative

        phi_x, _, _ = self.forward(x)
        phi_y, _,  _ = self.forward(y)
        _, _, phi_UC = self.forward(x_neighbors)
        _, _, phi_UC_negative = self.forward(x_negative)
        phi_UC_negative = phi_UC_negative.transpose(0, 1)
        _, phi_IC, _ = self.forward(y_neighbors)

        _, phi_IC_negative, _ = self.forward(y_negative)
        #print(phi_IC_negative)
        phi_IC_negative = phi_IC_negative.transpose(0, 1)

        phi_negative_x, _, _ = self.forward(negative_x)
        phi_negative_y, _, _ = self.forward(negative_y)

        loss = self.loss(phi_x, phi_y, phi_IC, phi_UC, phi_IC_negative, phi_UC_negative,
                         phi_negative_x, phi_negative_y)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y, x_neighbors, y_neighbors, x_negative, y_negative, negative_x, negative_y = valid_batch

        # if x >= 1000:
        #     x, y, = y, x
        #     x_neighbors, y_neighbors = y_neighbors, x_neighbors
        #     x_negative, y_negative = y_negative, x_negative

        phi_x, _, _ = self.forward(x)
        phi_y, _, _ = self.forward(y)
        _, _, phi_UC = self.forward(x_neighbors)
        _, _, phi_UC_negative = self.forward(x_negative)
        phi_UC_negative = phi_UC_negative.transpose(0, 1)
        _, phi_IC, _ = self.forward(y_neighbors)

        _, phi_IC_negative, _ = self.forward(y_negative)
        # print(phi_IC_negative)
        phi_IC_negative = phi_IC_negative.transpose(0, 1)

        phi_negative_x, _, _ = self.forward(negative_x)
        phi_negative_y, _, _ = self.forward(negative_y)

        loss = self.loss(phi_x, phi_y, phi_IC, phi_UC, phi_IC_negative, phi_UC_negative,
                         phi_negative_x, phi_negative_y)
        return loss

    def loss(self, phi_user, phi_item,  phis_IC, phis_UC, phi_IC_negative,
             phi_UC_negative, phi_negative_x, phi_negative_y):  # implement the loss function
        alpha = 0.1
        lam = 0.1

        reg = 1.0
        #loss is probably log sigmoid
        #should be able to modify direct similarity as needed
        #for the implicit feedback however, it is log-sigmoid
        loss_DS = - torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_user, phi_item),dim=1)))  # + negative sampling
        #negative sampling
        loss_DS += torch.mean(torch.sum(F.logsigmoid(torch.sum(torch.mul(phi_negative_x, phi_negative_y), dim=2)), dim=1))
        #my understanding of rating based completion is effectively MSE
        #loss_DS = self.loss_mse(ratings, torch.mul(phis[0], phis[1]))
        #dimensions may only work for sampling one neighbor
        loss_NS = - torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_item, phis_IC), dim=1))) - torch.mean(
            F.logsigmoid(torch.sum(torch.mul(phi_user, phis_UC), dim=1)))
        for i in range(len(phi_IC_negative)):
            loss_NS += torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_item, phi_IC_negative[i]), dim=1)))
            loss_NS += torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_user, phi_UC_negative[i]), dim=1)))
        # norm =    only for overfitting
        loss = alpha * (loss_DS + lam * loss_NS)  # + reg*norm)
        return loss
