import pytorch_lightning as pl
import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class Model(pl.LightningModule):
    def __init__(self, embedding_dim=32):
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.phi = nn.Embedding(11000, embedding_dim)
        self.phi_IC = nn.Embedding(11000, embedding_dim)
        self.phi_UC = nn.Embedding(11000, embedding_dim)
        self.lr = 0.01
        self.loss_mse = nn.MSELoss()
        self.num_samples_neigbors = 3
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.alpha = 0.05
        self.lam = 0.1
        self.lav = 0.025

    def forward(self, x):
        return self.phi(x), self.phi_IC(x), self.phi_UC(x)

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.phi.parameters(), 'weight_decay': self.lav, 'lr': self.lr},
            {'params': self.phi_IC.parameters()},
            {'params': self.phi_UC.parameters()}
        ], lr=self.lr, momentum=0.9)
        #return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        # for i, x in enumerate(train_batch):
        #     train_batch[i] = x.to(cu)
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
        phi_IC_negative = phi_IC_negative.transpose(0, 1)

        phi_negative_x, _, _ = self.forward(negative_x)
        phi_negative_y, _, _ = self.forward(negative_y)

        loss = self.loss(phi_x, phi_y, phi_IC, phi_UC, phi_IC_negative, phi_UC_negative,
                         phi_negative_x, phi_negative_y)

        self.log("Training Loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y, x_neighbors, y_neighbors, x_negative, y_negative, negative_x, negative_y = valid_batch

        x_negative, y_negative = y_negative, x_negative

        phi_x, _, _ = self.forward(x)
        phi_y, _, _ = self.forward(y)
        _, _, phi_UC = self.forward(x_neighbors)
        phi_UC = phi_UC.transpose(0, 1)
        _, _, phi_UC_negative = self.forward(x_negative)
        phi_UC_negative = phi_UC_negative.transpose(0, 1)
        _, phi_IC, _ = self.forward(y_neighbors)
        phi_IC = phi_IC.transpose(0, 1)
        _, phi_IC_negative, _ = self.forward(y_negative)
        # print(phi_IC_negative)
        phi_IC_negative = phi_IC_negative.transpose(0, 1)

        phi_negative_x, _, _ = self.forward(negative_x)
        phi_negative_y, _, _ = self.forward(negative_y)

        loss = self.loss(phi_x, phi_y, phi_IC, phi_UC, phi_IC_negative, phi_UC_negative,
                         phi_negative_x, phi_negative_y)
        self.log("Validation Loss", loss)
        return loss

    # def ranking_loss(self, phi_user, phi_positive, phi_negative, phi_IC, phis_UC, phi_IC_negative,
    #                  phi_UC_negative, phi_negative_x, phi_negative_y):
    #     loss_DS = - torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_user, phi_item), dim=1) -
    #                            torch.sum(torch.mul(phi_user, phi_item_negative), dim=1)))
    #     loss_NS = 0
    #     for i in range(self.num_samples_neighbors):
    #         loss_NS += - torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_item, phis_IC[i]), dim=1))) - torch.mean(
    #             F.logsigmoid(torch.sum(torch.mul(phi_user, phis_UC[i]), dim=1)))
    #
    #     loss_DS += torch.mean(
    #         torch.sum(F.logsigmoid(torch.sum(torch.mul(phi_negative_x, phi_negative_y), dim=2)), dim=1))
    #
    #
    #
    #     for i in range(len(phi_IC_negative)):
    #         loss_NS += torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_item, phi_IC_negative[i]), dim=1)))
    #         loss_NS += torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_user, phi_UC_negative[i]), dim=1)))
    #     # for w in self.phi.weight:
    #     #     norm += torch.norm(w)
    #     loss = self.alpha * (loss_DS + self.lam * loss_NS)  # + reg*norm)
    #     return loss

    def loss(self, phi_user, phi_item,  phis_IC, phis_UC, phi_IC_negative,
             phi_UC_negative, phi_negative_x, phi_negative_y):  # implement the loss function

        #loss is probably log sigmoid
        #should be able to modify direct similarity as needed
        #for the implicit feedback however, it is log-sigmoid
        loss_DS = - torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_user, phi_item),dim=1)))  # + negative sampling
        #negative sampling
        loss_DS += torch.mean(torch.sum(F.logsigmoid(torch.sum(torch.mul(phi_negative_x, phi_negative_y), dim=2)), dim=1))
        #my understanding of rating based completion is effectively MSE
        #loss_DS = self.loss_mse(ratings, torch.mul(phis[0], phis[1]))
        #dimensions may only work for sampling one neighbor
        loss_NS = 0
        for i in range(len(phis_IC)):
            print(phis_IC.size())
            print(phi_item.size())
            loss_NS += - torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_item, phis_IC[i]), dim=1))) - torch.mean(
                F.logsigmoid(torch.sum(torch.mul(phi_user, phis_UC[i]), dim=1)))

        for i in range(len(phi_IC_negative)):
            loss_NS += torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_item, phi_IC_negative[i]), dim=1)))
            loss_NS += torch.mean(F.logsigmoid(torch.sum(torch.mul(phi_user, phi_UC_negative[i]), dim=1)))
        # for w in self.phi.weight:
        #     norm += torch.norm(w)
        loss = self.alpha * (loss_DS + self.lam * loss_NS)  # + reg*norm)
        return loss
