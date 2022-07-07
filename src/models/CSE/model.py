import pytorch_lightning as pl
import torch.optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class model(pl.LightningModule):
    def __init__(self):
        super(model, self).__init__()
        embedding_dim = 10
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
        x, ratings, negative_samples = train_batch

        phi, phi_IC, phi_UC = self.forward(x)
        loss = self.loss(phi, phi_IC, phi_UC, ratings)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        phi, phi_IC, phi_U = self.forward(x)
        loss = self.loss(phi, phi_IC, phi_U, y)

    def loss(self, phis, phis_IS, phis_UC, ratings):  # implement the loss function
        alpha = 0.1
        lam = 0.1
        reg = 1.0
        phis = phis.transpose(0, 1)
        phis_IC = phis_IS.transpose(0, 1)
        phis_UC = phis_UC.transpose(0, 1)
        #loss is probably log sigmoid
        #should be able to modify direct similarity as needed
        #for the implicit feedback however, it is log-sigmoid
        loss_DS = - torch.mean(F.logsigmoid(torch.mul(phis[0], phis[1])))  # + negative sampling
        #my understanding of rating based completion is effectively MSE
        #loss_DS = self.loss_mse(ratings, torch.mul(phis[0], phis[1]))
        loss_NS = - torch.mean(F.logsigmoid(torch.mul(phis[0], phis_IC[1]))) - torch.mean(
            F.logsigmoid(torch.mul(phis[0], phis_UC[0])))
        # norm =    only for overfitting
        loss = alpha * (loss_DS + lam * loss_NS)  # + reg*norm)
        return loss
