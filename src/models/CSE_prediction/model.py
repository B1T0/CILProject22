import pytorch_lightning as pl
import torch.optim
import torch.nn as nn


class Prediction(pl.LightningModule):
    def __init__(self, model,freeze = False):
        super(Prediction, self).__init__()
        self.embedding_dim = model.embedding_dim
        self.model = model
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        self.lr = 0.0001
        self.output_layer = nn.Sequential(
            nn.Linear(4*self.embedding_dim, 2*self.embedding_dim),
            nn.Linear(2*self.embedding_dim, 2*self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2*self.embedding_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        user_emb = self.model.phi(x)
        user_ngh_emb = self.model.phi_UC(x)

        item_emb = self.model.phi(y)
        item_ngh_emb = self.model.phi_IC(y)

        emb = torch.cat([user_emb, user_ngh_emb, item_emb, item_ngh_emb], dim=1)

        output = self.output_layer(emb)
        return 5*output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        # for i, x in enumerate(train_batch):
        #     train_batch[i] = x.to(cu)
        #print(train_batch)
        x, y, rating = train_batch
        pred = self.forward(x, y)
        pred = pred.squeeze()
        loss = self.loss(pred, rating)
        self.log("Training Loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y, rating = valid_batch
        pred = self.forward(x, y)
        ored = pred.squeeze()
        loss = self.loss(pred, rating)
        self.log("Training Loss", loss)
        return loss

