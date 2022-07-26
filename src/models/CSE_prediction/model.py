import pytorch_lightning as pl
import torch.optim
import torch.nn as nn


class Prediction(pl.LightningModule):
    def __init__(self, model,freeze = False, lr=1e-3):
        super(Prediction, self).__init__()
        self.embedding_dim = model.embedding_dim
        self.model = model
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
        self.lr = lr
        self.output_layer = nn.Sequential(

            #nn.Linear(4*self.embedding_dim, self.embedding_dim),
            nn.Linear(7*self.embedding_dim, 4*self.embedding_dim),
            nn.Linear(4*self.embedding_dim, 4*self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4*self.embedding_dim, 2*self.embedding_dim),
            nn.Linear(2*self.embedding_dim, 2*self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2*self.embedding_dim, 1),

            nn.Sigmoid()
        )
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        user_emb = self.model.phi(x)
        #user_ngh_emb = self.model.phi_UC(x)
        user_IC_emb = self.model.phi_IC(x)

        item_emb = self.model.phi(y)

        #item_ngh_emb = self.model.phi_IC(y)
        item_UC_emb = self.model.phi_UC(y)
        #emb = torch.cat([user_emb, user_ngh_emb, item_emb, item_ngh_emb], dim=1)
        emb = torch.cat([user_emb, user_IC_emb, item_emb, item_UC_emb, torch.mul(user_emb, item_emb),
                         torch.mul(user_emb, item_UC_emb), torch.mul(item_emb, user_IC_emb)], dim=1)
# =======
#         item_ngh_emb = self.model.phi_IC(y)
#
#         emb = torch.cat([user_emb, user_ngh_emb, item_emb, item_ngh_emb,
#                          torch.mul(user_emb, item_emb), torch.mul(user_emb, item_ngh_emb),
#                          torch.mul(item_emb, user_ngh_emb)], dim=1)
# >>>>>>> 3ab1061ff36ed993d4c725ae2f2423e9eccf2d35

        output = self.output_layer(emb)
        return 5*output

    def configure_optimizers(self):
        print("Using mixed learn rates")
        return torch.optim.Adam([{'params': self.model.parameters(), 'lr': 1e-5},
            {'params': self.output_layer.parameters()}], lr=self.lr)


    def training_step(self, train_batch, batch_idx):
        # for i, x in enumerate(train_batch):
        #     train_batch[i] = x.to(cu)
        #print(train_batch)
        x, y, rating = train_batch
        pred = self.forward(x, y)
        pred = pred.squeeze()
        loss = self.loss(pred, rating.float())
        self.log("Training Loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y, rating = valid_batch
        pred = self.forward(x, y)
        pred = pred.squeeze()
        loss = self.loss(pred, rating.float())
        self.log("Training Loss", loss)
        return loss

