"""
Basic implementation of a Vanilla AE
"""

from abc import ABC
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from src.utils.losses import MaskedMSELoss
import numpy as np


class AutoEncoder(pl.LightningModule, ABC):
    def __init__(
        self,input_width=1000, hidden_dims=[512, 256, 128], lr=1e-4, weight_decay=1e-5, dropout=0.0, **kwargs
    ):
        super().__init__()
        self.input_shape = input_width
        self.lr = lr
        self.weight_decay = weight_decay 
        self.dropout = nn.Dropout(dropout)
        self.loss = MaskedMSELoss

        # Define Modules
        self.input_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_width, hidden_dims[0]), nn.ReLU()
        )
        self.encoding_modules = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU())
            for i in np.arange(0, len(hidden_dims) - 1)
        )
        self.decoding_modules = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i - 1]), nn.ReLU())
            for i in np.arange(len(hidden_dims) - 1, 0, -1)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], input_width),
         )
        self.dropout = nn.Dropout(0.25)

    def _encode(self, x, debug):
        x = self.input_layer(x)
        if debug: print(f"after input layer {x.shape}")
        for layer in self.encoding_modules:
            x = layer(x)
            x = self.dropout(x)
            if debug: print(f"x encoding {x.shape}")
        return x

    def _decode(self, x, debug):
        for layer in self.decoding_modules:
            x = layer(x)
            self.dropout(x)
            if debug: print(f"x decoding {x.shape}")
        x = self.output_layer(x)
        return x 

    def forward(self, x, debug=False):
        if debug: print(f"input shape {x.shape}")
        x = self.dropout(x)
        x = self._encode(x, debug)
        x = self._decode(x, debug)
        if debug: print(f"x output {x.shape}")
        bs = x.shape[0]
        x = x.view(bs, self.input_shape)
        return x # (batch_size, channels, width, height)

    def training_step(self, batch, batch_idx):
        loss, x_hat = self._common_step(batch, batch_idx, "train")
        #print(f"train loss {loss}")
        return {"loss": loss, "x_hat": x_hat}

    def validation_step(self, batch, batch_idx):
        loss, x_hat = self._common_step(batch, batch_idx, "valid")
        #print(f"validation loss {loss}")
        return {"valid_loss": loss, "x_hat": x_hat}

    def test_step(self, batch, batch_idx):
        loss, x_hat = self._common_step(batch, batch_idx, "test")
        return {"loss": loss, "x_hat": x_hat}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        x_hat = self(x)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _prepare_batch(self, batch):
        """Prepare batch."""
        x, nan_mask = batch[0], batch[1]
        #print(f"x {x[:5]}")
        #print(f"nan_mask {nan_mask[:5]}")
        # print(f"x shape {x.shape}")
        x = x.cuda()
        nan_mask = nan_mask.cuda()
        # compute the non-nan mask 
        #nan_mask = torch.isnan(x)
        # makenans in x to 0 
        x[~nan_mask] = 0
        # return x.view(x.size(0), -1)
        return x, nan_mask

    def _common_step(self, batch, batch_idx, stage: str):
        x, nan_mask = self._prepare_batch(batch)
        x_hat = self(x)
        loss = self.loss(x_hat, x, nan_mask)
        if stage != 'test':
            self.log(f"{stage}_loss", loss, on_step=True)
        return loss, x_hat
