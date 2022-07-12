"""
Basic implementation of a Vanilla AE
"""

from abc import ABC
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
import numpy as np


class AutoEncoder(pl.LightningModule, ABC):
    def __init__(
        self, input_channels=1, input_width=1000, input_height=1, hidden_dims=[256, 128, 64], lr=1e-4, **kwargs
    ):
        super().__init__()
        self.input_shape = (input_channels, input_width, input_height)
        self.lr = lr
        # Define Modules
        self.input_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_channels * input_width * input_height, hidden_dims[0]), nn.ReLU()
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
            nn.Linear(hidden_dims[0], input_channels * input_width * input_height)
        )

    def _encode(self, x):
        x = self.input_layer(x)
        for layer in self.encoding_modules:
            x = layer(x)
        return x, x # dummy output for logger instead of mu, logvar 

    def _decode(self, x):
        for layer in self.decoding_modules:
            x = layer(x)
        x = self.output_layer(x)
        bs = x.shape[0]
        x = x.view(bs, *self.input_shape)
        return x 

    def _reparameterize(self, mu, logvar):
        """
        Dummy method for logging purposes 
        """
        return mu 

    def forward(self, x):
        # print(f"input shape {x.shape}")
        x = self.input_layer(x)
        # print(f"after input layer {x.shape}")
        for layer in self.encoding_modules:
            x = layer(x)
            # print(f"x encoding {x.shape}")
        for layer in self.decoding_modules:
            x = layer(x)
            # print(f"x decoding {x.shape}")
        x = self.output_layer(x)
        # print(f"x output {x.shape}")
        bs = x.shape[0]
        x = x.view(bs, *self.input_shape)
        return x # (batch_size, channels, width, height)

    def training_step(self, batch, batch_idx):
        loss, x_hat = self._common_step(batch, batch_idx, "train")
        return {"loss": loss, "x_hat": x_hat}

    def validation_step(self, batch, batch_idx):
        loss, x_hat = self._common_step(batch, batch_idx, "valid")
        return {"val_loss": loss, "x_hat": x_hat}

    def test_step(self, batch, batch_idx):
        loss, x_hat = self._common_step(batch, batch_idx, "test")
        return {"loss": loss, "x_hat": x_hat}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        x_hat = self(x)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _prepare_batch(self, batch):
        """Prepare batch."""
        print(len(batch))
        x = batch[0]
        # print(f"x shape {x.shape}")
        x = x.to(self.device)
        # return x.view(x.size(0), -1)
        return x 

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        x_hat = self(x)
        loss = F.mse_loss(x, x_hat)
        if stage != 'test':
            self.log(f"{stage}_loss", loss, on_step=True)
        return loss, x_hat
