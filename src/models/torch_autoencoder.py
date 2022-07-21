"""
Basic implementation of a Vanilla AE
"""

from abc import ABC
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from src.utils.losses import MaskedMSELoss
from src.utils.utils import get_activation_function
import numpy as np


class AutoEncoder(torch.nn.Module):
    def __init__(
        self, input_channels=1, input_width=1000, input_height=1, hidden_dims=[512, 256, 128], lr=1e-4, activation='relu', **kwargs
    ):
        super().__init__()
        self.input_shape = input_width
        self.lr = lr
        self.activation_fn = get_activation_function(activation)
        self.loss = MaskedMSELoss()

        # Define Modules
        self.input_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_channels * input_width * input_height, hidden_dims[0]), self.activation_fn
        )
        self.encoding_modules = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), self.activation_fn)
            for i in np.arange(0, len(hidden_dims) - 1)
        )
        self.decoding_modules = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i - 1]), self.activation_fn)
            for i in np.arange(len(hidden_dims) - 1, 0, -1)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], input_channels * input_width * input_height)
        )
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, debug=False, dropout=True):
        if debug: print(f"input shape {x.shape}")
        x = self.input_layer(x)
        if debug: print(f"after input layer {x.shape}")
        for layer in self.encoding_modules:
            x = layer(x)
            if dropout: x = self.dropout(x)
            if debug: print(f"x encoding {x.shape}")
        for layer in self.decoding_modules:
            x = layer(x)
            if dropout: x = self.dropout(x)
            if debug: print(f"x decoding {x.shape}")
        x = self.output_layer(x)
        if debug: print(f"x output {x.shape}")
        bs = x.shape[0]
        x = x.view(bs, self.input_shape)
        return x # (batch_size, channels, width, height)