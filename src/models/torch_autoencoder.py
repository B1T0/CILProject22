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
        self, input_width=10000, hidden_dims=[512, 256, 128], activation='relu', **kwargs):
        super().__init__()
        self.input_shape = input_width
        self.activation_fn = get_activation_function(activation)
        self.loss = MaskedMSELoss()

        # Define Modules
        self.input_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_width, hidden_dims[0]), self.activation_fn
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
            nn.Linear(hidden_dims[0], input_width),
        )
        self.dropout = nn.Dropout(0.25)

    def _encode(self, x, debug=False, dropout=True):
        if debug: print(f"input shape {x.shape}")
        x = self.input_layer(x)
        if debug: print(f"after input layer {x.shape}")
        for layer in self.encoding_modules:
            x = layer(x)
            if dropout: x = self.dropout(x)
            if debug: print(f"x encoding {x.shape}")
        return x

    def _decode(self, x, debug=False, dropout=True):
        if debug: print(f"input shape {x.shape}")
        for layer in self.decoding_modules:
            x = layer(x)
            if dropout: x = self.dropout(x)
            if debug: print(f"x decoding {x.shape}")
        x = self.output_layer(x)
        if debug: print(f"x output {x.shape}")
        bs = x.shape[0]
        x = x.view(bs, self.input_shape)
        return x

    def forward(self, x, debug=False, dropout=True):
        encoded = self._encode(x, debug, dropout)
        decoded = self._decode(encoded, debug, dropout)
        return decoded # (batch_size, channels, width, height)
