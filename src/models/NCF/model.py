import numpy as np
import pandas as pd
from subprocess import call
from sklearn.metrics import mean_squared_error
import math
import torch
from torch import dropout, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax
import copy
from time import sleep

class NCF(nn.Module):

    def __init__(self):
        super(NCF, self).__init__()
        self.latent_dim = 64
        self.dropout = 0.4
        self.layer_dim = 512
        print(f'Model-Hyperparameters: "latent_dim": {self.latent_dim}, "dropout": {self.dropout}, "layer_dim": {self.layer_dim}')
        
        self.embedding_user = torch.nn.Embedding(num_embeddings=10000, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=1000, embedding_dim=self.latent_dim)
        self.user_1 = nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                                    torch.nn.Dropout(self.dropout),
                                    nn.Tanh(),
                                    )
        self.item_1 = nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                                    torch.nn.Dropout(self.dropout),
                                    nn.Tanh(),
                                    )
        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features=3*self.latent_dim, out_features=self.latent_dim),
                                    torch.nn.Dropout(self.dropout),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
                                    torch.nn.Dropout(self.dropout),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(in_features=self.latent_dim, out_features=40),
                                    torch.nn.Dropout(self.dropout),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(in_features=40, out_features=5),
                                    )


    def forward(self, user_id, item_id):
        user_embedding = self.embedding_user(user_id.long()-1)
        item_embedding = self.embedding_item(item_id.long()-1)
        x = torch.cat((user_embedding, item_embedding, torch.mul(user_embedding, item_embedding)), dim=1)
        x = self.dense(x)
        return x

def rmse(preds, true):
    # one hot to index
    pred_scalar = torch.zeros(preds.shape[0]).cuda()
    for i, pred in enumerate(preds):
        for j, p in enumerate(pred):
            pred_scalar[i] += p * (j+1)
    return torch.sqrt(torch.mean((pred_scalar-true)**2))

# Prediction
def predict(model, user_id, item_id):
    with torch.no_grad():
        output = model(user_id, item_id)
        return output.item()