from re import A
import pytorch_lightning as pl
import numpy as np 
import pandas as pd 
from src.utils.utils import read_data 

from torch.utils.data import random_split, DataLoader, TensorDataset, dataset
import torch 
from sklearn.preprocessing import StandardScaler


class CrossDataset(dataset.Dataset):
    """
    Custom dataset to sample concatenated row i and column j over data matrix
    Also return the value for user i, movie j 
    """
    def __init__(self, matrix, transform=None):
        self.matrix = matrix 
        self.shape = matrix.shape
        # count nans in matrix
        self.num_nans = np.isnan(matrix).sum()
        # number of non nans
        self.num_non_nans = self.shape[0] * self.shape[1] - self.num_nans
        # get indices of non-nan values
        self.non_nan_indices = np.where(~np.isnan(matrix))


    def __len__(self):
        return self.num_non_nans

    def __getitem__(self, idx):
        # compute indices of row and column to sample from matrix
        # i = idx // self.shape[1] # row 
        # j = idx % self.shape[1] # column
        # concat i-th row with j-th column
        i, j = self.non_nan_indices[0][idx], self.non_nan_indices[1][idx]
        return {
            'x': torch.unsqueeze(torch.cat((self.matrix[i, :], self.matrix[:, j]), dim=0), dim=0),
            'y': self.matrix[i, j]
        }



class Cross_DataModule(pl.LightningDataModule):
    """
    Our dataset iterates over the whole data matrix, where for entry i,j
    we sample the user i's data (1000) from the dataset concatenated with movie j's ratings (10000x1)
    I.e. we have a combination of the user's data and the movie's ratings.
    We want to predict the user i's rating on movie j 
    """
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, validation_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.data_matrix = torch.Tensor(read_data(self.data_dir))
        # standardize data 
        self.scaler = StandardScaler()
        self.data_matrix = self.scaler.fit_transform(self.data_matrix)
        # create tensor 
        self.data_matrix = torch.Tensor(self.data_matrix)
        print(f"Data matrix shape: {self.data_matrix.shape}")

    def setup(self, stage: str = None):
        self.dataset = CrossDataset(self.data_matrix)
        length = len(self.dataset)
        train_fraction = int((1-self.validation_split) * length)
        val_fraction = length - train_fraction
        self.user_train, self.user_val = random_split(self.dataset,
                            [train_fraction, val_fraction],
                            generator=torch.Generator().manual_seed(0))
        # print dataset sizes
        print(f"train split size {len(self.user_train)}")
        print(f"val split size {len(self.user_val)}")

    def train_dataloader(self):
        return DataLoader(self.user_train, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.user_val, batch_size=self.batch_size, num_workers=6)

    def test_dataloader(self):
        # return val for now 
        return DataLoader(self.user_val, batch_size=self.batch_size, num_workers=6)

    def predict_dataloader(self):
        # return val for now 
        return DataLoader(self.user_val, batch_size=self.batch_size, num_workers=6)

    def teardown(self, stage: str = None):
        pass 
    