from re import A
import pytorch_lightning as pl
import numpy as np 
import pandas as pd 
from src.utils.utils import read_data 

from torch.utils.data import random_split, DataLoader, TensorDataset
import torch 
from sklearn.preprocessing import StandardScaler


class User_DataModule(pl.LightningDataModule):
    """
    Data module where we sample each user's data from the dataset
    """
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, validation_split: float = 0):
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
        length = len(self.data_matrix)
        train_fraction = int((1-self.validation_split) * length)
        val_fraction = length - train_fraction
        self.dataset = TensorDataset(self.data_matrix)
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
    