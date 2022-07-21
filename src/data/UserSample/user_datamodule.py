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
    def __init__(self, split_number=0, data_dir: str = "path/to/dir", batch_size: int = 32, validation_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_number = split_number
        #self.validation_split = validation_split
        
    def setup(self, stage: str = None):
        self.data_matrix = torch.Tensor(read_data(self.data_dir + f"/data_train.csv")) # inference on the whole matrix 
        self.train_matrix = torch.Tensor(read_data(self.data_dir + f"/train_split_{self.split_number}.csv"))
        self.val_matrix = torch.Tensor(read_data(self.data_dir + f"/test_split_{self.split_number}.csv"))
        # standardize data 
        self.inference_scaler = StandardScaler()
        self.data_matrix = self.inference_scaler.fit_transform(self.data_matrix)
        self.train_scaler = StandardScaler()
        self.trainmatrix = self.train_scaler.fit_transform(self.train_matrix)
        self.val_scaler = StandardScaler()
        self.val_matrix = self.val_scaler.fit_transform(self.val_matrix)
        # create tensor 
        self.train_tensor = torch.unsqueeze(torch.Tensor(self.train_matrix), dim=1)
        self.val_tensor = torch.unsqueeze(torch.Tensor(self.val_matrix), dim=1)
        self.inference_tensor = torch.unsqueeze(torch.Tensor(self.data_matrix), dim=1)

        assert(self.train_tensor.shape == self.val_tensor.shape)
        print(f"Train matrix shape: {self.data_matrix.shape}")
        print(f"Val matrix shape: {self.val_matrix.shape}")
        #length = len(self.data_matrix)
        #train_fraction = int((1-self.validation_split) * length)
        #val_fraction = length - train_fraction
        #self.dataset = TensorDataset(self.data_matrix)
        #self.user_train, self.user_val = random_split(self.dataset,
         #                   [train_fraction, val_fraction],
          #                  generator=torch.Generator().manual_seed(0))
        # print dataset sizes
        #print(f"train split size {len(self.user_train)}")
        #print(f"val split size {len(self.user_val)}")

    def train_dataloader(self):
        return DataLoader(self.train_tensor, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_tensor, batch_size=self.batch_size, num_workers=6)

    def test_dataloader(self):
        # return val for now 
        return DataLoader(self.val_tensor, batch_size=self.batch_size, num_workers=6)

    def predict_dataloader(self):
        # return val for now 
        return DataLoader(self.val_tensor, batch_size=self.batch_size, num_workers=6)

    def teardown(self, stage: str = None):
        pass 
    