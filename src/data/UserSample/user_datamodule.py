from re import A
import pytorch_lightning as pl
import numpy as np 
import pandas as pd 
from src.utils.utils import read_data 

from torch.utils.data import DataLoader, TensorDataset
import torch 
from sklearn.preprocessing import StandardScaler


class User_DataModule(pl.LightningDataModule):
    """
    Data module where we sample each user's data from the dataset
    """
    def __init__(self, split_number=0, data_dir: str = "path/to/dir", batch_size: int = 32, item_based: bool = False, scale: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_number = split_number
        self.item_based = item_based
        self.scale = scale 

    def setup(self, stage: str = None):
        self.data_matrix = torch.Tensor(read_data(self.data_dir + f"/data_train.csv")) # inference on the whole matrix 
        self.train_matrix = torch.Tensor(read_data(self.data_dir + f"/train_split_{self.split_number}.csv"))
        self.val_matrix = torch.Tensor(read_data(self.data_dir + f"/test_split_{self.split_number}.csv"))

        # transpose if item-based encoding 
        if self.item_based:
            self.train_tensor = self.train_tensor.t()
            self.val_tensor = self.val_tensor.t()
            self.inference_tensor = self.inference_tensor.t()

        # standardize data 
        if self.scale: 
            self.inference_scaler = StandardScaler()
            self.data_matrix = self.inference_scaler.fit_transform(self.data_matrix)
            # self.train_scaler = StandardScaler()
            # self.trainmatrix = self.train_scaler.fit_transform(self.train_matrix)
            # self.val_scaler = StandardScaler()
            # self.val_matrix = self.val_scaler.fit_transform(self.val_matrix)

        # create tensors
        self.train_tensor = torch.Tensor(self.train_matrix)
        self.val_tensor = torch.Tensor(self.val_matrix)
        self.inference_tensor = torch.Tensor(self.data_matrix)
        # create non-nan masks
        self.train_mask = ~torch.isnan(self.train_tensor)
        self.val_mask = ~torch.isnan(self.val_tensor)
        self.inference_mask = ~torch.isnan(self.inference_tensor)

        # create datasets from tensors 
        self.train_dataset = TensorDataset(self.inference_tensor, self.train_mask)
        self.val_dataset = TensorDataset(self.inference_tensor, self.val_mask)
        #self.inference_dataset = TensorDataset(self.inference_tensor, self.inference_mask)
        
        print(f"Train matrix shape: {self.data_matrix.shape}")
        print(f"Val matrix shape: {self.val_matrix.shape}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6)

    def test_dataloader(self):
        # return val for now 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6)

    def predict_dataloader(self):
        # return val for now 
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6)

    def teardown(self, stage: str = None):
        pass 
    