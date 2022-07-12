from re import A
import pytorch_lightning as pl
import numpy as np 
import pandas as pd 
from torch.utils.data import random_split, DataLoader



def read_data(train_path, number_of_users=10000, number_of_movies=1000, impute_value=np.nan):
    data_pd = pd.read_csv(train_path) 
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # Create data matrix
    data = np.full((number_of_users , number_of_movies), impute_value)
    for user, movie ,pred in zip(users, movies, predictions): 
        data[user][movie] = pred
    return data 


class User_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        self.data_matrix = read_data(self.data_dir)
        length = len(self.data_matrix)
        train_fraction = 0.8 * length 
        val_fraction = length - train_fraction
        self.user_train, self.user_val = random_split(self.data_matrix, [train_fraction, val_fraction])

    def train_dataloader(self):
        return DataLoader(self.user_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.user_val, batch_size=self.batch_size)

    def test_dataloader(self):
        # return val for now 
        return DataLoader(self.user_val, batch_size=self.batch_size)

    def predict_dataloader(self):
        # return val for now 
        return DataLoader(self.user_val, batch_size=self.batch_size)

    def teardown(self, stage: str = None):
        pass 
    