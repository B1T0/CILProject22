import torch 
from re import A
import pytorch_lightning as pl
import numpy as np 
import pandas as pd 
from src.utils.utils import read_data 

from torch.utils.data import random_split, DataLoader, TensorDataset
import torch 
from sklearn.preprocessing import StandardScaler



def get_user_dataloaders(data_dir, split_number, batch_size=32):
    """
    return train and validation and test dataloaders 
    """
    data_matrix = torch.Tensor(read_data(data_dir + f"/data_train.csv")) # inference on the whole matrix 
    train_matrix = torch.Tensor(read_data(data_dir + f"/train_split_{split_number}.csv"))
    val_matrix = torch.Tensor(read_data(data_dir + f"/test_split_{split_number}.csv"))
    # standardize data 
    inference_scaler = StandardScaler()
    data_matrix = inference_scaler.fit_transform(data_matrix)
    train_scaler = StandardScaler()
    train_matrix = train_scaler.fit_transform(train_matrix)
    val_scaler = StandardScaler()
    val_matrix = val_scaler.fit_transform(val_matrix)
    # create tensor 
    train_tensor = torch.unsqueeze(torch.Tensor(train_matrix), dim=1)
    val_tensor = torch.unsqueeze(torch.Tensor(val_matrix), dim=1)
    inference_tensor = torch.unsqueeze(torch.Tensor(data_matrix), dim=1)
    # create datasets and dataloaders 
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    #inference_dataset = TensorDataset(inference_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    #inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    return train_dataloader, val_dataloader, inference_tensor, inference_scaler 
