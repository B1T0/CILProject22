import torch 
from re import A
import pytorch_lightning as pl
import numpy as np 
import pandas as pd 
from src.utils.utils import read_data 

from torch.utils.data import random_split, DataLoader, TensorDataset
import torch 
from sklearn.preprocessing import StandardScaler



def get_user_dataloaders(data_dir, split_number, batch_size=32, item_based=False):
    """
    return train and validation and test dataloaders 
    """
    data_matrix = torch.Tensor(read_data(data_dir + f"/data_train.csv")) # inference on the whole matrix 
    train_matrix = torch.Tensor(read_data(data_dir + f"/train_split_{split_number}.csv"))
    val_matrix = torch.Tensor(read_data(data_dir + f"/test_split_{split_number}.csv"))
    # compute the non-nan mask 
    data_matrix = torch.nan_to_num(data_matrix, nan=0)
    train_matrix_nan_mask = ~torch.isnan(train_matrix)
    train_matrix = torch.nan_to_num(train_matrix, nan=0)
    val_matrix_nan_mask = ~torch.isnan(val_matrix)
    val_matrix = torch.nan_to_num(val_matrix, nan=0)
    # print fraction of nans
    print(f"Sparsity in train: {torch.sum(train_matrix_nan_mask).item() / (train_matrix_nan_mask.shape[0] * train_matrix_nan_mask.shape[1])}")
    print(f"Sparsity in val: {torch.sum(val_matrix_nan_mask).item() / (val_matrix_nan_mask.shape[0] * val_matrix_nan_mask.shape[1])}")
    # standardize data 
    inference_scaler = StandardScaler()
    data_matrix = inference_scaler.fit_transform(data_matrix)
    train_scaler = StandardScaler()
    train_matrix = train_scaler.fit_transform(train_matrix)
    val_scaler = StandardScaler()
    val_matrix = val_scaler.fit_transform(val_matrix)
    # if item-based, transform matrices 
    if item_based:
        data_matrix = data_matrix.t()
        train_matrix = train_matrix.t()
        val_matrix = val_matrix.t()
        train_matrix_nan_mask = train_matrix_nan_mask.t()
        val_matrix_nan_mask = val_matrix_nan_mask.t()
    # print shapes 
    print(f"data_matrix shape: {data_matrix.shape}")
    print(f"train_matrix shape: {train_matrix.shape}")
    print(f"val_matrix shape: {val_matrix.shape}")
    
    # create tensor 
    train_tensor = torch.unsqueeze(torch.Tensor(train_matrix), dim=1)
    val_tensor = torch.unsqueeze(torch.Tensor(val_matrix), dim=1)
    inference_tensor = torch.unsqueeze(torch.Tensor(data_matrix), dim=1)
    # create datasets and dataloaders 
    train_dataset = TensorDataset(train_tensor, train_matrix_nan_mask)
    val_dataset = TensorDataset(val_tensor, val_matrix_nan_mask)
    #inference_dataset = TensorDataset(inference_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    # print sizes of train and validation dataloaders 
    print(f"train_dataloader size: {len(train_dataloader)}")
    print(f"val_dataloader size: {len(val_dataloader)}")
    #inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    return train_dataloader, val_dataloader, inference_tensor, inference_scaler 
