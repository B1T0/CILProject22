"""
Create dataloaders depending on settings in config.py
"""

from datetime import datetime
import random
import os
import json 

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch 
import pandas as pd
import torch 
from src.models.vae import VAE
from tqdm import tqdm 
import numpy as np 


from config import config
from src.data.datasets import PadChestDataset, ISICDataset
from src.utils.utils import get_data_path


def get_autoencoder_dataloaders():
    if config["mode"] == "train":
        label_fname = "training_subset_labels.csv"
    elif config["mode"] == "dev":
        label_fname = "development_subset_labels.csv"
    else:
        raise ValueError(f"Invalid mode in config.py: {config['mode']}")

    if config['dataset'] == 'padchest':
        Dataset = PadChestDataset
    elif config['dataset'] == 'isic':
        Dataset = ISICDataset
    else:
        raise ValueError(f"Invalid dataset in config.py: {config['dataset']}")

    df = pd.read_csv(get_data_path() / "processed" / label_fname)

    df = df[df.apply(
        lambda x: os.path.exists(
            get_data_path() / "processed" / "images" / "224" / x.ImageID),
        axis=1
    )]

    # Split the test set with a fixed seed such that it is fixed
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    # Split train and validation randomly
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=random.seed(datetime.now())
    )
    print(f"len train {len(train_df)}, len val {len(val_df)}, len test {len(test_df)}")

    # Create the datasets
    train_dataset = Dataset(train_df, size=(config["input_width"], config["input_height"]))
    val_dataset = Dataset(val_df, size=(config["input_width"], config["input_height"]))
    test_dataset = Dataset(test_df, size=(config["input_width"], config["input_height"]))
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    validation_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    return train_dataloader, validation_dataloader, test_dataloader


def get_classifier_dataloaders(encoder_dir:str):
    print(f"Loading encoder from {encoder_dir}")
    # check if encoded tensors already exist in encoder_dir
    if not os.path.exists(f"{encoder_dir}/X_encoded.npz") or not os.path.exists(f"{encoder_dir}/y.npz"):
        print(f"Encoding datasets")
        X, y = encode_dataset(encoder_dir)
        np.savez(f"{encoder_dir}/X_encoded.npz", X_train=X['train'], X_val=X['val'], X_test=X['test'])
        np.savez(f"{encoder_dir}/y.npz", y_train=y['train'], y_val=y['val'], y_test=y['test'])
        print(f"Saving encoded datasets as {encoder_dir}/X_encoded.npz and {encoder_dir}/y.npz")
        X_train, X_val, X_test = X["train"], X["val"], X["test"]
        y_train, y_val, y_test = y["train"], y["val"], y["test"]
        print(f"Saved dataset lenghts: {len(X_train)}, {len(X_val)}, {len(X_test)}")
    else:
        print(f"Loading encoded datasets")
        X = np.load(f"{encoder_dir}/X_encoded.npz")
        y = np.load(f"{encoder_dir}/y.npz")
        X_train, X_val, X_test = X["X_train"], X["X_val"], X["X_test"]
        y_train, y_val, y_test = y["y_train"], y["y_val"], y["y_test"]
        print(f"Loaded dataset lengths: {len(X_train)}, {len(X_val)}, {len(X_test)}")
    
    # create datasets for train, val and test
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    # create dataloaders for train, val and test
    train_dataloader = DataLoader(train_dataset, batch_size=config["classifier_batch_size"], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config["classifier_batch_size"], shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=config["classifier_batch_size"], shuffle=False, num_workers=4)
    print(f"Created datasets with {len(train_dataset)} train, {len(val_dataset)} val and {len(test_dataset)} test samples")
    return train_dataloader, val_dataloader, test_dataloader
    


def encode_dataset(model_dir:str):
    """
    Loads the model from model_dir, encodes the padchest datasets and returns them as np.arrays 
    """
    (
        train_dataloader,
        validation_dataloader,
        test_dataloader,
    ) = get_padchest_dataloaders()

    # load hyperparameters from json file
    with open(f"{model_dir}/hyperparameters.json", "r") as f:
        params = json.load(f)
    # find checkpoint file with highest epoch number 
    checkpoint_files = [    f for f in os.listdir(model_dir) if f.endswith(".ckpt")]
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("=")[1].split("-")[0]))
    checkpoint_file = checkpoint_files[-1]

    CKPT_PATH = os.path.join(model_dir, checkpoint_file)
    model = VAE(**params)
    checkpoint = torch.load(CKPT_PATH)
    
    print(f"Loading encoding model from {CKPT_PATH}")

    X = {}
    y = {}
    for mode, dataloader in zip(
        ['train', 'val', 'test'], 
        [train_dataloader, validation_dataloader, test_dataloader]
        ):
        print(f"Encoding {mode} set")
        X[mode] = []
        y[mode] = []
        # iterate over test dataset
        for i, datapoint in tqdm(enumerate(dataloader)):
            x = datapoint[0]
            label = datapoint[1:]
            # get latent representation
            mu, log_var = model._encode(x)
            z = model._reparameterize(mu, log_var)
            X[mode].append(z.detach().numpy())
            #print(np.concatenate(label, 1).shape)
            y[mode].append(np.concatenate(label, 1))

        X[mode] = np.vstack(np.array(X[mode]))
        y[mode] = np.vstack(np.array(y[mode]))
    print(f"Encoding done")
    return X, y 
    