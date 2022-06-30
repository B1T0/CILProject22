"""
Create dataloaders depending on settings in config.py
"""

from datetime import datetime
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import config
from src.data.datasets import PadChestDataset
from src.utils.utils import get_data_path


def get_padchest_dataloaders(num_workers=4):
    if config["mode"] == "train":
        label_fname = "training_subset_labels.csv"
    elif config["mode"] == "dev":
        label_fname = "development_subset_labels.csv"
    else:
        raise ValueError(f"Invalid mode in config.py: {config['mode']}")

    df = pd.read_csv(get_data_path() / "processed" / label_fname)

    # Split the test set with a fixed seed such that it is fixed
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    # Split train and validation randomly
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=random.seed(datetime.now())
    )

    # Create the datasets
    train_dataset = PadChestDataset(train_df)
    val_dataset = PadChestDataset(val_df)
    test_dataset = PadChestDataset(test_df)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # TODO: use lightning datamodule instead!
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers
    )
    validation_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers
    )
    return train_dataloader, validation_dataloader, test_dataloader

