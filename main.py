import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.CSE.model import Model
import os
import time
from src.data.Contrastive.graph_datamodule import Triplet_Dataset
from torch.utils.data import DataLoader

print(torch.cuda.device_count())


path = '/home/jimmy/CILProject22/data/raw/train_split_0.csv'
val_path = '/home/jimmy/CILProject22/data/raw/test_split_0.csv'
EPOCH = 50
bs = 16


def main():
    model = Model()
    print('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    print(optimizer)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_pretrain_norm_sgd"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    print('Creating Dataloaders')
    dataset = Triplet_Dataset(file_path=path, n_users=1000, n_items=10000, k=2)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=6)
    val_dataset = Triplet_Dataset(file_path=val_path, n_users=1000, n_items=10000, k=2)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=6)

    print('Beginning Training')
    for i in range(EPOCH):
        print(f'Epoch {i}')
        train_loss = 0
        model.train()
        for batch in tqdm(dataloader):
            for i, x in enumerate(batch):
                batch[i] = x.to('cuda:0')
            loss = model.training_step(batch, batch_idx=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
        print(f'Train Loss: {train_loss/len(dataloader)}')
        model.eval()
        print(f'Validating')
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_dataloader):
                for i, x in enumerate(batch):
                    batch[i] = x.to('cuda:0')
                loss = model.validation_step(batch, batch_idx=0)
                val_loss += loss
            print(f'Val Loss: {val_loss/len(val_dataloader)}')

        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'loss': train_loss
        }, log_dir+f'/model_{i}.pth')


if __name__ == "__main__":
    main()