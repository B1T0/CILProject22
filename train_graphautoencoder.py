import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.GraphAutoencoder.model import GraphAutoencoder
import os
import time
from src.data.GraphAutoencoder.graph_datamodule import Graph_Dataset
from src.utils.logger import Logger
from torch.utils.data import DataLoader
import logging
import sys

print(torch.cuda.device_count())

path = 'data/raw/train_split_'
val_path = 'data/raw/test_split_'
LOSS = 'MSE'
NUM_SPLITS = 5
EPOCH = 100
bs = 16
EARLY_STOPPING = 15
EMBEDDING_DIM = 100
train_on_splits = False
K = 2
lr = 1e-4


def train_model(log_dir, file_path, dataloader, val_dataloader=None, split=None):
    model = GraphAutoencoder(latent_dim=EMBEDDING_DIM, lr=lr, file_path=file_path, n_users=10000, n_items=1000, loss=LOSS)
    print('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print(optimizer)

    j = None

    best_val_loss = None
    early_stopping = 0
    for epoch in range(EPOCH):
        print(f'Epoch {epoch}')
        train_loss = 0
        train_eval = 0
        model.train()
        for batch in tqdm(dataloader):
            for i, x in enumerate(batch):
                if x is not None:
                    batch[i] = x.to('cuda:0')
            loss, eval = model.training_step(batch, batch_idx=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            eval += eval
        print(f'Train Loss: {train_loss / len(dataloader)}')
        print(f'Train Eval: {train_eval/len(dataloader)}')
        model.eval()
        print(torch.multinomial(torch.ones(size=(5,)), num_samples=4, generator=dataloader.generator))
        if j is None:
            j = (model.phi(torch.arange(1, 4, device='cuda:0')), model.phi_IC(torch.arange(1, 4, device='cuda:0')),
                 model.phi_UC(torch.arange(1, 4, device='cuda:0')))
        else:
            val = (model.phi(torch.arange(1, 4, device='cuda:0')), \
                   model.phi_IC(torch.arange(1, 4, device='cuda:0')), model.phi_UC(torch.arange(1, 4, device='cuda:0')))
            # val = model.phi(torch.arange(1,4, device='cuda:0'))
            print(f'embedding_diff: {torch.norm(val[0] - j[0])}')
            # ic and uc embeddings don't change massively....
            print(f'IC embedding_diff: {torch.norm(val[1] - j[1])}')
            print(f'UC embedding_diff: {torch.norm(val[2] - j[2])}')
            j = val
        if val_dataloader is not None:
            print(f'Validating')
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in tqdm(val_dataloader):
                    for i, x in enumerate(batch):
                        batch[i] = x.to('cuda:0')
                    loss = model.validation_step(batch, batch_idx=0)
                    val_loss += loss
                print(f'Val Loss: {val_loss / len(val_dataloader)}')
                if best_val_loss is None:
                    best_val_loss = val_loss
                elif val_loss < best_val_loss:
                    early_stopping = 0
                    best_val_loss = val_loss
                    logging.info(f'New best model in epoch {epoch} {best_val_loss}')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': train_loss
                    }, log_dir + f'/model_best_{split}.pth')
        scheduler.step()
        if epoch % 4 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_loss
            }, log_dir + f'/model_{epoch}_{split}.pth')
        if val_dataloader is not None:
            early_stopping += 1
            if early_stopping > EARLY_STOPPING:
                break


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_pretrain_norm_sgd"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")

    # Obtain datamodule based on config settings for dataset
    logging.info("Created data module.")

    print('Creating Dataloaders')

    print('Beginning Training')
    if train_on_splits:
        for split in range(NUM_SPLITS):
            dataset = Triplet_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000, k=K)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
            val_dataset = Triplet_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000, k=K)
            val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
            train_model(log_dir, dataloader, val_dataloader, split)
    else:
        dataset = Triplet_Dataset(file_path=f'data/raw/data_train.csv', n_items=1000, n_users=10000, k=K)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
        train_model(log_dir, dataloader)


if __name__ == "__main__":
    main()
