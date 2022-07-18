import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.CSE.model import Model
from src.models.CSE_prediction.model import Prediction
import os
import time
from src.data.Graph.graph_datamodule import Graph_Dataset
from torch.utils.data import DataLoader

print(torch.cuda.device_count())


path = '/home/jimmy/CILProject22/data/raw/train_split_0.csv'
val_path = '/home/jimmy/CILProject22/data/raw/test_split_0.csv'
model_path = '/home/jimmy/CILProject22/reports/logs/20220709-162957_pretrain_norm_sgd/model_20.pth'
EPOCH = 50
bs = 16


def main():
    pretrained = Model()
    checkpoint = torch.load(model_path)
    pretrained.load_state_dict(checkpoint['model_state_dict'])
    model = Prediction(pretrained)
    print('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    print(optimizer)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_finetuning"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    print('Creating Dataloaders')
    dataset = Graph_Dataset(file_path=path, n_users=1000, n_items=10000, threshold=False)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=6)
    val_dataset = Graph_Dataset(file_path=val_path, n_users=1000, n_items=10000, threshold=False)
    val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=6)

    print('Beginning Training')
    best_val_loss = None
    for epoch in range(EPOCH):
        print(f'Epoch {epoch}')
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
        print(torch.multinomial(torch.ones(size=(5,)), num_samples=4, generator=dataloader.generator))
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
        val_loss = val_loss / len(val_dataloader)
        if best_val_loss is None:
            best_val_loss = val_loss
        elif best_val_loss > val_loss:
            best_val_loss = val_loss
            print(f'Saving new best model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_loss
            }, log_dir+f'/model_best.pth')


if __name__ == "__main__":
    main()