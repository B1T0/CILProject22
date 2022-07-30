import torch
from src.models.SVGppGAttention.model import GraphAttention
import os
import time
from src.data.GraphAutoencoder.graph_datamodule import Graph_Dataset
from src.utils.logger import Logger
from torch.utils.data import DataLoader

from utils import train_model
import logging
import sys

print(torch.cuda.device_count())

path = '/home/jimmy/CILProject22/data/raw/train_split_'
val_path = '/home/jimmy/CILProject22/data/raw/test_split_'
whole_data = '/home/jimmy/CILProject22/data/raw/data_train.csv'
LOSS = 'MSE'
NUM_SPLITS = 5
EPOCH = 50
bs = 32
EARLY_STOPPING = 5
EMBEDDING_DIM = 48  # 32 #64 #100
train_on_splits = True
lr = 1e-4


def begin_training(file_path, log_dir, dataloader, val_dataloader=None, split=None,
                   save_period=10, patience=20, epochs=50, global_mean = 0):
    model = GraphAttention(latent_dim=EMBEDDING_DIM, lr=lr, file_path=file_path, global_mean=global_mean, n_users=10000, n_items=1000,
                             loss=LOSS)
    print('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    return train_model(model, scheduler, optimizer, log_dir, dataloader, val_dataloader, split,
                       save_period, patience, epochs)


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_graph_auto_encoder_{EMBEDDING_DIM}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")


    dataset = Graph_Dataset(file_path=whole_data, n_items=1000, n_users=10000)
    global_mean = dataset.gm

    if train_on_splits:
        best_losses = []
        for split in range(NUM_SPLITS):
            dataset = Graph_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)

            val_dataset = Graph_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000)
            val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
            print('Beginning Training')
            best_val_loss = begin_training( path + f'{split}.csv', log_dir, dataloader,
                                            val_dataloader, split, global_mean=global_mean)
            best_losses.append(best_val_loss)

        print(f'Best Val Losses : {best_losses} avg {sum(best_losses) / NUM_SPLITS}')
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
