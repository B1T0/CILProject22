import torch
from src.models.GraphUserEncoder.model import GraphUserEncoder
import os
import time
from src.data.RowDataset.graph_datamodule import RowDataset
from src.utils.logger import Logger
from torch.utils.data import DataLoader
import logging
import sys

from src.utils.utils import train_model

path = '/home/jimmy/CILProject22/data/raw/train_split_'
val_path = '/home/jimmy/CILProject22/data/raw/test_split_'
LOSS = 'MSE'
TRAIN_MODE = "user_mode"  # defining user as movie embeddings and recreating user rows
# TRAIN_MODE = "movie_mode"
# TRAIN_MODE = "alternating"

NUM_SPLITS = 5
EPOCH = 200
bs = 32
EARLY_STOPPING = 30
EMBEDDING_DIM =  16 #20 #24 #best current #40 overfitting # 32, # 64 overfitting
train_on_splits = True
lr = 1e-3


def train_model_non_alternating(log_dir, file_path, dataloader, user_mode=True, val_dataloader=None, split=None):
    if user_mode:
        mode = 'user_mode'
    else:
        mode = 'movie_mode'
    print(f'Using Training Mode {mode}')
    logging.info(f'Using Training Mode {mode}')
    model = GraphUserEncoder(latent_dim=EMBEDDING_DIM, lr=lr, file_path=file_path, n_users=10000, n_items=1000,
                             mode=mode,
                             loss=LOSS)
    print('Moving model to cuda')

    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_model(model, scheduler, optimizer, log_dir, dataloader, val_dataloader, split)

def train_model_alternating(log_dir, file_path, dataloader_user, dateloader_movie, user_mode=True,
                            val_dataloader_user=None, val_dataloader_movie=None, split=None):
    raise NotImplementedError()


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_graphautoencoder_{EMBEDDING_DIM}_{TRAIN_MODE}"
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

    best_losses = []
    if train_on_splits:
        for split in range(NUM_SPLITS):
            if TRAIN_MODE == 'user_mode':
                dataset = Graph_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000, user=True)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
                val_dataset = Graph_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000, user=True)
                val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
                best_val_loss =train_model_non_alternating(log_dir, path + f'{split}.csv', dataloader=dataloader, user_mode=True,
                                            val_dataloader=val_dataloader, split=split)
            elif TRAIN_MODE == 'movie_mode':
                dataset = Graph_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000, user=False)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
                val_dataset = Graph_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000,
                                            user=False)
                val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
                best_val_loss = train_model_non_alternating(log_dir, path + f'{split}.csv', dataloader=dataloader, user_mode=False,
                                            val_dataloader=val_dataloader, split=split)
            else:
                raise NotImplementedError()

            best_losses.append(best_val_loss)

        print(f'Best Val Losses : {best_losses} avg {sum(best_losses)/NUM_SPLITS}')




if __name__ == "__main__":
    main()
