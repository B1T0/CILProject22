import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_model
from src.data.GraphALS.graph_datamodule import Graph_Dataset
from src.models.GraphAttention.model import GraphAttention
from src.utils.logger import Logger, log

print(f'Available Cuda Devices {torch.cuda.device_count()}')

path = '/home/jimmy/CILProject22/data/raw/train_split_'
# test only
#path = '/home/jimmy/CILProject22/data/raw/test_split_'
val_path = '/home/jimmy/CILProject22/data/raw/test_split_'
FINETUNING = False
SAVED_DIR = '/home/jimmy/CILProject22/reports/logs/20220729-130520_graph_attention_split_2-2_128_0.2_48_64_movie_mode'
LOSS = 'MSE'
TRAIN_MODE = "user_mode"  # defining user as movie embeddings and recreating user rows
# TRAIN_MODE = "movie_mode"
# TRAIN_MODE = "alternating"

NUM_SPLITS = 5
EPOCH = 500
bs = 32
EARLY_STOPPING = 40
EMBEDDING_DIM = 64
GRAPH_HIDDEN = 32
HIDDEN = 32
ALPHA = 0.2
STEP_SIZE_SCHEDULER = 50
STEP_SIZE_START = 50
train_on_splits = True
lr = 1e-4
DROPOUT = 0.0

MODEL_SAVE = 50
SAVE_EPOCH = 50
SPLIT_START = 2


def train_model_splits(log_dir, file_path, dataloader, user_mode=True, val_dataloader=None, split=None):
    if user_mode:
        mode = 'user_mode'
    else:
        mode = 'movie_mode'
    log(f'Using Training Mode {mode}')
    model = GraphAttention(latent_dim=EMBEDDING_DIM, lr=lr, file_path=file_path, n_users=10000, n_items=1000,
                           mode=mode, graph_hidden=GRAPH_HIDDEN, hidden=HIDDEN, alpha=ALPHA,
                           loss=LOSS, dropout=DROPOUT)
    if FINETUNING:
        log(f'Loading pretrained models')
        model_path = SAVED_DIR + f'/model_best_{split}.pth'
        # print("Using hardcoded model path")
        # model_path = '/home/jimmy/CILProject22/reports/logs/20220729-003141_graph_attention_128_0.2_64_64_movie_mode/model_30_0.pth'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    log('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE_SCHEDULER, gamma=0.3, verbose=True)

    train_model(model, optimizer, scheduler, log_dir, dataloader, val_dataloader, split, SA)


def train_model_alternating(log_dir, file_path, dataloader_user, dateloader_movie, user_mode=True,
                            val_dataloader_user=None, val_dataloader_movie=None, split=None):
    raise NotImplementedError()


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_graph_attention_split_{SPLIT_START}-{NUM_SPLITS - 1}_{EMBEDDING_DIM}_{ALPHA}_{HIDDEN}_{GRAPH_HIDDEN}_{TRAIN_MODE}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")

    # Obtain datamodule based on config settings for dataset

    best_losses = []
    if train_on_splits:
        for split in range(SPLIT_START, NUM_SPLITS):
            log(f'Split {split}:  Creating Dataloader')
            if TRAIN_MODE == 'user_mode':
                dataset = Graph_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000, user=True)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
                val_dataset = Graph_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000, user=True)
                val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
                best_val_loss = train_model_splits(log_dir, path + f'{split}.csv', dataloader=dataloader,
                                                   user_mode=True,
                                                   val_dataloader=val_dataloader, split=split)
            elif TRAIN_MODE == 'movie_mode':
                dataset = Graph_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000, user=False)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
                val_dataset = Graph_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000,
                                            user=False)
                val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
                best_val_loss = train_model_splits(log_dir, path + f'{split}.csv', dataloader=dataloader,
                                                   user_mode=False,
                                                   val_dataloader=val_dataloader, split=split)
            else:
                raise NotImplementedError()
            log(f'Best Val Losses : {best_losses} avg {sum(best_losses) / (NUM_SPLITS - SPLIT_START + split)}')
            best_losses.append(best_val_loss)
            log("Waiting to prevent memory issues")
            time.sleep(10)
        log(f'Best Val Losses : {best_losses} avg {sum(best_losses) / (NUM_SPLITS-SPLIT_START)}')


if __name__ == "__main__":
    main()
