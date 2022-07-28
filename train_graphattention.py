import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.GraphALS.graph_datamodule import Graph_Dataset
from src.models.GraphAttention.model import GraphAttention
from src.utils.logger import Logger, log

print(f'Available Cuda Devices {torch.cuda.device_count()}')

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
EMBEDDING_DIM = 32
GRAPH_HIDDEN = 32
HIDDEN = 32
ALPHA = 0.2
STEP_SIZE_SCHEDULER = 20
train_on_splits = True
lr = 1e-3


def train_model_splits(log_dir, file_path, dataloader, user_mode=True, val_dataloader=None, split=None):
    if user_mode:
        mode = 'user_mode'
    else:
        mode = 'movie_mode'
    log(f'Using Training Mode {mode}')
    model = GraphAttention(latent_dim=EMBEDDING_DIM, lr=lr, file_path=file_path, n_users=10000, n_items=1000,
                             mode=mode, graph_hidden=GRAPH_HIDDEN, hidden=HIDDEN, alpha=ALPHA,
                             loss=LOSS)
    log('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE_SCHEDULER, gamma=0.1)

    log(optimizer)
    best_val_loss = None
    early_stopping = 0
    for epoch in range(EPOCH):
        log(f'Epoch {epoch}')

        train_loss = 0
        model.train()
        for batch in tqdm(dataloader):
            for i, x in enumerate(batch):
                if x is not None:
                    batch[i] = x.to('cuda:0')
            loss = model.training_step(batch, 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

        log(f'Train Loss: {train_loss / len(dataloader)}')

        model.eval()

        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in tqdm(val_dataloader):
                    for i, x in enumerate(batch):
                        batch[i] = x.to('cuda:0')
                    loss = model.validation_step(batch, 0)
                    val_loss += loss
                log(f'Val Loss: {val_loss / len(val_dataloader)}')

                if best_val_loss is None:
                    best_val_loss = val_loss
                elif val_loss < best_val_loss:
                    log(f'New best model in epoch {epoch} {best_val_loss}')
                    early_stopping = 0
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': train_loss
                    }, log_dir + f'/model_best_{split}.pth')

        scheduler.step()
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': train_loss
            }, log_dir + f'/model_{epoch}_{split}.pth')
        if val_dataloader is not None:
            early_stopping += 1
            if early_stopping > EARLY_STOPPING:
                break

    if best_val_loss is not None:
        log(f"Best Val Loss of split {split} {best_val_loss/len(val_dataloader)}")
        return best_val_loss/len(val_dataloader)


def train_model_alternating(log_dir, file_path, dataloader_user, dateloader_movie, user_mode=True,
                            val_dataloader_user=None, val_dataloader_movie=None, split=None):
    raise NotImplementedError()


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_graph_attention_{EMBEDDING_DIM}_{ALPHA}_{HIDDEN}_{GRAPH_HIDDEN}_{TRAIN_MODE}"
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
        for split in range(NUM_SPLITS):
            log(f'Split {split}:  Creating Dataloader')
            if TRAIN_MODE == 'user_mode':
                dataset = Graph_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000, user=True)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
                val_dataset = Graph_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000, user=True)
                val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
                best_val_loss =train_model_splits(log_dir, path + f'{split}.csv', dataloader=dataloader, user_mode=True,
                                            val_dataloader=val_dataloader, split=split)
            elif TRAIN_MODE == 'movie_mode':
                dataset = Graph_Dataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000, user=False)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5)
                val_dataset = Graph_Dataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000,
                                            user=False)
                val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5)
                best_val_loss = train_model_splits(log_dir, path + f'{split}.csv', dataloader=dataloader, user_mode=False,
                                            val_dataloader=val_dataloader, split=split)
            else:
                raise NotImplementedError()

            best_losses.append(best_val_loss)

        print(f'Best Val Losses : {best_losses} avg {sum(best_losses)/NUM_SPLITS}')




if __name__ == "__main__":
    main()
