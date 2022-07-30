import pandas as pd
from tqdm import tqdm
import torch
from src.models.SVD_pp.model import SVDPP
import os
import time
from src.data.SVDPP.graph_datamodule import SVDPPDataset, collate_fn
from src.data.Eval.dataset import get_eval_dataset
from src.utils.logger import Logger
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys

print(torch.cuda.device_count())

print("Using testing config")
path = '/home/jimmy/CILProject22/data/raw/cv10/train_split_'
val_path = '/home/jimmy/CILProject22/data/raw/cv10/test_split_'
FULL_DATASET = '/home/jimmy/CILProject22/data/raw/data_train.csv'
#FULL_DATASET = '/home/jimmy/CILProject22/data/raw/cv10/test_split_0.csv'
EVAL_PATH = '/home/jimmy/CILProject22/data/external/sampleSubmission.csv'

NUM_SPLITS = 10

EPOCH = 30
bs = 32
EARLY_STOPPING = 5
EMBEDDING_DIM = 30  # 20 #24 #best current #40 overfitting # 32, # 64 overfitting
train_on_splits = True
lr = 1e-3
wd = 0.05


def train_loop(model, optimizer, scheduler, dataloader, log_dir, val_dataloader=None, split=0):
    best_val_loss = None
    early_stopping = 0
    for epoch in range(EPOCH):
        print(f'Epoch {epoch}')
        train_loss = 0
        model.train()
        for batch in tqdm(dataloader):
            loss = model.training_step(batch, 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
        print(f'Train Loss: {train_loss / len(dataloader)}')
        logging.info(f'Train Loss: {train_loss / len(dataloader)}')
        model.eval()
        if val_dataloader is not None:
            print(f'Validating')
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in tqdm(val_dataloader):
                    loss = model.validation_step(batch, 0)
                    val_loss += loss
                print(f'Val Loss: {val_loss / len(val_dataloader)}')

                if best_val_loss is None:
                    best_val_loss = val_loss
                elif val_loss < best_val_loss:
                    print(f'New best model in epoch {epoch} {best_val_loss}')
                    early_stopping = 0
                    best_val_loss = val_loss
                    logging.info(f'New best model in epoch {epoch} {best_val_loss}')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': train_loss
                    }, log_dir + f'/model_best_{split}.pth')

        scheduler.step()
        if epoch % 20 == 0:
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
        print(f"Best Val Loss of split {split} {best_val_loss / len(val_dataloader)}")
        logging.info(f"Best Val Loss of split {split} {best_val_loss / len(val_dataloader)}")
        return best_val_loss / len(val_dataloader), model


def predict(model, log_dir, split=0):
    model.eval()
    idx, predictions = [], []

    dataset = SVDPPDataset(file_path=EVAL_PATH, n_items=1000, n_users=10000)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch, _ = batch
            x1 = []
            for i, j in enumerate(batch):
                x1.append(j.to('cuda:0'))
            prediction = model.forward(x1)
            user_idx = batch[0]
            item_idx = batch[1]
            prediction = prediction.cpu().numpy()
            for i in range(len(user_idx)):
                idx.append(f'r{user_idx[i]}_c{item_idx[i]}')
                predictions.append(prediction[i])

    df = pd.DataFrame({'Id': idx, 'Prediction': predictions})
    df.to_csv(log_dir + f'/submission_{split}.csv', index=False)


def train_model(log_dir, dataloader, global_mean,
                val_dataloader=None, split=None):
    model = SVDPP(num_users=10000, num_items=1000, embedding_dim=EMBEDDING_DIM,
                  global_mean=global_mean, lr=lr, wd=wd)
    print('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    best_val_score, model = train_loop(model, optimizer, scheduler, dataloader, log_dir, val_dataloader, split)

    predict(model, log_dir)
    return best_val_score


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_svdpp_{EMBEDDING_DIM}"
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
    full_dataset = SVDPPDataset(file_path=FULL_DATASET, n_items=1000, n_users=10000)
    global_mean = full_dataset.global_mean

    best_losses = []
    if train_on_splits:
        for split in range(NUM_SPLITS):
            dataset = SVDPPDataset(file_path=path + f'{split}.csv', n_items=1000, n_users=10000)
            dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=5, collate_fn=collate_fn)
            val_dataset = SVDPPDataset(file_path=val_path + f'{split}.csv', n_items=1000, n_users=10000)
            val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=5, collate_fn=collate_fn)

            best_val_loss = train_model(log_dir, dataloader=dataloader,
                                        val_dataloader=val_dataloader, split=split,
                                        global_mean=global_mean)

            best_losses.append(best_val_loss)

        print(f'Best Val Losses : {best_losses} avg {sum(best_losses) / NUM_SPLITS}')


if __name__ == "__main__":
    main()
