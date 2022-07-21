from tqdm import tqdm
import torch
from src.models.CSE.model import Model
from src.models.CSE_prediction.model import Prediction
import os
import time
from src.data.Graph.graph_datamodule import Graph_Dataset
from torch.utils.data import DataLoader
import sys
from src.utils.logger import Logger
import logging

print(torch.cuda.device_count())

SPLIT = 5
data_dir = 'reports/logs/run_2_5'
model_dir = 'reports/logs/20220709-162957_pretrain_norm_sgd'
EPOCH = 50
bs = 16
lr = 1e-4


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_finetuning"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")
    for i in range(SPLIT):
        print(f'Training split {i}')
        logging.info(f"Training split {i}")
        pretrained = Model()
        model_path = model_dir + f'/model_best_{i}.pth'
        checkpoint = torch.load(model_path)
        pretrained.load_state_dict(checkpoint['model_state_dict'])
        model = Prediction(pretrained, lr=lr)
        print('Moving model to cuda')
        model = model.to('cuda:0')
        optimizer = model.configure_optimizers()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        print(optimizer)
        # Create logging file
        print('Creating Dataloader')
        path = data_dir + f'/training_split_{i}.csv'
        val_path = data_dir + f'/test_split_{i}.csv'
        dataset = Graph_Dataset(file_path=path, n_users=1000, n_items=10000, threshold=False)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=6)
        val_dataset = Graph_Dataset(file_path=val_path, n_users=1000, n_items=10000, threshold=False)
        val_dataloader = DataLoader(val_dataset, batch_size=bs, num_workers=6)
        print('Beginning Training')
        best_val_loss = None
        for epoch in range(EPOCH):
            print(f'Epoch {epoch}')
            logging.info(f'Epoch {epoch}')
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
            print(f'Train Loss: {train_loss / len(dataloader)}')
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
                print(f'Val Loss: {val_loss / len(val_dataloader)}')
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
                }, log_dir + f'/model_best_{i}.pth')
            scheduler.step()


if __name__ == "__main__":
    main()
