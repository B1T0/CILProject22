import torch
from src.models.model_loader import get_model, get_model_params, predict_model
import os
import time
from src.data.GraphAutoencoder.graph_datamodule import Graph_Dataset
from src.utils.logger import Logger, log
from torch.utils.data import DataLoader

from src.utils.utils import train_model
from src.data.util import get_dataset
import logging
import sys

path = '/home/jimmy/CILProject22/data/raw/train_split_'
val_path = '/home/jimmy/CILProject22/data/raw/test_split_'
whole_data = '/home/jimmy/CILProject22/data/raw/data_train.csv'
LOSS = 'MSE'
BEGIN_SPLITS = 3
NUM_SPLITS = 5
EPOCH = 50
bs = 32
EARLY_STOPPING = 5
EMBEDDING_DIM = 32  # 32 #64 #100
train_on_splits = True
lr = 1e-3


def begin_training(args, model_name, file_name, log_dir, dataloader, val_dataloader=None, split=None):
    model_params = args['model_params']  # graph models need this
    model = get_model(model_name, model_params, file_name)

    print('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    print(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    return train_model(model, scheduler, optimizer, log_dir, dataloader, val_dataloader, split,
                       args['save_period'], args['patience'], args['epochs'])

def main():
    #model_name = "GraphAttention"
    #model_name = 'GraphAutoencoder'
    #model_name = 'GraphUserEncoder'
    model_name = 'SVDGraphAttention'
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_graph_auto_encoder_{EMBEDDING_DIM}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")
    args = get_model_params(model_name)
    print(args)

    if train_on_splits:
        best_losses = []
        for split in range(BEGIN_SPLITS, NUM_SPLITS):
            dataset = get_dataset(model_name, val_path + f'{split}.csv', args['model_params'])
            dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=5)
            #val_dataset = get_dataset(model_name, val_path + f'{split}.csv', args['model_params'])
            #val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], num_workers=5)
            val_dataloader = dataloader
            print('Beginning Training')
            best_val_loss = begin_training(args, model_name, path + f'{split}.csv', log_dir, dataloader,
                                           val_dataloader, split)
            best_losses.append(best_val_loss)

        print(f'Best Val Losses : {best_losses} avg {sum(best_losses) / NUM_SPLITS}')
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
