
import logging
import os
import sys
import time

import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.model_loader import get_model, predict_model, get_model_params
from utils.logger import Logger
from config import config

"""
Prediction file for model prediction over all training splits, 
uses basic config.py and hyperparameter.py for configuration
"""


SUBMISSION_PATH = 'data/external/sampleSubmission.csv'
PATH = config['train_path']
START_SPLIT = config['start_split']
NUM_SPLITS = config['num_splits']
model_name = config['model']
MODEL_DIR = config['predict_dir']


def get_submission_dataloader(path):
    """
    returns dataloader used for submission prediction
    :param path:
    :return:
    """
    indices_users = []
    indices_movies = []
    df = pd.read_csv(path)
    for i, x in df.iterrows():
        name, _ = x['Id'], x['Prediction']
        user, movie = name.replace('c', '').replace('r', '').split('_')
        user, movie = int(user) - 1, int(movie) - 1
        indices_users.append(user)
        indices_movies.append(movie)

    users = torch.tensor(indices_users).to('cuda:0')
    items = torch.tensor(indices_movies).to('cuda:0')
    dataset = TensorDataset(users, items)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader


def concatenate(concat_dir):
    """
    averages all k model predictions into a single prediction
    """
    print('Begin Concatenation')
    predictions = []
    idx = []
    for i in range(NUM_SPLITS):
        df = pd.read_csv(concat_dir + f'/submission_{i}.csv')
        predictions.append(df['Prediction'].to_numpy())
        idx.append(df['Id'])
    predictions = torch.tensor(predictions)
    print(predictions.size())
    predictions = torch.mean(predictions, dim=0)
    print(predictions.size())
    predictions = predictions.squeeze()
    df = pd.DataFrame({'Id': idx[0], 'Prediction': predictions})
    df.to_csv(concat_dir + f'/submission_{model_name}_avg.csv', index=False)

def prediction(args, file_name, model_name, model_dir, split=None):
    model_params = args['model_params']  # graph models need this
    model = get_model(model_name, model_params, file_name)
    model_path = model_dir + f'/model_best_{split}.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Moving model to cuda')
    dataloader = get_submission_dataloader(SUBMISSION_PATH)
    model.eval()
    model = model.to('cuda:0')

    predict_model(model_name, model, args, dataloader, split)

def main():

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_{model_name}_predictions"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")
    args = get_model_params(model_name)


    for split in range(START_SPLIT, NUM_SPLITS):
        file_name = PATH + f'{split}.csv'
        prediction(args, file_name, model_name, MODEL_DIR, split)
