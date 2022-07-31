import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch


def predict_graph_autoencoder(log_dir, model, args, dataloader, split=0):
    """
    Prediction function used to predict single model
    :param log_dir: directory to save prediction files to
    :param model: model
    :param args: -
    :param dataloader: dataloader for predictions
    :param split: # for file name
    :return:
    """
    print(f'Prediction split {split}')
    idx_split, predictions_split = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            user, movie = batch
            prediction = model.forward(model.embeddings.weight, movie, user)
            user_idx = user + 1
            item_idx = movie + 1
            prediction = prediction.cpu().numpy()
            for i in range(len(user_idx)):
                idx_split.append(f'r{user_idx[i]}_c{item_idx[i]}')
                predictions_split.append(prediction[i])

        df = pd.DataFrame({'Id': idx_split, 'Prediction': predictions_split})
        df.to_csv(log_dir + f'/train_split_{split}.csv', index=False)


