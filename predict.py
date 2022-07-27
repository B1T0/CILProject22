import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.CSE.model import Model
from src.models.CSE_prediction.model import Prediction
from torch.utils.data import TensorDataset
import os
import time
from src.data.Graph.graph_datamodule import Graph_Dataset
from torch.utils.data import DataLoader

print(torch.cuda.device_count())

path = '/home/jimmy/CILProject22/data/external/sampleSubmission.csv'
# model_path = '/home/jimmy/CILProject22/reports/logs/20220710-164447_finetuning/model_best.pth'
#model_path = '/home/jimmy/CILProject22/reports/logs/20220714-232851_finetuning/model_best.pth'
model_path = '/home/jimmy/CILProject22/reports/logs/20220727-023447_pretrain_norm_sgd/'
EPOCH = 50
bs = 16
n_users = 1000


def main():


    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_prediction"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    print('Creating Dataloaders')

    model.eval()
    indices_users = []
    indices_movies = []
    df = pd.read_csv(path)
    for i, x in df.iterrows():
        name, _ = x['Id'], x['Prediction']
        user, movie = name.replace('c', '').replace('r', '').split('_')
        user, movie = int(user) - 1, int(movie) - 1
        indices_users.append(user)
        indices_movies.append(movie + n_users)

    users = torch.tensor(indices_users).to('cuda:0')
    items = torch.tensor(indices_movies).to('cuda:0')
    dataset = TensorDataset(users, items)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
    idx, predictions = [], []
    print('Beginning Prediction')

    with torch.no_grad():
        for batch in tqdm(dataloader):
            user, item = batch
            prediction = model.forward(user, item)
            user_idx = user + 1
            item_idx = item - n_users + 1
            prediction = prediction.cpu().numpy()
            for i in range(len(user_idx)):
                idx.append(f'r{user_idx[i]}_c{movie_idx[i]}')
                predictions.append(int(prediction[i]))

    df = pd.DataFrame({'Id': idx, 'Prediction': predictions})
    df.to_csv(log_dir + '/submission_pretrained_mse_pc.csv', index=False)


if __name__ == "__main__":
    main()
