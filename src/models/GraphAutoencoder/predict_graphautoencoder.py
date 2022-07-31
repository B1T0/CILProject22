import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.GraphAutoencoder.model import GraphAutoencoder
from torch.utils.data import TensorDataset
import os
import time
from src.data.Graph.graph_datamodule import Graph_Dataset
from torch.utils.data import DataLoader

print(torch.cuda.device_count())

# path = '/home/jimmy/CILProject22/data/external/sampleSubmission.csv'
path = '/home/jimmy/CILProject22/data/raw/test_split_'
# model_path = '/home/jimmy/CILProject22/reports/logs/20220710-164447_finetuning/model_best.pth'
# model_path = '/home/jimmy/CILProject22/reports/logs/20220714-232851_finetuning/model_best.pth'
# model_dir = '/home/jimmy/CILProject22/reports/logs/20220727-023447_pretrain_norm_sgd'
model_dir = '/home/jimmy/CILProject22/reports/logs/20220727-131830_pretrain_norm_sgd'
graph_paths = '/home/jimmy/CILProject22/data/raw/train_split_'
EPOCH = 50
bs = 128
SPLIT = 5
EMBEDDING_DIM = 64  # 100


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_prediction"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    print('Creating Dataloaders')

    print('Beginning Prediction')
    for split in range(SPLIT):

        indices_users = []
        indices_movies = []
        df = pd.read_csv(path + f'{split}.csv')
        for i, x in df.iterrows():
            name, _ = x['Id'], x['Prediction']
            user, movie = name.replace('c', '').replace('r', '').split('_')
            user, movie = int(user) - 1, int(movie) - 1
            indices_users.append(user)
            indices_movies.append(movie)

        users = torch.tensor(indices_users).to('cuda:0')
        items = torch.tensor(indices_movies).to('cuda:0')
        dataset = TensorDataset(users, items)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
        idx, predictions = [], []

        print(f'Prediction split {split}')
        idx_split, predictions_split = [], []
        file_path = graph_paths + f'{split}.csv'
        model = GraphAutoencoder(EMBEDDING_DIM, 10000, 1000, file_path)
        model_path = model_dir + f'/model_best_{split}.pth'

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Moving model to cuda')
        model.eval()
        model = model.to('cuda:0')
        with torch.no_grad():
            for batch in tqdm(dataloader):
                user, movie = batch
                # print(user)
                # print(movie)
                # break
                prediction = model.forward(model.embeddings.weight, movie, user)
                user_idx = user + 1
                item_idx = movie + 1
                prediction = prediction.cpu().numpy()
                for i in range(len(user_idx)):
                    idx_split.append(f'r{user_idx[i]}_c{item_idx[i]}')
                    predictions_split.append(prediction[i])

            if len(idx) == 0:
                print("Appending ids")
                idx.append(idx_split)

            df = pd.DataFrame({'Id': idx_split, 'Prediction': predictions_split})
            df.to_csv(log_dir + f'/train_split_{split}.csv', index=False)

            predictions.append(predictions_split)

    # predictions = torch.tensor(predictions)
    # print(predictions.size())
    # predictions = torch.mean(predictions, dim=0)
    # print(predictions.size())
    # predictions = predictions.squeeze()
    # df = pd.DataFrame({'Id': idx[0], 'Prediction': predictions})
    # df.to_csv(log_dir + '/submission_graphauto_encoder.csv', index=False)


def predict_graph_autoencoder(log_dir, model, args, dataloader, split=0):

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


if __name__ == "__main__":
    main()
