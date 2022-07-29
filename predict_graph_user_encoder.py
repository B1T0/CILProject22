import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.GraphALS.model import GraphAutoencoder
from torch.utils.data import TensorDataset
import os
import time
from torch.utils.data import DataLoader

print(torch.cuda.device_count())

path = '/home/jimmy/CILProject22/data/external/sampleSubmission.csv'
model_dir = '/home/jimmy/CILProject22/reports/logs/20220728-215328_graphautoencoder_16_user_mode'
graph_paths = '/home/jimmy/CILProject22/data/raw/train_split_'
EPOCH = 50
bs = 128
SPLIT = 5

MODE = 'user_mode'
EMBEDDING_DIM = 16



def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_graph_user_encoder_prediction"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    print('Creating Dataloaders')

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
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
    idx, predictions = [], []
    print('Beginning Prediction')
    for split in range(SPLIT):
        print(f'Prediction split {split}')
        idx_split, predictions_split = [], []
        file_path = graph_paths + f'{split}.csv'
        model = GraphAutoencoder(latent_dim=EMBEDDING_DIM, file_path=file_path, n_users=10000, n_items=1000,
                               mode=MODE)

        model_path = model_dir + f'/model_best_{split}.pth'

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Moving model to cuda')
        model.eval()
        model = model.to('cuda:0')
        with torch.no_grad():
            if MODE == 'user_mode':
                i = 10000
            elif MODE == 'movie_mode':
                i = 1000

            rating_matrix = model.forward(torch.arange(0, i))
            print(rating_matrix.size())
            #user must be first dimension
            if MODE =='user_mode':
                rating_matrix = rating_matrix.transpose(0, 1)
            print(rating_matrix.size())
            for batch in tqdm(dataloader):
                user, movie = batch
                #print(user)
                prediction = rating_matrix[movie, user]
                user_idx = user + 1
                item_idx = movie + 1
                prediction = prediction.cpu().numpy()
                for i in range(len(user_idx)):
                    idx_split.append(f'r{user_idx[i]}_c{item_idx[i]}')
                    predictions_split.append(prediction[i])

            if len(idx) == 0:
                print("Appending ids")
                idx.append(idx_split)

            df = pd.DataFrame({'Id': idx[0], 'Prediction': predictions_split})
            df.to_csv(log_dir + f'/submission_{split}.csv', index=False)

            predictions.append(predictions_split)

    predictions = torch.tensor(predictions)
    print(predictions.size())
    predictions = torch.mean(predictions, dim=0)
    print(predictions.size())
    predictions = predictions.squeeze()
    df = pd.DataFrame({'Id': idx[0], 'Prediction': predictions})
    df.to_csv(log_dir + '/submission_graph_user_encoder.csv', index=False)


if __name__ == "__main__":
    main()
