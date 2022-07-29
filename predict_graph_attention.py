import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.GraphAttention.model import GraphAttention
from torch.utils.data import TensorDataset
import os
import time
from torch.utils.data import DataLoader

print(torch.cuda.device_count())

path = '/home/jimmy/CILProject22/data/external/sampleSubmission.csv'
#model_dir = '/home/jimmy/CILProject22/reports/logs/20220727-131830_pretrain_norm_sgd'
#model_dir = '/home/jimmy/CILProject22/reports/logs/20220729-003141_graph_attention_128_0.2_64_64_movie_mode'
model_dir = '/home/jimmy/CILProject22/reports/logs/20220729-130520_graph_attention_split_2-2_128_0.2_48_64_movie_mode'
#model_path = '/home/jimmy/CILProject22/reports/logs/20220729-003141_graph_attention_128_0.2_64_64_movie_mode/model_best_0.pth'
graph_paths = '/home/jimmy/CILProject22/data/raw/train_split_'
concat_dir = '/home/jimmy/CILProject22/reports/logs/20220729-135056_graph_attention_prediction'
EPOCH = 50
bs = 128
SPLIT_BEGIN = 4
SPLIT = 5

#MODE = 'user_mode'
MODE = "movie_mode"
EMBEDDING_DIM = 128
GRAPH_HIDDEN = 64
HIDDEN = 48
ALPHA = 0.2
# EMBEDDING_DIM = 64
# GRAPH_HIDDEN = 48
# HIDDEN = 64
# ALPHA = 0.2
NODE_DROPOUT = 0.10
# EMBEDDING_DIM = 32
# GRAPH_HIDDEN = 32
# HIDDEN = 32

PREDICT_SPLITS = False


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_graph_attention_prediction"
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
    for split in range(SPLIT_BEGIN, SPLIT):
        print(f'Prediction split {split}')
        idx_split, predictions_split = [], []
        file_path = graph_paths + f'{split}.csv'
        model = GraphAttention(latent_dim=EMBEDDING_DIM, file_path=file_path, n_users=10000, n_items=1000,
                               mode=MODE, graph_hidden=GRAPH_HIDDEN, hidden=HIDDEN, alpha=ALPHA)

        model_path = model_dir + f'/model_best_{split}.pth'
        #print("Using hardcoded model path")
        #model_path = '/home/jimmy/CILProject22/reports/logs/20220729-003141_graph_attention_128_0.2_64_64_movie_mode/model_30_0.pth'

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Moving model to cuda')
        model.eval()
        model = model.to('cuda:0')
        with torch.no_grad():
            if MODE == 'user_mode':
                max_idx = 10000
            elif MODE == 'movie_mode':
                max_idx = 1000
            rows = TensorDataset(torch.arange(0, max_idx).cuda())
            rows_loader = DataLoader(rows, batch_size=1, shuffle=False)
            ratings = []
            for batch in tqdm(rows_loader):
                ratings.append(model.forward(batch))

            print(ratings[0].size())
            rating_matrix = torch.cat(ratings, dim=0)
            print(rating_matrix.size())
            #user must be first dimension
            if MODE =='user_mode':
                rating_matrix = rating_matrix.transpose(0, 1)
            for batch in tqdm(dataloader):
                user, movie = batch

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

            #predictions.append(predictions_split)

    #predictions = torch.tensor(predictions)


def concatenate():
    print('Begin Concatenation')
    predictions = []
    idx = []
    for i in range(SPLIT):
        df = pd.read_csv(concat_dir + f'/submission_{i}.csv')
        predictions.append(df['Prediction'].to_numpy())
        idx.append(df['Id'])
    predictions = torch.tensor(predictions)
    print(predictions.size())
    predictions = torch.mean(predictions, dim=0)
    print(predictions.size())
    predictions = predictions.squeeze()
    df = pd.DataFrame({'Id': idx[0], 'Prediction': predictions})
    df.to_csv(concat_dir + '/submission_graphattention_avg.csv', index=False)


if __name__ == "__main__":
    # main()
    concatenate()
