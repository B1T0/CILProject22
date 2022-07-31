import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
import time
from torch.utils.data import DataLoader


def predict_graph_attention(log_dir, model, args, dataloader, split=0):
    """
    prediction function for Graph Attention User Encoder
    first creates full rating matrix before subsampling the given submission entries
    :param log_dir: directory to save file to
    :param model: model
    :param args: -
    :param dataloader: dataloader for submission indices
    :param split: # for name
    """

    MODE = args['train_mode']

    idx_split, predictions_split = [], []
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
        # user must be first dimension
        if MODE == 'user_mode':
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

        df = pd.DataFrame({'Id': idx_split, 'Prediction': predictions_split})
        df.to_csv(log_dir + f'/submission_{split}.csv', index=False)

        time.sleep(10)


