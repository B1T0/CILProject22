import logging
import os
import sys
import time

import torch
from surprise import Dataset, Reader
import pandas as pd
from surprise import SVDpp, accuracy
from surprise import dump

from src.utils.logger import Logger

path = 'data/raw/train_split_'

train_path = 'data/raw/cv10/train_split_'
val_path = 'data/raw/cv10/test_split_'
pred_file = 'data/external/sampleSubmission_converted.csv'
pred_dir = 'reports/logs/20220730-164827_svdpp_30'
log_dir = 'models'
TRAIN = False
n_factors = 20  #
weight_decay = 0.0015  # 0.1


def preprocess(file_path, splits=5):
    for i in range(splits):
        print(f"Converting split {i}")
        df = pd.read_csv(file_path + f'{i}.csv')
        indices_users, indices_movies = [], []
        values = []
        for j, x in df.iterrows():
            name, val = x['Id'], x['Prediction']
            user, movie = name.replace('c', '').replace('r', '').split('_')
            indices_users.append(user)
            indices_movies.append(movie)
            values.append(val)
        data = {'user': indices_users, 'movies': indices_movies, 'values': values}
        df = pd.DataFrame(data)
        df.to_csv(file_path + f'{i}_converted.csv', index=False)


def load_dataset(file_dir, split=0):
    df = pd.read_csv(file_dir + f'{split}_converted.csv')
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user', 'movies', 'values']], reader)
    return data


def train_split(log_directory, split=0, n_factors=30, l2=0.1):
    train_data = load_dataset(train_path, split)
    test_data = load_dataset(val_path, split)

    algo = SVDpp(verbose=True, n_factors=n_factors, n_epochs=30, reg_all=l2)
    trainset = train_data.build_full_trainset()
    algo.fit(trainset)
    testset = test_data.build_full_trainset().build_testset()
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    logging.info(f'{split}: rmse: {rmse}')
    dump.dump(log_directory + f'/svdpp_default_{split}_{n_factors}_{l2}', algo=algo)


def predict_split(log_directory, save_directory, split=0, n_factors=30, l2=0.1):
    print("Loading model")
    _, loaded_algo = dump.load(save_directory + f'/svdpp_default_{split}_{n_factors}_{l2}')

    df = pd.read_csv(pred_file)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user', 'movies', 'values']], reader)
    predict_set = data.build_full_trainset().build_testset()
    predictions = loaded_algo.test(predict_set)

    names = []
    vals = []
    for i in predictions:
        names.append(f'r{i.uid}_c{i.iid}')
        vals.append(f'{i.est}')

    data = {'Id': names, 'Prediction': vals}
    df = pd.DataFrame(data)
    df.to_csv(log_directory + f'/submission_{split}.csv', index=False)


def concatenate(concat_dir, splits=10):
    print('Begin Concatenation')
    predictions = []
    idx = []
    for i in range(splits):
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


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_svdpp_{n_factors}_train_{TRAIN}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)
    logging.info("Started logging.")

    # Obtain datamodule based on config settings for dataset
    logging.info("Created data module.")
    # preprocess(train_path)
    # preprocess(val_path)
    if TRAIN:
        print("Beginning Training")
        for i in range(0, 10):
            train_split(log_dir, i)
    else:
        print("Beginning Prediction")
        for i in range(0, 10):
            predict_split(log_dir, pred_dir, i)
        concatenate('/home/jimmy/CILProject22/reports/logs/20220731-002724_svdpp_20', 10)


if __name__ == "__main__":
    main()
