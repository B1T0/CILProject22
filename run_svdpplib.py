import logging
import os
import sys
import time

from surprise import Dataset, Reader
import pandas as pd
from surprise import SVDpp, accuracy
from surprise import dump

from src.utils.logger import Logger

path = 'data/raw/'

train_path = 'data/raw/train_split_'
val_path = 'data/raw/test_split_'
log_dir = 'models'

n_factors = 30
weight_decay = 0.1


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


def train_split(log_directory, split=0, n_factors = 30, l2=0.1):
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


def main():
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/{run_id}_svdpp_{n_factors}"
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

    for i in range(0, 10):
        train_split(log_dir, i)


if __name__ == "__main__":
    main()
