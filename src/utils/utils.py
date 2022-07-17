from pathlib import Path
import numpy as np 
import pandas as pd 
import torch 


def get_data_path():
    return Path(__file__).parent.parent.parent / 'data'

def read_data(train_path, impute_value=np.nan, number_of_users=10000, number_of_movies=1000):
    data_pd = pd.read_csv(train_path) 
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # Create data matrix
    data = np.full((number_of_users , number_of_movies), impute_value)
    for user, movie ,pred in zip(users, movies, predictions): 
        data[user][movie] = pred
    return data 

def clip_data(data, clip_high=5, clip_low=1):
    data[data > clip_high] = clip_high
    data[data < clip_low] = clip_low
    return data

def write_submission(data, submission_file_path, save_path, save_rounded=False):
    # clip data first 
    data = clip_data(data)
    
    # write submission
    data_pd = pd.read_csv(submission_file_path) 
    test_users, test_movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    with open(save_path, 'w') as f: 
        f.write('Id,Prediction\n')
        for (user, movie) in zip(test_users, test_movies): 
            f.write("r{}_c{},{}\n".format(user+1, movie+1, data[user, movie]))

    if save_rounded:
        data = np.round(data)
        with open(save_path.replace('.csv', '_rounded.csv'), 'w') as f: 
            f.write('Id,Prediction\n')
            for (user, movie) in zip(test_users, test_movies): 
                f.write("r{}_c{},{}\n".format(user+1, movie+1, np.round(data[user, movie])))

def inference_for_submission(model, data, scaler, save_path, save_rounded, submission_file_path='/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv'):
    # set the nans to 0 for inference as in training 
    data = torch.nan_to_num(data, nan=0)
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    predictions = predictions.cpu().numpy()
    # undo the scaling
    predictions = scaler.inverse_transform(predictions)
    write_submission(predictions, submission_file_path, save_path, save_rounded)
    return predictions
    