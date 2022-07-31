from pathlib import Path
from src.utils.losses import MaskedMSELoss
import numpy as np 
import pandas as pd 
import torch 
import math 

def get_activation_function(activation_function_name):
    if activation_function_name == 'relu':
        return torch.nn.ReLU()
    elif activation_function_name == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation_function_name == 'tanh':
        return torch.nn.Tanh()
    elif activation_function_name == 'linear':
        return torch.nn.Linear()
    else:
        raise ValueError("Unknown activation function")

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
    #print(f"data shape {data.shape}")
    # write submission
    data_pd = pd.read_csv(submission_file_path) 
    test_users, test_movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    with open(save_path, 'w') as f: 
        f.write('Id,Prediction\n')
        for (user, movie) in zip(test_users, test_movies): 
            f.write("r{}_c{},{}\n".format(user+1, movie+1, data[user, movie]))

    if save_rounded:
        # if data is not numpy array then convert it to numpy array
        if not isinstance(data, np.ndarray):
            data = data.cpu().numpy()
        data = np.round(data)
        with open(save_path.replace('.csv', '_rounded.csv'), 'w') as f: 
            f.write('Id,Prediction\n')
            for (user, movie) in zip(test_users, test_movies): 
                f.write("r{}_c{},{}\n".format(user+1, movie+1, data[user, movie]))

def inference_for_submission(model, data_module, save_path, save_rounded, submission_file_path='/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', remove_bias=True):
    # set the nans to 0 for inference as in training 

    data = torch.Tensor(data_module.inference_tensor)
    data = torch.nan_to_num(data, nan=0).cuda()
    model.eval().cuda()
    with torch.no_grad():
        predictions = model(data).cpu().numpy()
    model.train()

    # transpose data and predictions if necessary
    if data_module.item_based:
        data = data.cpu().numpy().T
        predictions = predictions.T

    write_submission(predictions, submission_file_path, save_path, save_rounded)

    if remove_bias:
        print(f"Computing user-based bias")
        data = torch.squeeze(data)
        predictions = remove_user_bias(predictions, data, correction='mse')

        write_submission(predictions, submission_file_path, save_path.replace('.csv', '_bias_removed.csv'), save_rounded)

    return predictions

def remove_user_bias(predictions, correction='mse'):
    """
    Minimize per user over the known (non-nan) entries and obtain a constant bias per user 
    """
    predictions = torch.Tensor(predictions).cuda()
    data = torch.Tensor(data).cuda()
    # compute non nan mask of data 
    non_nan_mask = torch.isnan(data) == False
    user_bias_mse = torch.zeros(data.shape[0])
    user_bias_l1 = torch.zeros(data.shape[0])
    for user in range(data.shape[0]):
        user_data = data[user]
        user_pred = predictions[user]
        user_mask = non_nan_mask[user]
        user_mse = (user_data[user_mask] - user_pred[user_mask]) ** 2
        user_l1 = user_data[user_mask] - user_pred[user_mask]
        # compute the biases 
        user_bias_mse[user] = torch.sign(torch.mean(user_l1)) * math.sqrt(torch.mean(user_mse))
        user_bias_l1[user] = torch.mean(user_l1)
        #print(user_data, user_pred, user_mask)
        #print("user mse", user_mse)
        #print("user l1", user_l1)
    if correction == 'l1':
        return predictions + user_bias_l1.unsqueeze(1).cuda()
    elif correction == 'mse':
        return predictions + user_bias_mse.unsqueeze(1).cuda()
    else:
        raise ValueError("Unknown correction method")

def eval_submission(predictions, data_module):
    # create tensors if necessary
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.Tensor(predictions)
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor(data)

    loss = MaskedMSELoss()
    loss_value = loss(predictions, data_module.inference_tensor, data_module.validation_mask).item()
    return loss_value
    