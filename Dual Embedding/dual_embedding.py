"""
Implementation of the dual embedding model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
from torch.utils.data import Dataset 
import pandas as pd 
import numpy as np 
import torch.optim as optim
import time 
import os, sys 
from config import config 
import csv 

def activation(input, kind):
  """
  activation function
  """
  if kind == 'selu':
    return F.selu(input)
  elif kind == 'relu':
    return F.relu(input)
  elif kind == 'relu6':
    return F.relu6(input)
  elif kind == 'sigmoid':
    return F.sigmoid(input)
  elif kind == 'tanh':
    return F.tanh(input)
  elif kind == 'elu':
    return F.elu(input)
  elif kind == 'lrelu':
    return F.leaky_relu(input)
  elif kind == 'swish':
    return input*F.sigmoid(input)
  elif kind == 'none':
    return input
  else:
    raise ValueError('Unknown non-linearity type')

def MSEloss(inputs, targets, mask=None):
  """
  Mean squared error loss
  """
  if mask == None:
    mask = targets != 0
  num_ratings = torch.sum(mask.float())
  criterion = nn.MSELoss(reduction='sum')
  mse = criterion(inputs * mask.float(), targets * mask.float())
  # print(f"inputs {inputs[:5]}, \ntargets {targets[:5]}, \nmask {mask[:5]}")
  # print(f"mse {mse}, num_ratings {num_ratings}")
  return mse, num_ratings

def RMSE(inputs, targets):
  """
  Root mean squared error
  """
  return torch.sqrt(torch.mean((inputs - targets)**2))

class AutoEncoder(nn.Module):
  def __init__(self, layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.5, last_layer_activations=True):
    """
    Describes an AutoEncoder model
    :param layer_sizes: Encoder network description. Should start with feature size (e.g. dimensionality of x).
    For example: [10000, 1024, 512] will result in:
      - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
      - decoder 2 layers: 512x1024 and 1024x10000.
    :param nl_type: (default 'selu') Type of no-linearity
    :param is_constrained: (default: True) Should constrain decoder weights
    :param dp_drop_prob: (default: 0.0) Dropout drop probability
    :param last_layer_activations: (default: True) Whether to apply activations on last decoder layer
    """
    super(AutoEncoder, self).__init__()
    self._dp_drop_prob = dp_drop_prob
    self._last_layer_activations = last_layer_activations
    if dp_drop_prob > 0:
      self.drop = nn.Dropout(dp_drop_prob)
    self._last = len(layer_sizes) - 2
    self._nl_type = nl_type
    self.encode_w = nn.ParameterList(
      [nn.Parameter(torch.rand(layer_sizes[i + 1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)])
    for ind, w in enumerate(self.encode_w):
      weight_init.xavier_uniform_(w)

    self.encode_b = nn.ParameterList(
      [nn.Parameter(torch.zeros(layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)])

    reversed_enc_layers = list(reversed(layer_sizes))

    self.is_constrained = is_constrained
    if not is_constrained:
      self.decode_w = nn.ParameterList(
      [nn.Parameter(torch.rand(reversed_enc_layers[i + 1], reversed_enc_layers[i])) for i in range(len(reversed_enc_layers) - 1)])
      for ind, w in enumerate(self.decode_w):
        weight_init.xavier_uniform(w)
    self.decode_b = nn.ParameterList(
      [nn.Parameter(torch.zeros(reversed_enc_layers[i + 1])) for i in range(len(reversed_enc_layers) - 1)])

    print("******************************")
    print("******************************")
    print(layer_sizes)
    print("Dropout drop probability: {}".format(self._dp_drop_prob))
    print("Encoder pass:")
    for ind, w in enumerate(self.encode_w):
      print(w.data.size())
      print(self.encode_b[ind].size())
    print("Decoder pass:")
    if self.is_constrained:
      print('Decoder is constrained')
      for ind, w in enumerate(list(reversed(self.encode_w))):
        print(w.transpose(0, 1).size())
        print(self.decode_b[ind].size())
    else:
      for ind, w in enumerate(self.decode_w):
        print(w.data.size())
        print(self.decode_b[ind].size())
    print("******************************")
    print("******************************")


  def encode(self, x):
    for ind, w in enumerate(self.encode_w):
      x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
    if self._dp_drop_prob > 0: # apply dropout only on code layer
      x = self.drop(x)
    return x

  def decode(self, z):
    if self.is_constrained:
      for ind, w in enumerate(list(reversed(self.encode_w))): # constrained autoencode re-uses weights from encoder
        z = activation(input=F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[ind]),
                     # last layer or decoder should not apply non linearities
                     kind=self._nl_type if ind!=self._last or self._last_layer_activations else 'none')
        #if self._dp_drop_prob > 0 and ind!=self._last: # and no dp on last layer
        #  z = self.drop(z)
    else:
      for ind, w in enumerate(self.decode_w):
        z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]),
                     # last layer or decoder should not apply non linearities
                     kind=self._nl_type if ind!=self._last or self._last_layer_activations else 'none')
        #if self._dp_drop_prob > 0 and ind!=self._last: # and no dp on last layer
        #  z = self.drop(z)
    return z

  def forward(self, x):
    return self.decode(self.encode(x))

class MLP(nn.Module):
  def __init__(self, layer_sizes=[256,128,64,32,1], nl_type='selu', dp_drop_prob=0.2):
    """
    Describes an MLP model
    :param layer_sizes: Encoder network description. Should start with feature size (e.g. dimensionality of x).
    For example: [10000, 1024, 512] will result in:
      - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
      - decoder 2 layers: 512x1024 and 1024x10000.
    :param nl_type: (default 'selu') Type of no-linearity
    :param is_constrained: (default: True) Should constrain decoder weights
    :param dp_drop_prob: (default: 0.0) Dropout drop probability
    :param last_layer_activations: (default: True) Whether to apply activations on last decoder layer
    """
    super(MLP, self).__init__()
    self._dp_drop_prob = dp_drop_prob
    self.layer_sizes = layer_sizes
    self.activation = nl_type
    self.layers = nn.ModuleList([nn.Linear(layer_sizes[ind], layer_sizes[ind + 1]) for ind in range(len(layer_sizes) - 1)])
    if dp_drop_prob > 0:
      self.drop = nn.Dropout(dp_drop_prob)

  def forward(self, x):
      for ind, layer in enumerate(self.layers):
        x = activation(input=layer(x), kind=self.activation)
        if self._dp_drop_prob > 0 and ind < len(self.layers)-2:
          x = self.drop(x)
      return x
    
class EmbeddingModel(nn.Module):
  """
  Describes an embedding model using two emeddings and a MLP
  """
  def __init__(self, mlp, item_encoder, user_encoder):
    super(EmbeddingModel, self).__init__()
    self.item_encoder = item_encoder
    self.user_encoder = user_encoder
    self.mlp = mlp

  def forward(self, user_vector, item_vector):
    # print(f"shapes of user and item vectors: {user_vector.shape}, {item_vector.shape}")
    user_embeddings = self.user_encoder.encode(user_vector)
    item_embeddings = self.item_encoder.encode(item_vector)
    input = torch.cat([user_embeddings, item_embeddings], 1)
    return self.mlp(input)

def read_data(train_path, impute_value=0, number_of_users=10000, number_of_movies=1000):
    data_pd = pd.read_csv(train_path) 
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # Create data matrix
    data = np.full((number_of_users , number_of_movies), impute_value)
    for user, movie ,pred in zip(users, movies, predictions): 
        data[user][movie] = pred
    return data 

def eval_model(model, dataloader):
  model.eval()
  model = model.cuda()
  total_loss = 0 
  with torch.no_grad():
    for batch in dataloader:
      user_vector, item_vector, rating = batch['user_vec'].cuda(), batch['item_vec'].cuda(), batch['rating'].cuda()
      output = model(user_vector.cuda(), item_vector.cuda())
      rating = rating.unsqueeze(1)
      loss = RMSE(output, rating)
      total_loss += loss.item()
  model.train()
  return total_loss / len(dataloader)
   
def remove_user_bias(gt, pred):
    """
    Minimize per user over the known (non-nan) entries and obtain a constant bias per user 
    """
    gt = torch.Tensor(gt)
    pred = torch.Tensor(pred)
    user_residual = torch.zeros(gt.shape[0])
    non_nan_mask = gt != 0 
    user_bias_mse = torch.zeros(gt.shape[0])
    for i, user in enumerate(range(gt.shape[0])):
        user_data = gt[i]
        user_pred = pred[i]
        user_mask = non_nan_mask[i]
        # collapse to entries that are non-zero
        user_data = user_data[user_mask]
        user_pred = user_pred[user_mask]
        user_residual = (user_data - user_pred).mean()
        user_bias_mse[i] = user_residual
    # remove bias from pred matrix 
    pred_corrected =  pred + user_bias_mse.unsqueeze(1)
    return pred_corrected, user_bias_mse

class RatingDataset(Dataset):
    def __init__(self, data_matrix, index_matrix):
        self.data_matrix = data_matrix
        self.index_matrix = index_matrix
        # count nonzeros of index matrix
        self.nonzero_count = torch.sum(index_matrix != 0)
        # compute indices of nonzeros
        self.nonzero_indices = torch.nonzero(index_matrix).squeeze()

    def __len__(self):
        return self.nonzero_count

    def __getitem__(self, idx):
        i, j = self.nonzero_indices[idx]
        # return i-th row and j-th column of data matrix
        row = self.data_matrix[i,:]
        col = self.data_matrix[:,j]
        rating = self.index_matrix[i,j]
        sample = {'user_vec': row, 'item_vec': col, 'rating': rating}
        return sample 

def get_dataloaders(data_dir, split_number, batch_size=32):
    """
    return train and validation and test dataloaders 
    """
    data_matrix = torch.Tensor(read_data(data_dir + f"/data_train.csv")) # inference on the whole matrix 
    train_matrix = torch.Tensor(read_data(data_dir + f"/train_split_{split_number}.csv"))
    val_matrix = torch.Tensor(read_data(data_dir + f"/test_split_{split_number}.csv"))  
    # substitute nans with 0s 
    train_matrix[train_matrix == np.nan] = 0
    val_matrix[val_matrix == np.nan] = 0
    data_matrix[data_matrix == np.nan] = 0
    # compute train and validation mask == 0 
    train_mask = torch.zeros(data_matrix.shape)
    train_mask[train_matrix != 0] = 1
    val_mask = torch.zeros(data_matrix.shape)
    val_mask[val_matrix != 0] = 1
    data_mask = torch.zeros(data_matrix.shape)
    data_mask[data_matrix != 0] = 1
    # assert no nan in matrices 
    assert torch.sum(torch.isnan(train_matrix)) == 0
    assert torch.sum(torch.isnan(val_matrix)) == 0
    assert torch.sum(torch.isnan(data_matrix)) == 0
    # create dataloaders
    train_dataset = RatingDataset(data_matrix, train_matrix)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = RatingDataset(data_matrix, val_matrix)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, data_matrix, val_mask

def clip_data(data, clip_high=5, clip_low=1):
    """
    clip data to [clip_low, clip_high]
    """
    data[data > clip_high] = clip_high
    data[data < clip_low] = clip_low
    return data

def write_submission(model, data, submission_file_path, save_path):
    """
    write submission file
    """
    data_pd = pd.read_csv(submission_file_path) 
    test_users, test_movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    preds = [] 
    model = model.cuda() 
    # create dataloader from test_users and test_movies
    indices_to_predict = torch.zeros(data.shape)
    indices_to_predict[test_users, test_movies] = 1
    test_dataset = RatingDataset(data, indices_to_predict)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    for i, batch in enumerate(test_dataloader):
        user_vector, item_vector, rating = batch['user_vec'].cuda(), batch['item_vec'].cuda(), batch['rating'].cuda()
        output = model(user_vector.cuda(), item_vector.cuda())
        preds.append(output.cpu().detach().numpy())
    preds = np.concatenate(preds, axis=0) 
    # write submission 
    with open(save_path, 'w') as f: 
        f.write('Id,Prediction\n')
        for (user, movie, pred) in zip(test_users, test_movies, preds): 
            f.write("r{}_c{},{}\n".format(user+1, movie+1, pred[0]))
    return preds 

def write_predictions(preds, submission_file_path, save_path):
    """
    write predictions to submission file
    """
    data_pd = pd.read_csv(submission_file_path) 
    test_users, test_movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    preds = clip_data(preds)
    with open(save_path, 'w') as f: 
        f.write('Id,Prediction\n')
        for i, (user, movie) in enumerate(zip(test_users, test_movies)): 
            # get user row and movie col from data matrix 
            pred = clip_data(preds[i])
            f.write("r{}_c{},{}\n".format(user+1, movie+1, preds[i][0]))
    return np.array(preds)

def run_epoch(log_dir, epoch, model, train_dataloader, valid_dataloader, optimizer, criterion, min_val_loss, no_improvement_count, verbose=True, print_frequency=2, batch_frequency=200, patience=5):
    if verbose: print(f"Train epoch: {epoch}")

    # training 
    if verbose: print(f"Training...")
    model.train()
    train_loss = 0
    for i, batch in enumerate(valid_dataloader):
        user_vec, item_vec, rating = batch['user_vec'].cuda(), batch['item_vec'].cuda(), batch['rating'].cuda()
        optimizer.zero_grad()
        z = model(user_vec, item_vec)
        rating = rating.unsqueeze(1)
        loss = criterion(z, rating)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if i>0 and i % batch_frequency == 0:
            print(f"Epoch: {epoch}, batch: {i}/{len(train_dataloader)}, train MSE: {train_loss/i}")
    if epoch % print_frequency == 0: print(f"Train MSE epoch {epoch}: {train_loss / len(train_dataloader)}")

    # validation
    if verbose: print(f"Validation...")
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            user_vec, item_vec, rating = batch['user_vec'].cuda(), batch['item_vec'].cuda(), batch['rating'].cuda()
            z = model(user_vec, item_vec)
            rating = rating.unsqueeze(1)
            loss = RMSE(z, rating)
            valid_loss += loss.item()
            if i>0 and i % batch_frequency == 0:
              print(f"Epoch: {epoch}, batch: {i}/{len(valid_dataloader)}, valid RMSE: {valid_loss/i}")
            
    if epoch % print_frequency == 0: print(f"Valid loss epoch {epoch}: {valid_loss / len(valid_dataloader)}")
    if verbose: print(f"Done epoch: {epoch}")

    if valid_loss/len(valid_dataloader) < min_val_loss:
      min_val_loss = valid_loss/len(valid_dataloader)
      torch.save(model.state_dict(), log_dir + f"/best_model.pt")
      print(f"Saved model in epoch {epoch}, val RMSE {valid_loss/len(valid_dataloader)}")
      no_improvement_count = 0
    else:
      no_improvement_count += 1
      if no_improvement_count > patience:
        print(f"Early stopping at epoch {epoch}")
        return train_loss/len(train_dataloader), valid_loss/len(valid_dataloader), min_val_loss, no_improvement_count, True # early stopping

    return train_loss/len(train_dataloader), valid_loss/len(valid_dataloader), min_val_loss, no_improvement_count, False # return train loss, valid loss, min val loss, no improvement count, early stopping


def main():
    # settings 
    nb_splits = 5
    split_dir = '5fold' if nb_splits == 5 else '10fold'
    epochs = 20
    batch_size = 512
    lr = 1e-3
    weight_decay = 1e-4
    dropout = 0

    item_log_dir = "/home/ubuntu/projects/CILProject22/reports/logs/20220727-104134_ItemEncoder_bs64_lr0.001_wd0.01_do0.5_ls[10000, 128, 128]_submit_unb_True_0.99636/UserAutoEncoder_split_4/best_model.pt"
    user_log_dir = "/home/ubuntu/projects/CILProject22/reports/logs/20220727-074319_UserAutoEncoder_bs32_lr0.001_wd0.001_do0.5_ls[1000, 128, 32, 32, 32]_submit_unb_True_0.98823/UserAutoEncoder_split_0/best_model.pt"
    # Create encoders 
    item_model = AutoEncoder(
          layer_sizes=[10000,128,128], 
          nl_type='selu', 
          is_constrained=True, 
          dp_drop_prob=0.5, 
          last_layer_activations=True
          ).cuda()
    user_model = AutoEncoder(
          layer_sizes=[1000,128,32,32,32], 
          nl_type='selu', 
          is_constrained=True, 
          dp_drop_prob=0.5, 
          last_layer_activations=True
          ).cuda()
    item_model.load_state_dict(torch.load(item_log_dir))
    user_model.load_state_dict(torch.load(user_log_dir))

    mlp_layer_sizes = [128+32, 128, 16, 1]

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_Embedding_MLP" + f"_bs{batch_size}" + f"_lr{lr}" + f"_wd{weight_decay}" + f"_do{dropout}" + f"_ls{mlp_layer_sizes}"
    predictions = None 
    losses = []
    for i in range(nb_splits):
        print(f"----- Split {i+1} / {nb_splits} -----")
        # Create model directory and Logger
        log_dir = f"reports/logs/{run_id}/Embedding_split_{i}"
        if not os.path.exists(log_dir):
          os.makedirs(log_dir)
        
        # Create model
        mlp = MLP(
          layer_sizes=mlp_layer_sizes,
          dp_drop_prob=0.25,
        ).cuda()
        embedding_model = EmbeddingModel(
          mlp=mlp,
          item_encoder = item_model,
          user_encoder=user_model
        ).cuda()

        train_loader, valid_loader, data_matrix, valid_mask = get_dataloaders(f'./data/raw/{split_dir}', split_number=i, batch_size=batch_size)
        optimizer = optim.AdamW(embedding_model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()

        # save model as best model to init 
        torch.save(embedding_model.state_dict(), log_dir + f"/best_model.pt")

        min_val_loss = np.inf
        no_improvement_count = 0
        for epoch in range(epochs):  
            train_loss, valid_loss, min_val_loss, no_improvement_count, early_stopped = run_epoch(log_dir, epoch, embedding_model, train_loader, valid_loader, optimizer, criterion, min_val_loss, no_improvement_count, verbose=False)
            if early_stopped:
                break
        
        # load best validation embedding model 
        embedding_model.load_state_dict(torch.load(log_dir + f"/best_model.pt"))
        print(f"Loaded model")
        # evaluate model 
        val_RMSE = eval_model(embedding_model, valid_loader)
        print(f"Model validation RMSE: {val_RMSE}")
        losses.append(val_RMSE)
        # write validation submission  
        write_submission(embedding_model, data_matrix, f"./data/raw/{split_dir}/test_split_{i}.csv", f"{log_dir}/valid_submission.csv")
        # write test submission 
        preds = write_submission(embedding_model, data_matrix, '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', f"{log_dir}/test_submission.csv")
        if predictions is None:
            predictions = preds 
        else:
            predictions += preds
    
    # write submission
    predictions /= nb_splits
    write_predictions(predictions, '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', f'reports/logs/{run_id}/ensemble_submission.csv')

    print(f"Finished run")
    # Save losses and avg to csv 
    with open(f'reports/logs/{run_id}' + '/all_losses.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(losses)
        writer.writerow([np.mean(losses)])


        

if __name__ == "__main__":
    main()