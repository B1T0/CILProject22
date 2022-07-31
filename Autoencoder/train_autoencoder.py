# Copyright (c) 2017 NVIDIA Corporation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
import pandas as pd 
import numpy as np 
import torch.optim as optim
import time 
import os, sys 
from config import config 
import csv 

def activation(input, kind):
  #print("Activation: {}".format(kind))
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
  if mask == None:
    mask = targets != 0
  num_ratings = torch.sum(mask.float())
  criterion = nn.MSELoss(reduction='sum')
  mse = criterion(inputs * mask.float(), targets * mask.float())
  # print(f"inputs {inputs[:5]}, \ntargets {targets[:5]}, \nmask {mask[:5]}")
  # print(f"mse {mse}, num_ratings {num_ratings}")
  return mse, num_ratings

def RMSE(inputs, targets, mask=None):
  if mask == None:
    mask = targets != 0
  num_ratings = torch.sum(mask.float())
  criterion = nn.MSELoss(reduction='sum')
  mse = criterion(inputs * mask.float(), targets * mask.float())
  return torch.sqrt(mse / num_ratings)

class AutoEncoder(nn.Module):
  def __init__(self, layer_sizes, nl_type='selu', is_constrained=True, dp_drop_prob=0.5, last_layer_activations=True, verbose=False):
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
    self.verbose = verbose 
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

    if self.verbose: print("******************************")
    if self.verbose: print("******************************")
    if self.verbose: print(layer_sizes)
    if self.verbose: print("Dropout drop probability: {}".format(self._dp_drop_prob))
    if self.verbose: print("Encoder pass:")
    for ind, w in enumerate(self.encode_w):
      if self.verbose:   print(w.data.size())
      if self.verbose:   print(self.encode_b[ind].size())
    if self.verbose: print("Decoder pass:")
    if self.is_constrained:
      if self.verbose:   print('Decoder is constrained')
      for ind, w in enumerate(list(reversed(self.encode_w))):
        if self.verbose:     print(w.transpose(0, 1).size())
        if self.verbose:     print(self.decode_b[ind].size())
    else:
      for ind, w in enumerate(self.decode_w):
        if self.verbose: print(w.data.size())
        if self.verbose: print(self.decode_b[ind].size())
    if self.verbose: print("******************************")
    if self.verbose: print("******************************")


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

def read_data(train_path, impute_value=0, number_of_users=10000, number_of_movies=1000):
    data_pd = pd.read_csv(train_path) 
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    # Create data matrix
    data = np.full((number_of_users , number_of_movies), impute_value)
    for user, movie ,pred in zip(users, movies, predictions): 
        data[user][movie] = pred
    return data 

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

def get_dataloaders(data_dir, split_number, batch_size=32, item_based=False):
    """
    return train and validation and test dataloaders 
    """
    data_matrix = torch.Tensor(read_data(data_dir + f"/data_train.csv")) # inference on the whole matrix 
    train_matrix = torch.Tensor(read_data(data_dir + f"/train_split_{split_number}.csv"))
    val_matrix = torch.Tensor(read_data(data_dir + f"/test_split_{split_number}.csv"))  
    # if item_based, we transpose the matrix 
    if item_based:
        data_matrix = data_matrix.t()
        train_matrix = train_matrix.t()
        val_matrix = val_matrix.t()
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
    train_dataset = torch.utils.data.TensorDataset(data_matrix, train_mask)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(data_matrix, val_mask)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader, data_matrix, val_mask

def clip_data(data, clip_high=5, clip_low=1):
    data[data > clip_high] = clip_high
    data[data < clip_low] = clip_low
    return data

def write_submission(data, submission_file_path, save_path, item_based=False):
    if item_based:
        data = data.T # numpy array 
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

def run_epoch(log_dir, epoch, model, train_dataloader, valid_dataloader, optimizer, scheduler, criterion, min_val_loss, no_improvement_count, verbose=True,print_frequency=30, patience=15):
    if verbose: print(f"Train epoch: {epoch}")

    # training 
    if verbose: print(f"Training...")
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_dataloader):
        x, mask = batch[0].cuda(), batch[1].cuda()
        optimizer.zero_grad()
        z = model(x)
        mse, num_ratings = criterion(z, x, mask)
        #print(mse.size())
        #print(num_ratings.size())
        #print(mask.size())
        loss = mse / num_ratings if num_ratings > 0 else 0
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if epoch % print_frequency == 0: print(f"Train loss epoch {epoch}: {train_loss / len(train_dataloader)}")

    # validation
    if verbose: print(f"Validation...")
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_dataloader):
            x, mask = batch[0].cuda(), batch[1].cuda()
            z = model(x)
            mse, num_ratings = criterion(z, x, mask)
            loss = mse / num_ratings if num_ratings > 0 else 0
            valid_loss += loss.item()
            
    if epoch % print_frequency == 0: print(f"Valid loss epoch {epoch}: {valid_loss / len(valid_dataloader)}")
    if verbose: print(f"Done epoch: {epoch}")

    if valid_loss/len(valid_dataloader) < min_val_loss:
      min_val_loss = valid_loss/len(valid_dataloader)
      torch.save(model.state_dict(), log_dir + f"/best_model.pt")
      print(f"Saved model in epoch {epoch}, val loss {valid_loss/len(valid_dataloader)}")
      no_improvement_count = 0
    else:
      no_improvement_count += 1
      if no_improvement_count > patience:
        print(f"Early stopping at epoch {epoch}")
        return train_loss/len(train_dataloader), valid_loss/len(valid_dataloader), min_val_loss, no_improvement_count, True # early stopping

    scheduler.step() # update learning rate
 

    return train_loss/len(train_dataloader), valid_loss/len(valid_dataloader), min_val_loss, no_improvement_count, False # return train loss, valid loss, min val loss, no improvement count, early stopping


def main():

    nb_folds = 5 
    fold_dir = "5fold" if nb_folds == 5 else "10fold"
    # approach 
    item_based = False
    # settings 
    epochs = 150
    batch_size = 64
    lr = 1e-3 
    weight_decay = 1e-3
    dropout = 0.5
    lr_decay_every = 30 # epochs
    lr_decay = 0.1 # multiplier
    input_size = 10000 if item_based else 1000
    layer_sizes = [input_size,32,32,32]
    submit_unbiased = False 

    model_name = "ItemEncoder" if item_based else "UserEncoder"
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_{model_name}" + f"_bs{batch_size}" + f"_lr{lr}" + f"_wd{weight_decay}" + f"_do{dropout}" + f"_ls{layer_sizes}" + f"_submit_unb_{submit_unbiased}" + f"lr_decay_every{lr_decay_every}" + f"_lr_decay{lr_decay}"
    predictions = None 
    losses = []
    for i in range(nb_folds):
        print(f"----- Split {i} / {nb_folds} -----")
        # Create model directory and Logger
        log_dir = f"reports/logs/{run_id}/{config['model']}_split_{i}"
        if not os.path.exists(log_dir):
          os.makedirs(log_dir)
        # Create model 
        model = AutoEncoder(
          layer_sizes=layer_sizes, 
          nl_type='selu', 
          is_constrained=True, 
          dp_drop_prob=dropout, 
          last_layer_activations=True
          )
        train_loader, valid_loader, data_matrix, valid_mask = get_dataloaders('./data/raw' + f"/{fold_dir}", split_number=i, batch_size=batch_size, item_based=item_based)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_every, gamma=lr_decay)
        criterion = MSEloss

        model.cuda() 
        min_val_loss = np.inf
        no_improvement_count = 0
        for epoch in range(epochs):  
            train_loss, valid_loss, min_val_loss, no_improvement_count, early_stopped = run_epoch(log_dir, epoch, model, train_loader, valid_loader, optimizer, scheduler, criterion, min_val_loss, no_improvement_count, verbose=False)
            if early_stopped:
                break
        # load best validation model 
        model.load_state_dict(torch.load(log_dir + f"/best_model.pt"))

        # run inference on data matrix 
        model.eval()
        with torch.no_grad():
            z = model(data_matrix.cuda())
            model_loss = RMSE(z, data_matrix.cuda(), valid_mask.cuda())
            z = z.cpu().numpy()
            print(f"Model RMSE on validation data: {model_loss}")
            # write validation submission 
            print(f"Writing validation submission...")
            write_submission(z, f"./data/raw/{fold_dir}/test_split_{i}.csv", f"{log_dir}/valid_submission.csv", item_based=item_based)
            print(f"Done.")
            # write test submission
            print(f"Writing test submission...")
            write_submission(z, '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', f"{log_dir}/test_submission.csv", item_based=item_based)
            print(f"Done.")
            if submit_unbiased:
              # remove user biases 
              bias_removed, user_biases = remove_user_bias(data_matrix, z)
              model_unbiased_mse, model_num_ratings = RMSE(bias_removed.cuda(), data_matrix.cuda(), valid_mask.cuda()) 
              model_unbiased_loss = model_unbiased_mse / model_num_ratings
              print(f"Model MSE on test data without user biases: {model_unbiased_mse / model_num_ratings}")
              z = bias_removed.cpu().numpy()
              model_loss = model_unbiased_loss
              write_submission(bias_removed, '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', f"{log_dir}/submission_unbiased.csv", item_based=item_based)
            losses.append(model_loss)
            predictions = z if predictions is None else predictions + z
    
    # write submission
    predictions = predictions / nb_folds
    write_submission(predictions, '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', f'reports/logs/{run_id}/ensemble_submission.csv', item_based=item_based)

    print(f"Finished run")
    # Save losses and avg to csv 
    with open(f'reports/logs/{run_id}' + '/all_losses.csv', 'w') as f:
        writer = csv.writer(f)
        losses = [loss.item() for loss in losses]
        writer.writerow(losses)
        writer.writerow([np.mean(losses)])
        

if __name__ == "__main__":
    main()