import numpy as np
import pandas as pd
from subprocess import call
from sklearn.metrics import mean_squared_error
import math
import torch
from torch import dropout, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax
import copy
from time import sleep
from model import NCF, rmse, predict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Is Cuda available: ', torch.cuda.is_available())

# Hyperparameters
split_numbers = 5
n_users = 10000
n_items = 1000
epochen = 100
patience_ = 20
learning_rate = 5*1e-4
weight_decay_ = 1*1e-4
batch_size_ = 128
# ideas: layers, embeddings, dropout, learning_rate, batch_size, optimizer

print('Loading Submission Data...')
sample_sub = pd.read_csv("../../../data/raw/sampleSubmission.csv")
to_predict = torch.Tensor([(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in sample_sub['Id']]).cuda()
user_id_sub, item_id_sub = torch.transpose(to_predict, 0, 1)[0].long(), torch.transpose(to_predict, 0, 1)[1].long()

print(f'Hyperparameters: "split_numbers": {split_numbers}, "n_users": {n_users}, "n_items": {n_items}, "epochen": {epochen}, "patience_": {patience_}, "learning_rate": {learning_rate}, "weight_decay_": {weight_decay_}')

# Define the training loop
def train(model, train_dataloader, test_dataloader, epochs, lr, weight_decay, patience):
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Define the metrics
    metrics = {'train_ce': [], 'test_ce': [], 'train_rmse': [], 'test_rmse': [], 'train_rmse_epoch': [], 'test_rmse_epoch': []}
    best_val_rsme = 100
    best_epoch = 0
    # Train the model
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch+1, epochs))
        # Training
        model.train()
        for i, (ids, rating) in enumerate(train_dataloader):
            if i == 0:
                metrics['train_rmse_epoch'] = []
            user_id, item_id = torch.transpose(ids, 0, 1)[0], torch.transpose(ids, 0, 1)[1]
            # Forward pass
            output = model(user_id, item_id)
            # Compute loss
            loss = loss_fn(output, rating.long()-1)
            # Backward pass
            optimizer.zero_grad()
            # Update weights
            loss.backward()
            optimizer.step()
            # Update metrics
            rmse_loss = rmse(softmax(output, dim=1), rating.long()).cpu().detach().numpy()
            metrics['train_ce'].append(loss.item())
            metrics['train_rmse'].append(rmse_loss)
            metrics['train_rmse_epoch'].append(rmse_loss)
            if i % 2000 == 0:
                print('\tBatch {}/{}'.format(i+1, len(train_dataloader)))
                print('\t\tLoss: {}'.format(loss.item()))
                print('\t\tRMSE: {}'.format(np.mean(metrics['train_rmse_epoch'])))
        # Validation
        model.eval()
        with torch.no_grad():
            for i, (ids, rating) in enumerate(test_dataloader):
                if i == 0:
                    metrics['test_rmse_epoch'] = []
                user_id, item_id = torch.transpose(ids, 0, 1)[0], torch.transpose(ids, 0, 1)[1]
                # Forward pass
                output = model(user_id, item_id)
                # Compute loss
                loss = loss_fn(output, rating.long()-1)
                # Update metrics
                rmse_loss = rmse(softmax(output, dim=1), rating.long()).cpu().detach().numpy()
                metrics['test_ce'].append(loss.item())
                metrics['test_rmse'].append(rmse_loss)
                metrics['test_rmse_epoch'].append(rmse_loss)
                if i % 500 == 0:
                    print('\tBatch {}/{}'.format(i+1, len(test_dataloader)))
                    print('\t\tLoss: {}'.format(loss.item()))
                    print('\t\tRMSE: {}'.format(np.mean(metrics['test_rmse_epoch'])))
        # Print metrics
        print('Epoch: {}, Train CE: {:.4f}, Test CE: {:.4f}, Train_RSME: {:.4f}, Test_RMSE: {:.4f}'.format(epoch+1, np.mean(metrics['train_ce']), np.mean(metrics['test_ce']), np.mean(metrics['train_rmse_epoch']), np.mean(metrics['test_rmse_epoch'])))
        # Keep track of the best model and do
        if best_val_rsme > np.mean(metrics['test_rmse_epoch']):
            best_val_rsme = np.mean(metrics['test_rmse_epoch'])
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            print('Best epoch: {}, Best RMSE: {:.4f}'.format(best_epoch+1, best_val_rsme))
        if best_epoch <= epoch - patience:
            print('Early stopping')
            print('Best epoch: {}, Best RMSE: {:.4f}'.format(best_epoch+1, best_val_rsme))
            torch.save(best_model.state_dict(), f'./best_model_{split_number}.pth')
            return best_model, metrics
    return model, metrics

for split_number in range(split_numbers):
    print(f'Split number: {split_number}')

    df_train = pd.read_csv(f'../../../data/raw/cross_validation/train_split_{split_number}.csv')
    df_test = pd.read_csv(f'../../../data/raw/cross_validation/test_split_{split_number}.csv')

    print('Loading the data...')
    dic_train = {
        'X': [(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in df_train['Id']],
        'y': [float(x) for x in df_train['Prediction']],
    }
    dic_test = {
        'X': [(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in df_test['Id']],
        'y': [float(x) for x in df_test['Prediction']],
    }


    print(np.asarray(dic_train['X']).shape, np.asarray(dic_train['y']).shape)

    train_dataset = TensorDataset(torch.Tensor(dic_train['X']).cuda(), torch.Tensor(dic_train['y']).cuda())
    test_dataset = TensorDataset(torch.Tensor(dic_test['X']).cuda(), torch.Tensor(dic_test['y']).cuda())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_, shuffle=True)


    print('Training the model...')
    model = NCF().cuda()
    model, _ = train(model, train_dataloader, test_dataloader, epochs=epochen, lr=learning_rate, weight_decay=weight_decay_, patience=patience_)
    print('Training complete.')

    # # Load the model
    # print('Loading the model...')
    # model = NCF()
    # model.load_state_dict(torch.load(f'./best_model_{split_number}.pth'))

    # Predict
    print('Predicting...')

    output = model(user_id_sub, item_id_sub)

    output_scalars = torch.zeros(output.shape[0]).cuda()
    for i, pred in enumerate(softmax(output, dim=1)):
        if i % 100000 == 0:
            print(i)
        for j, p in enumerate(pred):
            output_scalars[i] += p * (j+1)

    sample_sub.Prediction = output_scalars.cpu().detach().numpy()
    sample_sub.to_csv(f"../../../data/NCF_{split_number}.csv", index=False)

    print('Done!')



'''
average prediction over the 5 splits (Ensemble).
'''

split_numbers = 5
avg_pred = torch.Tensor(np.zeros(1176952)).cuda()

sample_sub = pd.read_csv("../../../data/raw/sampleSubmission.csv")
to_predict = torch.Tensor([(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in sample_sub['Id']]).cuda()
user_id, item_id = torch.transpose(to_predict, 0, 1)[0].long(), torch.transpose(to_predict, 0, 1)[1].long()

for split_number in range(split_numbers):
    # load predictions
    print(f'Loading predictions_{split_number}...')
    preds = pd.read_csv(f"../../../data/NCF_{split_number}.csv")
    avg_pred += torch.Tensor(preds['Prediction'].values).cuda()
    print('Done!')

print('Average predictions...')
avg_pred /= split_numbers
sample_sub.Prediction = avg_pred.cpu().detach().numpy()
sample_sub.to_csv(f"../../../data/NCF_avg.csv", index=False)
print('Done!')