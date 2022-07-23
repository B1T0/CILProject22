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

# Load the data
df_full = pd.read_csv('/home/dustin/Documents/Study/Master2/CILProject22/data_raw/data_train.csv')
df_train = pd.read_csv('/home/dustin/Documents/Study/Master2/CILProject22/data_raw/cross_validation/train_split_4.csv')
df_test = pd.read_csv('/home/dustin/Documents/Study/Master2/CILProject22/data_raw/cross_validation/test_split_4.csv')

print('Loading the data...')
dic_full = {
    'X': [(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in df_full['Id']],
    'y': [float(x) for x in df_full['Prediction']],
}
dic_train = {
    'X': [(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in df_train['Id']],
    'y': [float(x) for x in df_train['Prediction']],
}
dic_test = {
    'X': [(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in df_test['Id']],
    'y': [float(x) for x in df_test['Prediction']],
}


print('Normalizing the data...')
n_users = 10000
n_items = 1000

print(np.asarray(dic_train['X']).shape, np.asarray(dic_train['y']).shape)

train_dataset = TensorDataset(torch.Tensor(dic_train['X']), torch.Tensor(dic_train['y']))
test_dataset = TensorDataset(torch.Tensor(dic_test['X']), torch.Tensor(dic_test['y']))

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Define the model
class NCF(nn.Module):

    def __init__(self):
        super(NCF, self).__init__()
        self.latent_dim = 32
        self.dropout = 0.25
        self.embedding_user = torch.nn.Embedding(num_embeddings=10000, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=1000, embedding_dim=self.latent_dim)
        # self.user_1 = nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim, dropout=self.dropout),
        #                             nn.ReLU(),
        #                             nn.BatchNorm1d(self.latent_dim),)
        # self.item_1 = nn.Sequential(nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim, dropout=self.dropout),
        #                             nn.ReLU(),
        #                             nn.BatchNorm1d(self.latent_dim),)
        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features=3*self.latent_dim, out_features=128),
                                    torch.nn.Dropout(self.dropout),
                                    torch.nn.ReLU(),
                                    #torch.nn.BatchNorm1d(128),
                                    torch.nn.Linear(in_features=128, out_features=128),
                                    torch.nn.Dropout(self.dropout),
                                    torch.nn.ReLU(),
                                    #torch.nn.BatchNorm1d(128),
                                    torch.nn.Linear(in_features=128, out_features=20),
                                    torch.nn.Dropout(self.dropout),
                                    torch.nn.ReLU(),
                                    #torch.nn.BatchNorm1d(20),
                                    torch.nn.Linear(in_features=20, out_features=5),
                                    )


    def forward(self, user_id, item_id):
        user_embedding = self.embedding_user(user_id.long()-1)
        item_embedding = self.embedding_item(item_id.long()-1)
        x = torch.cat((user_embedding, item_embedding, torch.mul(user_embedding, item_embedding)), dim=1)
        x = self.dense(x)
        return x


# Define the training loop
def train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, weight_decay=0.0001):
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Define the metrics
    metrics = {'train_ce': [], 'test_ce': [], 'train_rmse': [], 'test_rmse': []}
    # Train the model
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        # Training
        model.train()
        for i, (ids, rating) in enumerate(train_dataloader):
            user_id, item_id = torch.transpose(ids, 0, 1)[0], torch.transpose(ids, 0, 1)[1]
            # Forward pass
            output = model(user_id, item_id)
            # Compute loss
            # ratings_one_hot = one_hot_encode(rating)
            loss = loss_fn(output, rating.long()-1)
            # Backward pass
            optimizer.zero_grad()
            # Update weights
            loss.backward()
            optimizer.step()
            # Update metrics
            rmse_loss = rmse(softmax(output), rating.long())
            metrics['train_ce'].append(loss.item())
            metrics['train_rmse'].append(rmse_loss)
            if i % 5000 == 0:
                print('\tBatch {}/{}'.format(i+1, len(train_dataloader)))
                print('\t\tLoss: {}'.format(loss.item()))
                print('\t\tRMSE: {}'.format(rmse_loss))
        # Validation
        model.eval()
        with torch.no_grad():
            for i, (ids, rating) in enumerate(test_dataloader):
                user_id, item_id = torch.transpose(ids, 0, 1)[0], torch.transpose(ids, 0, 1)[1]
                # Forward pass
                output = model(user_id, item_id)
                # Compute loss
                # ratings_one_hot = one_hot_encode(rating)
                loss = loss_fn(output, rating.long()-1)
                # Update metrics
                rmse_loss = rmse(softmax(output), rating.long())
                metrics['test_ce'].append(loss.item())
                metrics['test_rmse'].append(rmse_loss)
                if i % 5000 == 0:
                    print('\tBatch {}/{}'.format(i+1, len(test_dataloader)))
                    print('\t\tLoss: {}'.format(loss.item()))
                    print('\t\tRMSE: {}'.format(rmse_loss))
        # Print metrics
        print('Epoch: {}, Train CE: {:.4f}, Test CE: {:.4f}, Train_RSME: {:.4f}, Test_RMSE: {:.4f}'.format(epoch, np.mean(metrics['train_ce']), np.mean(metrics['test_ce']), np.mean(metrics['train_rmse']), np.mean(metrics['test_rmse'])))
    return metrics

def one_hot_encode(nums, num_classes=5):
    one_hots = torch.zeros(nums.shape[0], num_classes)
    for i, num in enumerate(nums.long()):
        one_hots[i, num-1] = 1
    return one_hots

def rmse(preds, true):
    # one hot to index
    pred_scalar = torch.zeros(preds.shape[0])
    for i, pred in enumerate(preds):
        for j, p in enumerate(pred):
            pred_scalar[i] += p * (j+1)
    return torch.sqrt(torch.mean((pred_scalar-true)**2))

# Prediction
def predict(model, user_id, item_id):
    with torch.no_grad():
        output = model(user_id, item_id)
        return output.item()


print('Training the model...')
model = NCF()
metrics = train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, weight_decay=0.0001)