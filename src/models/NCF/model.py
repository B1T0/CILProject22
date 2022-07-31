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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

# Load the data
split_number = 4
df_full = pd.read_csv('/home/ubuntu/CILProject22/data/raw/data_train.csv')
df_train = pd.read_csv(f'/home/ubuntu/CILProject22/data/raw/cross_validation/train_split_{split_number}.csv')
df_test = pd.read_csv(f'/home/ubuntu/CILProject22/data/raw/cross_validation/test_split_{split_number}.csv')

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


n_users = 10000
n_items = 1000

print(np.asarray(dic_train['X']).shape, np.asarray(dic_train['y']).shape)
x_train = torch.Tensor(dic_train['X'])
print(x_train.is_cuda)
x_train.cuda()
print(x_train.is_cuda)

train_dataset = TensorDataset(torch.Tensor(dic_train['X']).cuda(), torch.Tensor(dic_train['y']).cuda())
test_dataset = TensorDataset(torch.Tensor(dic_test['X']).cuda(), torch.Tensor(dic_test['y']).cuda())

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
def train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, weight_decay=0.0001, early_stopping=True, patience=5):
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
        print('Epoch {}/{}'.format(epoch+1, epochs))
        # Training
        model.train()
        for i, (ids, rating) in enumerate(train_dataloader):
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
            if i % 5000 == 0:
                print('\tBatch {}/{}'.format(i+1, len(train_dataloader)))
                print('\t\tLoss: {}'.format(loss.item()))
                print('\t\tRMSE: {}'.format(np.mean(metrics['train_rmse_epoch'])))
                metrics['train_rmse_epoch'] = []
        # Validation
        model.eval()
        with torch.no_grad():
            for i, (ids, rating) in enumerate(test_dataloader):
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
                if i % 5000 == 0:
                    print('\tBatch {}/{}'.format(i+1, len(test_dataloader)))
                    print('\t\tLoss: {}'.format(loss.item()))
                    print('\t\tRMSE: {}'.format(np.mean(metrics['test_rmse_epoch'])))
                    metrics['test_rmse_epoch'] = []
        # Print metrics
        print('Epoch: {}, Train CE: {:.4f}, Test CE: {:.4f}, Train_RSME: {:.4f}, Test_RMSE: {:.4f}'.format(epoch+1, np.mean(metrics['train_ce']), np.mean(metrics['test_ce']), np.mean(metrics['train_rmse']), np.mean(metrics['test_rmse'])))
        # Keep track of the best model and do
        if best_val_rsme > np.mean(metrics['test_rmse']):
            best_val_rsme = np.mean(metrics['test_rmse'])
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            print('Best epoch: {}, Best RMSE: {:.4f}'.format(best_epoch+1, best_val_rsme))
            torch.save(model.state_dict(), './best_model.pth')
        if best_epoch <= epoch - patience:
            print('Early stopping')
            print('Best epoch: {}, Best RMSE: {:.4f}'.format(best_epoch+1, best_val_rsme))
            return best_model, metrics
    return model, metrics

def rmse(preds, true):
    # one hot to index
    pred_scalar = torch.zeros(preds.shape[0]).cuda()
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
model.cuda()
model, metrics = train(model, train_dataloader, test_dataloader, epochs=10, lr=0.001, weight_decay=0.0001)
print('Training complete.')

# Save the model
torch.save(model.state_dict(), './model_afterTraining.pt')

# # Load the model
# model = NCF()
# model.load_state_dict(torch.load('./best_model.pt'))

# Predict
print('Predicting...')

sample_sub = pd.read_csv("/home/ubuntu/CILProject22/data/raw/sampleSubmission.csv")

to_predict = torch.Tensor([(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in sample_sub['Id']]).cuda()
user_id, item_id = torch.transpose(to_predict, 0, 1)[0].long(), torch.transpose(to_predict, 0, 1)[1].long()
output = model(user_id, item_id)

output_scalars = torch.zeros(output.shape[0]).cuda()
for i, pred in enumerate(softmax(output, dim=1)):
    for j, p in enumerate(pred):
        output_scalars[i] += p * (j+1)
    # if pred[4] > 0.8:
    #     output_scalars[i] = 5

sample_sub.Prediction[:100] = output_scalars.cpu().detach().numpy()
sample_sub.to_csv(f"/home/ubuntu/CILProject22/data/NCF_{split_number}.csv", index=False)
print('Done!')
