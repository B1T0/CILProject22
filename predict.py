import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch
from src.models.CSE.model import Model
from src.models.CSE_prediction.model import Prediction
from torch.utils.data import TensorDataset
import os
import time
from src.data.Graph.graph_datamodule import Graph_Dataset
from torch.utils.data import DataLoader

print(torch.cuda.device_count())

path = '/home/jimmy/CILProject22/data/external/sampleSubmission.csv'
# model_path = '/home/jimmy/CILProject22/reports/logs/20220710-164447_finetuning/model_best.pth'
#model_path = '/home/jimmy/CILProject22/reports/logs/20220714-232851_finetuning/model_best.pth'
model_path = '/home/jimmy/CILProject22/reports/logs/20220725-230933_finetuning/model_best.pth'
EPOCH = 50
bs = 16
n_users = 1000


def main():
    pretrained = Model()
    model = Prediction(pretrained)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Moving model to cuda')
    model = model.to('cuda:0')
    optimizer = model.configure_optimizers()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    print(optimizer)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"/home/jimmy/CILProject22/reports/logs/{run_id}_prediction"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Create logging file

    print('Creating Dataloaders')

    model.eval()
    indices_i = []
    indices_j = []
    df = pd.read_csv(path)
    for i, x in df.iterrows():
        name, _ = x['Id'], x['Prediction']
        movie, user = name.replace('c', '').replace('r', '').split('_')
        movie, user = int(movie) - 1, int(user) - 1
        indices_i.append(user)
        indices_j.append(movie + n_users)

    users = torch.tensor(indices_i).to('cuda:0')
    items = torch.tensor(indices_j).to('cuda:0')
    dataset = TensorDataset(users, items)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
    idx, predictions = [], []
    print('Beginning Prediction')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            user, item = batch
            prediction = model.forward(user, item)
            user_idx = user + 1
            item_idx = item - n_users + 1
            prediction = prediction.cpu().numpy()
            for i in range(len(user_idx)):
                idx.append(f'r{item_idx[i]}_c{user_idx[i]}')
                predictions.append(int(prediction[i]))

    df = pd.DataFrame({'Id': idx, 'Prediction': predictions})
    df.to_csv(log_dir + '/submission_pretrained_mse_pc.csv', index=False)


if __name__ == "__main__":
    main()
