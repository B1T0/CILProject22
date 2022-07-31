import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from src.models.NCF.model import NCF


'''
Load model from previous training and do predictions, which are averaged over the 5 splits (Ensemble).
'''

split_numbers = 5
predicting = False
avg_pred = torch.Tensor(np.zeros(1176952)).cuda()

sample_sub = pd.read_csv("../../../data/raw/sampleSubmission.csv")
to_predict = torch.Tensor([(int(str(x).partition("_")[0][1:]),int(str(x).partition("_")[2][1:])) for x in sample_sub['Id']]).cuda()
user_id, item_id = torch.transpose(to_predict, 0, 1)[0].long(), torch.transpose(to_predict, 0, 1)[1].long()

for split_number in range(split_numbers):

    if predicting:
        # Load the model
        print(f'Loading the model_{split_number}...')
        model = NCF().cuda()
        model.load_state_dict(torch.load(f'./best_model_{split_number}.pth'))
        model.eval()

        # Predict
        print('Predicting...')
        output = softmax(model(user_id, item_id), dim=1)

        print('computing predictions...')
        output_scalars = torch.zeros(output.shape[0]).cuda()
        print(output_scalars.shape)
        for i, pred in enumerate(output):
            if i % 100000 == 0:
                print(i)
            for j, p in enumerate(pred):
                output_scalars[i] += p * (j+1)

        # avg_pred += output_scalars
        sample_sub.Prediction = output_scalars.cpu().detach().numpy()
        sample_sub.to_csv(f"../../../data/NCF_{split_number}.csv", index=False)

    else:
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