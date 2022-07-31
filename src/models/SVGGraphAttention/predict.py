import torch
from tqdm import tqdm


def predict_svd_graph_attention(log_dir, model, args, dataloader, split=0):

    print(f'Prediction split {split}')
    idx_split, predictions_split = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            user, movie = batch
            prediction = model.forward(movie, user)
            user_idx = user + 1
            item_idx = movie + 1
            prediction = prediction.cpu().numpy()
            for i in range(len(user_idx)):
                idx_split.append(f'r{user_idx[i]}_c{item_idx[i]}')
                predictions_split.append(prediction[i])

        df = pd.DataFrame({'Id': idx_split, 'Prediction': predictions_split})
        df.to_csv(log_dir + f'/train_split_{split}.csv', index=False)