import pandas as pd
from tqdm import tqdm
import torch

print(torch.cuda.device_count())

# path = '/home/jimmy/CILProject22/data/external/sampleSubmission.csv'
# path = '/home/jimmy/CILProject22/data/raw/data_train.csv'
path = '/home/jimmy/CILProject22/data/raw/train_split_'
model_dir = '/home/jimmy/CILProject22/reports/logs/20220728-215328_graphautoencoder_16_user_mode'
graph_paths = '/home/jimmy/CILProject22/data/raw/train_split_'
EPOCH = 50
bs = 128
SPLIT = 5

MODE = 'user_mode'
EMBEDDING_DIM = 16


def predict_graph_user_encoder(log_dir, args, model, dataloader, split=0):
    print(f'Prediction split {split}')

    MODE = args['train_mode']
    idx_split, predictions_split = [], []
    with torch.no_grad():
        if MODE == 'user_mode':
            i = 10000
        elif MODE == 'movie_mode':
            i = 1000

        rating_matrix = model.forward(torch.arange(0, i))
        print(rating_matrix.size())
        # user must be first dimension
        if MODE == 'user_mode':
            rating_matrix = rating_matrix.transpose(0, 1)
        print(rating_matrix.size())

        for batch in tqdm(dataloader):
            user, movie = batch
            # print(user)
            prediction = rating_matrix[movie, user]
            user_idx = user + 1
            item_idx = movie + 1
            prediction = prediction.cpu().numpy()
            for i in range(len(user_idx)):
                idx_split.append(f'r{user_idx[i]}_c{item_idx[i]}')
                predictions_split.append(prediction[i])

        df = pd.DataFrame({'Id': idx_split, 'Prediction': predictions_split})
        df.to_csv(log_dir + f'/submission_{split}_train.csv', index=False)


if __name__ == "__main__":
    main()
