import pandas as pd
from tqdm import tqdm
import torch



def predict_graph_user_encoder(log_dir, args, model, dataloader, split=0):
    """
     prediction function for Graph User Encoder
    first creates full rating matrix before subsampling the given submission entries
    :param log_dir: directory to save file to
    :param model: model
    :param args: -
    :param dataloader: dataloader for submission indices
    :param split: # for name
    """
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
