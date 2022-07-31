import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import time  
from config import config 
import os 
import csv 


def RMSE(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))

def read_data(train_path, impute_value=0, number_of_users=10000, number_of_movies=1000):
    data_pd = pd.read_csv(train_path) 
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

def write_predictions(preds, submission_file_path, save_path):
    data_pd = pd.read_csv(submission_file_path) 
    test_users, test_movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    with open(save_path, 'w') as f: 
        f.write('Id,Prediction\n')
        print(f"len preds {len(preds)} vs len test {len(test_users)}")
        for i, (user, movie) in enumerate(zip(test_users, test_movies)): 
            # get user row and movie col from data matrix 
            f.write("r{}_c{},{}\n".format(user+1, movie+1, preds[i]))
    return np.array(preds)

class BiasSGD:
    def __init__(self,
                 number_of_users,
                 number_of_movies,
                 log_dir,
                 hidden_size=12,
                 regularization_matrix=0.08,
                 regularization_vector=0.04):

        self.number_of_users = number_of_users
        self.number_of_movies = number_of_movies
        self.log_dir = log_dir
        self.hidden_size = hidden_size
        self.regularization_matrix = regularization_matrix
        self.regularization_vector = regularization_vector

        # initialize matrices U and V
        def initialize(d0, d1, minimum=-0.02, maximum=0.02):
            return np.random.rand(d0, d1) * (maximum - minimum) + minimum

        self.U = initialize(self.number_of_users, self.hidden_size)
        self.V = initialize(self.number_of_movies, self.hidden_size)

        self.biasU = np.zeros(self.number_of_users)
        self.biasV = np.zeros(self.number_of_movies)

    def fit(self,
            train_users,
            train_movies,
            train_ratings,
            valid_users=None,
            valid_movies=None,
            valid_ratings=None,
            num_epochs=5,
            decay=1.5,
            lr=0.05,
            decay_every=5,
            verbose=True,
            patience=20):
        """
        Parameters
        ----------
        train_users             array of user ids for train
        train_movies            array of movie ids for train
        train_ratings           array of ratings for train
        valid_users             array of user ids for validation
        valid_movies            array of movie ids for validation
        valid_ratings           array of ratings for validation
        num_epochs              number of epochs
        decay                   divide the learning rate by this number every requested epochs
        lr                      initial learning rate
        decay_every             number of epoch every which to decay the learning rate
        verbose
        """

        validation = False
        if valid_users is not None and valid_movies is not None and valid_ratings is not None:
            validation = True

        average_rating = np.mean(train_ratings)

        min_val_err = np.inf
        for epoch in range(1, num_epochs + 1):
            users_train_sh, movies_train_sh, ratings_train_sh = shuffle(train_users, train_movies, train_ratings)

            for user, movie, rating in zip(users_train_sh, movies_train_sh, ratings_train_sh):
                U_d = self.U[user, :]
                V_n = self.V[movie, :]

                biasU_d = self.biasU[user]
                biasV_n = self.biasV[movie]

                guess = U_d.dot(V_n) + biasU_d + biasV_n

                delta = rating - guess

                try:
                    new_U_d = U_d + lr * (delta * V_n - self.regularization_matrix * U_d)
                    new_V_n = V_n + lr * (delta * U_d - self.regularization_matrix * V_n)

                    new_biasU_d = biasU_d + lr * (delta - self.regularization_vector * (biasU_d + biasV_n - average_rating))
                    new_biasV_n = biasV_n + lr * (delta - self.regularization_vector * (biasV_n + biasU_d - average_rating))

                except FloatingPointError:
                    continue
                else:
                    self.U[user, :] = new_U_d
                    self.V[movie, :] = new_V_n

                    self.biasU[user] = new_biasU_d
                    self.biasV[movie] = new_biasV_n

            if validation and verbose:
                predictions = self.predict(valid_users, valid_movies)
                val_err = RMSE(valid_ratings, predictions)
                print('Validation error at epoch', epoch, 'is', val_err)
                if val_err < min_val_err:
                    min_val_err = val_err
                    self.save(self.log_dir)
                    patience = 30
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping at epoch', epoch)
                        return 

            if epoch % decay_every == 0:
                lr /= decay

    def predict(self, test_users, test_movies):

        predictions = list()

        for user, movie in zip(test_users, test_movies):
            predictions.append(self.U[user, :].dot(self.V[movie, :]) + self.biasU[user] + self.biasV[movie])

        return np.array(predictions)

    def save(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        np.save(os.path.join(log_dir, 'U.npy'), self.U)
        np.save(os.path.join(log_dir, 'V.npy'), self.V)
        np.save(os.path.join(log_dir, 'biasU.npy'), self.biasU)
        np.save(os.path.join(log_dir, 'biasV.npy'), self.biasV)
        print('Model saved')

    def load(self, log_dir):
        self.U = np.load(os.path.join(log_dir, 'U.npy'))
        self.V = np.load(os.path.join(log_dir, 'V.npy'))
        self.biasU = np.load(os.path.join(log_dir, 'biasU.npy'))
        self.biasV = np.load(os.path.join(log_dir, 'biasV.npy'))
        print('Model loaded')


def main():

    nb_folds = 5
    fold_dir = '5fold' if nb_folds == 5 else '10fold'
    data_path = './data/raw'
    lr = 1e-2
    decay_factor = 5
    epochs = 1
    decay_every = 75
    hidden_size = 128

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"_BSGD"  + f"_lr{lr}" + f"_decay{decay_factor}" + f"_epochs{epochs}" + f"_decay_every{decay_every}" + f"_hidden_size{hidden_size}"

    losses = list()
    test_predictions = None 
    for i in range(nb_folds): # over all splits 
        print(f"------------------------------------------------------")
        print(f"Running split {i+1} / {nb_folds}")

        log_dir = f"reports/logs/{run_id}/split_{i}"
        if not os.path.exists(log_dir):
          os.makedirs(log_dir)

        model = BiasSGD(
            number_of_users=10000, 
            number_of_movies=1000,
            log_dir=log_dir,
            hidden_size=hidden_size
            )

        train_users, train_movies, train_ratings = read_data(data_path + f"/{fold_dir}" + '/train_split_' + str(i) + '.csv')
        valid_users, valid_movies, valid_ratings = read_data(data_path + f"/{fold_dir}" + '/test_split_' + str(i) + '.csv')

        model.fit(
            train_users, 
            train_movies, 
            train_ratings,
            valid_users=valid_users,
            valid_movies=valid_movies,
            valid_ratings=valid_ratings,
            num_epochs=epochs,
            decay=decay_factor, 
            lr=lr,
            decay_every=decay_every, 
            verbose=True
            )
        
        # load best validation error model
        model.load(log_dir)
        # predict on validation set 
        preds_bSGD_val = model.predict(valid_users, valid_movies)
        print('Validation RMSE on split', i+1, 'is', RMSE(valid_ratings, preds_bSGD_val))
        losses.append(RMSE(valid_ratings, preds_bSGD_val))
        # save validation prediction 
        write_predictions(preds_bSGD_val, data_path + f"/{fold_dir}" + '/test_split_' + str(i) + '.csv', f"{log_dir}/val_submission.csv")
        
        # predict on test set
        test_users, test_movies, test_ratings = read_data('/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv') 
        preds_bSGD_test = model.predict(test_users, test_movies)
        if test_predictions is None:
            test_predictions = preds_bSGD_test
        else:
            test_predictions += preds_bSGD_test
        # evaluate mse on validation set
        #write_predictions(preds_bSGD_test, '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', f"{log_dir}/test_submission.csv")
        print(f"------------------------------------------------------")   
    
    print(f"Finished run")
    # Save losses and avg to csv 
    with open(f'reports/logs/{run_id}' + '/all_losses.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(losses)
        writer.writerow([np.mean(losses)])

    # average predictions on all splits 
    test_predictions /= nb_folds
    write_predictions(test_predictions, '/home/ubuntu/projects/CILProject22/data/submission/sampleSubmission.csv', f"reports/logs/{run_id}/avg_submission.csv")

if __name__ == "__main__":
    main()