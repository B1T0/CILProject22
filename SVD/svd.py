import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
import pandas as pd
from tqdm.notebook import tqdm


def load_rating_df(path) -> pd.DataFrame:
    """Load the rating dataframe from the given path."""
    df = pd.read_csv(path)
    rating_df = df.Id.str.split("_", expand=True)
    rating_df.columns = ["user", "movie"]
    rating_df["user"] = rating_df["user"].str.lstrip("r").astype(int) - 1
    rating_df["movie"] = rating_df["movie"].str.lstrip("c").astype(int) - 1
    rating_df["rating"] = df.Prediction
    return rating_df


def create_rating_matrix(rating_df) -> np.ndarray:
    """Compute the rating matrix based on the given rating dataframe"""
    rating_matrix = np.empty((10000, 1000))
    rating_matrix.fill(np.nan)
    rating_matrix[
        rating_df.user.values, rating_df.movie.values
    ] = rating_df.rating.values
    return rating_matrix


rmses = []
for split in range(10):
    rating_df = load_rating_df(f"../data/train_split_{split}.csv")
    # Hyperparameter tuning
    KF = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in tqdm(KF.split(rating_df), total=10, leave=False):
        X_train = create_rating_matrix(rating_df.iloc[train_index])
        X_test = create_rating_matrix(rating_df.iloc[test_index])
        # standardize by column
        mu = np.nanmean(X_train, axis=1)
        sigma = np.nanstd(X_train, axis=1)
        X_train = (X_train - mu[:, np.newaxis]) / sigma[:, np.newaxis]
        # fillna
        X_train[np.isnan(X_train)] = 0
        # compute SVD on the training data
        U, s, V = np.linalg.svd(X_train, full_matrices=False)
        # iterate over the number of used components
        best_k, best_rmse = 0, np.inf
        for k in range(15):
            # compute the predicted ratings
            X_hat = np.dot(U[:, :k], np.dot(np.diag(s[:k]), V[:k, :]))
            X_hat = X_hat * sigma[:, np.newaxis] + mu[:, np.newaxis]
            # compute the RMSE
            rmse = np.sqrt(np.nanmean((X_hat - X_test) ** 2))
            # # print rmse
            # print(f"k={k}, rmse={rmse}")
            # update the best k and best RMSE
            if rmse < best_rmse:
                best_k, best_rmse = k, rmse
            if k > best_k + 3:
                break
        # print(f"Best k: {best_k}")
        # print(f"Best RMSE: {best_rmse}")
    # Compute split score
    X_train = create_rating_matrix(rating_df.iloc[:])
    rating_df = load_rating_df(f"../data/test_split_{split}.csv")
    X_test = create_rating_matrix(rating_df.iloc[:])
    # standardize by column
    mu = np.nanmean(X_train, axis=1)
    sigma = np.nanstd(X_train, axis=1)
    X_train = (X_train - mu[:, np.newaxis]) / sigma[:, np.newaxis]
    # fillna
    X_train[np.isnan(X_train)] = 0
    # compute SVD on the training data
    U, s, V = np.linalg.svd(X_train, full_matrices=False)
    X_hat = np.dot(U[:, :best_k], np.dot(np.diag(s[:best_k]), V[:best_k, :]))
    X_hat = X_hat * sigma[:, np.newaxis] + mu[:, np.newaxis]
    # compute the RMSE
    rmse = np.sqrt(np.nanmean((X_hat - X_test) ** 2))
    print(f"Split {split}: RMSE={rmse}, k={best_k}")
    rmses.append(rmse)


print(f"Mean RMSE: {np.mean(rmses)}")

### Prepare Submission

best_k = 7

rating_df = load_rating_df(f"../data/data_train.csv")
X_train = create_rating_matrix(rating_df.iloc[:])
# standardize by column
mu = np.nanmean(X_train, axis=1)
sigma = np.nanstd(X_train, axis=1)
X_train = (X_train - mu[:, np.newaxis]) / sigma[:, np.newaxis]
# fillna
X_train[np.isnan(X_train)] = 0
# compute SVD on the training data
U, s, V = np.linalg.svd(X_train, full_matrices=False)
X_hat = np.dot(U[:, :best_k], np.dot(np.diag(s[:best_k]), V[:best_k, :]))
X_hat = X_hat * sigma[:, np.newaxis] + mu[:, np.newaxis]

sub_df = load_rating_df("../data/sampleSubmission.csv")
sub_y_hat = np.clip(X_hat[sub_df.user, sub_df.movie], 1, 5)
submission = pd.read_csv("../data/sampleSubmission.csv")
submission["Prediction"] = sub_y_hat
submission.to_csv(f"../data/submission_svd.csv", index=False)
