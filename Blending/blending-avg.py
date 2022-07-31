from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from tqdm.notebook import tqdm

### Data

# Test split data
split_pred_dfs = []
for split in range(5):
    split_pred_dfs.append(
        pd.read_csv(f"../data/cross_validation/test_split_{split}.csv", index_col=0)
    )

y_df = pd.concat(split_pred_dfs, axis=0, keys=[f"split_{i}" for i in range(5)])

# prediction data
pred_path = Path("../data/predictions/blending")
model_folders = [
    pred_path / "bsgd",
    pred_path / "graph_ae",
    pred_path / "graph_ue",
    pred_path / "item_encoder",
    pred_path / "mot_init_dot",
    pred_path / "ncf",
    pred_path / "user_ae",
]

model_pred_dfs = []
for model_f in model_folders:
    model_preds = []
    for split in range(5):
        model_preds.append(pd.read_csv(model_f / f"split{split}.csv", index_col=0))
    model_preds = pd.concat(model_preds, axis=0, keys=[f"split_{i}" for i in range(5)])
    model_pred_dfs.append(model_preds)

X_df = pd.concat(model_pred_dfs, axis=1)
X_df.columns = [p.name for p in model_folders]
X_df.graph_ae = X_df.graph_ae.str.lstrip("[").str.rstrip("]").astype(float)


for col in X_df.columns:
    rmses = 0
    for i in range(5):
        rmse = np.sqrt(
            np.mean(
                (y_df.Prediction[f"split_{i}"].values - X_df[col][f"split_{i}"].values)
                ** 2
            )
        )
        rmses += rmse
        # print(f"{col, i} - RMSE: {rmse}")
    print(f"{col} - RMSES: {rmses / 5}")

### Training


# for i in tqdm(range(5), leave=False):
#     tree = DecisionTreeRegressor(random_state=42)

#     X_train = X_df.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#     y_train = y_df.Prediction.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#     tree.fit(X_train, y_train)

#     X_test = X_df.loc[f"split_{i}", :].values
#     y_test = y_df.Prediction.loc[f"split_{i}", :].values
#     y_hat = tree.predict(X_test)
#     rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))
#     print(f"RMSE: {rmse}")


# for d in range(5, 10):
#     for i in tqdm(range(5), leave=False):
#         if i == 0:
#             print(f"max_depth={d}")
#         tree = DecisionTreeRegressor(random_state=42, max_depth=d)

#         # print("Fitting tree...")
#         X_train = X_df.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#         y_train = y_df.Prediction.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#         tree.fit(X_train, y_train)

#         # print("Testing tree...")
#         X_test = X_df.loc[f"split_{i}", :].values
#         y_test = y_df.Prediction.loc[f"split_{i}", :].values
#         y_hat = tree.predict(X_test)
#         rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))
#         print(f"RMSE: {rmse}")


# for nexp in range(7, 15):
#     n = 2 ** nexp
#     for i in tqdm(range(5), leave=False):
#         if i == 0:
#             print(f"n_estimators={n}")
#         tree = RandomForestRegressor(random_state=42, n_estimators=n)

#         # print("Fitting tree...")
#         X_train = X_df.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#         y_train = y_df.Prediction.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#         tree.fit(X_train, y_train)

#         # print("Testing tree...")
#         X_test = X_df.loc[f"split_{i}", :].values
#         y_test = y_df.Prediction.loc[f"split_{i}", :].values
#         y_hat = tree.predict(X_test)
#         rmse = np.sqrt(np.mean((y_test - y_hat) ** 2))
#         print(f"RMSE: {rmse}")


# # Note: hyperparameter tuning is not necessary for this model as almost no impact on RMSE
# for i in tqdm(range(5), leave=False):
#     reg = Lasso(alpha=0.01, positive=True, random_state=42)

#     X_train = X_df.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#     y_train = y_df.Prediction.loc[[f"split_{j}" for j in range(5) if j != i], :].values
#     reg.fit(X_train, y_train)

#     X_test = X_df.loc[f"split_{i}", :].values
#     y_test = y_df.Prediction.loc[f"split_{i}", :].values
#     y_hat = reg.predict(X_test)
#     rmse = np.sqrt(np.mean((y_test - np.clip(y_hat, 1, 5)) ** 2))
#     print(f"RMSE: {rmse}")

# for model, coef in zip(model_folders, reg.coef_):
#     print(f"{model.name} - {coef}")


# Note: hyperparameter tuning is not necessary for this model as almost no impact on RMSE
for i in tqdm(range(5), leave=False):
    reg = Ridge(alpha=0.01, positive=True, random_state=42)

    X_train = X_df.loc[[f"split_{j}" for j in range(5) if j != i], :].values
    y_train = y_df.Prediction.loc[[f"split_{j}" for j in range(5) if j != i], :].values
    reg.fit(X_train, y_train)

    X_test = X_df.loc[f"split_{i}", :].values
    y_test = y_df.Prediction.loc[f"split_{i}", :].values
    y_hat = reg.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - np.clip(y_hat, 1, 5)) ** 2))
    print(f"RMSE: {rmse}")

    for model, coef in zip(model_folders, reg.coef_):
        print(f"{model.name} {coef:.3f}", end=" - ")

    print("\n")


reg = Ridge(alpha=0.01, positive=True, random_state=42)

X_train = X_df.values
y_train = y_df.Prediction.values
reg.fit(X_train, y_train)

### Submission


# submission data
sub_dfs = []
for model_f in model_folders:
    model_preds = pd.read_csv(model_f / "submission.csv", index_col=0)
    sub_dfs.append(model_preds)
sub_df = pd.concat(sub_dfs, axis=1)

y_hat = reg.predict(sub_df.values).clip(1, 5)

sub_pred_df = pd.read_csv("../data/sampleSubmission.csv")
sub_pred_df.Prediction = y_hat
sub_pred_df.Prediction = sub_pred_df.Prediction.clip(1, 5)
sub_pred_df.to_csv(f"../ridge_ensemble_avg.csv", index=False)
