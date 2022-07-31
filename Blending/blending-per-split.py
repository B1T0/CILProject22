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

# Note: hyperparameter tuning is not necessary for this model
regs = []
for i in tqdm(range(5), leave=False):
    reg = Ridge(alpha=0.01, positive=True, random_state=42)

    X_train = X_df.loc[f"split_{i}", :].values
    y_train = y_df.Prediction.loc[f"split_{i}", :].values
    reg.fit(X_train, y_train)

    regs.append(reg)

    X_test = X_df.loc[[f"split_{j}" for j in range(5) if j != i], :].values
    y_test = y_df.Prediction.loc[[f"split_{j}" for j in range(5) if j != i], :].values
    y_hat = reg.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - np.clip(y_hat, 1, 5)) ** 2))
    print(f"RMSE: {rmse}")

    for model, coef in zip(model_folders, reg.coef_):
        print(f"{model.name} {coef:.3f}", end=" - ")

    print("\n")


### Submission


# submission data
sub_dfs = []
for i in range(5):
    split_dfs = []
    for model_f in model_folders:
        if (model_f / f"sub{i}.csv").exists():
            model_preds = pd.read_csv(model_f / f"sub{i}.csv", index_col=0)
        else:
            model_preds = pd.read_csv(model_f / "submission.csv", index_col=0)
        split_dfs.append(model_preds)
    sub_dfs.append(pd.concat(split_dfs, axis=1))


y_hats = []

for i in range(5):
    y_hats.append(regs[i].predict(sub_dfs[i].values))

y_hat = np.mean(y_hats, axis=0).clip(1, 5)


sub_pred_df = pd.read_csv("../data/sampleSubmission.csv")
sub_pred_df.Prediction = y_hat
sub_pred_df.Prediction = sub_pred_df.Prediction.clip(1, 5)
sub_pred_df.to_csv(f"../ridge_ensemble_persplit_ncfavg.csv", index=False)
