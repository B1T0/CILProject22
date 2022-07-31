# Creation of cross-validation splits

import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns
from zipfile import ZipFile

df = pd.read_csv("../data/data_train.csv")
CV = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(CV.split(df)):
    print("TRAIN:", train_index, "TEST:", test_index)
    df.iloc[train_index].to_csv(f"../data/train_split_{i}.csv", index=False)
    df.iloc[test_index].to_csv(f"../data/test_split_{i}.csv", index=False)

file_name = "../data/cross_validation.zip"
with ZipFile(file_name, "w") as zip_file:
    for i in range(10):
        zip_file.write(f"../data/train_split_{i}.csv", f"train_split_{i}.csv")
        zip_file.write(f"../data/test_split_{i}.csv", f"test_split_{i}.csv")
