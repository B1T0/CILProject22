from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pandas as pd

def collate_tuples(batch):
    # print(batch)
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    # print(batch[0].size())
    return tuple(batch)


class Graph_Dataset(Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation
    """
    def __init__(self, file_path, n_users, n_items):
        super(Graph_Dataset, self).__init__()
        df = pd.read_csv(file_path)

        self.graph = torch.sparse_coo_tensor()
        self.len = len(df)
        l = []
        for i in df.iterrows():
            name, val = i
            i, j = name.split('_')
            self.graph[i, j+n_users] = val
            l.append([i, j + n_users])
        for i in range(n_users):
            self.graph[i, i] = 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        #have to ignore identity indices
        return self.graph.indices[idx]

class Graph_DataModule(pl.LightningDataModule):
    def __init__(self, file_dir, batch_size=32, collate_fn=None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = None  # define dataset specific transforms here
        self.collate_fn = collate_fn
        self.file_dir = file_dir


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        There are also data operations you might want to perform on every GPU. Use setup() to do things like:
            - count number of classes
            - build vocabulary
            - perform train/val/test splits
            - create datasets
            - apply transforms (defined explicitly in your datamodule)
        :param stage: fit, test, predict
        :return: Nothing
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = Graph_Dataset(file_path=self.file_dir+'/train.csv', n_users=1000, n_items=18000)
            self.val_set = Graph_Dataset(file_path=self.file_dir+'/val.csv', n_users=1000, n_items=18000)

        if stage == "test" or stage is None:
            self.test_set = Graph_Dataset(file_path=self.file_dir + '/val.csv', n_users=1000, n_items=18000)

        if stage == "predict" or stage is None:
            self.predict_set = NotImplemented

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def predict_dataloader(self):
        raise NotImplementedError("We do not have a predict set in this datamodule")