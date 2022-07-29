from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

class SVDPPDataset(Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation

    if necessary, it is possible to get gradients with regards to sparse COO matrix
    using torch.sparse.mm
    """

    def __init__(self, file_path, n_users, n_items):
        super(SVDPPDataset, self).__init__()
        df = pd.read_csv(file_path)

        # self.graph = torch.sparse_coo_tensor()
        self.len = len(df)
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        indices_i = []
        indices_j = []
        values = []

        for i, x in df.iterrows():
            name, val = x['Id'], x['Prediction']
            user, movie = name.replace('c', '').replace('r', '').split('_')
            user, movie = int(user), int(movie)

            indices_i.append(user)
            indices_j.append(movie)
            values.append(float(val))
        # l.append([user, movie + n_users])
        self.graph_coo = torch.sparse_coo_tensor(torch.tensor([indices_i, indices_j]),
                                             torch.tensor(values), size=[self.n_users+1, self.n_items+1]).coalesce()

        self.graph_csr = self.graph_coo.to_sparse_csr()
        self.global_mean = torch.mean(self.graph_coo.values().float())

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # have to ignore identity indices

        user, item, rating = self.graph_coo.indices()[0, idx], self.graph_coo.indices()[1, idx], self.graph_coo.values()[idx]
        rated_items = self.graph_csr.col_indices()[self.graph_csr.crow_indices()[user]:
                                                   self.graph_csr.crow_indices()[user+1]]
        rated_items_count = len(rated_items)
        return user, item, rated_items, rated_items_count, rating


def collate_fn(batch):
    rated_item_max_len = np.max(list(
        map(lambda row: row[3], batch)))
    u_id_tensor = torch.tensor(list(
        map(lambda row: row[0], batch)), dtype=torch.int64)
    m_id_tensor = torch.tensor(list(
        map(lambda row: row[1], batch)), dtype=torch.int64)
    rates_items_tensor = pad_sequence(list(
        map(lambda row: row[2], batch)), batch_first=True)
    rated_count = torch.tensor(list(
        map(lambda row: row[3], batch)), dtype=torch.float32)
    r_tensor = torch.stack(list(
        map(lambda row: row[4], batch)))
    return (u_id_tensor, m_id_tensor, rates_items_tensor, rated_count), r_tensor



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
            self.train_set = Graph_Dataset(file_path=self.file_dir + '/train.csv', n_users=1000, n_items=18000)
            self.val_set = Graph_Dataset(file_path=self.file_dir + '/val.csv', n_users=1000, n_items=18000)

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
