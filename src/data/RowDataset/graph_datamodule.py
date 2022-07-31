from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pandas as pd


class RowDataset(Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation

    if necessary, it is possible to get gradients with regards to sparse COO matrix
    using torch.sparse.mm
    """

    def __init__(self, file_path, n_items, n_users, user='user_mode'):
        super(RowDataset, self).__init__()

        print(file_path)
        self.df = pd.read_csv(str(file_path))

        # self.graph = torch.sparse_coo_tensor()
        if user == 'user_mode':
            self.user = True
        else:
            self.user = False
        print(f'Dataset user mode: {user}')
        #gives out users
        self.len = len(self.df)
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        indices_i = []
        indices_j = []
        values = []
        print("Iterating over dataset")
        for i, x in self.df.iterrows():
            name, val = x['Id'], x['Prediction']
            user, movie = name.replace('c', '').replace('r', '').split('_')
            movie, user = int(movie) - 1, int(user) - 1
            val = val
            if movie > self.n_items:
                raise Exception(f'Movie id {movie} > n_items {self.n_items}')
            if user > self.n_users:
                raise Exception(f'User id {user} > n_users {self.n_users}')

            indices_i.append(movie)
            indices_j.append(user)
            values.append(val)

        self.graph = torch.sparse_coo_tensor(torch.tensor([indices_i, indices_j]),
                                              torch.tensor(values), size=[self.n_items, self.n_users]).coalesce().to_dense()
        self.len = self.n_items
        if self.user:
            self.graph = self.graph.transpose(0, 1)
            self.len = self.n_users
        print(f'Graph {self.graph.size()}')
        print(f'{self.len}')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # have to ignore identity ind
        # return idx, self.graph[idx, :]
        return idx, self.graph[idx]


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
