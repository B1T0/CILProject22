from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
import pandas as pd
from src.data.Graph.graph_datamodule import Graph_Dataset


# def collate_tuples(batch):
#     #print(len(batch))
#     #print(batch)
#     batch = list(zip(*batch))
#     #print(batch)
#     for i in range(len(batch)):
#         batch[i] = torch.stack(batch[i])
#     # print(batch[0].size())
#     return tuple(batch)

class Triplet_Dataset(Graph_Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation

    probably should have been implemented as iter
    """

    def __init__(self, file_path, n_users, n_items, k, length=100000):
        super(Triplet_Dataset, self).__init__(file_path, n_users, n_items, False, True)
        self.degrees = torch.sparse.sum(self.binary_graph)
        self.len = len(self.binary_graph.indices()[0])
        self.k_neighborhood = self.binary_graph
        for i in range(k):
            self.k_neighborhood = torch.sparse.mm(self.k_neighborhood, self.binary_graph)
        # convert to cst
        self.k_neighborhood = self.k_neighborhood.to_sparse_csr()
        # self.sampler = BatchSampler(WeightedRandomSampler(self.degrees, self.n, replacement=True), 3, drop_last=True)
        # self.batches = torch.tensor(list(self.sampler))
        self.num_samples = 1
        self.m = 5
        self.sampler = torch.distributions.categorical.Categorical(
            torch.ones(11000)
        )
        self.edge_indices = torch.tensor(
            np.apply_along_axis(self.edge_idx_to_vector, 0, self.binary_graph.indices().numpy()))

    def edge_idx_to_vector(self, a):
        return a[0] * 11000 + a[1]

    def vector_to_edge_idx(self, a):
        return (a - a % 11000) / 11000, a % 11000  # for some reason casting did not work

    def reset(self):
        self.sampler = BatchSampler(WeightedRandomSampler(self.degrees, self.n, replacement=True), 3, drop_last=True)
        self.batches = torch.tensor(list(self.sampler))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x, y = self.binary_graph.indices()[:, idx]
        x_neighbors = self.k_neighborhood.col_indices()[
                      self.k_neighborhood.crow_indices()[x]:self.k_neighborhood.crow_indices()[x + 1]]

        y_neighbors = self.k_neighborhood.col_indices()[
                      self.k_neighborhood.crow_indices()[y]:self.k_neighborhood.crow_indices()[y + 1]]

        x_n = x_neighbors[torch.multinomial(torch.ones_like(x_neighbors).float(), num_samples=self.num_samples)]
        y_n = y_neighbors[torch.multinomial(torch.ones_like(y_neighbors).float(), num_samples=self.num_samples)]

        neg_idx = None
        for _ in range(3):
            sample_x = self.sampler.sample(sample_shape=(2, self.m))
            sample_y = self.sampler.sample(sample_shape=(2, self.m))
            sample = sample_x * 11000 + sample_y
            mask = torch.isin(sample, self.edge_indices)
            if neg_idx is not None:
                mask |= torch.isin(sample, neg_idx)
            sample = sample[~mask]
            neg_idx = sample if neg_idx is None else torch.cat([neg_idx, sample])
            if neg_idx.numel() >= self.m:
                neg_idx = neg_idx[:self.m]
                break
        neg_idx = torch.tensor(np.apply_along_axis(self.vector_to_edge_idx, 0, neg_idx.numpy())).long()
        w = torch.ones(11000)
        w[x_neighbors] = 0
        # not k-neighbors of user
        x_negative = torch.multinomial(w, num_samples=self.m)
        w = torch.ones(11000)
        w[y_neighbors] = 0
        # not k-neighbors of item
        y_negative = torch.multinomial(w, num_samples=self.m)

        return self.binary_graph.indices()[0, idx], self.binary_graph.indices()[
            1, idx], x_n, y_n, x_negative, y_negative, neg_idx[0], neg_idx[1]


class Graph_DataModule(pl.LightningDataModule):

    def __init__(self, file_dir, k=2, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = None  # define dataset specific transforms here
        # self.collate_fn = collate_tuples
        self.file_dir = file_dir
        self.k = 2

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
            self.train_set = Triplet_Dataset(file_path=self.file_dir + '/train_split_0.csv', n_users=1000, n_items=10000,
                                             k=self.k)
            self.val_set = Triplet_Dataset(file_path=self.file_dir + '/test_split_0.csv', n_users=1000, n_items=10000,
                                           k=self.k)

        if stage == "test" or stage is None:
            self.test_set = Triplet_Dataset(file_path=self.file_dir + '/val.csv', n_users=1000, n_items=10000, k=self.k)

        if stage == "predict" or stage is None:
            self.predict_set = NotImplemented

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,  # collate_fn=self.collate_fn,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,  # collate_fn=self.collate_fn,
                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,  # collate_fn=self.collate_fn,
                          shuffle=True)

    def predict_dataloader(self):
        raise NotImplementedError("We do not have a predict set in this datamodule")
