from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
import pandas as pd
from src.data.Graph.graph_datamodule import Graph_Dataset
import time


# def collate_tuples(batch):
#     #print(len(batch))
#     #print(batch)
#     batch = list(zip(*batch))
#     #print(batch)
#     for i in range(len(batch)):
#         batch[i] = torch.stack(batch[i])
#     # print(batch[0].size())
#     return tuple(batch)

def vector_to_edge_idx(a):
    row = a.div(11000, rounding_mode='floor')
    col = a % 11000
    return row.long(), col.long()  # for some reason casting did not work


def edge_idx_to_vector(a):
    row, col = a
    return row * 11000 + col


class Triplet_Dataset(Graph_Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation

    probably should have been implemented as iter
    """

    def __init__(self, file_path, n_users, n_items, k, length=100000, device='cuda:0'):
        super(Triplet_Dataset, self).__init__(file_path, n_users, n_items, False, True)
        self.degrees = torch.sparse.sum(self.binary_graph)
        self.anti_degrees = torch.ones(n_users + n_items) * (n_users + n_items -1) -self.degrees
        self.len = len(self.binary_graph.indices()[0])

        self.binary_graph = self.binary_graph
        self.k_neighborhood = self.binary_graph

        print('Computing Neighborhood')
        for i in range(k):
            print(f'{i} neighborhood')
            self.k_neighborhood = torch.sparse.mm(self.k_neighborhood, self.binary_graph)
        # convert to cst

        self.k_neighborhood = self.k_neighborhood.to_sparse_csr()
        self.binary_graph = self.binary_graph#.to(device)
        # self.sampler = BatchSampler(WeightedRandomSampler(self.degrees, self.n, replacement=True), 3, drop_last=True)
        # self.batches = torch.tensor(list(self.sampler))
        self.num_samples = 1
        self.device = device
        self.m = 5
        self.sampler = torch.distributions.categorical.Categorical(
            torch.ones(11000)
        )
        self.indices = self.binary_graph.indices()#.to(device)
        self.neighborhood = self.binary_graph.to_sparse_csr()
        self.neighborhood_col = self.neighborhood.col_indices()
        self.neighborhood_crow = self.neighborhood.crow_indices()

        self.edge_indices = edge_idx_to_vector(self.indices)
        self.col_indices = self.k_neighborhood.col_indices()#.to(device)
        self.crow_indices = self.k_neighborhood.crow_indices()#.to(device)

    def reset(self):
        self.sampler = BatchSampler(WeightedRandomSampler(self.degrees, self.n, replacement=True), 3, drop_last=True)
        self.batches = torch.tensor(list(self.sampler))

    def __len__(self):
        return self.len


    def negative_edges(self):
        # neg_idx = None
        # for _ in range(3):
        #     t5 = time.time()
        #     sample_x = torch.multinomial(torch.ones(11000), num_samples=(2 * self.m)).reshape(2, self.m)
        #     sample_y = torch.multinomial(torch.ones(11000), num_samples=(2 * self.m)).reshape(2, self.m)
        #     # self.sampler.sample(sample_shape=(2, self.m)).reshape)
        #     sample = sample_x * 11000 + sample_y
        #     t6 = time.time()
        #     print(f'sampling edges part {t6 - t5}')
        #     mask = torch.isin(sample, self.edge_indices)
        #     if neg_idx is not None:
        #         mask |= torch.isin(sample, neg_idx)
        #     t7 = time.time()
        #     print(f'is in time {t7 - t6}')
        #     sample = sample[~mask]
        #     neg_idx = sample if neg_idx is None else torch.cat([neg_idx, sample])
        #     if neg_idx.numel() >= self.m:
        #         neg_idx = neg_idx[:self.m]
        #         break
        #
        # row, col = vector_to_edge_idx(neg_idx)
        # return row, col
        #implicit assumption that every user has a negative edge
        row = torch.multinomial(self.anti_degrees, num_samples=self.m)
        col = []
        for x in range(len(row)):
            x_neighbors = self.neighborhood_col[
                          self.neighborhood_crow[x]:self.neighborhood_crow[x + 1]]
            w = torch.ones(11000)
            w[x_neighbors] = 0
            w[x] = 0
            # not k-neighbors of user
            x_negative = torch.multinomial(w, num_samples=1)
            col.append(x_negative)
        return row, torch.tensor(col)

    def __getitem__(self, idx):
        x, y = self.indices[:, idx]

        #t = time.time()
        x_neighbors = self.col_indices[
                      self.crow_indices[x]:self.crow_indices[x + 1]]

        y_neighbors = self.col_indices[
                      self.crow_indices[y]:self.crow_indices[y + 1]]
        #t1 = time.time()
        #print(f'Retrieving neighbors {t1-t}')
        x_n = x_neighbors[torch.multinomial(torch.ones_like(x_neighbors).float(), num_samples=self.num_samples)]
        y_n = y_neighbors[torch.multinomial(torch.ones_like(y_neighbors).float(), num_samples=self.num_samples)]
        #t2 = time.time()
        #print(f'Sampling neighbors{t2-t1}')

        #t3 = time.time()
        row, col = self.negative_edges()
        #print(f'Sampling negative edges {t3-t2}')
        w = torch.ones(11000)
        w[x_neighbors] = 0
        # not k-neighbors of user
        x_negative = torch.multinomial(w, num_samples=self.m)
        w = torch.ones(11000)
        w[y_neighbors] = 0
        # not k-neighbors of item
        y_negative = torch.multinomial(w, num_samples=self.m)
        #t4= time.time()
        #print(f'Sampling negative neighbors {t4-t3}')
        return x, y, x_n, y_n, x_negative, \
               y_negative, row, col


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
            self.train_set = Triplet_Dataset(file_path=self.file_dir + '/train_split_0.csv', n_users=1000,
                                             n_items=10000,
                                             k=self.k)
            self.val_set = Triplet_Dataset(file_path=self.file_dir + '/test_split_0.csv', n_users=1000, n_items=10000,
                                           k=self.k)

        if stage == "test" or stage is None:
            self.test_set = Triplet_Dataset(file_path=self.file_dir + '/val.csv', n_users=1000, n_items=10000, k=self.k)

        if stage == "predict" or stage is None:
            self.predict_set = NotImplemented

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,  # collate_fn=self.collate_fn,
                          shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,  # collate_fn=self.collate_fn,
                          shuffle=True)

    def predict_dataloader(self):
        raise NotImplementedError("We do not have a predict set in this datamodule")
