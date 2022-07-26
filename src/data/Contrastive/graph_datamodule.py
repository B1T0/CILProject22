from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler, Dataset
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


class GraphSamplingDataset(Dataset):
    """
    while theoretically an iterable dataset, abusing Dataset API
    works out to smoother implementation

    probably should have been implemented as iter
    """

    def __init__(self, file_path, n_items, n_users, k, device='cuda:0', seed=23151461614):
        super(GraphSamplingDataset, self).__init__()

        assert k % 2 == 0

        df = pd.read_csv(file_path)
        # self.graph = torch.sparse_coo_tensor()
        self.len = len(df)
        self.n_users = n_users
        self.n_items = n_items
        self.n = n_users + n_items
        self.threshold = 2.5
        indices_i = []
        indices_j = []
        values = []

        for i, x in df.iterrows():
            name, val = x['Id'], x['Prediction']
            user, movie = name.replace('c', '').replace('r', '').split('_')
            user, movie = int(movie) - 1, int(user) - 1
            if user > n_users:
                raise Exception(f"More users in file")
            if movie > n_items:
                raise Exception(f"More movies in file")
            if val > self.threshold:
                indices_i.append(movie)
                indices_j.append(user + n_items)
                values.append(val)
                #
                indices_i.append(user + n_items)
                indices_j.append(movie)
                values.append(val)

        # l.append([user, movie + n_users])
        self.graph = torch.sparse_coo_tensor(torch.tensor([indices_i, indices_j]),
                                             torch.tensor(values), size=[self.n, self.n]).coalesce()

        self.binary_graph = torch.sparse_coo_tensor(torch.tensor([indices_i, indices_j]),
                                                    torch.ones(size=(len(indices_i),)),
                                                    size=[self.n, self.n]).coalesce()
        print('1')
        for idx in self.binary_graph.indices():
            if idx[0] < 1000 and idx[1] < 1000:
                print(f'Binary Graph not bipartite {idx[0]} {idx[1]}')
                print(f'{self.binary_graph[idx[0], idx[1]]}')
            if idx[0] > 1000 and idx[1] > 1000:
                print(f'Binary Graph not bipartite {idx[0]} {idx[1]}')
                print(f'{self.binary_graph[idx[0], idx[1]]}')

        self.degrees = torch.sparse.sum(self.binary_graph, dim=1).to_dense()
        print(self.degrees.size())
        self.anti_degrees = torch.ones_like(self.degrees) * (self.n - 1) - self.degrees
        print(torch.min(self.anti_degrees))
        print(torch.max(self.anti_degrees))
        self.len = len(self.binary_graph.indices()[0])
        print(self.len)
        self.k_neighborhood = self.binary_graph

        print('Computing Neighborhood')
        for i in range(k - 1):
            print(f'{i} neighborhood')
            self.k_neighborhood = torch.sparse.mm(self.k_neighborhood, self.binary_graph)

        self.k_neighborhood_plus = torch.sparse.mm(self.k_neighborhood, self.binary_graph)
        # convert to cst

        self.k_degrees = torch.count_nonzero(self.k_neighborhood.to_dense(), dim=1)
        self.k_degrees_plus = torch.count_nonzero(self.k_neighborhood_plus.to_dense(), dim=1)
        print(f'k: {k}')
        print(f'Len k_degrees: {len(self.k_degrees)}')
        print(f'Max k-neighbors: {torch.max(self.k_degrees)}')
        print(f'Min k-neighbors: {torch.min(self.k_degrees)}')

        print(f'k+1: {k + 1}')
        print(f'Len k_degrees: {len(self.k_degrees_plus)}')
        print(f'Max k-neighbors: {torch.max(self.k_degrees_plus)}')
        print(f'Min k-neighbors: {torch.min(self.k_degrees_plus)}')

        self.k_neighborhood = self.k_neighborhood.to_sparse_csr()
        self.k_neighborhood_plus = self.k_neighborhood_plus.to_sparse_csr()
        self.binary_graph = self.binary_graph  # .to(device)
        # self.sampler = BatchSampler(WeightedRandomSampler(self.degrees, self.n, replacement=True), 3, drop_last=True)
        # self.batches = torch.tensor(list(self.sampler))
        self.num_samples = 1
        self.device = device
        self.m = 5
        self.sampler = torch.distributions.categorical.Categorical(
            torch.ones(self.n)
        )
        self.indices = self.binary_graph.indices()  # .to(device)
        self.neighborhood = self.binary_graph.to_sparse_csr()
        self.neighborhood_col = self.neighborhood.col_indices()
        self.neighborhood_crow = self.neighborhood.crow_indices()

        # self.edge_indices = self.edge_idx_to_vector(self.indices)
        self.col_indices = self.k_neighborhood.col_indices()  # .to(device)
        self.crow_indices = self.k_neighborhood.crow_indices()  # .to(device)
        self.values = self.k_neighborhood.values()

        self.col_indices1 = self.k_neighborhood_plus.col_indices()
        self.crow_indices1 = self.k_neighborhood_plus.crow_indices()
        self.values1 = self.k_neighborhood_plus.values()
        self.generator = torch.Generator()
        # self.generator.manual_seed(2495021)
        self.generator.manual_seed(seed=seed)

    def is_bipartite(self):
        for idx in self.indices:
            if idx[0] < self.n_items and idx[1] < self.n_items:
                print(f'Binary Graph not bipartite: {idx[0]} {idx[1]}')
                return False
            if idx[0] >= self.n_items and idx[1] >= self.n_items:
                print(f'k Graph not bipartite: {idx[0]} {idx[1]}')
                return False
        return True

    def __len__(self):
        return self.len

    def vector_to_edge_idx(self, a):
        row = a.div(self.n, rounding_mode='floor')
        col = a % self.n
        return row.long(), col.long()  # for some reason casting did not work

    def edge_idx_to_vector(self, a):
        row, col = a
        return row * self.n + col

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
        # implicit assumption that every user has a negative edge
        row = torch.multinomial(self.anti_degrees, num_samples=self.m, generator=self.generator)
        col = []
        for x in range(len(row)):
            x_neighbors = self.neighborhood_col[
                          self.neighborhood_crow[x]:self.neighborhood_crow[x + 1]]
            w = torch.ones(self.n)
            w[x_neighbors] = 0
            w[x] = 0
            # not k-neighbors of user
            x_negative = torch.multinomial(w, num_samples=1, generator=self.generator)
            col.append(x_negative)
        return row, torch.tensor(col)

    def sample_k_walks(self, x, y, k_plus=False):
        if k_plus:
            col_indices, crow_indices, values = self.col_indices1, self.crow_indices1, self.values1
        else:
            col_indices, crow_indices, values = self.col_indices, self.crow_indices, self.values

        ii_neighbors = col_indices[
                       crow_indices[x]:crow_indices[x + 1]]
        ii_walks = values[crow_indices[x]:crow_indices[x + 1]]  # random walk weights

        uu_neighbors = col_indices[
                       crow_indices[y]:crow_indices[y + 1]]
        uu_walks = values[crow_indices[y]:crow_indices[y + 1]]
        # item to item neighbor
        ii_neighbor = ii_neighbors[torch.multinomial(ii_walks, num_samples=self.num_samples,
                                                     generator=self.generator)]
        uu_neighbor = uu_neighbors[torch.multinomial(uu_walks.float(), num_samples=self.num_samples,
                                                     generator=self.generator)]

        w = torch.ones(self.n)
        w[ii_neighbors] = 0
        # not k-neighbors of user
        if torch.max(w) < 1:
            ii_negative = torch.full((self.m,), x)
        else:
            ii_negative = torch.multinomial(w, num_samples=self.m, generator=self.generator)
        w = torch.ones(self.n)
        w[uu_neighbors] = 0
        if torch.max(w) < 1 or torch.sum(w) <= 0:
            uu_negative = torch.full((self.m,), x)
        # negative k_neighbors: k-neighbors of item
        else:
            uu_negative = torch.multinomial(w, num_samples=self.m, generator=self.generator)
        return ii_neighbor, uu_neighbor, ii_negative, uu_negative

    def __getitem__(self, idx):
        x, y = self.indices[:, idx]
        if x >= self.n_items:
            x, y = y, x
        # neighborhood retrieval & sampling
        # graph does not look bipartite
        ii_neighbor, uu_neighbor, ii_negative, uu_negative = self.sample_k_walks(x, y, k_plus=False)
        # k+1 walk
        # item --> user

        iu_neighbor, ui_neighbor, iu_negative, ui_negative = self.sample_k_walks(x, y, k_plus=True)
        # negative edge sampling
        neg_item, neg_user = self.negative_edges()

        # negative k-walks sampling
        return x, y, ii_neighbor, uu_neighbor, ii_negative, uu_negative, iu_neighbor, ui_neighbor, \
               iu_negative, ui_negative, neg_item, neg_user


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
            self.train_set = GraphSamplingDataset(file_path=self.file_dir + '/train_split_0.csv', n_users=1000,
                                                  n_items=10000,
                                                  k=self.k)
            self.val_set = GraphSamplingDataset(file_path=self.file_dir + '/test_split_0.csv', n_users=1000,
                                                n_items=10000,
                                                k=self.k)

        if stage == "test" or stage is None:
            self.test_set = GraphSamplingDataset(file_path=self.file_dir + '/val.csv', n_users=1000, n_items=10000,
                                                 k=self.k)

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
