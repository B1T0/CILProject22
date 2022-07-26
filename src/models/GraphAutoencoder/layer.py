import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight)

    def forward(self, input, adj):
        # currently no batch support on sparse multiplication
        # input = F.dropout(input, self.dropout, self.training)
        # #support = torch.mm(input, self.weight.weight)
        # support = self.weight(input)
        # print(adj.size())
        # print(support.size())
        # output = torch.smm(adj, support)

        input = F.dropout(input, self.dropout, self.training)
        # support = torch.mm(input, self.weight.weight)
        support = self.weight(input)
        # print(adj.size())
        # print(support.size())
        output = torch.smm(adj, support.squeeze())
        # output = torch.matmul(adj.to_dense(), support)
        return output


class ScaledSigmoid(nn.Module):


    def forward(self, x):
        return 4* F.sigmoid(x) + 1