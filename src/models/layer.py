import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution layer employing propagation
    through sparse matrix multiplication
    """
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # self.weight = nn.Linear(in_features, out_features)
        # self.reset_parameters()

    def forward(self, input, adj, weight):
        input = F.dropout(input, self.dropout, self.training)
        # support = torch.mm(input, self.weight.weight)
        support = input.mm(weight)
        # print(adj.size())
        # print(support.size())
        output = torch.sparse.mm(adj, support.squeeze())
        # output = torch.matmul(adj.to_dense(), support)
        return output


class GraphSelfAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphSelfAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, weight):
        Wh = torch.mm(input, weight)  # weight.shape (out_features, in_features)
        # input.shape: (N, in_features), Wh.shape: (N, out_features)
        # have to rewrite this
        e = self._prepare_attentional_mechanism_input(Wh)
        #print(e.size())
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        # additional activation for concatenation later
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)





    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




class ScaledSigmoid(nn.Module):

    def forward(self, x):
        return 4 * F.sigmoid(x) + 1
