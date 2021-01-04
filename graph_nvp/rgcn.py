import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.argparser import args

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class Switch(nn.Module):
    def __init__(self):
        super(Switch, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class RelationGraphConvolution(nn.Module):
    """
    Relation GCN layer.
    """
    def __init__(self, in_features, out_features, edge_dim=3, aggregate='sum', dropout=0., use_relu=True, bias=False):
        '''
        :param in/out_features: scalar of channels for node embedding
        :param edge_dim: dim of edge type, virtual type not included
        '''
        super(RelationGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.aggregate = aggregate
        if use_relu:
            self.act = nn.ReLU()
        elif args.use_switch:
            self.act = Switch()
        else:
            self.act = None

        self.weight = nn.Parameter(torch.FloatTensor(
            self.edge_dim, self.in_features, self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(
                self.edge_dim, 1, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.graph_linear_self = GraphLinear(in_features, out_features)
        self.graph_linear_edge = GraphLinear(in_features, out_features * edge_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        typically d=9 e=3
        :return:
        updated x with shape (batch, N, d)
        '''
        x = F.dropout(x, p=self.dropout, training=self.training)  # (b, N, d)

        batch_size = x.size(0)

        mb, node, ch = x.shape

        # --- self connection, apply linear function ---
        hs = self.graph_linear_self(x)
        # --- relational feature, from neighbor connection ---
        # Expected number of neighbors of a vertex
        # Since you have to divide by it, if its 0, you need to arbitrarily set it to 1
        m = self.graph_linear_edge(x)
        m = m.view(mb, node, self.out_features, self.edge_dim)
        m = m.permute(0, 3, 1, 2)
        # m: (batchsize, edge_type, node, ch)
        # hr: (batchsize, edge_type, node, ch)
        hr = torch.matmul(adj, m)
        # hr: (batchsize, node, ch)
        hr = torch.sum(hr, 1)#dim=1)
        return hs + hr


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    def _get_embs_node(self, x, adj):
        """
        Args:
            x: current node feature matrix with shape (batch, N, 9)
            adj: current adjacency feature matrix with shape (batch, 4, N, N)
        Returns:
            graph embedding for updating node features with shape (batch, d)
        """

        batch_size = x.size(0)
        adj = adj[:, :3]  # (batch, 3, N, N)

        node_emb = self.rgcn(x, adj)  # (batch, N, d)
        if self.is_batchNorm:
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch, N, d)

        graph_emb = torch.sum(node_emb, dim=1, keepdim=False).contiguous()  # (batch, d)
        return graph_emb


class GraphAggregation(nn.Module):

    def __init__(self, in_features=128, out_features=64, b_dim=4, dropout=0.):
        super(GraphAggregation, self).__init__() #+b_dim
        self.sigmoid_linear = nn.Sequential(nn.Linear(in_features, out_features),
                                            nn.Sigmoid())  #+b_dim
        self.tanh_linear = nn.Sequential(nn.Linear(in_features, out_features),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.switch = Switch()

    def forward(self, input, activation):
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i,j), 1)
        if args.use_switch:
            output = self.switch(output)
        else:
            output = activation(output) if activation is not None\
                     else output
        output = self.dropout(output)
        return output

class GraphLinear(nn.Module):
    """Graph Linear layer.

    This function assumes its input is 3-dimensional.
    Differently from :class:`chainer.functions.linear`, it applies an affine
    transformation to the third axis of input `x`.

    .. seealso:: :class:`torch.nn.Linear`
    """
    def __init__(self, *argv, **kwargs):
        super(GraphLinear, self).__init__()
        #self.linear = spectral_norm(nn.Linear(*argv, **kwargs))
        self.linear = nn.Linear(*argv, **kwargs)

    def __call__(self, x):
        """Forward propagation.
        Args:
            x (:class:`torch.Tensor`)
                Input array that should be a float array whose ``dim`` is 3.
        Returns:
            :class:`torch.Tensor`:
                A 3-dimeisional array.
        """
        # (minibatch, atom, ch)
        s0, s1, s2 = x.size()
        x = x.view(s0 * s1, s2)
        x = self.linear(x)
        x = x.view(s0, s1, -1)
        return x

class RGCN(nn.Module):
    def __init__(self, nfeat, nhid=256, nout=128, aggout=64, edge_dim=3, num_layers=3, dropout=0., normalization=False):
        '''
        :num_layars: the number of layers in each R-GCN
        '''
        super(RGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.nout = nout
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.dropout = dropout
        self.emb = Linear(nfeat, nfeat, bias=False)

        self.gc1 = RelationGraphConvolution(
            nfeat, nhid, edge_dim=self.edge_dim, aggregate='sum', use_relu=True, dropout=self.dropout, bias=False)
        self.gc2 = nn.ModuleList([RelationGraphConvolution(nhid, nhid, edge_dim=self.edge_dim, aggregate='sum',
                                                           use_relu=True, dropout=self.dropout, bias=False)
                                  for i in range(self.num_layers-2)])
        self.gc3 = RelationGraphConvolution(
            nhid, nout, edge_dim=self.edge_dim, aggregate='sum', use_relu=False, dropout=self.dropout, bias=False)

        self.agg = GraphAggregation(nout, aggout, b_dim=edge_dim, dropout=dropout)
        self.output_layer = nn.Linear(aggout, 1)

    def forward(self, x, adj):
        '''
        :param x: (batch, N, d)
        :param adj: (batch, E, N, N)
        :return:
        '''
        # TODO: Add normalization for adacency matrix
        x = self.emb(x)
        x = self.gc1(x, adj)
        for i in range(self.num_layers-2):
            x = self.gc2[i](x, adj)  # (#node, #class)
        x = self.gc3(x, adj)  # (batch, N, d)
        #graph_emb = torch.sum(x, dim=1, keepdim=False).contiguous()
        x = self.agg(x, torch.tanh) # [256,64]
        return x

    def _get_embs_node(self, x, adj):
        """
        Args:
            x: current node feature matrix with shape (batch, N, 9)
            adj: current adjacency feature matrix with shape (batch, 4, N, N)
        Returns:
            graph embedding for updating node features with shape (batch, d)
        """
        batch_size = x.size(0)
        adj = adj[:, :3]  # (batch, 3, N, N)

        node_emb = self.rgcn(x, adj)  # (batch, N, d)
        if self.is_batchNorm:
            node_emb = self.batchNorm(node_emb.transpose(1, 2)).transpose(1, 2)  # (batch, N, d)

        graph_emb = torch.sum(node_emb, dim=1, keepdim=False).contiguous()  # (batch, d)
        return graph_emb