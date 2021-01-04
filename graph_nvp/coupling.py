import torch
import torch.nn as nn
from graph_nvp.mlp import MLP
from graph_nvp.rgcn import RGCN
from utils.argparser import args

def create_inv_masks(masks):
    inversed_masks = masks.clone()
    inversed_masks[inversed_masks > 0] = 2
    inversed_masks[inversed_masks == 0] = 1
    inversed_masks[inversed_masks == 2] = 0
    return inversed_masks

class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = self.weight.exp() * x
        return x

class Coupling(nn.Module):

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False):
        super(Coupling, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_bonds = num_relations
        self.num_features = num_features

        self.adj_size = self.num_nodes * self.num_nodes * self.num_relations
        self.x_size = self.num_nodes * self.num_features
        self.apply_batch_norm = batch_norm
        self.mask = mask.to(args.device)
        self.inversed_mask = create_inv_masks(self.mask).to(args.device)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError


class AffineAdjCoupling(Coupling):

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False, num_masked_cols=1,
                 ch_list=None):
        super(AffineAdjCoupling, self).__init__(num_nodes, num_relations, num_features, mask,
                                                batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.ch_list = ch_list
        self.adj_size = num_nodes * num_nodes * num_relations
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size

        self.mlp = MLP(ch_list, in_size=self.in_size)
        self.lin = nn.Linear(ch_list[-1], 2 * self.out_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.batch_norm = nn.BatchNorm1d(self.in_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.rescale = Rescale()

    def forward(self, adj):
        masked_adj = adj[:, :, self.mask>0].to(args.device)
        log_s, t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        s = self.sigmoid(log_s + 2)
        s = s.expand(adj.shape)
        log_det_jacobian = torch.sum(torch.log(torch.abs(s)), axis=(1, 2, 3))
        return adj, log_det_jacobian

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask>0].to(args.device)
        log_s, t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        s = self.sigmoid(log_s + 2)
        s = s.expand(adj.shape)
        adj = adj * self.mask + (((adj - t)/s) * self.inversed_mask)
        return adj, None

    def _s_t_functions(self, adj):
        x = adj.view(adj.shape[0], -1).to(args.device)
        if self.apply_batch_norm:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = self.tanh(y)
        y = self.lin(y)
        y = self.rescale(y)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = s.view(y.shape[0], self.num_relations, self.num_nodes, 1).to(args.device)
        t = t.view(y.shape[0], self.num_relations, self.num_nodes, 1).to(args.device)
        return s, t

class AffineNodeFeatureCoupling(Coupling):

    def __init__(self, num_nodes, num_bonds, num_features, mask,
                 batch_norm=False, input_type='float',
                 num_masked_cols=1, ch_list=None):
        super(AffineNodeFeatureCoupling, self).__init__(num_nodes, num_bonds,
                                                        num_features, mask, batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.out_size = num_features * num_masked_cols
        self.rgcn = RGCN(num_features, nhid=128, nout=ch_list['hidden'][0], edge_dim=self.num_bonds,
                         num_layers=args.num_gcn_layer, dropout=0., normalization=False).to(args.device)
        self.lin1 = nn.Linear(ch_list['hidden'][0], out_features=ch_list['hidden'][1])
        self.lin2 = nn.Linear(ch_list['hidden'][1], out_features=2*self.out_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.batch_norm = nn.BatchNorm1d(ch_list['hidden'][0])
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.rescale = Rescale()

    def forward(self, x, adj):
        masked_x = x * self.mask
        s, t = self._s_t_functions(masked_x, adj)
        x = masked_x + x * (s * self.inversed_mask) + t * self.inversed_mask
        log_det_jacobian = torch.sum(torch.log(torch.abs(s)), axis=(1, 2))
        return x, log_det_jacobian

    def reverse(self, y, adj):
        masked_y = y * self.mask
        s, t = self._s_t_functions(masked_y, adj)
        x = masked_y + (((y - t)/s) * self.inversed_mask)
        return x, None

    def _s_t_functions(self, x, adj):
        h = self.rgcn(x, adj)
        batch_size = x.shape[0]
        if self.apply_batch_norm:
            h = self.batch_norm(h)
        h = self.lin1(h)
        h = self.tanh(h)
        h = self.lin2(h)
        h = self.rescale(h)
        s = h[:, :self.out_size]
        t = h[:, self.out_size:]
        s = self.sigmoid(s + 2)

        t = t.view(batch_size, 1, self.out_size)
        t = t.expand(batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size).to(args.device)
        s = s.view(batch_size, 1, self.out_size)
        s = s.expand(batch_size, int(self.num_nodes / self.num_masked_cols), self.out_size).to(args.device)
        return s, t


class AdditiveAdjCoupling(Coupling):

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False,
                 num_masked_cols=1, ch_list=None):
        super(AdditiveAdjCoupling, self).__init__(num_nodes, num_relations,
                                                  num_features, mask,
                                                  batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.adj_size = num_nodes * num_nodes * num_relations
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size
        self.mlp = MLP(ch_list, in_size=self.in_size)
        self.lin = nn.Linear(ch_list[-1], out_features=self.out_size)
        self.batch_norm = nn.BatchNorm1d(self.in_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.tanh = nn.Tanh()
        self.rescale = Rescale()

    def forward(self, adj):
        masked_adj = adj[:, :, self.mask>0].to(args.device)
        t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        adj = adj + t * self.inversed_mask
        return adj, torch.zeros(1, device=args.device)

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask>0].to(args.device)
        t = self._s_t_functions(masked_adj)
        t = t.expand(adj.shape)
        adj = adj - t * self.inversed_mask
        return adj, None

    def _s_t_functions(self, adj):
        adj = adj.view(adj.shape[0], -1)
        x = adj.clone()
        if self.apply_batch_norm:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = self.tanh(y)
        y = self.lin(y) 
        y = self.rescale(y)
        y = y.view(y.shape[0], self.num_relations, self.num_nodes, 1)
        return y

class AdditiveNodeFeatureCoupling(Coupling):
    def __init__(self, num_nodes, num_bonds, num_features,
                 mask,
                 batch_norm=False, ch_list=None,
                 input_type='float', num_masked_cols=1):
        super(AdditiveNodeFeatureCoupling, self).__init__(num_nodes, num_bonds,
                                                          num_features, mask,
                                                          batch_norm=batch_norm)
        self.num_masked_cols = num_masked_cols
        self.out_size = num_features * num_masked_cols
        self.rgcn = RGCN(num_features, nhid=128, nout=ch_list['hidden'][0], edge_dim=self.num_bonds,
                         num_layers=args.num_gcn_layer, dropout=0., normalization=False).to(args.device)
        self.lin1 = nn.Linear(ch_list['hidden'][0], out_features=ch_list['hidden'][1])
        self.lin2 = nn.Linear(ch_list['hidden'][1], out_features=self.out_size)
        self.scale_factor = torch.zeros(1, device=args.device)
        self.batch_norm = nn.BatchNorm1d(ch_list['hidden'][0])
        self.tanh = nn.Tanh()
        self.rescale = Rescale()

    def forward(self, x, adj):
        masked_x = x * self.mask
        batch_size = x.shape[0]
        t = self._s_t_functions(masked_x, adj)
        t = t.view(batch_size, 1, self.out_size)
        t = t.expand(batch_size, int(self.num_nodes/self.num_masked_cols), self.out_size)
        if self.num_masked_cols > 1:
             t = t.view(batch_size, self.num_nodes, self.num_features)
        x = x + t * self.inversed_mask
        return x, torch.zeros(1, device=args.device)

    def reverse(self, y, adj):
        masked_y = y * self.mask
        batch_size = y.shape[0]
        t = self._s_t_functions(masked_y, adj)
        t = t.view(batch_size, 1, self.out_size)
        t = t.expand(batch_size, int(self.num_nodes/self.num_masked_cols), self.out_size)
        if self.num_masked_cols > 1:
             t = t.view(batch_size, self.num_nodes, self.num_features)
        y = y - t * self.inversed_mask
        return y, None

    def _s_t_functions(self, x, adj):
        h = self.rgcn(x, adj)
        if self.apply_batch_norm:
            h = self.batch_norm(h)
        h = self.lin1(h)
        h = self.tanh(h)
        h = self.lin2(h)
        h = self.rescale(h)
        return h
