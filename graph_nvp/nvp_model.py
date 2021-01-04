import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.argparser import args
from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.coupling import AffineNodeFeatureCoupling, AffineAdjCoupling, \
    AdditiveNodeFeatureCoupling, AdditiveAdjCoupling

def gaussian_nll(x, mean, ln_var, reduce='sum'):
    if reduce not in ('sum', 'mean', 'no'):
        raise ValueError(
            'only \'sum\', \'mean\' and \'no\' are valid for \'reduce\', but '
            '\'%s\' is given' % reduce)

    x_prec = torch.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + torch.log(torch.tensor([2 * np.pi], device=args.device, dtype=torch.float32))) / 2 - x_power
    if reduce == 'sum':
        return torch.sum(loss)
    elif reduce == 'mean':
        return torch.mean(loss)
    else:
        return loss

class GraphNvpModel(nn.Module):
    def __init__(self, hyperparams: Hyperparameters):
        super(GraphNvpModel, self).__init__()
        self.hyperparams = hyperparams
        self._init_params(hyperparams)
        self._need_initialization = False
        if self.masks is None:
            self._need_initialization = True
            self.masks = dict()
            self.masks['node'] = self._create_masks('node')
            self.masks['channel'] = self._create_masks('channel')
        self.num_bonds = self.num_relations
        self.num_atoms = self.num_nodes
        assert self.num_bonds+1 == self.num_features
        self.adj_size = self.num_atoms * self.num_atoms * self.num_relations
        self.x_size = self.num_atoms * self.num_features
        #self.prior_ln_var = torch.tensor([0.7],device=args.device, dtype=torch.float32)
        self.prior_ln_var = nn.Parameter(torch.zeros([1]))  
        nn.init.constant_(self.prior_ln_var, 1e-5)
        self.constant_pi = torch.tensor([3.1415926535], device=args.device, dtype=torch.float32)
        # AffineNodeFeatureCoupling found to be unstable.
        channel_coupling = AffineNodeFeatureCoupling
        node_coupling = AffineAdjCoupling
        if self.additive_transformations:
            channel_coupling = AdditiveNodeFeatureCoupling
            node_coupling = AdditiveAdjCoupling
            print("Additive Transformations")
        transforms = []
        for i in range(self.num_coupling['channel']):
            transforms += [channel_coupling(self.num_nodes, self.num_relations, self.num_features,
                                 self.masks['channel'][i % self.num_masks['channel']],
                                 num_masked_cols=int(self.num_nodes / self.num_masks['channel']),
                                 ch_list=self.gnn_channels,
                                 batch_norm=self.apply_batch_norm)]
        for i in range(self.num_coupling['node']):
            transforms += [node_coupling(self.num_nodes, self.num_relations, self.num_features,
                          self.masks['node'][i % self.num_masks['node']],
                          num_masked_cols=int(self.num_nodes / self.num_masks['channel']),
                          batch_norm=self.apply_batch_norm,
                          ch_list=self.mlp_channels)]
        self.transforms = nn.ModuleList(transforms)

    def forward(self, adj, x):
        h = x.clone()
        sum_log_det_jacs_x = torch.zeros(h.shape[0], device=args.device, requires_grad=True)
        sum_log_det_jacs_adj = torch.zeros(h.shape[0], device=args.device, requires_grad=True)
        # forward step of channel-coupling layers
        for i in range(self.num_coupling['channel']):
            h, log_det_jacobians = self.transforms[i](h, adj)
            sum_log_det_jacs_x = sum_log_det_jacs_x + log_det_jacobians 
        #adj = adj + 0.9 * torch.rand(adj.shape, device=args.device)
        for i in range(self.num_coupling['channel'], len(self.transforms)):
            adj, log_det_jacobians = self.transforms[i](adj)
            sum_log_det_jacs_adj = sum_log_det_jacs_adj + log_det_jacobians 

        adj = adj.view(adj.shape[0], -1)
        h = h.view(h.shape[0], -1)
        out = [h, adj]
        return out, [sum_log_det_jacs_x, sum_log_det_jacs_adj]

    def reverse(self, z, x_size, true_adj=None):
        """
        Returns a molecule, given its latent vector.
        :param z: latent vector. Shape: [B, N*N*M + N*T]
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
        :param true_adj: used for testing. An adjacency matrix of a real molecule
        :return: adjacency matrix and feature matrix of a molecule
        """
        # NOTE z is a list
        batch_size = z.shape[0]
        z_x = z[:, :x_size]
        z_adj = z[:, x_size:]
        temperature = 1.0
        if true_adj is None:
            h_adj = z_adj.view(batch_size, self.num_relations, self.num_nodes, self.num_nodes)
            # First, the adjacency coupling layers are applied in reverse order to get h_adj
            for i in reversed(range(self.num_coupling['channel'], len(self.transforms))):
                h_adj, log_det_jacobians = self.transforms[i].reverse(h_adj)
            adj = h_adj
            adj = adj + adj.permute(0, 1, 3, 2)
            adj = adj / 2.0
            adj = F.softmax(adj, dim=1)
        else:
            adj = true_adj

        h_x = z_x.view(batch_size, self.num_nodes, self.num_features) 
        # channel coupling layers
        for i in reversed(range(self.num_coupling['channel'])):
            h_x, log_det_jacobians = self.transforms[i].reverse(h_x, adj)
        return adj, h_x

    def _init_params(self, hyperparams):
        self.num_nodes = hyperparams.num_nodes
        self.num_relations = hyperparams.num_relations
        self.num_features = hyperparams.num_features
        self.masks = hyperparams.masks

        self.apply_batch_norm = args.apply_batch_norm
        self.additive_transformations = args.additive_transformations

        self.num_masks = hyperparams.num_masks
        self.num_coupling = hyperparams.num_coupling
        self.mask_size = hyperparams.mask_size
        self.mlp_channels = hyperparams.mlp_channels
        self.gnn_channels = hyperparams.gnn_channels

    def _create_masks(self, type):
        masks = []
        num_cols = int(self.num_nodes / self.hyperparams.num_masks[type])
        if type == 'node':
            # Columns of the adjacency matrix is masked
            for i in range(self.hyperparams.num_masks[type]):
                node_mask = torch.ones([self.num_nodes, self.num_nodes])
                for j in range(num_cols):
                    node_mask[:, i + j] = 0.0
                masks.append(node_mask)
        elif type == 'channel':
            # One row (one node) of the feature matrix is masked
            num_cols = int(self.num_nodes / self.hyperparams.num_masks[type])
            for i in range(self.hyperparams.num_masks[type]):
                ch_mask = torch.ones([self.num_nodes, self.num_features])
                for j in range(num_cols):
                    ch_mask[i * num_cols + j, :] = 0.0
                masks.append(ch_mask)
        return masks

    def log_prob(self, z, logdet):
        # z = [h, adj], logdet: [sum_log_det_jacs_x, sum_log_det_jacs_adj]
        logdet[0] = logdet[0] - self.x_size
        logdet[1] = logdet[1] - self.adj_size

        ll_node = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0] ** 2))
        ll_node = ll_node.sum(-1)  # (B)

        ll_edge = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[1] ** 2))
        ll_edge = ll_edge.sum(-1)  # (B)

        ll_node += logdet[0]  # ([B])
        ll_edge += logdet[1]  # ([B])

        nll = -(ll_node.mean() / self.x_size + ll_edge.mean() / self.adj_size) / 2.0
        return nll

    def save_hyperparams(self, path):
        self.hyperparams.save(path)

    def load_hyperparams(self, path):
        """
        loads hyper parameters from a json file
        :param path:
        :return:
        """
        hyperparams = Hyperparameters(path=path)
        self._init_params(hyperparams)

