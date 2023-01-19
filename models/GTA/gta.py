import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from models.GTA.model import Informer
from models.GTA.tconv import TemporalBlock
import torch.nn.functional as F


class AdaGCNConv(MessagePassing):
    def __init__(self, num_nodes, in_channels, out_channels, improved=False,
                 add_self_loops=False, normalize=True, bias=True, init_method='all'):
        super(AdaGCNConv, self).__init__(aggr='add', node_dim=0)  # "Max" aggregation.
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.bias = bias
        self.init_method = init_method

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._init_graph_logits_()

        self.reset_parameters()

    def _init_graph_logits_(self):
        if self.init_method == 'all':
            logits = .8 * torch.ones(self.num_nodes ** 2, 2)
            logits[:, 1] = 0
        elif self.init_method == 'random':
            logits = 1e-3 * torch.randn(self.num_nodes ** 2, 2)
        elif self.init_method == 'equal':
            logits = .5 * torch.ones(self.num_nodes ** 2, 2)
        else:
            raise NotImplementedError('Initial Method %s is not implemented' % self.init_method)

        self.register_parameter('logits', Parameter(logits, requires_grad=True))

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        if self.normalize:
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops, dtype=x.dtype)

        z = torch.nn.functional.gumbel_softmax(self.logits, hard=True)

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None, z=z)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_weight, z):
        if edge_weight is None:
            return x_j * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))
        else:
            return edge_weight.view([-1] + [1] * (x_j.dim() - 1)) * x_j * z[:, 0].contiguous().view(
                [-1] + [1] * (x_j.dim() - 1))

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GraphTemporalEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, seq_len, num_levels, kernel_size=3, dropout=0.02, device=torch.device('cuda:0')):
        super(GraphTemporalEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.num_levels = num_levels
        self.device = device
        assert (kernel_size - 1) // 2

        self.tc_modules = torch.nn.ModuleList([])
        self.gc_modules = torch.nn.ModuleList([])
        self.gc_module = AdaGCNConv(num_nodes, seq_len, seq_len)
        for i in range(num_levels):
            dilation_size = 2 ** i
            self.tc_modules.extend(
                [TemporalBlock(num_nodes, num_nodes, kernel_size=kernel_size, stride=1, dilation=dilation_size,
                               padding=(kernel_size - 1) * dilation_size // 2, dropout=dropout)])
            self.gc_modules.extend([AdaGCNConv(num_nodes, seq_len, seq_len)])

        source_nodes, target_nodes = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                source_nodes.append(j)
                target_nodes.append(i)
        self.edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=self.device)

    def forward(self, x):
        # >> (bsz, seq_len, num_nodes)
        x = x.permute(0, 2, 1)  # >> (bsz, num_nodes, seq_len)

        x = self.tc_modules[0](x)  # >> (bsz, num_nodes, seq_len)
        x = self.gc_modules[0](x.transpose(0, 1), self.edge_index).transpose(0, 1)  # >> (bsz, num_nodes, seq_len)
        # output = x

        for i in range(1, self.num_levels):
            x = self.tc_modules[i](x)  # >> (bsz, num_nodes, seq_len)
            x = self.gc_module(x.transpose(0, 1), self.edge_index).transpose(0, 1)  # >> (bsz, num_nodes, seq_len)
            # output += x

        # return output.transpose(1, 2) # >> (bsz, seq_len, num_nodes)
        return x.transpose(1, 2)


class GTA(torch.nn.Module):
    def __init__(self, num_nodes, c_out, input_len, label_len, num_levels,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 dropout=0.0, activation='gelu', LIN=True,
                 device=torch.device('cuda:0')):
        super(GTA, self).__init__()
        self.num_nodes = num_nodes
        self.LIN = LIN
        self.output_v = c_out
        self.input_len = input_len
        self.label_len = label_len
        self.num_levels = num_levels
        self.device = device

        self.gt_embedding = GraphTemporalEmbedding(num_nodes, input_len, num_levels, kernel_size=3,
                                                   dropout=dropout, device=device)
        self.model = Informer(num_nodes, c_out, input_len, label_len,
                              factor, d_model, n_heads, e_layers, d_layers,
                              dropout, activation)
        self.norm = nn.LayerNorm(input_len, eps=0, elementwise_affine=False)

    def forward(self, x_enc, drop=0):
        if drop:
            x_enc[:, -2 ** (drop - 1) - 1:-1, :self.output_v] = 0
        if self.LIN:
            x_enc[:, :, :self.output_v] = self.norm(x_enc[:, :, :self.output_v].permute(0, 2, 1)).transpose(1, 2)
        x_enc[torch.isinf(x_enc)] = 0
        x_enc[torch.isnan(x_enc)] = 0
        gt = x_enc[:, -1:, :].clone()
        gt = gt[:, :, :self.output_v]

        enc_input = x_enc.clone()
        enc_input[:, -1:, :self.output_v] = 0

        batch_x = self.gt_embedding(enc_input)  # >> (bsz, seq, num_nodes)
        output = self.model(batch_x)
        return output, gt
