import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, variate, out_variate, input_len, edge_index_sets, d_model=64, out_layer_inter_dim=256,
                 out_layer_num=1, topk=20, dropout=0.2, LIN=True):
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets
        edge_index = edge_index_sets[0]
        self.embedding = nn.Embedding(variate, d_model)
        self.bn_outlayer_in = nn.BatchNorm1d(d_model)
        self.output_v = out_variate
        self.LIN = LIN

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_len, d_model, inter_dim=2 * d_model, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(d_model * edge_set_num, variate, out_layer_num, inter_num=out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(input_len, eps=0, elementwise_affine=False)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, x_enc, drop=0):
        if drop:
            x_enc[:, -2 ** (drop - 1) - 1:-1, :self.output_v] = 0
        if self.LIN:
            x_enc[:, :, :self.output_v] = self.norm(x_enc[:, :, :self.output_v].permute(0, 2, 1)).transpose(1, 2)
        x_enc[torch.isinf(x_enc)] = 0
        x_enc[torch.isnan(x_enc)] = 0

        enc_input = x_enc.clone()
        enc_input[:, -1:, :self.output_v] = 0

        gt = x_enc[:, -1:, :].clone()
        gt = gt[:, :, :self.output_v]
        edge_index_sets = self.edge_index_sets
        enc_input = enc_input.transpose(0, 1)
        batch_num, all_feature, node_num = x_enc.shape

        x = enc_input.contiguous().view(-1, all_feature)
        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(x_enc.device)

            batch_edge_index = self.cache_edge_index_sets[i]
            all_embeddings = self.embedding(torch.arange(node_num).to(x_enc.device))

            weights = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).contiguous().view(-1, 1), weights.norm(dim=-1)
                                      .contiguous().view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji
            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten()\
                .to(x_enc.device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(x_enc.device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings)

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.contiguous().view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(x_enc.device)
        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.transpose(1, 2)

        return out[:, :, :self.output_v], gt
