import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DiagMask():
    def __init__(self, B, H, L, Diag, device="cpu"):
        with torch.no_grad():
            ones_mat = torch.ones(L, L, dtype=int).to(device)
            mask_list = []
            for i in range(Diag):
                down_mat = torch.tril(ones_mat, diagonal=2 ** i - 1)
                cur_mask = down_mat.bool()
                cur_mask[:, -1] = 0
                cur_mask = cur_mask.unsqueeze(0).unsqueeze(0).expand(B, H // Diag, L, L)
                mask_list.append(cur_mask)
            self._mask = torch.cat(mask_list, dim=1)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, attention_dropout=0.1, Diag=0):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.Diag = Diag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            attn_mask = DiagMask(B, H, L, self.Diag, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, Diag=0):
        super(AttentionLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        assert(n_heads % Diag == 0)

        self.inner_attention = FullAttention(mask_flag=True, attention_dropout=dropout, Diag=Diag)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, Res=True):
        super(EncoderLayer, self).__init__()
        d_ff = 4 * d_model
        self.attention = AttentionLayer(d_model, n_heads, dropout, Diag=4)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        self.Res = Res

    def forward(self, x):
        new_x = self.attention(x, x, x)
        if self.Res:
            x = x + self.dropout(new_x)
        else:
            x = self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)
