import torch
import torch.nn as nn

from models.RT.ConvBlock import ConvBlock
from models.RT.embed import DataEmbedding


class Res_block(nn.Module):
    def __init__(self, d_model, kernel, dropout, block_nums, input_len, variate, pool=True):
        super(Res_block, self).__init__()
        if pool:
            pro_conv = [ConvBlock(d_model * (2 ** i), d_model * (2 ** (i + 1)),
                                  kernel=kernel, dropout=dropout, pool=pool)
                        for i in range(block_nums)]
        else:
            pro_conv = [ConvBlock(d_model, d_model,
                                  kernel=kernel, dropout=dropout, pool=pool)
                        for _ in range(block_nums)]
        self.pro_conv = nn.ModuleList(pro_conv)
        last_dim = d_model * (2 ** block_nums)
        self.F = nn.Flatten()
        if pool:
            self.projection = nn.Conv1d(in_channels=last_dim * input_len // (2 ** block_nums),
                                        out_channels=variate, kernel_size=1, bias=False)
        else:
            self.projection = None
        self.variate = variate

    def forward(self, x):
        for conv in self.pro_conv:
            x = conv(x)
        if self.projection is not None:
            F_out = self.F(x.permute(0, 2, 1)).unsqueeze(-1)
            x_out = self.projection(F_out).squeeze().contiguous().view(-1, self.variate, 1)
            x_out = x_out.transpose(1, 2)
        else:
            x_out = x
        return x_out


class RF(nn.Module):
    def __init__(self, variate, out_variate, input_len,
                 kernel=3, block_nums=3, d_model=64, pyramid=1, LIN=True, dropout=0.0):
        super(RF, self).__init__()
        print("Start Embedding")
        # Enbeddinging
        self.enc_bed = [DataEmbedding(variate, d_model, dropout)
                        for i in range(pyramid)]
        self.enc_bed = nn.ModuleList(self.enc_bed)
        self.output_v = out_variate
        self.LIN = LIN

        assert (pyramid <= block_nums)
        self.input_len = input_len
        self.input = variate
        self.d_model = d_model
        print("Embedding finished")

        Res_blocks = [Res_block(d_model, kernel, dropout, block_nums - i,
                                self.input_len // (2 ** i), self.output_v)
                      for i in range(pyramid)]
        self.Res_blocks = nn.ModuleList(Res_blocks)
        self.norm = nn.LayerNorm(input_len, eps=0, elementwise_affine=False)

    def forward(self, x_enc, drop=0):
        # local_norm PrcAMF
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

        enc_out = 0
        i = 0
        for embed, RT_b in zip(self.enc_bed, self.Res_blocks):
            embed_enc = embed(enc_input[:, -self.input_len // (2 ** i):, :])
            enc_out += RT_b(embed_enc)
            i += 1
        enc_out = enc_out / i

        final_out = enc_out

        return final_out, gt  # [B, L, D]

