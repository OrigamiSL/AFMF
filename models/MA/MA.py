import torch
import torch.nn as nn


class MA(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, out_variate, input_len, LIN=True):
        super(MA, self).__init__()
        self.seq_len = input_len
        self.output_v = out_variate

        self.norm = nn.LayerNorm(input_len, eps=0, elementwise_affine=False)
        self.LIN = LIN

    def forward(self, x_enc, drop=0):
        # x: [Batch, Input length, Channel]
        if drop:
            x_enc[:, -2 ** (drop - 1) - 1:-1, :self.output_v] = 0
        if self.LIN:
            x_enc[:, :, :self.output_v] = self.norm(x_enc[:, :, :self.output_v].permute(0, 2, 1)).transpose(1, 2)
        x_enc[torch.isinf(x_enc)] = 0
        x_enc[torch.isnan(x_enc)] = 0
        gt = x_enc[:, -1:, :].clone()
        gt = gt[:, :, :self.output_v]

        x_out = torch.mean(x_enc[:, :-1, :self.output_v], dim=1, keepdim=True)

        return x_out[:, :, :self.output_v], gt
