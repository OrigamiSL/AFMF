import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, variate, out_variate, input_len,
                 kernel=3, LIN=True):
        super(DLinear, self).__init__()
        self.seq_len = input_len
        self.output_v = out_variate

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel)
        self.Linear_Seasonal = nn.Linear(self.seq_len, 1)
        self.Linear_Trend = nn.Linear(self.seq_len, 1)
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

        enc_input = x_enc.clone()
        enc_input[:, -1:, :self.output_v] = 0

        seasonal_init, trend_init = self.decompsition(enc_input)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x_out = seasonal_output + trend_output
        x_out = x_out.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x_out[:, :, :self.output_v], gt
