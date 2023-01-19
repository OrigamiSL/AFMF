import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dropout=0, s=1):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.kernel = kernel
        self.downConv = weight_norm(nn.Conv1d(in_channels=c_in,
                                              out_channels=c_in,
                                              padding=(kernel - 1) // 2,
                                              stride=s,
                                              kernel_size=kernel))
        self.activation1 = nn.GELU()
        self.actConv = weight_norm(nn.Conv1d(in_channels=c_in,
                                             out_channels=c_out,
                                             padding=(kernel - 1) // 2,
                                             stride=1,
                                             kernel_size=kernel))
        self.activation2 = nn.GELU()
        self.sampleConv = nn.Conv1d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=1) if c_in != c_out else None
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if s != 1 else None

    def forward(self, x):
        y = x.clone()
        if self.sampleConv is not None:
            y = self.sampleConv(y.permute(0, 2, 1)).transpose(1, 2)
        if self.pool is not None:
            y = self.pool(y.permute(0, 2, 1)).transpose(1, 2)
        x = self.dropout(self.downConv(x.permute(0, 2, 1)))
        x = self.activation1(x).transpose(1, 2)
        x = self.dropout(self.actConv(x.permute(0, 2, 1)))
        x = self.activation2(x).transpose(1, 2)
        x = x + y
        return x


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dropout=0, pool=True):
        super(ConvBlock, self).__init__()
        FE_block = ConvLayer
        if pool:
            self.conv = nn.Sequential(
                FE_block(c_in, c_in, kernel, dropout, s=2),
                FE_block(c_in, c_out, kernel, dropout, s=1)
            )
        else:
            self.conv = nn.Sequential(
                FE_block(c_in, c_in, kernel, dropout, s=1),
                FE_block(c_in, c_out, kernel, dropout, s=1)
            )

    def forward(self, x):
        x = self.conv(x)
        return x
