import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTNet(nn.Module):
    def __init__(self, input_len, input_dim, hidRNN, hidCNN, hidSkip, skip, CNN_kernel, highway_window,
                 dropout, c_out, LIN=True):
        super(LSTNet, self).__init__()

        self.P = input_len
        self.input_dim = input_dim
        self.c_out = c_out
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, input_dim))  #
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, c_out)
        else:
            self.linear1 = nn.Linear(self.hidR, c_out)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.LIN = LIN
        self.norm = nn.LayerNorm(input_len, eps=0, elementwise_affine=False)

    def forward(self, x, drop=0):
        if drop:
            x[:, -2 ** (drop - 1) - 1:-1, :self.c_out] = 0
        if self.LIN:
            x[:, :, :self.c_out] = self.norm(x[:, :, :self.c_out].permute(0, 2, 1)).transpose(1, 2)
        x[torch.isinf(x)] = 0
        x[torch.isnan(x)] = 0

        enc_input = x.clone()
        enc_input[:, -1:, :self.c_out] = 0

        gt = x[:, -1:, :].clone()
        gt = gt[:, :, :self.c_out]
        
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.input_dim)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()

            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)
        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.input_dim)
            res = res + z[:, :self.c_out]

        res = res.contiguous().view(-1, 1, self.c_out)

        return res, gt
