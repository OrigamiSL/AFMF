import torch
from torch import nn
import torch.nn.functional as F
from utils.embed import DataEmbedding


class DeepAR(nn.Module):
    def __init__(self, variate, out_variate, input_len, d_model=512, num_layers=3, LIN=True):
        super(DeepAR, self).__init__()
        self.input_len = input_len
        self.LIN = LIN
        self.output_v = out_variate
        self.embed = DataEmbedding(c_in=variate, d_model=d_model)
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.num_layers = num_layers
        self.d_model = d_model

        self.F = nn.Flatten()  # FC Layer replaces former linear projection layer
        self.FC = nn.Linear(self.input_len * d_model, out_variate)
        self.norm = nn.LayerNorm(input_len, eps=0, elementwise_affine=False)

    def forward(self, x_enc, drop=0):
        # local_norm and mask
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

        enc_input = self.embed(enc_input)
        h_0 = torch.zeros([self.num_layers, x_enc.shape[0], self.d_model]).to(x_enc.device)
        out, h_n = self.rnn(enc_input, h_0)  # out:[batch_sz,seq,hidden_sz]
        output = self.F(out)
        output = self.FC(output).contiguous().view(-1, 1, self.output_v)

        return output, gt
