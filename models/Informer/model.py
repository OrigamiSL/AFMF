import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Informer.encoder import Encoder, EncoderLayer, ConvLayer
from models.Informer.decoder import Decoder, DecoderLayer
from models.Informer.attn import FullAttention, ProbAttention, AttentionLayer
from models.Informer.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, input_len, label_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu', distil=True, mix=True, LIN=True):
        super(Informer, self).__init__()
        self.label_len = label_len
        self.attn = attn
        self.LIN = LIN
        self.output_v = c_out

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
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

        x_enc[:, -1:, :self.output_v] = 0
        x_dec = x_enc[:, - self.label_len - 1:, :].clone()

        enc_out = self.enc_embedding(x_enc[:, :-1, :])
        enc_out = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        return dec_out[:, -1:, :], gt  # [B, L, D]