# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from utils.embed import DataEmbedding
from models.Autoformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.Autoformer.Autoformer_EncDec import Encoder, EncoderLayer, \
    Decoder, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, variate, out_variate, input_len, label_len,
                 moving_avg, d_model,
                 dropout, factor, n_heads, activation,
                 e_layers, d_layers, LIN):
        super(Autoformer, self).__init__()
        self.seq_len = input_len
        self.label_len = label_len
        self.pred_len = 1
        self.LIN = LIN
        self.output_v = out_variate

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
            self.decomp2 = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
            self.decomp2 = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding(out_variate, d_model, dropout, position=False)
        self.dec_embedding = DataEmbedding(out_variate, d_model, dropout, position=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    out_variate,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, out_variate, bias=True)
        )
        self.norm = nn.LayerNorm(input_len, eps=0, elementwise_affine=False)

    def forward(self, x_enc, drop=0):
        # local_norm
        if drop:
            x_enc[:, -2 ** (drop - 1) - 1:-1, :self.output_v] = 0
        if self.LIN:
            x_enc[:, :, :self.output_v] = self.norm(x_enc[:, :, :self.output_v].permute(0, 2, 1)).transpose(1, 2)
        x_enc[torch.isinf(x_enc)] = 0
        x_enc[torch.isnan(x_enc)] = 0
        # decomp init
        gt = x_enc[:, -1:, :].clone()
        gt = gt[:, :, :self.output_v]

        x_input = x_enc[:, :-1, :self.output_v].clone()

        mean = torch.mean(x_input, dim=1).unsqueeze(1)
        season = torch.zeros(x_input.shape[0], 1, self.output_v).to(x_input.device)

        season_init, trend_init = self.decomp(x_input)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([season_init[:, -self.label_len:, :], season], dim=1)
        # enc
        enc_out = self.enc_embedding(x_input)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final
        output = trend_part + seasonal_part
        return output[:, -1:, :], gt
