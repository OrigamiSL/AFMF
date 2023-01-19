import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.GTA.encoder import Encoder, EncoderLayer, ConvLayer
from models.GTA.decoder import Decoder, DecoderLayer
from models.GTA.attn import FullAttention, ProbAttention, AttentionLayer
from models.GTA.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, variate, out_variate, input_len, label_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 dropout=0.0, activation='gelu'):
        super(Informer, self).__init__()
        self.label_len = label_len
        d_ff = d_model * 4

        # Encoding
        self.enc_embedding = DataEmbedding(variate, d_model, dropout, position=True)
        self.dec_embedding = DataEmbedding(variate, d_model, dropout, position=True)

        # Attention
        Attn = ProbAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
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
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, out_variate, bias=True)

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc[:, :-1, :])
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_inp = torch.zeros_like(x_enc[:, -1:, :]).to(x_enc.device)
        dec_inp = torch.cat([x_enc[:, - self.label_len - 1:-1, :], dec_inp], dim=1).to(x_enc.device)

        dec_out = self.dec_embedding(dec_inp)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -1:, :]  # [B, L, D]
