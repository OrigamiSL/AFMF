import torch
from torch import nn
from models.MTAD_GAT.Modules import ConvLayer, RNNDecoder, FeatureAttentionLayer, Forecasting_Model, \
    ReconstructionModel, \
    GRULayer, TemporalAttentionLayer


class MTAD_GAT(nn.Module):
    def __init__(self, variate, out_variate, input_len, kernel_size=7, feat_gat_embed_dim=None,
                 time_gat_embed_dim=None, use_gatv2=True, gru_n_layers=1, gru_hid_dim=150, forecast_n_layers=1,
                 forecast_hid_dim=150, recon_n_layers=1, recon_hid_dim=150, dropout=0.2, alpha=0.2, LIN=True):

        super(MTAD_GAT, self).__init__()
        self.name = 'MTAD_GAT'
        self.variate = variate
        self.output_v = out_variate
        self.LIN = LIN
        self.input_len = input_len

        self.conv = ConvLayer(variate, kernel_size)
        self.feature_gat = FeatureAttentionLayer(variate, input_len, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(variate, input_len, dropout, alpha, time_gat_embed_dim,
                                                   use_gatv2)
        self.gru = GRULayer(3 * variate, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_variate, forecast_n_layers,
                                                   dropout)
        self.recon_model = ReconstructionModel(input_len, gru_hid_dim, recon_hid_dim, out_variate, recon_n_layers,
                                               dropout)
        self.norm = nn.LayerNorm(self.input_len, eps=0, elementwise_affine=False)

    def forward(self, x_enc, drop=0):
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

        x = self.conv(enc_input)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.contiguous().view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)
        return predictions, recons, gt
