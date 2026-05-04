"""
TST (Time Series Transformer) adapted for dual-stream CNN+Flow features
Based on: "A Transformer-based Framework for Multivariate Time Series Representation Learning" (KDD'21)
Paper: https://arxiv.org/abs/2010.02803

Key adaptations:
1. Dual-stream architecture for CNN (2048-dim) and Flow (179-dim) features
2. Independent TST encoder for each stream
3. Late fusion for regression output
4. Maintains fair comparison: no additional tricks beyond baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ==================== Positional Encoding ====================
class FixedPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding
    """
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ==================== Transformer Encoder Layer with LayerNorm ====================
class TransformerLayerNormEncoderLayer(nn.Module):
    """
    Transformer encoder layer with LayerNorm (more stable than BatchNorm)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """
        src: [seq_len, batch_size, d_model]
        """
        # Self-attention with pre-norm
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # Feedforward with pre-norm
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


# ==================== TST Encoder ====================
class TSTEncoder(nn.Module):
    """
    Time Series Transformer Encoder
    """
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers,
                 dim_feedforward, dropout=0.1, activation='gelu'):
        super().__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        # Project input to d_model
        self.project_inp = nn.Linear(feat_dim, d_model)

        # Positional encoding
        self.pos_enc = FixedPositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        # Transformer encoder layers (use LayerNorm for stability)
        encoder_layer = TransformerLayerNormEncoderLayer(
            d_model, n_heads, dim_feedforward, dropout, activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.activation = F.gelu if activation == "gelu" else F.relu
        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X):
        """
        X: [batch_size, seq_length, feat_dim]
        Returns: [batch_size, seq_length, d_model]
        """
        # Permute to [seq_length, batch_size, feat_dim]
        inp = X.permute(1, 0, 2)

        # Project to d_model and scale
        inp = self.project_inp(inp) * math.sqrt(self.d_model)

        # Add positional encoding
        inp = self.pos_enc(inp)

        # Transformer encoding
        output = self.transformer_encoder(inp)

        # Activation
        output = self.activation(output)

        # Permute back to [batch_size, seq_length, d_model]
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)

        return output


# ==================== Dual-Stream TST ====================
class TST_DualStream(nn.Module):
    """
    Dual-stream TST for CNN + Flow features
    Fair comparison: matches baseline training configuration
    """
    def __init__(self, cnn_dim=2048, flow_dim=179, seq_len=15,
                 d_model=256, n_heads=8, num_layers=3,
                 dim_feedforward=1024, dropout=0.2):
        super().__init__()

        # CNN stream TST encoder
        self.cnn_encoder = TSTEncoder(
            feat_dim=cnn_dim,
            max_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )

        # Flow stream TST encoder
        self.flow_encoder = TSTEncoder(
            feat_dim=flow_dim,
            max_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )

        # Pooling for temporal aggregation
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fusion dimension: 2 streams * d_model
        fusion_dim = d_model * 2

        # Regression head (late fusion)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, cnn_x, flow_x):
        """
        cnn_x: [batch_size, seq_len, cnn_dim]
        flow_x: [batch_size, seq_len, flow_dim]
        """
        # Encode through TST
        cnn_encoded = self.cnn_encoder(cnn_x)  # [bs, seq_len, d_model]
        flow_encoded = self.flow_encoder(flow_x)  # [bs, seq_len, d_model]

        # Temporal pooling: [bs, seq_len, d_model] -> [bs, d_model]
        cnn_pooled = self.pool(cnn_encoded.transpose(1, 2)).squeeze(-1)
        flow_pooled = self.pool(flow_encoded.transpose(1, 2)).squeeze(-1)

        # Concatenate features
        combined = torch.cat([cnn_pooled, flow_pooled], dim=1)

        # Regression
        output = self.regressor(combined)

        return output.squeeze()
