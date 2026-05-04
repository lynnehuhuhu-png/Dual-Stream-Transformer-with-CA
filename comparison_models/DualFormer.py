"""
DualFormer adapted for dual-stream CNN+Flow features
Based on: "DualFormer: Dual-stream Transformer for Video Action Recognition"

Key adaptations:
1. Dual-stream architecture for CNN (2048-dim) and Flow (179-dim) features
2. Cross-modal attention between streams
3. Late fusion for regression output
4. Maintains fair comparison: no additional tricks beyond baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== Positional Encoding ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== Cross-Modal Attention ====================
class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between two streams
    Similar to DualFormer's cross-modal interaction
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Stream1 -> Stream2 attention
        self.cross_attn_1to2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Stream2 -> Stream1 attention
        self.cross_attn_2to1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stream1, stream2):
        """
        stream1: [batch_size, seq_len, d_model]
        stream2: [batch_size, seq_len, d_model]
        """
        # Stream1 attends to Stream2
        attn_1to2, _ = self.cross_attn_1to2(
            query=stream1,
            key=stream2,
            value=stream2
        )
        stream1 = self.norm1(stream1 + self.dropout(attn_1to2))

        # Stream2 attends to Stream1
        attn_2to1, _ = self.cross_attn_2to1(
            query=stream2,
            key=stream1,
            value=stream1
        )
        stream2 = self.norm2(stream2 + self.dropout(attn_2to1))

        return stream1, stream2


# ==================== Transformer Encoder Layer ====================
class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer with Pre-Norm architecture
    More stable than Post-Norm for small batch sizes
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """
        Pre-Norm architecture: normalize before attention/FFN
        """
        # Self-attention with Pre-Norm
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal
        )
        src = src + self.dropout1(src2)

        # Feedforward with Pre-Norm
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


# ==================== DualStream Encoder ====================
class DualStreamEncoder(nn.Module):
    """
    DualFormer encoder: alternates between self-attention and cross-modal attention
    Each layer performs:
    1. Self-attention within each stream
    2. Cross-modal attention between streams
    """
    def __init__(self, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Self-attention layers for each stream
        self.self_attn_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Cross-modal attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, stream1, stream2):
        """
        stream1: [batch_size, seq_len, d_model] - CNN stream
        stream2: [batch_size, seq_len, d_model] - Flow stream
        """
        for i in range(self.num_layers):
            # Self-attention within each stream
            stream1 = self.self_attn_layers[i](stream1)
            stream2 = self.self_attn_layers[i](stream2)

            # Cross-modal attention between streams
            stream1, stream2 = self.cross_attn_layers[i](stream1, stream2)

        return stream1, stream2


# ==================== DualFormer Model ====================
class DualFormer_DualStream(nn.Module):
    """
    DualFormer adapted for rainfall prediction

    Architecture:
    1. Input projection: CNN(2048) -> d_model, Flow(179) -> d_model
    2. Positional encoding
    3. DualStream encoder with alternating self-attention and cross-modal attention
    4. Temporal pooling (mean + max)
    5. Late fusion and regression head
    """
    def __init__(
        self,
        cnn_dim=2048,
        flow_dim=179,
        seq_len=15,
        d_model=256,
        n_heads=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.2
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection layers
        self.cnn_proj = nn.Linear(cnn_dim, d_model)
        self.flow_proj = nn.Linear(flow_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # DualStream encoder
        self.encoder = DualStreamEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Regression head (late fusion)
        self.regressor = nn.Sequential(
            nn.Linear(d_model * 4, 512),  # *4 for mean+max pooling from both streams
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cnn_x, flow_x):
        """
        cnn_x: [batch_size, seq_len, cnn_dim]
        flow_x: [batch_size, seq_len, flow_dim]
        """
        # Input projection
        cnn_stream = self.cnn_proj(cnn_x)      # [B, T, d_model]
        flow_stream = self.flow_proj(flow_x)   # [B, T, d_model]

        # Positional encoding
        cnn_stream = self.pos_encoder(cnn_stream)
        flow_stream = self.pos_encoder(flow_stream)

        # DualStream encoding with cross-modal attention
        cnn_stream, flow_stream = self.encoder(cnn_stream, flow_stream)

        # Temporal pooling (mean + max)
        cnn_mean = cnn_stream.mean(dim=1)      # [B, d_model]
        cnn_max = cnn_stream.max(dim=1)[0]     # [B, d_model]
        flow_mean = flow_stream.mean(dim=1)    # [B, d_model]
        flow_max = flow_stream.max(dim=1)[0]   # [B, d_model]

        # Late fusion
        fused = torch.cat([cnn_mean, cnn_max, flow_mean, flow_max], dim=1)  # [B, d_model*4]

        # Regression
        output = self.regressor(fused).squeeze(-1)  # [B]

        return output
