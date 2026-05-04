"""
PatchTST adapted for dual-stream CNN+Flow features
Adapted from: https://github.com/yuqinie98/PatchTST

Key adaptations:
1. Dual-stream architecture for CNN (2048-dim) and Flow (179-dim) features
2. Independent PatchTST backbone for each stream
3. Late fusion for regression output
4. Maintains fair comparison: no additional tricks beyond baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional


# ==================== RevIN (Reversible Instance Normalization) ====================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


# ==================== Positional Encoding ====================
def positional_encoding(q_len, d_model):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe - pe.mean()
    pe = pe / (pe.std() * 10)
    return nn.Parameter(pe, requires_grad=True)


# ==================== Transpose Layer ====================
class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


# ==================== Multi-Head Attention ====================
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))
        self.scale = math.sqrt(self.d_k)

    def forward(self, Q, K=None, V=None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        q = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_k)
        output = self.proj(output)
        return output


# ==================== Transformer Encoder Layer ====================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_dropout=0., dropout=0., norm='BatchNorm'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads, attn_dropout, dropout)
        self.dropout_attn = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout_ffn = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout_attn(src2)
        src = self.norm_attn(src)

        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.norm_ffn(src)
        return src


# ==================== PatchTST Backbone ====================
class PatchTST_Backbone(nn.Module):
    def __init__(self, c_in, seq_len, patch_len, stride, d_model=128, n_heads=8,
                 n_layers=3, d_ff=256, dropout=0.1, use_revin=True):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.c_in = c_in

        # Calculate number of patches
        self.patch_num = int((seq_len - patch_len) / stride + 1)

        # RevIN
        self.use_revin = use_revin
        if self.use_revin:
            self.revin_layer = RevIN(c_in, affine=True)

        # Patch embedding
        self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(self.patch_num, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, dropout, norm='BatchNorm')
            for _ in range(n_layers)
        ])

        self.d_model = d_model

    def forward(self, x):
        # x: [bs, c_in, seq_len]
        bs = x.size(0)

        # RevIN normalization
        if self.use_revin:
            x = x.permute(0, 2, 1)  # [bs, seq_len, c_in]
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)  # [bs, c_in, seq_len]

        # Patching: [bs, c_in, seq_len] -> [bs, c_in, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2)  # [bs, c_in, patch_len, patch_num]

        # Patch embedding
        x = x.permute(0, 1, 3, 2)  # [bs, c_in, patch_num, patch_len]
        x = self.W_P(x)  # [bs, c_in, patch_num, d_model]

        # Reshape for channel-independent processing
        x = x.reshape(bs * self.c_in, self.patch_num, self.d_model)

        # Add positional encoding
        x = self.dropout(x + self.W_pos)

        # Transformer encoding
        for layer in self.encoder_layers:
            x = layer(x)

        # Reshape back: [bs * c_in, patch_num, d_model] -> [bs, c_in, patch_num, d_model]
        x = x.reshape(bs, self.c_in, self.patch_num, self.d_model)

        return x


# ==================== Dual-Stream PatchTST ====================
class PatchTST_DualStream(nn.Module):
    """
    Dual-stream PatchTST for CNN + Flow features
    Fair comparison: matches baseline training configuration
    """
    def __init__(self, cnn_dim=2048, flow_dim=179, seq_len=15,
                 patch_len=3, stride=3, d_model=128, n_heads=8,
                 n_layers=3, d_ff=256, dropout=0.1, use_revin=True):
        super().__init__()

        # CNN stream PatchTST
        self.cnn_backbone = PatchTST_Backbone(
            c_in=cnn_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_revin=use_revin
        )

        # Flow stream PatchTST
        self.flow_backbone = PatchTST_Backbone(
            c_in=flow_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            use_revin=use_revin
        )

        # Calculate feature dimensions after patching
        patch_num = int((seq_len - patch_len) / stride + 1)
        cnn_feat_dim = cnn_dim * patch_num * d_model
        flow_feat_dim = flow_dim * patch_num * d_model
        fusion_dim = cnn_feat_dim + flow_feat_dim

        # Regression head (late fusion)
        self.regressor = nn.Sequential(
            nn.Flatten(),
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
        cnn_x: [bs, seq_len, cnn_dim]
        flow_x: [bs, seq_len, flow_dim]
        """
        # Transpose to [bs, dim, seq_len] for PatchTST
        cnn_x = cnn_x.permute(0, 2, 1)
        flow_x = flow_x.permute(0, 2, 1)

        # Process through backbones
        cnn_feat = self.cnn_backbone(cnn_x)  # [bs, cnn_dim, patch_num, d_model]
        flow_feat = self.flow_backbone(flow_x)  # [bs, flow_dim, patch_num, d_model]

        # Concatenate and regress
        combined = torch.cat([cnn_feat, flow_feat], dim=1)
        output = self.regressor(combined)

        return output.squeeze()
