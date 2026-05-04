"""
iTransformer模型 - 适配降雨强度预测任务
基于论文: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024)

核心思想：
- 反转Transformer架构：在变量维度而非时间维度应用注意力
- 将每个时间序列变量视为token
- 适配双流输入：CNN特征 + 光流特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding_inverted(nn.Module):
    """反转的数据嵌入层"""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [Batch, Seq_len, Num_vars]
        x = x.permute(0, 2, 1)  # [Batch, Num_vars, Seq_len]
        x = self.value_embedding(x)  # [Batch, Num_vars, d_model]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class iTransformer(nn.Module):
    """
    iTransformer模型 - 适配降雨强度预测

    输入：
    - cnn_x: [Batch, Seq_len, 2048] - CNN特征
    - flow_x: [Batch, Seq_len, 179] - 光流特征

    输出：
    - pred: [Batch] - 降雨强度预测值
    """
    def __init__(self,
                 cnn_dim=2048,
                 flow_dim=179,
                 seq_len=15,
                 d_model=512,
                 n_heads=8,
                 e_layers=4,
                 d_ff=2048,
                 dropout=0.1,
                 use_norm=True):
        super(iTransformer, self).__init__()

        self.seq_len = seq_len
        self.use_norm = use_norm

        # 特征预处理：将CNN和光流特征投影到统一维度
        self.cnn_proj = nn.Linear(cnn_dim, d_model // 2)
        self.flow_proj = nn.Linear(flow_dim, d_model // 2)

        # 合并后的特征维度
        self.num_vars = d_model  # CNN和Flow合并后的变量数

        # 反转嵌入层
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)

        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation='gelu')
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 回归头（修复：使用平均池化而非展平）
        self.projector = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, cnn_x, flow_x):
        """
        前向传播

        Args:
            cnn_x: [Batch, Seq_len, 2048]
            flow_x: [Batch, Seq_len, 179]

        Returns:
            pred: [Batch] - 降雨强度预测
        """
        batch_size = cnn_x.size(0)

        # 特征投影
        cnn_feat = self.cnn_proj(cnn_x)  # [B, L, d_model//2]
        flow_feat = self.flow_proj(flow_x)  # [B, L, d_model//2]

        # 拼接特征
        x = torch.cat([cnn_feat, flow_feat], dim=-1)  # [B, L, d_model]

        # 归一化（可选）
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # 反转嵌入：[B, L, d_model] -> [B, d_model, d_model]
        enc_out = self.enc_embedding(x)

        # Transformer编码
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        enc_out = self.norm(enc_out)

        # 平均池化：[B, d_model, d_model] -> [B, d_model]
        enc_out = enc_out.mean(dim=1)

        # 回归预测
        pred = self.projector(enc_out).squeeze(-1)

        return pred
