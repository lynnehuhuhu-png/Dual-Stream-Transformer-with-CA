"""
ResNet-LSTM adapted for dual-stream CNN+Flow features
Based on the paper's methodology for rainfall intensity estimation

Key adaptations:
1. Dual-stream architecture for CNN (2048-dim) and Flow (179-dim) features
2. Independent LSTM processing for each stream
3. Late fusion for regression output
4. Maintains fair comparison: no additional tricks beyond baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMStream(nn.Module):
    """LSTM stream for processing temporal features"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = h_n[-1]  # [batch_size, hidden_dim]

        # Apply normalization and dropout
        output = self.layer_norm(last_hidden)
        output = self.dropout(output)

        return output


class ResNetLSTM_DualStream(nn.Module):
    """
    Dual-stream ResNet-LSTM for CNN + Flow features
    Fair comparison: matches baseline training configuration
    """
    def __init__(self, cnn_dim=2048, flow_dim=179, seq_len=15,
                 hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()

        # CNN stream LSTM
        self.cnn_stream = LSTMStream(
            input_dim=cnn_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Flow stream LSTM
        self.flow_stream = LSTMStream(
            input_dim=flow_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Fusion dimension
        fusion_dim = hidden_dim * 2

        # Regression head (late fusion)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, cnn_x, flow_x):
        """
        cnn_x: [batch_size, seq_len, cnn_dim]
        flow_x: [batch_size, seq_len, flow_dim]
        """
        # Process through LSTM streams
        cnn_feat = self.cnn_stream(cnn_x)
        flow_feat = self.flow_stream(flow_x)

        # Concatenate features
        combined = torch.cat([cnn_feat, flow_feat], dim=1)

        # Regression
        output = self.regressor(combined)

        return output.squeeze()
