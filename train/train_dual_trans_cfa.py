"""
双流Transformer改进版（5折交叉验证）
核心改进：
1. 双流独立编码
2. DCMA跨模态注意力交互
3. 可学习门控融合
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdvancedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class DirectionalCrossModalAttention(nn.Module):
    """方向性跨模态注意力模块"""
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.cnn_to_flow = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.flow_to_cnn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, cnn_feat, flow_feat):
        # CNN引导光流
        flow_attended, _ = self.cnn_to_flow(flow_feat, cnn_feat, cnn_feat)
        flow_out = self.norm1(flow_feat + flow_attended)

        # 光流引导CNN
        cnn_attended, _ = self.flow_to_cnn(cnn_feat, flow_feat, flow_feat)
        cnn_out = self.norm2(cnn_feat + cnn_attended)

        # 可学习门控融合
        alpha = torch.sigmoid(self.alpha)
        cnn_fused = alpha * cnn_out + (1 - alpha) * cnn_feat
        flow_fused = alpha * flow_out + (1 - alpha) * flow_feat

        return cnn_fused, flow_fused


class ImprovedTwoStreamTransformer(nn.Module):
    def __init__(self, cnn_dim=2048, flow_dim=179, seq_len=15, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model

        # CNN特征处理器
        self.cnn_processor = nn.Sequential(
            nn.Linear(cnn_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, d_model),
        )

        # 光流特征处理器
        self.flow_processor = nn.Sequential(
            nn.Linear(flow_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, d_model),
        )

        self.pos_encoder = AdvancedPositionalEncoding(d_model)

        # CNN独立Transformer（前2层）
        cnn_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.cnn_transformer_1 = nn.TransformerEncoder(cnn_encoder_layer, num_layers=2)
        self.cnn_transformer_2 = nn.TransformerEncoder(cnn_encoder_layer, num_layers=2)

        # 光流独立Transformer（前2层）
        flow_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.flow_transformer_1 = nn.TransformerEncoder(flow_encoder_layer, num_layers=2)
        self.flow_transformer_2 = nn.TransformerEncoder(flow_encoder_layer, num_layers=2)

        # DCMA跨模态交互（插在第2层后）
        self.dcma = DirectionalCrossModalAttention(d_model, nhead)

        self.pool_1s = nn.AdaptiveAvgPool1d(1)
        self.cnn_conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.cnn_conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.flow_conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.flow_conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)

        # 融合后的回归头
        self.regressor = nn.Sequential(
            nn.Linear(d_model * 6, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self.cnn_norm = nn.LayerNorm(d_model)
        self.flow_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, cnn_x, flow_x):
        bs, seq_len = cnn_x.size(0), cnn_x.size(1)

        # 处理特征
        cnn_feat = self.cnn_processor(cnn_x.view(-1, cnn_x.size(-1))).view(bs, seq_len, self.d_model)
        flow_feat = self.flow_processor(flow_x.view(-1, flow_x.size(-1))).view(bs, seq_len, self.d_model)

        # 前2层独立编码
        cnn_feat = self.pos_encoder(cnn_feat)
        flow_feat = self.pos_encoder(flow_feat)

        cnn_encoded = self.cnn_transformer_1(cnn_feat)
        flow_encoded = self.flow_transformer_1(flow_feat)

        # DCMA跨模态交互
        cnn_encoded, flow_encoded = self.dcma(cnn_encoded, flow_encoded)

        # 后2层继续编码
        cnn_encoded = self.cnn_transformer_2(cnn_encoded)
        flow_encoded = self.flow_transformer_2(flow_encoded)

        # 残差归一化
        cnn_encoded = self.cnn_norm(cnn_encoded + cnn_feat)
        flow_encoded = self.flow_norm(flow_encoded + flow_feat)

        # CNN多尺度池化
        cnn_t1 = self.pool_1s(cnn_encoded.transpose(1, 2)).squeeze(-1)
        cnn_t3 = self.cnn_conv3(cnn_encoded.transpose(1, 2)).mean(dim=-1)
        cnn_t5 = self.cnn_conv5(cnn_encoded.transpose(1, 2)).mean(dim=-1)

        # 光流多尺度池化
        flow_t1 = self.pool_1s(flow_encoded.transpose(1, 2)).squeeze(-1)
        flow_t3 = self.flow_conv3(flow_encoded.transpose(1, 2)).mean(dim=-1)
        flow_t5 = self.flow_conv5(flow_encoded.transpose(1, 2)).mean(dim=-1)

        # 后期融合
        combined = torch.cat([cnn_t1, cnn_t3, cnn_t5, flow_t1, flow_t3, flow_t5], dim=-1)
        return self.regressor(combined).squeeze()


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    preds, targets = [], []

    for cnn_x, flow_x, y in tqdm(loader, desc="Training", leave=False):
        cnn_x, flow_x, y = cnn_x.to(device), flow_x.to(device), y.to(device)

        if np.random.rand() < 0.5:
            lam = np.random.beta(0.5, 0.5)
            idx = torch.randperm(y.size(0)).to(device)
            cnn_x = lam * cnn_x + (1 - lam) * cnn_x[idx]
            flow_x = lam * flow_x + (1 - lam) * flow_x[idx]
            y = lam * y + (1 - lam) * y[idx]

        if np.random.rand() < 0.3:
            cnn_x = cnn_x + torch.randn_like(cnn_x) * 0.1
            flow_x = flow_x + torch.randn_like(flow_x) * 0.1

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            pred = model(cnn_x, flow_x)
            huber_loss = F.huber_loss(pred, y, delta=1.0)
            mse_loss = F.mse_loss(pred, y)
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters()) * 1e-5
            loss = huber_loss + 0.1 * mse_loss + l2_reg

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds.extend(pred.detach().cpu().numpy())
        targets.extend(y.cpu().numpy())

    return total_loss / len(loader), r2_score(targets, preds)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for cnn_x, flow_x, y in loader:
            cnn_x, flow_x, y = cnn_x.to(device), flow_x.to(device), y.to(device)
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                pred = model(cnn_x, flow_x)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    targets, preds = np.array(targets), np.array(preds)
    return r2_score(targets, preds), mean_absolute_error(targets, preds)


def main():
    from sklearn.model_selection import KFold

    print("=" * 80)
    print("🌧️  双流Transformer改进版（5折交叉验证）")
    print("=" * 80)

    F_cnn = np.load("datatest1/train_video_feature_resnet50.npy")
    F_flow = np.load("datatest1/train_video_optical_flow_resnet50.npy")
    df = pd.read_csv("datatest1/train_video_label_resnet50.csv")
    y = df["RAINFALL INTENSITY"].values

    print(f"✅ 数据: CNN{F_cnn.shape}, Flow{F_flow.shape}, 样本{len(y)}")
    print("-" * 80)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    print("⚙️ 开始 5 折交叉验证...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(F_cnn)):
        print(f"{'='*80}\n折 {fold+1}/5\n{'='*80}")

        train_dataset = TensorDataset(
            torch.FloatTensor(F_cnn[train_idx]),
            torch.FloatTensor(F_flow[train_idx]),
            torch.FloatTensor(y[train_idx])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(F_cnn[val_idx]),
            torch.FloatTensor(F_flow[val_idx]),
            torch.FloatTensor(y[val_idx])
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        model = ImprovedTwoStreamTransformer(
            cnn_dim=2048, flow_dim=179, seq_len=15, d_model=512, nhead=8, num_layers=4
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
        scaler = GradScaler()

        best_val_r2 = -np.inf
        patience_counter = 0

        for epoch in range(100):
            train_loss, train_r2 = train_epoch(model, train_loader, optimizer, scaler, device)
            val_r2, val_mae = evaluate(model, val_loader, device)
            scheduler.step()

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")

        print(f"✅ 折 {fold+1} 最佳R²: {best_val_r2:.4f}\n")
        fold_results.append(best_val_r2)

    print("=" * 80)
    print("📊 5折交叉验证结果")
    print("=" * 80)

    for i, r2 in enumerate(fold_results):
        print(f"折 {i+1}: R² = {r2:.4f}")

    mean_r2 = np.mean(fold_results)
    std_r2 = np.std(fold_results)

    print(f"\n{'='*80}")
    print(f"平均R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"{'='*80}\n")

    print("📌 对比:")
    print(f"   基础Transformer: 74.87%")
    print(f"   改进Transformer: {mean_r2*100:.2f}%")
    print(f"   提升: {(mean_r2 - 0.7487)*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
