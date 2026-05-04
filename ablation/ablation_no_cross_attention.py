"""
消融实验：去掉跨流注意力机制
目的：证明跨流注意力对模型性能的贡献

与完整模型的唯一区别：
- ❌ 移除 CrossStreamAttention 模块
- ❌ 移除相关参数和计算
- ✅ 保留其他所有组件（统计特征、双流架构、多尺度池化等）
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# 固定随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
# 确保PyTorch的确定性行为
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 时序统计特征提取 ====================
def extract_temporal_statistics(features):
    """提取时序统计特征（与完整模型相同）"""
    N, seq_len, feature_dim = features.shape
    stats_list = []

    for i in range(N):
        seq = features[i]

        # 基础统计量
        mean_feat = np.mean(seq, axis=0)
        std_feat = np.std(seq, axis=0)
        max_feat = np.max(seq, axis=0)
        min_feat = np.min(seq, axis=0)

        # 时序趋势
        time_steps = np.arange(seq_len).reshape(-1, 1)
        slopes = []
        for j in range(feature_dim):
            try:
                slope, _, _, _, _ = stats.linregress(time_steps.flatten(), seq[:, j])
                if np.isnan(slope) or np.isinf(slope):
                    slope = 0.0
                slopes.append(slope)
            except:
                slopes.append(0.0)
        slopes = np.array(slopes)

        # 变化率
        change_rate = seq[-1] - seq[0]

        # 高阶统计量
        skewness = stats.skew(seq[:, :10], axis=0)
        kurtosis = stats.kurtosis(seq[:, :10], axis=0)
        skewness = np.nan_to_num(skewness, nan=0.0, posinf=0.0, neginf=0.0)
        kurtosis = np.nan_to_num(kurtosis, nan=0.0, posinf=0.0, neginf=0.0)

        # 自相关系数
        autocorr = []
        for j in range(min(10, feature_dim)):
            if np.std(seq[:, j]) > 1e-6:
                try:
                    corr = np.corrcoef(seq[:-1, j], seq[1:, j])[0, 1]
                    if np.isnan(corr) or np.isinf(corr):
                        corr = 0.0
                    autocorr.append(corr)
                except:
                    autocorr.append(0.0)
            else:
                autocorr.append(0.0)
        autocorr = np.array(autocorr)

        # 拼接统计特征
        sample_indices = np.linspace(0, feature_dim-1, min(32, feature_dim), dtype=int)
        stats_feat = np.concatenate([
            mean_feat[sample_indices],
            std_feat[sample_indices],
            max_feat[sample_indices],
            min_feat[sample_indices],
            slopes[sample_indices],
            change_rate[sample_indices],
            skewness,
            kurtosis,
            autocorr,
        ])
        stats_feat = np.nan_to_num(stats_feat, nan=0.0, posinf=0.0, neginf=0.0)
        stats_list.append(stats_feat)

    return np.array(stats_list)


# ==================== 位置编码 ====================
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


# ==================== 消融模型：无跨流注意力 ====================
class TwoStreamTransformer_NoCrossAttn(nn.Module):
    """
    消融实验模型：去掉跨流注意力

    保留的组件：
    - ✅ 双流架构（CNN流 + 光流流）
    - ✅ 时序统计特征
    - ✅ 独立Transformer编码器
    - ✅ 多尺度时序池化

    移除的组件：
    - ❌ 跨流注意力机制
    """
    def __init__(self, cnn_dim=2048, flow_dim=179, cnn_stats_dim=234, flow_stats_dim=234,
                 seq_len=15, d_model=512, nhead=8, num_layers=4):
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

        # 统计特征处理器
        self.cnn_stats_processor = nn.Sequential(
            nn.Linear(cnn_stats_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        self.flow_stats_processor = nn.Sequential(
            nn.Linear(flow_stats_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        self.pos_encoder = AdvancedPositionalEncoding(d_model)

        # ❌ 没有跨流注意力模块

        # CNN独立Transformer
        cnn_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.cnn_transformer = nn.TransformerEncoder(cnn_encoder_layer, num_layers=num_layers)

        # 光流独立Transformer
        flow_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.flow_transformer = nn.TransformerEncoder(flow_encoder_layer, num_layers=num_layers)

        # 多尺度池化
        self.pool_1s = nn.AdaptiveAvgPool1d(1)
        self.pool_3s = nn.AvgPool1d(3, stride=1)
        self.pool_5s = nn.AvgPool1d(5, stride=1)

        # 融合后的回归头
        fusion_dim = d_model * 6 + 128 * 2
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

        self.cnn_norm = nn.LayerNorm(d_model)
        self.flow_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, cnn_x, flow_x, cnn_stats, flow_stats):
        bs, seq_len = cnn_x.size(0), cnn_x.size(1)

        # 处理原始特征
        cnn_feat = self.cnn_processor(cnn_x.view(-1, cnn_x.size(-1))).view(bs, seq_len, self.d_model)
        flow_feat = self.flow_processor(flow_x.view(-1, flow_x.size(-1))).view(bs, seq_len, self.d_model)

        # 添加位置编码
        cnn_feat = self.pos_encoder(cnn_feat)
        flow_feat = self.pos_encoder(flow_feat)

        # ❌ 跳过跨流注意力，直接进入独立编码

        # CNN独立编码
        cnn_encoded = self.cnn_transformer(cnn_feat)
        cnn_encoded = self.cnn_norm(cnn_encoded + cnn_feat)

        # 光流独立编码
        flow_encoded = self.flow_transformer(flow_feat)
        flow_encoded = self.flow_norm(flow_encoded + flow_feat)

        # CNN多尺度池化
        cnn_t1 = self.pool_1s(cnn_encoded.transpose(1, 2)).squeeze(-1)
        cnn_t3 = self.pool_3s(cnn_encoded.transpose(1, 2)).mean(dim=-1)
        cnn_t5 = self.pool_5s(cnn_encoded.transpose(1, 2)).mean(dim=-1)

        # 光流多尺度池化
        flow_t1 = self.pool_1s(flow_encoded.transpose(1, 2)).squeeze(-1)
        flow_t3 = self.pool_3s(flow_encoded.transpose(1, 2)).mean(dim=-1)
        flow_t5 = self.pool_5s(flow_encoded.transpose(1, 2)).mean(dim=-1)

        # 处理统计特征
        cnn_stats_feat = self.cnn_stats_processor(cnn_stats)
        flow_stats_feat = self.flow_stats_processor(flow_stats)

        # 后期融合
        combined = torch.cat([
            cnn_t1, cnn_t3, cnn_t5,
            flow_t1, flow_t3, flow_t5,
            cnn_stats_feat, flow_stats_feat
        ], dim=-1)

        return self.regressor(combined).squeeze()


# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    preds, targets = [], []

    for cnn_x, flow_x, cnn_stats, flow_stats, y in tqdm(loader, desc="Training", leave=False):
        cnn_x = cnn_x.to(device)
        flow_x = flow_x.to(device)
        cnn_stats = cnn_stats.to(device)
        flow_stats = flow_stats.to(device)
        y = y.to(device)

        # Mixup数据增强
        if np.random.rand() < 0.5:
            lam = np.random.beta(0.5, 0.5)
            idx = torch.randperm(y.size(0)).to(device)
            cnn_x = lam * cnn_x + (1 - lam) * cnn_x[idx]
            flow_x = lam * flow_x + (1 - lam) * flow_x[idx]
            cnn_stats = lam * cnn_stats + (1 - lam) * cnn_stats[idx]
            flow_stats = lam * flow_stats + (1 - lam) * flow_stats[idx]
            y = lam * y + (1 - lam) * y[idx]

        # 噪声增强
        if np.random.rand() < 0.3:
            cnn_x = cnn_x + torch.randn_like(cnn_x) * 0.1
            flow_x = flow_x + torch.randn_like(flow_x) * 0.1

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            pred = model(cnn_x, flow_x, cnn_stats, flow_stats)
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
        for cnn_x, flow_x, cnn_stats, flow_stats, y in loader:
            cnn_x = cnn_x.to(device)
            flow_x = flow_x.to(device)
            cnn_stats = cnn_stats.to(device)
            flow_stats = flow_stats.to(device)
            y = y.to(device)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                pred = model(cnn_x, flow_x, cnn_stats, flow_stats)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    targets, preds = np.array(targets), np.array(preds)
    return r2_score(targets, preds), mean_absolute_error(targets, preds), mean_squared_error(targets, preds) ** 0.5


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("🔬 消融实验：去掉跨流注意力机制")
    print("=" * 80)
    print("保留组件：✅ 双流架构 ✅ 统计特征 ✅ 多尺度池化")
    print("移除组件：❌ 跨流注意力")
    print("=" * 80)

    # 加载数据
    F_flow = np.load("../datatest1/train_video_optical_flow_resnet50.npy")
    df = pd.read_csv("../datatest1/train_video_label_resnet50.csv")
    y = df["RAINFALL INTENSITY"].values

    print(f"✅ 数据: Flow{F_flow.shape}, 样本{len(y)}")

    # 提取光流统计特征
    print("🔄 提取光流时序统计特征...")
    F_flow_stats = extract_temporal_statistics(F_flow)
    print(f"✅ 光流统计特征: {F_flow_stats.shape}")
    print("-" * 80)

    # 加载数据划分索引
    train_idx = np.load("../train_idx_dev.npy")
    val_idx = np.load("../val_idx_dev.npy")

    print(f"📊 数据划分")
    print(f"   训练集: {len(train_idx)} 个样本")
    print(f"   验证集: {len(val_idx)} 个样本")
    print("-" * 80)

    # 加载CNN特征
    train_feature_file = "../datatest1/train_video_feature_dev.npy"
    val_feature_file = "../datatest1/val_video_feature_dev.npy"

    if not os.path.exists(train_feature_file) or not os.path.exists(val_feature_file):
        print(f"❌ 特征文件不存在")
        return

    F_cnn_train = np.load(train_feature_file)
    F_cnn_val = np.load(val_feature_file)
    print(f"✅ 训练集CNN特征: {F_cnn_train.shape}")
    print(f"✅ 验证集CNN特征: {F_cnn_val.shape}")

    # 提取CNN统计特征
    print("🔄 提取CNN时序统计特征...")
    F_cnn_train_stats = extract_temporal_statistics(F_cnn_train)
    F_cnn_val_stats = extract_temporal_statistics(F_cnn_val)
    print(f"✅ CNN统计特征: Train{F_cnn_train_stats.shape}, Val{F_cnn_val_stats.shape}")
    print("-" * 80)

    # 创建数据集
    train_dataset = TensorDataset(
        torch.FloatTensor(F_cnn_train),
        torch.FloatTensor(F_flow[train_idx]),
        torch.FloatTensor(F_cnn_train_stats),
        torch.FloatTensor(F_flow_stats[train_idx]),
        torch.FloatTensor(y[train_idx])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(F_cnn_val),
        torch.FloatTensor(F_flow[val_idx]),
        torch.FloatTensor(F_cnn_val_stats),
        torch.FloatTensor(F_flow_stats[val_idx]),
        torch.FloatTensor(y[val_idx])
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"✅ 数据加载器创建完成")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   验证批次: {len(val_loader)}")
    print("-" * 80)

    # 创建模型
    model = TwoStreamTransformer_NoCrossAttn(
        cnn_dim=2048,
        flow_dim=179,
        cnn_stats_dim=F_cnn_train_stats.shape[1],
        flow_stats_dim=F_flow_stats.shape[1],
        seq_len=15,
        d_model=512,
        nhead=8,
        num_layers=4
    ).to(device)

    print(f"✅ 模型创建完成")
    print(f"   设备: {device}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 80)

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    scaler = GradScaler()

    print(f"✅ 优化器配置完成")
    print(f"   优化器: AdamW (lr=1e-4, weight_decay=5e-4)")
    print(f"   调度器: CosineAnnealingWarmRestarts (T_0=10)")
    print(f"   混合精度: 启用")
    print("-" * 80)

    # 训练循环
    best_val_r2 = -np.inf
    best_val_mae = float('inf')
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    max_epochs = 100
    patience = 20

    print(f"🚀 开始训练...")
    print(f"   最大轮数: {max_epochs}")
    print(f"   早停耐心: {patience}")
    print("=" * 80)
    print(f"{'Epoch':<8} | {'Train R²':<10} | {'Val R²':<10} | {'Val MAE':<10} | {'Val RMSE':<10} | {'Best':<6}")
    print("-" * 80)

    for epoch in range(max_epochs):
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer, scaler, device)
        val_r2, val_mae, val_rmse = evaluate(model, val_loader, device)
        scheduler.step()

        is_best = False
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            patience_counter = 0
            is_best = True
            # 保存最佳模型
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), 'results/best_model_no_cross_attn.pth')
        else:
            patience_counter += 1

        best_marker = "✓" if is_best else ""
        print(f"{epoch+1:<8} | {train_r2:<10.4f} | {val_r2:<10.4f} | {val_mae:<10.4f} | {val_rmse:<10.4f} | {best_marker:<6}")

        if patience_counter > patience:
            print("-" * 80)
            print(f"⏹️  Early stopping at epoch {epoch+1}")
            break

    print("=" * 80)
    print(f"✅ 训练完成！")
    print(f"   最佳验证R²: {best_val_r2:.4f} (Epoch {best_epoch})")
    print(f"   最佳验证MAE: {best_val_mae:.4f}")
    print(f"   最佳验证RMSE: {best_val_rmse:.4f}")
    print("=" * 80)

    # 保存结果
    results = {
        "experiment": "Ablation: No Cross-Stream Attention",
        "best_epoch": best_epoch,
        "best_val_r2": float(best_val_r2),
        "best_val_mae": float(best_val_mae),
        "best_val_rmse": float(best_val_rmse),
        "params": sum(p.numel() for p in model.parameters())
    }

    import json
    with open('results/ablation_no_cross_attn_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ 结果已保存至: results/ablation_no_cross_attn_results.json")


if __name__ == "__main__":
    main()

