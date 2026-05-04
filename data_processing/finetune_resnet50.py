"""finetune-resnet50-rainfall

ResNet50微调脚本 - 端到端学习降雨特征
策略：
1. 加载预训练ResNet50
2. 冻结前面的层，只微调layer4 + FC
3. 在降雨视频上端到端训练
4. 输出：微调后的特征提取器
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class RainfallVideoDataset(Dataset):
    """降雨视频数据集（从预处理帧加载）"""
    def __init__(self, frames_array, indices, labels, transform=None):
        self.frames = frames_array[indices]  # (N, 15, 224, 224, 3)
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frames = self.frames[idx]  # (15, 224, 224, 3)
        label = self.labels[idx]

        # 转换为tensor
        frame_tensors = []
        for frame in frames:
            tensor = self.transform(frame)
            frame_tensors.append(tensor)

        video_tensor = torch.stack(frame_tensors)  # (15, 3, 224, 224)
        return video_tensor, torch.FloatTensor([label])


class FineTunedResNet50(nn.Module):
    """微调的ResNet50模型"""
    def __init__(self, num_frames=15, freeze_layers=True):
        super().__init__()
        self.num_frames = num_frames

        # 加载预训练ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # 冻结前面的层（只微调layer4）
        if freeze_layers:
            for name, param in resnet.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        # 提取特征部分（去掉FC层）
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # 时序聚合层（简单平均池化）
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # 新的回归头（增强正则化）
        self.regressor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        """
        x: (batch, T, C, H, W)
        """
        batch_size, T, C, H, W = x.size()

        # 展平时序维度
        x = x.view(batch_size * T, C, H, W)

        # 提取特征
        features = self.feature_extractor(x)  # (batch*T, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch*T, 2048)

        # 恢复时序维度
        features = features.view(batch_size, T, 2048)  # (batch, T, 2048)

        # 时序聚合
        features = features.transpose(1, 2)  # (batch, 2048, T)
        features = self.temporal_pool(features).squeeze(-1)  # (batch, 2048)

        # 回归预测
        output = self.regressor(features)

        return output.squeeze()

    def extract_features(self, x):
        """提取特征（用于后续保存）"""
        batch_size, T, C, H, W = x.size()
        x = x.view(batch_size * T, C, H, W)

        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.squeeze(-1).squeeze(-1)

        features = features.view(batch_size, T, 2048)
        return features


def train_epoch(model, loader, optimizer, scaler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    preds, targets = [], []

    for videos, labels in tqdm(loader, desc="Training", leave=False):
        videos = videos.to(device)
        labels = labels.to(device).squeeze()

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            pred = model(videos)
            loss = F.huber_loss(pred, labels, delta=1.0) + 0.1 * F.mse_loss(pred, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        pred_np = pred.detach().cpu().numpy()
        label_np = labels.cpu().numpy()
        preds.append(pred_np.flatten())
        targets.append(label_np.flatten())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return total_loss / len(loader), r2_score(targets, preds)


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.squeeze()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                pred = model(videos)

            pred_np = pred.cpu().numpy()
            label_np = labels.numpy()
            preds.append(pred_np.flatten())
            targets.append(label_np.flatten())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return r2_score(targets, preds), mean_absolute_error(targets, preds)


def main():
    print("=" * 80)
    print("🌧️  ResNet50微调 - 端到端学习降雨特征（从预处理帧）")
    print("=" * 80)

    # 加载预处理的视频帧
    frames_file = "datatest1/video_frames_preprocessed.npy"
    print(f"加载预处理帧: {frames_file}")
    all_frames = np.load(frames_file)  # (1091, 15, 224, 224, 3)
    print(f"✅ 帧数组形状: {all_frames.shape}")

    # 加载标签
    video_dir = "D:/paper/SARID-main/SARID/video-new"
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    labels_list = [os.path.basename(f).split('_') for f in video_files]
    df = pd.DataFrame(labels_list)
    y = df.iloc[:, 1].astype(float).values  # 降雨强度

    print(f"✅ 样本数量: {len(y)}")
    print(f"✅ 标签范围: [{y.min():.2f}, {y.max():.2f}]")
    print("-" * 80)

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    print("⚙️ 开始 5 折交叉验证（微调ResNet50）...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_frames)):
        print(f"{'='*80}\n折 {fold+1}/5\n{'='*80}")

        # 创建数据集（从预处理帧）
        train_dataset = RainfallVideoDataset(all_frames, train_idx, y[train_idx])
        val_dataset = RainfallVideoDataset(all_frames, val_idx, y[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # 创建模型
        model = FineTunedResNet50(num_frames=15, freeze_layers=True).to(device)

        # 只优化未冻结的参数
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
        scaler = GradScaler()

        best_val_r2 = -np.inf
        patience_counter = 0

        for epoch in range(50):  # 微调不需要太多epoch
            train_loss, train_r2 = train_epoch(model, train_loader, optimizer, scaler, device)
            val_r2, val_mae = evaluate(model, val_loader, device)
            scheduler.step()

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f'weights/finetuned_resnet50_fold{fold+1}.pth')
            else:
                patience_counter += 1

            if patience_counter > 10:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:2d} | Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")

        print(f"✅ 折 {fold+1} 最佳R²: {best_val_r2:.4f}\n")
        fold_results.append(best_val_r2)

    # 汇总结果
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
    print(f"   预训练ResNet50（冻结）:  71.53%")
    print(f"   微调ResNet50:            {mean_r2*100:.2f}%")
    print(f"   提升: {(mean_r2 - 0.7153)*100:.2f}%")
    print("=" * 80)

    print("\n💡 下一步:")
    print("1. 如果微调有效（R² > 73%），重新提取特征")
    print("2. 使用微调后的模型提取特征，替换原有的ResNet50特征")
    print("3. 重新训练Transformer模型")


if __name__ == "__main__":
    os.makedirs('weights', exist_ok=True)
    main()
