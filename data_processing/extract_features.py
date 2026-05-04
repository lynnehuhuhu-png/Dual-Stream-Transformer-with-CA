"""
按fold提取微调特征（防止数据泄露）
关键：每个fold只提取训练集特征，验证集使用预训练特征

针对RTX 3060 12GB优化：
- batch_size=8（平衡速度和显存）
- 自动清理显存
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FineTunedResNet50(nn.Module):
    """微调的ResNet50模型（与finetune_resnet50_rainfall.py完全一致）"""
    def __init__(self, num_frames=15):
        super().__init__()
        self.num_frames = num_frames

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def extract_features(self, x):
        """提取特征（不经过回归头）"""
        batch_size, T, C, H, W = x.size()
        x = x.view(batch_size * T, C, H, W)

        with torch.no_grad():
            features = self.feature_extractor(x)
            features = features.squeeze(-1).squeeze(-1)

        features = features.view(batch_size, T, 2048)
        return features


def extract_features_for_fold(model, frames_array, indices, batch_size=8):
    """为指定的索引提取特征"""
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_features = []
    selected_frames = frames_array[indices]  # 只选择训练集的帧

    for i in tqdm(range(0, len(selected_frames), batch_size), desc="提取特征"):
        batch_frames = selected_frames[i:i+batch_size]  # (batch, 15, 224, 224, 3)

        # 转换为tensor
        batch_tensors = []
        for video_frames in batch_frames:
            frame_tensors = []
            for frame in video_frames:
                tensor = transform(frame)
                frame_tensors.append(tensor)
            batch_tensors.append(torch.stack(frame_tensors))

        batch_tensor = torch.stack(batch_tensors).to(device)  # (batch, 15, 3, 224, 224)

        # 提取特征
        with torch.no_grad():
            features = model.extract_features(batch_tensor)  # (batch, 15, 2048)

        all_features.append(features.cpu().numpy())

        # 清理显存
        del batch_tensor, features
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return np.concatenate(all_features, axis=0)


def main():
    print("=" * 80)
    print("🚀 按fold提取微调特征（防数据泄露版本）")
    print("=" * 80)

    # 配置
    FRAMES_FILE = "datatest1/video_frames_preprocessed.npy"
    OUTPUT_DIR = "datatest1"
    BATCH_SIZE = 8  # 针对3060 12GB优化

    # 检查文件是否存在
    if not os.path.exists(FRAMES_FILE):
        print(f"❌ 预处理帧文件不存在: {FRAMES_FILE}")
        print("请先运行 preprocess_video_frames.py 预处理视频")
        return

    # 加载预处理帧
    print(f"\n加载预处理帧: {FRAMES_FILE}")
    all_frames = np.load(FRAMES_FILE)
    print(f"✅ 帧数组形状: {all_frames.shape}")

    # 加载标签（用于KFold划分）
    import glob
    video_dir = "D:/paper/SARID-main/SARID/video-new"
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    labels_list = [os.path.basename(f).split('_') for f in video_files]
    df = pd.DataFrame(labels_list)
    y = df.iloc[:, 1].astype(float).values

    print(f"✅ 样本数量: {len(y)}")
    print("-" * 80)

    # 使用相同的KFold划分（random_state=42，与微调保持一致）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("⚙️ 开始按fold提取特征...\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_frames)):
        print(f"{'='*80}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*80}")
        print(f"训练集样本数: {len(train_idx)}")
        print(f"验证集样本数: {len(val_idx)}")

        # 检查模型文件
        model_path = f'weights/finetuned_resnet50_fold{fold+1}.pth'
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请先运行 finetune_resnet50_rainfall.py 训练模型")
            continue

        # 加载该fold的微调模型
        print(f"加载模型: {model_path}")
        model = FineTunedResNet50(num_frames=15).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ 模型加载完成")

        # 只提取训练集特征
        print(f"\n提取训练集特征 (batch_size={BATCH_SIZE})...")
        train_features = extract_features_for_fold(model, all_frames, train_idx, batch_size=BATCH_SIZE)

        # 保存
        output_file = os.path.join(OUTPUT_DIR, f'train_video_feature_finetuned_fold{fold+1}.npy')
        np.save(output_file, train_features)

        print(f"✅ Fold {fold+1} 训练集特征: {train_features.shape}")
        print(f"✅ 已保存到: {output_file}\n")

        # 清理显存
        del model, train_features
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("=" * 80)
    print("✅ 所有fold的特征提取完成!")
    print("=" * 80)

    print("\n📊 生成的文件:")
    for fold in range(1, 6):
        output_file = os.path.join(OUTPUT_DIR, f'train_video_feature_finetuned_fold{fold}.npy')
        if os.path.exists(output_file):
            features = np.load(output_file)
            print(f"  Fold {fold}: {output_file} - {features.shape}")

    print("\n💡 下一步:")
    print("1. 运行 Bestbaseline_finetuned.py 训练Transformer")
    print("2. 验证集将使用预训练ResNet50特征（无数据泄露）")
    print("=" * 80)


if __name__ == "__main__":
    main()
