import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from models.DualFormer_adapted import DualFormer_DualStream

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for cnn_x, flow_x, y in loader:
            cnn_x, flow_x, y = cnn_x.to(device, non_blocking=True), flow_x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                pred = model(cnn_x, flow_x)
            preds.extend(pred.cpu().numpy().tolist() if pred.dim() > 0 else [pred.cpu().item()])
            targets.extend(y.cpu().numpy().tolist() if y.dim() > 0 else [y.cpu().item()])
    targets, preds = np.array(targets), np.array(preds)
    return r2_score(targets, preds), mean_absolute_error(targets, preds), np.sqrt(mean_squared_error(targets, preds))

def evaluate_by_rainfall_level(model, loader, device):
    """按降雨强度分级评估模型性能"""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for cnn_x, flow_x, y in loader:
            cnn_x, flow_x, y = cnn_x.to(device, non_blocking=True), flow_x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                pred = model(cnn_x, flow_x)
            preds.extend(pred.cpu().numpy().tolist() if pred.dim() > 0 else [pred.cpu().item()])
            targets.extend(y.cpu().numpy().tolist() if y.dim() > 0 else [y.cpu().item()])

    preds = np.array(preds)
    targets = np.array(targets)

    levels = ['Drizzle (0-1]', 'Light (1-3]', 'Moderate (3-6]', 'Heavy (>6]']
    results = {}

    for level in levels:
        if level == 'Drizzle (0-1]':
            mask = targets <= 1
        elif level == 'Light (1-3]':
            mask = (targets > 1) & (targets <= 3)
        elif level == 'Moderate (3-6]':
            mask = (targets > 3) & (targets <= 6)
        else:
            mask = targets > 6

        if mask.sum() > 0:
            level_targets = targets[mask]
            level_preds = preds[mask]
            mae = mean_absolute_error(level_targets, level_preds)
            rmse = np.sqrt(mean_squared_error(level_targets, level_preds))
            count = mask.sum()
            percentage = (count / len(targets)) * 100
            results[level] = {'mae': mae, 'rmse': rmse, 'count': count, 'percentage': percentage}
        else:
            results[level] = {'mae': 0.0, 'rmse': 0.0, 'count': 0, 'percentage': 0.0}

    return results

def print_rainfall_level_results(results):
    """以表格形式打印分级评估结果"""
    print("\n" + "=" * 80)
    print("📊 按降雨强度分级评估结果")
    print("=" * 80)
    print(f"{'Rainfall Level':<20} | {'MAE ↓':<10} | {'RMSE ↓':<10} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 80)

    for level, metrics in results.items():
        print(f"{level:<20} | {metrics['mae']:<10.4f} | {metrics['rmse']:<10.4f} | "
              f"{metrics['count']:<10} | {metrics['percentage']:<9.1f}%")

    print("=" * 80)

def train_epoch(model, loader, optimizer, scaler, device, accumulation_steps=4):
    """Gradient accumulation: batch_size=8, accumulation=4 => effective_batch_size=32"""
    model.train()
    total_loss, preds, targets = 0, [], []
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (cnn_x, flow_x, y) in enumerate(tqdm(loader, desc="Training", leave=False)):
        cnn_x, flow_x, y = cnn_x.to(device, non_blocking=True), flow_x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Mixup augmentation (same as baseline)
        if np.random.rand() < 0.5:
            lam = np.random.beta(0.5, 0.5)
            idx = torch.randperm(y.size(0)).to(device)
            cnn_x = lam * cnn_x + (1 - lam) * cnn_x[idx]
            flow_x = lam * flow_x + (1 - lam) * flow_x[idx]
            y = lam * y + (1 - lam) * y[idx]

        # Noise augmentation (same as baseline)
        if np.random.rand() < 0.3:
            cnn_x += torch.randn_like(cnn_x) * 0.1
            flow_x += torch.randn_like(flow_x) * 0.1

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            pred = model(cnn_x, flow_x)

            # Check for NaN in predictions
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"\nWarning: NaN/Inf detected in predictions at batch {batch_idx}, skipping...")
                continue

            loss = F.huber_loss(pred, y, delta=1.0) + 0.1 * F.mse_loss(pred, y) + sum(p.pow(2.0).sum() for p in model.parameters()) * 1e-5
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        # Update every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        preds.extend(pred.detach().cpu().numpy().tolist() if pred.dim() > 0 else [pred.detach().cpu().item()])
        targets.extend(y.cpu().numpy().tolist() if y.dim() > 0 else [y.cpu().item()])

    return total_loss / len(loader), r2_score(np.array(targets), np.array(preds))

def main():
    print("="*80 + "\n🔬 DualFormer对比实验\n" + "="*80)

    # Load data (same as baseline)
    F_flow = np.load("../datatest1/train_video_optical_flow_resnet50.npy")
    df = pd.read_csv("../datatest1/train_video_label_resnet50.csv")
    y = df["RAINFALL INTENSITY"].values

    # Load train/val indices (same split as baseline)
    train_idx = np.load("../train_idx_dev.npy")
    val_idx = np.load("../val_idx_dev.npy")

    # Load CNN features (same as baseline)
    F_cnn_train = np.load("../datatest1/train_video_feature_dev.npy")
    F_cnn_val = np.load("../datatest1/val_video_feature_dev.npy")

    print(f"✅ 数据加载完成: Train{len(train_idx)}, Val{len(val_idx)}")

    # Create dataloaders (ultra-low batch size + gradient accumulation)
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(F_cnn_train),
            torch.FloatTensor(F_flow[train_idx]),
            torch.FloatTensor(y[train_idx])
        ),
        batch_size=8,  # Reduced: 32->8 (accumulation=4 => effective=32)
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(F_cnn_val),
            torch.FloatTensor(F_flow[val_idx]),
            torch.FloatTensor(y[val_idx])
        ),
        batch_size=8,  # Reduced: 32->8
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    model = DualFormer_DualStream(
        cnn_dim=2048,
        flow_dim=179,
        seq_len=15,
        d_model=256,
        n_heads=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.2
    ).to(device)

    # Memory optimization
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    print(f"✅ 模型创建完成 | 参数量: {sum(p.numel() for p in model.parameters()):,}\n" + "-"*80)

    # Optimizer and scheduler (same as baseline)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    scaler = GradScaler()

    best_val_r2, best_val_mae, best_val_rmse, best_epoch, patience_counter = -np.inf, float("inf"), float("inf"), 0, 0

    print(f"🚀 开始训练\n" + "="*80 + f"\n{'Epoch':<8} | {'Train R²':<10} | {'Val R²':<10} | {'Val MAE':<10} | {'Val RMSE':<10} | {'Best':<6} | {'Patience':<8}\n" + "-"*80)

    for epoch in range(100):
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer, scaler, device)
        val_r2, val_mae, val_rmse = evaluate(model, val_loader, device)
        scheduler.step()

        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        is_best = False
        if val_r2 > best_val_r2:
            best_val_r2, best_val_mae, best_val_rmse, best_epoch, patience_counter, is_best = val_r2, val_mae, val_rmse, epoch+1, 0, True
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), "results/best_DualFormer.pth")
        else:
            patience_counter += 1

        print(f"{epoch+1:<8} | {train_r2:<10.4f} | {val_r2:<10.4f} | {val_mae:<10.4f} | {val_rmse:<10.4f} | {'✓' if is_best else '':<6} | {patience_counter}/20")

        if patience_counter > 20:
            print("-"*80 + f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

    print("="*80 + f"\n✅ 训练完成！\n   最佳验证R²: {best_val_r2:.4f} (Epoch {best_epoch})\n   最佳验证MAE: {best_val_mae:.4f}\n   最佳验证RMSE: {best_val_rmse:.4f}\n" + "="*80)

    # 加载最佳模型进行分级评估
    print("\n" + "=" * 80)
    print("📊 加载最佳模型进行分级评估...")
    print("=" * 80)

    model.load_state_dict(torch.load('results/best_DualFormer.pth'))

    # 进行分级评估
    level_results = evaluate_by_rainfall_level(model, val_loader, device)

    # 打印结果
    print_rainfall_level_results(level_results)

    import json
    with open("results/DualFormer_results.json", "w") as f:
        json.dump({
            "model": "DualFormer",
            "best_epoch": best_epoch,
            "best_val_r2": best_val_r2,
            "best_val_mae": best_val_mae,
            "best_val_rmse": best_val_rmse,
            "params": sum(p.numel() for p in model.parameters()),
            "level_evaluation": level_results
        }, f, indent=4)
    print("✅ 结果已保存至: results/DualFormer_results.json")

if __name__ == "__main__":
    main()
