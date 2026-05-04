import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from models.iTransformer_adapted import iTransformer

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for cnn_x, flow_x, y in loader:
            cnn_x, flow_x, y = cnn_x.to(device), flow_x.to(device), y.to(device)
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                pred = model(cnn_x, flow_x)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
    targets, preds = np.array(targets), np.array(preds)
    return r2_score(targets, preds), mean_absolute_error(targets, preds), np.sqrt(mean_squared_error(targets, preds))

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss, preds, targets = 0, [], []
    for cnn_x, flow_x, y in tqdm(loader, desc="Training", leave=False):
        cnn_x, flow_x, y = cnn_x.to(device), flow_x.to(device), y.to(device)
        if np.random.rand() < 0.5:
            lam = np.random.beta(0.5, 0.5)
            idx = torch.randperm(y.size(0)).to(device)
            cnn_x = lam * cnn_x + (1 - lam) * cnn_x[idx]
            flow_x = lam * flow_x + (1 - lam) * flow_x[idx]
            y = lam * y + (1 - lam) * y[idx]
        if np.random.rand() < 0.3:
            cnn_x += torch.randn_like(cnn_x) * 0.1
            flow_x += torch.randn_like(flow_x) * 0.1
        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            pred = model(cnn_x, flow_x)
            loss = F.huber_loss(pred, y, delta=1.0) + 0.1 * F.mse_loss(pred, y) + sum(p.pow(2.0).sum() for p in model.parameters()) * 1e-5
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item()
        preds.extend(pred.detach().cpu().numpy())
        targets.extend(y.cpu().numpy())
    return total_loss / len(loader), r2_score(targets, preds)

def main():
    print("=" * 80)
    print("iTransformer Comparison Experiment")
    print("=" * 80)
    
    F_flow = np.load("../datatest1/train_video_optical_flow_resnet50.npy")
    df = pd.read_csv("../datatest1/train_video_label_resnet50.csv")
    y = df["RAINFALL INTENSITY"].values
    train_idx = np.load("../train_idx_dev.npy")
    val_idx = np.load("../val_idx_dev.npy")
    F_cnn_train = np.load("../datatest1/train_video_feature_dev.npy")
    F_cnn_val = np.load("../datatest1/val_video_feature_dev.npy")
    print(f"Data loaded: Train={len(train_idx)}, Val={len(val_idx)}")
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(F_cnn_train), torch.FloatTensor(F_flow[train_idx]), torch.FloatTensor(y[train_idx])), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(F_cnn_val), torch.FloatTensor(F_flow[val_idx]), torch.FloatTensor(y[val_idx])), batch_size=32, shuffle=False, num_workers=0)
    
    model = iTransformer(cnn_dim=2048, flow_dim=179, seq_len=15, d_model=512, n_heads=8, e_layers=4, d_ff=2048, dropout=0.1, use_norm=True).to(device)
    print(f"Model created | Params: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 80)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    scaler = GradScaler()
    
    best_val_r2, best_val_mae, best_val_rmse, best_epoch, patience_counter = -np.inf, float("inf"), float("inf"), 0, 0
    print("Training started...")
    print("=" * 80)
    print(f"{'Epoch':<8} | {'Train R2':<10} | {'Val R2':<10} | {'Val MAE':<10} | {'Val RMSE':<10} | {'Best':<6} | {'Patience':<8}")
    print("-" * 80)
    
    for epoch in range(100):
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer, scaler, device)
        val_r2, val_mae, val_rmse = evaluate(model, val_loader, device)
        scheduler.step()
        is_best = False
        if val_r2 > best_val_r2:
            best_val_r2, best_val_mae, best_val_rmse, best_epoch, patience_counter, is_best = val_r2, val_mae, val_rmse, epoch+1, 0, True
            os.makedirs("results", exist_ok=True)
            torch.save(model.state_dict(), "results/best_iTransformer.pth")
        else:
            patience_counter += 1
        print(f"{epoch+1:<8} | {train_r2:<10.4f} | {val_r2:<10.4f} | {val_mae:<10.4f} | {val_rmse:<10.4f} | {'*' if is_best else '':<6} | {patience_counter}/20")
        if patience_counter > 20:
            print("-" * 80)
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("=" * 80)
    print("Training completed!")
    print(f"Best Val R2: {best_val_r2:.4f} (Epoch {best_epoch})")
    print(f"Best Val MAE: {best_val_mae:.4f}")
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print("=" * 80)
    
    import json
    with open("results/iTransformer_results.json", "w") as f:
        json.dump({"model": "iTransformer", "best_epoch": best_epoch, "best_val_r2": best_val_r2, "best_val_mae": best_val_mae, "best_val_rmse": best_val_rmse, "params": sum(p.numel() for p in model.parameters())}, f, indent=4)
    print("Results saved to: results/iTransformer_results.json")

if __name__ == "__main__":
    main()
