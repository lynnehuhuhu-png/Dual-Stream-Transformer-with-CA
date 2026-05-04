# DualTrans-CFA

Official implementation of **DualTrans-CFA**: Dual-Stream Transformer with Cross-modal Fusion Attention for video-based rainfall intensity estimation.
This repository is directly associated with the manuscript submitted to *The Visual Computer*:
**Spatio-Temporal Dual-Stream Transformer with Cross-Modal Attention for Robust Urban Rainfall Intensity Estimation**
## Model Architecture

- Dual-stream Transformer encoding (CNN features + Optical Flow features)
- **DCMA** (Bidirectional cross-modal attention for cross-stream interaction) for cross-stream interaction
- Learnable gated fusion
- Multi-scale temporal pooling
## Reproducibility
To facilitate reproducibility, this repository provides:
- implementation of DualTrans-CFA;
- comparison models used in the manuscript;
- ablation experiment scripts;
- preprocessing and feature extraction scripts;
- fixed experimental protocol with a train/validation/test split ratio of 7:1:2;
- configuration details for reproducing the reported results.

All experiments in the manuscript use a fixed random seed of 42.
## Project Structure

```
DualTrans-CFA-release/
├── model/
│   └── dual_trans_cfa.py          # Main model (DualTrans-CFA)
├── comparison_models/
│   ├── ResNet_LSTM.py
│   ├── TST.py
│   ├── PatchTST.py
│   ├── iTransformer.py
│   └── DualFormer.py
├── train/
│   ├── train_dual_trans_cfa.py    # Train our model
│   ├── train_ResNet_LSTM.py
│   ├── train_TST.py
│   ├── train_PatchTST.py
│   ├── train_iTransformer.py
│   └── train_DualFormer.py
├── data/
│   ├── preprocess_video_frames.py # Step 1: extract frames
│   ├── finetune_resnet50.py       # Step 2: finetune ResNet50
│   └── extract_features.py        # Step 3: extract CNN+flow features
└── ablation/
    ├── ablation_no_cross_attention.py
    ├── ablation_no_stats.py
    └── ablation_single_scale.py
```

## Usage

### 1. Prepare Data
#### Dataset

The experiments are conducted on the SARID dataset. Due to dataset redistribution restrictions, the raw videos are not included in this repository. Please obtain the SARID dataset from the original dataset providers.

After downloading the dataset, organize the raw videos as follows:

```text
dataset/
├── raw_videos/
├── labels/
└── metadata/
```bash
python data/preprocess_video_frames.py
python data/finetune_resnet50.py
python data/extract_features.py
```

### 2. Train DualTrans-CFA

```bash
python train/train_dual_trans_cfa.py
```

### 3. Run Comparison Experiments

```bash
python train/train_ResNet_LSTM.py
python train/train_TST.py
python train/train_PatchTST.py
python train/train_iTransformer.py
python train/train_DualFormer.py
```

## Requirements

```bash
pip install -r requirements.txt
```

## Comparison Models

| Model | Reference |
|---|---|
| ResNet-LSTM | He et al., 2016 + Hochreiter et al., 1997 |
| TST | Zerveas et al., 2021 |
| PatchTST | Nie et al., 2023 |
| iTransformer | Liu et al., 2024 |
| DualFormer | Liang et al., 2022 |
| **DualTrans-CFA (Ours)** | — |
