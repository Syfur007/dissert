# Medical Image Segmentation Boilerplate Framework

A modular, production-ready PyTorch framework designed for training and benchmarking deep learning architectures (like U-Net and Attention U-Net) on medical image segmentation tasks.

---

## Key Features

*   **Modular Architecture**: Built from reusable blocks (`ConvBlock`, `ResBlock`, `AttentionGate`).
*   **Flexible Data Handling**: Random splitting, list-based loading, and automated K-Fold Cross-Validation.
*   **Comprehensive Benchmarking**: Computes accuracy metrics (Dice, mIoU, HD95, ASD) and complexity metrics (Parameters, FLOPs, Throughput/FPS).
*   **Pipeline Features**: Loguru logging, TensorBoard experiment tracking, checkpoint manager with training resume support, and hyperparameter search.

---

## Directory Structure

```directory
dissert/
├── configs/             # YAML configurations (base parameters & hyperparameter search)
├── datasets/            # Dataset loaders, data modules, and augmentation transforms
├── models/              # U-Net, Attention U-Net, and registry
├── utils/               # Checkpoints, logging, and evaluation metrics
├── train.py             # Model training (Standard or K-Fold CV)
├── eval.py              # Test set evaluation (Single-checkpoint or K-Fold Ensemble)
├── search.py            # Hyperparameter grid search
└── requirements.txt     # Python package requirements
```

---

## Setup & Quickstart

To operate this repository, use the `thesis` Conda environment:

```bash
# Activate your conda environment
conda activate thesis

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 1. Run Sanity Checks
Verify model building, dataset loader flow, and metrics logic before starting:
```bash
python -m unittest discover -s . -p "*test*"  # Or run custom sanity checks
```
To run the internal verification script:
```bash
python datasets/dataset.py  # Runs basic loader checks
```

---

## Usage Guide

### 2. Model Training

Training parameters are configured in `configs/base_config.yaml`. You can override configurations via command line flags:

*   **Train All Folds Sequentially (default)**:
    `configs/base_config.yaml` ships with `k_fold.enabled: true`, so this plain command runs the *full K-Fold cross-validation sweep*, not a single train/val split:
    ```bash
    python train.py --config configs/base_config.yaml --epochs 50 --batch-size 8
    ```
*   **Train a Single Fold (K-Fold CV)**:
    ```bash
    python train.py --fold 0
    ```
*   **Standard Training** (single train/val split, no cross-validation):
    Set `k_fold.enabled: false` in your config — see `configs/base_test_config.yaml` for a working example — then run:
    ```bash
    python train.py --config configs/base_test_config.yaml
    ```
*   **Resume Training**:
    ```bash
    python train.py --resume
    ```

---

### 3. Model Evaluation

Inference throughput and segmentation performance are calculated on the test set:

*   **Evaluate a Specific Checkpoint**:
    ```bash
    python eval.py --fold 0
    ```
*   **Evaluate as a K-Fold Ensemble**:
    Evaluates predictions averaged across all trained folds:
    ```bash
    python eval.py --ensemble
    ```

---

### 4. Hyperparameter Search

Grid search configurations are loaded from `configs/search_config.yaml`:
```bash
python search.py --base-config configs/base_config.yaml --search-config configs/search_config.yaml
```
Outputs trial metrics to `search_results/search_report.md` and `search_results/search_summary.csv`.
