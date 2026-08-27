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

Pinned to `torch==1.13.1` / `torchvision==0.14.1`. The plain `pip install -r requirements.txt`
above pulls CPU-only wheels; for a CUDA build, install torch/torchvision from the CUDA wheel
index first (see the comment block at the top of `requirements.txt` for the exact command —
cu117 or cu116 depending on your driver) before installing the rest of the file.

`requirements.txt` also pins `mamba-ssm==1.0.1` / `causal-conv1d==1.1.1` (Phase 6's Mamba/VSS
model family). These are CUDA extensions with no prebuilt wheels on PyPI — `pip install
mamba-ssm` will try to build from source and typically fails without `nvcc` and a matching CUDA
toolkit. Download the prebuilt wheel matching your GPU's CUDA version, torch 1.13, your Python
version, and cxx11-ABI setting from the projects' GitHub Releases instead (URLs and matching
notes in `requirements.txt`'s comment block), and `pip install <wheel file>` directly. If no
matching wheel exists for your deployment environment, skip both packages — the Mamba model
family falls back to a pure-PyTorch reference scan (`models/auxiliary/ss2d_ref.py`, Phase 6)
automatically when `mamba_ssm` isn't importable.

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

Training parameters are configured in `configs/base.yaml`, composed with a dataset fragment
(`configs/dataset/*.yaml`) and a model-size fragment (`configs/model/<family>/*.yaml`) — see any
file under `configs/experiment/` for a working example of the `compose:` list. You can override
configurations via command line flags:

*   **Train All Folds Sequentially (default)**:
    `configs/base.yaml` ships with `k_fold.enabled: true`, so this plain command runs the *full
    K-Fold cross-validation sweep*, not a single train/val split:
    ```bash
    python train.py --config configs/experiment/mkunet/mkunet_t_clinicdb.yaml --epochs 50 --batch-size 8
    ```
*   **Train a Single Fold (K-Fold CV)**:
    ```bash
    python train.py --fold 0
    ```
*   **Standard Training** (single train/val split, no cross-validation):
    Set `k_fold.enabled: false` in your config, then run as usual:
    ```bash
    python train.py --config configs/experiment/mkunet/mkunet_t_clinicdb.yaml
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
python search.py --base-config configs/base.yaml --search-config configs/search_config.yaml
```
Outputs trial metrics to `search_results/search_report.md` and `search_results/search_summary.csv`.
