# Deep Learning Segmentation Pipeline with K-Fold Cross Validation

Create a clean, efficient, and simple PyTorch boilerplate framework for testing deep learning architectures on medical image segmentation tasks. The framework will support modular model creation, dataset split management, optional K-Fold cross-validation, logging, experiment tracking, hyperparameter search, and standard medical image segmentation benchmarking metrics (Dice, HD95, mIoU, ASD, Params, FLOPs, Throughput).

## User Review Required

> [!IMPORTANT]
> The framework requires a robust environment with PyTorch and standard scientific computing packages (numpy, scikit-learn, medpy, albumentations, etc.). These packages are already specified in the user's `requirements.txt`.
> We will configure the datasets to point to default locations under `data/`, e.g., `data/polyp/ClinicDB`, `data/polyp/ColonDB`, etc., but they will be fully customizable via YAML configuration files.

## Proposed Changes

We will create a modular directory structure under `/home/syfur/Workspace/dissert`:
- `configs/`: Directory containing YAML configuration files.
- `models/`: Modular model blocks and architectures.
- `data/`: Dataset loaders, datamodules, and transforms.
- `utils/`: Metrics, checkpoints, and logging utilities.
- `train.py`: The main training and validation pipeline (supporting K-Fold).
- `eval.py`: The evaluation pipeline on test sets.
- `search.py`: Hyperparameter search utility.

---

### Configurations

#### [NEW] [base_config.yaml](file:///home/syfur/Workspace/dissert/configs/base_config.yaml)
Defines all default hyperparameters, including model architectures, dataset paths, training schedules, cross-validation parameters, loss functions, and logging parameters.

#### [NEW] [search_config.yaml](file:///home/syfur/Workspace/dissert/configs/search_config.yaml)
Defines the hyperparameter search grid/random configurations (e.g., learning rates, batch sizes, model backbones).

---

### Models Module

#### [NEW] [blocks.py](file:///home/syfur/Workspace/dissert/models/blocks.py)
Contains reusable PyTorch blocks:
- `ConvBlock`: Standard Convolution-BatchNorm-ReLU block.
- `ResBlock`: Residual convolution block.
- `DoubleConv`: Double convolution block (used in standard U-Net).
- `EncoderBlock`: MaxPool followed by double convolution.
- `DecoderBlock`: Up-convolution/Up-sample followed by double convolution with skip-connection.
- `AttentionBlock`: Self-attention / additive attention gate (used in Attention U-Net).

#### [NEW] [unet.py](file:///home/syfur/Workspace/dissert/models/unet.py)
A modular U-Net implementation constructed using the blocks in `blocks.py`. It will support flexible depth and channel configurations.

#### [NEW] [attention_unet.py](file:///home/syfur/Workspace/dissert/models/attention_unet.py)
An Attention U-Net implementation utilizing the reusable attention and convolutional blocks.

#### [NEW] [registry.py](file:///home/syfur/Workspace/dissert/models/registry.py)
A registry system to register and build models by name (e.g., `get_model("unet", num_classes=1)`).

---

### Data Module

#### [NEW] [dataset.py](file:///home/syfur/Workspace/dissert/data/dataset.py)
A clean, generic `MedicalSegmentationDataset` class that handles loading image-mask pairs from a folder or list of files, applying Albumentations augmentations, and formatting them for training.

#### [NEW] [datamodule.py](file:///home/syfur/Workspace/dissert/data/datamodule.py)
A data manager class that:
- Reads datasets (Polyp, Skin, Breast Cancer, etc.) based on configs.
- Automatically handles Train/Val/Test splits or lists.
- Generates data loaders for normal single runs or K-Fold indices.

#### [NEW] [transforms.py](file:///home/syfur/Workspace/dissert/data/transforms.py)
Standard training and validation augmentation pipelines using Albumentations (e.g., rotations, flips, brightness adjustment, normalizations).

---

### Utilities Module

#### [NEW] [metrics.py](file:///home/syfur/Workspace/dissert/utils/metrics.py)
Calculates key metrics:
- Dice score, Mean IoU (mIoU).
- Hausdorff Distance (HD95) and Average Surface Distance (ASD) via `medpy.metric.binary` with robust fallback.
- Throughput (frames per second / images per second).
- Model parameters and FLOPs using `thop`/`ptflops` (or custom estimators).

#### [NEW] [logger.py](file:///home/syfur/Workspace/dissert/utils/logger.py)
Sets up consistent logging with `loguru` and TensorBoard experiment tracking.

#### [NEW] [checkpoint.py](file:///home/syfur/Workspace/dissert/utils/checkpoint.py)
Handles saving checkpoints, managing the "best" model checkpoint based on a target validation metric, and resuming training from a saved checkpoint state.

---

### Pipelines

#### [NEW] [train.py](file:///home/syfur/Workspace/dissert/train.py)
The central training entry point. Based on the config, it:
- Sets random seeds.
- Discovers datasets and initializes the DataModule.
- Handles either standard Train/Val/Test split training OR K-Fold cross-validation (training K separate models and tracking average fold validation scores).
- Updates the best checkpoint and saves training progress for resume capability.

#### [NEW] [eval.py](file:///home/syfur/Workspace/dissert/eval.py)
The evaluation script. It loads a saved checkpoint, calculates validation/test set metrics (including throughput, HD95, ASD), and generates a performance report.

#### [NEW] [search.py](file:///home/syfur/Workspace/dissert/search.py)
Runs a simple hyperparameter search (grid search or random search) across values defined in the search config, launching individual training runs.

---

## Verification Plan

### Automated Tests
1. **Model Sanity Test**: Instantiate UNet and AttentionUNet, run a dummy tensor shape verification, and print FLOPs/parameters.
2. **Dataset & Loader Test**: Test loading `ClinicDB` (if present) or a mock dataset directory to verify the Albumentations pipeline and train/val split.
3. **Training & Resuming Test**: Run a 2-epoch training run, save checkpoint, resume from checkpoint, and verify training picks up at epoch 3.
4. **K-Fold Test**: Run a 3-fold cross validation test run with 1 epoch per fold to verify fold splitting, logging, and performance aggregation.
5. **Evaluation Test**: Run the evaluation script on the trained checkpoint to output a full metrics suite (Dice, mIoU, HD95, ASD, FLOPs, Throughput).

### Manual Verification
- Review the generated metrics, logs, and TensorBoard events.
- Document files and features in the final walkthrough report.
