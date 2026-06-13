# EMCAD-Style Repo Skeleton — Architecture Notes (Revised)

> **Revision purpose:** Weak links identified and patched. Alternatives chosen for simplicity and reproducibility without rewriting the core design.

---



## 1. Top-Level Pipeline Split

The repository has two separate experimentation tracks:

- **3D** medical segmentation — Synapse and ACDC datasets.
- **2D** binary segmentation — polyp datasets.

Both tracks share `lib/networks.py` for the model wrapper and `utils/utils.py` for losses and metrics. Dataset loading, training loops, evaluation, and checkpoint paths are track-specific.

**Rule:** Any logic used by both tracks belongs in `utils/`. Any logic specific to one track stays in its own files.

---

## 2. Config and Run Identity

### Problem (original)
The experiment name encodes encoder, kernel sizes, dilation mode, aggregation mode, LGAG kernel size, expansion factor, activation, supervision mode, dataset, image size, batch size, learning rate, seed, and run index — all concatenated into a directory name. This breaks on Windows (260-char path limit), makes `ls` unreadable, and means one argument rename silently creates a new experiment directory instead of continuing an old one.

### Fix
Each experiment gets a **short slug** (8-char hash of the full config) plus a human-readable prefix. The full config is saved as `config.yaml` inside the experiment folder at launch time.

```
snapshots/
  synapse_a3f9c1b2/
    config.yaml        ← full hyperparameter record
    log.txt
    log/               ← TensorBoard
    best.pth
    last.pth
```

**`utils/paths.py`** — single source of truth for all path construction:

```python
import hashlib, yaml
from pathlib import Path

def make_run_id(cfg: dict, prefix: str) -> str:
    blob = yaml.dump(cfg, sort_keys=True).encode()
    slug = hashlib.sha256(blob).hexdigest()[:8]
    return f"{prefix}_{slug}"

def snapshot_dir(base: str, run_id: str) -> Path:
    return Path(base) / run_id

def best_ckpt(snap: Path) -> Path:
    return snap / "best.pth"

def last_ckpt(snap: Path) -> Path:
    return snap / "last.pth"
```

Both `train_*.py` and `test_*.py` import from `utils/paths.py`. The `config.yaml` written at launch time is the authoritative record; test scripts read it back to reconstruct paths rather than re-parsing CLI args.

---

## 3. 3D Synapse / ACDC Wiring

### Entry point

- `train_synapse.py` is the main launcher.
- It loads a base `configs/synapse_default.yaml`, then applies CLI overrides.
- It calls `make_run_id()` and `snapshot_dir()` from `utils/paths.py`.
- It saves the resolved config to `snapshot_dir/config.yaml` before training starts.
- It instantiates `EMCADNet` and calls `trainer_synapse()` in `trainer.py`.

### Data flow

- `utils/dataset_synapse.py` defines `Synapse_dataset`.
- `utils/dataset_ACDC.py` defines `ACDCdataset`.
- Training transforms: random rotation, flip, resize.
- Each sample returns `image`, `label`, `case_name`.
- Validation and test loaders use `batch_size=1`.

### Model flow

- `lib/networks.py` defines `EMCADNet`.
- The 1→3 channel stem is an explicit constructor kwarg (`stem_channels=3`), not a silent internal conversion. Pass `stem_channels=1` to skip it.
- Backbone is selected from PVTv2 or ResNet variants via config.
- Features pass to the EMCAD decoder, which returns four tensors at different scales.
- Output heads map decoder features to class logits.
- Final prediction uses the last (full-resolution) output tensor.

### Training flow

- `trainer.py` calls `setup_logging(snapshot_path)` from `utils/logging.py`.
- `utils/logging.py` creates both the `log.txt` file handler and the TensorBoard writer, and returns both. This is the only place either is initialized.
- Optimizer: AdamW.
- Loss: cross-entropy + Dice, combined via supervision mode setting.
- The training loop logs scalar loss and learning rate through the unified logger.
- Validation runs after each epoch via `evaluate_volume(mode="val")`.

### Checkpointing

- `last.pth` is saved every epoch.
- `best.pth` is saved when validation performance improves.
- `epoch_N.pth` is saved at configured intervals and at the final epoch.
- All checkpoint paths come from `utils/paths.py`.
- Inference calls `best_ckpt(snap)` first; falls back to `last_ckpt(snap)`.

---

## 4. 2D Polyp Wiring

### Entry point

- `train_polyp.py` trains a **single run** identified by a config slug.
- A separate `run_polyp_replicates.sh` (or equivalent launcher) calls `train_polyp.py` five times with different seeds. This makes each run independently restartable and log-separable.

```bash
# run_polyp_replicates.sh
for SEED in 0 1 2 3 4; do
  python train_polyp.py --config configs/polyp_default.yaml --seed $SEED
done
```

### Data flow

- `utils/dataloader_polyp.py` defines `PolypDataset` and `get_loader()`.
- Images are read with OpenCV; masks are converted to binary targets.
- Training uses augmentation when enabled.
- Test-time samples return image, mask, original shape, and filename.

### Model flow

- Same `EMCADNet` wrapper, single-channel binary head.
- Multiple input scales are handled inside the loader, not the model.
- Training and testing consume only the final output tensor.

### Training flow

- AdamW + cosine scheduler.
- Structure loss: weighted BCE + IoU.
- Validation runs on both `test` and `val` splits each epoch.
- Best validation Dice triggers `best.pth` save.
- The test Dice at the best validation epoch is tracked and logged.

### Checkpointing and reporting

- `last.pth` every epoch; `best.pth` on improvement.
- `test_polyp.py` reads the checkpoint path from `utils/paths.py` using the same config and seed.
- Per-case results exported to Excel; summary row appended to a persistent workbook.
- Prediction masks written to `predictions/<run_id>/`.

---

## 5. Shared Utilities (`utils/`)

### `utils/paths.py`
Single source of truth for experiment slugs and all checkpoint/log paths. Imported by every train and test script.

### `utils/logging.py`
Single setup function used by all train scripts:

```python
def setup_logging(snapshot_path: str):
    """Returns (python_logger, tensorboard_writer)."""
    ...
```

No train script should create its own file handler or TensorBoard writer directly.

### `utils/utils.py`
Shared losses, metrics, seed utility, and model inspection helpers:

- `DiceLoss` (3D path)
- `structure_loss()` (2D path, BCE + IoU)
- `set_seed(seed)` — call once at the start of every train script
- `count_params(model)` and `count_flops(model, input_size)`
- `evaluate_volume(model, loader, mode, save_preds=False)` — unified validation/test function

### `utils/dataset_synapse.py`, `utils/dataset_ACDC.py`, `utils/dataloader_polyp.py`
Dataset-specific logic stays here. No dataset knowledge belongs in the model or trainer.

---

## 6. Losses and Metrics

### 3D path
- `DiceLoss` in `utils/utils.py`.
- `evaluate_volume(mode="val")` — per-class Dice, no file writing.
- `evaluate_volume(mode="test", save_preds=True)` — adds prediction saving and HD95.

### 2D path
- `structure_loss()` in `utils/utils.py`.
- Dice and IoU helpers in `utils/utils.py`.
- `test_polyp.py` computes Dice, IoU, sensitivity, specificity, precision, HD95.

---

## 7. Logging Conventions

Both tracks use identical logging setup via `utils/logging.py`:

| Artifact | Location |
|----------|----------|
| Python log | `snapshots/<run_id>/log.txt` |
| TensorBoard | `snapshots/<run_id>/log/` |
| Polyp run logs | `logs/<run_id>.log` (symlink to snapshot log) |

Logged values per step: learning rate, total loss, component losses.  
Logged values per epoch: validation Dice (mean and per-class for 3D), validation IoU (2D).

---

## 8. Extension Points for Experiments

To test new architectures, change only the relevant file:

| Goal | File to change |
|------|---------------|
| Swap or compose encoders / output heads | `lib/networks.py` |
| New decoder blocks or gating modules | `lib/decoders.py` |
| New supervision strategies or branch mixing | `trainer.py` |
| New data formats or augmentations | `utils/dataset_*.py` |
| New metrics or loss terms | `utils/utils.py` |
| New experiment naming or path rules | `utils/paths.py` |
| New CLI flags | `train_*.py` and `test_*.py` |

---

## 9. Skeleton Checklist for a New Repo

Copy this structure:

```
project/
├── configs/
│   ├── synapse_default.yaml
│   ├── acdc_default.yaml
│   └── polyp_default.yaml
├── lib/
│   ├── networks.py          # EMCADNet wrapper (thin)
│   └── decoders.py          # decoder blocks and gating
├── utils/
│   ├── paths.py             # ← NEW: all path construction
│   ├── logging.py           # ← NEW: unified logging setup
│   ├── utils.py             # losses, metrics, set_seed, evaluate_volume
│   ├── dataset_synapse.py
│   ├── dataset_ACDC.py
│   └── dataloader_polyp.py
├── train_synapse.py         # loads config → make_run_id → trainer
├── train_polyp.py           # single run; seeds from CLI
├── run_polyp_replicates.sh  # ← NEW: outer loop for 5 runs
├── test_synapse.py
├── test_polyp.py
├── trainer.py               # logging, optimizer, val loop
├── snapshots/               # created at runtime
└── logs/                    # created at runtime
```

### Non-negotiable rules

1. **One path construction function.** `utils/paths.py` is the only place experiment slugs and checkpoint paths are computed. Train and test scripts import it; they do not re-derive paths from arguments.
2. **Config first, CLI second.** Every script loads a base YAML; CLI flags are overrides only. The resolved config is saved to `snapshot_dir/config.yaml` before any training starts.
3. **One logging setup.** `utils/logging.py` initializes both file and TensorBoard logging. No script creates its own handlers.
4. **Seed up front.** Every train script calls `set_seed(cfg.seed)` as the first statement after config resolution.
5. **Dataset logic stays in dataset files.** No preprocessing, normalization, or augmentation lives in the model or trainer.
6. **Thin model wrapper.** `lib/networks.py` selects backbone and wires decoder. Block-level experiments stay in `lib/decoders.py`.
7. **Single evaluate function.** `evaluate_volume()` in `utils/utils.py` handles both validation and test; the `save_preds` flag controls file output.
8. **Run replication is external.** `train_polyp.py` runs one replicate. The shell script or job scheduler drives multiple seeds.
