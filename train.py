import os
import argparse
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from tqdm import tqdm

from models import get_model
from datasets import SegmentationDataModule
from utils import (
    setup_logger, 
    TensorBoardTracker, 
    CheckpointManager, 
    compute_dataset_metrics, 
    get_flops_and_params
)

# --- CUSTOM LOSSES FOR SEGMENTATION ---

class DiceLoss(nn.Module):
    """Dice Loss for binary or multi-class image segmentation."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        if logits.shape[1] == 1:
            # Binary segmentation
            probs = torch.sigmoid(logits)
            probs = probs.view(-1)
            targets = targets.view(-1).float()
            intersection = (probs * targets).sum()
            dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
            return 1. - dice
        else:
            # Multi-class segmentation.
            # `targets` may arrive as integer class-index maps (N, H, W) or
            # (N, 1, H, W), or already as one-hot (N, C, H, W). Normalize to
            # one-hot here so this loss is safe to use standalone
            # (loss_type: 'dice'), not only when pre-converted by ComboLoss.
            num_classes = logits.shape[1]
            probs = torch.softmax(logits, dim=1)

            if targets.ndim == 4 and targets.shape[1] == num_classes:
                one_hot_targets = targets.float()
            else:
                if targets.ndim == 4 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                one_hot_targets = nn.functional.one_hot(
                    targets.long(), num_classes=num_classes
                ).permute(0, 3, 1, 2).float()

            dice_loss = 0.0
            for c in range(num_classes):
                p_c = probs[:, c].reshape(-1)
                t_c = one_hot_targets[:, c].reshape(-1)
                intersection = (p_c * t_c).sum()
                dice = (2. * intersection + self.smooth) / (p_c.sum() + t_c.sum() + self.smooth)
                dice_loss += (1. - dice)
            return dice_loss / num_classes


class CrossEntropyLossWrapper(nn.Module):
    """
    Thin wrapper around nn.CrossEntropyLoss that normalizes target shape/dtype
    before delegating. Masks loaded from disk aren't guaranteed to already be
    squeezed to (N, H, W) and cast to int64, so calling nn.CrossEntropyLoss
    directly on raw masks can throw or, worse, silently misbehave.
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        return self.ce(logits, targets.long())


class ComboLoss(nn.Module):
    """Combines BCE (or CE) and Dice Loss to leverage both voxel-wise and region-wise objectives."""
    def __init__(self, num_classes=1, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

        if num_classes == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # targets shape for BCE is (N, 1, H, W)
        # targets shape for CE is (N, H, W) containing class indices
        if logits.shape[1] == 1:
            ce_loss = self.ce(logits, targets.float())
            dice_loss = self.dice(logits, targets)
        else:
            # Multi-class setup
            # If targets has channel dimension, we need to convert to long indices for CE
            if targets.ndim == 4 and targets.shape[1] == 1:
                ce_targets = targets.squeeze(1).long()
            elif targets.ndim == 4 and targets.shape[1] > 1:
                ce_targets = targets.argmax(dim=1).long()
            else:
                ce_targets = targets.long()

            ce_loss = self.ce(logits, ce_targets)

            # DiceLoss now normalizes target format internally, so we can
            # pass the integer class-index targets straight through instead
            # of duplicating the one-hot conversion here.
            dice_loss = self.dice(logits, ce_targets)

        return self.bce_weight * ce_loss + self.dice_weight * dice_loss

# --- SEEDING FOR REPRODUCIBILITY ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- TRAINING PIPELINE LOOP ---

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger, scaler=None, grad_clip_norm=None):
    """
    scaler: a torch.cuda.amp.GradScaler instance to enable AMP, or None to train in full precision.
    grad_clip_norm: max gradient norm for clip_grad_norm_, or None to disable clipping.
    """
    model.train()
    running_loss = 0.0
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                # Gradients must be unscaled before clipping, otherwise we'd
                # be clipping against the artificially inflated scaled norm
                # rather than the true gradient norm.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device, logger, use_amp=False):
    model.eval()
    running_loss = 0.0
    
    preds_list = []
    gts_list = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            if use_amp:
                # No GradScaler needed here -- there's no backward pass to
                # protect from fp16 underflow, just the memory/speed benefit
                # of running the forward pass in mixed precision.
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            
            # Prepare predictions and ground truths for metrics
            if outputs.shape[1] == 1:
                # Binary
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).cpu().numpy().astype(np.uint8)
            else:
                # Multiclass: take the most likely class per pixel so shapes
                # match the (H, W) integer ground-truth masks. Without this
                # argmax, preds stays as (C, H, W) softmax probabilities,
                # which breaks Dice/IoU/HD95/ASD computation and therefore
                # corrupts the very metric used for checkpoint selection.
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
                
            preds_list.extend([p for p in preds])
            gts_list.extend([m.cpu().numpy().astype(np.uint8) for m in masks])
            
    val_loss = running_loss / len(dataloader.dataset)
    
    # Calculate Dice, IoU, HD95, ASD
    metrics = compute_dataset_metrics(preds_list, gts_list)
    metrics['loss'] = val_loss
    
    return metrics

def run_training(config, fold=None):
    """
    Run standard single training or train a specific fold in K-Fold cross validation.
    """
    training_cfg = config['training']
    dataset_cfg = config['dataset']
    kfold_cfg = config.get('k_fold', {})
    chk_cfg = config.get('checkpoint', {})
    log_cfg = config.get('logging', {})
    
    device = torch.device(training_cfg['device'] if torch.cuda.is_available() else "cpu")
    set_seed(training_cfg['seed'])
    
    # Prefix naming for folds
    fold_prefix = f"_fold{fold}" if fold is not None else ""
    experiment_name = f"{log_cfg['experiment_name']}{fold_prefix}"
    
    logger = setup_logger(log_cfg['log_dir'], experiment_name)
    logger.info(f"Using device: {device}")
    
    # Create datamodule and load data loaders
    dm = SegmentationDataModule(config)
    
    if fold is not None:
        logger.info(f"Initializing fold {fold}/{kfold_cfg['n_splits']-1} loaders...")
        train_loader, val_loader = dm.get_fold_loaders(fold)
    else:
        logger.info("Initializing standard train/val loaders...")
        train_loader, val_loader = dm.get_standard_loaders()
        
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model_cfg = config['model']
    model = get_model(**model_cfg).to(device)
    
    # Show parameters and FLOPs complexity
    try:
        flops, params = get_flops_and_params(model, (1, model_cfg['in_channels'], dataset_cfg['img_height'], dataset_cfg['img_width']))
        logger.info(f"Model Summary | Parameters: {params:,} | FLOPs: {flops:,}")
    except Exception as e:
        logger.warning(f"Could not compute model FLOPs summary: {e}")
        
    # Setup loss function
    if training_cfg['loss_type'] == 'bce':
        criterion = nn.BCEWithLogitsLoss() if model_cfg['out_channels'] == 1 else CrossEntropyLossWrapper()
    elif training_cfg['loss_type'] == 'dice':
        criterion = DiceLoss()
    else:
        criterion = ComboLoss(num_classes=model_cfg['out_channels'])
        
    # Setup optimizer
    if training_cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=training_cfg['lr'], weight_decay=training_cfg['weight_decay'])
    elif training_cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=training_cfg['lr'], momentum=training_cfg['momentum'], weight_decay=training_cfg['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=training_cfg['lr'], weight_decay=training_cfg['weight_decay'])
        
    # Setup scheduler
    scheduler = None
    if training_cfg['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=training_cfg['epochs'])
    elif training_cfg['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=training_cfg['lr_step_size'], gamma=training_cfg['lr_gamma'])

    # --- AMP (mixed precision) and gradient clipping, both config-driven ---
    # training.amp: bool, default False. Only takes effect on CUDA -- AMP's
    # speed/memory benefit comes from Tensor Cores, and GradScaler's loss
    # scaling logic is built around fp16 underflow on GPU, so a CPU run with
    # amp: true just falls back to full precision with a warning.
    amp_requested = training_cfg.get('amp', False)
    use_amp = amp_requested and device.type == 'cuda'
    if amp_requested and not use_amp:
        logger.warning("AMP requested in config but CUDA is not available/selected; training in full precision.")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Automatic Mixed Precision (AMP) training enabled.")

    # training.grad_clip_norm: float, default None (disabled). Typical
    # segmentation values are 1.0-5.0; start at 1.0 if unsure.
    grad_clip_norm = training_cfg.get('grad_clip_norm', None)
    if grad_clip_norm is not None:
        logger.info(f"Gradient clipping enabled with max_norm={grad_clip_norm}")

    # Checkpoint and Experiment tracking managers
    checkpoint_dir = os.path.join(chk_cfg.get('save_dir', 'checkpoints'), log_cfg['experiment_name'])
    chk_manager = CheckpointManager(
        save_dir=checkpoint_dir, 
        monitor_metric=chk_cfg.get('monitor_metric', 'val_dice'), 
        mode=chk_cfg.get('mode', 'max')
    )
    
    tracker = TensorBoardTracker(log_cfg['tb_dir'], experiment_name)
    
    start_epoch = 1
    
    # Resume training logic
    if chk_cfg.get('resume', False):
        chk_path = chk_cfg.get('checkpoint_path')
        if not chk_path:
            # Fallback to loading the last fold checkpoint
            chk_path = os.path.join(checkpoint_dir, f"last{fold_prefix}.pth")
            
        if os.path.exists(chk_path):
            start_epoch, loaded_metric, _ = chk_manager.load(chk_path, model, optimizer, scheduler)
            start_epoch += 1  # start from next epoch
            if loaded_metric is not None:
                # Restore the "best so far" tracker on chk_manager itself.
                # is_better()/best_metric (used below and as this function's
                # return value) live on chk_manager, not on a local variable,
                # so without this resumed training would silently reset the
                # best-so-far to -inf/inf and could overwrite a genuinely
                # better earlier checkpoint on the very next epoch.
                chk_manager.best_metric = loaded_metric
            # Note: chk_manager.load() does not currently restore GradScaler
            # state (its signature only takes model/optimizer/scheduler), so
            # a resumed AMP run starts with a fresh loss-scale factor rather
            # than the one in effect when training was paused. This is
            # harmless in practice -- GradScaler re-calibrates its scale
            # within a handful of iterations -- but if you want bit-for-bit
            # resume behavior, extend CheckpointManager to also persist
            # scaler.state_dict().
        else:
            logger.warning(f"No checkpoint found at {chk_path}. Starting training from scratch.")
            
    # Training Loop
    logger.info(f"Starting training loop from epoch {start_epoch} to {training_cfg['epochs']}...")
    for epoch in range(start_epoch, training_cfg['epochs'] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            scaler=scaler, grad_clip_norm=grad_clip_norm
        )
        val_metrics = validate(model, val_loader, criterion, device, logger, use_amp=use_amp)
        
        if scheduler:
            scheduler.step()
            
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        val_metrics['lr'] = current_lr
        val_metrics['train_loss'] = train_loss
        
        # Log to stdout
        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | Val mIoU: {val_metrics['miou']:.4f} | "
            f"Val HD95: {val_metrics['hd95']:.2f} | Val ASD: {val_metrics['asd']:.2f} | LR: {current_lr:.6f}"
        )
        
        # Track metrics in TensorBoard
        tracker.log_dict(val_metrics, step=epoch, prefix="epoch")
        
        # Checkpoint evaluation
        monitored_val = val_metrics[chk_manager.monitor_metric.replace("val_", "")] # extract metric name (dice, miou, loss)
        is_best = chk_manager.is_better(monitored_val)
        
        chk_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metric_val=monitored_val,
            fold=fold,
            is_best=is_best
        )
        
    tracker.close()
    logger.info(f"Training of {experiment_name} completed.")
    return chk_manager.best_metric

# --- ENTRY POINT & CONFIG CLI PARSER ---

def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to configuration file")
    parser.add_argument("--fold", type=int, default=None, help="Specific K-Fold index to train (starts from 0)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--model", type=str, default=None, help="Override model architecture name")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision training")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Override gradient clipping max norm")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Command line overrides
    if args.fold is not None:
        # Force K-Fold to be enabled internally if fold index is requested
        config['k_fold']['enabled'] = True
    if args.resume:
        config['checkpoint']['resume'] = True
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.model is not None:
        config['model']['name'] = args.model
    if args.dataset_dir is not None:
        config['dataset']['root'] = args.dataset_dir
    if args.amp:
        config['training']['amp'] = True
    if args.grad_clip_norm is not None:
        config['training']['grad_clip_norm'] = args.grad_clip_norm
        
    kfold_cfg = config.get('k_fold', {})
    
    if kfold_cfg.get('enabled', False) and args.fold is None:
        # Train all folds sequentially if fold index is not explicitly requested
        n_splits = kfold_cfg.get('n_splits', 5)
        run_folds = kfold_cfg.get('run_folds') or list(range(n_splits))
        
        print(f"Starting K-Fold Cross Validation ({n_splits} splits) over folds: {run_folds}...")
        fold_scores = []
        for f in run_folds:
            print(f"\n=================== TRAINING FOLD {f} ===================")
            best_score = run_training(config, fold=f)
            fold_scores.append(best_score)
            
        print("\n=================== K-FOLD SUMMARY ===================")
        print(f"Monitored metric ({config['checkpoint']['monitor_metric']}) across folds:")
        for idx, score in zip(run_folds, fold_scores):
            print(f"Fold {idx}: {score:.4f}")
        print(f"Mean Score: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    else:
        # Train a single standard model (or a specific single fold index)
        run_training(config, fold=args.fold)

if __name__ == "__main__":
    main()