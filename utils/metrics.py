import time
import numpy as np
# Monkey-patch numpy.bool to fix medpy compatibility in newer numpy versions
if not hasattr(np, "bool"):
    np.bool = bool

import torch
from loguru import logger

def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_binary_metrics(pred, gt):
    """
    Compute binary segmentation metrics on numpy arrays.
    
    Args:
        pred (np.ndarray): Binary predictions (0 or 1).
        gt (np.ndarray): Binary ground truth masks (0 or 1).
        
    Returns:
        dict: Dice, IoU, HD95, and ASD metrics.
    """
    metrics = {}
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    
    # Compute intersection and totals
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()
    total_sum = pred_sum + gt_sum
    
    if total_sum == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * intersection) / total_sum
        iou = intersection / union if union > 0 else 0.0
        
    metrics['dice'] = dice
    metrics['iou'] = iou
    
    # Compute HD95 and ASD metrics
    try:
        from medpy.metric.binary import hd95, asd
        
        # Only defined if both predictions and ground truth have at least one foreground pixel
        if pred_sum > 0 and gt_sum > 0:
            metrics['hd95'] = hd95(pred_b, gt_b)
            metrics['asd'] = asd(pred_b, gt_b)
        else:
            # If both are empty, distance is 0. If only one is empty, set default/penalty distance
            metrics['hd95'] = 0.0 if (pred_sum == 0 and gt_sum == 0) else 999.0
            metrics['asd'] = 0.0 if (pred_sum == 0 and gt_sum == 0) else 999.0
    except Exception:
        # Fallback if medpy fails or is not installed
        metrics['hd95'] = 999.0
        metrics['asd'] = 999.0
        
    return metrics

def compute_dataset_metrics(preds, gts):
    """
    Calculate average metrics over a complete validation/test dataset.

    Args:
        preds (list of np.ndarray): Channel-wise or binary predictions.
            For multiclass, each element must have shape (C, H, W) with one
            channel per class.  For binary, shape is (H, W) or (1, H, W).
        gts (list of np.ndarray): Ground truth labels matching preds.

    Returns:
        dict: Keys ``dice``, ``miou``, ``hd95``, ``asd`` — macro-averaged
              across all samples.  Also includes ``per_class`` sub-dict with
              per-class mean vectors (multiclass only):
              ``{'dice': [...], 'iou': [...], 'hd95': [...], 'asd': [...]}``.
              Binary inputs return ``per_class: {}``.
    """
    dice_list = []
    iou_list  = []
    hd95_list = []
    asd_list  = []

    # Per-class accumulators: class_index → list of per-sample values
    per_class_dice: dict = {}
    per_class_iou:  dict = {}
    per_class_hd95: dict = {}
    per_class_asd:  dict = {}
    is_multiclass = False

    for p, g in zip(preds, gts):
        if p.ndim == 3:
            if p.shape[0] == 1:
                # ── binary (1, H, W) ─────────────────────────────────────
                p_sq = p.squeeze(0)
                g_sq = g.squeeze(0)
                m = get_binary_metrics(p_sq, g_sq)
                dice_list.append(m['dice'])
                iou_list.append(m['iou'])
                hd95_list.append(m['hd95'])
                asd_list.append(m['asd'])
            else:
                # ── multiclass (C, H, W) ──────────────────────────────────
                is_multiclass = True
                class_dices, class_ious, class_hd95, class_asds = [], [], [], []
                for c in range(p.shape[0]):
                    m = get_binary_metrics(p[c], g[c])
                    class_dices.append(m['dice'])
                    class_ious.append(m['iou'])
                    class_hd95.append(m['hd95'])
                    class_asds.append(m['asd'])

                    # Accumulate per-class
                    per_class_dice.setdefault(c, []).append(m['dice'])
                    per_class_iou.setdefault(c, []).append(m['iou'])
                    per_class_hd95.setdefault(c, []).append(m['hd95'])
                    per_class_asd.setdefault(c, []).append(m['asd'])

                dice_list.append(np.mean(class_dices))
                iou_list.append(np.mean(class_ious))
                hd95_list.append(np.mean(class_hd95))
                asd_list.append(np.mean(class_asds))
        else:
            # ── flat binary (H, W) ────────────────────────────────────────
            m = get_binary_metrics(p, g)
            dice_list.append(m['dice'])
            iou_list.append(m['iou'])
            hd95_list.append(m['hd95'])
            asd_list.append(m['asd'])

    # Build per-class summary (only non-empty for multiclass runs)
    per_class: dict = {}
    if is_multiclass:
        n_cls = max(per_class_dice.keys()) + 1 if per_class_dice else 0
        per_class = {
            'dice': [float(np.mean(per_class_dice.get(c, [0.0]))) for c in range(n_cls)],
            'iou':  [float(np.mean(per_class_iou.get(c,  [0.0]))) for c in range(n_cls)],
            'hd95': [float(np.mean(per_class_hd95.get(c, [0.0]))) for c in range(n_cls)],
            'asd':  [float(np.mean(per_class_asd.get(c,  [0.0]))) for c in range(n_cls)],
        }

    return {
        'dice':      float(np.mean(dice_list)) if dice_list else 0.0,
        'miou':      float(np.mean(iou_list))  if iou_list  else 0.0,
        'hd95':      float(np.mean(hd95_list)) if hd95_list else 0.0,
        'asd':       float(np.mean(asd_list))  if asd_list  else 0.0,
        'per_class': per_class,
    }

def measure_throughput(model, dataloader, device, num_warmup=5):
    """
    Measure system inference throughput in frames/second (FPS).
    
    Args:
        model (nn.Module): The model to benchmark.
        dataloader (DataLoader): DataLoader for testing.
        device (torch.device): Device running computation (CPU/CUDA).
        num_warmup (int): Warmup batches before time tracking.
    """
    model.eval()
    total_samples = 0
    total_time = 0.0
    
    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            _ = model(images)
            if i >= num_warmup:
                break
                
    # Measurement loop
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            _ = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            total_samples += images.size(0)
            total_time += (end_time - start_time)
            
    throughput = total_samples / total_time if total_time > 0 else 0.0
    return throughput

def log_model_summary(model, input_shape, logger, log_dir=None):
    """
    Profile model complexity (thop → ptflops → param count fallback),
    log a human-readable MACs/Params line, and optionally write a
    torchinfo layer table to log_dir/model_summary.txt.

    Returns:
        (flops, params) as raw integers for downstream use.
    """
    import os

    # ── 1. Complexity profiling ──────────────────────────────────────────
    flops, params = 0, count_parameters(model)
    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape).to(device)

    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        flops, params = int(flops), int(params)
    except Exception:
        try:
            from ptflops import get_model_complexity_info
            macs, params = get_model_complexity_info(
                model, input_shape[1:], as_strings=False,
                print_per_layer_stat=False, verbose=False
            )
            flops, params = int(2 * macs), int(params)
        except Exception:
            pass

    def _human(val, suffix=""):
        for unit in ("", "K", "M", "G", "T"):
            if abs(val) < 1000:
                return f"{val:.2f} {unit}{suffix}".strip()
            val /= 1000
        return f"{val:.2f} P{suffix}"

    if flops > 0:
        logger.info(f"Model Complexity | MACs: {_human(flops // 2, 'Mac')} | Params: {_human(params)}")
    else:
        logger.info(f"Model Complexity | Params: {params:,}")

    # ── 2. torchinfo layer table ─────────────────────────────────────────
    if log_dir:
        try:
            from torchinfo import summary
            s = summary(model, input_size=input_shape, verbose=0)
            summary_path = os.path.join(log_dir, "model_summary.txt")
            with open(summary_path, "w") as f:
                f.write(str(s))
            logger.info(f"Saved model layer summary → {summary_path}")
        except Exception as e:
            logger.debug(f"torchinfo summary unavailable: {e}")

    return flops, params

