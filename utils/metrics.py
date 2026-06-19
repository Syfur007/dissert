import time
import numpy as np
import torch
from loguru import logger

def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops_and_params(model, input_size=(1, 3, 256, 256)):
    """
    Calculate FLOPs and parameter counts using 'thop' or 'ptflops' packages.
    Falls back to parameter counting if profiling packages are not available.
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)
    
    # Try thop
    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return int(flops), int(params)
    except Exception:
        pass
        
    # Try ptflops
    try:
        from ptflops import get_model_complexity_info
        # Input size shape format expected is (C, H, W)
        macs, params = get_model_complexity_info(
            model, input_size[1:], as_strings=False, print_per_layer_stat=False, verbose=False
        )
        # FLOPs approximate to 2 * MACs
        return int(2 * macs), int(params)
    except Exception:
        pass
        
    # Fallback
    params = count_parameters(model)
    return 0, params

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
        gts (list of np.ndarray): Ground truth labels.
        
    Returns:
        dict: Averaged Dice, mIoU, HD95, and ASD.
    """
    dice_list = []
    iou_list = []
    hd95_list = []
    asd_list = []
    
    for p, g in zip(preds, gts):
        if p.ndim == 3:
            # Handle class channels
            if p.shape[0] == 1:
                p_sq = p.squeeze(0)
                g_sq = g.squeeze(0)
                m = get_binary_metrics(p_sq, g_sq)
                dice_list.append(m['dice'])
                iou_list.append(m['iou'])
                if m['hd95'] < 999.0: hd95_list.append(m['hd95'])
                if m['asd'] < 999.0: asd_list.append(m['asd'])
            else:
                # Multi-class average
                class_dices, class_ious, class_hd95, class_asds = [], [], [], []
                for c in range(p.shape[0]):
                    m = get_binary_metrics(p[c], g[c])
                    class_dices.append(m['dice'])
                    class_ious.append(m['iou'])
                    if m['hd95'] < 999.0: class_hd95.append(m['hd95'])
                    if m['asd'] < 999.0: class_asds.append(m['asd'])
                dice_list.append(np.mean(class_dices))
                iou_list.append(np.mean(class_ious))
                if class_hd95: hd95_list.append(np.mean(class_hd95))
                if class_asds: asd_list.append(np.mean(class_asds))
        else:
            m = get_binary_metrics(p, g)
            dice_list.append(m['dice'])
            iou_list.append(m['iou'])
            if m['hd95'] < 999.0: hd95_list.append(m['hd95'])
            if m['asd'] < 999.0: asd_list.append(m['asd'])
            
    return {
        'dice': float(np.mean(dice_list)) if dice_list else 0.0,
        'miou': float(np.mean(iou_list)) if iou_list else 0.0,
        'hd95': float(np.mean(hd95_list)) if hd95_list else 0.0,
        'asd': float(np.mean(asd_list)) if asd_list else 0.0
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
