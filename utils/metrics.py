"""
utils/metrics.py — model-profiling helpers (param count, throughput, layer
summary). Segmentation-quality metrics (Dice/IoU/HD95/ASD/...) moved to the
top-level metrics/ package in Phase 1 of IMPLEMENTATION_PLAN.md — see
metrics/aggregate.py's compute_dataset_metrics(), the direct replacement for
what used to live here as get_binary_metrics()/compute_dataset_metrics().

What's left here is explicitly Phase 10's territory (profiling/deployment
module) and stays until that phase gives it a proper home in profiling/ —
see IMPLEMENTATION_PLAN.md's Phase 10 section, which calls out
measure_throughput's warmup=5 loop and log_model_summary by name as things
it supersedes.
"""
import time

import torch
from loguru import logger

def count_parameters(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        for i, (images, _, _meta) in enumerate(dataloader):
            images = images.to(device)
            _ = model(images)
            if i >= num_warmup:
                break

    # Measurement loop
    with torch.no_grad():
        for images, _, _meta in dataloader:
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

