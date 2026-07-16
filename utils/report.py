"""
utils/report.py — Comprehensive Evaluation Report Generator
============================================================
Generates a rich, structured evaluation report including:
  - Full YAML configuration snapshot
  - Segmentation metrics (Dice, mIoU, HD95, ASD, Precision, Recall, Specificity, F2)
  - Model size information (MB / GB)
  - Complexity metrics (FLOPs, MACs, Parameters)
  - Efficiency metrics (Throughput FPS, Latency ms, GPU memory)
  - Hardware / environment context (CUDA, PyTorch version, CPU, hostname)
  - Evaluation context (timestamp, checkpoint path, number of test samples)

Usage:
    from utils.report import EvaluationReporter

    reporter = EvaluationReporter(config, args, logger)
    reporter.set_model_info(model, flops, params, throughput, checkpoint_path)
    reporter.set_eval_results(metrics, num_samples, eval_duration_s)
    reporter.save(report_dir)      # writes .md and .json
    reporter.print_console()       # prints formatted table to stdout
"""

import os
import json
import time
import platform
import socket
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from tabulate import tabulate
    _TABULATE = True
except ImportError:
    _TABULATE = False


# ---------------------------------------------------------------------------
# Helper formatters
# ---------------------------------------------------------------------------

def _human_bytes(n_bytes: float) -> str:
    """Return a human-readable size string (B → KB → MB → GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.3f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.3f} PB"


def _human_flops(flops: int) -> str:
    """Return a human-readable FLOPs/MACs string (K → M → G → T)."""
    if flops == 0:
        return "N/A"
    for unit in ("", "K", "M", "G", "T"):
        if abs(flops) < 1_000:
            return f"{flops:.2f} {unit}FLOPs".strip()
        flops /= 1_000
    return f"{flops:.2f} PFLOPs"


def _safe_fmt(value, fmt_str=".4f", suffix="") -> str:
    """Format a numeric value safely, returning 'N/A' on None or 0."""
    if value is None:
        return "N/A"
    try:
        return format(float(value), fmt_str) + suffix
    except (ValueError, TypeError):
        return "N/A"


# ---------------------------------------------------------------------------
# Extended metric computation
# ---------------------------------------------------------------------------

def compute_extended_metrics(preds: list, gts: list) -> dict:
    """
    Compute an extended set of segmentation metrics beyond Dice/mIoU.

    Adds per-image:
        - Precision  (Positive Predictive Value)
        - Recall     (Sensitivity / True Positive Rate)
        - Specificity (True Negative Rate)
        - F2-Score   (β=2, rewards recall twice as much as precision)
        - Accuracy   (pixel-level)

    All scores are macro-averaged across the test set.

    Args:
        preds: List of np.ndarray predictions (binary or single-channel).
        gts:   List of np.ndarray ground-truth masks (binary or single-channel).

    Returns:
        dict with keys: precision, recall, specificity, f2, accuracy.
    """
    prec_list, rec_list, spec_list, f2_list, acc_list = [], [], [], [], []

    for p, g in zip(preds, gts):
        # Flatten to 1-D boolean arrays
        p_b = p.astype(bool).ravel()
        g_b = g.astype(bool).ravel()

        tp = int(np.logical_and(p_b, g_b).sum())
        fp = int(np.logical_and(p_b, ~g_b).sum())
        fn = int(np.logical_and(~p_b, g_b).sum())
        tn = int(np.logical_and(~p_b, ~g_b).sum())

        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        beta2 = 4.0  # β² = 4 for F2
        denom_f2 = beta2 * precision + recall
        f2 = (1 + beta2) * precision * recall / denom_f2 if denom_f2 > 0 else 0.0

        prec_list.append(precision)
        rec_list.append(recall)
        spec_list.append(specificity)
        f2_list.append(f2)
        acc_list.append(accuracy)

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        "precision":   _mean(prec_list),
        "recall":      _mean(rec_list),
        "specificity": _mean(spec_list),
        "f2":          _mean(f2_list),
        "accuracy":    _mean(acc_list),
    }


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------

def get_model_disk_size(checkpoint_path) -> int:
    """Return on-disk checkpoint size in bytes. Accepts a single path or a
    list of paths; sums across the list (e.g. all fold checkpoints in an
    ensemble -- os.path.getsize() on a directory silently returns the
    directory inode size, not the contained files' size)."""
    paths = checkpoint_path if isinstance(checkpoint_path, (list, tuple)) else [checkpoint_path]
    total = 0
    for p in paths:
        try:
            total += os.path.getsize(p)
        except (OSError, TypeError):
            pass
    return total


def get_model_memory_size(model: nn.Module) -> int:
    """
    Estimate model size in bytes as the sum of all parameter and buffer tensors.
    This matches the in-RAM (float32) footprint, which is also the weight-only
    footprint in a checkpoint for FP32 models.
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes


def get_latency_stats(model: nn.Module, input_shape: tuple, device: torch.device,
                      n_runs: int = 50) -> dict:
    """
    Measure single-sample inference latency statistics (mean, median, std, p95).

    Args:
        model:       Evaluated model (already on `device`).
        input_shape: (C, H, W) — single sample shape.
        device:      Target device.
        n_runs:      Number of repeated forward passes.

    Returns:
        dict: mean_ms, median_ms, std_ms, p95_ms
    """
    model.eval()
    dummy = torch.randn(1, *input_shape).to(device)
    latencies = []

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_ms":   float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms":    float(np.std(latencies)),
        "p95_ms":    float(np.percentile(latencies, 95)),
    }


def get_gpu_memory_usage(device: torch.device) -> dict:
    """
    Query peak / current VRAM usage from PyTorch for CUDA devices.

    Returns dict with keys allocated_mb, reserved_mb, peak_mb.
    All values are 0 for CPU devices.
    """
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
    try:
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved  = torch.cuda.memory_reserved(device)  / (1024 ** 2)
        peak      = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return {"allocated_mb": allocated, "reserved_mb": reserved, "peak_mb": peak}
    except Exception:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}


def get_environment_info() -> dict:
    """Collect hardware / software environment metadata."""
    info = {
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname":       socket.gethostname(),
        "python_version": platform.python_version(),
        "os":             f"{platform.system()} {platform.release()}",
        "cpu":            platform.processor() or platform.machine(),
        "torch_version":  torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"]    = torch.version.cuda
        info["gpu_name"]        = torch.cuda.get_device_name(0)
        info["gpu_count"]       = torch.cuda.device_count()
        total_mem = torch.cuda.get_device_properties(0).total_memory
        info["gpu_total_memory"] = _human_bytes(total_mem)
    return info


# ---------------------------------------------------------------------------
# Main reporter class
# ---------------------------------------------------------------------------

class EvaluationReporter:
    """
    Aggregates all evaluation data and renders rich Markdown + JSON reports.

    Workflow:
        reporter = EvaluationReporter(config, args, logger)
        reporter.set_model_info(model, flops, params, throughput, checkpoint_path)
        reporter.set_eval_results(metrics, preds, gts, num_samples, eval_duration_s)
        reporter.save(report_dir)
        reporter.print_console()
    """

    def __init__(self, config: dict, args, logger=None):
        self.config  = config
        self.args    = args
        self.logger  = logger
        self._env    = get_environment_info()

        # Filled by set_model_info()
        self._model_name       = config.get("model", {}).get("name", "unknown")
        self._dataset_name     = config.get("dataset", {}).get("name", "unknown")
        self._params           = 0
        self._flops            = 0
        self._throughput       = 0.0
        self._checkpoint_path  = ""
        self._checkpoint_size  = 0
        self._model_mem_size   = 0
        self._latency          = {}
        self._gpu_mem          = {}
        self._device           = None

        # Filled by set_eval_results()
        self._metrics_base     = {}
        self._metrics_ext      = {}
        self._per_class        = {}   # per_class sub-dict from compute_dataset_metrics
        self._num_samples      = 0
        self._eval_duration_s  = 0.0
        self._is_multiclass    = False

        # Ensemble flag
        self._is_ensemble = getattr(args, "ensemble", False)

    # ------------------------------------------------------------------
    # Data population
    # ------------------------------------------------------------------

    def set_model_info(
        self,
        model: nn.Module,
        flops: int,
        params: int,
        throughput: float,
        checkpoint_path: str = "",
        measure_latency: bool = True,
    ):
        """
        Populate model-level metrics.

        Args:
            model:           The evaluated model (nn.Module).
            flops:           Pre-computed FLOPs (from get_flops_and_params).
            params:          Pre-computed parameter count.
            throughput:      Throughput in images/sec (from measure_throughput).
            checkpoint_path: Path to the loaded checkpoint file.
            measure_latency: If True, run a latency benchmark (adds ~5 sec).
        """
        self._params  = params
        self._flops   = flops
        self._throughput = throughput
        self._checkpoint_path = checkpoint_path
        self._checkpoint_size = get_model_disk_size(checkpoint_path)
        self._model_mem_size  = get_model_memory_size(model)
        self._device = next(model.parameters()).device
        self._gpu_mem = get_gpu_memory_usage(self._device)

        if measure_latency:
            device = next(model.parameters()).device
            cfg_ds = self.config.get("dataset", {})
            cfg_m  = self.config.get("model", {})
            input_shape = (
                cfg_m.get("in_channels", 3),
                cfg_ds.get("img_height", 352),
                cfg_ds.get("img_width", 352),
            )
            try:
                if self.logger:
                    self.logger.info("Measuring single-sample latency statistics...")
                self._latency = get_latency_stats(model, input_shape, device)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"Could not measure latency: {exc}")
                self._latency = {}

    def set_eval_results(
        self,
        base_metrics: dict,
        preds: list,
        gts: list,
        num_samples: int,
        eval_duration_s: float,
        is_multiclass: bool = False,
    ):
        """
        Populate evaluation results.

        Args:
            base_metrics:    Dict from compute_dataset_metrics (dice/miou/hd95/asd).
            preds:           Raw prediction list (for extended metric computation).
            gts:             Raw ground-truth list (for extended metric computation).
            num_samples:     Number of test images evaluated.
            eval_duration_s: Wall-clock seconds the evaluation loop took.
            is_multiclass:   Precision/Recall/Specificity/F2 assume binary
                             masks (bool cast); for multiclass label maps this
                             would silently collapse all foreground classes
                             into one, so they're skipped instead.
        """
        self._metrics_base    = base_metrics
        self._per_class       = base_metrics.get("per_class", {})
        self._num_samples     = num_samples
        self._eval_duration_s = eval_duration_s
        self._is_multiclass   = is_multiclass

        if preds and gts and not is_multiclass:
            try:
                self._metrics_ext = compute_extended_metrics(preds, gts)
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"Could not compute extended metrics: {exc}")
                self._metrics_ext = {}
        else:
            self._metrics_ext = {}

        # set_model_info() snapshots GPU memory before the main eval loop
        # (so latency measurement isn't contaminated by it), which means it
        # misses the loop that usually drives actual peak usage. Re-snapshot
        # now that the full pass has run.
        if self._device is not None:
            self._gpu_mem = get_gpu_memory_usage(self._device)
        else:
            self._metrics_ext = {}

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _config_yaml(self) -> str:
        """Full config as valid YAML text (handles arbitrary nesting, unlike
        a hand-rolled one-level dict loop)."""
        return yaml.safe_dump(self.config, sort_keys=False, default_flow_style=False)

    def _config_section_md(self) -> str:
        return "## Experiment Configuration\n\n```yaml\n" + self._config_yaml() + "```\n"

    def _checkpoint_display(self) -> str:
        cp = self._checkpoint_path
        if isinstance(cp, (list, tuple)):
            if not cp:
                return "N/A"
            return f"{len(cp)} fold checkpoints (dir: {os.path.dirname(cp[0])})"
        return cp or "N/A"

    def _per_class_table(self) -> list:
        """Build a per-class metric table (rows) for multiclass reports.

        Returns an empty list if no per-class data is available.
        """
        pc = self._per_class
        if not pc:
            return []

        dice_vals = pc.get("dice", [])
        iou_vals  = pc.get("iou",  [])
        hd95_vals = pc.get("hd95", [])
        asd_vals  = pc.get("asd",  [])
        n_cls     = max(len(dice_vals), len(iou_vals))
        if n_cls == 0:
            return []

        cfg_names = self.config.get("dataset", {}).get("class_names", [])

        rows = [["━━ PER-CLASS BREAKDOWN ━━", "", "", "", ""]]
        rows.append(["Class", "Dice", "IoU", "HD95", "ASD"])
        for c in range(n_cls):
            name = cfg_names[c] if cfg_names and c < len(cfg_names) else f"Class {c}"
            rows.append([
                name,
                _safe_fmt(dice_vals[c] if c < len(dice_vals) else None),
                _safe_fmt(iou_vals[c]  if c < len(iou_vals)  else None),
                _safe_fmt(hd95_vals[c] if c < len(hd95_vals) else None, ".2f"),
                _safe_fmt(asd_vals[c]  if c < len(asd_vals)  else None, ".2f"),
            ])
        return rows

    def _metrics_table(self) -> list:
        """Build a flat list-of-rows for tabulate (2-column format)."""
        m   = self._metrics_base
        ext = self._metrics_ext
        ext_note = " (N/A — multiclass)" if self._is_multiclass else ""

        rows = [
            # --- Segmentation quality ---
            ["━━ SEGMENTATION QUALITY ━━", ""],
            ["Dice / F1-Score",          _safe_fmt(m.get("dice"),  ".4f")],
            ["mean IoU (mIoU)",          _safe_fmt(m.get("miou"),  ".4f")],
            ["Precision" + ext_note,                _safe_fmt(ext.get("precision"),   ".4f")],
            ["Recall (Sensitivity)" + ext_note,      _safe_fmt(ext.get("recall"),      ".4f")],
            ["Specificity" + ext_note,               _safe_fmt(ext.get("specificity"), ".4f")],
            ["F2-Score (β=2)" + ext_note,            _safe_fmt(ext.get("f2"),          ".4f")],
            ["Pixel Accuracy" + ext_note,            _safe_fmt(ext.get("accuracy"),    ".4f")],
            ["HD95 (Hausdorff 95%)",
             f"{m.get('hd95'):.2f} px" if (m.get("hd95", 0) > 0) else "N/A"],
            ["ASD (Avg Surface Dist)",
             f"{m.get('asd'):.2f} px"  if (m.get("asd",  0) > 0) else "N/A"],

            # --- Evaluation context ---
            ["━━ EVALUATION CONTEXT ━━", ""],
            ["Model Architecture",       self._model_name],
            ["Dataset",                  self._dataset_name],
            ["Mode",                     "Ensemble" if self._is_ensemble else "Single"],
            ["Test Samples",             f"{self._num_samples:,}"],
            ["Eval Duration",            f"{self._eval_duration_s:.2f} s"],
            ["Checkpoint",               self._checkpoint_display()],

            # --- Model size ---
            ["━━ MODEL SIZE ━━", ""],
            ["Parameters (total)",       f"{self._params:,}"],
            ["Parameters (M)",           f"{self._params / 1e6:.3f} M"],
            ["Model Memory Footprint",   _human_bytes(self._model_mem_size)],
            ["Checkpoint File Size",
             _human_bytes(self._checkpoint_size) if self._checkpoint_size else "N/A"],

            # --- Complexity ---
            ["━━ COMPLEXITY ━━", ""],
            ["FLOPs",                    _human_flops(self._flops)],
            ["FLOPs (raw)",              f"{self._flops:,}" if self._flops else "N/A"],
            ["MACs (≈ FLOPs / 2)",
             _human_flops(self._flops // 2) if self._flops else "N/A"],
        ]

        # --- Efficiency ---
        rows.append(["━━ EFFICIENCY ━━", ""])
        rows.append(["Throughput (FPS)",          f"{self._throughput:.2f} img/sec"])
        if self._latency:
            rows.append(["Latency (mean)",        f"{self._latency.get('mean_ms', 0):.2f} ms"])
            rows.append(["Latency (median)",      f"{self._latency.get('median_ms', 0):.2f} ms"])
            rows.append(["Latency (std)",         f"{self._latency.get('std_ms', 0):.2f} ms"])
            rows.append(["Latency (P95)",         f"{self._latency.get('p95_ms', 0):.2f} ms"])
        if self._gpu_mem:
            rows.append(["GPU Allocated",         f"{self._gpu_mem.get('allocated_mb', 0):.1f} MB"])
            rows.append(["GPU Reserved",          f"{self._gpu_mem.get('reserved_mb', 0):.1f} MB"])
            rows.append(["GPU Peak",              f"{self._gpu_mem.get('peak_mb', 0):.1f} MB"])

        # --- Environment ---
        rows.append(["━━ ENVIRONMENT ━━", ""])
        rows.append(["Timestamp",                 self._env.get("timestamp", "N/A")])
        rows.append(["Hostname",                  self._env.get("hostname", "N/A")])
        rows.append(["OS",                        self._env.get("os", "N/A")])
        rows.append(["CPU",                       self._env.get("cpu", "N/A")])
        rows.append(["Python",                    self._env.get("python_version", "N/A")])
        rows.append(["PyTorch",                   self._env.get("torch_version", "N/A")])
        if self._env.get("cuda_available"):
            rows.append(["CUDA",                  self._env.get("cuda_version", "N/A")])
            rows.append(["GPU",                   self._env.get("gpu_name", "N/A")])
            rows.append(["GPU Memory (total)",    self._env.get("gpu_total_memory", "N/A")])
        else:
            rows.append(["CUDA", "Not available"])

        return rows

    def _metrics_table_2col(self) -> list:
        """Alias kept for backward compat; returns same as _metrics_table."""
        return self._metrics_table()

    # ------------------------------------------------------------------
    # Public output methods
    # ------------------------------------------------------------------

    def print_console(self):
        """Print a formatted table to stdout."""
        rows   = self._metrics_table()
        header = ["Metric", "Value"]

        border = "=" * 68
        print(f"\n{border}")
        exp_name = self.config.get("logging", {}).get("experiment_name", "Evaluation")
        print(f"  EVALUATION REPORT — {exp_name}".center(68))
        print(border)
        print("\n-- Experiment Configuration --\n")
        print(self._config_yaml())
        print(border)

        if _TABULATE:
            print(tabulate(rows, headers=header, tablefmt="github"))
        else:
            # Fallback plain text
            print(f"{'Metric':<40} {'Value'}")
            print("-" * 68)
            for row in rows:
                print(f"{row[0]:<40} {row[1]}")

        # Per-class breakdown (multiclass only)
        pc_rows = self._per_class_table()
        if pc_rows:
            print("\n" + border)
            print("  PER-CLASS METRIC BREAKDOWN".center(68))
            print(border)
            if _TABULATE:
                print(tabulate(pc_rows[2:], headers=pc_rows[1], tablefmt="github"))
            else:
                print(f"{'Class':<25} {'Dice':>8} {'IoU':>8} {'HD95':>8} {'ASD':>8}")
                print("-" * 60)
                for row in pc_rows[2:]:
                    print(f"{row[0]:<25} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")

        print(border + "\n")

    def save(self, report_dir: str, filename_prefix: str = ""):
        """
        Save a Markdown report and a JSON data dump.

        Args:
            report_dir:      Directory to write reports into.
            filename_prefix: Optional prefix (e.g. experiment name).
        """
        os.makedirs(report_dir, exist_ok=True)
        ensemble_tag = "ensemble_" if self._is_ensemble else ""
        base_name = f"{filename_prefix}_{ensemble_tag}report" if filename_prefix else f"{ensemble_tag}report"

        md_path   = os.path.join(report_dir, f"{base_name}.md")
        json_path = os.path.join(report_dir, f"{base_name}.json")

        self._write_markdown(md_path)
        self._write_json(json_path)

        if self.logger:
            self.logger.info(f"Saved Markdown report  → {md_path}")
            self.logger.info(f"Saved JSON data dump   → {json_path}")

    def _write_markdown(self, path: str):
        exp_name = self.config.get("logging", {}).get("experiment_name", "Evaluation")
        rows     = self._metrics_table()
        header   = ["Metric", "Value"]

        with open(path, "w") as fh:
            fh.write(f"# Evaluation Report — {exp_name}\n\n")
            fh.write(f"*Generated at {self._env.get('timestamp', '')}*\n\n")

            fh.write("---\n\n")

            # Metrics table
            fh.write("## Results Summary\n\n")
            if _TABULATE:
                fh.write(tabulate(rows, headers=header, tablefmt="github"))
            else:
                fh.write(f"| {'Metric':<42} | {'Value':<30} |\n")
                fh.write(f"|{'-'*44}|{'-'*32}|\n")
                for row in rows:
                    fh.write(f"| {row[0]:<42} | {row[1]:<30} |\n")
            fh.write("\n\n---\n\n")

            # Per-class breakdown
            pc_rows = self._per_class_table()
            if pc_rows:
                fh.write("## Per-Class Metrics\n\n")
                pc_header = pc_rows[1]  # ["Class", "Dice", "IoU", "HD95", "ASD"]
                pc_data   = pc_rows[2:]
                if _TABULATE:
                    fh.write(tabulate(pc_data, headers=pc_header, tablefmt="github"))
                else:
                    fh.write(f"| {'Class':<25} | {'Dice':>8} | {'IoU':>8} | {'HD95':>8} | {'ASD':>8} |\n")
                    fh.write(f"|{'-'*27}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|\n")
                    for row in pc_data:
                        fh.write(f"| {row[0]:<25} | {row[1]:>8} | {row[2]:>8} | {row[3]:>8} | {row[4]:>8} |\n")
                fh.write("\n\n---\n\n")

            # Full config
            fh.write(self._config_section_md())

    def _write_json(self, path: str):
        # Strip per_class from the flat metrics dict (it lives under its own key)
        base_flat = {k: v for k, v in self._metrics_base.items() if k != "per_class"}
        data = {
            "config":      self.config,
            "experiment":  self.config.get("logging", {}).get("experiment_name", ""),
            "timestamp":   self._env.get("timestamp", ""),
            "is_ensemble": self._is_ensemble,
            "is_multiclass": self._is_multiclass,
            "checkpoint":  self._checkpoint_path,
            "num_samples": self._num_samples,
            "eval_duration_s": self._eval_duration_s,
            "metrics": {
                **base_flat,
                **self._metrics_ext,
            },
            "per_class_metrics": self._per_class,
            "model": {
                "name":              self._model_name,
                "params":            self._params,
                "params_M":          round(self._params / 1e6, 4),
                "flops":             self._flops,
                "flops_human":       _human_flops(self._flops),
                "memory_bytes":      self._model_mem_size,
                "memory_human":      _human_bytes(self._model_mem_size),
                "checkpoint_bytes":  self._checkpoint_size,
                "checkpoint_human":  _human_bytes(self._checkpoint_size) if self._checkpoint_size else "N/A",
            },
            "efficiency": {
                "throughput_fps": self._throughput,
                "latency":        self._latency,
                "gpu_memory_mb":  self._gpu_mem,
            },
            "environment": self._env,
        }

        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)