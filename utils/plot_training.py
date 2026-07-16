"""
utils/plot_training.py — Offline static training curve dumper.

Reads the TensorBoard event file written during training and dumps one
matplotlib PNG per scalar tag.  Useful for grabbing paper figures without
opening TensorBoard.

Usage (programmatic):
    from utils.plot_training import plot_training_curves
    plot_training_curves(tb_log_dir="runs/my_exp", out_dir="logs/my_exp/plots")

Usage (CLI):
    python -m utils.plot_training --tb_dir runs/my_exp --out_dir logs/my_exp/plots
"""

from __future__ import annotations

import os
import argparse
from typing import List, Optional

# Default scalar tags logged by TensorBoardCallback
_DEFAULT_TAGS = [
    "epoch/train_loss",
    "epoch/train_dice",
    "epoch/train_iou",
    "epoch/loss",      # val loss
    "epoch/dice",      # val dice
    "epoch/miou",
    "epoch/hd95",
    "epoch/asd",
    "epoch/lr",
]


def plot_training_curves(
    tb_log_dir: str,
    out_dir: str,
    tags: Optional[List[str]] = None,
    dpi: int = 150,
) -> List[str]:
    """Read a TensorBoard event file and save one PNG per scalar tag.

    Args:
        tb_log_dir: Path to the TensorBoard run directory (contains
                    ``events.out.tfevents.*`` files).
        out_dir:    Directory where PNGs are written.  Created if absent.
        tags:       Explicit list of scalar tags to plot.  Defaults to the
                    standard tags logged by this pipeline.
        dpi:        Output PNG resolution.

    Returns:
        List of paths to the saved PNG files.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
            STORE_EVERYTHING_SIZE_GUIDANCE,
        )
    except ImportError:
        # Attempt tensorboardX fallback
        try:
            from tensorboardX.event_file_loader import EventFileLoader  # noqa: F401
        except ImportError:
            pass
        print(
            "[plot_training] tensorboard package not found — cannot read event files. "
            "Install with: pip install tensorboard"
        )
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Load the event accumulator
    ea = EventAccumulator(
        tb_log_dir,
        size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE,
    )
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if not available_tags:
        print(f"[plot_training] No scalar events found in {tb_log_dir}")
        return []

    plot_tags = tags if tags is not None else _DEFAULT_TAGS
    # Keep only tags that actually exist in the event file
    plot_tags = [t for t in plot_tags if t in available_tags]
    # Also include any extra tags present in the file but not in our default list
    extra = [t for t in available_tags if t not in plot_tags]
    plot_tags = plot_tags + extra

    saved_paths: List[str] = []

    for tag in plot_tags:
        try:
            events = ea.Scalars(tag)
        except Exception:
            continue
        if not events:
            continue

        steps  = [e.step  for e in events]
        values = [e.value for e in events]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values, linewidth=2, color="#4C72B0")

        # Smoothed overlay (exponential moving average) when there are enough points
        if len(values) >= 5:
            alpha = 0.3
            smoothed = [values[0]]
            for v in values[1:]:
                smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
            ax.plot(steps, smoothed, linewidth=1.5, color="#DD8452",
                    linestyle="--", label="EMA (α=0.3)")
            ax.legend(fontsize=9)

        # Friendly axis label
        ylabel = tag.split("/")[-1].replace("_", " ").title()
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(tag, fontsize=12)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        safe_name = tag.replace("/", "_").replace(" ", "_")
        out_path  = os.path.join(out_dir, f"{safe_name}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Dump TensorBoard scalar tags to matplotlib PNGs."
    )
    parser.add_argument(
        "--tb_dir", required=True,
        help="Path to TensorBoard run directory (contains events.out.tfevents.*)",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Directory to write PNG files into.",
    )
    parser.add_argument(
        "--tags", nargs="*", default=None,
        help="Explicit list of scalar tags to plot (default: all standard pipeline tags).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    saved = plot_training_curves(
        tb_log_dir=args.tb_dir,
        out_dir=args.out_dir,
        tags=args.tags,
        dpi=args.dpi,
    )
    for p in saved:
        print(f"Saved → {p}")
    print(f"\nTotal: {len(saved)} plot(s) saved to {args.out_dir}")


if __name__ == "__main__":
    _cli()
