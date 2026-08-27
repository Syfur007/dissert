"""
Dataset statistics utility.

Computes per-channel pixel mean/std (Welford's parallel algorithm, fully
vectorised per image) and per-class pixel frequency for any registered
dataset.  Output is printed in YAML-pasteable format for direct insertion
into dataset configs.

Usage
-----
    # Compute stats on the train split (default) — pass any experiment
    # config that composes the dataset you want stats for (only its
    # `dataset:` section is read):
    python -m datasets.stats --config configs/experiment/mkunet/mkunet_t_clinicdb.yaml

    # Include val + test in the computation:
    python -m datasets.stats --config configs/experiment/mkunet/mkunet_t_clinicdb.yaml --split all
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow ``python -m datasets.stats`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config


def compute_dataset_stats(pairs: list, split_label: str = "split", is_multiclass: bool = False) -> dict:
    """
    Compute per-channel mean/std and per-class pixel frequency over *pairs*.

    Uses Welford's parallel merge formulation (fully vectorised per image)
    to avoid loading the entire dataset into RAM simultaneously.

    Parameters
    ----------
    pairs       : list of (img_path, mask_path)
    split_label : label used in progress output
    is_multiclass : if False (binary segmentation — the common case here),
        masks are binarized at 127 before counting, so antialiased/
        compressed source masks that aren't strictly {0, 255} (e.g.
        ClinicDB) report a clean background/foreground split instead of
        ~100+ spurious pseudo-classes, one per intermediate grey value. If
        True, raw pixel values are counted as-is (each value is a real
        class index).

    Returns
    -------
    dict with keys ``mean``, ``std``, ``class_freq``, ``n_images``.
    """
    n_channels = 3
    n_images   = 0
    n_pixels   = np.zeros(n_channels, dtype=np.float64)  # running pixel count per channel
    mean       = np.zeros(n_channels, dtype=np.float64)
    M2         = np.zeros(n_channels, dtype=np.float64)   # sum of squared deviations
    px_hist: dict[int, int] = {}

    total = len(pairs)
    report_every = max(1, total // 10)

    for i, (img_path, mask_path) in enumerate(pairs):
        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"  [WARN] skipping unreadable pair: {img_path}", file=sys.stderr)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        h, w, _ = img.shape
        n_px    = h * w

        # Parallel Welford merge: combine running stats with this image's stats.
        img_flat = img.reshape(-1, n_channels)      # (n_px, 3)
        img_mean = img_flat.mean(axis=0)            # (3,)
        img_var  = img_flat.var(axis=0)             # (3,)

        new_n   = n_pixels + n_px
        delta   = img_mean - mean
        mean   += delta * (n_px / np.maximum(new_n, 1))
        M2     += img_var * n_px + delta ** 2 * (n_pixels * n_px / np.maximum(new_n, 1))
        n_pixels = new_n
        n_images += 1

        class_mask = mask if is_multiclass else (mask > 127).astype(np.uint8)
        for cls, cnt in zip(*np.unique(class_mask, return_counts=True)):
            px_hist[int(cls)] = px_hist.get(int(cls), 0) + int(cnt)

        if (i + 1) % report_every == 0 or (i + 1) == total:
            print(f"  [{split_label}] {i+1}/{total} images processed.", file=sys.stderr)

    if n_images == 0:
        raise RuntimeError("No readable image pairs found.")

    std       = np.sqrt(M2 / np.maximum(n_pixels - 1, 1))
    total_px  = sum(px_hist.values())
    class_freq = {
        cls: cnt / total_px for cls, cnt in sorted(px_hist.items())
    }

    return {
        "mean":       mean.tolist(),
        "std":        std.tolist(),
        "class_freq": class_freq,
        "n_images":   n_images,
    }


def _get_pairs(config: dict, split: str) -> list:
    from datasets.polyp.clinicdb import ClinicDB
    from datasets.polyp.colondb  import ColonDB

    HANDLERS = {ClinicDB.NAME: ClinicDB, ColonDB.NAME: ColonDB}
    ds_cfg = config["dataset"]
    name   = ds_cfg["name"].lower()

    if name not in HANDLERS:
        raise ValueError(
            f"stats.py does not know how to enumerate '{name}'. "
            f"Registered: {list(HANDLERS)}."
        )

    handler = HANDLERS[name](ds_cfg)
    splits  = ["train", "val", "test"] if split == "all" else [split]

    pairs = []
    for sp in splits:
        ds = handler.get_dataset(sp, transform=None)
        pairs.extend(ds.pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-channel pixel mean/std and class balance for a dataset."
    )
    parser.add_argument("--config", required=True, help="Path to dataset YAML config.")
    parser.add_argument(
        "--split", default="train",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to compute stats over (default: train).",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    pairs = _get_pairs(config, args.split)
    print(
        f"Computing stats over {len(pairs)} pairs "
        f"(dataset={config['dataset']['name']}, split={args.split}) ...",
        file=sys.stderr,
    )

    is_multiclass = config.get("model", {}).get("out_channels", 1) > 1
    stats = compute_dataset_stats(pairs, split_label=args.split, is_multiclass=is_multiclass)

    fmt_mean = [round(v, 4) for v in stats["mean"]]
    fmt_std  = [round(v, 4) for v in stats["std"]]

    print("\n# Paste into your dataset config (under 'dataset:'):")
    print(f"  norm_mean: {fmt_mean}")
    print(f"  norm_std:  {fmt_std}")
    print(f"\n# Pixel class balance ({stats['n_images']} images):")
    for cls, freq in stats["class_freq"].items():
        print(f"  class {cls}: {freq:.4%}")


if __name__ == "__main__":
    main()
