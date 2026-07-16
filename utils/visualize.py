"""
utils/visualize.py — Diagnostic visualisation helpers for test-time analysis.

Three standalone functions that consume accumulated predictions / labels and
save publication-ready PNGs.  No display calls; all output goes to files.

Dependencies
------------
Required:  matplotlib, numpy
Optional:  seaborn  (prettier confusion-matrix heat map; falls back gracefully)
           scikit-learn (ROC / PR curve computation; silently skips if absent)
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _flat_binary(arr: np.ndarray) -> np.ndarray:
    """Return a 1-D boolean array from any mask/pred shape."""
    return arr.astype(bool).ravel()


# ---------------------------------------------------------------------------
# 1. Confusion Matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    save_path: str,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Compute and save a confusion-matrix heatmap as a PNG.

    Works for both binary (2×2) and multiclass (N×N) cases.

    For binary inputs the arrays are flattened to 1-D and thresholded at 0.5.
    For multiclass inputs the first channel or argmax label is used depending
    on whether ``preds`` elements are (C, H, W) arrays or (H, W) index maps.

    Args:
        preds:       List of per-image predictions.
        gts:         List of per-image ground-truth masks.
        save_path:   Full path (including filename) to write the PNG.
        class_names: Optional list of class label strings for axis ticks.
        normalize:   If True, normalise each row to sum to 1 (default).
        title:       Plot title.
    """
    try:
        from sklearn.metrics import confusion_matrix as sk_cm
    except ImportError:
        # Manual fallback for binary case only
        sk_cm = None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(save_path)

    # ── Flatten predictions & labels to 1-D integer arrays ────────────────
    y_pred_all, y_true_all = [], []
    for p, g in zip(preds, gts):
        p = np.asarray(p)
        g = np.asarray(g)
        if p.ndim == 3 and p.shape[0] > 1:
            # multiclass (C, H, W) → argmax index map
            p = np.argmax(p, axis=0)
            g = np.argmax(g, axis=0) if (g.ndim == 3 and g.shape[0] > 1) else g.squeeze()
        elif p.ndim == 3 and p.shape[0] == 1:
            p = (p.squeeze(0) > 0.5).astype(int)
            g = (g.squeeze(0) > 0.5).astype(int) if g.ndim == 3 else (g > 0.5).astype(int)
        else:
            p = (p > 0.5).astype(int)
            g = (g > 0.5).astype(int)
        y_pred_all.append(p.ravel())
        y_true_all.append(g.ravel())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    # ── Compute confusion matrix ───────────────────────────────────────────
    n_cls = int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(n_cls))
    if sk_cm is not None:
        cm = sk_cm(y_true, y_pred, labels=labels).astype(float)
    else:
        # Pure-numpy fallback (binary only)
        cm = np.zeros((2, 2), dtype=float)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, n_cls), max(5, n_cls - 1)))

    try:
        import seaborn as sns
        sns.heatmap(
            cm, annot=True, fmt=".2f" if normalize else "d",
            cmap="Blues", ax=ax,
            xticklabels=class_names or labels,
            yticklabels=class_names or labels,
            linewidths=0.5,
        )
    except ImportError:
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1 if normalize else None)
        plt.colorbar(im, ax=ax)
        ticks = np.arange(n_cls)
        tick_labels = class_names or [str(i) for i in labels]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i in range(n_cls):
            for j in range(n_cls):
                val = format(cm[i, j], fmt)
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. ROC Curves
# ---------------------------------------------------------------------------

def save_roc_curve(
    probs: List[np.ndarray],
    gts: List[np.ndarray],
    save_path: str,
    class_names: Optional[Sequence[str]] = None,
    title: str = "ROC Curve",
) -> None:
    """Compute and save ROC curves (one curve per class in multiclass mode).

    Args:
        probs:       List of per-image *soft probability* arrays.
                     Binary: shape (1, H, W) or (H, W), values in [0, 1].
                     Multiclass: shape (C, H, W), softmax probabilities.
        gts:         List of per-image ground-truth masks (same convention).
        save_path:   Full path to write the PNG.
        class_names: Optional label strings.
        title:       Plot title.
    """
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        return  # silently skip if sklearn absent

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(save_path)

    probs_arr = [np.asarray(p) for p in probs]
    gts_arr   = [np.asarray(g) for g in gts]

    is_multiclass = probs_arr[0].ndim == 3 and probs_arr[0].shape[0] > 1
    n_cls = probs_arr[0].shape[0] if is_multiclass else 1

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_cls, 2)))

    for c in range(n_cls):
        if is_multiclass:
            y_score = np.concatenate([p[c].ravel() for p in probs_arr])
            y_true  = np.concatenate([
                (np.argmax(g, axis=0) if (g.ndim == 3 and g.shape[0] > 1) else g.squeeze()).ravel() == c
                for g in gts_arr
            ]).astype(int)
        else:
            p0 = probs_arr[0]
            y_score = np.concatenate([
                (p.squeeze() if p.ndim == 3 else p).ravel() for p in probs_arr
            ])
            y_true  = np.concatenate([
                ((g.squeeze() if g.ndim == 3 else g) > 0.5).astype(int).ravel()
                for g in gts_arr
            ])

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        label = (class_names[c] if class_names and c < len(class_names) else f"Class {c}")
        ax.plot(fpr, tpr, color=colors[c], lw=2, label=f"{label} (AUC={roc_auc:.3f})")

        if not is_multiclass:
            break

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Precision-Recall Curves
# ---------------------------------------------------------------------------

def save_pr_curve(
    probs: List[np.ndarray],
    gts: List[np.ndarray],
    save_path: str,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Precision-Recall Curve",
) -> None:
    """Compute and save Precision-Recall curves.

    Args:
        probs:       Per-image soft probability arrays (same convention as
                     ``save_roc_curve``).
        gts:         Per-image ground-truth masks.
        save_path:   Full path to write the PNG.
        class_names: Optional label strings.
        title:       Plot title.
    """
    try:
        from sklearn.metrics import precision_recall_curve, auc as sk_auc
    except ImportError:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(save_path)

    probs_arr = [np.asarray(p) for p in probs]
    gts_arr   = [np.asarray(g) for g in gts]

    is_multiclass = probs_arr[0].ndim == 3 and probs_arr[0].shape[0] > 1
    n_cls = probs_arr[0].shape[0] if is_multiclass else 1

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_cls, 2)))

    for c in range(n_cls):
        if is_multiclass:
            y_score = np.concatenate([p[c].ravel() for p in probs_arr])
            y_true  = np.concatenate([
                (np.argmax(g, axis=0) if (g.ndim == 3 and g.shape[0] > 1) else g.squeeze()).ravel() == c
                for g in gts_arr
            ]).astype(int)
        else:
            y_score = np.concatenate([
                (p.squeeze() if p.ndim == 3 else p).ravel() for p in probs_arr
            ])
            y_true  = np.concatenate([
                ((g.squeeze() if g.ndim == 3 else g) > 0.5).astype(int).ravel()
                for g in gts_arr
            ])

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = sk_auc(recall, precision)
        label  = (class_names[c] if class_names and c < len(class_names) else f"Class {c}")
        ax.plot(recall, precision, color=colors[c], lw=2,
                label=f"{label} (AP={pr_auc:.3f})")

        if not is_multiclass:
            break

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
