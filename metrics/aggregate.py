"""
metrics/aggregate.py — the single legitimate source of dataset-level
aggregate segmentation metrics. Per the spec, nothing downstream (eval.py's
reports, Phase 9's statistics module, Phase 14's reporting layer) should
recompute its own aggregate from raw predictions — everything reads
compute_dataset_metrics()'s output or write_per_image_parquet()'s file.

Supersedes utils/metrics.py's compute_dataset_metrics(): same (preds, gts)
-> dict contract and the same binary/multiclass shape handling, plus:
precision/recall/specificity/f2/accuracy (previously utils/report.py's
independent compute_extended_metrics()), nsd, *_excluded_n counts for the
boundary metrics' new undefined-when-one-empty behaviour (see boundary.py),
dice_p5/dice_p25, and the two lesion-free-subset detection aggregates.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .boundary import asd as _asd
from .boundary import hd95 as _hd95
from .boundary import nsd as _nsd
from .detection import fpr_on_normals as _fpr_on_normals
from .detection import precision_recall_specificity_f2_accuracy as _prsfa
from .detection import specificity_on_lesion_free_subset as _spec_lesion_free
from .region import dice_iou as _dice_iou

# Documents, in one place, the convention every function in this package
# follows for an empty prediction/ground-truth mask, so a report reader
# doesn't have to dig through source to know what a reported "average HD95"
# actually averaged over. Referenced from run manifests/reports, not just
# this docstring.
EMPTY_MASK_CONVENTION: Dict[str, Any] = {
    "dice_iou_both_empty": 1.0,
    "hd95_asd_both_empty": 0.0,
    "nsd_both_empty": 1.0,
    "hd95_asd_nsd_exactly_one_empty": (
        "undefined — excluded from the dataset average, counted in the "
        "matching *_excluded_n field, not penalised with a fixed constant"
    ),
    "precision_recall_specificity_f2_accuracy_zero_denominator": 0.0,
}


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_dataset_metrics(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    probs: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute every canonical metric over a full validation/test set.

    Args:
        preds: hard per-image predictions. Binary: (H, W) or (1, H, W).
            Multiclass: (C, H, W), one channel per class.
        gts: ground truth, matching preds' shapes.
        probs: optional soft (pre-threshold) predictions, same shapes as
            preds — enables the ``ece`` field. Binary only; omitted (or
            multiclass) leaves ``ece`` out of the result rather than
            computing a value against a definition that doesn't apply.

    Returns:
        dict with ``dice``, ``miou``, ``hd95``, ``asd``, ``nsd``,
        ``hd95_excluded_n``, ``asd_excluded_n``, ``nsd_excluded_n``,
        ``dice_p5``, ``dice_p25``, ``precision``, ``recall``,
        ``specificity``, ``f2``, ``accuracy``, ``per_class`` (multiclass
        only, else ``{}``), and — binary only — ``fpr_on_normals``,
        ``specificity_lesion_free``, and (when *probs* is given) ``ece``.
    """
    dice_list: List[float] = []
    iou_list: List[float] = []
    hd95_list: List[float] = []
    asd_list: List[float] = []
    nsd_list: List[float] = []
    hd95_excluded = asd_excluded = nsd_excluded = 0

    prec_list: List[float] = []
    rec_list: List[float] = []
    spec_list: List[float] = []
    f2_list: List[float] = []
    acc_list: List[float] = []

    per_class_dice: Dict[int, list] = {}
    per_class_iou: Dict[int, list] = {}
    per_class_hd95: Dict[int, list] = {}
    per_class_asd: Dict[int, list] = {}
    is_multiclass = False

    # Flat 2-D views, accumulated only for binary samples — feeds the
    # lesion-free-subset detection aggregates below.
    flat_preds_2d: List[np.ndarray] = []
    flat_gts_2d: List[np.ndarray] = []

    for p, g in zip(preds, gts):
        if p.ndim == 3 and p.shape[0] > 1:
            # ── multiclass (C, H, W) ─────────────────────────────────────
            is_multiclass = True
            class_dices, class_ious = [], []
            class_hd95, class_asds = [], []

            for c in range(p.shape[0]):
                p2d, g2d = p[c], g[c]
                d, i = _dice_iou(p2d, g2d)
                h = _hd95(p2d, g2d)
                a = _asd(p2d, g2d)

                class_dices.append(d)
                class_ious.append(i)
                per_class_dice.setdefault(c, []).append(d)
                per_class_iou.setdefault(c, []).append(i)

                if h is not None:
                    class_hd95.append(h)
                    per_class_hd95.setdefault(c, []).append(h)
                else:
                    hd95_excluded += 1
                if a is not None:
                    class_asds.append(a)
                    per_class_asd.setdefault(c, []).append(a)
                else:
                    asd_excluded += 1

                det = _prsfa(p2d, g2d)
                prec_list.append(det["precision"])
                rec_list.append(det["recall"])
                spec_list.append(det["specificity"])
                f2_list.append(det["f2"])
                acc_list.append(det["accuracy"])

            dice_list.append(_mean(class_dices))
            iou_list.append(_mean(class_ious))
            if class_hd95:
                hd95_list.append(_mean(class_hd95))
            if class_asds:
                asd_list.append(_mean(class_asds))
        else:
            # ── binary: (1, H, W) or flat (H, W) ────────────────────────
            p2d = p.squeeze(0) if p.ndim == 3 else p
            g2d = g.squeeze(0) if g.ndim == 3 else g

            d, i = _dice_iou(p2d, g2d)
            h = _hd95(p2d, g2d)
            a = _asd(p2d, g2d)
            n = _nsd(p2d, g2d)

            dice_list.append(d)
            iou_list.append(i)
            if h is not None:
                hd95_list.append(h)
            else:
                hd95_excluded += 1
            if a is not None:
                asd_list.append(a)
            else:
                asd_excluded += 1
            if n is not None:
                nsd_list.append(n)
            else:
                nsd_excluded += 1

            det = _prsfa(p2d, g2d)
            prec_list.append(det["precision"])
            rec_list.append(det["recall"])
            spec_list.append(det["specificity"])
            f2_list.append(det["f2"])
            acc_list.append(det["accuracy"])

            flat_preds_2d.append(p2d)
            flat_gts_2d.append(g2d)

    per_class: Dict[str, list] = {}
    if is_multiclass:
        n_cls = max(per_class_dice.keys()) + 1 if per_class_dice else 0
        per_class = {
            "dice": [float(np.mean(per_class_dice.get(c, [0.0]))) for c in range(n_cls)],
            "iou": [float(np.mean(per_class_iou.get(c, [0.0]))) for c in range(n_cls)],
            "hd95": [float(np.mean(per_class_hd95.get(c, [0.0]))) for c in range(n_cls)],
            "asd": [float(np.mean(per_class_asd.get(c, [0.0]))) for c in range(n_cls)],
        }

    result: Dict[str, Any] = {
        "dice": _mean(dice_list),
        "miou": _mean(iou_list),
        "hd95": _mean(hd95_list),
        "asd": _mean(asd_list),
        "hd95_excluded_n": hd95_excluded,
        "asd_excluded_n": asd_excluded,
        "dice_p5": float(np.percentile(dice_list, 5)) if dice_list else 0.0,
        "dice_p25": float(np.percentile(dice_list, 25)) if dice_list else 0.0,
        "precision": _mean(prec_list),
        "recall": _mean(rec_list),
        "specificity": _mean(spec_list),
        "f2": _mean(f2_list),
        "accuracy": _mean(acc_list),
        "per_class": per_class,
    }

    if not is_multiclass:
        # NSD and the lesion-free-subset aggregates assume a single
        # foreground class; left out of multiclass results rather than
        # computed against a definition that doesn't apply.
        result["nsd"] = _mean(nsd_list)
        result["nsd_excluded_n"] = nsd_excluded
        if flat_preds_2d:
            result["fpr_on_normals"] = _fpr_on_normals(flat_preds_2d, flat_gts_2d)
            result["specificity_lesion_free"] = _spec_lesion_free(flat_preds_2d, flat_gts_2d)

        if probs is not None:
            from .calibration import pixelwise_ece
            result["ece"] = pixelwise_ece(probs, gts)

    return result


def write_per_image_parquet(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    path: str,
    image_ids: Optional[List[Any]] = None,
    extra_columns: Optional[Dict[str, List[Any]]] = None,
) -> str:
    """Write one row per (pred, gt) pair — per-image dice/iou/hd95/asd/nsd —
    to a Parquet file. The only legitimate source of *downstream* per-image
    statistics (Phase 9's paired significance tests read this file directly
    rather than re-deriving per-image values from a summary).

    Multiclass samples get their per-image dice/iou/hd95/asd as the
    class-mean (matching compute_dataset_metrics' per-image aggregation);
    ``nsd`` is left null for multiclass rows (see compute_dataset_metrics'
    docstring on why NSD is binary-only).

    Args:
        image_ids: optional row identifier per sample (defaults to a plain
            0-based index) — pass the actual filenames/subject_ids once
            Phase 3's ``meta`` dict is available, so a row can be traced
            back to a specific test image.
        extra_columns: optional ``{column_name: [value_per_sample]}`` to
            merge in verbatim (e.g. subject_id, source_dataset once Phase 3
            lands) without this function needing to know about them ahead
            of time.
    """
    import pandas as pd

    rows = []
    for idx, (p, g) in enumerate(zip(preds, gts)):
        is_multiclass_sample = p.ndim == 3 and p.shape[0] > 1
        if is_multiclass_sample:
            d_list, i_list, h_list, a_list = [], [], [], []
            for c in range(p.shape[0]):
                d, i = _dice_iou(p[c], g[c])
                d_list.append(d)
                i_list.append(i)
                h = _hd95(p[c], g[c])
                a = _asd(p[c], g[c])
                if h is not None:
                    h_list.append(h)
                if a is not None:
                    a_list.append(a)
            row = {
                "dice": _mean(d_list),
                "iou": _mean(i_list),
                "hd95": _mean(h_list) if h_list else None,
                "asd": _mean(a_list) if a_list else None,
                "nsd": None,
            }
        else:
            p2d = p.squeeze(0) if p.ndim == 3 else p
            g2d = g.squeeze(0) if g.ndim == 3 else g
            d, i = _dice_iou(p2d, g2d)
            row = {
                "dice": d,
                "iou": i,
                "hd95": _hd95(p2d, g2d),
                "asd": _asd(p2d, g2d),
                "nsd": _nsd(p2d, g2d),
            }

        row["image_id"] = image_ids[idx] if image_ids is not None else idx
        if extra_columns:
            for col, values in extra_columns.items():
                row[col] = values[idx]
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["image_id"] + [c for c in df.columns if c != "image_id"]
    df = df[cols]
    df.to_parquet(path, index=False)
    return path
