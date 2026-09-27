"""
datasets/preprocess.py — one-time dataset manifest + near-duplicate
detection.

build_manifest() computes per-image (path, subject_id, split, mask_empty,
resolution) ONCE per dataset version — a training run reads this file
rather than re-deriving the same fixed, pre-augmentation facts every epoch
inside Dataset.__getitem__ (where "mask_empty" in particular would be both
wasteful to recompute and ambiguous post-augmentation, since a crop can
move a lesion in or out of frame run to run).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def build_manifest(
    pairs_by_split: Dict[str, Sequence[Tuple[str, str]]],
    subject_id_fn: Optional[Callable[[str], str]] = None,
    out_path: Optional[str] = None,
) -> List[dict]:
    """One row per ``(img_path, mask_path)`` pair across every split.

    Row keys: ``path``, ``mask_path``, ``split``, ``subject_id``,
    ``mask_empty``, ``height``, ``width``.

    Args:
        pairs_by_split: ``{"train": [(img_path, mask_path), ...], ...}``.
        subject_id_fn: defaults to
            ``datasets.dataset._default_subject_id_fn`` (filename stem) —
            the same default ``MedicalSegmentationDataset`` uses, so a
            manifest built here and subject_ids seen at training time never
            disagree.
        out_path: when given, also writes the manifest as JSON.
    """
    from .dataset import _default_subject_id_fn

    subject_id_fn = subject_id_fn or _default_subject_id_fn

    rows: List[dict] = []
    for split, pairs in pairs_by_split.items():
        for img_path, mask_path in pairs:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found while building manifest: {mask_path}")
            h, w = mask.shape[:2]
            rows.append(
                {
                    "path": img_path,
                    "mask_path": mask_path,
                    "split": split,
                    "subject_id": subject_id_fn(img_path),
                    # Same 127 threshold Dataset.__getitem__ binarizes at —
                    # see its comment on why (antialiased/compressed source
                    # masks aren't strictly {0, 255}).
                    "mask_empty": bool(not (mask > 127).any()),
                    "height": int(h),
                    "width": int(w),
                }
            )

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    return rows


# ---------------------------------------------------------------------------
# Deduplication (phash + SSIM) — mandatory for BUSI per spec
# ---------------------------------------------------------------------------

def _phash(image: np.ndarray, hash_size: int = 8) -> int:
    """DCT-based perceptual hash, as a single Python int (its low
    ``hash_size**2`` bits are the hash). No extra dependency — the
    ``imagehash`` package isn't pinned in this repo, and the whole
    algorithm is a handful of lines on top of the already-pinned
    scipy/opencv.
    """
    import scipy.fftpack

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(
        gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(resized, axis=0), axis=1)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()

    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup(
    pairs: Sequence[Tuple[str, str]],
    phash_threshold: int = 4,
    ssim_threshold: float = 0.95,
) -> List[int]:
    """Return indices into *pairs* to exclude as near-duplicates.

    Two-stage: phash (cheap; O(n) hash computation + Hamming-distance
    comparison) narrows candidate pairs, then SSIM (expensive; only run on
    phash-flagged candidates) confirms. **Mandatory for BUSI per the spec**
    — that dataset is known to contain duplicated images across its
    benign/malignant/normal folders, and a duplicate split across
    train/test would silently inflate test-set performance.

    Keeps the *first* occurrence (by input order) of each near-duplicate
    group; returns the indices of the rest, to exclude.

    This is a one-time preprocessing step (see module docstring) — O(n²) in
    the worst case (every image within phash_threshold of every other), but
    the phash prefilter makes the expensive SSIM call rare in practice for
    a real dataset, where near-duplicates are a small fraction of all
    pairs.
    """
    from skimage.metrics import structural_similarity as ssim

    hashes: List[int] = []
    images: List[np.ndarray] = []
    for img_path, _ in pairs:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found during dedup: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        hashes.append(_phash(img))

    n = len(pairs)
    excluded: set = set()
    for i in range(n):
        if i in excluded:
            continue
        for j in range(i + 1, n):
            if j in excluded:
                continue
            if _hamming(hashes[i], hashes[j]) > phash_threshold:
                continue
            gray_i = cv2.cvtColor(cv2.resize(images[i], (256, 256)), cv2.COLOR_RGB2GRAY)
            gray_j = cv2.cvtColor(cv2.resize(images[j], (256, 256)), cv2.COLOR_RGB2GRAY)
            score = ssim(gray_i, gray_j)
            if score >= ssim_threshold:
                excluded.add(j)

    return sorted(excluded)
