"""
datasets/splits.py — leakage guards and published-split utilities.

Phase 3 of IMPLEMENTATION_PLAN.md: a leakage guard is meaningless if only
new datasets carry it, so every function here is meant to run against any
dataset's actual (subject_id, split) assignment — including the existing
ClinicDB/ColonDB handlers, not just BUSI/ISIC18.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class LeakageError(Exception):
    """Raised when the same subject appears in more than one split."""


class ExternalDatasetError(Exception):
    """Raised when a train/val loader is requested for a dataset marked
    external (held-out-evaluation-only) — see
    datasets.datamodule._GenericHandler's ``external`` flag."""


class TestLoaderGuardError(Exception):
    """Raised by datasets.datamodule.*.get_test_loader() when the supplied
    token wasn't minted by orchestration.ledger.LedgerWriter.issue_test_token()."""
    # Tells pytest not to try collecting this as a test class just because
    # its name starts with "Test" — it's an exception, not a test.
    __test__ = False


def assert_no_subject_overlap(
    train_ids: Iterable[str],
    val_ids: Iterable[str],
    test_ids: Iterable[str] = (),
) -> None:
    """Raise LeakageError if any subject_id appears in more than one of
    train/val/test.

    *test_ids* defaults to empty since K-Fold's train/val split has no test
    set of its own at this granularity (the dataset handler's
    get_kfold_pairs() already excludes the test split entirely — see
    datasets/polyp/clinicdb.py's get_kfold_pairs() docstring).
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlaps = {
        "train/val": train_set & val_set,
        "train/test": train_set & test_set,
        "val/test": val_set & test_set,
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        detail = "; ".join(
            f"{k}: {sorted(v)[:10]}{' ...' if len(v) > 10 else ''}" for k, v in bad.items()
        )
        raise LeakageError(f"Subject overlap detected between splits — {detail}")


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_published_split(list_path: str) -> Tuple[List[str], str]:
    """Read a ``train_list_*.txt``-style file (one filename per line) —
    formalises what datasets/polyp/{clinicdb,colondb}.py already do ad hoc
    — and return ``(names, sha1_of_file)``. The hash lets a run manifest
    record exactly which version of a published split file produced a
    run's train/val/test membership, so a silently-edited list file shows
    up as a manifest mismatch rather than an untraceable result drift.
    """
    with open(list_path) as f:
        names = [line.strip() for line in f if line.strip()]
    return names, _sha1_file(list_path)


def duplicate_cross_check(
    pairs_by_split: Dict[str, Sequence[Tuple[str, str]]],
    hash_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Hash every image across every split and report any content-identical
    duplicate that landed in more than one split.

    Closes the gap in ``_GenericHandler``'s unguarded random shuffle
    (``datasets/datamodule.py``): a byte-identical image can still be
    assigned to both train and test purely by chance under a random split,
    which a filename-based (or even subject_id-based, if subject_id is
    itself derived from the filename) leakage check can't catch, since the
    filenames genuinely differ.

    Args:
        pairs_by_split: ``{"train": [(img_path, mask_path), ...], ...}``.
        hash_fn: override the per-image hash (defaults to exact-duplicate
            SHA1 of file bytes). Pass ``datasets.preprocess._phash`` (via a
            small wrapper) for a near-duplicate, not just exact-duplicate,
            check — exact hashing is what this function defaults to because
            it's cheap enough to always run, unlike phash+SSIM, which is
            reserved for the explicit ``datasets.preprocess.dedup()`` step.

    Returns:
        ``{content_hash: [(split, img_path, mask_path), ...]}`` — only for
        hashes seen in more than one split. Empty dict = no cross-split
        duplicates found.
    """
    hash_fn = hash_fn or _sha1_file
    seen: Dict[str, List[Tuple[str, str, str]]] = {}
    for split, pairs in pairs_by_split.items():
        for img_path, mask_path in pairs:
            h = hash_fn(img_path)
            seen.setdefault(h, []).append((split, img_path, mask_path))

    return {
        h: entries
        for h, entries in seen.items()
        if len({split for split, _, _ in entries}) > 1
    }
