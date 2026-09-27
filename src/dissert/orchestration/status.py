"""
orchestration/status.py — read-only inspection of an experiment's state.

Answers "what state is this experiment in, and what would it take to
continue it?" from the artifacts on disk, for anything driving this repo
from the outside: a scheduler script chaining bounded sessions on
session-limited compute, a resume wrapper, a progress dashboard.

The point is that **this module decides what a resume needs**, not its
caller. A caller that reimplements the path layout or reaches into ``.pth``
files silently breaks the first time either changes; a caller that reads
``checkpoint_files`` off this module keeps working.

Deliberately import-light: no torch at module scope, so a caller can
inspect an experiment without paying for (or even having) the training
stack. ``verify=True`` opts into a real checkpoint load and lazy-imports
torch only then.

Two entry points, matching the two units a caller cares about:

- :func:`describe_run` — one *run root*,
  ``<output_dir>/<experiment_name>/<hash7>-s<seed>[-r<repeat>]/``, i.e. the
  directory ``orchestration.runid.experiment_paths()["root"]`` returns. It
  covers every fold of that one (config_hash, seed, repeat).
- :func:`describe_experiment` — the whole
  ``<output_dir>/<experiment_name>/`` tree. The default sweep is 3 seeds x
  3 repeats (9+ run roots), so a caller driving one session almost always
  wants this one.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional

# Worst-first. A caller acts on the *worst* thing that happened across the
# folds of a run (or the runs of an experiment), so the rollup surfaces
# that rather than an average.
#
# "running" ranks above "interrupted" on purpose: on an artifact captured
# after the fact it means the process never reached manifest.finish() —
# hard-killed mid-epoch — which needs attention in a way a clean,
# self-imposed budget stop does not.
_STATUS_PRECEDENCE = ["failed", "running", "interrupted", "pending", "done"]


def _rollup(statuses: List[str]) -> str:
    """The worst status in *statuses*, per :data:`_STATUS_PRECEDENCE`.

    An unrecognised status wins over everything known — a caller should
    look at something this module doesn't understand, not have it silently
    outranked by "done"."""
    if not statuses:
        return "pending"
    unknown = [s for s in statuses if s not in _STATUS_PRECEDENCE]
    if unknown:
        return sorted(unknown)[0]
    return min(statuses, key=_STATUS_PRECEDENCE.index)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """Parse *path*, or None if it's absent/unreadable/corrupt. A
    half-written manifest must degrade into "unknown", never into an
    exception — this module is what a caller uses to *diagnose* a broken
    run, so it cannot itself fall over on one."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _is_resumable(checkpoint_dir: str, verify: bool = False) -> bool:
    """Whether ``last.pth`` in *checkpoint_dir* is a usable resume point.

    Presence plus non-zero size by default: ``utils.checkpoint.atomic_torch_save``
    writes through a temp file and ``os.replace``, so a torn ``last.pth``
    isn't a reachable state and paying to deserialise a multi-GB checkpoint
    to learn that would be waste. *verify* forces the real load anyway, for
    a caller that has just moved the file between machines and wants proof
    rather than inference.
    """
    path = os.path.join(checkpoint_dir, "last.pth")
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
    except OSError:
        return False

    if not verify:
        return True

    try:
        import torch  # local: keeps this module importable without torch
        torch.load(path, map_location="cpu")
        return True
    except Exception:  # noqa: BLE001 — any failure to load means "not resumable"
        return False


def _fold_dirs(run_dir: str) -> List[str]:
    """Every checkpoint directory under *run_dir* that holds a manifest.

    Both layouts ``orchestration.runid.experiment_paths`` produces:
    ``checkpoints/`` for a non-CV run, ``checkpoints/fold<N>/`` for a
    K-Fold one."""
    found = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "fold*")))
    dirs = [d for d in found if os.path.isdir(d)]
    if not dirs:
        plain = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(plain):
            dirs = [plain]
    return dirs


def _relpath(path: str) -> str:
    """*path* relative to the current working directory (the repo root, for
    anything run the way this repo's scripts are). Falls back to the
    absolute path when the two are on different drives/roots."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def describe_run(experiment_dir: str, verify: bool = False) -> Dict[str, Any]:
    """Describe one run root.

    Args:
        experiment_dir: ``<output_dir>/<experiment_name>/<hash7>-s<seed>[-r<repeat>]/``
            — what ``orchestration.runid.experiment_paths()["root"]`` returns.
        verify: actually load each ``last.pth`` to decide ``resumable``,
            instead of trusting its presence. Costs a full deserialise per
            fold; see :func:`_is_resumable`.

    Returns a dict with:
        ``run_id``, ``experiment_name``, ``seed``, ``repeat``, ``config_hash``
            identity, read from ``run_meta.json`` — never parsed back out of
            the directory name.
        ``status``
            ``pending``/``running``/``done``/``interrupted``/``failed``,
            rolled up across folds worst-first.
        ``resumable``
            whether every not-yet-``done`` fold has a resume point. A run
            with nothing left to do is *not* resumable — there is nothing
            to resume.
        ``epochs_completed`` / ``total_epochs``
            progress of the *least*-advanced unfinished fold, since that is
            what determines the work remaining. Cumulative across sessions.
        ``has_fold_splits``
            whether the shared fold partition file is present. It lives
            beside the run roots, not inside one, so it is easy to forget
            when moving a run between machines — hence surfacing it.
        ``checkpoint_files``
            repo-relative paths that must travel with this run for a resume
            to be correct. Kept minimal; see below.
        ``folds``
            per-fold detail, in fold order.

    ``checkpoint_files`` includes ``best.pth`` and not only ``last.pth``:
    ``train.run_training`` re-seeds ``CheckpointManager.best_metric`` from
    ``best.pth`` on resume, so without it a resumed session compares every
    epoch against ``-inf`` and overwrites a genuinely better checkpoint
    with a worse one. It also includes the shared fold-splits file, whose
    absence would make a resumed K-Fold run rebuild its partition and train
    on different data than the checkpoint it loaded.
    """
    run_dir = os.path.abspath(experiment_dir)
    meta = _read_json(os.path.join(run_dir, "run_meta.json")) or {}
    config_hash = meta.get("config_hash")

    # The fold partition is shared by every seed/repeat of one config hash,
    # so it sits one level up, beside the run roots.
    exp_root = os.path.dirname(run_dir)
    fold_splits: Optional[str] = None
    if config_hash:
        candidate = os.path.join(exp_root, f"{config_hash[:7]}-fold_splits.json")
        if os.path.isfile(candidate):
            fold_splits = candidate
    else:
        # No run_meta.json to read the hash from (a run killed before it
        # wrote one). Fall back to the glob, but only when it's unambiguous.
        matches = sorted(glob.glob(os.path.join(exp_root, "*-fold_splits.json")))
        if len(matches) == 1:
            fold_splits = matches[0]

    folds: List[Dict[str, Any]] = []
    checkpoint_files: List[str] = []
    if fold_splits:
        checkpoint_files.append(_relpath(fold_splits))
    run_meta_path = os.path.join(run_dir, "run_meta.json")
    if os.path.isfile(run_meta_path):
        checkpoint_files.append(_relpath(run_meta_path))

    for chk_dir in _fold_dirs(run_dir):
        manifest = _read_json(os.path.join(chk_dir, "manifest.json")) or {}
        resumable = _is_resumable(chk_dir, verify=verify)
        basename = os.path.basename(chk_dir)
        fold_idx = manifest.get("fold")
        if fold_idx is None and basename.startswith("fold"):
            fold_idx = int(basename[len("fold"):])

        folds.append({
            "fold": fold_idx,
            "run_id": manifest.get("run_id"),
            # No manifest at all means the directory exists but nothing ever
            # finished writing one — "pending" is the honest reading.
            "status": manifest.get("status", "pending"),
            "epochs_completed": manifest.get("epochs_completed", 0),
            "total_epochs": manifest.get("total_epochs"),
            "wall_seconds": manifest.get("wall_seconds"),
            "resumable": resumable,
        })

        for name in ("last.pth", "best.pth", "manifest.json"):
            path = os.path.join(chk_dir, name)
            if os.path.isfile(path):
                checkpoint_files.append(_relpath(path))

    status = _rollup([f["status"] for f in folds])

    # Only unfinished folds bear on "what's left to do" — a finished fold
    # neither needs nor has a reason to keep a resume point.
    pending = [f for f in folds if f["status"] != "done"]
    resumable = bool(pending) and all(f["resumable"] for f in pending)
    # Progress of the least-advanced fold that still has work, since that is
    # what determines the remaining work. Once every fold is done there is no
    # laggard, and the honest number is what the folds actually reached — not
    # the 0 an empty-sequence default would give.
    counted = pending or folds
    epochs_completed = min((f["epochs_completed"] or 0 for f in counted), default=0)
    total_epochs = next(
        (f["total_epochs"] for f in folds if f["total_epochs"] is not None), None
    )

    return {
        "run_id": next((f["run_id"] for f in folds if f["run_id"]), None),
        "experiment_name": meta.get("experiment_name"),
        "seed": meta.get("seed"),
        "repeat": meta.get("repeat"),
        "config_hash": config_hash,
        "status": status,
        "resumable": resumable,
        "epochs_completed": epochs_completed,
        "total_epochs": total_epochs,
        "has_fold_splits": fold_splits is not None,
        "checkpoint_files": checkpoint_files,
        "folds": folds,
    }


def describe_experiment(experiment_dir: str, verify: bool = False) -> Dict[str, Any]:
    """Describe every run under one ``<output_dir>/<experiment_name>/`` tree.

    The default sweep produces one run root per (seed, repeat) — 9 of them
    at ``train.DEFAULT_SEEDS`` x ``train.DEFAULT_REPEATS`` — so a caller
    driving a bounded session needs the whole set's state, not one run's.

    Same keys as :func:`describe_run` where they carry over (``status`` and
    ``resumable`` roll up across runs by the same rules), plus ``runs``
    with each run's full dict. ``checkpoint_files`` is the de-duplicated
    union: staging exactly that set reproduces enough of this directory for
    every unfinished run to continue.
    """
    exp_root = os.path.abspath(experiment_dir)

    run_dirs = sorted(
        d for d in glob.glob(os.path.join(exp_root, "*"))
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "checkpoints"))
    )
    runs = [describe_run(d, verify=verify) for d in run_dirs]

    checkpoint_files: List[str] = []
    for run in runs:
        for path in run["checkpoint_files"]:
            if path not in checkpoint_files:
                checkpoint_files.append(path)

    pending = [r for r in runs if r["status"] != "done"]
    return {
        "experiment_name": next(
            (r["experiment_name"] for r in runs if r["experiment_name"]),
            os.path.basename(exp_root),
        ),
        "config_hash": next((r["config_hash"] for r in runs if r["config_hash"]), None),
        "status": _rollup([r["status"] for r in runs]),
        "resumable": bool(pending) and all(r["resumable"] for r in pending),
        "n_runs": len(runs),
        "n_done": sum(1 for r in runs if r["status"] == "done"),
        "has_fold_splits": any(r["has_fold_splits"] for r in runs),
        "checkpoint_files": checkpoint_files,
        "runs": runs,
    }
