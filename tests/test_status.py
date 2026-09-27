"""
tests/test_status.py — orchestration.status, the read-only "what state is
this experiment in?" API.

Built against hand-written directory trees rather than real training runs:
the module's whole job is to read artifacts off disk, so the artifacts are
what needs varying (missing manifests, corrupt ones, a run killed before it
wrote run_meta.json), and none of those states are convenient to produce by
actually training.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from dissert.orchestration.status import describe_experiment, describe_run


# ---------------------------------------------------------------------------
# Tree builders
# ---------------------------------------------------------------------------

HASH = "abc1234def5678"


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def _make_run(
    exp_root,
    seed=7,
    repeat=None,
    folds=(None,),
    status="done",
    epochs_completed=10,
    total_epochs=10,
    write_last=True,
    write_best=True,
    write_run_meta=True,
):
    """Build one run root the way orchestration.runid.experiment_paths lays
    it out, and return its path."""
    suffix = f"-r{repeat}" if repeat is not None else ""
    run_dir = os.path.join(exp_root, f"{HASH[:7]}-s{seed}{suffix}")

    if write_run_meta:
        _write_json(os.path.join(run_dir, "run_meta.json"), {
            "experiment_name": os.path.basename(exp_root),
            "seed": seed, "repeat": repeat, "config_hash": HASH,
        })

    for fold in folds:
        chk = os.path.join(run_dir, "checkpoints")
        if fold is not None:
            chk = os.path.join(chk, f"fold{fold}")
        _write_json(os.path.join(chk, "manifest.json"), {
            "run_id": f"R-{HASH[:7]}-s{seed}-f{fold if fold is not None else '-'}",
            "status": status,
            "fold": fold,
            "epochs_completed": epochs_completed,
            "total_epochs": total_epochs,
            "wall_seconds": 123.4,
        })
        for name, write in (("last.pth", write_last), ("best.pth", write_best)):
            if write:
                with open(os.path.join(chk, name), "wb") as f:
                    f.write(b"not-a-real-checkpoint")

    return run_dir


def _make_fold_splits(exp_root):
    path = os.path.join(exp_root, f"{HASH[:7]}-fold_splits.json")
    _write_json(path, {"n_splits": 2, "config_hash": HASH, "folds": []})
    return path


# ---------------------------------------------------------------------------
# describe_run
# ---------------------------------------------------------------------------

def test_describe_run_reads_identity_from_run_meta(tmp_path):
    exp_root = str(tmp_path / "exp")
    run_dir = _make_run(exp_root, seed=7, repeat=2)

    info = describe_run(run_dir)
    assert info["experiment_name"] == "exp"
    assert info["seed"] == 7
    assert info["repeat"] == 2
    assert info["config_hash"] == HASH
    assert info["status"] == "done"
    assert info["total_epochs"] == 10


def test_describe_run_done_run_is_not_resumable(tmp_path):
    """Nothing left to do means nothing to resume — 'resumable' answers
    "can this be continued", not "does a checkpoint exist"."""
    run_dir = _make_run(str(tmp_path / "exp"), status="done")
    info = describe_run(run_dir)
    assert info["resumable"] is False
    # ...but a finished run still reports the epochs it actually reached,
    # rather than the 0 that "no unfinished folds" would otherwise give.
    assert info["epochs_completed"] == 10


def test_describe_run_interrupted_run_is_resumable(tmp_path):
    run_dir = _make_run(
        str(tmp_path / "exp"), status="interrupted", epochs_completed=4, total_epochs=10,
    )
    info = describe_run(run_dir)
    assert info["status"] == "interrupted"
    assert info["resumable"] is True
    assert info["epochs_completed"] == 4


def test_describe_run_interrupted_without_checkpoint_is_not_resumable(tmp_path):
    run_dir = _make_run(str(tmp_path / "exp"), status="interrupted", write_last=False)
    info = describe_run(run_dir)
    assert info["status"] == "interrupted"
    assert info["resumable"] is False


def test_describe_run_empty_last_pth_is_not_resumable(tmp_path):
    run_dir = _make_run(str(tmp_path / "exp"), status="interrupted")
    open(os.path.join(run_dir, "checkpoints", "last.pth"), "wb").close()
    assert describe_run(run_dir)["resumable"] is False


def test_describe_run_checkpoint_files_are_relative_and_minimal(tmp_path, monkeypatch):
    exp_root = str(tmp_path / "exp")
    _make_fold_splits(exp_root)
    run_dir = _make_run(exp_root, status="interrupted")

    monkeypatch.chdir(tmp_path)
    files = describe_run(run_dir)["checkpoint_files"]

    assert all(not os.path.isabs(p) for p in files)
    names = sorted(os.path.basename(p) for p in files)
    # best.pth is mandatory: train.py re-seeds CheckpointManager.best_metric
    # from it on resume, so without it a resumed session would overwrite a
    # better checkpoint with a worse one.
    assert names == ["abc1234-fold_splits.json", "best.pth", "last.pth",
                     "manifest.json", "run_meta.json"]
    assert all(os.path.exists(os.path.join(str(tmp_path), p)) for p in files)


def test_describe_run_omits_absent_files(tmp_path):
    run_dir = _make_run(str(tmp_path / "exp"), write_best=False)
    names = {os.path.basename(p) for p in describe_run(run_dir)["checkpoint_files"]}
    assert "best.pth" not in names
    assert "last.pth" in names


def test_describe_run_kfold_folds_and_rollup(tmp_path):
    exp_root = str(tmp_path / "exp")
    _make_fold_splits(exp_root)
    run_dir = _make_run(exp_root, folds=(0, 1), status="done")

    # Fold 1 stopped early; the run as a whole is therefore interrupted.
    manifest_path = os.path.join(run_dir, "checkpoints", "fold1", "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest.update(status="interrupted", epochs_completed=3)
    _write_json(manifest_path, manifest)

    info = describe_run(run_dir)
    assert [f["fold"] for f in info["folds"]] == [0, 1]
    assert info["status"] == "interrupted"
    assert info["has_fold_splits"] is True
    # The laggard fold determines the work remaining.
    assert info["epochs_completed"] == 3


def test_describe_run_rollup_is_worst_first(tmp_path):
    exp_root = str(tmp_path / "exp")
    run_dir = _make_run(exp_root, folds=(0, 1), status="interrupted")

    path = os.path.join(run_dir, "checkpoints", "fold0", "manifest.json")
    with open(path) as f:
        manifest = json.load(f)
    manifest["status"] = "failed"
    _write_json(path, manifest)

    assert describe_run(run_dir)["status"] == "failed"


def test_describe_run_running_outranks_interrupted(tmp_path):
    """A manifest stuck at 'running' means the process was hard-killed and
    never got to record an outcome — that needs attention in a way a clean
    budget stop does not."""
    run_dir = _make_run(str(tmp_path / "exp"), folds=(0, 1), status="interrupted")
    path = os.path.join(run_dir, "checkpoints", "fold1", "manifest.json")
    with open(path) as f:
        manifest = json.load(f)
    manifest["status"] = "running"
    _write_json(path, manifest)

    assert describe_run(run_dir)["status"] == "running"


def test_describe_run_missing_fold_splits(tmp_path):
    run_dir = _make_run(str(tmp_path / "exp"))
    info = describe_run(run_dir)
    assert info["has_fold_splits"] is False
    assert not any("fold_splits" in p for p in info["checkpoint_files"])


def test_describe_run_falls_back_to_glob_without_run_meta(tmp_path):
    """A run killed before it wrote run_meta.json still resolves its shared
    fold-splits file, as long as the fallback is unambiguous."""
    exp_root = str(tmp_path / "exp")
    _make_fold_splits(exp_root)
    run_dir = _make_run(exp_root, write_run_meta=False)

    info = describe_run(run_dir)
    assert info["config_hash"] is None
    assert info["has_fold_splits"] is True


def test_describe_run_tolerates_corrupt_manifest(tmp_path):
    """A half-written manifest degrades to 'pending', it does not raise —
    this module is what a caller uses to diagnose a broken run."""
    run_dir = _make_run(str(tmp_path / "exp"))
    with open(os.path.join(run_dir, "checkpoints", "manifest.json"), "w") as f:
        f.write("{ this is not json")

    info = describe_run(run_dir)
    assert info["status"] == "pending"
    assert info["folds"][0]["epochs_completed"] == 0


def test_describe_run_on_empty_directory(tmp_path):
    empty = tmp_path / "exp" / "abc1234-s7"
    empty.mkdir(parents=True)
    info = describe_run(str(empty))
    assert info["status"] == "pending"
    assert info["resumable"] is False
    assert info["folds"] == []


# ---------------------------------------------------------------------------
# describe_experiment
# ---------------------------------------------------------------------------

def test_describe_experiment_covers_every_run(tmp_path):
    exp_root = str(tmp_path / "exp")
    _make_fold_splits(exp_root)
    _make_run(exp_root, seed=7, status="done")
    _make_run(exp_root, seed=42, status="interrupted")
    _make_run(exp_root, seed=1337, status="done")

    info = describe_experiment(exp_root)
    assert info["n_runs"] == 3
    assert info["n_done"] == 2
    assert info["status"] == "interrupted"
    assert info["resumable"] is True
    assert info["experiment_name"] == "exp"
    assert info["config_hash"] == HASH


def test_describe_experiment_all_done(tmp_path):
    exp_root = str(tmp_path / "exp")
    _make_run(exp_root, seed=7, status="done")
    _make_run(exp_root, seed=42, status="done")

    info = describe_experiment(exp_root)
    assert info["status"] == "done"
    assert info["resumable"] is False


def test_describe_experiment_checkpoint_files_are_deduped(tmp_path):
    """The shared fold-splits file belongs to every run — it must appear in
    the union exactly once."""
    exp_root = str(tmp_path / "exp")
    _make_fold_splits(exp_root)
    _make_run(exp_root, seed=7, status="interrupted")
    _make_run(exp_root, seed=42, status="interrupted")

    files = describe_experiment(exp_root)["checkpoint_files"]
    assert len(files) == len(set(files))
    assert sum("fold_splits" in p for p in files) == 1


def test_describe_experiment_ignores_sibling_report_files(tmp_path):
    """Combined reports and fold-split files sit beside the run roots; only
    directories holding checkpoints/ are runs."""
    exp_root = str(tmp_path / "exp")
    _make_fold_splits(exp_root)
    _write_json(os.path.join(exp_root, "exp.json"), {"metrics": {}})
    _make_run(exp_root, seed=7)

    assert describe_experiment(exp_root)["n_runs"] == 1


def test_describe_experiment_on_empty_directory(tmp_path):
    exp_root = tmp_path / "exp"
    exp_root.mkdir()
    info = describe_experiment(str(exp_root))
    assert info["n_runs"] == 0
    assert info["status"] == "pending"
    assert info["resumable"] is False


# ---------------------------------------------------------------------------
# Import weight
# ---------------------------------------------------------------------------

def test_status_module_imports_without_torch(tmp_path, monkeypatch):
    """This module is meant to be importable by an external driver that has
    no training stack installed, so nothing at module scope may need torch."""
    monkeypatch.setitem(sys.modules, "torch", None)
    for name in list(sys.modules):
        if name == "dissert.orchestration.status":
            monkeypatch.delitem(sys.modules, name)

    import importlib
    module = importlib.import_module("dissert.orchestration.status")

    run_dir = _make_run(str(tmp_path / "exp"), status="interrupted")
    # Presence-based resumability must not touch torch either.
    assert module.describe_run(run_dir)["resumable"] is True


def test_verify_rejects_unloadable_checkpoint(tmp_path):
    """verify=True actually deserialises — the dummy byte payload these
    fixtures write is not a real checkpoint and must be reported as such."""
    pytest.importorskip("torch")
    run_dir = _make_run(str(tmp_path / "exp"), status="interrupted")
    assert describe_run(run_dir)["resumable"] is True
    assert describe_run(run_dir, verify=True)["resumable"] is False
