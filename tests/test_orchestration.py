"""
tests/test_orchestration.py — Phase 1: config schema, run identity
(config_hash/run_id), manifest, ledger, orchestration.runner, and the first
version of test_determinism.

Phase 6 extends test_determinism to also cover the Mamba model family
(expected to genuinely trip the non-determinism warning on its fused-kernel
path); this version only has the existing (non-SSM) model families to
verify against, all of which are expected to run bit-for-bit identically
under a fixed seed on CPU — see IMPLEMENTATION_PLAN.md's Phase 6 section.
"""
from __future__ import annotations

import copy
import json
import os
import time

import pydantic
import pytest
import torch

from dissert.orchestration.budget import WallClockBudget
from dissert.orchestration.ledger import LedgerWriter
from dissert.orchestration.manifest import build_manifest
from dissert.orchestration.runid import config_hash, experiment_id, experiment_paths, run_id
from dissert.orchestration.runner import run_sweep
from dissert.config.schema import validate_config
from dissert.cli.train import run_training
from dissert.training.determinism import (
    get_recorded_nondeterminism,
    reset_recorded_nondeterminism,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_accepts_valid_config(tiny_config):
    # tiny_config is already validate_config()'s own output; re-validating
    # it must be a no-op (idempotent).
    assert validate_config(copy.deepcopy(tiny_config)) == tiny_config


def test_schema_rejects_unknown_key(tiny_config):
    bad = copy.deepcopy(tiny_config)
    bad["dataset"]["totally_bogus_key"] = 1
    with pytest.raises(pydantic.ValidationError):
        validate_config(bad)


def test_schema_rejects_missing_required_field(tiny_config):
    bad = copy.deepcopy(tiny_config)
    del bad["dataset"]["name"]
    with pytest.raises(pydantic.ValidationError):
        validate_config(bad)


def test_schema_rejects_wrong_type(tiny_config):
    bad = copy.deepcopy(tiny_config)
    bad["training"]["lr"] = "not-a-float"
    with pytest.raises(pydantic.ValidationError):
        validate_config(bad)


def test_schema_model_section_is_permissive(tiny_config):
    # model: forwards arbitrary kwargs (mk_unet's channels/depths/... etc)
    # straight to get_model — extra keys must NOT raise.
    cfg = copy.deepcopy(tiny_config)
    cfg["model"]["some_arch_specific_kwarg"] = [1, 2, 3]
    validated = validate_config(cfg)
    assert validated["model"]["some_arch_specific_kwarg"] == [1, 2, 3]


def test_schema_optional_none_fields_do_not_shadow_downstream_defaults(tiny_config):
    # dataset.norm_mean is unset here; the validated dict must NOT carry an
    # explicit `norm_mean: None` (that would make
    # ds_cfg.get("norm_mean", _IMAGENET_MEAN) return None instead of
    # falling back to _IMAGENET_MEAN in datasets/transforms.py).
    assert "norm_mean" not in tiny_config["dataset"]
    assert "norm_std" not in tiny_config["dataset"]


# ---------------------------------------------------------------------------
# config_hash / run_id
# ---------------------------------------------------------------------------

def test_config_hash_stable_across_seed(tiny_config):
    a = copy.deepcopy(tiny_config)
    b = copy.deepcopy(tiny_config)
    b["training"]["seed"] = 999
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_real_change(tiny_config):
    a = copy.deepcopy(tiny_config)
    b = copy.deepcopy(tiny_config)
    b["training"]["lr"] = 0.5
    assert config_hash(a) != config_hash(b)


def test_config_hash_stable_across_experiment_name(tiny_config):
    a = copy.deepcopy(tiny_config)
    b = copy.deepcopy(tiny_config)
    b["logging"]["experiment_name"] = "totally_different_name"
    assert config_hash(a) == config_hash(b)


def test_config_hash_stable_across_output_dir(tiny_config):
    a = copy.deepcopy(tiny_config)
    b = copy.deepcopy(tiny_config)
    b["output_dir"] = "some/other/output/dir"
    assert config_hash(a) == config_hash(b)


def test_config_hash_stable_across_checkpoint_bookkeeping(tiny_config):
    a = copy.deepcopy(tiny_config)
    b = copy.deepcopy(tiny_config)
    b["checkpoint"]["checkpoint_path"] = "/some/other/checkpoint.pth"
    b["checkpoint"]["resume"] = not b["checkpoint"].get("resume", True)
    b["checkpoint"]["periodic_save_every"] = 999
    assert config_hash(a) == config_hash(b)


def test_config_hash_stable_across_stats_section(tiny_config):
    a = copy.deepcopy(tiny_config)
    b = copy.deepcopy(tiny_config)
    a["stats"] = {"family": "fam_a", "comparators": [], "min_meaningful_diff": 0.01, "alpha": 0.05}
    b["stats"] = {"family": "fam_b", "comparators": ["x"], "min_meaningful_diff": 0.02, "alpha": 0.1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_checkpoint_monitor_metric(tiny_config):
    # monitor_metric/mode are deliberately NOT stripped — they change which
    # checkpoint gets selected as "best", i.e. the actual result.
    a = copy.deepcopy(tiny_config)
    b = copy.deepcopy(tiny_config)
    b["checkpoint"]["monitor_metric"] = "val_loss"
    assert config_hash(a) != config_hash(b)


def test_run_id_format(tiny_config):
    h = config_hash(tiny_config)
    assert run_id(h, seed=7, fold=2) == f"R-{h[:7]}-s7-f2"
    assert run_id(h, seed=7, fold=None) == f"R-{h[:7]}-s7-f-"  # non-CV run
    # repeat=None (default) omits the repeat segment entirely — same
    # historical format as above, unaffected by the repeat feature existing.
    assert run_id(h, seed=7, fold=2, repeat=None) == f"R-{h[:7]}-s7-f2"


def test_run_id_format_with_repeat(tiny_config):
    h = config_hash(tiny_config)
    assert run_id(h, seed=7, fold=2, repeat=1) == f"R-{h[:7]}-s7-r1-f2"
    assert run_id(h, seed=7, fold=None, repeat=0) == f"R-{h[:7]}-s7-r0-f-"


def test_experiment_id_and_paths_with_repeat(tiny_config):
    h = config_hash(tiny_config)
    assert experiment_id(h, seed=7) == f"{h[:7]}-s7"  # unaffected, repeat=None
    assert experiment_id(h, seed=7, repeat=2) == f"{h[:7]}-s7-r2"

    paths_no_repeat = experiment_paths("outputs/experiments", "exp", h, 7)
    paths_r0 = experiment_paths("outputs/experiments", "exp", h, 7, repeat=0)
    paths_r1 = experiment_paths("outputs/experiments", "exp", h, 7, repeat=1)

    # Each repeat gets its own independent root — never collides with the
    # legacy unsuffixed path or with another repeat.
    assert len({paths_no_repeat["root"], paths_r0["root"], paths_r1["root"]}) == 3
    assert paths_r0["checkpoints"] != paths_r1["checkpoints"]

    # fold_splits and combined_report stay shared across seeds *and*
    # repeats — never scoped into any one repeat's root.
    assert paths_r0["fold_splits"] == paths_r1["fold_splits"] == paths_no_repeat["fold_splits"]
    assert paths_r0["combined_report"] == paths_r1["combined_report"] == paths_no_repeat["combined_report"]
    # seed_combined_report is per-seed (shared across that seed's repeats),
    # not per-repeat.
    assert paths_r0["seed_combined_report"] == paths_r1["seed_combined_report"]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_round_trip(tmp_path, tiny_config):
    rid = run_id(config_hash(tiny_config), seed=42, fold=0)
    manifest = build_manifest(rid, tiny_config, seed=42, fold=0)
    manifest.start()
    manifest.finish(status="done")
    path = tmp_path / "manifest.json"
    manifest.save(str(path))

    with open(path) as f:
        data = json.load(f)
    assert data["run_id"] == rid
    assert data["status"] == "done"
    assert data["config_hash"] == config_hash(tiny_config)
    assert data["resolved_config"]["dataset"]["name"] == "synthetic_test_dataset"
    assert data["nondeterministic_ops"] == []


def test_manifest_records_nondeterminism(tiny_config):
    manifest = build_manifest("R-test", tiny_config, seed=42)
    manifest.record_nondeterminism("some op has no deterministic implementation")
    assert manifest.data["nondeterministic_ops"] == [
        "some op has no deterministic implementation"
    ]


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def test_ledger_append_and_has_done_run(tmp_path):
    ledger = LedgerWriter(str(tmp_path / "ledger"))
    assert not ledger.has_done_run("R-abc")
    ledger.append_run_row(run_id="R-abc", status="done")
    assert ledger.has_done_run("R-abc")


def test_ledger_rejects_unknown_field(tmp_path):
    ledger = LedgerWriter(str(tmp_path / "ledger"))
    with pytest.raises(ValueError):
        ledger.append_run_row(run_id="R-abc", not_a_real_column=1)


# ---------------------------------------------------------------------------
# orchestration.runner: idempotent sweep skip
# ---------------------------------------------------------------------------

def test_run_sweep_idempotent_skip(tmp_path, tiny_config):
    calls = []

    def fake_train(config, fold=None, run_id=None, repeat=None):
        calls.append(run_id)
        return 0.42

    ledger_dir = str(tmp_path / "ledger")

    results_1 = run_sweep(
        tiny_config, seeds=[0, 1], folds=[None], train_fn=fake_train,
        ledger_dir=ledger_dir,
    )
    assert all(r["status"] == "done" for r in results_1)
    assert len(calls) == 2

    results_2 = run_sweep(
        tiny_config, seeds=[0, 1], folds=[None], train_fn=fake_train,
        ledger_dir=ledger_dir,
    )
    assert all(r["status"] == "skipped-done" for r in results_2)
    assert len(calls) == 2  # fake_train not called again


def test_run_sweep_repeats_axis(tmp_path, tiny_config):
    """Each (seed, repeat) combination gets its own run_id/manifest — a
    repeat is never mistaken for "already done" by another repeat of the
    same seed, and the ledger records which repeat index produced each row."""
    calls = []

    def fake_train(config, fold=None, run_id=None, repeat=None):
        calls.append((run_id, repeat))
        return 0.42

    ledger_dir = str(tmp_path / "ledger")

    results = run_sweep(
        tiny_config, seeds=[0], folds=[None], repeats=[0, 1, 2],
        train_fn=fake_train, ledger_dir=ledger_dir,
    )
    assert all(r["status"] == "done" for r in results)
    assert len(calls) == 3
    assert len({rid for rid, _ in calls}) == 3  # every repeat gets a distinct run_id

    with open(os.path.join(ledger_dir, "runs.csv")) as f:
        import csv
        rows = list(csv.DictReader(f))
    assert sorted(row["repeat"] for row in rows) == ["0", "1", "2"]

    # Re-running with the same repeats is a full no-op — same idempotent
    # skip as the seed/fold axes already have.
    results_2 = run_sweep(
        tiny_config, seeds=[0], folds=[None], repeats=[0, 1, 2],
        train_fn=fake_train, ledger_dir=ledger_dir,
    )
    assert all(r["status"] == "skipped-done" for r in results_2)
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Wall-clock budgets (orchestration.budget) and the interrupted status
# ---------------------------------------------------------------------------

def test_budget_exhausted_by():
    budget = WallClockBudget(1.0, start=time.monotonic())
    # Nothing spent yet: a projection well inside the hour fits, one past
    # the whole hour does not.
    assert not budget.exhausted_by(0.0)
    assert not budget.exhausted_by(60.0)
    assert budget.exhausted_by(3600.1)
    assert 0 < budget.remaining() <= 3600.0


def test_budget_already_spent():
    """A budget whose deadline is in the past is exhausted even by zero
    further work — this is what makes a session that has no time left stop
    before running an epoch rather than after."""
    budget = WallClockBudget(0.001, start=time.monotonic() - 3600.0)
    assert budget.exhausted_by(0.0)
    assert budget.remaining() < 0


def test_budget_rejects_nonpositive():
    with pytest.raises(ValueError):
        WallClockBudget(0)


def test_run_sweep_gates_on_exhausted_budget(tmp_path, tiny_config):
    """No time left => no combination is started at all, and every one is
    reported (not silently dropped) so the caller can see what remains."""
    calls = []

    def fake_train(config, fold=None, run_id=None, repeat=None, budget=None):
        calls.append(run_id)
        return 0.42

    results = run_sweep(
        tiny_config, seeds=[0, 1], folds=[None], train_fn=fake_train,
        ledger_dir=str(tmp_path / "ledger"),
        budget=WallClockBudget(0.001, start=time.monotonic() - 3600.0),
    )
    assert [r["status"] for r in results] == ["skipped-budget", "skipped-budget"]
    assert calls == []
    # Nothing ran, so nothing may have been recorded as if it had.
    assert not list((tmp_path).glob("**/manifest.json"))


def test_run_sweep_records_interrupted_and_retries_it(tmp_path, tiny_config):
    """A run that stops itself on budget is 'interrupted', not 'done' — and
    is therefore picked up again by the next sweep instead of being skipped."""
    from dissert.training.determinism import record_manifest_extra

    calls = []

    def budget_stopped_train(config, fold=None, run_id=None, repeat=None, budget=None):
        calls.append(run_id)
        record_manifest_extra("stopped_on_budget", True)
        record_manifest_extra("epochs_completed", 3)
        record_manifest_extra("total_epochs", 10)
        return 0.42

    ledger_dir = str(tmp_path / "ledger")
    # A budget with plenty of time left, so the gate above never fires and
    # the run itself is what stops.
    budget = WallClockBudget(10.0)

    results = run_sweep(
        tiny_config, seeds=[0], folds=[None], train_fn=budget_stopped_train,
        ledger_dir=ledger_dir, budget=budget,
    )
    assert [r["status"] for r in results] == ["interrupted"]

    mpath = os.path.join(
        experiment_paths(
            tiny_config["output_dir"], tiny_config["logging"]["experiment_name"],
            config_hash(tiny_config), 0,
        )["checkpoints"],
        "manifest.json",
    )
    with open(mpath) as f:
        manifest = json.load(f)
    assert manifest["status"] == "interrupted"
    assert manifest["epochs_completed"] == 3
    assert manifest["total_epochs"] == 10
    assert manifest["wall_seconds"] is not None
    # No last.pth was written by the fake trainer, so this must report False
    # rather than optimistically claiming a resume point exists.
    assert manifest["resumable"] is False

    # The whole point: an interrupted run is retried, unlike a done one.
    run_sweep(
        tiny_config, seeds=[0], folds=[None], train_fn=budget_stopped_train,
        ledger_dir=ledger_dir, budget=budget,
    )
    assert len(calls) == 2


def test_run_sweep_without_budget_keeps_legacy_train_fn_signature(tmp_path, tiny_config):
    """No budget => train_fn is called exactly as before, so an injected
    function written against the original signature (which does not accept
    `budget`) keeps working."""
    def legacy_train(config, fold=None, run_id=None, repeat=None):
        return 0.42

    results = run_sweep(
        tiny_config, seeds=[0], folds=[None], train_fn=legacy_train,
        ledger_dir=str(tmp_path / "ledger"),
    )
    assert [r["status"] for r in results] == ["done"]


def test_manifest_records_wall_seconds(tiny_config):
    """wall_seconds is set for every finished run, unlike gpu_hours which is
    None on a CPU box — a caller estimating remaining sessions needs a
    duration off every manifest."""
    manifest = build_manifest("R-test", tiny_config, seed=0)
    assert manifest.data["wall_seconds"] is None
    assert manifest.data["epochs_completed"] == 0
    assert manifest.data["resumable"] is False
    manifest.start().finish(status="interrupted")
    assert manifest.data["status"] == "interrupted"
    assert manifest.data["wall_seconds"] >= 0


def test_trainer_stops_on_budget(tiny_config_factory):
    """Real training, real budget: a budget that is already spent stops the
    run before its first epoch, leaving epochs_completed at 0 so a caller
    can tell this session made no progress."""
    cfg = tiny_config_factory()
    cfg["training"]["epochs"] = 3

    spent = WallClockBudget(0.001, start=time.monotonic() - 3600.0)
    best = run_training(cfg, fold=None, budget=spent)

    from dissert.training.determinism import get_recorded_manifest_extras
    extras = get_recorded_manifest_extras()
    assert extras["stopped_on_budget"] is True
    assert extras["epochs_completed"] == 0
    assert extras["total_epochs"] == 3
    assert best is not None


class _BudgetAfter:
    """Budget stub that reports "exhausted" only from the *n*-th check on.

    Timing a real budget to trip between two epochs of a sub-second test run
    would be flaky; what actually needs asserting is that a stop *between*
    epochs leaves consistent state, which this pins down exactly.
    """

    max_hours = 1.0

    def __init__(self, trip_on_check: int):
        self.trip_on_check = trip_on_check
        self.checks = 0

    def exhausted_by(self, projected_seconds=0.0):
        self.checks += 1
        return self.checks >= self.trip_on_check

    def elapsed(self):
        return 0.0

    def remaining(self):
        return 0.0


def test_trainer_stops_between_epochs(tiny_config_factory):
    """Stopping mid-run leaves epochs_completed at the last epoch whose
    checkpoint is on disk — the two can never disagree, which is what a
    resume depends on."""
    cfg = tiny_config_factory()
    cfg["training"]["epochs"] = 4

    # Checked once per epoch: pass for epoch 1, trip before epoch 2.
    run_training(cfg, fold=None, budget=_BudgetAfter(trip_on_check=2))

    from dissert.training.determinism import get_recorded_manifest_extras
    extras = get_recorded_manifest_extras()
    assert extras["stopped_on_budget"] is True
    assert extras["epochs_completed"] == 1
    assert extras["total_epochs"] == 4
    assert len(extras["epoch_seconds"]) == 1

    chk = experiment_paths(
        cfg["output_dir"], cfg["logging"]["experiment_name"],
        config_hash(cfg), cfg["training"]["seed"],
    )["checkpoints"]
    last = torch.load(os.path.join(chk, "last.pth"), map_location="cpu")
    assert last["epoch"] == extras["epochs_completed"]


def test_trainer_budget_stop_ends_staged_run(tiny_config_factory):
    """A budget stop inside one stage ends the whole run — otherwise the
    next stage would start straight through the deadline."""
    cfg = tiny_config_factory()
    cfg["stages"] = [
        {"epochs": 2, "lr": 0.01, "freeze": []},
        {"epochs": 2, "lr": 0.001, "freeze": []},
    ]

    run_training(cfg, fold=None, budget=_BudgetAfter(trip_on_check=2))

    from dissert.training.determinism import get_recorded_manifest_extras
    extras = get_recorded_manifest_extras()
    assert extras["stopped_on_budget"] is True
    assert extras["epochs_completed"] == 1
    # Summed across stages, so progress is readable without the config.
    assert extras["total_epochs"] == 4


def test_trainer_without_budget_runs_every_epoch(tiny_config_factory):
    """The new machinery is inert when no budget is given — the pre-existing
    behaviour is unchanged."""
    cfg = tiny_config_factory()
    cfg["training"]["epochs"] = 2

    run_training(cfg, fold=None)

    from dissert.training.determinism import get_recorded_manifest_extras
    extras = get_recorded_manifest_extras()
    assert "stopped_on_budget" not in extras
    assert extras["epochs_completed"] == 2
    assert extras["total_epochs"] == 2


# ---------------------------------------------------------------------------
# Determinism (the star of Phase 1)
# ---------------------------------------------------------------------------

def test_seed_everything_reproduces_torch_rng():
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_determinism(tmp_path, tiny_config_factory):
    """Train the same tiny model on the same tiny data twice, from the same
    seed: the two runs must produce bit-identical final weights and an
    identical monitored metric, and torch's determinism guard must not have
    recorded any non-deterministic op along the way (this model/data has
    none, on CPU, at this torch pin).
    """
    results = {}
    for run_name in ("a", "b"):
        cfg = tiny_config_factory()
        cfg["output_dir"] = str(tmp_path / f"outputs_{run_name}")
        reset_recorded_nondeterminism()

        best_metric = run_training(cfg, fold=None)

        ckpt_path = os.path.join(
            experiment_paths(
                cfg["output_dir"], cfg["logging"]["experiment_name"],
                config_hash(cfg), cfg["training"]["seed"],
            )["checkpoints"],
            "last.pth",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        results[run_name] = {
            "best_metric": best_metric,
            "state_dict": ckpt["model_state_dict"],
            "nondeterminism": get_recorded_nondeterminism(),
        }

    assert results["a"]["best_metric"] == results["b"]["best_metric"]
    assert results["a"]["nondeterminism"] == []
    assert results["b"]["nondeterminism"] == []

    sd_a, sd_b = results["a"]["state_dict"], results["b"]["state_dict"]
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"weights diverged at {key}"
