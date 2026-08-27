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

import pydantic
import pytest
import torch

from orchestration.ledger import LedgerWriter
from orchestration.manifest import build_manifest
from orchestration.runid import config_hash, run_id
from orchestration.runner import run_sweep
from orchestration.schema import validate_config
from train import run_training
from training.determinism import (
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


def test_run_id_format(tiny_config):
    h = config_hash(tiny_config)
    assert run_id(h, seed=7, fold=2) == f"R-{h[:7]}-s7-f2"
    assert run_id(h, seed=7, fold=None) == f"R-{h[:7]}-s7-f-"  # non-CV run


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

    def fake_train(config, fold=None, run_id=None):
        calls.append(run_id)
        return 0.42

    artifacts_dir = str(tmp_path / "artifacts")
    ledger_dir = str(tmp_path / "artifacts" / "ledger")

    results_1 = run_sweep(
        tiny_config, seeds=[0, 1], folds=[None], train_fn=fake_train,
        artifacts_dir=artifacts_dir, ledger_dir=ledger_dir,
    )
    assert all(r["status"] == "done" for r in results_1)
    assert len(calls) == 2

    results_2 = run_sweep(
        tiny_config, seeds=[0, 1], folds=[None], train_fn=fake_train,
        artifacts_dir=artifacts_dir, ledger_dir=ledger_dir,
    )
    assert all(r["status"] == "skipped-done" for r in results_2)
    assert len(calls) == 2  # fake_train not called again


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
        cfg["checkpoint"]["save_dir"] = str(tmp_path / f"checkpoints_{run_name}")
        reset_recorded_nondeterminism()

        best_metric = run_training(cfg, fold=None)

        ckpt_path = os.path.join(
            cfg["checkpoint"]["save_dir"], cfg["logging"]["experiment_name"], "last.pth"
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
