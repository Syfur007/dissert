"""
tests/test_sweep.py — Phase 14: orchestration/sweep.py, spec §15's
budget-aware successor to search.py. test_sweep_cannot_see_test itself
(the spec-named static import check) lives in tests/test_data_contract.py
alongside train.py/search.py/orchestration.runner's own checks.
"""
from __future__ import annotations

import time

import pytest

from orchestration.sweep import get_grid_paths_and_values, run_budgeted_sweep


def test_get_grid_paths_and_values_nested():
    grid = {"training": {"lr": [0.01, 0.001]}, "model": {"name": ["a", "b"]}}
    paths, values = get_grid_paths_and_values(grid)
    assert paths == [["training", "lr"], ["model", "name"]]
    assert values == [[0.01, 0.001], ["a", "b"]]


def _base_config():
    return {"training": {"lr": 0.0}, "model": {"name": "x"}, "checkpoint": {"mode": "max"}}


def test_run_budgeted_sweep_stops_early_when_budget_exceeded():
    def fake_trial(cfg):
        time.sleep(0.05)
        return cfg["training"]["lr"] * 100

    grid = {"training": {"lr": [0.001, 0.01, 0.1, 1.0, 10.0]}}
    tiny_budget = (0.05 / 3600) * 2.5  # room for ~2 trials
    result = run_budgeted_sweep(_base_config(), grid, budget_gpu_hours=tiny_budget, seed=1, trial_fn=fake_trial)

    assert result["stopped_early"] is True
    assert 0 < len(result["trials"]) < result["n_grid_combinations"]
    assert result["best_trial"] is not None


def test_run_budgeted_sweep_runs_full_grid_with_ample_budget():
    def fake_trial(cfg):
        return cfg["training"]["lr"] * 100

    grid = {"training": {"lr": [0.001, 0.01, 0.1, 1.0, 10.0]}}
    result = run_budgeted_sweep(_base_config(), grid, budget_gpu_hours=1000.0, seed=1, trial_fn=fake_trial)

    assert len(result["trials"]) == 5
    assert result["stopped_early"] is False
    assert result["best_trial"]["params"]["training.lr"] == 10.0
    assert result["best_trial"]["objective"] == pytest.approx(1000.0)


def test_run_budgeted_sweep_min_mode_picks_lowest_objective():
    def fake_trial(cfg):
        return cfg["training"]["lr"]

    grid = {"training": {"lr": [0.001, 0.01, 0.1]}}
    result = run_budgeted_sweep(_base_config(), grid, budget_gpu_hours=1000.0, seed=1, trial_fn=fake_trial, mode="min")
    assert result["best_trial"]["params"]["training.lr"] == pytest.approx(0.001)


def test_run_budgeted_sweep_failed_trial_does_not_kill_sweep():
    def flaky_trial(cfg):
        if cfg["training"]["lr"] == 0.01:
            raise RuntimeError("boom")
        return cfg["training"]["lr"]

    grid = {"training": {"lr": [0.001, 0.01, 0.1]}}
    result = run_budgeted_sweep(_base_config(), grid, budget_gpu_hours=1000.0, seed=1, trial_fn=flaky_trial)

    assert len(result["trials"]) == 3
    statuses = [t["status"] for t in result["trials"]]
    assert any("failed" in s for s in statuses)
    assert result["best_trial"] is not None  # the two successful trials still yield a best


def test_run_budgeted_sweep_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        run_budgeted_sweep(_base_config(), {"training": {"lr": [0.1]}}, budget_gpu_hours=0, trial_fn=lambda c: 0.0)
    with pytest.raises(ValueError):
        run_budgeted_sweep(_base_config(), {"training": {"lr": [0.1]}}, budget_gpu_hours=-1.0, trial_fn=lambda c: 0.0)


def test_run_budgeted_sweep_rejects_empty_grid():
    with pytest.raises(ValueError):
        run_budgeted_sweep(_base_config(), {}, budget_gpu_hours=1.0, trial_fn=lambda c: 0.0)


def test_run_budgeted_sweep_rejects_invalid_mode():
    with pytest.raises(ValueError):
        run_budgeted_sweep(_base_config(), {"training": {"lr": [0.1]}}, budget_gpu_hours=1.0, trial_fn=lambda c: 0.0, mode="bogus")


def test_run_budgeted_sweep_all_failures_gives_no_best_trial():
    def always_fails(cfg):
        raise RuntimeError("nope")

    grid = {"training": {"lr": [0.1, 0.2]}}
    result = run_budgeted_sweep(_base_config(), grid, budget_gpu_hours=1000.0, trial_fn=always_fails)
    assert result["best_trial"] is None
