"""
tests/test_reporting.py — Phase 14: reporting layer
(reporting/tables.py, figures.py, inventory.py).

The test the plan names for this phase: test_reporting_blocks.
"""
from __future__ import annotations

import os

import pytest

from orchestration.ledger import LedgerWriter
from reporting.figures import _pareto_front_indices, render_critical_difference_figure, render_degradation_curve_figure, render_pareto_frontier_figure
from reporting.inventory import ARTEFACT_INVENTORY, audit_artefact_inventory
from reporting.tables import (
    BlockingRuleError,
    check_minimum_seeds,
    check_no_dirty_tree_runs,
    check_saliency_sanitized,
    check_stats_entries_present,
    read_runs_ledger,
    read_stats_ledger,
    render_main_comparison_table,
)


def _seed_clean_ledger(ledger_dir, models=("gmk_unet", "unet"), seeds=(1, 2, 3), dirty=False):
    ledger = LedgerWriter(ledger_dir)
    for model in models:
        for seed in seeds:
            ledger.append_run_row(
                run_id=f"{model}-s{seed}", config_hash="abc", experiment_name="exp",
                model_name=model, dataset_name="clinicdb", seed=seed, fold="",
                status="done", start_time="t0", end_time="t1", gpu_hours=0.1,
                best_metric=0.8, monitor_metric="val_dice", git_commit="deadbeef",
                git_dirty=dirty, manifest_path="m.json",
            )
    return ledger_dir


def _results(models=("gmk_unet", "unet"), seeds=(1, 2, 3)):
    out = []
    for model, base in zip(models, [0.78, 0.70]):
        for seed in seeds:
            out.append({"model": model, "dataset": "clinicdb", "seed": seed, "dice": base + seed * 0.001, "miou": base - 0.1, "hd95": 30.0})
    return out


# ---------------------------------------------------------------------------
# test_reporting_blocks (spec's named test) — every blocking rule genuinely
# refuses to emit a table for the corresponding bad input.
# ---------------------------------------------------------------------------

def test_reporting_blocks(tmp_path):
    ledger_dir = str(tmp_path / "ledger")

    # 1. dirty-tree run
    _seed_clean_ledger(ledger_dir, dirty=True)
    runs = read_runs_ledger(ledger_dir)
    with pytest.raises(BlockingRuleError):
        check_no_dirty_tree_runs(runs)

    # 2. under-seeded config
    ledger_dir2 = str(tmp_path / "ledger2")
    _seed_clean_ledger(ledger_dir2, seeds=(1, 2), dirty=False)
    runs2 = read_runs_ledger(ledger_dir2)
    with pytest.raises(BlockingRuleError):
        check_minimum_seeds(runs2)

    # 3. missing stats entry
    with pytest.raises(BlockingRuleError):
        check_stats_entries_present(["a_vs_b", "a_vs_c"], [{"comparison": "a_vs_b"}])

    # 4. unsanitised saliency
    with pytest.raises(BlockingRuleError):
        check_saliency_sanitized(["fig1", "fig2"], {"fig1": True})

    # render_main_comparison_table must propagate a block, not silently render
    with pytest.raises(BlockingRuleError):
        render_main_comparison_table(_results(), ledger_dir, snapshot_id="snap")


def test_reporting_does_not_block_clean_inputs(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir, dirty=False)
    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)  # must not raise
    check_minimum_seeds(runs)  # must not raise
    check_stats_entries_present(["a_vs_b"], [{"comparison": "a_vs_b"}])  # must not raise
    check_saliency_sanitized(["fig1"], {"fig1": True})  # must not raise

    out = render_main_comparison_table(_results(), ledger_dir, snapshot_id="snap")
    assert "gmk_unet" in out["csv"]
    assert "\\begin{tabular}" in out["latex"]


def test_check_minimum_seeds_only_counts_done_runs(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    ledger = LedgerWriter(ledger_dir)
    for seed, status in [(1, "done"), (2, "done"), (3, "failed")]:
        ledger.append_run_row(
            run_id=f"r{seed}", config_hash="h", experiment_name="e", model_name="m", dataset_name="d",
            seed=seed, fold="", status=status, start_time="t0", end_time="t1", gpu_hours=0.1,
            best_metric=0.8, monitor_metric="val_dice", git_commit="c", git_dirty=False, manifest_path="m.json",
        )
    runs = read_runs_ledger(ledger_dir)
    with pytest.raises(BlockingRuleError):
        check_minimum_seeds(runs)  # only 2 "done" seeds


def test_read_runs_ledger_missing_file_returns_empty(tmp_path):
    assert read_runs_ledger(str(tmp_path / "nonexistent")) == []


def test_read_stats_ledger_missing_file_returns_empty(tmp_path):
    assert read_stats_ledger(str(tmp_path / "nonexistent")) == []


def test_render_main_comparison_table_rejects_empty_results(tmp_path):
    ledger_dir = str(tmp_path / "ledger")
    _seed_clean_ledger(ledger_dir)
    with pytest.raises(ValueError):
        render_main_comparison_table([], ledger_dir, snapshot_id="snap")


# ---------------------------------------------------------------------------
# figures.py
# ---------------------------------------------------------------------------

def test_pareto_front_indices_hand_computed():
    rows = [
        {"model": "A", "dice": 0.9, "cost": 100},
        {"model": "B", "dice": 0.85, "cost": 50},
        {"model": "C", "dice": 0.7, "cost": 200},
    ]
    front = _pareto_front_indices(rows, "dice", "cost", higher_metric_is_better=True)
    assert front == {0, 1}  # C is dominated by A on both axes


def test_render_pareto_frontier_figure_smoke():
    rows = [
        {"model": "A", "dice": 0.9, "cost": 100},
        {"model": "B", "dice": 0.85, "cost": 50},
    ]
    fig = render_pareto_frontier_figure(rows, "dice", "cost", snapshot_id="snap")
    assert len(fig.axes) == 1


def test_render_pareto_frontier_figure_rejects_empty():
    with pytest.raises(ValueError):
        render_pareto_frontier_figure([], "dice", "cost", snapshot_id="snap")


def test_render_critical_difference_figure_uses_real_nemenyi_output():
    from stats.ranking import nemenyi_posthoc

    scores = {"a": [0.9, 0.8, 0.85], "b": [0.7, 0.6, 0.65], "c": [0.8, 0.75, 0.78]}
    nem = nemenyi_posthoc(scores)
    fig = render_critical_difference_figure(nem, snapshot_id="snap")
    assert len(fig.axes) == 1


def test_render_degradation_curve_figure_smoke():
    curve = [{"severity": 0, "mean_dice": 1.0, "n": 5}, {"severity": 1, "mean_dice": 0.9, "n": 5}]
    fig = render_degradation_curve_figure(curve, title="test", snapshot_id="snap")
    assert len(fig.axes) == 1


def test_render_degradation_curve_figure_rejects_empty():
    with pytest.raises(ValueError):
        render_degradation_curve_figure([], title="test", snapshot_id="snap")


# ---------------------------------------------------------------------------
# inventory.py
# ---------------------------------------------------------------------------

def test_artefact_inventory_matches_spec_row_count():
    # spec §20 lists 19 manuscript items (Table 1-7, Fig.1-9 + S1, CD diagram)
    assert len(ARTEFACT_INVENTORY) == 19


def test_audit_artefact_inventory_reports_missing_when_nothing_exists(tmp_path):
    result = audit_artefact_inventory(str(tmp_path))
    assert result["total"] == len(ARTEFACT_INVENTORY)
    assert "Fig. 1 Architecture" in result["skipped"]  # manual, always skipped
    assert len(result["missing"]) > 0
    assert result["present"] == []


def test_audit_artefact_inventory_detects_present_artefact(tmp_path):
    reports_root = tmp_path / "reports"
    tables_dir = reports_root / "tables"
    os.makedirs(tables_dir)
    (tables_dir / "datasets.csv").write_text("a,b\n1,2\n")
    result = audit_artefact_inventory(str(reports_root))
    assert "Table 1 Datasets" in result["present"]
