"""
tests/test_report_aggregation.py — utils.report's two-level noise-reduction
aggregation: aggregate_repeat_reports (same seed, N identical re-runs) and
aggregate_seed_reports (N different seeds), sharing one mean/std core
(_aggregate_metric_reports).
"""
from __future__ import annotations

import json

import pytest

from dissert.orchestration.runid import config_hash, experiment_paths
from dissert.evaluation.report import aggregate_repeat_reports, aggregate_seed_reports


def _write_report(path, dice, iou):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(
            {
                "metrics": {"dice": dice, "miou": iou, "fpr_on_normals": None},
                "per_class_metrics": {"dice": [dice, dice - 0.1]},
            },
            fh,
        )


def test_aggregate_repeat_reports_mean_std(tmp_path, tiny_config):
    cfg = tiny_config
    h = config_hash(cfg)
    seed = 42
    paths = []
    for i, dice in enumerate([0.8, 0.9, 1.0]):
        p = tmp_path / f"repeat{i}" / "report.json"
        _write_report(p, dice=dice, iou=dice - 0.05)
        paths.append(str(p))

    out_path = aggregate_repeat_reports(cfg, paths, repeats=[0, 1, 2], seed=seed)

    expected_path = experiment_paths(
        cfg.get("output_dir", "outputs/experiments"), cfg["logging"]["experiment_name"], h, seed,
    )["seed_combined_report"]
    assert out_path == expected_path

    with open(out_path) as fh:
        combined = json.load(fh)

    assert combined["seed"] == seed
    assert combined["n_repeats"] == 3
    assert combined["repeats"] == [0, 1, 2]
    # metrics holds the repeat-mean as a flat scalar (same shape a raw
    # single-run report.json uses) — deliberately NOT nested as
    # {"mean","std"} — so this file is a drop-in report_paths entry for
    # aggregate_seed_reports next.
    assert combined["metrics"]["dice"] == pytest.approx(0.9)
    assert combined["metrics_repeat_std"]["dice"] == pytest.approx(0.0816496580927726)
    # A metric that's None in every report stays unset (mean/std both None) —
    # same convention metrics/aggregate.py itself uses for undefined values.
    assert combined["metrics"]["fpr_on_normals"] is None
    assert combined["metrics_repeat_std"]["fpr_on_normals"] is None
    assert combined["per_class_metrics"]["dice"] == pytest.approx([0.9, 0.8])


def test_aggregate_seed_reports_consumes_repeat_combined_reports(tmp_path, tiny_config):
    cfg = tiny_config
    h = config_hash(cfg)

    seed_paths = []
    for i, seed in enumerate([7, 42, 1337]):
        p = tmp_path / f"seed{seed}.json"
        _write_report(p, dice=0.7 + 0.1 * i, iou=0.6 + 0.1 * i)
        seed_paths.append(str(p))

    out_path = aggregate_seed_reports(cfg, seed_paths, seeds=[7, 42, 1337])

    expected_path = experiment_paths(
        cfg.get("output_dir", "outputs/experiments"), cfg["logging"]["experiment_name"], h, 7,
    )["combined_report"]
    assert out_path == expected_path

    with open(out_path) as fh:
        combined = json.load(fh)

    assert combined["n_seeds"] == 3
    assert combined["seeds"] == [7, 42, 1337]
    assert combined["metrics"]["dice"]["mean"] == pytest.approx(0.8)


def test_aggregate_seed_reports_chained_after_aggregate_repeat_reports(tmp_path, tiny_config):
    """The actual eval.py flow when repeats are in use: each seed's repeats
    are combined first (aggregate_repeat_reports), and THOSE output files —
    not raw per-run report.json — are what feeds aggregate_seed_reports.
    Regression test for a real bug this exact chaining hit: an earlier
    version of aggregate_repeat_reports wrote its per-metric mean nested as
    {"mean", "std"} (matching aggregate_seed_reports' own terminal shape),
    which aggregate_seed_reports then choked on trying to np.mean() a dict.
    """
    cfg = tiny_config

    seed_combined_paths = []
    for seed_idx, seed in enumerate([7, 42]):
        repeat_paths = []
        for i in range(2):
            p = tmp_path / f"seed{seed}" / f"repeat{i}" / "report.json"
            _write_report(p, dice=0.8 + 0.05 * seed_idx + 0.01 * i, iou=0.7)
            repeat_paths.append(str(p))
        seed_combined_paths.append(
            aggregate_repeat_reports(cfg, repeat_paths, repeats=[0, 1], seed=seed)
        )

    combined_path = aggregate_seed_reports(cfg, seed_combined_paths, seeds=[7, 42])
    with open(combined_path) as fh:
        combined = json.load(fh)

    assert combined["n_seeds"] == 2
    assert isinstance(combined["metrics"]["dice"]["mean"], float)
    assert isinstance(combined["metrics"]["dice"]["std"], float)
