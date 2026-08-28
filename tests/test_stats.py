"""
tests/test_stats.py — Phase 9: statistics module (stats/tests.py,
effectsize.py, correction.py, ranking.py, stats/__init__.py's
run_family_comparison orchestration).
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon
from statsmodels.stats.multitest import multipletests

from stats import run_family_comparison
from stats.correction import holm_bonferroni
from stats.effectsize import cliffs_delta, paired_median_diff
from stats.ranking import friedman_test, nemenyi_posthoc
from stats.tests import bootstrap_ci, meaningfulness_gate, wilcoxon_paired_test


# ---------------------------------------------------------------------------
# tests.py
# ---------------------------------------------------------------------------

def test_wilcoxon_paired_test_matches_scipy_directly():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 30)
    b = rng.normal(0.75, 0.05, 30)
    ref = wilcoxon(a, b)
    out = wilcoxon_paired_test(a, b)
    assert out["statistic"] == pytest.approx(ref.statistic)
    assert out["p_value"] == pytest.approx(ref.pvalue)
    assert out["n"] == 30


def test_wilcoxon_paired_test_identical_arrays_returns_p1():
    a = [0.5, 0.6, 0.7]
    out = wilcoxon_paired_test(a, a)
    assert out == {"statistic": 0.0, "p_value": 1.0, "n": 3}


def test_wilcoxon_paired_test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        wilcoxon_paired_test([1, 2, 3], [1, 2])


def test_wilcoxon_paired_test_rejects_empty():
    with pytest.raises(ValueError):
        wilcoxon_paired_test([], [])


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    vals = rng.normal(0.8, 0.02, 200)
    ci = bootstrap_ci(vals, n_resamples=5000, seed=1)
    assert ci["point_estimate"] == pytest.approx(np.mean(vals))
    assert ci["ci_low"] < ci["point_estimate"] < ci["ci_high"]
    assert ci["n"] == 200


def test_bootstrap_ci_deterministic_given_seed():
    rng = np.random.default_rng(2)
    vals = rng.normal(0.5, 0.1, 50)
    a = bootstrap_ci(vals, seed=42)
    b = bootstrap_ci(vals, seed=42)
    assert a == b


def test_bootstrap_ci_accepts_median_statistic():
    rng = np.random.default_rng(3)
    vals = rng.normal(0.5, 0.1, 50)
    ci = bootstrap_ci(vals, statistic=np.median, seed=1)
    assert ci["point_estimate"] == pytest.approx(np.median(vals))


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


@pytest.mark.parametrize(
    "diff,min_diff,p,alpha,expected",
    [
        (0.05, 0.01, 0.01, 0.05, "statistically significant and practically meaningful"),
        (0.005, 0.01, 0.01, 0.05, "statistically significant but below the pre-registered meaningful-difference threshold"),
        (0.05, 0.01, 0.5, 0.05, "not statistically significant despite exceeding the meaningful-difference threshold (underpowered?)"),
        (0.005, 0.01, 0.5, 0.05, "neither statistically significant nor practically meaningful"),
    ],
)
def test_meaningfulness_gate_four_branches(diff, min_diff, p, alpha, expected):
    assert meaningfulness_gate(diff, min_diff, p, alpha) == expected


def test_meaningfulness_gate_uses_absolute_diff():
    # A negative observed diff (proposed worse) still counts as "meaningful"
    # if its magnitude clears the threshold.
    assert meaningfulness_gate(-0.05, 0.01, 0.01) == "statistically significant and practically meaningful"


# ---------------------------------------------------------------------------
# effectsize.py
# ---------------------------------------------------------------------------

def test_cliffs_delta_hand_computed():
    a, b = [3, 4, 5], [1, 2, 3]
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    expected = (greater - less) / (len(a) * len(b))
    assert cliffs_delta(a, b) == pytest.approx(expected)


def test_cliffs_delta_antisymmetric():
    a, b = [3, 4, 5], [1, 2, 3]
    assert cliffs_delta(a, b) == pytest.approx(-cliffs_delta(b, a))


def test_cliffs_delta_extremes():
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0


def test_cliffs_delta_rejects_empty():
    with pytest.raises(ValueError):
        cliffs_delta([], [1, 2])


def test_paired_median_diff():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 0.5, 0.5, 0.5])
    out = paired_median_diff(a, b)
    assert out["median_diff"] == pytest.approx(float(np.median(a - b)))
    assert out["mean_diff"] == pytest.approx(float(np.mean(a - b)))
    assert out["n"] == 4


def test_paired_median_diff_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired_median_diff([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# correction.py
# ---------------------------------------------------------------------------

def test_holm_bonferroni_matches_statsmodels_directly():
    pvals = [0.001, 0.02, 0.03, 0.04, 0.5]
    names = ["a", "b", "c", "d", "e"]
    out = holm_bonferroni(pvals, names)
    ref_reject, ref_corrected, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for o, rr, rc in zip(out, ref_reject, ref_corrected):
        assert o["reject"] == bool(rr)
        assert o["corrected_p_value"] == pytest.approx(rc)


def test_holm_bonferroni_preserves_input_order():
    pvals = [0.04, 0.001, 0.03]
    names = ["c", "a", "b"]
    out = holm_bonferroni(pvals, names)
    assert [o["comparison"] for o in out] == names


def test_holm_bonferroni_empty_input():
    assert holm_bonferroni([], []) == []


def test_holm_bonferroni_rejects_length_mismatch():
    with pytest.raises(ValueError):
        holm_bonferroni([0.1, 0.2], ["a"])


def test_holm_bonferroni_never_decreases_p_value():
    pvals = [0.001, 0.02, 0.03, 0.04, 0.5]
    out = holm_bonferroni(pvals, ["a", "b", "c", "d", "e"])
    for o in out:
        assert o["corrected_p_value"] >= o["p_value"] - 1e-12


# ---------------------------------------------------------------------------
# ranking.py
# ---------------------------------------------------------------------------

def test_friedman_test_matches_scipy_directly():
    rng = np.random.default_rng(0)
    scores = {
        "proposed": rng.normal(0.85, 0.03, 8).tolist(),
        "unet": rng.normal(0.78, 0.03, 8).tolist(),
        "emcad": rng.normal(0.80, 0.03, 8).tolist(),
    }
    out = friedman_test(scores)
    ref = friedmanchisquare(*[scores[k] for k in scores])
    assert out["statistic"] == pytest.approx(ref.statistic)
    assert out["p_value"] == pytest.approx(ref.pvalue)
    assert out["n_blocks"] == 8


def test_friedman_test_requires_at_least_three_methods():
    with pytest.raises(ValueError):
        friedman_test({"a": [1, 2], "b": [1, 2]})


def test_friedman_test_requires_equal_length_blocks():
    with pytest.raises(ValueError):
        friedman_test({"a": [1, 2, 3], "b": [1, 2], "c": [1, 2, 3]})


def test_nemenyi_avg_ranks_hand_computed():
    fixed = {"A": [3, 3, 3], "B": [2, 2, 2], "C": [1, 1, 1]}
    out = nemenyi_posthoc(fixed, higher_is_better=True)
    assert out["avg_ranks"] == {"A": 1.0, "B": 2.0, "C": 3.0}


def test_nemenyi_lower_is_better_flips_ranks():
    fixed = {"A": [3, 3, 3], "B": [2, 2, 2], "C": [1, 1, 1]}
    out = nemenyi_posthoc(fixed, higher_is_better=False)
    assert out["avg_ranks"] == {"A": 3.0, "B": 2.0, "C": 1.0}


def test_nemenyi_critical_difference_matches_demsar_2006():
    # Demsar (2006)'s published Nemenyi q_0.05 at k=3 is 2.343.
    q = float(studentized_range.ppf(0.95, 3, np.inf)) / np.sqrt(2)
    assert q == pytest.approx(2.343, abs=1e-3)


def test_nemenyi_pairwise_consistent_with_critical_difference():
    rng = np.random.default_rng(4)
    scores = {
        "proposed": rng.normal(0.85, 0.05, 10).tolist(),
        "unet": rng.normal(0.78, 0.05, 10).tolist(),
        "emcad": rng.normal(0.80, 0.05, 10).tolist(),
        "swin": rng.normal(0.79, 0.05, 10).tolist(),
    }
    out = nemenyi_posthoc(scores)
    assert len(out["pairwise"]) == 6  # 4 choose 2
    for pair in out["pairwise"]:
        assert pair["significant"] == (pair["rank_diff"] > out["critical_difference"])


def test_nemenyi_requires_at_least_three_methods():
    with pytest.raises(ValueError):
        nemenyi_posthoc({"a": [1, 2], "b": [1, 2]})


# ---------------------------------------------------------------------------
# stats/__init__.py — run_family_comparison orchestration
# ---------------------------------------------------------------------------

@pytest.fixture
def paired_scores():
    rng = np.random.default_rng(0)
    proposed = rng.normal(0.85, 0.03, 20)
    unet = proposed - rng.normal(0.05, 0.01, 20)
    emcad = proposed - rng.normal(0.01, 0.01, 20)
    return proposed.tolist(), unet.tolist(), emcad.tolist()


def test_run_family_comparison_writes_json(tmp_path, paired_scores):
    proposed, unet, emcad = paired_scores
    out_dir = tmp_path / "reports"
    result = run_family_comparison(
        family="mkunet_ablation",
        proposed_name="gmkunet_t",
        proposed_per_image=proposed,
        comparators={"unet": unet, "emcad": emcad},
        out_dir=str(out_dir),
        ledger_dir=None,
    )
    assert len(result["comparisons"]) == 2
    written = out_dir / "mkunet_ablation.json"
    assert written.exists()
    with open(written) as f:
        reloaded = json.load(f)
    assert reloaded["family"] == "mkunet_ablation"
    assert reloaded == {k: v for k, v in result.items() if k != "_written_to"}


def test_run_family_comparison_appends_ledger_stats_rows(tmp_path, paired_scores):
    proposed, unet, emcad = paired_scores
    ledger_dir = tmp_path / "ledger"
    run_family_comparison(
        family="mkunet_ablation",
        proposed_name="gmkunet_t",
        proposed_per_image=proposed,
        comparators={"unet": unet, "emcad": emcad},
        out_dir=None,
        ledger_dir=str(ledger_dir),
    )
    stats_csv = ledger_dir / "stats.csv"
    assert stats_csv.exists()
    with open(stats_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["comparison"] for r in rows} == {"gmkunet_t_vs_unet", "gmkunet_t_vs_emcad"}
    for r in rows:
        assert r["family"] == "mkunet_ablation"
        float(r["p_value"])
        float(r["corrected_p_value"])
        float(r["effect_size"])
        assert int(r["n"]) == 20


def test_run_family_comparison_holm_corrects_across_declared_family(tmp_path, paired_scores):
    proposed, unet, emcad = paired_scores
    result = run_family_comparison(
        family="f",
        proposed_name="p",
        proposed_per_image=proposed,
        comparators={"unet": unet, "emcad": emcad},
        out_dir=None,
        ledger_dir=None,
    )
    for comp in result["comparisons"]:
        assert comp["corrected_p_value"] >= comp["wilcoxon"]["p_value"] - 1e-12


def test_run_family_comparison_rejects_empty_comparators(paired_scores):
    proposed, _, _ = paired_scores
    with pytest.raises(ValueError):
        run_family_comparison("f", "p", proposed, {}, out_dir=None, ledger_dir=None)


def test_run_family_comparison_skips_side_effects_when_dirs_none(paired_scores):
    proposed, unet, _ = paired_scores
    result = run_family_comparison(
        "f", "p", proposed, {"unet": unet}, out_dir=None, ledger_dir=None
    )
    assert "_written_to" not in result
