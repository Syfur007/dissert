"""
stats/ — statistics module (Phase 9 of IMPLEMENTATION_PLAN.md), completing
S6's deferred "stats attached" gate.

run_family_comparison() is the orchestration entry point tying the four
primitives together (tests.py, correction.py, effectsize.py, ranking.py):
given a proposed method's per-image scores and one or more comparators'
(same images, paired), it runs the paired Wilcoxon test + both effect
sizes + a bootstrap CI for each, Holm-Bonferroni-corrects across the
declared family (every comparator passed in one call — the family is
fixed by construction, not a subset chosen after seeing results), applies
the meaningfulness gate, writes reports/json/stats/<family>.json (spec
§10's "Output" row), and appends one Stats-table row per comparison to
orchestration.ledger (Phase 1's LedgerWriter.append_stats_row — the
scaffolded-but-previously-unused STATS_FIELDS table this phase finally
populates).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .correction import holm_bonferroni
from .effectsize import cliffs_delta, paired_median_diff
from .ranking import friedman_test, nemenyi_posthoc
from .tests import bootstrap_ci, meaningfulness_gate, wilcoxon_paired_test


def run_family_comparison(
    family: str,
    proposed_name: str,
    proposed_per_image: Sequence[float],
    comparators: Dict[str, Sequence[float]],
    min_meaningful_diff: float = 0.01,
    alpha: float = 0.05,
    out_dir: Optional[str] = "reports/json/stats",
    ledger_dir: Optional[str] = "outputs/ledger",
) -> Dict[str, Any]:
    """
    Args:
        family: name of this declared comparison family — also the output
            filename stem (reports/json/stats/<family>.json).
        proposed_name: the proposed method's name (for labelling only).
        proposed_per_image: proposed method's per-image scores.
        comparators: ``{comparator_name: per_image_scores}`` — every value
            must be paired with proposed_per_image (same images, same
            order, same length). Every key here is one comparison in the
            family Holm-Bonferroni corrects across.
        out_dir: where to write ``<family>.json``; None skips writing (for
            a caller that only wants the returned dict, e.g. a test).
        ledger_dir: where to append Stats-table rows; None skips it.

    Returns:
        The same dict written to ``<family>.json``.
    """
    proposed_per_image = list(proposed_per_image)
    if not comparators:
        raise ValueError("run_family_comparison: comparators must be non-empty")

    comparisons: List[Dict[str, Any]] = []
    raw_p_values: List[float] = []
    names: List[str] = []

    for comp_name, comp_scores in comparators.items():
        comp_scores = list(comp_scores)
        wtest = wilcoxon_paired_test(proposed_per_image, comp_scores)
        delta = cliffs_delta(proposed_per_image, comp_scores)
        med_diff = paired_median_diff(proposed_per_image, comp_scores)
        verdict = meaningfulness_gate(
            med_diff["median_diff"], min_meaningful_diff, wtest["p_value"], alpha
        )
        comparisons.append({
            "comparator": comp_name,
            "wilcoxon": wtest,
            "cliffs_delta": delta,
            "paired_median_diff": med_diff,
            "meaningfulness_verdict": verdict,
        })
        raw_p_values.append(wtest["p_value"])
        names.append(comp_name)

    corrected = holm_bonferroni(raw_p_values, names, alpha=alpha)
    for comp, corr in zip(comparisons, corrected):
        comp["corrected_p_value"] = corr["corrected_p_value"]
        comp["reject_null"] = corr["reject"]

    result: Dict[str, Any] = {
        "family": family,
        "proposed": proposed_name,
        "proposed_bootstrap_ci_over_images": bootstrap_ci(proposed_per_image),
        "comparisons": comparisons,
        "alpha": alpha,
        "min_meaningful_diff": min_meaningful_diff,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{family}.json")
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, path)
        result["_written_to"] = path

    if ledger_dir:
        from orchestration.ledger import LedgerWriter

        ledger = LedgerWriter(ledger_dir)
        for comp in comparisons:
            ledger.append_stats_row(
                family=family,
                comparison=f"{proposed_name}_vs_{comp['comparator']}",
                metric="wilcoxon_p_value",
                p_value=comp["wilcoxon"]["p_value"],
                corrected_p_value=comp["corrected_p_value"],
                effect_size=comp["cliffs_delta"],
                n=comp["wilcoxon"]["n"],
                timestamp=result["timestamp"],
            )

    return result


__all__ = [
    "run_family_comparison",
    "wilcoxon_paired_test",
    "bootstrap_ci",
    "meaningfulness_gate",
    "holm_bonferroni",
    "cliffs_delta",
    "paired_median_diff",
    "friedman_test",
    "nemenyi_posthoc",
]
