"""
stats/ranking.py — cross-dataset ranking: Friedman test + Nemenyi
post-hoc, critical-difference diagram *data* (spec §10). Rendering the
actual diagram is Phase 14's reporting/figures.py's job (per the spec's
own artefact-inventory table: the source artefact is stats/ranking.json,
data, not an image) — this module produces exactly the numbers that JSON
needs (avg_ranks, critical_difference, which method pairs are/aren't
significantly different), not a plot.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def friedman_test(scores_by_method: Dict[str, List[float]]) -> Dict[str, object]:
    """scores_by_method: ``{method_name: [score_per_dataset_or_fold, ...]}``,
    every list the *same* length — Friedman needs a complete block design
    (every method scored on the same set of datasets/folds).
    """
    names = list(scores_by_method.keys())
    if len(names) < 3:
        raise ValueError("friedman_test: needs at least 3 methods to compare")
    arrays = [np.asarray(scores_by_method[n], dtype=float) for n in names]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"friedman_test: every method needs the same number of scores, got lengths {lengths}")

    stat, p = friedmanchisquare(*arrays)
    return {"statistic": float(stat), "p_value": float(p), "methods": names, "n_blocks": len(arrays[0])}


def nemenyi_posthoc(
    scores_by_method: Dict[str, List[float]], alpha: float = 0.05, higher_is_better: bool = True
) -> Dict[str, object]:
    """Nemenyi post-hoc test following a significant Friedman result:
    average rank per method and the critical difference (CD) two methods'
    average ranks must exceed to be called significantly different at
    *alpha* (Demšar, 2006).

    Args:
        higher_is_better: True for a metric like Dice (default) — scores
            are negated before ranking so rank 1 = best. False for a
            metric like HD95 where a lower value is better.
    """
    names = list(scores_by_method.keys())
    if len(names) < 3:
        raise ValueError("nemenyi_posthoc: needs at least 3 methods")
    arrays = np.array([scores_by_method[n] for n in names], dtype=float)  # (k, n)
    if higher_is_better:
        arrays = -arrays
    k, n = arrays.shape

    # rankdata ranks ascending within each block (column) -> rank 1 = best
    # after the higher_is_better negation above.
    ranks = np.apply_along_axis(rankdata, 0, arrays)
    avg_ranks = ranks.mean(axis=1)

    # Demšar (2006)'s Nemenyi q_alpha is the studentized-range critical
    # value divided by sqrt(2) — verified against the paper's published
    # table (e.g. q_0.05 at k=3 is 2.343): studentized_range.ppf(0.95, 3,
    # inf) / sqrt(2) == 2.343.
    q_alpha = float(studentized_range.ppf(1 - alpha, k, np.inf)) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            pairwise.append({
                "method_a": names[i], "method_b": names[j],
                "rank_diff": float(diff), "significant": bool(diff > cd),
            })

    return {
        "methods": names,
        "avg_ranks": {name: float(r) for name, r in zip(names, avg_ranks)},
        "critical_difference": float(cd),
        "alpha": alpha,
        "higher_is_better": higher_is_better,
        "pairwise": pairwise,
    }
