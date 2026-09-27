"""
stats/correction.py — Holm-Bonferroni multiple-comparison correction
(spec §10), applied across a *declared* comparison family — the set of
comparisons fixed before running any of them (declared in the experiment
config; see orchestration/schema.py's StatsConfig), not discovered by
running everything and correcting after the fact, which would be testing
against a different, invalid family.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from statsmodels.stats.multitest import multipletests


def holm_bonferroni(p_values: Sequence[float], family: Sequence[str], alpha: float = 0.05) -> List[Dict]:
    """p_values / family: parallel sequences — family[i] names what
    comparison p_values[i] belongs to (e.g. "proposed_vs_unet").

    Returns one dict per comparison — ``{"comparison", "p_value",
    "corrected_p_value", "reject"}`` — in the *same order* as the inputs
    (not sorted by p-value), so a caller can zip the result back against
    whatever else it's tracking per comparison.
    """
    if len(p_values) != len(family):
        raise ValueError(
            f"holm_bonferroni: p_values and family must be parallel (same "
            f"length), got {len(p_values)} vs {len(family)}"
        )
    if not p_values:
        return []
    reject, corrected, _, _ = multipletests(list(p_values), alpha=alpha, method="holm")
    return [
        {"comparison": name, "p_value": float(p), "corrected_p_value": float(c), "reject": bool(r)}
        for name, p, c, r in zip(family, p_values, corrected, reject)
    ]
