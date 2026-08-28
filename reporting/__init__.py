"""
reporting/ — Phase 14 of IMPLEMENTATION_PLAN.md, spec §16's REPORTING
LAYER: renders manuscript tables/figures from already-computed JSON/
Parquet/ledger-CSV artefacts only, never a checkpoint, never a recomputed
metric. Four blocking rules (dirty-tree run, under-seeded config, missing
stats entry, unsanitised saliency) are hard raises, not warnings.
"""
from __future__ import annotations

from .figures import render_critical_difference_figure, render_degradation_curve_figure, render_pareto_frontier_figure
from .inventory import ARTEFACT_INVENTORY, ArtefactEntry, audit_artefact_inventory
from .tables import (
    REQUIRED_SEEDS,
    BlockingRuleError,
    check_minimum_seeds,
    check_no_dirty_tree_runs,
    check_saliency_sanitized,
    check_stats_entries_present,
    provenance_footer,
    read_runs_ledger,
    read_stats_ledger,
    render_efficiency_table,
    render_main_comparison_table,
)

__all__ = [
    "BlockingRuleError",
    "REQUIRED_SEEDS",
    "check_no_dirty_tree_runs",
    "check_minimum_seeds",
    "check_stats_entries_present",
    "check_saliency_sanitized",
    "provenance_footer",
    "read_runs_ledger",
    "read_stats_ledger",
    "render_main_comparison_table",
    "render_efficiency_table",
    "render_degradation_curve_figure",
    "render_pareto_frontier_figure",
    "render_critical_difference_figure",
    "ArtefactEntry",
    "ARTEFACT_INVENTORY",
    "audit_artefact_inventory",
]
