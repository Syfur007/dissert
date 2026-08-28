"""
reporting/inventory.py — spec §20's ARTEFACT INVENTORY MAPPED TO THE
MANUSCRIPT table, as data plus an audit function — the same "codify the
audit itself" pattern tests/test_ci_audit.py established for spec §19's
test table, so the mapping lives in one place a future phase's rename
can't silently drift out of sync with.
"""
from __future__ import annotations

import os
from typing import Dict, List, NamedTuple


class ArtefactEntry(NamedTuple):
    manuscript_item: str
    produced_by: str  # pipeline stage(s), e.g. "S6+S7"
    source_artefact: str  # path relative to the reports/ root, per spec's own column


# Spec §20's table. Paths are normalised to be uniformly relative to the
# reports/ root (spec §16's Output row: "reports/tables/*.tex + *.csv,
# reports/figures/*.pdf, reports/json/*") — spec's own table mixes a
# reports/-prefixed path for Table 1 with bare filenames for everything
# else; the Table 1 entry is de-prefixed here so every source_artefact in
# this list means the same thing: "relative to reports/", not "relative
# to the repo root for this one row and reports/ for every other row".
ARTEFACT_INVENTORY: List[ArtefactEntry] = [
    ArtefactEntry("Table 1 Datasets", "S1", "tables/datasets.csv"),
    ArtefactEntry("Table 2 Main comparison", "S6+S7", "results.parquet"),
    ArtefactEntry("Table 2 Main comparison (stats)", "S6+S7", "stats/main.json"),
    ArtefactEntry("Table 3 External validation", "S15", "stats/external.json"),
    ArtefactEntry("Table 4 Ablation", "S8", "ablation.parquet"),
    ArtefactEntry("Table 5 Channel modes", "S9", "channels.parquet"),
    ArtefactEntry("Table 6 Channel attribution", "S10", "attribution/channels.json"),
    ArtefactEntry("Table 7 Efficiency", "S16", "profiling.json"),
    ArtefactEntry("Fig. 1 Architecture", "manual", ""),
    ArtefactEntry("Fig. 2 Pareto frontier", "S6+S16", "pareto.json"),
    ArtefactEntry("Fig. 3 Per-image distributions", "S6", "per_image.parquet"),
    ArtefactEntry("Fig. 4 Attribution bar + Shapley heat", "S10", "attribution/*.json"),
    ArtefactEntry("Fig. 5 Stage utilisation", "S10", "attribution/branch.json"),
    ArtefactEntry("Fig. 6 Shortcut / translation curves", "S11+S14", "robustness/geometric.json"),
    ArtefactEntry("Fig. 7 ERF and CKA", "S12", "analysis/*.json"),
    ArtefactEntry("Fig. 8 Calibration and retention", "S13", "uncertainty/*.json"),
    ArtefactEntry("Fig. 9 Qualitative + failure gallery", "S12", "failures/*"),
    ArtefactEntry("Fig. S1 Saliency panels (supplement)", "S10", "segcam/* with sanity/*"),
    ArtefactEntry("Critical-difference diagram", "S7", "stats/ranking.json"),
]


def audit_artefact_inventory(reports_root: str) -> Dict[str, object]:
    """Checks which of ``ARTEFACT_INVENTORY``'s source artefacts actually
    exist on disk under *reports_root* — a run-time completeness check
    (which manuscript items are producible *right now*), not a test of the
    mapping's correctness (that's a fixed table, verified by matching spec
    §20 directly, the same way ``tests/test_ci_audit.py`` verifies spec
    §19's test list).

    Entries with ``produced_by == "manual"`` (Fig. 1, hand-drawn) or an
    empty/glob-containing source path are reported as ``skipped`` — glob
    patterns (``attribution/*.json``) name a *family* of files, not one
    this function resolves; a caller wanting that resolved can glob
    *reports_root* directly using the same pattern.

    Returns ``{"present": [...], "missing": [...], "skipped": [...],
    "total": int}``, each list of manuscript_item strings.
    """
    present, missing, skipped = [], [], []
    for entry in ARTEFACT_INVENTORY:
        if entry.produced_by == "manual" or not entry.source_artefact or "*" in entry.source_artefact:
            skipped.append(entry.manuscript_item)
            continue
        full_path = os.path.join(reports_root, entry.source_artefact)
        if os.path.exists(full_path):
            present.append(entry.manuscript_item)
        else:
            missing.append(entry.manuscript_item)

    return {
        "present": present,
        "missing": missing,
        "skipped": skipped,
        "total": len(ARTEFACT_INVENTORY),
    }
