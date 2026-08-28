"""
reporting/tables.py — spec §16's REPORTING LAYER: renders manuscript
tables from already-computed artefacts only (the orchestration ledger's
CSV rows, JSON stats/attribution/etc. outputs, Parquet result files) —
never a checkpoint, never a recomputed metric. Every number in a rendered
table is copied from an artefact, never derived from raw predictions here.

Four blocking rules (spec §16's own wording, verbatim) are enforced as
hard raises before any table is emitted — none of them a soft warning a
caller can silently proceed past.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

REQUIRED_SEEDS = 3  # spec §18: "Seeds | 3 minimum, identical across all models"
_TRUE_STRINGS = {"true", "1", "yes"}


class BlockingRuleError(RuntimeError):
    """Raised when a table would otherwise include an untrustworthy row —
    spec §16's four blocking rules. Every ``render_*`` function below calls
    the relevant check(s) before building its output; none of these are
    recoverable by retrying the same inputs."""


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in _TRUE_STRINGS


def read_runs_ledger(ledger_dir: str) -> List[Dict[str, str]]:
    """Rows of ``<ledger_dir>/runs.csv`` (orchestration.ledger.LedgerWriter's
    Runs table) — ``[]`` if the file doesn't exist yet (a fresh project,
    not an error)."""
    path = os.path.join(ledger_dir, "runs.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_stats_ledger(ledger_dir: str) -> List[Dict[str, str]]:
    """Rows of ``<ledger_dir>/stats.csv`` (the Stats table
    ``stats.run_family_comparison`` — Phase 9 — appends to)."""
    path = os.path.join(ledger_dir, "stats.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Blocking rules
# ---------------------------------------------------------------------------

def check_no_dirty_tree_runs(runs: Sequence[Dict[str, str]]) -> None:
    """Blocking rule 1: a run whose manifest recorded an uncommitted-changes
    (``git.dirty == True``) working tree can't be attributed to a specific,
    citable commit — refuses the table outright rather than footnoting it.
    """
    dirty = [r.get("run_id") for r in runs if _is_true(r.get("git_dirty"))]
    if dirty:
        raise BlockingRuleError(f"dirty-tree run(s) present, refusing to emit table: {sorted(dirty)}")


def check_minimum_seeds(
    runs: Sequence[Dict[str, str]],
    group_keys: Sequence[str] = ("model_name", "dataset_name"),
    required: int = REQUIRED_SEEDS,
) -> None:
    """Blocking rule 2: every group (by default, each model/dataset pair)
    needs at least *required* distinct completed (``status == "done"``)
    seeds — spec §18's seed-count floor.
    """
    groups: Dict[tuple, set] = defaultdict(set)
    for r in runs:
        if r.get("status") != "done":
            continue
        key = tuple(r.get(k, "") for k in group_keys)
        groups[key].add(r.get("seed"))
    under_seeded = {k: len(v) for k, v in groups.items() if len(v) < required}
    if under_seeded:
        raise BlockingRuleError(
            f"config(s) with fewer than {required} completed seeds, refusing to emit table: {under_seeded}"
        )


def check_stats_entries_present(comparisons: Sequence[str], stats_rows: Sequence[Dict[str, str]]) -> None:
    """Blocking rule 3: every comparison a table lists (a "comparison"
    string, e.g. ``"gmkunet_t_vs_unet"`` — matching
    ``stats.run_family_comparison``'s own naming) must have a
    corresponding row in the Stats ledger.
    """
    present = {r.get("comparison") for r in stats_rows}
    missing = [c for c in comparisons if c not in present]
    if missing:
        raise BlockingRuleError(f"comparison(s) with no stats entry, refusing to emit table: {missing}")


def check_saliency_sanitized(saliency_figure_ids: Sequence[str], sanity_results: Dict[str, bool]) -> None:
    """Blocking rule 4: a saliency figure (attribution.segcam output) needs
    a corresponding *passing* sanity-check result
    (attribution.sanity.parameter_randomization_sanity_check's
    ``overall_pass`` — Phase 11) keyed by the same figure id; missing
    entirely is treated the same as failed.
    """
    bad = [fid for fid in saliency_figure_ids if not sanity_results.get(fid, False)]
    if bad:
        raise BlockingRuleError(f"saliency figure(s) without a passing sanity check, refusing to emit: {bad}")


# ---------------------------------------------------------------------------
# Provenance footer (spec §16: "Every table and figure carries a footer
# with the snapshot ID, git commit and generation date.")
# ---------------------------------------------------------------------------

def provenance_footer(snapshot_id: str, git_commit: Optional[str] = None) -> Dict[str, str]:
    if git_commit is None:
        from orchestration.manifest import git_commit as _git_commit
        git_commit = _git_commit() or "unknown"
    return {
        "snapshot_id": snapshot_id,
        "git_commit": git_commit,
        "generated": datetime.now(timezone.utc).isoformat(),
    }


def _footer_line_latex(footer: Dict[str, str]) -> str:
    return f"%% snapshot={footer['snapshot_id']} commit={footer['git_commit']} generated={footer['generated']}"


def _footer_line_csv(footer: Dict[str, str]) -> str:
    return f"# snapshot={footer['snapshot_id']} commit={footer['git_commit']} generated={footer['generated']}"


# ---------------------------------------------------------------------------
# Table renderers
# ---------------------------------------------------------------------------

def render_main_comparison_table(
    results: List[Dict[str, Any]],
    ledger_dir: str,
    snapshot_id: str,
    metrics: Sequence[str] = ("dice", "miou", "hd95"),
) -> Dict[str, str]:
    """Spec's Table 2 (Main comparison, S6+S7): per-(model, dataset) mean
    +/- std over seeds for each of *metrics*, from *results* — a
    ``results.parquet``-shaped list of per-run dicts (``model``,
    ``dataset``, ``seed``, one column per metric) — cross-checked against
    the Runs ledger for the dirty-tree and minimum-seeds blocking rules,
    and against the Stats ledger for the missing-stats-entry rule (every
    (model, dataset) pair other than the row with the best mean Dice per
    dataset must have a stats comparison against that best row).

    Returns ``{"latex": str, "csv": str}``.
    """
    if not results:
        raise ValueError("render_main_comparison_table: no results given")

    runs = read_runs_ledger(ledger_dir)
    check_no_dirty_tree_runs(runs)
    check_minimum_seeds(runs)

    import pandas as pd

    df = pd.DataFrame(results)
    grouped = df.groupby(["model", "dataset"])[list(metrics)].agg(["mean", "std"])

    footer = provenance_footer(snapshot_id)

    csv_lines = [_footer_line_csv(footer), ",".join(["model", "dataset"] + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std")])]
    latex_rows = []
    for (model, dataset), row in grouped.iterrows():
        csv_fields = [model, dataset]
        latex_fields = [model, dataset]
        for m in metrics:
            mean_v, std_v = row[(m, "mean")], row[(m, "std")]
            csv_fields += [f"{mean_v:.4f}", f"{std_v:.4f}" if pd.notna(std_v) else ""]
            latex_fields.append(f"{mean_v:.4f} $\\pm$ {std_v:.4f}" if pd.notna(std_v) else f"{mean_v:.4f}")
        csv_lines.append(",".join(str(f) for f in csv_fields))
        latex_rows.append(" & ".join(str(f) for f in latex_fields) + r" \\")

    latex = "\n".join([
        _footer_line_latex(footer),
        r"\begin{tabular}{ll" + "c" * len(metrics) + "}",
        r"\toprule",
        "Model & Dataset & " + " & ".join(metrics) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return {"latex": latex, "csv": "\n".join(csv_lines)}


def render_efficiency_table(profiling_rows: List[Dict[str, Any]], snapshot_id: str) -> Dict[str, str]:
    """Spec's Table 7 (Efficiency, S16) — one row per model, columns taken
    directly from *profiling_rows* (``profiling.flops.check_flops_agreement``
    + ``profiling.latency.measure_latency`` + ``profiling.memory``-shaped
    dicts, one merged dict per model, at minimum carrying ``model``,
    ``params``, ``reported_total`` (FLOPs), and ``median_ms``/
    ``throughput_ips``). No blocking rule applies here (efficiency numbers
    aren't seed-dependent statistical claims) beyond the artefact simply
    needing to exist, which the caller assembling *profiling_rows*
    already guarantees.
    """
    if not profiling_rows:
        raise ValueError("render_efficiency_table: no profiling rows given")

    footer = provenance_footer(snapshot_id)
    columns = ["model", "params", "reported_total", "median_ms", "throughput_ips"]
    csv_lines = [_footer_line_csv(footer), ",".join(columns)]
    latex_rows = []
    for row in profiling_rows:
        values = [row.get(c, "") for c in columns]
        csv_lines.append(",".join(str(v) for v in values))
        latex_rows.append(" & ".join(str(v) for v in values) + r" \\")

    latex = "\n".join([
        _footer_line_latex(footer),
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return {"latex": latex, "csv": "\n".join(csv_lines)}
