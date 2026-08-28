"""
reporting/figures.py — spec §16's REPORTING LAYER, figure side: renders
matplotlib figures directly from already-computed artefact shapes (this
module's inputs are exactly the return values of stats.ranking.nemenyi_posthoc,
profiling.flops.check_flops_agreement, robustness.common.degradation_curve,
etc. — no re-derivation of any number here, only plotting of what those
modules already computed). Every figure carries spec's provenance footer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .tables import provenance_footer


def _add_provenance_footer(fig, snapshot_id: str, git_commit: Optional[str] = None) -> None:
    footer = provenance_footer(snapshot_id, git_commit)
    fig.text(
        0.01, 0.01,
        f"snapshot={footer['snapshot_id']}  commit={footer['git_commit'][:12]}  generated={footer['generated']}",
        fontsize=6, color="gray", ha="left", va="bottom",
    )


def render_degradation_curve_figure(curve: List[Dict[str, Any]], title: str, snapshot_id: str):
    """spec's Fig. 6 (Shortcut/translation curves, S11+S14) and the general
    "Degradation curve: metric vs severity" figure every corruption family
    produces — *curve* is exactly ``robustness.common.degradation_curve``'s
    (or ``robustness.geometric.geometric_degradation_curve``'s) return
    value: a list of ``{"severity": int, "mean_dice": float, "n": int}``
    dicts, severity 0 = clean baseline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curve:
        raise ValueError("render_degradation_curve_figure: empty curve")

    severities = [row["severity"] for row in curve]
    dices = [row["mean_dice"] for row in curve]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(severities, dices, marker="o")
    ax.set_xlabel("Severity (0 = clean)")
    ax.set_ylabel("Dice")
    ax.set_title(title)
    ax.set_xticks(severities)
    ax.grid(True, alpha=0.3)
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig


def render_pareto_frontier_figure(
    profiling_rows: List[Dict[str, Any]],
    metric_key: str,
    cost_key: str,
    snapshot_id: str,
    higher_metric_is_better: bool = True,
):
    """spec's Fig. 2 (Pareto frontier, S6+S16) — *profiling_rows*: one dict
    per model, at minimum carrying ``"model"``, *metric_key* (e.g.
    ``"dice"``) and *cost_key* (e.g. ``profiling.flops.check_flops_agreement``'s
    ``"reported_total"``, or a latency figure). Points on the Pareto front
    (no other model is both better on *metric_key* and cheaper on
    *cost_key*) are highlighted and connected by a step line; dominated
    points are plotted but not connected.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not profiling_rows:
        raise ValueError("render_pareto_frontier_figure: no profiling rows given")

    pareto_indices = _pareto_front_indices(profiling_rows, metric_key, cost_key, higher_metric_is_better)

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, row in enumerate(profiling_rows):
        on_front = i in pareto_indices
        ax.scatter(
            row[cost_key], row[metric_key],
            color="tab:blue" if on_front else "lightgray",
            zorder=3 if on_front else 2,
        )
        ax.annotate(row.get("model", str(i)), (row[cost_key], row[metric_key]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    front_sorted = sorted((profiling_rows[i] for i in pareto_indices), key=lambda r: r[cost_key])
    if front_sorted:
        ax.plot([r[cost_key] for r in front_sorted], [r[metric_key] for r in front_sorted], color="tab:blue", linestyle="--", zorder=1)

    ax.set_xlabel(cost_key)
    ax.set_ylabel(metric_key)
    ax.set_title("Pareto frontier")
    ax.grid(True, alpha=0.3)
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig


def _pareto_front_indices(rows: List[Dict[str, Any]], metric_key: str, cost_key: str, higher_metric_is_better: bool) -> set:
    """Index set of the non-dominated rows: row i is on the front unless
    some other row j has metric >= i's metric (or <=, if lower-is-better)
    *and* cost <= i's cost, with at least one strict inequality (i.e. j
    Pareto-dominates i).
    """
    front = set()
    n = len(rows)
    for i in range(n):
        mi, ci = rows[i][metric_key], rows[i][cost_key]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            mj, cj = rows[j][metric_key], rows[j][cost_key]
            metric_at_least_as_good = (mj >= mi) if higher_metric_is_better else (mj <= mi)
            cost_at_least_as_good = cj <= ci
            strictly_better = (mj > mi if higher_metric_is_better else mj < mi) or (cj < ci)
            if metric_at_least_as_good and cost_at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(i)
    return front


def render_critical_difference_figure(nemenyi_result: Dict[str, Any], snapshot_id: str):
    """spec's "Critical-difference diagram" (S7) — *nemenyi_result* is
    exactly ``stats.ranking.nemenyi_posthoc``'s return value: average rank
    per method plotted on a single axis, with the critical-difference span
    drawn as a reference bar, and a horizontal bracket over any pair whose
    ranks are *not* significantly different (rank_diff <= critical_difference).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = nemenyi_result["methods"]
    avg_ranks = nemenyi_result["avg_ranks"]
    cd = nemenyi_result["critical_difference"]

    order = sorted(methods, key=lambda m: avg_ranks[m])
    ranks_sorted = [avg_ranks[m] for m in order]

    fig, ax = plt.subplots(figsize=(6, 2 + 0.3 * len(methods)))
    y_positions = list(range(len(order)))
    ax.scatter(ranks_sorted, y_positions, color="tab:blue", zorder=3)
    for y, m, r in zip(y_positions, order, ranks_sorted):
        ax.text(r, y, f"  {m} ({r:.2f})", va="center", fontsize=8)

    # CD reference bar, anchored at the best (lowest-rank) method.
    best_rank = ranks_sorted[0]
    ax.plot([best_rank, best_rank + cd], [-1, -1], color="black", linewidth=2, clip_on=False)
    ax.text(best_rank + cd / 2, -1.3, f"CD = {cd:.2f}", ha="center", fontsize=8, clip_on=False)

    # Brackets over statistically-indistinguishable pairs (adjacent in rank).
    bracket_y = len(order)
    for pair in nemenyi_result["pairwise"]:
        if not pair["significant"]:
            ia, ib = order.index(pair["method_a"]), order.index(pair["method_b"])
            lo, hi = sorted((ia, ib))
            ax.plot([ranks_sorted[lo], ranks_sorted[hi]], [bracket_y, bracket_y], color="tab:red", linewidth=1.5)
            bracket_y += 0.5

    ax.set_yticks([])
    ax.set_xlabel("Average rank")
    ax.set_title("Critical-difference diagram")
    ax.set_ylim(-2, bracket_y + 0.5)
    ax.invert_yaxis()
    _add_provenance_footer(fig, snapshot_id)
    fig.tight_layout()
    return fig
