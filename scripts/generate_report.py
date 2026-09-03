"""
scripts/generate_report.py — the reporting layer's one real CLI entry
point: glob eval.py's JSON report dumps (utils/report.py's
EvaluationReporter._write_json output — the "S6 Main comparison" and
"S16 Profiling" artefacts, already-computed, never re-touched here),
assemble them into the results/profiling row-shapes
reporting.tables.render_main_comparison_table/render_efficiency_table
expect, and write outputs/reports/*.{csv,tex}.

Only reads JSON (spec §16: "The reporting layer cannot read checkpoints
or recompute metrics") — every number in the emitted tables is copied
from an eval.py JSON dump's "metrics"/"model"/"efficiency" section, never
recomputed from a prediction array.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reporting.tables import BlockingRuleError, render_efficiency_table, render_main_comparison_table  # noqa: E402


def _load_eval_reports(pattern: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"generate_report: no eval.py JSON reports matched '{pattern}'")
    reports = []
    for p in paths:
        with open(p) as f:
            reports.append(json.load(f))
    return reports


def _to_results_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in reports:
        cfg = r.get("config", {})
        rows.append({
            "model": r["model"]["name"],
            "dataset": cfg.get("dataset", {}).get("name", "unknown"),
            "seed": cfg.get("training", {}).get("seed", 0),
            "dice": r["metrics"].get("dice"),
            "miou": r["metrics"].get("miou"),
            "hd95": r["metrics"].get("hd95"),
        })
    return rows


def _to_profiling_rows(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in reports:
        rows.append({
            "model": r["model"]["name"],
            "params": r["model"]["params"],
            "reported_total": r["model"]["flops"],
            "median_ms": r["efficiency"].get("latency", {}).get("median_ms"),
            "throughput_ips": r["efficiency"].get("throughput_fps"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render manuscript tables from eval.py JSON reports")
    parser.add_argument("--reports-glob", type=str, required=True, help="Glob for eval.py *_report.json files")
    parser.add_argument("--ledger-dir", type=str, default="outputs/ledger")
    parser.add_argument("--out-dir", type=str, default="outputs/reports")
    parser.add_argument("--snapshot-id", type=str, required=True)
    args = parser.parse_args()

    eval_reports = _load_eval_reports(args.reports_glob)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        main_table = render_main_comparison_table(_to_results_rows(eval_reports), args.ledger_dir, args.snapshot_id)
    except BlockingRuleError as exc:
        print(f"BLOCKED: main comparison table refused — {exc}", file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(args.out_dir, "main_comparison.csv"), "w") as f:
        f.write(main_table["csv"])
    with open(os.path.join(args.out_dir, "main_comparison.tex"), "w") as f:
        f.write(main_table["latex"])

    efficiency_table = render_efficiency_table(_to_profiling_rows(eval_reports), args.snapshot_id)
    with open(os.path.join(args.out_dir, "efficiency.csv"), "w") as f:
        f.write(efficiency_table["csv"])
    with open(os.path.join(args.out_dir, "efficiency.tex"), "w") as f:
        f.write(efficiency_table["latex"])

    print(f"Wrote {args.out_dir}/main_comparison.{{csv,tex}} and efficiency.{{csv,tex}} from {len(eval_reports)} eval report(s).")


if __name__ == "__main__":
    main()
