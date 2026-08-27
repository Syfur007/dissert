"""
orchestration/ledger.py — append-only run bookkeeping across four tables
(Runs, Compute, Test_Evals, Stats), one flat CSV file each under a ledger
directory.

CSV, not xlsx: xlsx is fragile under concurrent writers (two sweep processes
appending at once can corrupt the zip container); a CSV append is a single
``open(..., "a")`` + one row + close, safe under concurrent appenders in the
way an open-for-the-whole-run xlsx workbook handle is not.

Only ``Runs`` and ``Compute`` are populated by Phase 1 (via
:class:`LedgerWriter` from :mod:`orchestration.runner`). ``Test_Evals`` rows
are appended by Phase 3's ``issue_test_token()`` (the guarded test-loader
mechanism doesn't exist yet); ``Stats`` rows are appended by Phase 9. Both
tables' schemas are declared here now so every consumer of this module
agrees on the column set from the start, even though nothing writes to them
yet.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

RUNS_FIELDS = [
    "run_id", "config_hash", "experiment_name", "model_name", "dataset_name",
    "seed", "fold", "status", "start_time", "end_time", "gpu_hours",
    "best_metric", "monitor_metric", "git_commit", "git_dirty", "manifest_path",
]

COMPUTE_FIELDS = [
    "run_id", "wall_seconds", "gpu_hours", "peak_mem_mb", "device",
]

TEST_EVALS_FIELDS = [
    "run_id", "token", "issued_time", "config_hash", "checkpoint_path",
]

STATS_FIELDS = [
    "family", "comparison", "metric", "p_value", "corrected_p_value",
    "effect_size", "n", "timestamp",
]

_TABLE_FIELDS = {
    "runs": RUNS_FIELDS,
    "compute": COMPUTE_FIELDS,
    "test_evals": TEST_EVALS_FIELDS,
    "stats": STATS_FIELDS,
}


class LedgerWriter:
    """One instance per ledger directory (typically ``artifacts/ledger/``)."""

    def __init__(self, ledger_dir: str = "artifacts/ledger"):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def _table_path(self, table: str) -> Path:
        return self.ledger_dir / f"{table}.csv"

    def _append_row(self, table: str, row: Dict[str, Any]) -> None:
        fields = _TABLE_FIELDS[table]
        unknown = set(row) - set(fields)
        if unknown:
            raise ValueError(
                f"{table}: unknown field(s) {sorted(unknown)}; expected one of {fields}"
            )
        path = self._table_path(table)
        is_new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if is_new:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})

    def append_run_row(self, **kwargs: Any) -> None:
        self._append_row("runs", kwargs)

    def append_compute_row(self, **kwargs: Any) -> None:
        self._append_row("compute", kwargs)

    def append_test_eval_row(self, **kwargs: Any) -> None:
        self._append_row("test_evals", kwargs)

    def append_stats_row(self, **kwargs: Any) -> None:
        self._append_row("stats", kwargs)

    def has_done_run(self, run_id: str) -> bool:
        """True if *run_id* already has a row with ``status == "done"`` in
        the Runs table — used by :mod:`orchestration.runner` for the
        idempotent-skip check. (The manifest file is the primary source of
        truth for this — see ``orchestration.runner._load_existing_status``
        — this is a secondary check for when the ledger and manifest are
        consulted independently.)"""
        path = self._table_path("runs")
        if not path.exists():
            return False
        with open(path, newline="") as f:
            return any(
                row.get("run_id") == run_id and row.get("status") == "done"
                for row in csv.DictReader(f)
            )
