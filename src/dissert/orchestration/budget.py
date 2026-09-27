"""
orchestration/budget.py — wall-clock training budgets.

Some compute has a hard session limit: a shared-cluster slot with a maximum
walltime, a preemptible/spot instance, a hosted notebook runtime. When the
platform kills such a session it lands wherever training happened to be —
mid-epoch, possibly mid-write — and whether anything written up to that
point survives is not something to build on.

The fix is for training to stop *itself* at a clean epoch boundary before
the limit lands, and exit 0, so the session ends as a normal completion
whose output is preserved like any other. This module holds the one piece
of shared state that makes that possible: a deadline, plus the "will the
next unit of work still fit?" question every caller asks against it.

Two consumers, one budget object shared between them (see ``train.py``):

- ``training.trainer.Trainer`` — checks before each epoch, using the
  longest epoch observed so far as the projection.
- ``orchestration.runner.run_sweep`` — checks before each
  (seed, repeat, fold) combination, using the longest combination observed
  so far. Without this second level, a 9-to-45-run default sweep would
  cheerfully start a fresh run with three minutes left and burn the
  remaining session on setup for zero completed epochs.

Deliberately *not* a config field: a budget is a property of the machine a
session runs on, not of the experiment. Keeping it out of the config dict
keeps it out of ``orchestration.runid.config_hash``, so two sessions of the
same experiment under different budgets resolve to the same output
directory and the second can resume the first. It is threaded as a runtime
argument instead, exactly like ``fold`` and ``repeat``.
"""
from __future__ import annotations

import time
from typing import Optional


class WallClockBudget:
    """A deadline, measured in monotonic seconds from *start*.

    Args:
        max_hours: total wall time this budget allows.
        start: monotonic timestamp the budget is measured from. Defaults to
            "now"; ``train.py`` passes its module-import timestamp so the
            budget covers process startup (imports, CUDA init, dataset
            caching) and not just the training loop.

    ``time.monotonic()`` rather than ``time.time()``: an NTP step or a
    daylight-saving change must not move a deadline that a session's
    survival depends on.
    """

    def __init__(self, max_hours: float, start: Optional[float] = None):
        if max_hours <= 0:
            raise ValueError(f"max_hours must be positive, got {max_hours!r}.")
        self.max_hours = float(max_hours)
        self.max_seconds = float(max_hours) * 3600.0
        self.start = time.monotonic() if start is None else float(start)

    def elapsed(self) -> float:
        """Seconds spent since *start*."""
        return time.monotonic() - self.start

    def remaining(self) -> float:
        """Seconds left before the deadline. Negative once overrun."""
        return self.max_seconds - self.elapsed()

    def exhausted_by(self, projected_seconds: float = 0.0) -> bool:
        """True if doing *projected_seconds* more work would overrun.

        Callers pass the most pessimistic estimate they have for the next
        unit of work (the *longest* epoch/run observed so far, not the mean
        — overshooting is the one failure mode that loses a whole session,
        so the projection errs toward stopping early). With the default
        ``0.0`` this degenerates to "is the deadline already past", which is
        what a caller with no observations yet asks.
        """
        return self.elapsed() + projected_seconds > self.max_seconds

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"WallClockBudget(max_hours={self.max_hours}, "
            f"elapsed={self.elapsed():.1f}s, remaining={self.remaining():.1f}s)"
        )
