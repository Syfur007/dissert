# XDash resume contract — what dissert needs to provide

**Audience:** dissert. **Requested by:** XDash (`../XDash`), for automated
multi-leg training on Kaggle.
**Status:** proposal / not yet implemented on either side.

## 1. The problem

Kaggle terminates a notebook session at a hard platform limit (~9 h with a
GPU). Experiments that need longer than that currently cannot finish there
at all. XDash wants to run them as a **chain of legs**: train until the
budget is nearly spent, stop cleanly, carry the checkpoint forward as a
Kaggle dataset, and resume in the next notebook version — repeating until
the run reports `done`.

Almost all of that is XDash's job and is being built on the XDash side
(packaging checkpoints into a dataset, versioning it, re-pushing the
notebook, tracking legs). **Three things can only be done inside the
training process**, which is why this document exists.

## 2. Why a graceful stop is the whole design

The scheme must never depend on what Kaggle does to a session it kills.
Whether `/kaggle/working` output is committed after a hard timeout is
undocumented and not something to build on. So the training process must
**stop itself before the deadline and exit 0**, which makes the notebook a
normally-completed run whose output Kaggle commits like any other.

Everything below follows from that.

## 3. What already works (no change needed)

Verified while designing this — listed so nobody rebuilds it:

- `utils/checkpoint.py:57-93` — `save()` writes `last.pth` **atomically,
  every epoch**, carrying model, optimizer, scheduler, epoch and GradScaler
  state. That is a correct and sufficient resume point.
- `train.py:410,432` — `--resume` sets `checkpoint.resume`, and
  `train.py:297-299` loads `last.pth` from the fold-scoped checkpoint dir.
- `orchestration/runid.py:54` — `resume` is popped before the run-id hash,
  so a resumed leg keeps the **same `run_id`**. Legs therefore collapse into
  one ledger row rather than forking into several. This is exactly right for
  XDash and must stay true.

## 4. What XDash needs from dissert

### 4.1 A wall-clock training budget

New flag on `train.py` (name negotiable — XDash will call whatever you
choose):

```
--max-hours FLOAT        # stop cleanly once this much wall time is spent
```

Required semantics:

- Measure from process start.
- **Stop before exceeding it, not after.** After each epoch, project the
  next epoch's duration from observed epoch times; if
  `elapsed + projected > max_hours`, stop now. Overshooting the budget is
  the one failure mode that loses the whole leg, because Kaggle's own kill
  lands while the process is mid-epoch and possibly mid-write.
- Save `last.pth` before returning (the existing per-epoch save already
  covers this if the stop happens at an epoch boundary).
- Finalize the manifest with `status="interrupted"` (§4.2).
- **Exit 0.** A non-zero exit is indistinguishable from a real failure and
  will make XDash stop retrying the experiment.

A budget that expires before the *first* epoch completes should still exit 0
with `status="interrupted"` and `epochs_completed: 0`, so XDash can detect
"this leg made no progress" and stop chaining rather than looping forever on
a config whose epoch is longer than any available session.

### 4.2 An explicit `interrupted` status in the manifest

`orchestration/manifest.py` currently writes `pending` → `running` →
`done`/`failed` (plus `skipped-done` from the runner). There is no way to
say "stopped early, but healthy and resumable."

Add `interrupted`, and these fields alongside it:

| Field | Meaning |
| --- | --- |
| `status: "interrupted"` | Stopped on budget, not on error |
| `epochs_completed` | Epochs actually finished this run (cumulative across legs) |
| `total_epochs` | Target from the config, so XDash can show progress and estimate remaining legs |
| `resumable: true` | `last.pth` is present and loadable |
| `wall_seconds` | This leg's wall time, for XDash's own duration estimates |

Why XDash can't infer this: a hard-killed process never reaches
`manifest.finish()`, so the manifest stays `"running"` forever. XDash *can*
fall back on "downloaded artifact still says `running` + `last.pth` exists ⇒
it was interrupted", and will implement that so it isn't blocked on this
document — but that fallback cannot distinguish a budget stop from a crash
mid-epoch, and it only works at all if Kaggle happened to commit the output.
It is a stopgap, not the contract.

### 4.3 Fold splits must survive a resume

A resumed leg must train on **the same split** as the leg before it. If
`fold_splits.json` is regenerated on resume — from a fresh seed, a
re-shuffle, or a different dataset ordering — the legs silently train on
different data and the resulting run is invalid in a way no error message
will reveal.

Please confirm (and if necessary enforce) that `--resume` reuses the
existing `fold_splits.json` from the experiment directory and never rewrites
it. XDash will carry that file forward in the checkpoint dataset (§5) so it
is present before training starts.

### 4.4 A read-only status API for XDash's bridge

XDash already runs small scripts inside this repo's own interpreter
(`XDash/backend/bridge_scripts/`, executed with `bridge_python_executable`).
It needs to ask "what state is this experiment in?" without parsing `.pth`
files or reimplementing path layout.

Please expose something stable and importable, e.g.:

```python
# orchestration/status.py
def describe_run(experiment_dir: str) -> dict:
    """
    {
      "run_id": str,
      "status": "pending|running|done|interrupted|failed",
      "resumable": bool,          # last.pth present and loadable
      "epochs_completed": int,
      "total_epochs": int,
      "has_fold_splits": bool,
      "checkpoint_files": [str],  # repo-relative, what XDash must carry forward
    }
    """
```

`checkpoint_files` is the important one: **dissert decides what a resume
needs**, and XDash just ships that list. Otherwise XDash is guessing at your
layout, and every change to it silently breaks resume. Keep the list minimal
— it is uploaded and re-downloaded once per leg.

## 5. What XDash does with this (so the contract makes sense)

Per experiment, one Kaggle notebook and one Kaggle dataset, each versioned
per leg:

1. Leg 1: push notebook v1 with the training dataset attached; run with
   `--max-hours <budget>`.
2. Download output. Read `describe_run()`.
   - `done` → finished, register in the ledger, delete the checkpoint
     dataset.
   - `interrupted` + `resumable` → continue.
   - `failed` → stop, surface the error.
3. Stage `checkpoint_files` into a folder, `kaggle datasets create` (leg 1)
   or `kaggle datasets version` (later legs).
4. Push notebook v(N+1) with **both** datasets attached; the notebook copies
   the checkpoint dataset into the canonical experiment directory and adds
   `--resume`.
5. Repeat until `done` or a leg cap is hit.

`--max-hours` is set below the platform limit by a margin covering setup and
teardown. Setup is not small: the launch template's own comment notes that
resolving a `pip install` without a prebuilt wheel "can silently burn 20-40+
minutes", and that time comes out of the same session.

## 6. Summary of asks

| # | Ask | Where |
| --- | --- | --- |
| 1 | `--max-hours` with projected-epoch early stop, exit 0 | `train.py` |
| 2 | `interrupted` status + `epochs_completed`/`total_epochs`/`resumable`/`wall_seconds` | `orchestration/manifest.py` |
| 3 | Confirm `fold_splits.json` is reused, never regenerated, on `--resume` | `train.py` / orchestration |
| 4 | `orchestration/status.py:describe_run()` returning the dict in §4.4 | new module |

1 and 2 are what make chained legs possible at all. 3 is a correctness
guard — without it, resumed runs are quietly invalid. 4 is what keeps XDash
from hardcoding assumptions about this repo's layout.
