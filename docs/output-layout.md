# Output layout — migration reference (for XDash)

This documents dissert's current `outputs/` layout after the
experiment-oriented redesign, so XDash's own refactor (a separate repo,
`../XDash`, not touched by this change) can pick it up without
re-deriving it from dissert's source. `outputs/` is entirely gitignored
in both repos — nothing under it is ever committed.

By default, `train.py`/`eval.py` each sweep 3 seeds (`[7, 42, 1337]`)
automatically per experiment config (`--seed N` for a single explicit
seed instead) — see `SESSION_GROUPING_PLAN.md`. Cross-validation
(`k_fold.enabled`) defaults to off; when it's turned back on for a
specific experiment, all 3 seeds train/validate on the *same* fold
partition (only model-init/training randomness varies across seeds).

On top of that seed axis, each swept seed is by default also re-run 3
times (`--repeats N` to override, `--seed` bypasses repeats entirely):
same seed, re-seeded identically each time, to measure/average out
whatever noise survives fixed seeding — residual hardware/kernel
non-determinism (`training/determinism.py` — cudnn, fused kernels like
`mamba-ssm`'s), as distinct from the seed axis's deliberate model-init/
data-order variance. `repeat` is never part of the config dict, same as
`fold` — it's a pure runtime/path parameter.

## Target layout

```
outputs/experiments/<experiment_name>/
├── <hash7>-fold_splits.json     # K-fold only; shared across all seeds AND repeats of this config_hash
├── <hash7>-s<seed>-r0/          # experiment_id = "{config_hash[:7]}-s{seed}-r{repeat}"
│   ├── run_meta.json            # {experiment_name, seed, config_hash, created_at}
│   ├── checkpoints/
│   │   └── fold0/               # omitted (files directly under checkpoints/) for non-CV runs
│   │       ├── best.pth
│   │       ├── last.pth
│   │       ├── epoch_0010.pth   # periodic, optional
│   │       └── manifest.json    # orchestration-driven runs only
│   ├── logs/
│   │   ├── fold0.log
│   │   ├── fold1.log
│   │   └── eval.log
│   ├── tensorboard/
│   │   └── fold0/                   # omitted for non-CV runs
│   ├── plots/
│   │   └── fold0/                   # omitted for non-CV runs
│   │       ├── overlays/overlay_epoch_0010.png
│   │       └── curves/epoch_train_loss.png
│   └── eval/
│       ├── report.json          # or ensemble_report.json with --ensemble
│       ├── report.md
│       └── curves/{confusion_matrix,roc_curve,pr_curve}.png
├── <hash7>-s<seed>-r1/            # same shape, one per repeat of this seed
├── <hash7>-s<seed>-r2/
├── <hash7>-s<seed>.json           # seed_combined_report: mean±std over that seed's repeats
├── <hash7>-s<seed2>-r0/           # same shape, one per (seed, repeat) combination swept
├── ...
└── <experiment_name>.json        # combined report: mean±std over all swept seeds' <hash7>-s<seed>.json

outputs/ledger/{runs,compute,test_evals,stats}.csv
outputs/searches/{search,search_test}/{best_config.yaml,search_summary.csv,search_report.md}
outputs/reports/{main_comparison,efficiency}.{csv,tex}
```

Repeats are opt-out, not opt-in: `--repeats 1` (or `--seed`, which bypasses
the whole apparatus) drops the `-r{repeat}` suffix entirely, landing back
on the exact legacy `<hash7>-s<seed>/` layout above with no
`<hash7>-s<seed>.json` file — same `(None,)` "axis not in use" convention
`fold` already has.

The one source of truth for this layout in code is
`orchestration/runid.py`'s `experiment_id()` / `experiment_paths()` —
every producer (train.py, eval.py, orchestration/runner.py, search.py,
datasets/datamodule.py) calls into it rather than constructing paths
itself.

## Old → new mapping

| Old (`<experiment_name>-s<seed>/...`, no hash in path) | New (`<experiment_name>/<hash7>-s<seed>/...`) |
| --- | --- |
| `experiments/<experiment_name>-s<seed>/checkpoints/fold{N}/best.pth` | `experiments/<experiment_name>/<hash7>-s<seed>/checkpoints/fold{N}/best.pth` |
| `experiments/<experiment_name>-s<seed>/checkpoints/best.pth` (non-CV) | `experiments/<experiment_name>/<hash7>-s<seed>/checkpoints/best.pth` |
| `experiments/<experiment_name>-s<seed>/logs/*.log` | `experiments/<experiment_name>/<hash7>-s<seed>/logs/*.log` |
| `experiments/<experiment_name>-s<seed>/plots/fold{N}/...` | `experiments/<experiment_name>/<hash7>-s<seed>/plots/fold{N}/...` |
| `experiments/<experiment_name>-s<seed>/tensorboard/fold{N}/` | `experiments/<experiment_name>/<hash7>-s<seed>/tensorboard/fold{N}/` |
| `experiments/<experiment_name>-s<seed>/eval/report.{json,md}` | `experiments/<experiment_name>/<hash7>-s<seed>/eval/report.{json,md}` |
| `experiments/<experiment_name>-s<seed>/fold_splits.json` (per seed) | `experiments/<experiment_name>/<hash7>-fold_splits.json` (shared across all seeds of that hash — see below) |
| — (didn't exist) | `experiments/<experiment_name>/<experiment_name>.json` — combined, seed-averaged eval report |
| `artifacts/ledger/*.csv` | `ledger/*.csv` (Runs table gains a `repeat` column, empty string when repeats aren't in use — still a flat, global, cross-experiment index, never nested per-experiment) |
| `search_results/`, `search_test_results/` | `searches/search/`, `searches/search_test/` — sweep-level aggregates only; each trial is a full experiment at `experiments/<trial_name>/<hash7>-s<seed>/` |
| `reports/tables/*.{csv,tex}` | `reports/*.{csv,tex}` |

## New identifier: `experiment_id`

`experiment_id = "{config_hash[:7]}-s{seed}[-r{repeat}]"`, nested one
level under `experiment_name` — the one atomic, self-contained directory
for a given (config_hash, seed[, repeat]), shared by every fold trained
within it. `repeat` (like `fold`) is never part of the config dict itself
— a pure runtime/path parameter — so it needs no `config_hash` stripping;
the `-r{repeat}` segment is simply omitted when `repeat is None` (repeats
not in use for this run).

`config_hash` (`orchestration.runid.config_hash`) strips only
bookkeeping/presentation fields — `training.seed`, the whole `logging`
section (experiment_name + display knobs), `output_dir`,
`checkpoint.checkpoint_path`/`resume`/`periodic_save_every`, and `stats`
— everything else (`model`, `dataset`, `training.*` except seed, `k_fold`,
`checkpoint.monitor_metric`/`mode`, `early_stopping`, `stages`) is part of
the hash. Two configs that differ only in the stripped fields describe
the same experiment and hash identically; anything else produces a
different hash and therefore a different directory.

This **inverts** the old scheme's behavior: previously `experiment_id`
deliberately excluded the hash so `--resume` kept landing in the same
directory across an "ordinary" tweak (extending epochs, adjusting lr),
with `run_meta.json` catching genuine drift after the fact via a warning.
Now the hash is load-bearing in the path itself — any change to a hashed
field lands in a fresh directory, so `--resume` only ever continues a
byte-identical setup. `run_meta.json` is still written per
(experiment_name, seed) directory, but its drift check can now only fire
on a truncated-hash collision (two different full hashes sharing the same
first 7 hex chars), not on a genuine reconfiguration — that case is
structurally impossible to collide anymore.

## Shared fold splits, repeat-averaged and seed-averaged reports

Three files live directly under `experiments/<experiment_name>/`,
siblings to every `<hash7>-s<seed>[-r<repeat>]/` directory rather than
inside one:

- **`<hash7>-fold_splits.json`** (K-fold only) — the train/val partition
  for each fold, computed once per config_hash and reused by every seed
  *and repeat* swept under it (`datasets.datamodule.KFoldDataModule`). Its
  `KFold` `random_state` is derived from `config_hash`, not `training.seed`
  (and not repeat at all), so this is deterministic and reproducible
  independent of which seed/repeat runs first. It is also what makes a
  partition survive a resume: `config_hash` strips `checkpoint.resume`, so a
  resuming run resolves the same filename and reuses the cached partition. A
  mismatch is unreachable short of a truncated-hash collision, but would
  silently invalidate a resumed run if it did happen, so `KFoldDataModule`
  raises `datasets.splits.FoldSplitDriftError` there rather than regenerating
  (a non-resuming run keeps the existing warn-and-regenerate path).
- **`<hash7>-s<seed>.json`** (repeats only) — written by `eval.py`'s
  default multi-repeat path (`utils.report.aggregate_repeat_reports`)
  after all of one seed's repeats are evaluated: mean ± std, per metric,
  over that seed's `n_repeats` independent `eval/report.json` files. This
  is the inner half of two-level noise reduction — it isolates whatever
  noise survives *fixed* seeding (hardware/kernel non-determinism), as
  distinct from the seed axis's deliberate variance. Absent when repeats
  aren't in use (`--repeats 1` or `--seed`).
- **`<experiment_name>.json`** — written by `eval.py`'s default multi-seed
  path (`utils.report.aggregate_seed_reports`) after all swept seeds are
  evaluated: mean ± std, per metric, over each seed's report (its raw
  `eval/report.json` when repeats aren't in use, or its repeat-averaged
  `<hash7>-s<seed>.json` above when they are — `metrics` +
  `per_class_metrics` blocks only; `model`/`efficiency`/`environment`
  aren't averaged either level). Overwritten on every re-sweep under the
  same experiment_name — if that name is later re-run under a different
  config_hash, its old `<hash7>-s<seed>[-r<repeat>]/` directories stay on
  disk, but only the most recent hash's combined report survives at this
  path.

## Bounded sessions: run statuses and what a resume needs

A default sweep is 9 runs for a non-CV config (3 seeds × 3 repeats) and 45
for a 5-fold one, which routinely outlives one session on compute with a
hard walltime limit (a shared-cluster slot, a preemptible instance, a
hosted notebook runtime). `train.py --max-hours FLOAT` bounds a session:
training stops itself at a clean **epoch boundary** before the budget would
be overrun, exits **0**, and leaves a resumable `last.pth`. Re-running the
same command continues where it stopped. Without the flag nothing changes —
every check is skipped.

The budget is measured from process start and shared by the whole sweep, at
two levels: a run stops mid-training when the next epoch wouldn't fit
(projected from the *longest* epoch observed so far), and `run_sweep`
declines to start a further (seed, repeat, fold) whose projected duration
wouldn't fit (projected from the longest run so far). `--max-hours` is a
CLI argument only, never a config field — so it stays out of `config_hash`
and two sessions under different budgets share one directory.

`--max-hours` works on the `--seed` single-run bypass too, but that path
writes no manifest (it never has), so a budget stop there is clean and exits
0 yet is invisible to the status API below. Use the default sweep path when
a session needs to be inspectable.

Run statuses, as written to `checkpoints/[fold<N>/]manifest.json`:

| Status | Meaning |
| --- | --- |
| `pending` / `running` | Never started / started and not yet finished. A manifest still reading `running` after the fact means the process was **hard-killed** — it never reached `manifest.finish()` |
| `done` | Completed its epochs (or early-stopped). Skipped by later sweeps unless `--force` |
| `interrupted` | Stopped itself on the wall-clock budget. Healthy and resumable — **not** skipped by later sweeps, which is what makes a sweep resumable |
| `failed` | Raised. Recorded with the error; the rest of the sweep continues |

Every terminal status also carries `epochs_completed` (cumulative across
sessions, since a resume continues the epoch counter rather than restarting
it), `total_epochs`, `resumable`, and `wall_seconds`. `run_sweep`'s
*returned* statuses add two that are never written to a manifest:
`skipped-done` (idempotent skip) and `skipped-budget` (not started — no
time left).

`orchestration/status.py` reads all of this back:

```python
from orchestration.status import describe_experiment, describe_run

describe_experiment("outputs/experiments/<experiment_name>")   # whole sweep
describe_run("outputs/experiments/<experiment_name>/<hash7>-s<seed>")
```

Both return `status`/`resumable`/`epochs_completed`/`total_epochs`/
`has_fold_splits` plus **`checkpoint_files`** — the repo-relative paths that
must travel with a run for a resume to be *correct*, so callers moving runs
between machines never have to re-derive this layout:

- `checkpoints/[fold<N>/]last.pth` — the resume point.
- `checkpoints/[fold<N>/]best.pth` — **not optional.** `train.py` re-seeds
  `CheckpointManager.best_metric` from it on resume; without it the resumed
  session compares against `-inf` and overwrites a genuinely better
  checkpoint with a worse one.
- `checkpoints/[fold<N>/]manifest.json`, `run_meta.json` — status and identity.
- `<hash7>-fold_splits.json` — without it a resumed K-fold run rebuilds its
  partition and trains on different data than its checkpoint saw.

The module deliberately imports no torch at module scope, so it stays cheap
to import for inspection; `verify=True` opts into actually loading each
`last.pth` rather than trusting its presence (presence is normally enough —
`utils.checkpoint.atomic_torch_save` writes via temp-file + `os.replace`, so
a torn checkpoint isn't a reachable state).

## Listing "experiments"

Previously XDash reconciled three independent sources (config filenames,
a flat directory walk under `logs_dir`, and the ledger/manifest files).
A directory walk under `outputs/experiments/` is still sufficient on its
own, now two levels deep: each top-level entry is one `experiment_name`
(a group of seed runs, potentially spanning more than one `config_hash`
if the name was reused across a config change), and each child directory
is one `experiment_id`, with `run_meta.json` inside giving
`experiment_name`/`seed`/`repeat`/`config_hash` directly.

## Config knob

Everything above is rooted at one config field, `output_dir` (top-level,
default `"outputs/experiments"` — `configs/base.yaml`,
`orchestration/schema.py`'s `Config.output_dir`). This replaces the old
three independently-configurable-but-never-actually-independent
`checkpoint.save_dir` / `logging.log_dir` / `logging.tb_dir` fields.
