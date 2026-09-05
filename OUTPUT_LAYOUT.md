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

## Target layout

```
outputs/experiments/<experiment_name>/
├── <hash7>-fold_splits.json     # K-fold only; shared across all seeds of this config_hash
├── <hash7>-s<seed>/             # experiment_id = "{config_hash[:7]}-s{seed}"
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
├── <hash7>-s<seed2>/             # same shape, one per swept seed
├── <hash7>-s<seed3>/
└── <experiment_name>.json        # combined report: mean±std of eval metrics over all swept seeds

outputs/ledger/{runs,compute,test_evals,stats}.csv
outputs/searches/{search,search_test}/{best_config.yaml,search_summary.csv,search_report.md}
outputs/reports/{main_comparison,efficiency}.{csv,tex}
```

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
| `artifacts/ledger/*.csv` | `ledger/*.csv` (same CSV schemas, unchanged — still a flat, global, cross-experiment index, never nested per-experiment) |
| `search_results/`, `search_test_results/` | `searches/search/`, `searches/search_test/` — sweep-level aggregates only; each trial is a full experiment at `experiments/<trial_name>/<hash7>-s<seed>/` |
| `reports/tables/*.{csv,tex}` | `reports/*.{csv,tex}` |

## New identifier: `experiment_id`

`experiment_id = "{config_hash[:7]}-s{seed}"`, nested one level under
`experiment_name` — the one atomic, self-contained directory for a given
(config_hash, seed), shared by every fold trained within it.

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

## Shared fold splits, seed-averaged report

Two files live directly under `experiments/<experiment_name>/`, siblings
to every `<hash7>-s<seed>/` directory rather than inside one:

- **`<hash7>-fold_splits.json`** (K-fold only) — the train/val partition
  for each fold, computed once per config_hash and reused by every seed
  swept under it (`datasets.datamodule.KFoldDataModule`). Its `KFold`
  `random_state` is derived from `config_hash`, not `training.seed`, so
  this is deterministic and reproducible independent of which seed runs
  first.
- **`<experiment_name>.json`** — written by `eval.py`'s default multi-seed
  path (`utils.report.aggregate_seed_reports`) after all swept seeds are
  evaluated: mean ± std, per metric, over each seed's `eval/report.json`
  (`metrics` + `per_class_metrics` blocks only — `model`/`efficiency`/
  `environment` aren't averaged). Overwritten on every re-sweep under the
  same experiment_name — if that name is later re-run under a different
  config_hash, its old `<hash7>-s<seed>/` directories stay on disk, but
  only the most recent hash's combined report survives at this path.

## Listing "experiments"

Previously XDash reconciled three independent sources (config filenames,
a flat directory walk under `logs_dir`, and the ledger/manifest files).
A directory walk under `outputs/experiments/` is still sufficient on its
own, now two levels deep: each top-level entry is one `experiment_name`
(a group of seed runs, potentially spanning more than one `config_hash`
if the name was reused across a config change), and each child directory
is one `experiment_id`, with `run_meta.json` inside giving
`experiment_name`/`seed`/`config_hash` directly.

## Config knob

Everything above is rooted at one config field, `output_dir` (top-level,
default `"outputs/experiments"` — `configs/base.yaml`,
`orchestration/schema.py`'s `Config.output_dir`). This replaces the old
three independently-configurable-but-never-actually-independent
`checkpoint.save_dir` / `logging.log_dir` / `logging.tb_dir` fields.
