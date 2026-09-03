# Output layout — migration reference (for XDash)

This documents dissert's current `outputs/` layout after the
experiment-oriented redesign, so XDash's own refactor (a separate repo,
`../XDash`, not touched by this change) can pick it up without
re-deriving it from dissert's source. `outputs/` is entirely gitignored
in both repos — nothing under it is ever committed.

## Target layout

```
outputs/experiments/<experiment_id>/     # experiment_id = "{experiment_name}-s{seed}"
├── run_meta.json                # {experiment_name, seed, config_hash, created_at}
├── fold_splits.json             # K-fold only; shared across folds
├── checkpoints/
│   └── fold0/                   # omitted (files directly under checkpoints/) for non-CV runs
│       ├── best.pth
│       ├── last.pth
│       ├── epoch_0010.pth       # periodic, optional
│       └── manifest.json        # orchestration-driven runs only
├── logs/
│   ├── fold0.log
│   ├── fold1.log
│   └── eval.log
├── tensorboard/
│   └── fold0/                   # omitted for non-CV runs
├── plots/
│   └── fold0/                   # omitted for non-CV runs
│       ├── overlays/overlay_epoch_0010.png
│       └── curves/epoch_train_loss.png
└── eval/
    ├── report.json
    ├── report.md
    └── curves/{confusion_matrix,roc_curve,pr_curve}.png

outputs/ledger/{runs,compute,test_evals,stats}.csv
outputs/searches/{search,search_test}/{best_config.yaml,search_summary.csv,search_report.md}
outputs/reports/{main_comparison,efficiency}.{csv,tex}
```

The one source of truth for this layout in code is
`orchestration/runid.py`'s `experiment_id()` / `experiment_paths()` —
every producer (train.py, eval.py, orchestration/runner.py, search.py)
calls into it rather than constructing paths itself.

## Old → new mapping

| Old (flat, artifact-type-first) | New (nested, experiment-first) |
| --- | --- |
| `checkpoints/<experiment_name>/best_fold{N}.pth` | `experiments/<experiment_name>-s<seed>/checkpoints/fold{N}/best.pth` |
| `checkpoints/<experiment_name>/best.pth` (non-CV) | `experiments/<experiment_name>-s<seed>/checkpoints/best.pth` |
| `logs/<experiment_name>/fold{N}.log` | `experiments/<experiment_name>-s<seed>/logs/fold{N}.log` |
| `logs/<experiment_name>/eval.log` | `experiments/<experiment_name>-s<seed>/logs/eval.log` |
| `logs/<experiment_name>/overlays/overlay_epoch_{N}.png` | `experiments/<experiment_name>-s<seed>/plots/fold{N}/overlays/overlay_epoch_{N}.png` (was fold-unscoped — a real bug; now genuinely per-fold) |
| `logs/<experiment_name>/plots/*.png` (training curves) | `experiments/<experiment_name>-s<seed>/plots/fold{N}/curves/*.png` (was fold-unscoped — a real bug; now genuinely per-fold) |
| `logs/<experiment_name>/curves/{confusion_matrix,roc_curve,pr_curve}.png` | `experiments/<experiment_name>-s<seed>/eval/curves/*.png` |
| `logs/<experiment_name>/report.{json,md}` | `experiments/<experiment_name>-s<seed>/eval/report.{json,md}` |
| `runs/<experiment_name>_fold{N}/` (TensorBoard) | `experiments/<experiment_name>-s<seed>/tensorboard/fold{N}/` |
| `checkpoints/<experiment_name>/fold_splits.json` | `experiments/<experiment_name>-s<seed>/fold_splits.json` |
| `artifacts/runs/<run_id>/manifest.json` | `experiments/<experiment_name>-s<seed>/checkpoints/fold{N}/manifest.json` (or the exp root if `fold=None`) |
| `artifacts/ledger/*.csv` | `ledger/*.csv` (same CSV schemas, unchanged — still a flat, global, cross-experiment index, never nested per-experiment) |
| `search_results/`, `search_test_results/` | `searches/search/`, `searches/search_test/` — sweep-level aggregates only; **trials themselves are no longer a separate tree** — each trial is a full experiment at `experiments/<trial_name>-s<seed>/` |
| `reports/tables/*.{csv,tex}` | `reports/*.{csv,tex}` |

## New identifier: `experiment_id`

`experiment_id = "{experiment_name}-s{seed}"` — the one atomic,
self-contained directory for a given (experiment_name, seed), shared by
every fold trained within it. Deliberately **excludes** any config hash
so `train.py --resume` keeps working across ordinary config tweaks
(extending epochs, adjusting lr, etc.) — a same-name/same-seed rerun
lands back in the same directory rather than a fresh one.

Atomicity is instead enforced by `run_meta.json`, written once per
experiment_id with the config's hash (`orchestration.runid.config_hash`,
excludes `training.seed`) at creation time; every later run against the
same experiment_id compares its hash and logs a warning (never blocks)
on drift, so a genuine reconfiguration under the same name+seed is
visible rather than silently blended.

## Listing "experiments"

Previously XDash reconciled three independent sources (config filenames,
a flat directory walk under `logs_dir`, and the ledger/manifest files).
The new layout makes a directory walk under `outputs/experiments/`
sufficient on its own: each top-level entry *is* one experiment_id, with
`run_meta.json` inside giving `experiment_name`/`seed`/`config_hash`
directly, without needing to cross-reference the config directory or
parse directory-name conventions.

## Config knob

Everything above is rooted at one config field, `output_dir` (top-level,
default `"outputs/experiments"` — `configs/base.yaml`,
`orchestration/schema.py`'s `Config.output_dir`). This replaces the old
three independently-configurable-but-never-actually-independent
`checkpoint.save_dir` / `logging.log_dir` / `logging.tb_dir` fields.
