# Multi-seed-by-default training/eval + hash-scoped experiment layout

## Problem

Today `train.py`/`eval.py` each run exactly one seed per invocation
(`training.seed` from the config, or `eval.py --seed` to override it). A
study that wants 3 seeds per config needs 3 manual invocations per mode —
nothing loops it. `orchestration.runner.run_sweep` already loops
`seeds x folds` for training, but it's only wired up for orchestrated
sweeps (`scripts/reproduce.sh`, `search.py`), not for a bare
`python train.py --config X`. Eval has no sweep entry point at all.

Separately: `k_fold.enabled: true` is the `configs/base.yaml` default, so
every experiment also multiplies by up to 5 folds unless it opts out.

This file's earlier version proposed fixing the *launch-count* problem via
a pair of XDash-facing wrapper scripts (`scripts/run_seed_sweep.py`,
`scripts/eval_seed_sweep.py`) pointed to by `XDash/repos/dissert.yaml`.
That's dropped — whatever XDash needs from this is XDash's own concern.
What follows instead makes 3-seed averaging **the default behavior of
`train.py`/`eval.py` themselves**, with cross-validation off by default,
and reorganises `outputs/experiments/` so a config's identity (its hash)
is visible in the path instead of only inside `run_meta.json`.

## Summary of changes

1. `configs/base.yaml`: `k_fold.enabled` → `false`.
2. `orchestration/runid.py::config_hash`: exclude more bookkeeping fields
   (currently only `training.seed` is stripped).
3. `orchestration/runid.py::experiment_paths`: fold `config_hash` into the
   directory name — `outputs/experiments/<experiment_name>-s<seed>/` →
   `outputs/experiments/<experiment_name>/<hash7>-s<seed>/`. `fold_splits.json`
   moves out of the per-seed directory entirely — when both CV and the
   3-seed sweep are on, all 3 seeds must train/validate on the *same* fold
   partition (only the seed used for model init/training stochasticity
   should vary) — see §3b.
4. `train.py`: default CLI invocation sweeps `DEFAULT_SEEDS = [7, 42, 1337]`
   via `run_sweep` instead of training exactly one seed.
5. `eval.py`: default CLI invocation evaluates all 3 seeds in-process, then
   writes a combined, averaged report adjacent to the seed directories.
6. Docs/scripts that describe the old layout (`OUTPUT_LAYOUT.md`,
   `README.md`, `CODE_REVIEW.md`, `scripts/reproduce.sh`,
   `datasets/datamodule.py`, `utils/logger.py`) get updated in the same
   change so nothing describes a layout that no longer exists.

No runs exist yet under `outputs/experiments/` — this is a clean cutover,
no migration needed.

---

## 1. `k_fold.enabled: false`

`configs/base.yaml`:

```yaml
k_fold:
  enabled: false          # was: true
  n_splits: 5
  run_folds: null
```

Grep confirms no `configs/experiment/**/*.yaml` overrides `k_fold.enabled`
today, so this one edit turns CV off everywhere. `orchestration.schema.
KFoldConfig.enabled` can stay `True` as the pydantic fallback (defense in
depth for a config that omits the whole `k_fold:` block) — `base.yaml` is
composed into every experiment config and always wins.

## 2. `config_hash`: exclude bookkeeping fields

Current (`orchestration/runid.py:20`) strips only `training.seed`. Extend
the stripped set to everything that's true bookkeeping/presentation and
doesn't change what training or eval actually produces:

- `logging` (whole section) — `experiment_name` is a label, and
  `log_interval`/`save_overlays`/`overlay_save_every`/`overlay_n_samples`
  only affect which side-artifacts get written, not the model or metrics.
- `output_dir` — where to write, not what to run.
- `checkpoint.checkpoint_path`, `checkpoint.resume`,
  `checkpoint.periodic_save_every` — resume/save-cadence bookkeeping.
  `checkpoint.monitor_metric`/`checkpoint.mode` stay **in** the hash — they
  change which checkpoint is selected as "best", i.e. the actual result.
- `stats` (whole section) — declares which significance-testing family a
  result feeds into; doesn't change the run itself.

Everything else (`model`, `dataset`, `training.*` except `seed`, `k_fold`,
`checkpoint.monitor_metric`/`mode`, `early_stopping`, `stages`) stays in
the hash — those are the atomic, reproducible description of the setup.

```python
def config_hash(resolved_config: Dict[str, Any]) -> str:
    stripped = copy.deepcopy(resolved_config)
    training = stripped.get("training")
    if isinstance(training, dict):
        training.pop("seed", None)
    stripped.pop("logging", None)
    stripped.pop("output_dir", None)
    stripped.pop("stats", None)
    checkpoint = stripped.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint.pop("checkpoint_path", None)
        checkpoint.pop("resume", None)
        checkpoint.pop("periodic_save_every", None)
    canonical = json.dumps(stripped, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()
```

`tests/test_orchestration.py` needs a new case alongside
`test_config_hash_stable_across_seed`/`test_config_hash_changes_with_real_change`:
hash must stay stable across an `experiment_name` or `output_dir` change,
and still change on a real `training.lr`/`model.*` change (existing test
already covers the latter).

## 3. `experiment_paths`: hash-scoped directory

New layout: `outputs/experiments/<experiment_name>/<hash[:7]>-s<seed>/`
(same 7-hex-char truncation `run_id` already uses, for consistency).

```python
def experiment_id(config_hash_: str, seed: int) -> str:
    return f"{config_hash_[:7]}-s{seed}"

def experiment_paths(
    base_dir: str,
    experiment_name: str,
    config_hash_: str,
    seed: int,
    fold: Optional[int] = None,
) -> Dict[str, str]:
    root = os.path.join(base_dir, experiment_name, experiment_id(config_hash_, seed))
    ...  # unchanged below this line
    return {
        "root": root,
        "checkpoints": _fold_scoped("checkpoints"),
        "logs": os.path.join(root, "logs"),
        "tensorboard": _fold_scoped("tensorboard"),
        "plots": _fold_scoped("plots"),
        "eval": os.path.join(root, "eval"),
        "run_meta": os.path.join(root, "run_meta.json"),
        # Not seed-scoped — see §3b. One shared file per (experiment_name,
        # config_hash), sibling to every <hash7>-s<seed>/ dir it's used by.
        "fold_splits": os.path.join(
            base_dir, experiment_name, f"{config_hash_[:7]}-fold_splits.json"
        ),
        # combined-report path, adjacent to every <hash>-s<seed>/ directory
        # under this experiment_name, independent of hash/seed.
        "combined_report": os.path.join(base_dir, experiment_name, f"{experiment_name}.json"),
    }
```

`config_hash_` becomes a required positional arg — every call site already
computes (or can cheaply compute) it:

| Call site | Change needed |
| --- | --- |
| `train.py:100` (`experiment_paths(...)`) | already computes `resolved_config_hash` at line 89 — just pass it through |
| `eval.py:193`, `eval.py:245` | doesn't compute `config_hash` today except inside the `--allow-test-eval` branch — hoist that call to run unconditionally near the top of `main()`/the per-seed function |
| `orchestration/runner.py:41` (`_manifest_path`) | already has `h` computed at line 93 — thread it through `_manifest_path` |
| `datasets/datamodule.py:335` (`KFoldDataModule.__init__`) | needs `config_hash(config)` computed there too — also drives the fold-partition RNG now, not just the path; see §3b |

**Explicit behavior change, called out on purpose:** `OUTPUT_LAYOUT.md`
currently documents that `experiment_id` deliberately *excludes* the
config hash so `--resume` keeps landing in the same directory across
"ordinary" tweaks (extending epochs, adjusting lr). Folding the hash into
the path inverts that: any change to a hashed field now lands in a
*different* directory, so `--resume` only continues a byte-identical
(modulo excluded fields) setup. That's the intended effect of an
atomic/reproducible hash — flagging it here since it reverses a documented
design decision, not just refactors code.

With the hash now load-bearing in the path itself,
`check_and_record_run_meta`'s drift warning (`orchestration/runid.py:79`)
can only fire on a **truncated-hash collision** (two different full
hashes sharing the same first 7 hex chars) — keep the function, just
update its docstring; it's no longer catching "same name+seed, config
drifted" (structurally impossible now) but is still worth keeping as a
defensive check on the truncation.

## 3b. Fold splits shared across all 3 seeds

`datasets/datamodule.py::KFoldDataModule` currently derives the
train/val partition itself from the seed —
`KFold(n_splits=..., shuffle=True, random_state=self._seed)`
(`datasets/datamodule.py:367`) — and caches it at a per-seed path
(`datasets/datamodule.py:336`). So today, 3 seeds under CV don't just get
3 different weight-init/training trajectories, they also silently get 3
*different data partitions* — seed variance and fold-partition variance
are entangled, which confounds what a seed-averaged metric is supposed to
isolate.

Fix: derive the `KFold` `random_state` from `config_hash` instead of from
`training.seed`, and cache to the new hash-scoped (not seed-scoped) path
from §3's `experiment_paths()["fold_splits"]`. All 3 seeds of the same
config hash then resolve to the same cache file, compute (or find already
computed) the identical partition, and only diverge on model
init/training randomness from there.

```python
# datasets/datamodule.py — KFoldDataModule
def __init__(self, config: dict):
    super().__init__(config)
    ...
    self._config_hash = config_hash(config)   # new import: orchestration.runid.config_hash
    exp_name   = config.get("logging", {}).get("experiment_name", "experiment")
    output_dir = config.get("output_dir", "outputs/experiments")
    self._fold_file = experiment_paths(
        output_dir, exp_name, self._config_hash, self._seed
    )["fold_splits"]

def _load_or_create_fold_splits(self) -> list:
    n_splits = self.kf_cfg.get("n_splits", 5)

    if os.path.exists(self._fold_file):
        with open(self._fold_file) as f:
            cached = json.load(f)
        # keyed by config_hash now, not seed — every seed of this config
        # must hit this branch and reuse the same partition.
        if cached.get("n_splits") == n_splits and cached.get("config_hash") == self._config_hash:
            return cached["folds"]
        logger.warning(...)  # unchanged wording, s/seed/config_hash/

    all_pairs = np.array(self.handler.get_kfold_pairs())
    ...
    fold_seed = int(self._config_hash[:8], 16) % (2**31 - 1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    folds = [...]  # unchanged

    os.makedirs(os.path.dirname(self._fold_file), exist_ok=True)
    with open(self._fold_file, "w") as f:
        json.dump({"n_splits": n_splits, "config_hash": self._config_hash, "folds": folds}, f)
    return folds
```

A concurrent-write race is possible when `run_sweep` (or 3 manual
`train.py` launches) starts all 3 seeds close together and none of them
sees the cache file yet — each would independently compute and write it.
Since the computation is deterministic (`fold_seed` derived purely from
`config_hash`, not from wall-clock or process state), all 3 computations
produce byte-identical `folds` content regardless of write order — a
last-writer-wins race is harmless here, just a few redundant recomputes,
not a correctness bug. Not fixing with a lock unless it turns out to
matter in practice.

`KFoldDataModule`'s class docstring (`datasets/datamodule.py:294-317`)
needs its "Fold assignments are serialised to
`outputs/experiments/<experiment_name>-s<seed>/fold_splits.json`" line
rewritten to describe the new hash-scoped, seed-shared path and the
reason (§6 already flagged this file for a docstring update; this is the
same edit, just with the actual content specified now).

## 4. `train.py`: default multi-seed sweep

New CLI args (`parse_args`, `train.py:376`):

```python
parser.add_argument("--seed",  type=int, default=None,
                     help="Run exactly this one seed (bypasses the default 3-seed sweep).")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="Run exactly these seeds via run_sweep (default: DEFAULT_SEEDS).")
parser.add_argument("--force", action="store_true",
                     help="Re-run seeds whose manifest already says done (passed to run_sweep).")
```

`DEFAULT_SEEDS = [7, 42, 1337]` as a module-level constant.

In `main()` (`train.py:394`), after the existing CLI-override block, branch
on `args.seed`:

```python
if args.seed is not None:
    # Unchanged single-run path — exact current behavior, config's own
    # seed replaced by --seed. Still respects --fold / manual K-Fold loop.
    config["training"]["seed"] = args.seed
    kfold_cfg = config.get("k_fold", {})
    if kfold_cfg.get("enabled", False) and args.fold is None:
        ...  # existing K-Fold loop, unchanged
    else:
        run_training(config, fold=args.fold)
else:
    from orchestration.runner import run_sweep
    seeds = args.seeds or DEFAULT_SEEDS
    kfold_cfg = config.get("k_fold", {})
    if kfold_cfg.get("enabled", False):
        n_splits = kfold_cfg.get("n_splits", 5)
        folds = kfold_cfg.get("run_folds") or list(range(n_splits))
    else:
        folds = (None,)
    results = run_sweep(config, seeds=seeds, folds=folds, force=args.force)
    for r in results:
        print(f"run_id={r['run_id']} status={r['status']} "
              f"best_metric={r['best_metric']} error={r['error']}")
```

This reuses `run_sweep` exactly as `scripts/reproduce.sh`/`search.py`
already do — no new sweep engine, just wiring it up as `train.py`'s own
default path instead of something only orchestration callers reach.

## 5. `eval.py`: default multi-seed loop + combined report

Refactor the body of current `main()` (`eval.py:137`–`405`, everything
from `training_cfg = config['training']` down through
`reporter.save(...)`) into a function that runs one seed end-to-end:

```python
def evaluate_one(config: dict, args, seed: int) -> str:
    """Runs the full existing eval.py body for one seed. Returns the path
    to the report JSON it wrote (report.json or ensemble_report.json)."""
    config = copy.deepcopy(config)
    config["training"]["seed"] = seed
    ...  # exactly today's main() body
    reporter.save(report_dir=exp_paths["eval"], filename_prefix="")
    ensemble_tag = "ensemble_" if args.ensemble else ""
    return os.path.join(exp_paths["eval"], f"{ensemble_tag}report.json")
```

New CLI arg alongside the existing `--seed` (`eval.py:160`):

```python
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="Evaluate exactly these seeds, then combine (default: DEFAULT_SEEDS).")
```

`main()` becomes:

```python
config = load_config(args.config)
...  # existing --dataset_dir / --experiment-name overrides, unchanged

if args.seed is not None:
    evaluate_one(config, args, args.seed)   # current single-seed behavior, no combined report
else:
    seeds = args.seeds or DEFAULT_SEEDS
    report_paths = [evaluate_one(config, args, s) for s in seeds]
    aggregate_seed_reports(config, report_paths, seeds)
```

Each `evaluate_one` call needs its own test-eval token (the ledger's
`Test_Evals` row is per-run provenance) — mint with a seed-suffixed
`run_id` (`f"manual-eval-{experiment_name}-s{seed}"`) instead of today's
bare `f"manual-eval-{experiment_name}"` (`eval.py:212`), since that string
is otherwise identical across all 3 seeds in one invocation.

### Combined report

New function (`utils/report.py` or a new `orchestration/aggregate.py` —
open question, see below), given the 3 seeds' `report.json` paths:

```python
def aggregate_seed_reports(config: dict, report_paths: List[str], seeds: List[int]) -> str:
    reports = [json.load(open(p)) for p in report_paths]

    agg_metrics = {
        k: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for k in reports[0]["metrics"]
        for vals in [[r["metrics"][k] for r in reports]]
    }

    agg_per_class = {}
    for k in reports[0].get("per_class_metrics", {}):
        arr = np.array([r["per_class_metrics"][k] for r in reports])  # (n_seeds, n_classes)
        agg_per_class[k] = {"mean": arr.mean(axis=0).tolist(), "std": arr.std(axis=0).tolist()}

    out = {
        "experiment_name": config["logging"]["experiment_name"],
        "config_hash": config_hash(config),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "metrics": agg_metrics,
        "per_class_metrics": agg_per_class,
        "per_seed_reports": report_paths,
    }
    out_path = experiment_paths(
        config.get("output_dir", "outputs/experiments"),
        config["logging"]["experiment_name"], config_hash(config), seeds[0],
    )["combined_report"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out_path
```

Scope, per your answers above: averages only `metrics` (flat scalar dict —
dice/miou/hd95/asd/precision/recall/specificity/f2/accuracy) and
`per_class_metrics` (elementwise mean/std over each metric's per-class
list — see `metrics/aggregate.py:175-182` for that list's shape).
`model`/`efficiency`/`environment`/`config` blocks are **not** averaged —
`model.*` (params/flops) is constant across seeds of the same config
anyway, and `efficiency.*`/`environment.*` describe the measuring
machine/run, not the experiment's result.

**Known rough edge, flagging rather than silently deciding:** the
combined-report filename is exactly `<experiment_name>.json` with no hash
in it (per your literal spec). If `experiment_name` is later re-run under
a *different* config (different hash), its old `<hash>-s<seed>/`
directories stay on disk, but the new sweep's combined report overwrites
the old one — only the most recent hash's aggregate survives at that path
even though old seed directories are still there. Tell me if you want the
combined report keyed by hash too (e.g. `<experiment_name>/<hash7>.json`)
instead of bare `<experiment_name>.json` — that would need a small
deviation from your literal filename spec, so I'm not doing it unasked.

## 6. Docs/scripts to update (per your "update everything" answer)

- **`OUTPUT_LAYOUT.md`**: retarget the tree diagram, the old→new mapping
  table, and the `experiment_id` section to the new
  `<experiment_name>/<hash7>-s<seed>/` shape; rewrite the "deliberately
  excludes any config hash" paragraph to describe the new
  hash-in-path/`--resume`-is-now-hash-scoped behavior (§3 above); move
  `fold_splits.json` out of the per-seed tree entry and add it as a
  sibling `<hash7>-fold_splits.json` file next to the `<hash7>-s<seed>/`
  dirs and `<experiment_name>.json` (§3b).
- **`README.md:36`**: update the one-line output-path description.
- **`CODE_REVIEW.md`**: lines referencing `outputs/experiments/<experiment_name>-s<seed>/` (the `manifest.py` row, the `reproduce.sh` row).
- **`scripts/reproduce.sh`**: lines 65-169 build paths/globs off the old
  `${REPRODUCE_TAG}-s<seed>` convention directly (`--reports-glob
  "outputs/experiments/${REPRODUCE_TAG}-s*/eval/report.json"` etc.) — these
  need the experiment_name/hash split, which means either computing the
  hash in bash (awkward) or having reproduce.sh call a small Python helper
  to resolve the path. Needs its own look once the Python side lands —
  flagging as non-trivial rather than sketching bash here.
- **`datasets/datamodule.py`**: docstring at line ~298 + `__init__`/
  `_load_or_create_fold_splits` per §3b (hash-scoped shared fold splits,
  hash-derived `KFold` `random_state`).
- **`utils/logger.py:14`**: example path in its docstring.
- **`tests/test_orchestration.py`**: `experiment_paths` call sites (if any
  test calls it directly) need the new `config_hash_` positional arg;
  add the new config_hash-exclusion test from §2.

## 7. Non-goals / explicitly out of scope here

- Anything on the XDash side (session grouping, `dissert.yaml` script
  paths) — not touched, not this repo's concern per your framing.
- Migrating existing `outputs/experiments/` runs — none exist, moot.
- `scripts/run_seed_sweep.py`/`scripts/eval_seed_sweep.py` as separate
  wrapper scripts — superseded by making the sweep the default behavior
  of `train.py`/`eval.py` directly.
- Averaging training-side metrics (manifest `best_metric`, GPU-hours)
  across seeds — only the eval-side combined report was asked for.

## Open items I need a decision on before implementing

1. Where `aggregate_seed_reports` lives — `utils/report.py` (next to
   `EvaluationReporter`) or a new `orchestration/aggregate.py`. No strong
   pull either way from existing structure.
2. The combined-report-filename-vs-hash rough edge in §5 — keep the literal
   `<experiment_name>.json` (overwritten on every re-sweep under a new
   hash) or key it by hash too.
3. `scripts/reproduce.sh`'s bash-side path construction (§6) — worth a
   dedicated pass once the Python changes land, or fold into this same
   change?
