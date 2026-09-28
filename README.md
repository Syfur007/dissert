# dissert

**A config-driven PyTorch framework for medical image segmentation research**

> One YAML fully determines a run. Baseline and proposed architectures are trained, evaluated, and
> statistically validated under one shared pipeline — training, evaluation, statistics,
> attribution, robustness, uncertainty, profiling, and reporting are each a first-class module, not
> a notebook. [`docs/reference.md`](docs/reference.md) is the detailed, file-by-file contract; if
> code and that document disagree, one of them is a bug.

## 1 · QUICKSTART

| Step | Command |
| --- | --- |
| Install (CPU / no GPU) | `conda activate thesis && pip install --prefer-binary -e .[dev]` |
| Install (GPU training box) | `conda activate thesis && pip install --prefer-binary -e .[dev,gpu]` — pulls the pinned cu117 torch/torchvision build |
| Test | `pytest -v` — 359 tests |
| Train | `dissert-train --config configs/experiment/gmkunet/gmkunet_t_clinicdb.yaml` — sweeps 3 seeds by default |
| Train (one seed) | `... --seed 42` — bypasses the sweep, exact single run |
| Evaluate | `dissert-eval --config <same config> --allow-test-eval` — evaluates all 3 seeds, writes a combined report |
| Reproduce | `./scripts/reproduce.sh` — a real, reduced end-to-end pass over S1–S17 |

`dissert-train`/`dissert-eval`/`dissert-search`/`dissert-report` are console-script entry points
installed by `pip install -e .` (equivalently `python -m dissert.cli.train`, etc.). Temporary root
`train.py`/`eval.py` shims also exist purely for external tooling (XDash) still invoking
`python train.py`/`python eval.py` at the repo root — see the comment at the top of each file.

CUDA build and the Mamba fused-kernel dependency notes live in `pyproject.toml`'s dependency
comments.

## 2 · LAYOUT

| Layer | Package(s) | Purpose |
| --- | --- | --- |
| Config | `configs/`, `src/dissert/config/schema.py` | `compose:`-merged YAML, schema-validated on load |
| Data | `src/dissert/datasets/` | Loaders, channel construction, augmentation, leakage guards |
| Models | `src/dissert/models/` | UNet family, EMCAD, GMK-UNet, Mamba/VSS hybrid — one registry |
| Training | `src/dissert/training/`, `src/dissert/losses/` | Trainer, optimizer, determinism, declarative losses |
| Metrics | `src/dissert/metrics/` | The one canonical Dice / IoU / HD95 / ASD / NSD / ECE source |
| Orchestration | `src/dissert/orchestration/` | Run manifest, ledger, seed/fold sweeps |
| Analysis | `src/dissert/analysis/{stats,profiling,uncertainty,robustness,mechanism}/`, `src/dissert/xai/` | Significance testing, efficiency, explainability, robustness |
| Reporting | `src/dissert/reporting/` | Manuscript tables/figures, blocking rules |

Every run's output — checkpoints, logs, tensorboard, plots, eval report — lands under
`outputs/experiments/<experiment_name>/<config_hash7>-s<seed>/`, with a seed-averaged combined
report at `outputs/experiments/<experiment_name>/<experiment_name>.json`; see
[`docs/output-layout.md`](docs/output-layout.md) for the full tree and the exact fields
`config_hash` includes/excludes.

## 3 · MODELS

| Registry name | Kind | Notes |
| --- | --- | --- |
| `unet`, `attention_unet` | Baselines | Modular UNet; attention variant adds gated skip connections |
| `mk_unet` (+ `_s` / `_t`), `emcad` | Baselines | Multi-Kernel UNet (inverted-residual, multi-kernel depthwise, CBAM-style attention); EMCAD is PVTv2-backbone + multi-scale depthwise decoder |
| `gmk_unet` | Proposed | Geometry/colour channel groups (m1–m5), grouped attention gate, optional dual-skip |
| `mamba_unet` | Proposed | Dual-branch: MK-UNet encoder + 4-stage VSS/selective-scan auxiliary encoder, fused per stage (`add`/`concat`/`xattn`/`cbffm`) into a shared decoder |

`t`/`s`/`base`/`m`/`l` width presets exist for `mk_unet`/`gmk_unet` (`configs/model/{mkunet,gmkunet}/*.yaml`); 13 ready-to-run experiment configs ship under `configs/experiment/**` across ClinicDB/ColonDB. `dissert.models.build.build_width_matched()` searches width presets for the closest parameter-count match to a target model — the capacity-matched control every proposed-vs-baseline comparison needs, backed by `dissert.models.registry.ModelRegistry.get()`'s `budget_ceiling`/`allow_over_budget` guard against silently comparing models of different size. Mamba's fused CUDA scan (`mamba_ssm`) falls back automatically to a pure-PyTorch reference scan (`src/dissert/models/auxiliary/ss2d_ref.py`) when the extension isn't installed — recorded per run as `scan_impl` in the manifest.

## 4 · CHANNEL MODES

| Mode | Groups | Channels |
| --- | --- | --- |
| m1 | RGB | 3 |
| m2 | RGB + XY | 5 |
| m3 | RGB + YCbCr | 6 |
| m4 | RGB + XY + Rθ | 8 |
| m5 | RGB + XY + YCbCr + Rθ | 11 |

θ is encoded as sin/cos to avoid the branch cut. `coordonly_channels()` (XY+Rθ, no RGB) is the
shortcut-audit control used by `robustness.geometric.shortcut_audit`.

## 5 · DATA PIPELINE (`src/dissert/datasets/`)

- **Dataset handlers**: `clinicdb`, `colondb`, `busi`, `isic18` (registered, each with `get_dataset(split)`/`get_kfold_pairs()`), plus a generic flat-directory handler (`_GenericHandler`, supports `external: true` for a held-out-only dataset with no train loader).
- **Loading**: filename-based image/mask pairing with suffix tolerance, optional integrity `validate=True`, optional in-RAM `cache=True` (size-guarded), masks binarized at pixel value 127, stride-32 image-size snapping.
- **Splits**: `StandardSplitDataModule` (train/val/test ratios or published splits) and `KFoldDataModule` (sklearn `KFold`, off by default — see §7). `assert_no_subject_overlap()` raises `LeakageError`; `duplicate_cross_check()` catches near-duplicate leakage via perceptual hashing.
- **Test-set guard**: `get_test_loader(token)` on every DataModule raises `TestLoaderGuardError` without a token minted by `dissert.orchestration.ledger.LedgerWriter.issue_test_token()` — the test set can only be touched on purpose, and every touch is recorded.
- **Preprocessing**: `preprocess.build_manifest()` (path/subject/split/mask-empty/resolution manifest), `preprocess.dedup()` (perceptual hash + SSIM near-duplicate exclusion — mandatory for BUSI).
- **Augmentation**: `AugmentationPolicy` — one Albumentations pipeline per `(modality, dataset)`, geometric ops applied to image+mask together, then geometry channels (XY/Rθ) regenerated from the augmented frame so they never desync from the image. Augmentation intensity is modality-conditioned (colour / grayscale-ultrasound / grayscale-microscopy get different op tables).
- **Stats helper**: `python -m dissert.datasets.stats` computes per-channel mean/std and class frequency for pasting into a new dataset config.

## 6 · TRAINING (`src/dissert/training/`, `src/dissert/losses/`)

- **`Trainer.fit()`**: plain epoch loop or multi-stage (per-stage LR/freeze schedule, `training.stages` in config); AMP (`GradScaler`), gradient accumulation, gradient clipping (value or norm mode), multi-scale training, EMA shadow weights (validated under `ema.average_parameters()`, restored after).
- **Checkpointing**: `CheckpointManager` writes `best.pth`/`last.pth` atomically (temp-file + `os.replace`), embeds `config_hash`/`run_id`/git commit into every checkpoint; `PeriodicCheckpointCallback` adds `epoch_NNNN.pth` snapshots on a configurable interval. Resume restores model/optimizer/scheduler/scaler/EMA/RNG/EarlyStopper state.
- **Callbacks**: `TensorBoardCallback`, `PredictionOverlayCallback` (EMA-aware side-by-side prediction grids), `TrainingCurvePlotCallback` (offline PNGs rendered from the TensorBoard event file at train-end).
- **Determinism**: `seed_everything()` seeds python/numpy/torch/cuda + DataLoader workers and turns on `torch.use_deterministic_algorithms(warn_only=True)`; any operation that actually trips the guard is recorded (`get_recorded_nondeterminism()`) into the run manifest instead of silently passing or hard-crashing.
- **Optimizers/schedulers**: AdamW/Adam/SGD with parameter groups (biases, norm layers, and Mamba's `A_log`/`D` get zero weight decay); cosine/step/plateau/onecycle schedules.
- **Losses** (`losses.get_loss`): `bce`, `ce`, `dice`, `tversky`, `focal`, `structure` (boundary-weighted, PraNet/Polyp-PVT lineage), `combo`, and a declarative `compound` loss (`term_list=[(name, weight, schedule), ...]` with linear-ramp scheduling) — a schema-level redundancy guard rejects stacking overlapping loss families (e.g. `dice`+`tversky`) without an explicit override reason.

## 7 · MULTI-SEED WORKFLOW

`dissert-train`/`dissert-eval` sweep 3 seeds (`[7, 42, 1337]`) by default — a study needs one
invocation per config, not three:

| Flag | Effect |
| --- | --- |
| *(none)* | Sweep the default 3 seeds via `dissert.orchestration.runner.run_sweep` |
| `--seeds 7 42 1337 2024` | Sweep exactly these seeds instead |
| `--seed 42` | Run exactly one seed, bypassing the sweep entirely (the old single-run behavior) |
| `--force` (train only) | Re-run seeds whose manifest already reports `status: "done"` |

Cross-validation (`k_fold.enabled`) is **off by default**; when turned on, every swept seed
trains/validates on the *same* fold partition (the partition is derived from `config_hash`, not
from the seed, so seed-variance and fold-variance never get entangled) — only weight-init/training
stochasticity differs across seeds.

`dissert-eval`'s default multi-seed run writes one `eval/report.json` per seed plus a combined,
seed-averaged report (`dissert.evaluation.report.aggregate_seed_reports`: mean±std of every scalar
metric and per-class metric) at `outputs/experiments/<experiment_name>/<experiment_name>.json`.

## 8 · ORCHESTRATION & SWEEPS (`src/dissert/orchestration/`)

- **Config schema** (`src/dissert/config/schema.py`): Pydantic-validated `Config` — unknown keys or wrong types raise at load time. `ModelConfig` is the one permissive section (`extra="allow"`, forwarded as `**kwargs` to `get_model()`).
- **Run identity** (`runid.py`): `config_hash()` — SHA1 over the resolved config, excluding pure bookkeeping (seed, the whole `logging` section, `output_dir`, resume/save-cadence knobs, `stats`) so two configs that only differ in presentation hash identically; `run_id = R-{hash[:7]}-s{seed}-f{fold}` addresses one exact run everywhere (manifest, checkpoint metadata, ledger row).
- **Manifest** (`manifest.py`): per-run provenance — resolved config, config hash, git commit + dirty-tree flag, environment hash, hardware, timing, GPU-hours, wall-seconds, epochs completed/target, resumability, any recorded non-determinism. Status is `pending` → `running` → `done`/`interrupted`/`failed`; `interrupted` means the run stopped itself on a wall-clock budget (below) — healthy and resumable, as opposed to a manifest stranded at `running` by a hard kill.
- **Ledger** (`ledger.py`): append-only CSV tables — `Runs`, `Compute`, `Test_Evals` (every test-set touch, token-gated), `Stats` (significance-test results).
- **`run_sweep()`** (`runner.py`): expands seed×fold, idempotent-skips a combination whose manifest already says `"done"` unless `force=True`, wraps each in a manifest + ledger row. This is what `dissert-train`'s default multi-seed path (§7) and `scripts/reproduce.sh` both drive.
- **Wall-clock budgets** (`budget.py`): `dissert-train --max-hours FLOAT` bounds a whole session on compute with a hard walltime limit. Training stops itself at a clean *epoch boundary* before the budget would be overrun and exits 0, leaving a resumable `last.pth` and a `status: "interrupted"` manifest; `run_sweep` additionally declines to start a run whose projected duration no longer fits. Re-running the same command continues where it stopped (`interrupted` runs are retried, `done` ones skipped). Omit the flag and every check is inert. Deliberately a CLI argument and never a config field, so it stays out of `config_hash` and sessions under different budgets share one directory.
- **Run status API** (`status.py`): `describe_run()` / `describe_experiment()` — read-only inspection of an experiment's state (`status`, `resumable`, `epochs_completed`/`total_epochs`, `has_fold_splits`) plus `checkpoint_files`, the minimal repo-relative set that must travel with a run for a resume to be correct. Imports no torch at module scope.
- **Budgeted sweep** (`sweep.py`): `run_budgeted_sweep()` — trials run in seeded-shuffled order until measured wall-clock cost exceeds a GPU-hour budget; CLI via `python -m dissert.orchestration.sweep`.
- **Legacy grid/random search** (`src/dissert/cli/search.py`): expands `configs/search_config.yaml`'s `grid:` into a Cartesian product (or seeded random subset), one `dissert.cli.train.run_training()` call per trial (K-fold disabled), writes `search_summary.csv`/`search_report.md`/`best_config.yaml` under `outputs/searches/`.

## 9 · EVALUATION & METRICS

- **Canonical metrics** (`src/dissert/metrics/`, the only implementation training/eval/attribution/robustness ever import): `dice`/`iou` (region), `hd95`/`asd`/`nsd` (boundary — undefined-when-empty cases excluded and counted, never penalized with a fixed constant), `precision`/`recall`/`specificity`/`f2`/`accuracy` (detection), ECE (calibration). `compute_dataset_metrics()` returns macro averages, 5th/25th-percentile Dice, and a per-class breakdown.
- **`dissert-eval`**: loads a single checkpoint or all K-fold checkpoints as an ensemble (prefers EMA shadow weights when present); profiles FLOPs/params/latency/throughput; runs the guarded test loader; saves confusion-matrix/ROC/PR plots; writes a Markdown+JSON report per seed plus the combined report (§7). `--experiment-name` scopes a run without touching a real experiment's own directory; `--ensemble` evaluates every fold's checkpoint together.

## 10 · ANALYSIS SUITE

| Area | Package | Capabilities |
| --- | --- | --- |
| Statistics | `src/dissert/analysis/stats/` | Wilcoxon paired test, bootstrap CI, a meaningfulness gate, Cliff's delta / paired median diff (effect size), Holm–Bonferroni correction, Friedman test + Nemenyi post-hoc (critical-difference data) — tied together by `run_family_comparison()` for one declared model-comparison family |
| Profiling | `src/dissert/analysis/profiling/` | Analytic + fvcore FLOPs with an agreement check, latency (batch 1/16, warm-up + timed runs), peak GPU memory, ONNX/TorchScript/TensorRT export (timeout-guarded) |
| Attribution | `src/dissert/xai/` | Channel-group occlusion, exact Shapley values over channel groups, integrated gradients (captum), Mamba auxiliary-branch ablation, CBFFM fusion-gate probing, Seg-Grad-CAM / Seg-XRes-CAM, parameter/label randomization sanity checks |
| Uncertainty | `src/dissert/analysis/uncertainty/` | Deep ensemble over existing seeds (zero extra training cost) — predictive entropy, inter-seed variance, error-detection AUROC, retention curves |
| Robustness | `src/dissert/analysis/robustness/` | 8 photometric/acquisition corruptions × 5 severities (noise, blur, JPEG, brightness/contrast, gamma, resolution/resampling); geometric perturbations (translate/rotate/scale/off-centre-crop) with a shared-grid primitive that leaves geometry channels untouched; shortcut audit (coord-only-model Dice vs. threshold); frame-jitter sensitivity |
| Mechanism analysis | `src/dissert/analysis/mechanism/` | Effective Receptive Field (gradient-based), linear CKA between representations, a 6-category per-image failure taxonomy (success / missed-lesion / false-positive / under-/over-segmentation / boundary-only) |

## 11 · REPORTING (`src/dissert/reporting/`)

Reads only already-computed artefacts (JSON/Parquet/ledger CSV) — never a checkpoint, never a
recomputed metric. Four blocking rules are hard `BlockingRuleError` raises, not warnings:

- **No dirty-tree runs** — refuses to render a table if any contributing run's manifest has an uncommitted git tree.
- **Minimum seeds** — refuses an under-seeded comparison.
- **Stats entries present** — refuses an unstated/unsupported comparison claim.
- **Saliency sanitized** — refuses unsanitised attribution output in a figure.

`render_main_comparison_table()`/`render_efficiency_table()` produce CSV+LaTeX manuscript tables
with a provenance footer (snapshot ID, git commit, generation date); `figures.py` renders
degradation curves, Pareto frontiers (real non-dominated-point detection), and critical-difference
diagrams; `inventory.py` audits which of the manuscript's declared artefacts actually exist on disk.
`dissert-report` (`src/dissert/cli/report.py`) is the CLI entry point.

## 12 · GUARANTEES

| Guarantee | Enforced by |
| --- | --- |
| Test set touched once, on purpose | `dissert.datasets.datamodule.get_test_loader(token)` raises without a minted token |
| Every run is addressable | `run_id = R-{config_hash[:7]}-s{seed}-f{fold}`, in every manifest/checkpoint/artefact |
| No config-drift between models | Shared `AugmentationPolicy`; no per-model augmentation key in the schema |
| Reported tables are trustworthy | `dissert.reporting` refuses a dirty-tree run, an under-seeded config, an unstated comparison, or unsanitised saliency |

## 13 · TESTING

`pytest -v` — 359 tests, one file per implementation area: `test_orchestration.py`,
`test_metrics.py`, `test_data_contract.py`, `test_channels.py`, `test_optim.py`, `test_models.py`,
`test_mamba.py`, `test_losses.py`, `test_stats.py`, `test_profiling.py`, `test_attribution.py`,
`test_uncertainty.py`, `test_robustness.py`, `test_analysis.py`, `test_reporting.py`,
`test_sweep.py`, `test_ci_audit.py` (checks the test suite's own completeness rather than framework
behavior directly). CI (`.github/workflows/ci.yml`) runs the full suite on CPU wheels on every
push/PR, excluding `mamba-ssm`/`causal-conv1d` (no GPU on the runner — the suite only needs the
pure-PyTorch scan fallback).

## 14 · DOCUMENTS

| Document | Contents |
| --- | --- |
| [`docs/reference.md`](docs/reference.md) | Deep implementation reference — file-by-file, function-by-function |
| [`docs/output-layout.md`](docs/output-layout.md) | The exact `outputs/` directory tree, `experiment_id`/`config_hash` semantics |
| [`docs/design/session-grouping.md`](docs/design/session-grouping.md) | Design plan for session/repeat grouping in the output layout |
| [`docs/design/xdash-resume-contract.md`](docs/design/xdash-resume-contract.md) | The resume contract shared with the external XDash orchestrator |
| [`docs/design/transfer-xai-plan.md`](docs/design/transfer-xai-plan.md) | The repo-reorg / transfer-learning / XAI roadmap (this reorg is its Phase 0) |
| [`CHANGELOG.md`](CHANGELOG.md) | Build history, phase by phase, including real bugs found and fixed along the way |
