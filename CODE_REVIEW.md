# Technical Reference — `dissert`

Detailed, file-by-file map of the implementation: what each package/module does, its public
surface, and how the pieces connect. [`Technical_Framework_Spec.md`](Technical_Framework_Spec.md)
is the contract; this document is how the current code satisfies it. [`README.md`](README.md) is
the short quickstart. [`CHANGELOG.md`](CHANGELOG.md) has the phase-by-phase build history,
including real bugs found and fixed along the way.

---

## 1 · Entry-point scripts

| File | Role |
| --- | --- |
| `train.py` | Loads/validates a config, seeds RNGs (`training.determinism.seed_everything`), builds every component (DataModule, model, loss, optimizer/scheduler, `CheckpointManager`, `EarlyStopping`, EMA, TensorBoard tracker, callbacks), computes FLOPs/params via `profiling.flops.check_flops_agreement`, handles resume (restores model/optimizer/scheduler/scaler/EMA/RNG/EarlyStopper state), and drives the K-fold loop (per-fold try/except, mean±std summary). |
| `eval.py` | Loads a single checkpoint or all K-fold checkpoints as an `EnsembleModel`; prefers EMA shadow weights when a checkpoint carries them; profiles FLOPs/params/latency/throughput (`profiling.flops`/`profiling.latency`); runs the test set through the guarded loader (`--test-token`/`--allow-test-eval`); computes the canonical metric set (`metrics.aggregate.compute_dataset_metrics`); saves confusion-matrix/ROC/PR plots; writes a Markdown+JSON report (`utils.report.EvaluationReporter`). `--experiment-name` overrides `logging.experiment_name` for one run without touching a real experiment's own log/checkpoint directory. |
| `search.py` | Legacy grid/random hyperparameter search: expands `configs/search_config.yaml`'s `grid:` block into a Cartesian product (or a seeded random subset), runs `train.run_training()` per trial with K-fold disabled, writes `search_summary.csv`/`search_report.md`/`best_config.yaml`. Superseded by `orchestration/sweep.py` for budget-bounded sweeps but still functional. |
| `scripts/generate_report.py` | Globs `eval.py`'s JSON report dumps, assembles `results.parquet`/`profiling.json`-shaped rows, and calls `reporting.tables.render_main_comparison_table`/`render_efficiency_table`. Only reads JSON — never a checkpoint, never recomputes a metric. |
| `scripts/reproduce.sh` | Runs a reduced S1–S17 pass for real where a CLI exists (data/channel assertions, a multi-seed training sweep via `orchestration.runner.run_sweep`, guarded per-seed evaluation, a rendered comparison table) and prints the exact module/function to call for stages that are Python-API-only. Every run is scoped to `<experiment>_reproduce_<snapshot>` so it can never collide with a real experiment's logs/checkpoints — seed no longer needs manually folding into the name, since `outputs/experiments/<experiment_name>-s<seed>/` already isolates it (see `OUTPUT_LAYOUT.md`). Sets `PYTHONPATH` explicitly since some non-interactive shells don't add the repo root to `sys.path` on their own. |

---

## 2 · Configuration system

`utils/config.py`'s `load_config()` resolves a `compose:` list (recursive deep-merge, circular-include
detection, later fragments override earlier ones) and then validates the merged dict against
`orchestration/schema.py`'s Pydantic models (`validate=False` opts out, used by non-experiment
configs like `search_config.yaml`). Unknown keys or wrong-typed values raise at load time.

```
configs/
├── base.yaml                # every section documented inline; the sole source of defaults
├── dataset/{clinicdb,colondb,busi,isic18}.yaml
├── model/{mkunet,gmkunet,mamba}/{t,s,base,m,l}.yaml
├── training/emcad.yaml
├── experiment/**/*.yaml     # compose: [base, dataset, model] + a small override block
├── search_config.yaml       # search: {method, num_trials, seed, budget_gpu_hours}, grid: {...}
└── search_test_config.yaml
```

`dataset.name`/`dataset.root` have no default — an experiment config must compose a
`configs/dataset/<name>.yaml` fragment or set both explicitly. `ModelConfig` is the one
permissive schema section (`extra="allow"`): `model:` blocks forward verbatim as `**kwargs` to
`models.registry.get_model()`, since each model family has its own constructor signature.

---

## 3 · Data layer (`datasets/`)

| File | Role |
| --- | --- |
| `dataset.py` | `MedicalSegmentationDataset` — `__getitem__` returns `(image, mask, meta)`; `meta` carries `subject_id`, `source_dataset`, `spacing`, `artefact_flags` (`METADATA_KEYS`). Filename-based pairing with suffix tolerance, optional `validate=True` integrity check, optional in-RAM `cache=True` (size-guarded). Masks binarized at pixel value 127. |
| `datamodule.py` | `BaseDataModule` (stride-32 image-size snapping, seeded `DataLoader` factory, `persistent_workers=True`), `StandardSplitDataModule`, `KFoldDataModule` (sklearn `KFold`, fold cache invalidated on `n_splits`/seed mismatch). `get_test_loader(token)` on both raises `TestLoaderGuardError` without a token minted by `orchestration.ledger.LedgerWriter.issue_test_token()`. `DATASETS` registry: `clinicdb`, `colondb`, `busi`, `isic18`. `_GenericHandler` (unregistered flat-directory datasets) supports `external: bool` (no train loader when set). |
| `polyp/clinicdb.py`, `polyp/colondb.py` | `get_dataset(split)`/`get_kfold_pairs()` handlers for CVC-ClinicDB/ColonDB; `ARTEFACT_FLAGS` including `frame_level_only_no_video_grouping: true` (filenames are flat sequential integers, no recoverable source-video id for `GroupKFold`). |
| `busi.py`, `isic18.py` | Same handler interface; BUSI runs mandatory `preprocess.dedup()`, ISIC18 uses `splits.load_published_split("isic18")`. |
| `channels.py` | `build_channels(image, mode, order)` / `build_channels_from_groups()` for channel modes m1–m5 (see §4); `xy_channels()`, `r_theta_channels()` (θ as sin/cos to avoid the branch cut), `randproj_channels()`, `coordonly_channels()` controls; `modality_effective_channels()` (drops YCbCr for grayscale); `ycbcr_from_rgb_tensor()`/`ycbcr_from_rgb()` (the differentiable RGB→YCbCr conversion GMK-UNet's fusion branch consumes). |
| `augment.py` | `AugmentationPolicy` — built once per `(modality, ds_cfg)`, wraps `transforms.py`'s pipeline, enforces `ORDER=post` (geometric augmentation on image+mask, then `build_channels()` regenerates geometry channels from the augmented frame) and modality-conditioned op tables (colour / grayscale-ultrasound / grayscale-microscopy). |
| `splits.py` | `assert_no_subject_overlap()` (raises `LeakageError`), `load_published_split()`, `duplicate_cross_check()` (hash-based), `TestLoaderGuardError`. |
| `preprocess.py` | `build_manifest()` (path, subject_id, split, `mask_empty`, resolution), `dedup()` (perceptual hash + SSIM, exclusion list). |
| `transforms.py` | `build_transforms(img_height, img_width, ds_cfg)` — Albumentations pipelines; images bilinear, masks nearest, always. |
| `stats.py` | CLI (`python -m datasets.stats`) computing per-channel mean/std (Welford) and class frequency (binarized for binary tasks) for pasting into a dataset config. |

### Channel modes

| Mode | Groups | Channels |
| --- | --- | --- |
| m1 | RGB | 3 |
| m2 | RGB + XY | 5 |
| m3 | RGB + YCbCr | 6 |
| m4 | RGB + XY + Rθ | 8 |
| m5 | RGB + XY + YCbCr + Rθ | 11 |

`dataset.channel_mode`/`channel_order`/`modality` (`colour`/`grayscale`) drive `AugmentationPolicy`.
`coordonly_channels()` (XY+Rθ, no RGB) is the shortcut-audit control (`robustness.geometric.shortcut_audit`).

---

## 4 · Models (`models/`)

Decorator-based `MODEL_REGISTRY` (`registry.py`); `get_model(name=..., **kwargs)` is the single
construction entry point everywhere. `ModelRegistry.get()` takes `budget_ceiling`/`allow_over_budget`
— profiles the built model and raises `ModelBudgetExceededError` unless the override is explicit.

| Registry name(s) | File | Architecture |
| --- | --- | --- |
| `unet` | `baseline/unet.py` | Modular UNet (`DoubleConv`/`EncoderBlock`/`DecoderBlock` from `blocks.py`), configurable feature widths, bilinear or transposed-conv upsampling. |
| `attention_unet` | `baseline/attention_unet.py` | UNet + additive attention gates on skip connections. |
| `mk_unet`, `mk_unet_s`, `mk_unet_t` | `baseline/mk_unet.py` | Multi-Kernel UNet — inverted-residual blocks with parallel multi-kernel (1/3/5) depthwise convs, channel shuffle, grouped-conv `GroupedAttentionGate`, CBAM-style channel+spatial attention per decoder stage. T/S/base/M/L width presets via `configs/model/mkunet/*.yaml`'s `channels` list; `mk_unet_s`/`mk_unet_t` are separate registered subclasses, the rest select width via config. |
| `emcad` (`EMCADNet`) | `baseline/emcad.py` | PVTv2-backbone decoder (MSDC/MSCB multi-scale depthwise blocks); pulls backbone weights via `backbones.py`'s `get_backbone()`; `pretrain=False` builds offline without the checkpoint. Only model with no `in_channels` kwarg (backbone-driven). |
| `gmk_unet` (`GMK_UNet`) | `proposed/gmk_unet.py` | Geometry/colour-aware MK-UNet: multi-channel input (whatever `channel_mode` produces) through an MK-UNet-style encoder; `GroupedAttentionGate` (optional dual-skip via `use_rgb_skip`) gates each skip. |
| `mamba_unet` (`MambaUNet`) | `proposed/mamba_unet.py` | Dual-branch: MK-UNet primary encoder (reuses `mk_unet.py`'s blocks directly, not reimplemented) + a 4-stage VSS/selective-scan auxiliary encoder (`models/auxiliary/vss.py`), consuming the *same* multi-channel input as the primary branch (so channel-group attribution can attribute to a specific branch). Fused per stage via `models/fusion.py` (`add`/`concat`/`xattn`/`none`/`cbffm`) into a shared decoder (`models/decoder.py`, `skip_policy ∈ {primary_only, fused, both_concat}`). `scan_impl` property reports which selective-scan path actually ran (`mamba_ssm` fused kernel vs. `ss2d_ref` pure-PyTorch fallback), recorded in the run manifest. |

**Auxiliary / shared:**

| File | Role |
| --- | --- |
| `auxiliary/ss2d.py` | `SS2D` — multi-directional 2D selective scan; `scan_directions ∈ {1,2,4,8}`, direction set/traversal/merge (`sum`\|`learned`) as named constants with an ASCII-diagram docstring. Uses `mamba_ssm`'s fused CUDA kernel when importable, else falls back automatically. |
| `auxiliary/ss2d_ref.py` | `selective_scan_ref()` — pure-PyTorch reference scan, the same (A, B, C, Δ) recurrence `mamba_ssm`'s own reference implements. |
| `auxiliary/vss.py` | VSS block + PVM grouping, `DropPath`, per-stage depth — `build_vss_stage()`. |
| `build.py` | `build_width_matched(base_model_fn, candidates, input_shape, target_params, tol)` — searches width-preset candidates for the closest parameter-count match; the capacity-matched control every proposed-model comparison needs. |
| `fusion.py` | `AddFusion`, `ConcatFusion`, `NoneFusion`, `CrossAttentionFusion`, `CBFFM` (learned per-pixel spatial gate blending primary/auxiliary branches) — `build_fusion_stages()`. |
| `decoder.py` | `MambaDecoder` — single shared decoder, `skip_policy` config knob. |
| `blocks.py` | Shared primitives: `ConvBlock`, `ResBlock`, `DoubleConv`, `EncoderBlock`, `DecoderBlock`, `AttentionBlock`, `ChannelAttention`, `SpatialAttention`, `channel_shuffle`, `_init_weights` (normal/trunc-normal/xavier/kaiming/efficientnet-style fan-out). |
| `backbones.py`, `pvtv2.py`, `resnet.py` | `get_backbone(name, pretrained, pretrained_dir)` factory; PVTv2 and ResNet backbone implementations for EMCAD. |

---

## 5 · Training infrastructure

| File | Role |
| --- | --- |
| `training/trainer.py` | `Trainer.fit()` dispatches to `_fit_range()` (plain epoch loop) or `_fit_staged()` (multi-stage, per-stage LR/freeze; only rebuilds optimizer/scheduler when actually entering a fresh stage, not on a mid-stage resume). `train_one_epoch()`/`validate()` unpack the 3-tuple `(image, mask, meta)`. Supports multi-scale training, AMP (`GradScaler`), gradient accumulation, gradient clipping (value/norm), EMA (validates under shadow weights via `ema.average_parameters()`, restores raw weights after — `_persist_extra_state()` writes EMA/RNG/EarlyStopper state onto `best.pth` too, not just `last.pth`), callback hooks, early stopping. The rolling per-batch Dice/IoU is a cheap live-monitoring proxy only — never cite it as a result; the canonical value comes from `metrics.aggregate.compute_dataset_metrics` at validation/eval time. |
| `training/optimizers.py` | `build_optimizer(cfg, model)` — AdamW/Adam/SGD with parameter groups via `no_decay_group(model)` (biases, norm layers, and `A_log`/`D` SSM parameters get zero weight decay); `build_scheduler()` (cosine/step/plateau/onecycle/none), returns `(scheduler, step_mode)`. |
| `training/determinism.py` | `seed_everything(seed, deterministic=True)` — python/numpy/torch/cuda + DataLoader `worker_init_fn`, `torch.use_deterministic_algorithms(warn_only=True)`; a captured-warnings hook records any operation that actually trips the guard via `record_manifest_extra`/`get_recorded_nondeterminism()` rather than asserting reproducibility unconditionally. |
| `training/ema.py` | `EMA` — exponentially-averaged shadow parameters; `average_parameters()` context manager swaps live↔shadow weights. |
| `training/callbacks.py` | `Callback` base + `Trainer._call(hook, ...)`. Built-ins: `PeriodicCheckpointCallback`, `TensorBoardCallback`, `PredictionOverlayCallback` (EMA-aware overlay grids), `TrainingCurvePlotCallback` (offline PNGs from the TB event file). |
| `losses/` | `get_loss(name, num_classes, **kwargs)`: `bce`, `ce`, `dice`, `tversky`, `focal`, `structure` (binary-only boundary-weighted, PraNet/Polyp-PVT lineage), `combo`, `compound` (declarative `term_list=[(name, weight, schedule), ...]`, `losses/schedules.py`'s linear ramp). `losses/terms.py` holds each standalone term (incl. `boundary` — real distance-transform loss, `compute_or_load_distance_map()` cached to disk); `losses/compound.py`'s `REDUNDANT_TERM_FAMILIES` backs the schema-level redundancy guard (rejects stacking `dice`+`tversky` without an explicit `loss_override_reason`). |

---

## 6 · Metrics (`metrics/`)

The one canonical implementation — training, evaluation, attribution, and robustness all import
from here; nothing recomputes Dice independently.

| File | Role |
| --- | --- |
| `region.py` | `dice_iou()`/`dice()`/`iou()` — both-empty → 1.0, one-empty → 0.0. |
| `boundary.py` | `hd95()`/`asd()`/`nsd()` — undefined-when-either-empty is *excluded and counted* (`*_excluded_n`), not penalized with a fixed constant. |
| `detection.py` | `precision_recall_specificity_f2_accuracy()`, `fpr_on_normals()`, `specificity_on_lesion_free_subset()`. |
| `calibration.py` | ECE, equal-mass binning. |
| `aggregate.py` | `EMPTY_MASK_CONVENTION` (project constant), `compute_dataset_metrics()` (macro averages + 5th/25th percentile Dice + per-class breakdown), `write_per_image_parquet()` — the only legitimate source of downstream per-image aggregates. |

---

## 7 · Orchestration (`orchestration/`)

| File | Role |
| --- | --- |
| `schema.py` | Pydantic models (`Config`, `DatasetConfig`, `TrainingConfig`, `ModelConfig` (permissive), `CheckpointConfig`, `EarlyStoppingConfig`, `LoggingConfig`, `KFoldConfig`, `StatsConfig`, `SearchSweepConfig`); `validate_config()`. |
| `runid.py` | `config_hash(resolved_config)` (SHA1, seed/fold excluded), `run_id(hash, seed, fold)` → `R-{hash[:7]}-s{seed}-f{fold}`. |
| `manifest.py` | `RunManifest`/`build_manifest()` — resolved config, config hash, git commit + dirty-tree flag, env hash, hardware, start/end time, GPU-hours, `nondeterministic_ops`. Written to `outputs/experiments/<experiment_name>-s<seed>/checkpoints/fold<N>/manifest.json` (see `OUTPUT_LAYOUT.md`). |
| `ledger.py` | `LedgerWriter` — CSV-backed `Runs`/`Compute`/`Test_Evals`/`Stats` tables (`append_run_row`, `issue_test_token`, `has_test_token`, `append_stats_row`). |
| `runner.py` | `run_sweep(config, seeds, folds, ...)` — expands seed×fold, idempotent skip on `status: "done"` unless `force`, wraps each combination in manifest start/finish + ledger row. Zero reference to the guarded test loader (statically checked). |
| `sweep.py` | `run_budgeted_sweep(base_config, grid, budget_gpu_hours, ...)` — budget-aware successor to `search.py`; trials run in a seeded-shuffled order until measured wall-clock cost exceeds the budget. CLI: `python -m orchestration.sweep --base-config ... --search-config ...` (requires `search.budget_gpu_hours` in the search config). Zero reference to the guarded test loader (statically checked, same as `runner.py`). |

---

## 8 · Statistics (`stats/`)

| File | Role |
| --- | --- |
| `tests.py` | `wilcoxon_paired_test()`, `bootstrap_ci()`, `meaningfulness_gate()`. |
| `effectsize.py` | `cliffs_delta()`, `paired_median_diff()`. |
| `correction.py` | `holm_bonferroni()` (wraps `statsmodels.stats.multitest.multipletests`). |
| `ranking.py` | `friedman_test()`, `nemenyi_posthoc()` (critical-difference diagram data). |
| `__init__.py` | `run_family_comparison(family, proposed_name, proposed_per_image, comparators, ...)` — ties the above together for a *declared* comparison family, writes `reports/json/stats/<family>.json`, appends rows to the ledger's `Stats` table. |

---

## 9 · Profiling (`profiling/`)

| File | Role |
| --- | --- |
| `flops.py` | `analytic_flops()` (hand-written Conv2d/Linear formulas + a closed-form selective-scan count for `SS2D` submodules — fvcore's tracer has no handler for the scan and is prohibitively slow on its sequential loop, so `SS2D.forward` is stubbed during the fvcore pass), `fvcore_flops()`, `check_flops_agreement()` (raises `FlopsAgreementError` if the two disagree by more than 5%). |
| `latency.py` | `measure_latency()` — batch 1/16, ≥50 warm-up, ≥200 timed runs, explicit `cuda.synchronize`; `measure_latency_table()`. |
| `memory.py` | `measure_peak_memory()` (`torch.cuda.max_memory_allocated` after reset; `None`, not a fabricated 0.0, on CPU), `checkpoint_size_mb()`. |
| `export.py` | `export_all()` — ONNX/TorchScript/TensorRT, each attempt under a `SIGALRM` hard timeout (the Mamba model's sequential scan makes trace-based export slow enough to need one); `ok`/`fail` + error class, never a hard failure. |

---

## 10 · Attribution & explainability (`attribution/`)

Inference-only; every function runs from a checkpoint against the guarded test loader's token.

| File | Role |
| --- | --- |
| `common.py` | `resolve_group_slices()` (channel-group index boundaries from `AugmentationPolicy.effective_groups`), `compute_training_mean_image()`, `occlude_groups()`, `predict_hard()`. |
| `occlusion.py` | `run_channel_group_occlusion()` — per-group Dice drop vs. the training-set mean baseline. |
| `shapley.py` | `run_exact_shapley()` — exact Shapley over the active channel groups (2^k coalitions); `shapley_values_from_characteristic_function()` is the pure-combinatorics core. |
| `integrated_grads.py` | `run_integrated_gradients()` (captum `IntegratedGradients`, baseline = training mean image, per-channel/group attribution mass), `agreement_score()` vs. occlusion. |
| `branch.py` | `run_branch_ablation()`/`ablate_auxiliary_stage()` — zeroes a Mamba model's auxiliary-branch stage output via a forward hook, recomputes Dice. Mamba-family only. |
| `fusion_probe.py` | `run_fusion_probe()` — reads `CBFFM.gate`'s own forward output via a hook; correlates gate value with GT foreground fraction. CBFFM fusion mode only. |
| `segcam.py` | `seg_grad_cam()`/`seg_xres_cam()` — qualitative only; vanilla Grad-CAM deliberately not implemented. |
| `sanity.py` | `parameter_randomization_sanity_check()` (Adebayo et al. cascading randomisation, SSIM-scored), `label_randomization_sanity_check()`, `randomize_model_()`. |

---

## 11 · Uncertainty (`uncertainty/`)

| File | Role |
| --- | --- |
| `ensemble.py` | `predict_ensemble_members()` (per-member probability maps), `predictive_entropy()`, `inter_seed_variance()` — deep ensemble over existing seeds, zero extra training cost. |
| `retention.py` | `error_detection_auroc()`, `uncertainty_error_correlation()`, `retention_curve()` (Dice vs. fraction of most-uncertain images referred away). |

---

## 12 · Robustness (`robustness/`)

| File | Role |
| --- | --- |
| `corruptions.py` | 8 photometric/acquisition corruptions (Gaussian/speckle noise, blur, JPEG, brightness/contrast, gamma, resolution change, resampling), severities 1–5 (`SEVERITY_LEVELS`, shared). |
| `common.py` | `degradation_curve()`, `mean_corruption_error()`, `evaluate_under_corruption()`. |
| `geometric.py` | `translate`/`rotate`/`scale`/`off_centre_crop` (shared affine `grid_sample` primitive — content channels transformed, geometry channels left untouched since they're a pure function of grid shape), `geometric_degradation_curve()`, `shortcut_audit()` (coord-only-model Dice vs. a pre-registered threshold), `frame_jitter_sensitivity()`. |

---

## 13 · Mechanism analysis (`analysis/`)

| File | Role |
| --- | --- |
| `erf.py` | `compute_erf()` (gradient-based Effective Receptive Field, Luo et al. 2016), `erf_radius()` (weighted-RMS-radius summary). |
| `cka.py` | `linear_cka()` (Kornblith et al. 2019, Frobenius-norm identity), `cka_matrix()`, `flatten_spatial_features()`. |
| `failure_taxonomy.py` | `classify_failure()` — 6-category per-image classification (`success`, `missed_lesion`, `false_positive`, `under_segmentation`, `over_segmentation`, `boundary_only`); `failure_counts()`, `gallery_indices()`. |

---

## 14 · Reporting (`reporting/`)

Reads only already-computed artefacts (JSON, Parquet, ledger CSV rows) — never a checkpoint, never
a recomputed metric.

| File | Role |
| --- | --- |
| `tables.py` | Four blocking rules as hard `BlockingRuleError` raises (`check_no_dirty_tree_runs`, `check_minimum_seeds`, `check_stats_entries_present`, `check_saliency_sanitized`); `render_main_comparison_table()`, `render_efficiency_table()`; `provenance_footer()` (snapshot ID, git commit, generation date). |
| `figures.py` | `render_degradation_curve_figure()`, `render_pareto_frontier_figure()` (real non-dominated-point detection), `render_critical_difference_figure()`. |
| `inventory.py` | `ARTEFACT_INVENTORY` (the manuscript-item → source-artefact table as data), `audit_artefact_inventory()`. |

---

## 15 · Utilities (`utils/`)

| File | Role |
| --- | --- |
| `config.py` | `load_config()` — the composable YAML loader (§2). |
| `checkpoint.py` | `CheckpointManager` (`save`/`load`, `is_better()` with `min_delta` shared with `EarlyStopping`), `atomic_torch_save()` (temp-file + `os.replace`). `load()` reconciles a `module.`-prefix mismatch (DataParallel/DDP) and logs missing/unexpected keys; wraps `optimizer.load_state_dict()` in try/except for graceful degradation on a param-group-count mismatch (e.g. after an optimizer change). Embeds `config_hash`/`run_id`/git commit in every saved checkpoint. |
| `metrics.py` | `count_parameters()` only — FLOPs/latency/memory profiling moved to `profiling/`. |
| `early_stopping.py` | `EarlyStopping` — patience/`min_delta`, `state_dict()`/`load_state_dict()` for resume. |
| `logger.py` | `setup_logger()` (Loguru, per-experiment files), `TensorBoardTracker`. |
| `report.py` | `EvaluationReporter` — config snapshot, metrics, model size, FLOPs/params (from `profiling.flops`), throughput/latency (from `profiling.latency`)/GPU memory, environment info; writes `.md`+`.json`, prints a console table. `get_latency_stats()` delegates to `profiling.latency.measure_latency`. |
| `visualize.py` | `save_confusion_matrix`, `save_roc_curve`, `save_pr_curve`. |
| `plot_training.py` | `plot_training_curves()` — reads a TensorBoard event file, dumps one PNG per scalar tag. |

---

## 16 · Testing

`tests/` — one file per implementation area, 310 tests total (`pytest tests/ -v`):
`test_orchestration.py`, `test_metrics.py`, `test_data_contract.py`, `test_channels.py`,
`test_optim.py`, `test_models.py`, `test_mamba.py`, `test_losses.py`, `test_stats.py`,
`test_profiling.py`, `test_attribution.py`, `test_uncertainty.py`, `test_robustness.py`,
`test_analysis.py`, `test_reporting.py`, `test_sweep.py`, `test_ci_audit.py` (checks the suite's
own completeness against `Technical_Framework_Spec.md`'s §19 test table, not framework behaviour
directly). CI (`.github/workflows/ci.yml`) runs the full suite on CPU wheels on every push/PR,
excluding `mamba-ssm`/`causal-conv1d` (no GPU on the runner — the suite only needs the pure-PyTorch
scan fallback).

---

## 17 · Dashboard (`dashboard/` / `exp_dashboard/`)

A self-contained Flask + tmux web UI for running experiments without a terminal (configs
browser/editor, terminal launcher, scheduler, reports/history viewers, machine stats, TensorBoard
launcher). Both directories are gitignored, byte-identical duplicates, and fully decoupled from the
rest of the codebase (`backend/*.py` imports only `yaml`/`tmux`, never `utils`/`datasets`/`models`/
`training`) — none of Phases 0–14's dependencies or the config schema validator touch it. Not
exercised by this reference's rest; see `dashboard/README.md` if it's in use.

---

## 18 · Dependencies & environment

Pinned to `torch==1.13.1`/`torchvision==0.14.1`; the numeric stack (`numpy`/`scipy`/`scikit-learn`/
`scikit-image`/`pandas`) is pinned together to avoid an ABI mismatch against that torch/numpy pin.
`env/environment.lock` is a `pip freeze` snapshot of a known-good CPU-only resolution.

`mamba-ssm==1.0.1`/`causal-conv1d==1.1.1` (Mamba fused-kernel scan) are CUDA extensions with no
prebuilt PyPI wheel — install the matching GitHub-Release wheel for your CUDA/torch/Python/cxx11-ABI
combination (exact URLs in `requirements.txt`'s comment block), or skip both and let the model
family fall back to `models/auxiliary/ss2d_ref.py`'s pure-PyTorch scan automatically.
`onnx`/`pyarrow`/`fvcore`/`captum`/`statsmodels`/`pytest`/`pydantic` were added across Phases 1–11
for schema validation, per-image Parquet, profiling, attribution, statistics, and the test suite
respectively — see `requirements.txt`'s inline comments for which phase added what and why.
