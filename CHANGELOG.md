# Changelog

Tracks framework-level changes made while bringing `dissert` into compliance with
`Technical_Framework_Spec.md`, per `IMPLEMENTATION_PLAN.md`. Not a user-facing release log.

## Phase 0 — torch 1.11 → 1.13 environment bump (2026-08-27)

- `torch==1.11.0`/`torchvision==0.12.0` → `torch==1.13.1`/`torchvision==0.14.1`. All other
  pinned packages (`numpy`, `albumentations`, `timm`, `transformers`, `huggingface-hub`, ...)
  resolved cleanly against the new torch pin with no forced bumps.
- Added `pydantic>=2,<3`, `statsmodels`, `fvcore`, `captum` (installed and verified importable),
  and `mamba-ssm==1.0.1` / `causal-conv1d==1.1.1` (pinned for the Phase 6 Mamba/VSS model family;
  **not installable on the dev sandbox used for this work**, which has no `nvcc`/CUDA toolkit —
  `pip install` fails at metadata generation with `NameError: name 'bare_metal_version' is not
  defined`, the expected failure mode of their setup.py's CUDA-toolkit probe when `nvcc` is
  absent. Confirmed prebuilt wheels exist for CUDA 11.8 + torch 1.13 + Python 3.8 on both
  projects' GitHub Releases — see `requirements.txt`'s comment block for exact URLs — for
  installation on the actual GPU training machine. Until then, Phase 6 must use the
  `models/auxiliary/ss2d_ref.py` pure-PyTorch fallback in this sandbox.)
- Removed `torchaudio==0.11.0+cu113` from the `thesis` conda env: unpinned in `requirements.txt`,
  unimported anywhere in the codebase, and the only package `pip check` flagged as conflicting
  with the new torch pin. Left `mmcv-full`/`addict`/`yapf` installed but undeclared/unused
  (present in the env, imported nowhere, nothing depends on them) — not removed since they were
  pre-existing and not blocking anything.
- Fixed `compose: ["../../base_config.yaml"]` in 11 experiment configs under
  `configs/experiment/{gmkunet,mkunet}/*.yaml` — one directory level too many, resolved to the
  nonexistent `configs/base_config.yaml` instead of `configs/experiment/base_config.yaml`. This
  was a pre-existing bug (unrelated to the torch bump) that made every one of those configs
  unloadable; discovered while trying to run this phase's own reproducibility smoke test.
- **Reproducibility smoke test** (`checkpoints/gmkunet_t_clinicdb/best_fold0.pth`,
  `configs/experiment/gmkunet/gmkunet_t_clinicdb.yaml`, fold 0, ClinicDB test split, CPU — this
  sandbox has no GPU): `eval.py` under torch 1.11.0+cu113 vs. torch 1.13.1+cpu produced
  bit-identical results — Dice 0.7816, mIoU 0.6881, HD95 35.39px, ASD 13.31px, Precision 0.7636,
  Recall 0.8640 — in both runs. No drift beyond expected latency/throughput timing noise.
- **Training smoke test**: `mkunet_t_clinicdb`, fold 0, 2 epochs, CPU, torch 1.13.1 — dataloader
  caching, AMP CUDA-guard (correctly no-ops on CPU), optimizer/scheduler, checkpointing,
  EarlyStopping, and callbacks all ran without error; Dice rose 0.163 → 0.258 across the 2 epochs
  from a fresh init, as expected for a from-scratch run (not a resume, so not a like-for-like
  comparison to a historical checkpoint — the eval-side smoke test above is the actual
  before/after reproducibility check).
- **EMCAD / PVTv2**: model builds and runs a forward pass cleanly under torch 1.13. Could not
  verify the actual `torch.load(...)` pretrained-checkpoint path end-to-end — no
  `pretrained_pth/pvt/*.pth` file exists in this sandbox (gitignored, never downloaded here); the
  loader correctly warns and falls back to random init when the file is absent. No code change
  needed regardless — torch 1.13 is well below the 2.6 `weights_only` default flip.
- `requirements.txt`, `env/environment.lock` (new), `README.md` updated accordingly.

**Not verified in this sandbox** (no GPU, no CUDA toolkit, no internet-independent build tools):
CUDA-build install of torch/torchvision, `mamba-ssm`/`causal-conv1d` import or kernel execution,
real (non-random-init) PVTv2 checkpoint loading, any GPU-only behavior. These need to be repeated
on the actual training machine before treating Phase 0 as fully closed there.

## Phase 1 — Config schema, run identity, manifest, determinism, orchestration core (2026-08-27)

- **New `orchestration/` package**: `schema.py` (pydantic `Config`/`SearchSweepConfig` models
  mirroring every section of `configs/base.yaml`; `validate_config()`), `runid.py`
  (`config_hash()` — SHA1 over the resolved config with `training.seed` excluded;
  `run_id()` → `R-{hash[:7]}-s{seed}-f{fold}`, with `f-` for a non-CV run's fold segment),
  `manifest.py` (`RunManifest`/`build_manifest()` — resolved config, config hash, git commit +
  dirty flag, env hash, hardware, timing, `nondeterministic_ops`, written to
  `artifacts/runs/<run_id>/manifest.json`), `ledger.py` (`LedgerWriter`, CSV-backed
  Runs/Compute/Test_Evals/Stats tables — only Runs/Compute are populated yet), `runner.py`
  (`run_sweep()` — expands seed×fold, idempotent `status: done` skip, manifest+ledger wrapping
  around `train.run_training()`, one bad run doesn't stop the sweep).
- **`utils/config.py`**: `load_config()` now pipes its merged dict through
  `orchestration.schema.validate_config()` by default (`validate=False` opts out for a fragment
  loaded standalone) — an unknown key, a missing required key, or a wrong-typed value anywhere in
  the config tree now raises `pydantic.ValidationError` at load time instead of a confusing
  `KeyError`/silent no-op deep in train.py/eval.py.
- **New `training/determinism.py`**: single `seed_everything(seed)` (python/numpy/torch/cuda +
  `torch.use_deterministic_algorithms(True, warn_only=True)`), replacing train.py's local
  `set_seed()`. Captures any non-deterministic-op warning torch emits into
  `get_recorded_nondeterminism()` rather than silently dropping it. `eval.py` — which previously
  had **zero** seeding — now calls it too.
- **`utils/checkpoint.py`**: `CheckpointManager` now accepts `config_hash`/`run_id` and embeds
  them plus the current git commit into every saved `last*.pth`/`best*.pth`. `train.py`'s
  `run_training()` gained an optional `run_id` param (auto-computed from the config+seed+fold when
  omitted, so even a bare `python train.py` produces addressable checkpoints).
- **Bootstrapped `tests/`**: `conftest.py` (synthetic tiny-dataset fixtures — a handful of random
  PNGs generated at test time, no dependency on the gitignored real `data/`),
  `test_orchestration.py` (16 tests: schema accept/reject, config_hash/run_id, manifest round-trip,
  ledger, sweep idempotency, and `test_determinism` — trains a real tiny UNet twice from the same
  seed on synthetic data and asserts bit-identical final weights with zero recorded
  non-determinism). Added `pytest>=8` to `requirements.txt` (not explicitly listed in Phase 0's
  dependency set, but obviously required to run the test suite this phase bootstraps).
- **Config layout relocation** (the "layout reconciliation" work, done as part of this phase since
  it's what the new schema validates against): `configs/experiment/base_config.yaml` →
  `configs/base.yaml`; new `configs/dataset/{clinicdb,colondb}.yaml`,
  `configs/model/{mkunet,gmkunet}/{t,s,base,m,l}.yaml`, `configs/training/emcad.yaml` fragments
  populate the previously-empty scaffold directories; all ~12 real experiment configs now compose
  these instead of duplicating full `model:`/`dataset:` blocks inline. `train.py`/`eval.py`
  (`--config`) and `search.py` (`--base-config`) CLI defaults now point at
  `configs/experiment/mkunet/mkunet_t_clinicdb.yaml` — `configs/base.yaml` alone is deliberately
  **not** a runnable standalone config anymore (see next item).
- **Bug found and fixed**: `configs/base.yaml` used to default `dataset.name`/`dataset.root` to
  ClinicDB. `configs/experiment/mkunet/mkunet_s_colondb.yaml` composed only the base config and
  never overrode either field — so despite its filename, its `experiment_name`, and its checkpoint
  directory (`checkpoints/mkunet_s_colondb/`, which contains 5 real trained fold checkpoints) all
  claiming ColonDB, **it was actually training and evaluating on ClinicDB the entire time**. Fixed
  by (a) making `dataset.name`/`root` required with no default in the schema — a config that
  forgets to select a dataset now fails to load instead of silently picking one — and (b) wiring
  `mkunet_s_colondb.yaml` to actually compose `configs/dataset/colondb.yaml`. Verified: the fixed
  config now loads 303 train / 38 val ColonDB pairs (vs. ClinicDB's 440/110). **The existing
  checkpoints under `checkpoints/mkunet_s_colondb/` predate this fix and are ClinicDB results —
  do not cite them as ColonDB results in the dissertation; re-run that config to get a real
  ColonDB checkpoint.**
- Also fixed while touching these files: all ~11 `gmkunet`/`mkunet` experiment configs had a
  `compose: ["../../base_config.yaml"]` path one directory level too deep (resolved to the
  nonexistent `configs/base_config.yaml`), pre-existing and unrelated to Phase 0/1 — every one of
  these configs was already unloadable before this session. `README.md`/`datasets/stats.py`
  updated to match the new paths.
- Verified end-to-end against the real pipeline throughout, not just unit tests: `eval.py` on
  `gmkunet_t_clinicdb` reproduces Phase 0's exact baseline numbers after all of the above
  (Dice 0.7816, mIoU 0.6881); a real `train.py` run confirms checkpoints now embed a correct
  `run_id`/`config_hash`/`git_commit`; `orchestration.runner.run_sweep()` verified against both a
  fake and the real `train.run_training`, including idempotent re-run skipping; `search.py` still
  completes a full grid sweep (summary CSV, best-config YAML, markdown report) with per-trial
  failure isolation intact.

## Phase 2 — Canonical metrics module (2026-08-28)

- **New top-level `metrics/` package**, collapsing the three independently-drifted Dice/IoU/HD95/
  ASD implementations (`utils/metrics.py`'s `get_binary_metrics`/`compute_dataset_metrics`,
  `utils/report.py`'s `compute_extended_metrics`, and `training/trainer.py`'s per-batch rolling
  proxy) down to two: this package (canonical) and the rolling proxy (deliberately kept separate,
  see below). `region.py` (dice/iou — empty-mask convention ported unchanged: both-empty → 1.0),
  `boundary.py` (hd95/asd/nsd — **behaviour change**: exactly-one-empty is now `None`/excluded +
  counted, replacing the old ad hoc `999.0` penalty that silently distorted any mean/percentile by
  however many empty-vs-nonempty pairs happened to land in a given eval; both-empty stays 0.0/1.0),
  `detection.py` (precision/recall/specificity/F2/accuracy — ported from `utils/report.py`
  unchanged, including its zero-denominator→0.0 convention; **note this doesn't match region.py's
  both-empty→1.0 convention** — pre-existing inconsistency, not introduced here, left as-is since
  Phase 2's scope was consolidation + the one documented HD95/ASD/NSD change, not a second
  unscoped convention change; adds `fpr_on_normals`/`specificity_on_lesion_free_subset`, both
  `None` — not 0.0 — when a dataset has no lesion-free images, which is true of ClinicDB/ColonDB),
  `calibration.py` (ECE, equal-mass binning — new, no prior implementation existed),
  `aggregate.py` (`EMPTY_MASK_CONVENTION` constant documenting every convention above in one place,
  `compute_dataset_metrics()`, `dice_p5`/`dice_p25`, `write_per_image_parquet()`).
- `utils/metrics.py` shrunk to just `count_parameters`/`measure_throughput`/`log_model_summary`
  (Phase 10's territory, not moved) — `get_binary_metrics`/`compute_dataset_metrics` deleted, not
  deprecated-and-reexported, since every call site was migrated in this same session rather than
  across a longer rollout. `utils/report.py`'s `compute_extended_metrics` deleted outright; its
  `EvaluationReporter.set_eval_results()` no longer takes `preds`/`gts` (no longer needed — the
  canonical `base_metrics` dict already carries precision/recall/specificity/f2/accuracy) and its
  "(N/A — multiclass)" table note is gone, since the new implementation computes those correctly
  for multiclass too (a genuine improvement: the old per-image TP/FP/FN/TN on a raw multiclass
  label map would have collapsed classes together, so it was skipped entirely before; the new
  version does a proper per-class one-vs-rest decomposition instead).
  `eval.py`/`training/trainer.py` now `from metrics import compute_dataset_metrics` directly
  instead of via `utils`'s re-export.
- Added `pyarrow==17.0.0` to `requirements.txt` for `write_per_image_parquet()` — the last release
  with a Python 3.8 wheel (18.0.0+ dropped 3.8; bump alongside a future Python bump).
- `training/trainer.py`'s rolling per-batch Dice/IoU (`train_one_epoch`, computed from raw logits,
  no medpy) is **unchanged** per the plan — it's a deliberately separate, cheap tensor-native
  computation for the progress bar/TensorBoard, not a second canonical implementation. Its
  docstring now says explicitly: never cite `train_dice`/`train_iou` in a results table.
  `validate()`'s metrics (the ones actually logged/checkpointed as `val_dice` etc.) already called
  the canonical function before this phase and still do, now pointed at the new package.
- **New `tests/test_metrics.py`** (7 tests): `test_metric_conventions` (every empty-mask rule in
  `EMPTY_MASK_CONVENTION`, checked against real function calls, not just read from the constant),
  exclusion-counting, the lesion-free-subset aggregates, an ECE sanity check, a Parquet
  round-trip, and `test_rolling_tracks_canonical_at_epoch_end` — builds a real `Trainer`, calls
  `train_one_epoch()` directly to capture the actual rolling `(dice, iou)` return value, then
  independently re-evaluates the same post-epoch weights on the same training data via the
  canonical path and asserts they land in the same ballpark (measured drift on the synthetic
  fixture: ~0.017; asserted tolerance: 0.35, generous on purpose to avoid flakiness while still
  catching a real convention-divergence bug). 23/23 tests passing repo-wide.
- Verified end-to-end against the real pipeline: `eval.py` on `gmkunet_t_clinicdb` still reproduces
  the exact baseline (Dice 0.7816, mIoU 0.6881, ..., all bit-identical to Phase 0/1) and its JSON
  report now carries `nsd`, `dice_p5`/`dice_p25`, `hd95_excluded_n`/`asd_excluded_n`, `ece`, and
  `fpr_on_normals`/`specificity_lesion_free` (`None` for ClinicDB, confirming the "no lesion-free
  images" case is detected correctly rather than silently reporting 0.0/1.0); a real multi-epoch
  `train.py` run confirms the canonical validation path (now backed by `metrics/`) still trains,
  checkpoints, and logs `val_dice`/`val_miou`/`val_hd95`/`val_asd` correctly.

## Phase 3 — Data-layer retrofit: (image, mask, meta) contract, leakage guards, BUSI/ISIC18 (2026-08-28)

- **`Dataset.__getitem__` now returns `(image, mask, meta)`** instead of `(image, mask)` —
  `meta = {subject_id, source_dataset, spacing, artefact_flags}`, applied project-wide (ClinicDB/
  ColonDB included, not just new datasets — a leakage guard is meaningless otherwise).
  `spacing` is `()`, not `None`, for "no spacing metadata" — confirmed torch 1.13's default
  `DataLoader` collate raises on a bare `None` inside a dict value but happily collates an empty
  sequence into `[]`. Single coordinated change across every consumer in one pass: found via a
  full-repo unpacking-pattern search, **5 call sites across 4 files**, not the ~4 named in the
  plan — `training/trainer.py` (`train_one_epoch`, `_validate_model`), `eval.py` (`evaluate`),
  `training/callbacks.py` (`PredictionOverlayCallback`), and **`utils/metrics.py`'s
  `measure_throughput`** (missed by an initial grep since it unpacks `(images, _)`, not
  `(images, masks)` — caught by actually running `eval.py --allow-test-eval` end-to-end afterward,
  not just by the grep). ClinicDB/ColonDB's `subject_id` honestly defaults to frame-level identity
  (filename stem) via each handler's new `ARTEFACT_FLAGS = {"frame_level_only_no_video_grouping":
  True}` — the caveat that used to live only in `KFoldDataModule`'s docstring now has a
  machine-readable home.
- **New `datasets/splits.py`**: `assert_no_subject_overlap()` (`LeakageError`),
  `load_published_split()` (reads + SHA1-hashes a `train_list_*.txt`-style file),
  `duplicate_cross_check()` (exact-hash cross-split duplicate detection — closes the gap in
  `_GenericHandler`'s unguarded random shuffle), plus `ExternalDatasetError` and
  `TestLoaderGuardError` (see below).
- **New `datasets/preprocess.py`**: `build_manifest()` (path/subject_id/split/`mask_empty`/
  resolution, computed once from the raw mask file — deliberately *not* inside `__getitem__`,
  where it would be both wasteful to recompute every epoch and ambiguous post-augmentation, since
  a crop can move a lesion in or out of frame run to run); `dedup()` (phash + SSIM, two-stage —
  mandatory for BUSI per spec).
- **Test-loader guard**: `StandardSplitDataModule.get_test_loader()` and
  `KFoldDataModule.get_test_loader()` now require a `token` argument, raising
  `TestLoaderGuardError` unless it was minted by the new
  `orchestration.ledger.LedgerWriter.issue_test_token()` (which appends a `Test_Evals` row as a
  side effect — "did this run touch the test set" is now a ledger fact, not a convention).
  `eval.py` gained `--test-token <TOKEN>` (pre-minted, e.g. from an orchestrated sweep) and
  `--allow-test-eval` (self-mints one for a manual run). Verified all four paths against real
  ClinicDB data: no token → refused with a clean error (not a raw traceback — added a
  `try/except TestLoaderGuardError` around the call, since a *bogus* token bypasses the earlier
  argparse-level check and needs its own handling); bogus token → refused; `--allow-test-eval` →
  works, ledger row recorded; pre-minted `--test-token` → works. `orchestration.runner`,
  `train.py`, `training/trainer.py`, and `search.py` contain zero references to `get_test_loader` —
  verified as a repo-wide static fact (`tests/test_data_contract.py::test_sweep_cannot_see_test`),
  so the sweep/training path structurally cannot reach the guarded test set.
- **`_GenericHandler` gains `external: bool`** (`dataset.external` in config): when set,
  `get_dataset("train")`/`get_dataset("val")` raise `ExternalDatasetError` instead of silently
  returning an empty loader — every sample goes to `"test"`. Registered handlers (ClinicDB et al.)
  don't support this; it's specifically for a held-out-only external cohort with no train/val split
  by design. Handler construction protocol changed to always pass `seed` (`DATASETS[name](cfg,
  seed)`, not `DATASETS[name](cfg)`) so a handler that needs to compute its own random split (BUSI)
  can, while ones that don't (ClinicDB/ColonDB/ISIC18 — published or official splits) just accept
  and ignore it.
- **New `datasets/busi.py`, `datasets/isic18.py`** — same `get_dataset()`/`get_kfold_pairs()`
  interface as ClinicDB/ColonDB, registered into `DATASETS`. **Structurally complete but UNVERIFIED
  against real data** — no BUSI or ISIC18 files exist anywhere reachable from this environment.
  Built against each dataset's publicly documented layout (BUSI: `benign/malignant/normal`
  subfolders, `<case>.png` + `<case>_mask.png`, no official split — hence a seeded random split via
  `dataset.split`, with mandatory `preprocess.dedup()`; ISIC18: the official
  `ISIC2018_Task1-2_*_Input`/`ISIC2018_Task1_*_GroundTruth` directory triad with a
  `<stem>_segmentation.png` mask suffix that needed its own pairing logic — the shared
  `MedicalSegmentationDataset` filename-matching only tolerates `_mask`/`_gt`). Verified the
  handler *interface* end-to-end against synthetic data built to match each documented layout
  (`tests/test_data_contract.py`) — get_dataset/get_kfold_pairs/mask-pairing/dedup all confirmed
  working structurally; **sanity-check directory names and the mask suffix against your actual
  download before first real use.** New `configs/dataset/busi.yaml`/`isic18.yaml` fragments, same
  "point this at real data, unverified" caveat.
- **New `tests/test_data_contract.py`** (13 tests): the four the plan names —
  `test_no_subject_overlap` (including a real check against ClinicDB's actual K-Fold split, not
  just synthetic ids), `test_external_never_trained`, `test_test_loader_guard`,
  `test_sweep_cannot_see_test` — plus contract round-trip tests (including through a real
  `DataLoader`, where the `None`-vs-`()` collate issue above actually surfaces),
  `duplicate_cross_check` unit tests, and the BUSI/ISIC18 handler-interface tests. 36/36 tests
  passing repo-wide.
- **Real finding**: ran the new `duplicate_cross_check()` against ColonDB's actual published
  split (not just as a unit-test exercise) and it found **35 of ColonDB's 379 images (9.2%) are
  byte-identical duplicates split across train/val/test** — 9 duplicate groups, most crossing the
  train/test boundary specifically (confirmed genuine: matching MD5 + file size + real image
  content, not blank/degenerate frames — e.g. `train/images/198.png` and `val/images/183.png` are
  byte-identical). This means any model trained on ColonDB's current published split (including
  the newly-fixed `mkunet_s_colondb` config from Phase 1) has some fraction of its "held-out" test
  images already memorisable from training. **Not fixed here** — per user instruction, no
  checkpoints were re-run this session, and de-duplicating a published split is a data-curation
  decision the user should make deliberately (e.g. via `datasets.preprocess.dedup()` or
  `datasets.splits.duplicate_cross_check()`-guided exclusion), not one this pass makes silently.
  Ran the same check against ClinicDB: 0 cross-split duplicates found.
- `MedicalSegmentationDataset.__init__` gained `source_dataset`/`subject_id_fn`/`artefact_flags`
  params (all optional, default to `""`/frame-level-stem/`{}`); `KFoldDataModule.get_fold_loaders()`
  now threads `source_dataset`/`artefact_flags` through explicitly (it constructs
  `MedicalSegmentationDataset` directly from cached fold pairs, bypassing the handler's
  `get_dataset()`, so couldn't inherit them the way `StandardSplitDataModule` does).

## Phase 4 — Channel construction module + modality-driven augmentation policy (2026-08-28)

- **New `datasets/channels.py`**: `build_channels(image, mode, order)` for m1 (rgb) / m2 (rgb+xy) /
  m3 (rgb+ycbcr) / m4 (rgb+xy+rtheta) / m5 (all), built from named channel *groups* (not hardcoded
  per-mode logic) so a caller can pin an explicit `order` — the group boundaries this produces are
  what Phase 11's per-group Shapley attribution will slice by. `xy_channels()` ([-1,1] position),
  `r_theta_channels()` (radius + sin/cos(θ) — sin/cos specifically to avoid the -π/π branch cut a
  raw angle channel would have along the negative x-axis), `randproj_channels()`/
  `coordonly_channels()` ablation controls, `modality_effective_channels()` (drops the `ycbcr`
  group for `modality=grayscale` — closes `models/baseline/emcad.py`'s old ad hoc
  "if grayscale, repeat to 3ch" gap at the *data* level instead of patching every model), per-mode
  channel-stats caching (`channel_norm_stats()`, hashed by dataset+mode+modality).
  `models/proposed/gmk_unet.py`'s private `_ycbcr()` now imports `ycbcr_from_rgb_tensor()` from
  here instead of defining its own — same BT.601 coefficients (`_YCBCR_COEFFS`), shared with the
  numpy path `build_channels()` uses, so the two can never numerically drift apart even though one
  runs on live GPU tensors during a forward pass and the other on numpy arrays at data-loading
  time (confirmed 0.0 max diff between the two on a real ClinicDB image). Verified the real
  `gmkunet_t_clinicdb` checkpoint still reproduces the exact baseline (Dice 0.7816) after this
  swap.
- **New `datasets/augment.py`**: `AugmentationPolicy`, built once per dataset from
  `(modality, ds_cfg)`. Enforces ORDER=post (geometric augmentation via Albumentations first, then
  `build_channels()` regenerates XY/Rθ from the *already-augmented* frame — a pre-augmentation XY
  channel would encode stale coordinates once the frame is flipped/rotated). Modality-conditioned
  augmentation intensity (spec §5.4's colour vs. grayscale-ultrasound vs. grayscale-microscopy):
  the config schema only exposes the coarse `dataset.modality: colour|grayscale` the spec fixes at
  dataset level, so the finer ultrasound-vs-microscopy split (needed only for augmentation
  intensity, not channel construction) is resolved from `dataset.name` via a small
  `_GRAYSCALE_SUBTYPE` table (`{"busi": "ultrasound"}` today; no microscopy dataset registered
  yet).
- Config schema gained `dataset.modality` (`colour|grayscale`, default `colour`), `channel_mode`
  (`m1`..`m5`, default `m1`), `channel_order`. All three are inert by default — every existing
  config stays on `m1`/`colour` (today's plain-RGB pipeline, byte-for-byte unchanged) unless a
  config explicitly opts into a different mode.
- **Deliberately not wired into the live training pipeline this phase**: `datasets/dataset.py`/
  `datasets/datamodule.py` still call `transforms.build_transforms()` directly, not
  `AugmentationPolicy` — no model in Phases 0–5 accepts non-RGB input, so there's nothing yet to
  consume `channel_mode` > m1. Phase 6 is explicitly where this gets wired in for real: the Mamba
  model's auxiliary VSS encoder is specified to consume the same multi-channel input as the
  primary encoder, via `dataset.channel_mode`.
- **New `tests/test_channels.py`** (9 tests): the five the plan names —
  `test_val_test_no_augment`, `test_mask_interpolation`, `test_channel_order`,
  `test_theta_continuity`, `test_grayscale_drops_colour` — plus a BT.601 reference-value check and
  an `AugmentationPolicy` dataset-specific intensity-scale check. Found and fixed two real bugs
  while writing these: the first `test_theta_continuity` draft swept a row that passed through the
  coordinate origin itself (a genuine r=0 singularity no encoding can smooth over, different from
  the branch-cut discontinuity sin/cos is actually meant to fix) rather than testing the branch cut
  itself — rewrote it to sweep a fixed negative-x *column* through y=0, which is where the branch
  cut actually lives, and added a direct comparison confirming a raw `atan2` channel really does
  jump there (~2π) while sin/cos doesn't. Also fixed a wrong transform-list index (`[-2]` instead
  of `[-3]`) in the augmentation-intensity test. 45/45 tests passing repo-wide.

## Phase 5 — Model registry hardening: capacity control, budget guard, parameter-group optimizer (2026-08-28)

- **New `models/build.py`**: `build_width_matched(base_model_fn, width_candidates, input_shape,
  target_params, target_flops, tol)` — searches a model family's already-exposed width-scaling
  knob (MK-UNet/GMK-UNet's channels-list presets) for a configuration matching a target
  param/FLOPs budget. Verified against real MK-UNet presets: finds an exact match (0.0 relative
  error) when the target is in the candidate list, correctly reports `within_tolerance: False`
  with the closest candidate otherwise.
- **`models/registry.py`**: `ModelRegistry.get()` (and `get_model()`) gained
  `budget_ceiling`/`allow_over_budget` — profiles every built model immediately and raises
  `ModelBudgetExceededError` unless the override is explicit. Applies to every registered model,
  including whichever one Phase 6 adds — the `@register` decorator API itself is unchanged.
- **`training/optimizers.py`**: `build_optimizer(cfg, model)` — signature changed from
  `build_optimizer(cfg, params)` (confirmed zero param-group splitting before this) to take the
  whole model, not `model.parameters()`, since the new `no_decay_group(model)` needs each
  parameter's dotted name and owning module to split by *name* (bias / normalisation-layer
  affine params / — forward-looking for Phase 6 — a parameter literally named `A_log` or `D`),
  not by tensor shape. Two param groups (`weight_decay` / `0.0`) built and logged for every
  optimizer choice (Adam/AdamW/SGD). Updated both call sites: `train.py`'s main path and
  `training/trainer.py`'s multi-stage `_fit_staged()` rebuild (which used to pass a
  `filter(requires_grad, ...)` iterator — `no_decay_group()` already skips frozen params via its
  own `requires_grad` check, so passing the whole model post-`_set_frozen()` produces the same
  trainable-only set without a separate filter step).
- **Real finding, fixed**: this param-group change breaks resuming from any checkpoint saved
  under the old single-group optimizer (`optimizer.load_state_dict()` raises `ValueError` on a
  param-group-count mismatch) — caught running the real training smoke test, not a unit test.
  Fixed in `utils/checkpoint.py`'s `CheckpointManager.load()`: catches that specific `ValueError`,
  logs a clear warning, and continues with a freshly-initialised optimizer rather than crashing
  the whole resume — model weights (the expensive part) still load and restore correctly; only
  Adam's per-parameter moment estimates are lost, once, for any checkpoint that predates this
  change. Verified end-to-end: a real resume from a pre-existing checkpoint now degrades
  gracefully with a warning instead of crashing, and training continues correctly from there.
- **New `tests/test_optim.py`** (4 tests) and **`tests/test_models.py`** (6 tests): the two the
  plan names — `test_param_groups`, `test_capacity_control_match` — plus frozen-param exclusion,
  SGD param-grouping, no-target/no-candidates validation, and the registry budget guard
  (raises/bypasses/not-triggered). 56/56 tests passing repo-wide.

## Phase 5.5 — MVP checkpoint reached (2026-08-28)

Verified per the plan's own criterion: ran `orchestration.runner.run_sweep()` across 2 real model
families (`mk_unet_t`, `gmk_unet_t`) × 2 seeds on real ClinicDB data — 4 real training runs, not
mocked. Confirmed: distinct `config_hash` per family (shared across seeds), distinct `run_id` per
(family, seed), correct manifests (resolved config, git commit + dirty flag, env hash, hardware,
timing) and ledger rows written for all 4; re-running the identical sweep call afterward correctly
reports all 4 as `skipped-done` with zero retraining. Along the way, found that the two seeds'
validation Dice was suspicious identical — investigated directly (not dismissed): confirmed via a
3-seed direct comparison that *training* metrics (loss, train Dice) do differ correctly per seed,
but a ~40K-parameter model after a single epoch collapses to an identical hard-thresholded
all-background validation prediction regardless of seed, which makes Dice/HD95/ASD — computed on
the thresholded output, not the raw logits — genuinely identical by construction, not a
determinism bug. (Also incidentally confirmed Phase 2's boundary-metric exclusion behaviour is
working as designed: an all-background prediction against ClinicDB's always-non-empty ground
truth excludes every sample from HD95/ASD, correctly producing 0.0 as "mean of zero excluded
samples," not a misleadingly "perfect" score — visible via `hd95_excluded_n` for anyone who checks
it, which is the point of tracking that field at all.)

Phases 0–5 now let a user run config-hashed, leakage-guarded, canonically-metriced experiments
across all 6 existing model families (5 baselines + GMK-UNet ×5 sizes) on ClinicDB/ColonDB (BUSI/
ISIC18 handlers are in place but unverified against real data — see Phase 3's entry), with a
working idempotent sweep/ledger. Per the plan, Phase 6 (Mamba) is dissertation-critical but not
MVP-blocking.

## Phase 6 — Mamba/VSS hybrid model family (2026-08-28)

New architecture built from spec §6 on top of Phases 0–5's hardened registry/channel/
capacity-control machinery. The spec deliberately leaves PVM's internals and the exact 8-direction
scan geometry undefined beyond naming the config surface ("Direction definitions documented in
code with an ASCII diagram") — every such choice below is this implementation's own, documented
as such, not a claim of matching a specific paper.

- **New `models/auxiliary/ss2d_ref.py`**: the core (A, B, C, Δ) selective-scan recurrence, pure
  PyTorch, sequential for-loop over the sequence length — the same recurrence `mamba-ssm`'s own
  `selective_scan_ref` computes, minus the fused kernel. Verified against a hand-computed closed
  form (A=0, constant Δ=1 reduces the recurrence to a plain cumulative sum) before building
  anything on top of it, and confirmed differentiable.
- **New `models/auxiliary/ss2d.py`**: the 2D multi-directional scan. `scan_directions ∈ {1,2,4,8}`
  as named direction sets (raster / raster-reverse / transposed / transposed-reverse, plus two
  diagonal traversals + their reverses for 8), `merge ∈ {sum, learned}`. Primary path imports
  `mamba_ssm`'s fused kernel when importable; falls back to `ss2d_ref` automatically otherwise
  (this sandbox has no `mamba_ssm` — confirmed `nvcc`-less in Phase 0 — so every test here exercises
  the fallback path). Every direction's flatten/unflatten round-trips exactly (verified per
  direction, including the two diagonal traversals, which use a cached argsort-based permutation
  index).
- **New `models/auxiliary/vss.py`**: `VSSBlock` (LayerNorm → in-proj → depthwise conv+SiLU → PVM →
  out-proj, residual + DropPath — `timm`'s `DropPath`, not reimplemented) and `build_vss_stage()`
  (per-stage depth, linearly-ramped stochastic depth). **PVM** ("Parallel Vision Mamba group
  count"): this implementation's interpretation is *G independent, narrower SS2D scans* (each
  with its own dt/B/C projection, `A_log`/`D`) instead of one full-width scan — chosen because
  that's what makes "PVM group count" a real capacity/efficiency knob rather than a no-op.
  `A_log` initialised via the standard Mamba/HiPPO-inspired scheme (`log(1..d_state)` per channel),
  not randomly.
- **New `models/fusion.py`**: `fusion ∈ {cbffm, concat, add, xattn, none}` + `fuse_stages`, one
  shared interface (`forward(f_primary, f_auxiliary) -> Tensor`) so `mamba_unet.py` never branches
  on which mode is active — contrasting deliberately with GMK-UNet's `ExponentialDecayGating`,
  hard-baked into that model's own `forward()`. `cbffm` (Cross-Branch Feature Fusion Module): a
  per-pixel spatial gate (not GMK-UNet's global per-branch scalar) blending per-branch-projected
  features.
- **New `models/decoder.py`**: `MambaDecoder`, `skip_policy ∈ {primary_only, fused, both_concat}`,
  reusing `models/blocks.py`'s existing `DecoderBlock`/CBAM primitives rather than reimplementing
  upsampling. **Real bug found and fixed**: the first version stopped one stage short of full input
  resolution (it omitted the equivalent of MK-UNet's `decoder5` refinement step, which upsamples
  from `t1`'s H/2 back to H with no skip connection) — a real forward pass silently returned a
  half-resolution output instead of raising. Caught by actually checking the output shape against
  the input, not assumed; fixed by adding the missing final-upsample + refinement stage; now a
  named regression test (`test_decoder_reaches_full_resolution`).
- **New `models/proposed/mamba_unet.py`** (`mamba_unet`, registered via the existing `@register`
  pattern): primary encoder is `models/baseline/mk_unet.py`'s actual `mk_irb_bottleneck`/
  `MultiKernelInvertedResidualBlock`, imported and reused, not reimplemented — MK-UNet and this
  model share identical primary-branch numerics. **Channel input decision** (verified, not just
  asserted): the auxiliary VSS encoder's stem takes the same `in_channels` as the primary encoder
  — tested with an 11-channel (Phase 4's m5 mode) input end-to-end. Auxiliary branch: a strided-conv
  stem + 4 VSS stages at H/2, H/4, H/8, H/16 (patch-merging downsample between stages), matching
  MK-UNet's `t1..t4` skip resolutions exactly (its own maxpool-after-every-stage convention) so
  per-stage fusion needs no resizing.
- **Phase 5 integration, verified for real**: this is the first model in the registry to actually
  define parameters named `A_log`/`D` — confirmed `no_decay_group()` (built forward-looking in
  Phase 5, before this model existed) correctly routes all of them (8 of each, for a 4-stage ×
  2-PVM-group config) into the no-weight-decay group. `build_width_matched`/`ModelRegistry`'s
  budget guard both verified against real `mamba_unet` instances too.
- **Manifest wiring** (per spec: "config exposes which path is active, and it's recorded in the
  run manifest"): `train.run_training()` records `model.scan_impl` (when the model exposes it) via
  a new side-channel in `training/determinism.py` (`record_manifest_extra`/
  `get_recorded_manifest_extras` — needed since `run_training()`'s return type is a bare float,
  depended on by `search.py`, and couldn't easily carry this without breaking that);
  `orchestration/runner.py` reads it back and calls `manifest.record()`. Verified end-to-end: a
  real sweep run's `manifest.json` correctly shows `"scan_impl": "ss2d_ref"` in this environment.
- **New `configs/model/mamba/t.yaml`** and `configs/experiment/mamba/mamba_t_clinicdb.yaml`,
  following the same composition pattern as every other model family's configs.
- **New `tests/test_mamba.py`** (39 tests): the scan recurrence against a hand-computed reference,
  all 8 scan directions' flatten/unflatten round-trip, `SS2D`/`VSSBlock`/fusion/decoder
  forward+backward across their full parameter space, the assembled model (including the channel_mode
  and no_decay_group integration checks above), and **`test_mamba_determinism`** — the test the
  plan names for this phase: trains the same tiny `mamba_unet` config twice from the same seed and
  branches on which scan implementation actually ran. On `ss2d_ref` (what runs in this sandbox):
  asserts bit-identical final weights and zero recorded non-determinism, exactly like every other
  model family's determinism test. On the `mamba_ssm` fused-kernel path (unreachable here):
  asserts non-determinism gets *recorded* rather than silently treated as reproducible, per the
  spec's explicit pitfall guard — untestable in this sandbox, but written so it asserts the right
  thing whenever it does run that way. 95/95 tests passing repo-wide.
- Full real-data verification: a genuine `train.py` run on real ClinicDB images (small resolution —
  see CPU-performance note below) trains end-to-end with correct optimizer param groups, correct
  checkpointing, correct canonical metrics; `eval.py` loads the resulting checkpoint and evaluates
  correctly. The existing baseline (`gmkunet_t_clinicdb`) remains bit-identical (Dice 0.7816)
  throughout — zero effect on any existing model family.
- **CPU performance note, not a bug**: `ss2d_ref`'s sequential Python for-loop scans a sequence of
  length H×W per direction — at this project's real training resolution (256×256, auxiliary stage 1
  = H/2 = 128×128 = 16,384 sequential steps), a full training run on this CPU-only sandbox is
  impractically slow; real end-to-end training smoke tests here used a reduced resolution (32–64px)
  specifically to keep the reference path tractable for verification. This is exactly the
  performance gap the `mamba-ssm` fused kernel exists to close (Phase 0's torch 1.13 bump was
  pinned specifically to make that kernel importable) — full-resolution training with this model
  family should happen on the real GPU machine with `mamba-ssm` installed, not via the `ss2d_ref`
  fallback.

## Phase 7 — Loss module: declarative compound losses, redundancy guard, boundary loss (2026-08-28)

- **New top-level `losses/` package**, replacing `training/losses.py` (fully migrated and
  deleted, not left as a shim — same one-session-migration approach as Phase 2's metrics move).
  `losses/terms.py`: standalone `bce`/`ce`/`dice` (ported, verified numerically identical to the
  old `DiceLoss`/`ComboLoss`'s BCE branch to 1e-6), new `tversky` (verified `tversky(alpha=beta=0.5)
  == dice` exactly — a real mathematical identity, not just a design note), new `focal`, and a real
  distance-transform `boundary()` loss (Kervadec et al. 2019: `mean(probs * signed_distance_map)`)
  with disk-cached distance-map precomputation (`compute_or_load_distance_map`, mirroring
  Phase 4's channel-stats cache pattern). `losses/schedules.py`: `linear`/`constant` weight ramps.
  `losses/compound.py`: `CompoundLoss(term_list, term_kwargs)` — every loss this project builds,
  including a "single-term" one like plain Dice, is a `CompoundLoss`, not a special case; one
  `forward()`/`set_epoch()` path for every `loss_type`. `StructureLoss` kept as its own named class
  (boundary-*weighted* BCE+IoU, not the spec's distance-transform boundary loss — the two are
  easy to conflate under one name, so they're kept explicitly distinct).
- **Numerical equivalence verified before deleting anything**: ran every `loss_type` value
  ("bce", "dice", "structure", "combo", "adaptive_guide_fusion", including the binary/multiclass
  branches) through both the old and new implementations side by side — all matched to 1e-6 — only
  then deleted `training/losses.py`. `AdaptiveGuideFusionLoss`'s learnable-alpha option was
  dropped, not preserved: it was already dead code (its own docstring required a dedicated
  optimizer param group `train.py` never actually built), and no config uses
  `loss_type: "adaptive_guide_fusion"` at all.
- **Redundancy guard** (`orchestration/schema.py`): a new `training.loss_terms` field (used only
  when `loss_type: "compound"`) is validated by a pydantic cross-field check —
  stacking two terms from `losses.compound.REDUNDANT_TERM_FAMILIES` (currently `{dice, tversky}`,
  generalizing the spec's literal "dice+iou" example since this repo has no separate "iou" loss
  term) without an explicit `training.loss_override_reason` raises at config-load time. Verified
  all three cases (rejected without override, accepted with override, non-redundant pairs need no
  override).
- **Schedule ramp verified in a real training run, not just unit-tested**: configured a compound
  loss with `dice`'s weight on a 3-epoch linear ramp from 0→1, traced the actual per-epoch
  effective weight during a real `train.py` run on ClinicDB — confirmed 0.333 at epoch 1, 0.667 at
  epoch 2, matching the ramp formula exactly. Wired via `Trainer._fit_range()` calling
  `criterion.set_epoch(epoch, end_epoch)` once per epoch (also fires correctly for multi-stage
  training, each stage getting its own local epoch/max_epoch view).
- **Scope decision, documented not silently dropped**: `boundary()`'s distance maps must be passed
  explicitly to `CompoundLoss.forward(distance_maps=...)` — not wired into `Trainer`'s live
  `criterion(outputs, masks)` call, since that would require threading mask paths through Phase 3's
  `meta` dict and into the Trainer, and `boundary` has zero current usage in any active config.
  Fully functional via direct API use (verified); raises a clear error if configured without a way
  to supply distance maps, rather than silently misbehaving.
- **New `tests/test_losses.py`** (29 tests): term correctness/gradients, the Dice↔Tversky identity,
  the empty-mask convention (cross-checked for consistency with `metrics.region`'s own convention),
  distance-map sign convention and disk-cache round-trip, `CompoundLoss` construction/forward/
  `set_epoch`, every `get_loss()` preset's behavior, and the redundancy guard (all three cases).
  124/124 tests passing repo-wide. Real-pipeline regression check unchanged (`gmkunet_t_clinicdb`
  still Dice 0.7816).

## Phase 8 — CI test-suite completion (2026-08-28)

- **New `.github/workflows/ci.yml`** (none existed before): installs CPU torch/torchvision wheels,
  the rest of `requirements.txt` minus `mamba-ssm`/`causal-conv1d` (CUDA extensions with no PyPI
  wheel — a standard GitHub-hosted runner has no `nvcc`, the exact failure mode confirmed locally
  in Phase 0), `pip check`, then `pytest tests/ -v` — mirrors this session's own working install
  recipe exactly, not a guess.
- **New `tests/test_ci_audit.py`**: codifies the audit itself rather than doing it once by hand —
  walks every `tests/test_*.py` file's AST and confirms all 13 of spec §19's tests not blocked on
  a later phase actually exist by name; the remaining 2 (`test_flops_agreement`,
  `test_reporting_blocks`) are asserted *explicitly skipped with their blocking phase named*, not
  silently absent, so deleting Phase 10/14 from this file's tracking dict (instead of adding the
  real test once that phase lands) is a visible change in review rather than the audit just
  staying green forever. Verified the detection logic actually catches a fabricated missing test
  before trusting it.
- **Audit result**: 13/15 spec §19 tests exist (`test_no_subject_overlap`,
  `test_external_never_trained`, `test_val_test_no_augment`, `test_mask_interpolation`,
  `test_channel_order`, `test_theta_continuity`, `test_grayscale_drops_colour`,
  `test_param_groups`, `test_metric_conventions`, `test_capacity_control_match`,
  `test_test_loader_guard`, `test_sweep_cannot_see_test`, `test_determinism`); the remaining 2 are
  correctly blocked on Phase 10/Phase 14 per the plan. Strengthened `test_val_test_no_augment`
  while auditing it: it previously only checked the val/test pipeline was *deterministic*, not
  that it literally "contains only resize + normalise" (spec §19's exact wording) — a
  fixed-seed augmentation op would have passed the old check without satisfying the real
  requirement. Now asserts the transform list composition directly.
- 126 passed, 2 correctly-skipped, repo-wide.

## Phase 9 — Statistics module (2026-08-28)

- **New `stats/` package** (spec §10): `tests.py` (`wilcoxon_paired_test`, `bootstrap_ci`,
  `meaningfulness_gate`), `effectsize.py` (`cliffs_delta`, `paired_median_diff`), `correction.py`
  (`holm_bonferroni`, wrapping `statsmodels.stats.multitest.multipletests(method="holm")`),
  `ranking.py` (`friedman_test`, `nemenyi_posthoc` for cross-dataset critical-difference ranking),
  and `stats/__init__.py`'s `run_family_comparison()` orchestration function tying all four
  together: given a proposed method's per-image scores and one or more comparators' (paired, same
  images), it runs Wilcoxon + both effect sizes + a bootstrap CI per comparator, Holm-Bonferroni-
  corrects across the whole declared family in one call (not a subset picked after seeing results),
  applies the meaningfulness gate, writes `reports/json/stats/<family>.json` (spec §10's Output
  row), and appends one row per comparison to `orchestration/ledger.py`'s `Stats` table — the
  `STATS_FIELDS` schema Phase 1 declared but left unpopulated is now actually written to.
- **New `orchestration/schema.py`: `StatsConfig`** — an optional `stats:` experiment-config
  section (`family`, `comparators`, `min_meaningful_diff`, `alpha`) so a comparison family is
  declared in the config *before* any result is seen, matching spec §10's requirement that the
  Holm-Bonferroni family be fixed in advance rather than assembled post hoc from whichever
  comparisons turned out significant. Not yet wired into `train.py`/`eval.py` (nothing in this repo
  yet calls `run_family_comparison()` from a pipeline stage — that lands with Phase 14's reporting
  layer / orchestration completion); the schema field and the orchestration function both exist and
  are independently verified now so that wiring is a small connecting step, not new design work.
- **Every primitive cross-checked against its own independent reference**, not just internally
  consistent: `wilcoxon_paired_test` against `scipy.stats.wilcoxon` directly; `holm_bonferroni`
  against `statsmodels.stats.multitest.multipletests` directly; `friedman_test` against
  `scipy.stats.friedmanchisquare` directly; `cliffs_delta` against a hand-computed pairwise count on
  a fixed tiny example (and checked antisymmetric: `δ(a,b) == -δ(b,a)`); the Nemenyi
  critical-difference formula (`studentized_range.ppf(1-α, k, ∞) / √2`) against Demšar (2006)'s
  published table value — `q_0.05` at `k=3` is `2.343`, reproduced to 3 decimal places.
- **New `tests/test_stats.py`** (37 tests): every module above, including the four
  `meaningfulness_gate` verdict branches, Holm correction never *decreasing* a p-value, the
  `run_family_comparison` JSON round-trip and ledger side effect (both independently toggleable via
  `out_dir=None`/`ledger_dir=None`), and the empty-family/length-mismatch guards on every function.
  163 passed, 2 correctly-skipped, repo-wide (126 prior + 37 new). `pip check` clean. Real-pipeline
  regression check unchanged (`gmkunet_t_clinicdb` still Dice 0.7816 / mIoU 0.6881).

## Phase 10 — Profiling and deployment module (2026-08-28)

- **New `profiling/` package** (spec §14): `flops.py` (analytic + fvcore-agreement-checked FLOP
  counting), `latency.py` (GPU/CPU latency + throughput, spec's exact protocol), `memory.py` (peak
  CUDA memory + checkpoint size), `export.py` (ONNX/TorchScript/TensorRT, `ok`/`fail` + error
  class). Supersedes `utils/metrics.py`'s `log_model_summary`/`measure_throughput` (named for
  deletion by the plan) — that module now holds only `count_parameters`.
- **`flops.py`**: fvcore's `FlopCountAnalysis` has no handler for `models.auxiliary.ss2d.SS2D`'s
  custom elementwise recurrence (spec's own named pitfall: "FLOP profilers silently return zero for
  the selective-scan op") — confirmed directly, and worse than "silently zero": fvcore's tracer
  *unrolls the scan's sequential Python for-loop one iteration at a time*, and an untimed
  `FlopCountAnalysis` call on a 64x64 `mamba_unet` was still running after 90 seconds before being
  killed. Its `einsum` handler also partially (not zero, not complete — missing the raw elementwise
  `h = deltaA*h + deltaB_u` step, no handler for a bare tensor `*`/`+`) attributes cost to the scan,
  which would have silently corrupted any comparison built on it. Fixed by temporarily monkey-patching
  every `SS2D` submodule's `forward` to a zero-cost passthrough for the duration of an fvcore trace
  (`_stubbed_scans`) — makes fvcore's SS2D contribution deterministically and exactly zero (matching
  the spec's documented failure mode precisely) and fast (a 64x64 `mamba_unet` trace: 90s+ →
  unmeasured-but-hanging fixed to ~0.1s; a realistic 256x256 "base"-size `mamba_unet`: 3.2s), so the
  independently hand-derived `_selective_scan_flops` formula (8·B·D·N·L true FLOPs per direction —
  verified against an explicit per-term manual derivation, not just architecturally reasoned) can
  replace it outright.
- **Real bug found and fixed**: `analytic_flops`'s `nn.Linear` formula computed
  `n_elements(output, excluding out_features) * in_features` — silently dropping the `out_features`
  factor entirely. Caught by testing `emcad` (PVTv2 backbone, real `nn.Linear` layers) end-to-end:
  fvcore's conv/linear subtotal disagreed with the analytic count by 78.6%, an order of magnitude
  too large to be a formula-precision issue. Traced via `fvcore`'s `by_module_and_operator()` down to
  one transformer block, hand-recomputed its 5 Linear calls from their real captured shapes, and
  matched fvcore's per-module number exactly — confirming the bug was in this module's formula, not
  fvcore's count. Fixed (`n_output_elements` now includes `out_features`); all 6 registered model
  families (`unet`, `attention_unet`, `mk_unet`, `gmk_unet`, `emcad`, `mamba_unet`) now agree with
  fvcore within 0.03%–4.6% (well under the 5% tolerance), verified directly, not assumed from the
  `unet`/`attention_unet` cases (which never exercised the buggy code path — no `nn.Linear` layers).
  Also discovered mid-investigation: fvcore's `conv`/`addmm` counts are MACs (halved), while its
  `batch_norm`/`upsample_bilinear2d` counts are already true-op-equivalent (not halved) — confirmed
  by a standalone `nn.BatchNorm2d` micro-test (`fvcore raw == 2 * n_elements`, not `1 *`) — so
  `check_flops_agreement` compares only the MAC-style operator subtotal (conv/linear, unit-normalised
  by doubling) against the hand-written formula, and folds fvcore's non-MAC-style operators as-is
  into the reported total rather than re-deriving them by hand.
- **`export.py`**: every export attempt (`try_export_torchscript`/`_onnx`/`_tensorrt`) is wrapped in a
  `SIGALRM`-based hard timeout (`_time_limit`, default 120s) — needed because TorchScript/ONNX export
  trace through the same slow sequential scan loop as fvcore's tracer; confirmed directly (a
  deliberately short `timeout_s=10` on `mamba_unet` produced two clean `fail: ExportTimeout` results
  in ~27s instead of hanging). `onnx` added to `requirements.txt`/installed in this sandbox specifically
  to verify the ONNX path for real rather than trusting an untested `except ImportError` branch — both
  TorchScript and ONNX export succeed genuinely for every non-Mamba registered model; TensorRT fails
  cleanly with `ModuleNotFoundError` (no `tensorrt` package, no GPU driver in this sandbox) — exactly
  the "failure is a reportable result, not a blocker" outcome spec calls for.
- **`train.py`/`eval.py` wired to the new package**: `check_flops_agreement` + `count_parameters`
  replace `log_model_summary`; `measure_latency` (batch=1, spec-minimum 50 warmup/200 runs) replaces
  `measure_throughput`. `utils/report.py`'s `get_latency_stats` (previously its own 10-warmup/50-run
  loop, under both spec minimums) now delegates to `profiling.latency.measure_latency` — the
  Markdown/JSON evaluation report's latency table is upgraded to the spec-compliant protocol without
  changing its rendering code. A `FlopsAgreementError` is caught and logged as a warning at both call
  sites (training/eval proceed with `flops=0` rather than aborting a run over a profiling gap) — the
  disagreement itself is real signal worth surfacing, not a reason to block.
- **Real-pipeline regression check**: `gmkunet_t_clinicdb` eval — Dice 0.7816 / mIoU 0.6881
  unchanged (the model-quality canary). Reported FLOPs changed (134.37M → 214.79M raw) because the
  measurement method changed (old thop-based estimate → new analytic+fvcore-agreement-checked count),
  not because anything about the model changed — expected and correct. Reported param count also
  changed (40,883 → 40,895): confirmed thop's own parameter count silently differs from
  `sum(p.numel() for p in model.parameters())` by 12 (isolated directly — both counts run against the
  identical freshly-constructed model, no `requires_grad=False` params exist to explain the gap) — a
  pre-existing thop quirk now bypassed by using `count_parameters` directly, not a new discrepancy
  introduced here.
- **New `tests/test_profiling.py`** (20 tests, including spec's own named `test_flops_agreement`,
  which asserts agreement for all 6 registered model families in one test): the hand-derived scan
  formula against an independent per-term manual computation, the `nn.Linear` regression case
  explicitly (asserts `out_features` is included), `check_flops_agreement`'s raise path (verified with
  `tolerance=0.0` against `mk_unet`'s real small residual error, not a fabricated scenario), latency's
  spec-minimum guards, CPU peak-memory correctly returning `None` (not a fabricated `0.0`), and the
  export timeout against a deliberately slow dummy module (fast, deterministic — doesn't depend on the
  real Mamba model's actual trace time). `tests/test_ci_audit.py` updated: `test_flops_agreement` moved
  from "blocked on Phase 10" to "exists" (only `test_reporting_blocks`/Phase 14 remains blocked).
  183 passed, 1 correctly-skipped, repo-wide (163 prior + 20 new). `pip check` clean.

## Phase 11 — Attribution and explainability module (2026-08-28)

- **New `attribution/` package** (spec §11): `common.py` (shared channel-group-slice resolution —
  reuses Phase 4's `datasets.augment.AugmentationPolicy.effective_groups`, the exact boundaries the
  dataset pipeline actually built the model's input from — plus the training-set mean image and the
  occlusion primitive both `occlusion.py` and `integrated_grads.py` need), `occlusion.py`
  (channel-group occlusion, Dice drop per group), `shapley.py` (exact Shapley over the active channel
  groups, 2^k coalitions), `integrated_grads.py` (captum `IntegratedGradients`, baseline = training
  mean image, per-channel/per-group attribution mass, agreement score vs. occlusion), `branch.py`
  (per-stage auxiliary-branch ablation, Mamba-family only), `fusion_probe.py` (CBFFM gate statistics
  vs. GT structure size), `segcam.py` (Seg-Grad-CAM / Seg-XRes-CAM, qualitative only — vanilla
  Grad-CAM deliberately not implemented, per spec), `sanity.py` (Adebayo et al. 2018-style cascading
  parameter randomisation + data-label randomisation checks, SSIM-scored).
- **Shapley combinatorics verified against three independent hand-computed games** before trusting it
  on any real model: a 2-player asymmetric game (φ_A=0.65, φ_B=0.35, hand-derived from the standard
  2-player formula), a 3-player additive game (v(S)=|S|, every player's exact Shapley value must be
  1.0 by symmetry), and a 4-player symmetric superadditive game (v(S)=|S|², every value must be 4.0) —
  all matched to 1e-9. The efficiency property (Σφ = v(all)−v(∅)) was then also confirmed on a real
  `mk_unet` forward pass end-to-end (occlusion's baseline/fully-occluded Dice vs. Shapley's value sum),
  not just on the synthetic toy games.
- **Real bug caught while wiring `attribution/branch.py`'s stage-ablation verification**: a first
  verification attempt registered the probe hook *before* entering the ablation context, so it fired
  first (PyTorch calls forward hooks in registration order, each seeing the prior hook's returned
  value) and captured the *pre*-ablation output — looked like `ablate_auxiliary_stage` wasn't zeroing
  anything. Not a bug in the module: reordering the probe hook to register *inside* the context (the
  order any real caller would use) confirmed stage 2's output is exactly zero and every other stage's
  is untouched.
- **Real (non-bug) finding from `sanity.py`'s own end-to-end test**: an untrained model's raw sigmoid
  output has essentially zero spatial variance (std ≈ 3e-5 across a 32×32 map, confirmed directly) —
  SSIM between two such near-flat maps (original vs. cascade-randomised) comes out high (0.956) purely
  because neither has real structure to disrupt, so the sanity check correctly reports a **fail** for
  that "saliency" method on that model. This is the check doing its job (catching a saliency source
  that isn't actually sensitive to learned weights), not a defect — the test was rewritten to certify
  against `seg_grad_cam`'s gradient-based map instead (confirmed non-degenerate: full [0,1] range with
  a forced non-empty target region), which correctly passes (SSIM drops well below threshold).
  `parameter_randomization_sanity_check` also directly verified to *fail* on a saliency function that
  ignores which model it's given (returns the original model's map regardless of the randomised copy
  passed in) — SSIM=1.0, `overall_pass=False`, confirming the check can't be trivially gamed.
- **`branch.py`/`fusion_probe.py` design note**: both use `register_forward_hook`'s documented ability
  to *replace* a wrapped module's return value (returning non-`None` from the hook) — `branch.py`
  zeroes one of `auxiliary_encoder`'s 4 returned stage tensors this way, with no change to
  `MambaUNet.forward()`'s own code; `fusion_probe.py` reads `CBFFM.gate`'s own forward output directly
  (the gate map it already computes internally, just never previously exposed) the same way. Both
  raise a clear `AttributeError`/`ValueError` immediately for a model/fusion-mode this analysis doesn't
  apply to (verified against `mk_unet` and a `fusion="add"` Mamba variant), rather than silently
  returning an empty or zero result.
- **New `tests/test_attribution.py`** (30 tests) covering all 8 submodules, including the 3
  hand-computed Shapley games, the real-model Shapley/occlusion efficiency-property cross-check, the
  branch-ablation hook-ordering-corrected direct verification, both fusion_probe guard rejections, and
  both sanity-check pass/fail directions (real saliency passes, input-insensitive/gamed saliency
  fails). 213 passed, 1 correctly-skipped, repo-wide (183 prior + 30 new). `pip check` clean.
  Real-pipeline regression check unchanged (`gmkunet_t_clinicdb` still Dice 0.7816 / mIoU 0.6881).

## Phase 12 — Uncertainty module (2026-08-28)

- **New `uncertainty/` package** (spec §12): `ensemble.py` (`predict_ensemble_members` — per-member
  probability maps stacked on a new leading dimension, reusing `eval.py`'s existing `EnsembleModel`
  concept of averaging several already-trained seed/fold checkpoints rather than introducing a second
  ensembling mechanism; `predictive_entropy` — per-pixel Shannon entropy of the ensemble mean,
  binary and multiclass; `inter_seed_variance` — per-pixel variance across members, the
  epistemic-uncertainty proxy) and `retention.py` (`error_detection_auroc` — AUROC treating
  "pixel wrong" as the label and uncertainty as the score; `uncertainty_error_correlation` — the
  linear-agreement complement; `retention_curve` — Dice as a function of the fraction of
  most-uncertain images referred away).
- **Every formula hand-verified against an independent closed form or library call before trusting
  it on model data**: `predictive_entropy`'s binary case at p=0.5 against ln(2) (the known maximum-
  entropy point) and at p=0.9 against the closed-form `-(0.9 ln 0.9 + 0.1 ln 0.1)`; the multiclass
  case against `ln(C)` for a uniform C=4 distribution (another known maximum-entropy identity);
  `inter_seed_variance` against `numpy.var`'s own population-variance computation directly;
  `error_detection_auroc` against a perfectly-separable synthetic case (AUROC must equal exactly
  1.0) plus both single-class-present degenerate cases (must return NaN, not a silently-wrong 0.5);
  `uncertainty_error_correlation` against `numpy.corrcoef` directly; `retention_curve` against a
  hand-computed 4-image example (retaining the 2 least-uncertain of 4 images at referred_fraction=0.5
  must give exactly the mean of those 2 images' own metric values, verified to 1e-9).
- **New `tests/test_uncertainty.py`** (14 tests) covering every formula above plus
  `predict_ensemble_members` end-to-end against 3 real (untrained but architecturally real) `mk_unet`
  instances, and the retention/AUROC/correlation functions' input-validation guards (shape mismatch,
  out-of-range fraction, empty input). 227 passed, 1 correctly-skipped, repo-wide (213 prior + 14
  new). `pip check` clean.

## Phase 13 — Robustness module + mechanism analysis (2026-08-28)

- **New `robustness/` package** (spec §13 + §11's shortcut-audit guarantee): `corruptions.py` (8
  photometric/acquisition corruptions — Gaussian noise, speckle, blur, JPEG compression,
  brightness/contrast, gamma, resolution change, resampling — severities 1-5 declared once and
  shared), `common.py` (the shared corrupt-batch-and-recompute-Dice driver, `degradation_curve`,
  `mean_corruption_error`), `geometric.py` (translate/rotate/scale/off-centre-crop via a shared affine
  ``grid_sample`` primitive, `geometric_degradation_curve`, `shortcut_audit`, `frame_jitter_sensitivity`
  — reuses Phase 4's channel-group machinery directly).
- **Geometry-channel design decision, made explicit and tested, not left implicit**: every geometric
  transform applies to content channel groups (rgb/ycbcr) and the mask, but deliberately leaves
  geometry channel groups (xy, rtheta) untouched — they are a pure function of output grid shape (see
  `datasets/channels.py`), not of pixel content, and every transform here preserves (H, W); this
  mirrors `datasets/augment.py`'s own existing "transform content, then rebuild geometry" ordering
  rather than spatially distorting the coordinate channels' encoded values. Verified directly: a
  synthetic bright-square-exactly-matching-mask image was translated/rotated/scaled/cropped, and the
  transformed content and transformed mask were confirmed to still overlap (content/mask co-registration
  holds under every transform, not just approximately).
- **Mask binarity verified, not assumed**: `grid_sample`'s bilinear interpolation would otherwise
  produce fractional mask values at transform boundaries; masks are resampled with nearest-neighbour
  and re-thresholded — confirmed the output mask's value set is exactly `{0.0, 1.0}` after rotation and
  translation, not merely "close to binary".
- **New `analysis/` package** (spec §12's mechanism-analysis module): `erf.py` (gradient-based
  Effective Receptive Field, Luo et al. 2016, + a weighted-RMS-radius summary), `cka.py` (linear
  Centered Kernel Alignment, Kornblith et al. 2019, via the Frobenius-norm identity rather than full
  Gram matrices), `failure_taxonomy.py` (6-category per-image failure classification — success,
  missed_lesion, false_positive, under/over-segmentation, boundary_only — + gallery indexing).
- **Every formula verified against an independent closed form before trusting it on model data**:
  `erf_radius` against the exact closed-form RMS radius of a uniform k×k square (`k/√6`) for k ∈
  {4,8,16,32}; `compute_erf` against a hand-built single-Conv2d model with a uniform (all-ones) 5×5
  kernel — the measured gradient was confirmed non-zero in *exactly* the theoretical 5×5 window (span
  ≤4 pixels in both axes) and its radius matched `5/√6`. `linear_cka` against four independent
  mathematical identities: self-similarity (`CKA(X,X)=1.0` exactly), rotation invariance
  (`CKA(X,Y)==CKA(X,Y@R)` for orthogonal R, to 1e-8), isotropic-scale invariance
  (`CKA(X,Y)==CKA(X,5Y)`), and near-zero similarity for independent random high-sample-count matrices.
  `classify_failure`'s all 6 categories individually verified against hand-constructed synthetic
  pred/gt pairs designed to land in exactly one category each (a tiny-corner prediction for
  under-segmentation, a full-image prediction for over-segmentation, a shifted-square prediction with
  a raised threshold for boundary_only).
- **New `tests/test_robustness.py`** (33 tests) and **`tests/test_analysis.py`** (26 tests): every
  corruption's shape/dtype/severity-guard, the content/mask alignment property across all four
  geometric transforms, a hand-built "identity-like" model (predictions directly track input
  brightness, sidestepping the untrained-random-weight near-constant-output confound seen throughout
  this session's synthetic-data checks) used to confirm blur genuinely degrades Dice while translation
  keeps image/mask correctly co-registered, both shortcut-audit threshold directions, all four CKA
  identities, the ERF closed-form checks, and every failure-taxonomy category plus its aggregation and
  gallery-index guards. 286 passed, 1 correctly-skipped, repo-wide (227 prior + 33 + 26 new). `pip
  check` clean. Real-pipeline regression check unchanged (`gmkunet_t_clinicdb` still Dice 0.7816 /
  mIoU 0.6881).

## Phase 14 — Reporting layer + orchestration completion (2026-08-28)

- **New `orchestration/sweep.py`** (spec §15's Sweep row): budget-aware successor to `search.py` —
  `run_budgeted_sweep()` runs grid trials in a seeded-shuffled order until measured cumulative
  wall-clock cost would exceed `budget_gpu_hours` (a required, project-level constant — spec's
  literal "takes a search space and a trial budget", not `search.py`'s bare `num_trials`), rather
  than a fixed trial count. Zero import-time dependency on `datasets`/`torch` (`trial_fn` defaults
  via a lazy import), matching `orchestration.runner.run_sweep`'s own discipline — verified by the
  same static check `search.py` already had, extended: `tests/test_data_contract.py::test_sweep_cannot_see_test`
  now also greps this file's source for the guarded test-loader function's name and expects zero
  hits. Added `budget_gpu_hours: Optional[float] = None` to `orchestration/schema.py`'s
  `SearchMetaConfig`. Verified with a fake `trial_fn`: correct early stop at a tiny budget (3/5
  trials), correct full-grid completion with an ample budget, correct best-trial selection under
  both "max" and "min" objective modes, and a failing trial logged without killing the sweep.
- **New `reporting/` package** (spec §16): `tables.py` (four blocking rules — dirty-tree run,
  under-seeded config, missing stats entry, unsanitised saliency — every one a hard `BlockingRuleError`
  raise, plus `render_main_comparison_table`/`render_efficiency_table` reading only the orchestration
  ledger's CSV rows and a `results.parquet`-shaped row list, with a provenance footer on every
  output), `figures.py` (Pareto frontier with real dominated-point detection, a critical-difference
  diagram consuming `stats.ranking.nemenyi_posthoc`'s actual return shape directly, a degradation-curve
  plot consuming `robustness.common.degradation_curve`'s actual return shape directly), `inventory.py`
  (spec §20's 19-row artefact-inventory table as data, normalised to be uniformly relative to
  `reports/` — spec's own table inconsistently prefixes exactly one row with `reports/`, found while
  writing a round-trip test against it — plus an audit function reporting which artefacts exist on
  disk today).
- **Every blocking rule verified to genuinely block AND to pass clean input** — not just unit-tested
  in isolation: seeded a real `orchestration.ledger.LedgerWriter` CSV with a dirty run, an
  under-seeded config, and (separately) a fully clean 2-model×3-seed ledger, and confirmed
  `render_main_comparison_table` itself (not just the underlying check functions) propagates a block
  end-to-end while rendering a correct LaTeX+CSV table end-to-end for the clean case.
- **New `scripts/generate_report.py`**: the reporting layer's one real CLI — globs `eval.py`'s
  existing JSON report dumps (`utils/report.py`'s `EvaluationReporter._write_json` output) and
  renders `reports/tables/*.{csv,tex}` from them. Verified against a genuinely real eval.py report:
  fed it `logs/gmkunet_t_clinicdb/report.json` (the standing regression-canary run) plus a
  synthetic 3-seed ledger, and confirmed the rendered CSV carries the exact canary numbers (Dice
  0.7816, mIoU 0.6881) copied through unchanged — the reporting layer never recomputes a metric,
  confirmed by construction here, not just by not calling a metrics function.
- **New `scripts/reproduce.sh`**: orchestrates S1-S17 honestly — S1/S2 (pytest assertions), S3/S4/S6
  (a real, reduced `orchestration.runner.run_sweep` call, one train.py run per seed), S15 (`eval.py
  --allow-test-eval`, once per seed), S17 (`scripts/generate_report.py`) are genuinely executed;
  S5/S7-S14/S16 (no standalone CLI exists for any of Phases 4/9/11/12/13's library modules) print the
  exact Python module/function to call instead of fabricating a command that doesn't exist.
  **Two real bugs found running this script live, not just reading it**: (1) a first draft reused
  the config's own bare `experiment_name` for every stage — running it against a real experiment's
  config silently overwrote that experiment's `logs/<name>/report.json` with the smoke run's own
  results (caught directly: `gmkunet_t_clinicdb`'s canary report briefly read Dice 0.0 mid-session,
  immediately re-generated from the untouched checkpoint to confirm no data was actually lost — only
  the report file, from a checkpoint that survived). Fixed by scoping every stage to a
  `<name>_reproduce_<snapshot>_s<seed>` experiment name, and by adding a matching `--experiment-name`
  override flag to `eval.py` so it can target the same scoped name. (2) `utils.checkpoint.CheckpointManager`'s
  save directory is keyed only by `experiment_name`, not seed — an initial multi-seed loop under one
  shared name would have had each seed's checkpoint silently overwrite the previous seed's before it
  was ever evaluated; fixed by giving each seed its own scoped name (`..._s<seed>`), one checkpoint
  directory and one `eval.py` call per seed. (3) discovered while testing this script specifically:
  in this harness, a `pytest`/`python` invocation launched as a background process does not reliably
  see the same import path an interactive foreground shell does (`ModuleNotFoundError: No module
  named 'orchestration'` even from the repo root, reproduced with a bare `pytest tests/...` outside
  the script entirely) — fixed defensively with an explicit `export PYTHONPATH="$(pwd):$PYTHONPATH"`
  after the script's `cd`, verified to resolve it. Live-ran the corrected script end-to-end (3 seeds
  x 1 epoch, `gmkunet_t_clinicdb`): S1/S2 assertions passed for real, S6 trained 3 real per-seed
  checkpoints, S15 evaluated each for real, S16's profiling numbers landed in each report — and S17
  then correctly **refused** to render a table, `BlockingRuleError: dirty-tree run(s) present`,
  because this repo's working tree genuinely is dirty (this session's own uncommitted Phase 9-14
  work) — the blocking rule firing on a real run against real repo state, not just in a synthetic
  unit test, is exactly the guarantee spec §16 asks for. The clean-tree "happy path" (table actually
  renders) is covered instead by `scripts/generate_report.py`'s own standalone verification above
  (real canary JSON + a hand-seeded clean 3-seed ledger) and by `tests/test_reporting.py`'s unit
  tests. Also discovered live: `eval.py`'s test-token minting constructs `LedgerWriter()` with no
  arguments, so it always writes to the hardcoded default `artifacts/ledger/` regardless of a
  caller's `$LEDGER_DIR` — every `--allow-test-eval` invocation this entire session (including every
  regression-canary recheck) has been appending to the *real* project ledger's `test_evals.csv`, not
  a scratch one. Left as-is rather than "fixed": that file accumulating a permanent record of every
  real test-set touch, including repeats, is spec's own explicit design ("repeats are surfaced, not
  hidden" — §21's pitfalls table) — `orchestration.runner.run_sweep`'s `ledger_dir` parameter, used
  for the Runs/Compute tables, is correctly caller-configurable and was not affected.
- **New `tests/test_sweep.py`** (9 tests) and **`tests/test_reporting.py`** (15 tests, including
  spec's own named `test_reporting_blocks`). `tests/test_ci_audit.py` updated: all 15 of spec §19's
  named tests now exist (`test_reporting_blocks` was the last one). 310 passed, 1 (pytest's own
  empty-parametrize placeholder, not a real skip) skipped, repo-wide (286 prior + 9 + 15 new). `pip
  check` clean. Real-pipeline regression check unchanged (`gmkunet_t_clinicdb` still Dice 0.7816 /
  mIoU 0.6881 — reconfirmed after the accidental overwrite above).
